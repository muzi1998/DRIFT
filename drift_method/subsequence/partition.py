# -*- coding: utf-8 -*-
"""Differentiable soft boundaries and adaptive sub-sequence assignment.

Given the long-term part of a user history, this module predicts *where* the
user's preference shifts and *how many* coherent sub-sequences the history should
be split into. Everything is differentiable, so the partition is learned jointly
with the generative objective instead of being fixed a priori.

Formulation
-----------
Per-position shift descriptor (``phi``), boundary intensity (``b``), cumulative
soft coordinate (``r``), per-user effective number of sub-sequences (``kappa_u``),
activity gate (``m``) and soft assignment (``a``)::

    phi_{u,t} = [ 1 - cos(x_t, x_{t-1}) ;
                  log(1 + dt_t) ;
                  JSD(P_{t-w:t}, P_{t:t+w}) ;
                  x_t - x_{t-1} ;
                  d_u ]
    b_{u,t}   = sigmoid(MLP_b(phi_{u,t}))
    S_{u,t}   = sum_{tau <= t} b_{u,tau}
    kappa_u   = K_min + (K_max - K_min) *
                sigmoid(theta0 + theta_d*d_u + theta_L*ell_u
                        + theta_b*b_bar + theta_v*v_u)
    r_{u,t}   = 1 + (kappa_u - 1) * S_{u,t} / (S_{u,L} + eps)
    m_{u,k}   = sigmoid((kappa_u - k + 0.5) / tau_m)
    a_{u,t,k} = m_{u,k} * exp(-(r_{u,t} - mu_k)^2 / tau_a) / normalizer

Implementation notes
--------------------
* ``dt_t`` uses real timestamps when available, as
  ``(ts[t] - ts[t-1]) / time_unit_scale`` in days, lower-bounded by
  ``cfg.ts_min_dt_floor``. When ``cfg.use_real_timestamp`` is False or no
  timestamps are provided it degenerates to ``dt = 1``, i.e. the constant
  ``log(2)``, which contributes no gradient.
* The local Jensen-Shannon divergence compares first-level SID histograms in the
  windows ``[t-w, t)`` and ``[t, t+w)``; it is computed in closed form from a
  cumulative one-hot tensor for efficiency.
* Candidate sub-sequence centers are fixed and uniform, ``mu_k = k``.
* Normalizing ``r`` by ``kappa_u`` (``coord_normalize="kappa"``) is what couples
  the assignment granularity to the per-user shift statistics; the ``"raw"`` and
  ``"K"`` variants are kept for reference.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_cos(a, b, eps=1e-8):
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


class SoftPartition(nn.Module):
    """Predict soft boundaries and sub-sequence assignments.

    Inputs are the long-term history embeddings ``(B, L_h, D)`` plus their
    first-level SID codes; outputs are the boundary intensities ``b_{u,t}``, the
    soft assignment ``a_{u,t,k}`` and the activity gate ``m_{u,k}``.
    """

    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K
        self.tau_a = cfg.tau_a
        self.tau_m = cfg.tau_m

        # MLP_b input: 1 (cos) + 1 (log dt) + 1 (jsd) + d_model (diff) + 1 (d_u)
        in_dim = 1 + 1 + 1 + d_model + 1
        self.mlp_b = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Fixed uniform sub-sequence centers mu_k = k.
        mu = torch.arange(1, self.K + 1, dtype=torch.float32)
        self.register_buffer("mu_k", mu)
        self.window_w = 3

    def kappa_u(self, d0, L_eff, b_bar, v_u):
        cfg = self.cfg
        ell_u = torch.log1p(L_eff) / math.log(1.0 + cfg.L_max_ref)
        logits = (
            cfg.theta0
            + cfg.theta_d * d0
            + cfg.theta_L * ell_u
            + cfg.theta_b * b_bar
            + cfg.theta_v * v_u
        )
        kappa = cfg.K_min + (cfg.K_max - cfg.K_min) * torch.sigmoid(logits)
        return kappa

    def _local_jsd(self, c1_long, mask_long, sid1_size):
        """Local JSD between the SID histograms before/after each position.

        Returns a ``(B, L_h)`` tensor.
        """
        B, L = c1_long.shape
        w = self.window_w
        # Closed-form window sums via a cumulative one-hot tensor.
        safe = c1_long.clamp(min=0, max=sid1_size - 1)
        oh = F.one_hot(safe, num_classes=sid1_size).float() * mask_long.float().unsqueeze(-1)
        cum = oh.cumsum(dim=1)
        # Windows [t-w, t-1] (left) and [t, t+w-1] (right).
        idx = torch.arange(L, device=c1_long.device)
        lo_a = (idx - w).clamp(min=0)
        hi_a = idx
        lo_b = idx
        hi_b = (idx + w).clamp(max=L)

        def _gather_slice(cum, hi, lo):
            # cum: (B, L, C); hi, lo: (L,)
            # Returns (B, L, C) window sums cum[hi] - cum[lo].
            zero = torch.zeros_like(cum[:, :1, :])
            cum_ext = torch.cat([zero, cum], dim=1)                  # (B, L+1, C)
            hi_idx = hi.long().unsqueeze(0).unsqueeze(-1).expand(B, L, cum.size(-1))
            lo_idx = lo.long().unsqueeze(0).unsqueeze(-1).expand(B, L, cum.size(-1))
            return cum_ext.gather(1, hi_idx) - cum_ext.gather(1, lo_idx)

        sum_a = _gather_slice(cum, hi_a, lo_a)
        sum_b = _gather_slice(cum, hi_b, lo_b)
        eps_p = self.cfg.jsd_smooth_eps
        Pa = (sum_a + eps_p) / (sum_a.sum(-1, keepdim=True) + eps_p * sid1_size + 1e-12)
        Pb = (sum_b + eps_p) / (sum_b.sum(-1, keepdim=True) + eps_p * sid1_size + 1e-12)
        M = 0.5 * (Pa + Pb)
        eps = 1e-12
        jsd = 0.5 * (
            (Pa * (Pa.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(-1)
            + (Pb * (Pb.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(-1)
        )
        return jsd

    def forward(
        self,
        x_long: torch.Tensor,        # (B, L_h, D)
        m_long: torch.Tensor,        # (B, L_h) bool
        c1_long: torch.Tensor,       # (B, L_h) long
        d0: torch.Tensor,            # (B,)
        sid1_size: int,
        history_ts_long: torch.Tensor = None,   # (B, L_h) float, in days
    ):
        cfg = self.cfg
        B, L_h, D = x_long.shape
        device = x_long.device

        # --- Build the shift descriptor phi_{u,t} -------------------------
        # Semantic jump 1 - cos(x_t, x_{t-1}); position 0 has no predecessor.
        cos_pair = _safe_cos(x_long[:, 1:, :], x_long[:, :-1, :])
        cos_pair = torch.cat([torch.zeros(B, 1, device=device), 1.0 - cos_pair], dim=1)  # (B, L_h)

        # log(1 + dt): prefer real timestamps (in days, lower-bounded);
        # otherwise fall back to the constant log(2) (no gradient).
        if getattr(cfg, "use_real_timestamp", False) and history_ts_long is not None:
            ts = history_ts_long
            dt = torch.zeros(B, L_h, device=device)
            dt[:, 1:] = (ts[:, 1:] - ts[:, :-1]).clamp(min=getattr(cfg, "ts_min_dt_floor", 1e-5))
            # Zero out dt when either endpoint is padding; log1p(0) = 0 is neutral.
            valid_pair = m_long[:, 1:] & m_long[:, :-1]
            dt[:, 1:] = dt[:, 1:] * valid_pair.float()
            log_dt = torch.log1p(dt)
        else:
            log_dt = torch.full((B, L_h), math.log(2.0), device=device)

        # Local distribution shift.
        jsd_local = self._local_jsd(c1_long, m_long, sid1_size)                          # (B, L_h)

        # diff: x_t - x_{t-1}
        diff = torch.cat([torch.zeros(B, 1, D, device=device), x_long[:, 1:, :] - x_long[:, :-1, :]], dim=1)

        d0_exp = d0.view(B, 1).expand(B, L_h)

        phi = torch.cat([
            cos_pair.unsqueeze(-1),
            log_dt.unsqueeze(-1),
            jsd_local.unsqueeze(-1),
            diff,
            d0_exp.unsqueeze(-1),
        ], dim=-1)                                                                       # (B, L_h, in_dim)

        b_logit = self.mlp_b(phi).squeeze(-1)                                            # (B, L_h)
        b = torch.sigmoid(b_logit)
        # No predecessor at position 0, so force b_{u,1} = 0.
        b = b * m_long.float()
        b[:, 0] = 0.0

        # --- Effective number of sub-sequences kappa_u -------------------
        L_eff = m_long.float().sum(dim=1)                                                # (B,)
        b_bar = b.sum(dim=1) / (L_eff - 1).clamp_min(1.0)
        v_u = 4.0 * b.var(dim=1, unbiased=False)
        kappa = self.kappa_u(d0, L_eff, b_bar, v_u)                                      # (B,)

        # --- Cumulative soft coordinate r_{u,t} (normalized by kappa_u) --
        S_ut = b.cumsum(dim=1)                                                           # (B, L_h)
        S_uL = S_ut[:, -1:].clamp_min(cfg.eps)                                           # (B, 1)
        if cfg.coord_normalize == "raw":
            r_ut = 1.0 + S_ut
        elif cfg.coord_normalize == "K":
            r_ut = 1.0 + (cfg.K - 1) * S_ut / S_uL
        else:  # "kappa" (default)
            r_ut = 1.0 + (kappa.unsqueeze(1) - 1.0) * S_ut / S_uL

        # --- Activity gate m_{u,k} ---------------------------------------
        k_idx = torch.arange(1, self.K + 1, device=device, dtype=torch.float32)          # (K,)
        m_uk = torch.sigmoid((kappa.unsqueeze(1) - k_idx.unsqueeze(0) + 0.5) / cfg.tau_m)   # (B, K)

        # --- Soft assignment a_{u,t,k}, shape (B, L_h, K) ----------------
        diff2 = (r_ut.unsqueeze(-1) - self.mu_k.view(1, 1, -1)) ** 2
        logit = -diff2 / cfg.tau_a
        # Multiplying by m_{u,k} after exp is numerically safer than adding log m.
        ex = torch.exp(logit) * m_uk.unsqueeze(1)                                        # (B, L_h, K)
        a_utk = ex / (ex.sum(dim=-1, keepdim=True) + cfg.eps)
        # Padding positions get zero mass so they never contribute downstream.
        a_utk = a_utk * m_long.float().unsqueeze(-1)

        out = {
            "b": b,                  # (B, L_h)
            "S_ut": S_ut,
            "r_ut": r_ut,
            "kappa": kappa,
            "m_uk": m_uk,
            "a_utk": a_utk,
        }
        if cfg.use_gumbel:
            # Gumbel-Softmax reparameterization of the assignment.
            logits_a = torch.log(a_utk + cfg.eps)
            g = -torch.log(-torch.log(torch.rand_like(logits_a) + cfg.eps) + cfg.eps)
            y = F.softmax((logits_a + g) / cfg.gumbel_tau, dim=-1)
            y = y * m_long.float().unsqueeze(-1)
            out["y_utk"] = y
        else:
            out["y_utk"] = a_utk

        return out
