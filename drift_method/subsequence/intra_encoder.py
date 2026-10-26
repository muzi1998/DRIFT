# -*- coding: utf-8 -*-
"""Intra-sub-sequence representation encoder.

Turns the soft partition of the long-term history into ``K`` sub-sequence
representations ``h_{u,k}``. The encoder kind is selected by
``cfg.intra_encoder`` (default ``"linear_mha"``):

* ``"linear_mha"``  — multi-head *linear* attention with membership weighting
  (default; cost ``O(L * K * d)``, no ``L x L`` matrix).
* ``"mha"``         — membership-weighted scaled dot-product attention.
* ``"sparse_mha"``  — membership-weighted attention plus per-query top-k.
* ``"linear_fusion"`` — the simpler weighted linear fusion implemented below,
  kept as a lightweight reference variant.

``linear_fusion`` computes::

    v_{u,t}      = W_v x_{u,t}
    r_{u,t,k}    = w_in^T tanh( W_in [ x_{u,t} ; log(1 + dt_t) ; e_p(t) ; d_u ] )
    lambda_{u,t,k} = y_{u,t,k} exp(r_{u,t,k}) / sum_tau y_{u,tau,k} exp(r_{u,tau,k})
    h_{u,k}      = LayerNorm( sum_t lambda_{u,t,k} v_{u,t} )

In every variant, sub-sequence ``k`` is representated only by items assigned to it,
because the soft assignment ``y_{u,t,k}`` gates the aggregation.

Notes
-----
* ``log(1 + dt_t)`` uses real timestamps when available; otherwise it degenerates
  to the constant ``log(2)`` and contributes no gradient.
* ``e_p(t)`` is a learned positional embedding shared across sub-sequences.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from drift_method.subsequence.encoders import build_subseq_encoder
except ImportError:  # direct-execution fallback
    from .encoders import build_subseq_encoder


class IntraSubsequenceEncoder(nn.Module):
    """Encode each historical sub-sequence into a single vector.

    The forward signature is stable across variants and always returns
    ``(h_uk, weight)``:

        h_uk:   (B, K, D)     one representation per sub-sequence
        weight: (B, L_h, K)   ``lambda`` for linear fusion, or the normalized
                              membership weights for the attention variants.
    """

    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        # Encoder kind; defaults to membership-weighted linear attention.
        self.intra_kind = getattr(cfg, "intra_encoder", "linear_mha")
        self.attn_enc = build_subseq_encoder(
            self.intra_kind, cfg, d_model,
            num_heads=getattr(cfg, "intra_num_heads", 4),
            topk=getattr(cfg, "intra_topk", 8),
        )

        # --- Layers used only by the "linear_fusion" variant -----------------
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # r_in input: x (D) + log dt (1) + e_p (dim_p) + d_u (1)
        self.dim_p = 16
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 4, self.dim_p)
        in_dim = d_model + 1 + self.dim_p + 1
        self.W_in = nn.Linear(in_dim, d_model, bias=True)
        self.w_in = nn.Linear(d_model, cfg.K, bias=False)   # score all K at once
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_long: torch.Tensor,        # (B, L_h, D)
        m_long: torch.Tensor,        # (B, L_h) bool
        y_utk: torch.Tensor,         # (B, L_h, K) soft assignment from partition
        d0: torch.Tensor,            # (B,) shift degree
        history_ts_long: torch.Tensor = None,  # (B, L_h) float, in days
    ):
        cfg = self.cfg
        B, L_h, D = x_long.shape
        device = x_long.device

        # --- Attention variants: membership-weighted aggregation ------------
        if self.attn_enc is not None:
            return self.attn_enc(x_long, m_long, y_utk)

        # --- Reference variant: weighted linear fusion ----------------------
        # log(1 + dt) from real timestamps, else the constant log(2).
        if getattr(cfg, "use_real_timestamp", False) and history_ts_long is not None:
            ts = history_ts_long
            dt = torch.zeros(B, L_h, device=device)
            dt[:, 1:] = (ts[:, 1:] - ts[:, :-1]).clamp(
                min=getattr(cfg, "ts_min_dt_floor", 1e-5))
            valid_pair = m_long[:, 1:] & m_long[:, :-1]
            dt[:, 1:] = dt[:, 1:] * valid_pair.float()
            log_dt = torch.log1p(dt).unsqueeze(-1)
        else:
            log_dt = torch.full((B, L_h, 1), math.log(2.0), device=device)

        pos = torch.arange(L_h, device=device).unsqueeze(0).expand(B, L_h)
        e_p = self.pos_emb(pos)                                     # (B, L_h, dim_p)
        d0_exp = d0.view(B, 1, 1).expand(B, L_h, 1)
        feat = torch.cat([x_long, log_dt, e_p, d0_exp], dim=-1)     # (B, L_h, in_dim)
        h_in = torch.tanh(self.W_in(feat))                          # (B, L_h, D)
        r_in = self.w_in(h_in)                                      # (B, L_h, K)

        # Normalize lambda_{u,t,k} over t for each sub-sequence k. Adding log(y)
        # in logit space suppresses non-member and padding positions.
        m_pad = m_long.float().unsqueeze(-1)                        # (B, L_h, 1)
        y_mask = (y_utk * m_pad).clamp(min=cfg.eps)                 # (B, L_h, K)
        logit = r_in + torch.log(y_mask + cfg.eps)
        logit = logit - logit.max(dim=1, keepdim=True).values       # numerical stability
        ex = torch.exp(logit)
        denom = ex.sum(dim=1, keepdim=True).clamp_min(cfg.eps)
        lam = ex / denom                                            # (B, L_h, K)

        v_t = self.W_v(x_long)                                      # (B, L_h, D)
        # h_{u,k} = sum_t lambda_{u,t,k} v_{u,t}
        h_uk = torch.einsum("bld,blk->bkd", v_t, lam)               # (B, K, D)
        h_uk = self.layer_norm(h_uk)
        return h_uk, lam
