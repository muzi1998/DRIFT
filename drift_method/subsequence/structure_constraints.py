# -*- coding: utf-8 -*-
"""Per-sub-sequence coherence statistics.

These three statistics quantify how coherent each learned sub-sequence is; they
are combined into the structure loss by ``drift_method/losses.py``, and they also
serve as interpretable diagnostics of the learned partition.

Let ``a_{u,t,k}`` be the soft assignment of position ``t`` to sub-sequence ``k``::

    xbar_{u,k}    = sum_t a_{u,t,k} x_{u,t} / sum_t a_{u,t,k}                (centroid)

    C^sem_{u,k}   = sum_t a_{u,t,k} (1 - cos(x_{u,t}, xbar_{u,k}))
                    / sum_t a_{u,t,k}                       (semantic compactness)

    C^time_{u,k}  = sum_{t>=2} a_{u,t,k} a_{u,t-1,k} log(1 + dt_t)
                    / sum_{t>=2} a_{u,t,k} a_{u,t-1,k}      (temporal continuity)

    C^sid_{u,k}   = sum_t a_{u,t,k} CE(y^{(1)}_{u,t}, ybar^{(1)}_{u,k})
                    / sum_t a_{u,t,k}                        (SID consistency)

Notes
-----
* ``C^time`` prefers real timestamps (in days) and degenerates to the constant
  ``log(2)`` when they are unavailable, in which case it has zero gradient.
* ``C^sid`` uses the *first-level SID code* as a categorical proxy for the item
  category; because ``ybar`` is the assignment-weighted mean of one-hot codes, this
  cross-entropy is exactly the entropy of the aggregated first-level distribution
  within the sub-sequence.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def stage_centroid(x_long: torch.Tensor, a_utk: torch.Tensor, eps: float = 1e-8):
    """Assignment-weighted centroid xbar_{u,k}; returns (B, K, D)."""
    weights = a_utk.sum(dim=1).clamp_min(eps)                  # (B, K)
    centroid = torch.einsum("bld,blk->bkd", x_long, a_utk) / weights.unsqueeze(-1)
    return centroid


def semantic_compactness(x_long, a_utk, eps=1e-8):
    """Semantic compactness C^sem_{u,k}; returns (B, K)."""
    cen = stage_centroid(x_long, a_utk, eps)                   # (B, K, D)
    # 1 - cos(x_{u,t}, xbar_{u,k}) for all (t, k) pairs.
    x_n = F.normalize(x_long, dim=-1, eps=eps)                  # (B, L, D)
    c_n = F.normalize(cen, dim=-1, eps=eps)                     # (B, K, D)
    cos_tk = torch.einsum("bld,bkd->blk", x_n, c_n)             # (B, L, K)
    one_minus = 1.0 - cos_tk
    num = (a_utk * one_minus).sum(dim=1)                        # (B, K)
    den = a_utk.sum(dim=1).clamp_min(eps)
    return num / den


def time_continuity(a_utk, history_ts_long=None, m_long=None, cfg=None, eps=1e-8):
    """Temporal continuity C^time_{u,k}; returns (B, K).

    Uses real timestamps (``history_ts_long``, in days) when available and
    degenerates to the constant ``log(2)`` otherwise, in which case the term
    contributes no gradient.
    """
    a_lag = a_utk[:, :-1, :]                     # (B, L-1, K)
    a_now = a_utk[:, 1:, :]                      # (B, L-1, K)
    pair = a_lag * a_now                         # (B, L-1, K)

    use_real = (
        cfg is not None
        and getattr(cfg, "use_real_timestamp", False)
        and history_ts_long is not None
    )
    if use_real:
        ts = history_ts_long
        dt = (ts[:, 1:] - ts[:, :-1]).clamp(min=getattr(cfg, "ts_min_dt_floor", 1e-5))  # (B, L-1)
        if m_long is not None:
            valid_pair = m_long[:, 1:] & m_long[:, :-1]
            dt = dt * valid_pair.float()
        log_dt = torch.log1p(dt).unsqueeze(-1)   # (B, L-1, 1)
        num = (pair * log_dt).sum(dim=1)         # (B, K)
    else:
        num = (pair * math.log(2.0)).sum(dim=1)
    den = pair.sum(dim=1).clamp_min(eps)
    return num / den


def sid_consistency(c1_long, m_long, a_utk, sid1_size: int, eps=1e-8):
    """First-level SID consistency C^sid_{u,k}; returns (B, K)."""
    safe = c1_long.clamp(min=0, max=sid1_size - 1)
    y_one = F.one_hot(safe, num_classes=sid1_size).float() * m_long.float().unsqueeze(-1)  # (B,L,C)
    weights = a_utk.sum(dim=1).clamp_min(eps).unsqueeze(-1)                                # (B,K,1)
    y_bar = torch.einsum("blc,blk->bkc", y_one, a_utk) / weights                           # (B,K,C)
    # CE(y_t, ybar_k) = -sum_c y_{t,c} log ybar_{k,c}
    log_y_bar = y_bar.clamp_min(1e-9).log()                                                # (B,K,C)
    # ce_{t,k} aggregated with the assignment weights below.
    ce_tk = -torch.einsum("blc,bkc->blk", y_one, log_y_bar)                                # (B,L,K)
    num = (a_utk * ce_tk).sum(dim=1)
    den = a_utk.sum(dim=1).clamp_min(eps)
    return num / den


class StructureConstraints(nn.Module):
    """Compute all three statistics; ``losses.py`` combines them."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x_long, m_long, c1_long, a_utk, sid1_size, history_ts_long=None):
        sem = semantic_compactness(x_long, a_utk, self.cfg.eps)
        time = time_continuity(
            a_utk,
            history_ts_long=history_ts_long,
            m_long=m_long,
            cfg=self.cfg,
            eps=self.cfg.eps,
        )
        sid = sid_consistency(c1_long, m_long, a_utk, sid1_size, self.cfg.eps)
        return {"sem": sem, "time": time, "sid": sid}
