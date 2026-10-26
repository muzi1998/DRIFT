# -*- coding: utf-8 -*-
"""Attention encoder that turns soft sub-sequence assignments into representations.

Each historical sub-sequence ``k`` must be represented **only** by its own member
items, otherwise the sub-sequence representations collapse into a single global
summary. We enforce this in a fully differentiable way: the soft assignment
``y_{u,t,k}`` produced by ``partition.SoftPartition`` is used as a per-member
weight when aggregating keys/values for sub-sequence ``k``, so items belonging to
other sub-sequences (or padding) are suppressed rather than hard-masked.

Three aggregation variants are provided:

* ``"linear"`` (default) — membership-weighted linear attention with an
  ``elu(x) + 1`` feature map. Cost is ``O(L * K * d)`` and it avoids materializing
  the ``L x L`` attention matrix, which matters for the long histories
  (up to 1000 items) used in our experiments.
* ``"dense"``  — membership-weighted scaled dot-product attention, where the
  log-membership is added to the scores as a soft mask.
* ``"sparse"`` — same as ``dense``, plus a per-query top-k restriction.

All variants return ``(h_uk, weight)`` with ``h_uk`` of shape ``(B, K, D)`` and
``weight`` of shape ``(B, L_h, K)``, the normalized per-item contribution that can
be inspected for interpretability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubseqAttnEncoder(nn.Module):
    """Learn one representation ``h_{u,k}`` per historical sub-sequence."""

    def __init__(self, cfg, d_model: int, attn_kind: str = "linear",
                 num_heads: int = 4, topk: int = 8):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K
        self.d_model = d_model
        self.attn_kind = attn_kind
        assert d_model % num_heads == 0, \
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.topk = topk
        self.eps = cfg.eps

        # One learnable query per candidate sub-sequence.
        self.stage_query = nn.Parameter(torch.randn(self.K, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def _split(self, x, B, N):
        """(B, N, D) -> (B, H, N, head_dim)."""
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x_long, m_long, y_utk):
        """Aggregate member items into per-sub-sequence representations.

        Args:
            x_long: (B, L_h, D) long-term history item embeddings.
            m_long: (B, L_h) bool, True for valid (non-padding) positions.
            y_utk:  (B, L_h, K) soft assignment from the partition module.

        Returns:
            h_uk:   (B, K, D) sub-sequence representations (layer-normalized).
            weight: (B, L_h, K) normalized membership weights.
        """
        B, L_h, D = x_long.shape
        K, H, hd = self.K, self.num_heads, self.head_dim

        # Membership weight w_{t,k} = y_{u,t,k} * valid_mask.
        w_tk = (y_utk * m_long.float().unsqueeze(-1)).clamp_min(0.0)

        k = self._split(self.k_proj(x_long), B, L_h)          # (B, H, L, hd)
        v = self._split(self.v_proj(x_long), B, L_h)          # (B, H, L, hd)
        q = self.q_proj(self.stage_query)                     # (K, D)
        q = q.view(1, K, H, hd).permute(0, 2, 1, 3).expand(B, H, K, hd)   # (B, H, K, hd)

        # Membership weights are shared across heads.
        w_bhlk = w_tk.permute(0, 2, 1).unsqueeze(1)           # (B, 1, K, L)

        if self.attn_kind == "linear":
            # Membership-weighted linear attention:
            #   KV_k = sum_t w_{t,k} * phi(k_t) (x) v_t
            qf = F.elu(q) + 1.0                               # (B, H, K, hd)
            kf = F.elu(k) + 1.0                               # (B, H, L, hd)
            w_bhkl = w_bhlk.expand(B, H, K, L_h)              # (B, H, K, L)
            kv = torch.einsum("bhkl,bhld,bhle->bhkde", w_bhkl, kf, v)
            k_sum = torch.einsum("bhkl,bhld->bhkd", w_bhkl, kf)            # (B, H, K, hd)
            z = torch.einsum("bhkd,bhkd->bhk", qf, k_sum).clamp_min(self.eps)
            out = torch.einsum("bhkd,bhkde->bhke", qf, kv) / z.unsqueeze(-1)
        else:
            # dense / sparse: dot-product scores plus a log-membership soft mask.
            scores = torch.einsum("bhkd,bhld->bhkl", q, k) / math.sqrt(hd)  # (B, H, K, L)
            log_w = torch.log(w_bhlk.expand(B, H, K, L_h).clamp_min(self.eps))
            scores = scores + log_w
            if self.attn_kind == "sparse":
                kk = min(self.topk, L_h)
                tv, _ = torch.topk(scores, kk, dim=-1)
                thr = tv[..., -1, None]
                scores = scores.masked_fill(scores < thr,
                                            torch.finfo(scores.dtype).min / 4)
            attn = F.softmax(scores, dim=-1)                                # (B, H, K, L)
            out = torch.einsum("bhkl,bhld->bhkd", attn, v)                  # (B, H, K, hd)

        h_uk = out.transpose(1, 2).contiguous().view(B, K, D)
        h_uk = self.layer_norm(h_uk)

        # Normalized membership weights, useful for interpretability.
        weight = w_tk / w_tk.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return h_uk, weight


def build_subseq_encoder(kind: str, cfg, d_model: int,
                         num_heads: int = 4, topk: int = 8):
    """Factory for the intra-sub-sequence encoder.

    ``kind`` values:
        ``"linear_fusion"`` -> ``None``; the caller then falls back to the simple
            weighted linear fusion implemented in ``IntraSubsequenceEncoder``.
        ``"linear_mha"`` / ``"mha"`` / ``"sparse_mha"`` -> ``SubseqAttnEncoder``
            with attention kind ``linear`` / ``dense`` / ``sparse``.
    """
    kind = (kind or "linear_fusion").lower()
    if kind == "linear_fusion":
        return None
    amap = {"linear_mha": "linear", "mha": "dense", "sparse_mha": "sparse"}
    if kind not in amap:
        raise ValueError(f"unknown intra-sub-sequence encoder kind: {kind}")
    return SubseqAttnEncoder(cfg, d_model, attn_kind=amap[kind],
                             num_heads=num_heads, topk=topk)
