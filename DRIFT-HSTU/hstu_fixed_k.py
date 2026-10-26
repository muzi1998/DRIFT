"""Fixed-K version of the v5 subsequence encoder for length experiments.

The boundary network still consumes user drift and the actual time gaps.  K is
the number of active subsequence slots, rather than an adaptive upper bound.
The original v5 module and its parameters are left unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from run_hstu_drift_c_v5 import SubseqRepresentationV3


class FixedKSubseqRepresentationV5(SubseqRepresentationV3):
    """Use exactly ``K_max`` active slots with v5's learned boundaries.

    Padding is excluded from item-to-slot attention.  For an entirely empty
    long history, the returned representations and assignments are zero.  Its
    slot mask stays valid, allowing downstream cross-attention to consume
    those zeros without an all-masked softmax.
    """

    def forward(self, x_long, mask_long, d_u0, log_dt=None):
        batch_size, long_len, hidden_dim = x_long.shape
        if self.K_max < 1:
            raise ValueError("K_max must be positive")
        if mask_long.shape != (batch_size, long_len):
            raise ValueError("mask_long must have shape (batch_size, long_len)")
        if d_u0.shape != (batch_size,):
            raise ValueError("d_u0 must have shape (batch_size,)")

        device = x_long.device
        mask_long = mask_long.bool()
        x_long = x_long.masked_fill(~mask_long.unsqueeze(-1), 0.0)
        slot_mask = torch.ones(
            (batch_size, self.K_max), dtype=torch.bool, device=device
        )
        if long_len == 0:
            return (
                x_long.new_zeros(batch_size, self.K_max, hidden_dim),
                slot_mask,
                x_long.new_zeros(batch_size, 0, self.K_max),
                x_long,
            )

        # The only change to the boundary construction is a fixed kappa.
        # d_u0 remains an input to mlp_b, as in the original v5 encoder.
        kappa = x_long.new_full((batch_size,), float(self.K_max))
        cos_pair = F.cosine_similarity(
            x_long[:, 1:, :], x_long[:, :-1, :], dim=-1, eps=1e-6
        )
        cos_pair = torch.cat(
            [x_long.new_zeros(batch_size, 1), 1.0 - cos_pair], dim=1
        )
        if log_dt is None:
            log_dt_long = x_long.new_full((batch_size, long_len), math.log(2.0))
        else:
            if log_dt.ndim != 2 or log_dt.shape[0] != batch_size or log_dt.shape[1] < long_len:
                raise ValueError("log_dt must provide at least long_len time gaps per user")
            log_dt_long = log_dt[:, -long_len:]
        difference = torch.cat(
            [x_long.new_zeros(batch_size, 1, hidden_dim),
             x_long[:, 1:, :] - x_long[:, :-1, :]],
            dim=1,
        )
        drift_features = d_u0[:, None, None].expand(batch_size, long_len, 1)
        boundary_features = torch.cat(
            [cos_pair.unsqueeze(-1), log_dt_long.unsqueeze(-1),
             difference, drift_features],
            dim=-1,
        )
        boundary = torch.sigmoid(self.mlp_b(boundary_features).squeeze(-1))
        valid_transition = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
             mask_long[:, 1:] & mask_long[:, :-1]],
            dim=1,
        )
        boundary = boundary * valid_transition.to(x_long.dtype)

        cumulative = boundary.cumsum(dim=1)
        total = cumulative[:, -1:].clamp_min(1e-6)
        slot_position = 1.0 + (kappa[:, None] - 1.0) * cumulative / total
        slot_index = torch.arange(
            1, self.K_max + 1, device=device, dtype=x_long.dtype
        )
        # Preserve both original temperatures and the soft slot weights.
        # With kappa=K every slot has gate > 0.5, so all K are active.
        slot_gate = torch.sigmoid(
            (kappa[:, None] - slot_index[None, :] + 0.5) / self.tau_m
        )
        distance = (slot_position.unsqueeze(-1) - self.mu_k.view(1, 1, -1)) ** 2
        unnormalized = torch.exp(-distance / self.tau_a) * slot_gate.unsqueeze(1)
        assignments = unnormalized / (unnormalized.sum(dim=-1, keepdim=True) + 1e-8)
        assignments = assignments * mask_long.unsqueeze(-1).to(x_long.dtype)

        values = self.W_v(x_long)
        attention_logits = self.w_in(torch.tanh(self.W_in(x_long)))
        attention_logits = attention_logits + torch.log(assignments.clamp_min(1e-8))
        # A finite minimum also makes all-padding rows differentiable and safe.
        attention_logits = attention_logits.masked_fill(
            ~mask_long.unsqueeze(-1), torch.finfo(attention_logits.dtype).min
        )
        item_weights = F.softmax(attention_logits, dim=1)
        item_weights = item_weights * mask_long.unsqueeze(-1).to(item_weights.dtype)
        representations = torch.einsum("bld,blk->bkd", values, item_weights)
        representations = self.layer_norm(representations)
        representations = self.ffn_norm(representations + self.ffn(representations))
        attended, _ = self.intra_attn(
            representations, representations, representations, need_weights=False
        )
        representations = self.intra_norm(representations + attended)
        # LayerNorm/FFN biases must not fabricate an interest for empty users.
        representations = representations.masked_fill(
            ~mask_long.any(dim=1)[:, None, None], 0.0
        )
        return representations, slot_mask, assignments, x_long
