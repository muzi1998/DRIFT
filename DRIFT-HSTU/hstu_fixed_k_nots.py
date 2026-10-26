"""Fixed K subsequence slots for Taobao MM, without time features.

The original NoTS encoder is preserved. This variant removes transitions from
padding, masks padding out of aggregation, and returns zeros for empty histories.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from run_hstu_drift_c_v5_taobaomm import SubseqRepresentationV3NoTS


class FixedKSubseqRepresentationV5NoTS(SubseqRepresentationV3NoTS):
    def forward(self, x_long, mask_long, d_u0):
        batch, length, dim = x_long.shape
        if self.K_max < 1 or mask_long.shape != (batch, length):
            raise ValueError('Invalid K or mask shape')
        if d_u0.shape != (batch,):
            raise ValueError('d_u0 must have shape (batch,)')
        mask_long = mask_long.bool()
        x_long = x_long.masked_fill(~mask_long.unsqueeze(-1), 0.0)
        slots = torch.ones(batch, self.K_max, device=x_long.device, dtype=torch.bool)
        if length == 0:
            return (x_long.new_zeros(batch, self.K_max, dim), slots,
                    x_long.new_zeros(batch, 0, self.K_max), x_long)

        cosine = F.cosine_similarity(x_long[:, 1:], x_long[:, :-1], dim=-1, eps=1e-6)
        cosine = torch.cat((x_long.new_zeros(batch, 1), 1.0-cosine), dim=1)
        diff = torch.cat((x_long.new_zeros(batch, 1, dim),
                          x_long[:, 1:]-x_long[:, :-1]), dim=1)
        features = torch.cat((cosine.unsqueeze(-1), diff,
                              d_u0[:, None, None].expand(batch, length, 1)), dim=-1)
        transition = torch.cat((mask_long.new_zeros(batch, 1),
                                mask_long[:, 1:] & mask_long[:, :-1]), dim=1)
        # The first real item never receives a boundary from preceding padding.
        boundary = torch.sigmoid(self.mlp_b(features).squeeze(-1))*transition.to(x_long.dtype)
        cumulative = boundary.cumsum(dim=1)
        position = 1.0+(self.K_max-1)*cumulative/cumulative[:, -1:].clamp_min(1e-6)
        kappa = x_long.new_full((batch, 1), float(self.K_max))
        slot_gate = torch.sigmoid((kappa-self.mu_k[None, :]+0.5)/self.tau_m)
        unnormalized = torch.exp(-(position[..., None]-self.mu_k[None, None, :])**2/self.tau_a)
        unnormalized = unnormalized*slot_gate[:, None, :]
        assignment = unnormalized/(unnormalized.sum(-1, keepdim=True)+1e-8)
        assignment = assignment*mask_long[..., None].to(x_long.dtype)
        values = self.W_v(x_long)
        logits = self.w_in(torch.tanh(self.W_in(x_long)))+assignment.clamp_min(1e-8).log()
        logits = logits.masked_fill(~mask_long[..., None], torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1)*mask_long[..., None].to(logits.dtype)
        representations = self.layer_norm(torch.einsum('bld,blk->bkd', values, weights))
        representations = self.ffn_norm(representations+self.ffn(representations))
        attended, _ = self.intra_attn(representations, representations, representations,
                                      need_weights=False)
        representations = self.intra_norm(representations+attended)
        representations = representations.masked_fill(~mask_long.any(1)[:, None, None], 0.0)
        return representations, slots, assignment, x_long
