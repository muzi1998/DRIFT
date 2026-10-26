# -*- coding: utf-8 -*-
"""In-batch recommendation losses.

The generative objective alone supervises SID tokens, not the user representation
itself. We therefore add in-batch softmax (InfoNCE) terms that pull a pooled user
representation towards its target item and push it away from the other targets in
the same batch::

    L = -(1/B) sum_i log[ exp(s(z_i, e_i) / tau) / sum_j exp(s(z_i, e_j) / tau) ]

where ``s`` is the cosine similarity (both sides are L2-normalized) and ``tau`` is
the temperature. This is used twice in the training entries:

* ``L_rec``     — with ``z_u`` pooled from the encoder states, which already carry
  the sub-sequence information injected by cross-attention.
* ``L_subrec``  — with ``z_sub`` pooled from the sub-sequence representations
  ``h_{u,k}``, giving the partition module a direct recommendation signal.

Item representation
-------------------
``item_repr="first"`` (the default used in all reported results) represents the
target item by the embedding of its **first-level** SID code only. This focuses the
contrastive signal on the coarsest, most discriminative level, which is also the
level that dominates beam-search errors. ``item_repr="mlp"`` instead fuses all
three level embeddings with ``ItemFusionMLP``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemFusionMLP(nn.Module):
    """Fuse the three level embeddings of an item into a single vector."""

    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1,
                 num_layers: int = 2):
        super().__init__()
        in_dim = 3 * d_model
        hidden = hidden_mult * d_model
        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, d_model))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, e0: torch.Tensor, e1: torch.Tensor,
                e2: torch.Tensor) -> torch.Tensor:
        """Args ``e0``, ``e1``, ``e2``: (B, D) level embeddings -> (B, D)."""
        x = torch.cat([e0, e1, e2], dim=-1)   # (B, 3D)
        return self.net(x)


def loss_rec_inbatch(
    z_u: torch.Tensor,
    labels: torch.Tensor,
    shared_emb: nn.Embedding,
    tau: float = 0.07,
    item_repr: str = "first",
    item_mlp: Optional[ItemFusionMLP] = None,
) -> torch.Tensor:
    """In-batch softmax recommendation loss.

    Equivalent to ``F.cross_entropy(logits, arange(B))`` with
    ``logits = normalize(z) @ normalize(e).T / tau``.

    Args:
        z_u:        (B, D) pooled user representation.
        labels:     (B, M) target item SID tokens (already offset).
        shared_emb: the backbone's shared embedding table.
        tau:        softmax temperature.
        item_repr:  ``"first"`` (use only the first-level code) or ``"mlp"``.
        item_mlp:   required when ``item_repr="mlp"``.

    Returns:
        Scalar loss.
    """
    # 1) Build the item representation e_j.
    if item_repr == "first":
        # First-level focus: labels[:, 0] is already an offset token id.
        e = shared_emb(labels[:, 0])                         # (B, D)
    elif item_repr == "mlp":
        assert item_mlp is not None, "item_repr='mlp' requires item_mlp"
        e0 = shared_emb(labels[:, 0])                        # (B, D)
        e1 = shared_emb(labels[:, 1])
        e2 = shared_emb(labels[:, 2])
        e = item_mlp(e0, e1, e2)                             # (B, D)
    else:
        raise ValueError(f"item_repr must be 'first' or 'mlp', got: {item_repr}")

    # 2) L2-normalize both sides (scaled cosine similarity).
    z = F.normalize(z_u, dim=-1)     # (B, D)
    e = F.normalize(e, dim=-1)       # (B, D)

    # 3) In-batch logits (B, B); the diagonal holds the positive pairs.
    logits = (z @ e.t()) / tau
    target = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, target)
