# -*- coding: utf-8 -*-
"""Item-level embeddings over flattened SID token sequences.

The data pipeline flattens every item into ``M`` SID tokens, so a history of ``L``
items arrives as a ``(B, L * M)`` token tensor (see ``model/dataset.py``). The
helpers here recover one embedding per *item*::

    x_{u,t} = aggregate_{m=1..M} Embed(token(level m of item t))

Two design choices matter:

* We reuse the backbone's ``shared`` embedding table rather than allocating a new
  one. This keeps the item semantics identical between the generative decoder and
  the subsequence modules, and adds no extra parameters.
* Levels are aggregated by summation by default, which matches the additive
  composition of the residual-quantized SID codes.

``split_recent_long`` implements the split between the long-term history ``H_u``
(partitioned into sub-sequences) and the recent window ``R_u`` (the ``r`` most
recent items, consumed directly by the encoder). Because sequences are left-padded,
the recent window is the right-hand slice.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def per_token_embed(shared: nn.Embedding, token_ids: torch.Tensor) -> torch.Tensor:
    """Embed a flattened SID token sequence with the backbone's shared table.

    Args:
        shared:    the backbone's ``nn.Embedding(vocab_size, d_model)``.
        token_ids: (B, L*M) long, already offset by ``_apply_offset_to_code``.

    Returns:
        (B, L*M, d_model) float tensor.
    """
    return shared(token_ids)


def per_item_embed(
    shared: nn.Embedding,
    token_ids: torch.Tensor,
    target_depth: int,
    pad_token_id: int = 0,
    aggregate: str = "sum",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate the ``M`` SID token embeddings of each item into ``x_{u,t}``.

    Args:
        token_ids:    (B, L*M) flattened, offset SID tokens.
        target_depth: number of SID levels per item (``M``).
        pad_token_id: padding token id.
        aggregate:    ``"sum"`` (default) or ``"mean"``.

    Returns:
        x_ut:      (B, L, d_model) per-item embeddings.
        item_mask: (B, L) bool, True where the item is not entirely padding.
    """
    B, LM = token_ids.shape
    assert LM % target_depth == 0, \
        f"token length {LM} is not a multiple of target_depth={target_depth}"
    L = LM // target_depth

    tok_emb = shared(token_ids)                                  # (B, L*M, D)
    tok_emb = tok_emb.view(B, L, target_depth, -1)               # (B, L, M, D)

    if aggregate == "sum":
        x_ut = tok_emb.sum(dim=2)                                # (B, L, D)
    elif aggregate == "mean":
        x_ut = tok_emb.mean(dim=2)
    else:
        raise ValueError(f"unknown aggregate mode: {aggregate}")

    tokens = token_ids.view(B, L, target_depth)
    item_mask = (tokens != pad_token_id).any(dim=2)              # (B, L)
    return x_ut, item_mask


def split_recent_long(
    x_ut: torch.Tensor,
    item_mask: torch.Tensor,
    recent_window_r: int,
):
    """Split per-item embeddings into long-term history and recent window.

    Args:
        x_ut:            (B, L, D) per-item embeddings.
        item_mask:       (B, L) validity mask.
        recent_window_r: size ``r`` of the recent window.

    Returns:
        ``(x_long, m_long, x_recent, m_recent)`` with shapes
        ``(B, L-r, D) / (B, L-r) / (B, r, D) / (B, r)``.
    """
    B, L, D = x_ut.shape
    r = min(recent_window_r, L)
    L_long = L - r
    if L_long <= 0:
        # Degenerate case: the history is shorter than the recent window, so the
        # whole sequence is treated as recent and the long-term part is empty.
        x_long = x_ut.new_zeros(B, 1, D)
        m_long = item_mask.new_zeros(B, 1, dtype=torch.bool)
        return x_long, m_long, x_ut, item_mask

    x_long = x_ut[:, :L_long, :]
    x_recent = x_ut[:, L_long:, :]
    m_long = item_mask[:, :L_long]
    m_recent = item_mask[:, L_long:]
    return x_long, m_long, x_recent, m_recent
