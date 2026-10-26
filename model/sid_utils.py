# -*- coding: utf-8 -*-
"""Semantic-ID (SID) token helpers.

The generative backbone flattens each item into ``M`` SID tokens (``M = 3`` in all
our experiments). Token offsets follow the convention used by the data pipeline
(``model/dataset.py::GenRecDataset._apply_offset_to_code``)::

    token(level i, code c) = c + i * CODEBOOK_SIZE + 1,     PAD -> 0

Hence the first-level (coarsest) code of an item is recovered from its first token
by subtracting the offset ``1``. The first-level code is the categorical proxy
used by the subsequence structure constraints.
"""

from __future__ import annotations

import torch


def decode_level1_codes(token_ids: torch.Tensor, target_depth: int,
                        codebook0: int, pad_token_id: int):
    """Recover first-level SID codes from a flattened token sequence.

    Args:
        token_ids:    (B, L * M) flattened, offset SID tokens.
        target_depth: number of SID levels per item (``M``).
        codebook0:    size of the first-level codebook.
        pad_token_id: padding token id (0 by convention).

    Returns:
        codes: (B, L) first-level codes in ``[0, codebook0)``, ``-1`` for padding.
        valid: (B, L) boolean mask, ``True`` for real (non-padding) positions.
    """
    batch_size, flat_len = token_ids.shape
    seq_len = flat_len // target_depth
    tokens = token_ids.view(batch_size, seq_len, target_depth)

    level1_tok = tokens[:, :, 0]                                  # (B, L)
    is_pad = level1_tok == pad_token_id
    codes = (level1_tok - 1).clamp(min=0, max=codebook0 - 1)
    codes = torch.where(is_pad, torch.full_like(codes, -1), codes)
    return codes, ~is_pad
