# -*- coding: utf-8 -*-
"""Sub-sequence structure regularizer.

The structure loss keeps each learned sub-sequence internally coherent. Given the
per-sub-sequence statistics produced by
``subsequence/structure_constraints.StructureConstraints``, it forms::

    L_sub = (1/B) * sum_u sum_k m_{u,k} * ( C^sem_{u,k}
                                          + lambda_t * C^time_{u,k}
                                          + lambda_c * C^sid_{u,k} )

where ``m_{u,k}`` is the activity gate, so inactive sub-sequences of users with a
small ``kappa_u`` are not penalized.

Two conventions are worth stating explicitly, since they determine how the
reported weights should be interpreted:

* The semantic term ``C^sem`` carries an implicit weight of ``1.0``; only the
  temporal and SID-consistency terms are rescaled by ``lambda_t`` / ``lambda_c``.
  The overall strength of the whole term is controlled by ``--lambda_sub`` in the
  training entry scripts.
* The sum over ``k`` is **not** divided by ``K``; normalization is over the batch
  dimension only.

When the dataset provides no real timestamps, ``C^time`` is a constant and thus
contributes no gradient, which effectively disables the temporal part.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_subseq_structure(stats_dict, m_uk, lam_t=0.5, lam_c=0.5):
    """Weighted sub-sequence structure loss.

    Args:
        stats_dict: dict with ``"sem"``, ``"time"``, ``"sid"``, each ``(B, K)``.
        m_uk:       (B, K) activity gate of each candidate sub-sequence.
        lam_t:      weight of the temporal-continuity term.
        lam_c:      weight of the first-level SID consistency term.

    Returns:
        Scalar loss, averaged over the batch (summed over ``K``).
    """
    sem = stats_dict["sem"]                                    # (B, K)
    time = stats_dict["time"]                                  # (B, K)
    sid = stats_dict["sid"]                                    # (B, K)
    weighted = m_uk * (sem + lam_t * time + lam_c * sid)       # (B, K)
    return weighted.sum(dim=1).mean()


def loss_contrastive(z_u: torch.Tensor, e_target: torch.Tensor, tau=0.07):
    """In-batch InfoNCE between a user representation and the target item.

    Kept as a minimal reference implementation; the training entries use the
    equivalent ``rec_losses.loss_rec_inbatch``, which additionally handles the
    choice of item representation.
    """
    z = F.normalize(z_u, dim=-1)
    e = F.normalize(e_target, dim=-1)
    logits = (z @ e.t()) / tau
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, labels)
