#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared training / inference core for both encoder-prefix variants.

Method overview
===============
A user history is split into a long-term part ``H_u`` and a recent window ``R_u``
(the ``r`` most recent items). The long-term part is adaptively partitioned into
``K`` preference sub-sequences and encoded into ``h_{u,k}``; the recent window is
fed to the generative encoder. The sub-sequence representations are injected in two
places:

1. **Inside every encoder block**, as an extra cross-attention between the
   self-attention and the FFN::

       baseline block:  Self-Attn -> FFN
       ours:            Self-Attn -> Cross-Attn(h_uk) -> FFN

   This is what makes the encoder output ``H_enc*`` carry the sub-sequence
   structure, so the pooled representation used by ``L_rec`` becomes discriminative
   with respect to the learned partition.

2. **In the decoder**, as the second cross-attention path of the dual-aware block,
   fused with the encoder memory by a learned gate::

       CA1 = CrossAttn(Q=h_dec, K/V=H_enc*)
       CA2 = CrossAttn(Q=h_dec, K/V=h_uk)
       g   = sigmoid(W [h_dec ; CA2])
       out = (1 - g) * CA1 + g * CA2

Data flow::

    prefix -> Encoder( Self-Attn -> Cross-Attn(h_uk) -> FFN ) -> H_enc*
      h_uk /                                                      |
    Decoder: CA1(K/V=H_enc*) + CA2(K/V=h_uk) -> gate -> SID logits

The two entry scripts differ **only** in the encoder prefix:

* ``train_drift_sidtier.py`` — prefix ``[sidtier ; recent]``
* ``train_drift_recent.py``  — prefix ``[recent]``

Training objective
==================
::

    L_total = L_CE
            + lambda_rec    * L_rec         # InfoNCE(pool(H_enc*), target)
            + lambda_subrec * L_subrec      # InfoNCE(pool(h_uk),   target)
            + lambda_sub    * L_sub         # sub-sequence structure regularizer

``L_CE`` is the SID token cross-entropy (implicit weight 1.0). Note that inside
``L_sub`` the semantic term also carries an implicit weight of 1.0, while
``--sub_lam_t`` / ``--sub_lam_c`` reweight only its temporal and SID-consistency
parts (see ``drift_method/losses.py``).

Only the backbone modules are reused here; no file outside this package is
modified at runtime.
"""

import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import paths

paths.ensure_import_paths()

# ---- Generative backbone (imported, never executed as a script) ----
import glass_backbone as gl

GLASS = gl.GLASS
GenRecDataset = gl.GenRecDataset
GenRecDataLoader = gl.GenRecDataLoader
set_seed = gl.set_seed
T5LayerNorm = gl.T5LayerNorm
T5Attention = gl.T5Attention

# ---- Preference-shift subsequence modules ----
from drift_method.config import DriftConfig
from drift_method.embedding.item_embedding import per_item_embed, split_recent_long
from drift_method.losses import loss_subseq_structure
from drift_method.rec_losses import loss_rec_inbatch
from drift_method.subsequence.intra_encoder import IntraSubsequenceEncoder
from drift_method.subsequence.partition import SoftPartition
from drift_method.subsequence.structure_constraints import StructureConstraints

# ---- Data helpers ----
from model.sid_utils import decode_level1_codes
from model.timestamped_data import (
    load_timestamps_into_dataset,
    make_timestamped_dataset_class,
)


# ============================================================
# Sub-sequence representation branch
# ============================================================
class SubseqRepresentation(nn.Module):
    """Adaptive partition + intra-sub-sequence encoding.

    Runs alongside the generative backbone and produces, for every user:

        h_uk:  (B, K, D)     sub-sequence representations
        m_uk:  (B, K)        activity gate (how many sub-sequences are active)
        a_utk: (B, L_h, K)   soft assignment, used by the structure constraints
    """

    def __init__(self, cfg: DriftConfig, shared_emb: nn.Embedding, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        # The backbone embedding table is shared, adding no extra item parameters.
        self.shared = shared_emb
        self.partition = SoftPartition(cfg, d_model)
        self.intra_enc = IntraSubsequenceEncoder(cfg, d_model)
        self.struct = StructureConstraints(cfg)

    def forward(self, history_token_ids, history_ts_long=None):
        cfg = self.cfg
        sid1 = cfg.codebook[0]
        device = history_token_ids.device

        # Per-item embeddings and first-level SID codes.
        x_full, m_full = per_item_embed(
            self.shared, history_token_ids,
            target_depth=cfg.target_depth, pad_token_id=cfg.pad_token_id,
            aggregate="sum")
        c1_full, _ = decode_level1_codes(
            history_token_ids, cfg.target_depth, sid1, cfg.pad_token_id)

        # Split off the recent window; only the long-term part is partitioned.
        x_long, m_long, _x_recent, _m_recent = split_recent_long(
            x_full, m_full, cfg.recent_window_r)
        L = x_full.size(1)
        r = min(cfg.recent_window_r, L)
        L_long = max(L - r, 1)
        c1_long = c1_full[:, :L_long].contiguous()
        if x_long.size(1) == 0:
            # Degenerate history: keep one position so shapes stay valid.
            x_long = x_full[:, :1, :].contiguous()
            m_long = m_full[:, :1, :].contiguous()
            c1_long = c1_full[:, :1, :].contiguous()

        B = x_long.size(0)
        # Shift degree placeholder; the partition module learns the boundaries
        # end-to-end, so no externally precomputed drift score is required.
        d0 = torch.zeros(B, device=device)

        ts_long = None
        if history_ts_long is not None:
            ts_long = history_ts_long[:, :x_long.size(1)].contiguous()

        part = self.partition(x_long, m_long, c1_long, d0, sid1,
                              history_ts_long=ts_long)
        h_uk, _weight = self.intra_enc(x_long, m_long, part["y_utk"], d0,
                                       history_ts_long=ts_long)

        return {
            "h_uk": h_uk,
            "m_uk": part["m_uk"],
            "a_utk": part["a_utk"],
            "kappa": part["kappa"],
            "x_long": x_long,
            "m_long": m_long,
            "c1_long": c1_long,
            "sid1_size": sid1,
        }


# ============================================================
# Cross-attention module inserted into each encoder block
# ============================================================
class HukCrossAttn(nn.Module):
    """Cross-attention over ``h_uk``, placed between Self-Attn and FFN.

    Reuses the backbone's ``T5LayerNorm`` / ``T5Attention`` so the injected path
    matches the rest of the encoder. Forward pass::

        h' = h + Dropout( CrossAttn( LayerNorm(h), K=h_uk, V=h_uk ) )

    Added parameters: ``num_encoder_layers`` copies of
    (LayerNorm + 4 projections), i.e. roughly
    ``4 * num_layers * d_model * inner_dim`` weights (~0.4M in our setting).
    """

    def __init__(self, d_model, num_heads, d_kv, dropout_rate):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model)
        self.cross_attn = T5Attention(d_model, num_heads, d_kv, dropout_rate,
                                      has_relative_bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, h_uk, huk_mask):
        """hidden_states: (B, S, D); h_uk: (B, K, D); huk_mask: (B, 1, 1, K)."""
        normed = self.layer_norm(hidden_states)
        ca_out, _ = self.cross_attn(normed, kv_states=h_uk, mask=huk_mask)
        return hidden_states + self.dropout(ca_out)


# ============================================================
# Encoder forward with h_uk injection
# ============================================================
def encode_with_huk_crossattn(raw_model, huk_cross_modules,
                              input_ids, attention_mask, h_uk,
                              sidtier=None, use_sidtier_prefix=True,
                              recent_token_len=None):
    """Run the encoder while injecting ``h_uk`` inside every block.

    The encoder block forward pass is unrolled manually so the extra
    cross-attention can be inserted between Self-Attn and FFN; all existing block
    weights are reused unchanged.

    Args:
        raw_model:          ``GLASS`` instance (unwrapped from DataParallel).
        huk_cross_modules:  ``nn.ModuleList`` of ``HukCrossAttn``, one per layer.
        input_ids:          (B, S) history SID tokens.
        attention_mask:     (B, S).
        h_uk:               (B, K, D) sub-sequence representations.
        sidtier:            (B, sidtier_dim) SID-tier feature, or None.
        use_sidtier_prefix: True  -> prefix ``[sidtier ; recent]``
                            False -> prefix ``[recent]``
        recent_token_len:   keep only the last ``recent_token_len`` tokens.

    Returns:
        ``(hidden_states, extended_mask, attn_mask)``.
    """
    # 0) Restrict the encoder to the recent window.
    if recent_token_len is not None and input_ids.size(1) > recent_token_len:
        input_ids = input_ids[:, -recent_token_len:]
        attention_mask = attention_mask[:, -recent_token_len:]

    B = input_ids.size(0)
    device = input_ids.device

    # 1) Recent-item embeddings.
    recent_emb = raw_model.shared(input_ids)

    # 2) Build the encoder prefix. h_uk is deliberately NOT concatenated here; it
    #    only enters through the in-block cross-attention, which avoids making the
    #    prefix redundant with the SID-tier token.
    prefix_tokens, prefix_masks = [], []
    if use_sidtier_prefix and raw_model.use_sidtier and sidtier is not None:
        sidtier_emb = raw_model.sidtier_mlp(sidtier).unsqueeze(1)   # (B, 1, D)
        prefix_tokens.append(sidtier_emb)
        prefix_masks.append(torch.ones((B, 1), device=device,
                                       dtype=attention_mask.dtype))

    if prefix_tokens:
        enc_in = torch.cat(prefix_tokens + [recent_emb], dim=1)
        enc_attn_mask = torch.cat(prefix_masks + [attention_mask], dim=1)
    else:
        enc_in = recent_emb
        enc_attn_mask = attention_mask

    # 3) All K sub-sequence slots are attendable; the activity gate m_uk already
    #    down-weights inactive ones inside h_uk itself.
    K_sub = h_uk.size(1)
    huk_attn_mask = torch.ones(B, K_sub, device=device, dtype=enc_in.dtype)
    huk_mask_ext = raw_model.get_extended_attention_mask(huk_attn_mask, enc_in.dtype)

    # 4) Unrolled encoder stack with the injected cross-attention.
    hidden_states = raw_model.encoder_dropout(enc_in)
    extended_mask = raw_model.get_extended_attention_mask(enc_attn_mask,
                                                          hidden_states.dtype)
    position_bias = None

    for i, block in enumerate(raw_model.encoder_blocks):
        # (a) Self-attention, reusing the block's own weights.
        normed = block.layer_norm1(hidden_states)
        attn_out, position_bias = block.self_attn(normed, mask=extended_mask,
                                                  position_bias=position_bias)
        hidden_states = hidden_states + block.dropout1(attn_out)

        # (b) Injected cross-attention over the sub-sequence representations.
        hidden_states = huk_cross_modules[i](hidden_states, h_uk, huk_mask_ext)

        # (c) Feed-forward, reusing the block's own weights.
        normed = block.layer_norm3(hidden_states)
        ff_out = block.ff_out(block.ff_dropout(block.act(block.ff_in(normed))))
        hidden_states = hidden_states + block.dropout3(ff_out)

    hidden_states = raw_model.encoder_norm(hidden_states)
    return hidden_states, extended_mask, enc_attn_mask


# ============================================================
# Decoder: second cross-attention path uses h_uk
# ============================================================
def decode_logits_with_huk(raw_model, labels, encoder_hidden_states, encoder_mask,
                           h_uk, huk_mask):
    """Teacher-forced decoding; the gated second path attends to ``h_uk``."""
    decoder_input_ids = raw_model._shift_right(labels)
    hidden_states = raw_model.shared(decoder_input_ids)
    hidden_states = raw_model.decoder_dropout(hidden_states)
    seq_len = decoder_input_ids.shape[1]
    causal_mask = raw_model.get_causal_mask(seq_len, hidden_states.dtype,
                                            hidden_states.device)
    position_bias = None
    for block in raw_model.decoder_blocks:
        hidden_states, position_bias, _gate = block(
            hidden_states, mask=causal_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_mask=encoder_mask,
            encoder_hidden_states_sim=h_uk,
            encoder_mask_sim=huk_mask,
            position_bias=position_bias)
    hidden_states = raw_model.decoder_norm(hidden_states)
    return raw_model.lm_head(hidden_states)


@torch.no_grad()
def generate_with_huk(raw_model, huk_cross_modules, input_ids, attention_mask,
                      h_uk, sidtier=None, use_sidtier_prefix=True,
                      beam_size=20, max_length=4, recent_token_len=None):
    """Beam search over the three SID levels with ``h_uk`` injection."""
    raw_model.eval()
    device = input_ids.device
    B = input_ids.size(0)
    K_beam = beam_size

    # 1) Encoder forward (prefix + in-block h_uk cross-attention).
    enc_hidden, enc_mask_ext, _ = encode_with_huk_crossattn(
        raw_model, huk_cross_modules, input_ids, attention_mask, h_uk,
        sidtier=sidtier, use_sidtier_prefix=use_sidtier_prefix,
        recent_token_len=recent_token_len)

    # 2) Mask for the decoder's h_uk path.
    K_sub = h_uk.size(1)
    huk_attn_mask = torch.ones(B, K_sub, device=device, dtype=enc_hidden.dtype)
    huk_mask_ext = raw_model.get_extended_attention_mask(huk_attn_mask,
                                                          enc_hidden.dtype)

    # 3) First decoding step from BOS.
    curr_ids = torch.full((B, 1), raw_model.bos_token_id, dtype=torch.long,
                          device=device)
    logits, _ = raw_model.forward_decoder(curr_ids, enc_hidden, enc_mask_ext,
                                          h_uk, huk_mask_ext)
    next_token_logits = logits[:, -1, :]
    topk_scores, topk_sids = torch.topk(
        torch.softmax(next_token_logits, dim=-1), K_beam, dim=-1)

    # 4) Expand every tensor along the beam dimension.
    enc_hidden = enc_hidden.repeat_interleave(K_beam, dim=0)
    enc_mask_ext = enc_mask_ext.repeat_interleave(K_beam, dim=0)
    h_uk_expanded = h_uk.repeat_interleave(K_beam, dim=0)
    huk_mask_expanded = huk_mask_ext.repeat_interleave(K_beam, dim=0)
    curr_ids = curr_ids.repeat_interleave(K_beam, dim=0)
    curr_ids = torch.cat([curr_ids, topk_sids.view(-1, 1)], dim=1)
    beam_scores = torch.log(topk_scores + 1e-9).view(-1)

    # 5) Remaining SID levels.
    for _step in range(max_length - 1):
        logits, _ = raw_model.forward_decoder(curr_ids, enc_hidden, enc_mask_ext,
                                              h_uk_expanded, huk_mask_expanded)
        next_token_logprobs = torch.log_softmax(logits[:, -1, :], dim=-1)
        next_scores = beam_scores.unsqueeze(1) + next_token_logprobs
        next_scores = next_scores.view(B, K_beam * raw_model.vocab_size)
        best_scores, best_indices = next_scores.topk(K_beam, dim=1)
        prev_beam_indices = best_indices // raw_model.vocab_size
        new_token_indices = best_indices % raw_model.vocab_size
        batch_base = (torch.arange(B, device=device) * K_beam).unsqueeze(1)
        global_indices = (batch_base + prev_beam_indices).view(-1)

        beam_scores = best_scores.view(-1)
        curr_ids = curr_ids[global_indices]
        curr_ids = torch.cat([curr_ids, new_token_indices.view(-1, 1)], dim=1)
        enc_hidden = enc_hidden[global_indices]
        enc_mask_ext = enc_mask_ext[global_indices]
        h_uk_expanded = h_uk_expanded[global_indices]
        huk_mask_expanded = huk_mask_expanded[global_indices]

    return curr_ids.view(B, K_beam, -1)


# ============================================================
# Evaluation
# ============================================================
def masked_mean_pool(hidden, mask_bool, eps=1e-8):
    """Mean-pool ``hidden`` over valid positions."""
    if mask_bool is None:
        return hidden.mean(dim=1)
    w = mask_bool.float().unsqueeze(-1)
    return (hidden * w).sum(dim=1) / (w.sum(dim=1) + eps)


def evaluate_huk(raw_model, huk_cross_modules, subseq, eval_loader, topk_list,
                 beam_size, device, use_sidtier_prefix=True,
                 recent_token_len=None):
    """Compute Recall@k, NDCG@k and SID prefix precision P(1), P(1,2), P(1,2,3)."""
    raw_model.eval()
    subseq.eval()
    huk_cross_modules.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    exact = {'pos_1': 0, 'pos_1_2': 0, 'pos_1_2_3': 0, 'total_samples': 0}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="  Evaluating", leave=False):
            input_ids = batch['history'].to(device)
            if input_ids.dim() == 3:
                input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            sidtier = batch.get('sidtier', None)
            if sidtier is not None:
                sidtier = sidtier.to(device)

            ts = batch.get('timestamps', None)
            if ts is not None:
                ts = ts.to(device)
            sub_out = subseq(input_ids, history_ts_long=ts)

            preds = generate_with_huk(
                raw_model, huk_cross_modules, input_ids, attention_mask,
                sub_out["h_uk"], sidtier=sidtier,
                use_sidtier_prefix=use_sidtier_prefix,
                beam_size=beam_size, max_length=4,
                recent_token_len=recent_token_len)

            # Drop BOS, keep the three SID levels.
            preds = preds[:, :, 1:4]
            labels = labels[:, 0:3]

            # A beam counts as a hit only if all three levels match.
            pos_index = torch.zeros((preds.shape[0], preds.shape[1]), dtype=torch.bool)
            for i in range(preds.shape[0]):
                cur_label = labels[i].tolist()
                for j in range(preds.shape[1]):
                    if preds[i, j].tolist() == cur_label:
                        pos_index[i, j] = True
                        break

            for k in topk_list:
                safe_k = min(k, pos_index.shape[1])
                recalls['Recall@' + str(k)].append(
                    pos_index[:, :safe_k].sum(dim=1).float().mean().item())
                ranks = torch.arange(1, pos_index.shape[-1] + 1)
                dcg = 1.0 / torch.log2(ranks + 1)
                dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float))
                ndcgs['NDCG@' + str(k)].append(
                    dcg[:, :safe_k].sum(dim=1).float().mean().item())

            # Prefix precision of the top-1 beam.
            top1_list = preds[:, 0, :].cpu().tolist()
            labels_list = labels.cpu().tolist()
            exact['total_samples'] += labels.size(0)
            for i in range(labels.size(0)):
                cur_pred, cur_label = top1_list[i], labels_list[i]
                if len(cur_pred) >= 1 and cur_pred[0] == cur_label[0]:
                    exact['pos_1'] += 1
                    if len(cur_pred) >= 2 and cur_pred[1] == cur_label[1]:
                        exact['pos_1_2'] += 1
                        if len(cur_pred) >= 3 and cur_pred[2] == cur_label[2]:
                            exact['pos_1_2_3'] += 1

    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    total = exact['total_samples'] or 1
    prob_stats = {
        'P_1': exact['pos_1'] / total,
        'P_1_2': exact['pos_1_2'] / total,
        'P_1_2_3': exact['pos_1_2_3'] / total,
    }
    return avg_recalls, avg_ndcgs, prob_stats


# ============================================================
# Loss terms
# ============================================================
def compute_l_sub_rec(h_uk, m_uk_bool, labels, shared_emb, tau, item_repr):
    """InfoNCE between the pooled sub-sequence representation and the target."""
    z_sub = masked_mean_pool(h_uk, m_uk_bool)
    return loss_rec_inbatch(z_u=z_sub, labels=labels, shared_emb=shared_emb,
                            tau=tau, item_repr=item_repr, item_mlp=None)


def compute_l_sub(struct_module, sub_out, lam_t, lam_c, history_ts_long=None):
    """Sub-sequence structure regularizer."""
    ts_long = None
    if history_ts_long is not None:
        ts_long = history_ts_long[:, :sub_out["x_long"].size(1)].contiguous()
    stats = struct_module(
        sub_out["x_long"], sub_out["m_long"], sub_out["c1_long"],
        sub_out["a_utk"], sub_out["sid1_size"], history_ts_long=ts_long)
    return loss_subseq_structure(stats, sub_out["m_uk"], lam_t=lam_t, lam_c=lam_c)


# ============================================================
# One training epoch
# ============================================================
def train_one_epoch(raw_model, huk_cross_modules, subseq, struct, loader,
                    optimizer, device, lambda_rec, lambda_subrec, lambda_sub,
                    rec_tau, rec_item_repr, sub_lam_t, sub_lam_c,
                    use_sidtier_prefix=True, recent_token_len=None,
                    desc="  train"):
    raw_model.train()
    subseq.train()
    struct.train()
    huk_cross_modules.train()
    agg = {"total": 0.0, "ce": 0.0, "rec": 0.0, "subrec": 0.0, "sub": 0.0, "n": 0}
    ce_fct = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        sidtier = batch.get('sidtier', None)
        if sidtier is not None:
            sidtier = sidtier.to(device)

        optimizer.zero_grad()

        # 1) Adaptive partition + sub-sequence encoding.
        ts = batch.get('timestamps', None)
        if ts is not None:
            ts = ts.to(device)
        sub_out = subseq(input_ids, history_ts_long=ts)
        h_uk = sub_out["h_uk"]
        m_uk_bool = (sub_out["m_uk"] >= 0.5)

        # 2) Encoder with in-block h_uk injection.
        enc_hidden, enc_mask_ext, enc_attn_mask = encode_with_huk_crossattn(
            raw_model, huk_cross_modules, input_ids, attention_mask, h_uk,
            sidtier=sidtier, use_sidtier_prefix=use_sidtier_prefix,
            recent_token_len=recent_token_len)

        # 3) Mask for the decoder's h_uk path.
        B, K_sub = h_uk.size(0), h_uk.size(1)
        huk_attn_mask = torch.ones(B, K_sub, device=device, dtype=enc_hidden.dtype)
        huk_mask_ext = raw_model.get_extended_attention_mask(huk_attn_mask,
                                                              enc_hidden.dtype)

        # 4) L_CE over the SID tokens.
        logits = decode_logits_with_huk(raw_model, labels, enc_hidden, enc_mask_ext,
                                        h_uk=h_uk, huk_mask=huk_mask_ext)
        loss_ce = ce_fct(logits.view(-1, raw_model.vocab_size), labels.view(-1))

        # 5) L_rec, pooled from H_enc* (already fused with h_uk).
        z_u = masked_mean_pool(enc_hidden, enc_attn_mask.bool())
        loss_rec = loss_rec_inbatch(z_u=z_u, labels=labels,
                                    shared_emb=raw_model.shared,
                                    tau=rec_tau, item_repr=rec_item_repr,
                                    item_mlp=None)

        # 6) L_subrec, pooled from h_uk.
        loss_subrec = compute_l_sub_rec(h_uk, m_uk_bool, labels,
                                        raw_model.shared, rec_tau, rec_item_repr)

        # 7) L_sub, the structure regularizer.
        loss_sub = compute_l_sub(struct, sub_out, sub_lam_t, sub_lam_c,
                                 history_ts_long=ts)

        total = (loss_ce
                 + lambda_rec * loss_rec
                 + lambda_subrec * loss_subrec
                 + lambda_sub * loss_sub)

        if not torch.isfinite(total):
            optimizer.zero_grad(set_to_none=True)
            logging.warning("[skip] non-finite total loss")
            continue

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(raw_model.parameters()) + list(subseq.parameters())
            + list(struct.parameters()) + list(huk_cross_modules.parameters()),
            max_norm=1.0)
        optimizer.step()

        agg["total"] += float(total.item())
        agg["ce"] += float(loss_ce.item())
        agg["rec"] += float(loss_rec.item())
        agg["subrec"] += float(loss_subrec.item())
        agg["sub"] += float(loss_sub.item())
        agg["n"] += 1

    n = max(1, agg["n"])
    return {k: (agg[k] / n if k != "n" else agg[k]) for k in agg}


# ============================================================
# Setup helpers
# ============================================================
def build_drift_cfg(args, codebook, device):
    """Build the ``DriftConfig`` used by the subsequence modules."""
    cfg = DriftConfig(
        codebook=tuple(codebook), target_depth=3, pad_token_id=args.pad_token_id,
        sid_layer1_size=codebook[0],
        max_seq_len=args.max_len, recent_window_r=args.recent_window_r,
        d_model=gl.D_MODEL, K=args.K,
        K_min=float(args.K_min),
        K_max=float(args.K_max) if args.K_max > 0 else float(args.K),
        seed=args.seed, device=str(device),
    )
    # Real timestamps are available for KuaiRec and ML-20M; for datasets without
    # them the temporal terms fall back to a constant (no gradient).
    cfg.use_real_timestamp = bool(args.use_timestamps)
    return cfg


def add_common_arguments(ap):
    """Register the CLI flags shared by both entry scripts."""
    # ---- data ----
    ap.add_argument('--data_root', default='kuairec_processed',
                    help='dataset directory, relative to the data root')
    ap.add_argument('--codebook', default='256,256,256',
                    help='per-level codebook sizes, e.g. "64,128,128" for TAOBAO-MM')
    ap.add_argument('--ts_suffix', action='store_true',
                    help='dataset directory carries the "_ts" suffix (timestamped)')
    ap.add_argument('--use_timestamps', type=int, default=1,
                    help='1: use real timestamps; 0: disable the temporal terms')
    ap.add_argument('--max_len', type=int, default=50,
                    help='history length in items (L)')
    # ---- optimization ----
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--infer_size', type=int, default=128)
    ap.add_argument('--num_epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--beam_size', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--early_stop', type=int, default=50)
    ap.add_argument('--pad_token_id', type=int, default=0)
    # ---- adaptive partition ----
    ap.add_argument('--K', type=int, default=8,
                    help='maximum number of historical sub-sequences')
    ap.add_argument('--K_min', type=float, default=1.0, help='lower bound of kappa_u')
    ap.add_argument('--K_max', type=float, default=-1.0,
                    help='upper bound of kappa_u; <=0 means use K')
    ap.add_argument('--recent_window_r', type=int, default=10,
                    help='recent-window size r')
    # ---- loss weights ----
    ap.add_argument('--lambda_rec', type=float, default=0.5)
    ap.add_argument('--lambda_subrec', type=float, default=0.5)
    ap.add_argument('--lambda_sub', type=float, default=0.05)
    ap.add_argument('--rec_tau', type=float, default=0.07,
                    help='temperature shared by L_rec and L_subrec')
    ap.add_argument('--rec_item_repr', default='first', choices=['first', 'mlp'],
                    help='target item representation in the in-batch losses')
    ap.add_argument('--sub_lam_t', type=float, default=0.5,
                    help='weight of the temporal term inside L_sub')
    ap.add_argument('--sub_lam_c', type=float, default=0.5,
                    help='weight of the SID-consistency term inside L_sub')
    return ap


def resolve_dataset_path(args, codebook):
    """Resolve the processed-dataset directory from relative parts."""
    name = f'glass_data_{codebook[0]}_{codebook[1]}_{codebook[2]}'
    if args.ts_suffix:
        name += '_ts'
    dataset_path = paths.data_path(args.data_root, name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"dataset directory not found: {dataset_path}\n"
            f"Set DRIFT_DATA_ROOT or pass --data_root accordingly.")
    return dataset_path


def resolve_split_file(dataset_path, default_name, max_len):
    """Pick the split parquet, honoring the ``seq_len<L>`` filename variant.

    Datasets prepared for a non-default history length store files as
    ``train_seq_len<L>_sim_augmented.parquet`` etc.; ``max_len == 50`` uses the
    plain names.
    """
    if int(max_len) == 50:
        return os.path.join(dataset_path, default_name)
    for prefix in ('train_', 'valid_', 'test_'):
        if default_name.startswith(prefix):
            seq_name = f'{prefix[:-1]}_seq_len{max_len}_' + default_name[len(prefix):]
            break
    else:
        seq_name = default_name
    seq_path = os.path.join(dataset_path, seq_name)
    if os.path.exists(seq_path):
        return seq_path
    logging.warning(f"[data] seq_len file missing, falling back to: {default_name}")
    return os.path.join(dataset_path, default_name)


def build_dataloaders(args, dataset_path, codebook):
    """Create the train / valid / test loaders with timestamp support."""
    TsDataset, TsLoader = make_timestamped_dataset_class(GenRecDataset,
                                                        GenRecDataLoader)
    files = {
        'train': resolve_split_file(dataset_path, gl.TRAIN_FILE_PATH, args.max_len),
        'valid': resolve_split_file(dataset_path, gl.VALID_FILE_PATH, args.max_len),
        'test': resolve_split_file(dataset_path, gl.TEST_FILE_PATH, args.max_len),
    }
    logging.info("[data] " + " ".join(
        f"{k}={os.path.basename(v)}" for k, v in files.items()))

    cb_max = max(codebook)
    # autoload_ts=False: we load timestamps from the exact split files below,
    # because filename globbing cannot distinguish the valid from the test split.
    train_ds = TsDataset(dataset_path=files['train'], mode='train',
                         max_len=args.max_len, codebook_size=cb_max,
                         autoload_ts=False)
    valid_ds = TsDataset(dataset_path=files['valid'], mode='evaluation',
                         max_len=args.max_len, PAD_TOKEN=0, codebook_size=cb_max,
                         autoload_ts=False)
    test_ds = TsDataset(dataset_path=files['test'], mode='evaluation',
                        max_len=args.max_len, PAD_TOKEN=0, codebook_size=cb_max,
                        autoload_ts=False)

    for ds, key in ((train_ds, 'train'), (valid_ds, 'valid'), (test_ds, 'test')):
        load_timestamps_into_dataset(ds, files[key], args.max_len)

    train_dl = TsLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    valid_dl = TsLoader(valid_ds, batch_size=args.infer_size, shuffle=False,
                        num_workers=0)
    test_dl = TsLoader(test_ds, batch_size=args.infer_size, shuffle=False,
                       num_workers=0)
    return train_dl, valid_dl, test_dl


def build_model_config(args, codebook, dataset_path, device):
    """Backbone hyper-parameters; SID-dependent sizes derive from the codebook."""
    return {
        'batch_size': args.batch_size, 'infer_size': args.infer_size,
        'num_epochs': args.num_epochs, 'lr': args.lr, 'device': str(device),
        'num_layers': 4, 'num_decoder_layers': 4,
        'd_model': gl.D_MODEL, 'd_ff': 1024, 'num_heads': 8, 'dropout_rate': 0.1,
        'vocab_size': 1000, 'pad_token_id': args.pad_token_id,
        # Derived from the codebook so a different SID configuration cannot cause
        # out-of-range BOS ids or SID-tier shape mismatches.
        'bos_token_id': sum(codebook) + 1,
        'sidtier_dim': codebook[0] * 4,
        'max_len': args.max_len, 'seed': args.seed, 'topk_list': [1, 3, 5, 10, 20],
        'beam_size': args.beam_size, 'd_kv': 32, 'feed_forward_proj': 'relu',
        'dataset_path': dataset_path,
        'use_gate': gl.USE_GATE, 'use_sidtier': gl.USE_SIDTIER,
    }


def run_training(args, variant, use_sidtier_prefix, log_tag):
    """Full training loop shared by both entry scripts.

    Args:
        args:                parsed CLI arguments.
        variant:             short identifier written into the summary JSON.
        use_sidtier_prefix:  True for prefix ``[sidtier ; recent]``,
                             False for prefix ``[recent]``.
        log_tag:             prefix used in log lines.
    """
    codebook = [int(x) for x in args.codebook.split(',')]
    assert len(codebook) == 3, "--codebook expects three comma-separated sizes"

    os.makedirs(args.out_dir, exist_ok=True)
    best_json = os.path.join(args.out_dir, f'{variant}_best.json')
    summary_json = os.path.join(args.out_dir, f'{variant}_summary.json')
    for f in (best_json, summary_json):
        if os.path.exists(f):
            raise FileExistsError(
                f"output file already exists, refusing to overwrite: {f}\n"
                f"Use a different --out_dir or remove it first.")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = resolve_dataset_path(args, codebook)
    config = build_model_config(args, codebook, dataset_path, device)
    logging.info(f"[{log_tag}] dataset={dataset_path} codebook={codebook} "
                 f"max_len={args.max_len} K={args.K} r={args.recent_window_r} "
                 f"lambda_rec={args.lambda_rec} lambda_subrec={args.lambda_subrec} "
                 f"lambda_sub={args.lambda_sub}")

    # ---- model + subsequence modules ----
    model = GLASS(config).to(device)
    raw_model = model
    drift_cfg = build_drift_cfg(args, codebook, device)
    subseq = SubseqRepresentation(drift_cfg, raw_model.shared, gl.D_MODEL).to(device)
    struct = StructureConstraints(drift_cfg).to(device)

    # One cross-attention module per encoder layer.
    huk_cross_modules = nn.ModuleList([
        HukCrossAttn(gl.D_MODEL, config['num_heads'], config['d_kv'],
                     config['dropout_rate'])
        for _ in range(config['num_layers'])
    ]).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        raw_model = model.module

    train_dl, valid_dl, test_dl = build_dataloaders(args, dataset_path, codebook)

    # NOTE: the shared item-embedding table is intentionally listed once via the
    # backbone and once via ``subseq`` (which reuses the same table). This matches
    # the configuration used for the reported results and is kept as-is for
    # reproducibility; PyTorch emits a duplicate-parameter warning for it.
    params = (list(raw_model.parameters()) + list(subseq.parameters())
              + list(struct.parameters()) + list(huk_cross_modules.parameters()))
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # The encoder consumes the recent window only; each item spans 3 SID tokens.
    recent_token_len = int(args.recent_window_r) * 3
    logging.info(f"[{log_tag}] recent_token_len={recent_token_len} "
                 f"(r={args.recent_window_r} x depth=3); branch max_len={args.max_len}")

    best_ndcg, early, best = 0.0, 0, {}
    t0 = time.time()
    ep = -1
    for ep in range(args.num_epochs):
        tr = train_one_epoch(
            raw_model, huk_cross_modules, subseq, struct, train_dl, optimizer,
            device,
            lambda_rec=args.lambda_rec, lambda_subrec=args.lambda_subrec,
            lambda_sub=args.lambda_sub, rec_tau=args.rec_tau,
            rec_item_repr=args.rec_item_repr,
            sub_lam_t=args.sub_lam_t, sub_lam_c=args.sub_lam_c,
            use_sidtier_prefix=use_sidtier_prefix,
            recent_token_len=recent_token_len, desc=f"  {log_tag}")
        logging.info(
            f"[Epoch {ep+1}] L_total={tr['total']:.4f} L_CE={tr['ce']:.4f} "
            f"L_rec={tr['rec']:.4f} L_subrec={tr['subrec']:.4f} "
            f"L_sub={tr['sub']:.4f}")

        recalls, ndcgs, prob = evaluate_huk(
            raw_model, huk_cross_modules, subseq, valid_dl, config['topk_list'],
            args.beam_size, device, use_sidtier_prefix=use_sidtier_prefix,
            recent_token_len=recent_token_len)
        n20 = ndcgs.get('NDCG@20', 0.0)
        logging.info(f"[Epoch {ep+1}] valid: " + " | ".join(
            f"{k}={v:.4f}" for k, v in list(recalls.items()) + list(ndcgs.items())))
        logging.info(f"[Epoch {ep+1}] valid props: P(1)={prob['P_1']:.4f} | "
                     f"P(1,2)={prob['P_1_2']:.4f} | P(1,2,3)={prob['P_1_2_3']:.4f}")

        # Model selection on validation NDCG@20; test is only evaluated on improvement.
        if n20 > best_ndcg:
            best_ndcg, early = n20, 0
            trc, tnd, tprob = evaluate_huk(
                raw_model, huk_cross_modules, subseq, test_dl, config['topk_list'],
                args.beam_size, device, use_sidtier_prefix=use_sidtier_prefix,
                recent_token_len=recent_token_len)
            logging.info(f"[Epoch {ep+1}] [BEST] test: " + " | ".join(
                f"{k}={v:.4f}" for k, v in list(trc.items()) + list(tnd.items())))
            logging.info(f"[Epoch {ep+1}] [BEST] test props: "
                         f"P(1)={tprob['P_1']:.4f} | P(1,2)={tprob['P_1_2']:.4f} | "
                         f"P(1,2,3)={tprob['P_1_2_3']:.4f}")
            best = {'best_epoch': ep + 1, 'best_valid_ndcg@20': best_ndcg,
                    'test_recall': trc, 'test_ndcg': tnd, 'test_prob': tprob}
            with open(best_json, 'w') as f:
                json.dump(best, f, ensure_ascii=False, indent=1)
        else:
            early += 1
            if early >= args.early_stop:
                logging.info(f"[early stop] at epoch {ep+1}")
                break

    with open(summary_json, 'w') as f:
        json.dump({'best': best, 'epochs_run': ep + 1,
                   'variant': variant,
                   'codebook': codebook,
                   'lambda_rec': args.lambda_rec,
                   'lambda_subrec': args.lambda_subrec,
                   'lambda_sub': args.lambda_sub,
                   'K': args.K, 'recent_window_r': args.recent_window_r,
                   'max_len': args.max_len,
                   'elapsed_sec': round(time.time() - t0, 1)},
                  f, ensure_ascii=False, indent=1)
    logging.info(f"[DONE] best={best}")
    return best
