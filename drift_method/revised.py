# -*- coding: utf-8 -*-
"""Revised DRIFT-TIGER with padding-safe partition and all-position fusion.

The backbone's T5 math is reused without changing existing entry points. The
recent encoder attends to every subsequence slot; the decoder uses the paper's
global memory at every SID step, including the first prediction from BOS.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import paths

paths.ensure_import_paths()
from model import glass_backbone as gl
from drift_method.config import DriftConfig
from drift_method.embedding.item_embedding import per_item_embed, split_recent_long
from drift_method.losses import loss_subseq_structure
from drift_method.rec_losses import loss_rec_inbatch
from drift_method.subsequence.intra_encoder import IntraSubsequenceEncoder
from drift_method.subsequence.partition import SoftPartition
from drift_method.subsequence.structure_constraints import StructureConstraints


class PaperDriftT5Block(gl.DualAwareT5Block):
    """Original T5 block plus the paper's local/global gated cross-attention.

    ``enable_global_ca=False`` selects the history-only decoder ablation.
    Neither the module-level ``gl.USE_GATE`` nor ``config['use_gate']`` controls
    this block. The caller passes subsequence memory and its chosen slot mask.
    """

    def __init__(self, config, is_decoder=False, has_relative_bias=False,
                 enable_global_ca=True):
        super().__init__(config, is_decoder=is_decoder,
                         has_relative_bias=has_relative_bias)
        self.enable_global_ca = bool(enable_global_ca)

    def forward(self, hidden_states, mask=None,
                encoder_hidden_states=None, encoder_mask=None,
                encoder_hidden_states_sim=None, encoder_mask_sim=None,
                position_bias=None):
        layer_gate_value = None

        normed_hidden_states = self.layer_norm1(hidden_states)
        attn_out, position_bias = self.self_attn(
            normed_hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = hidden_states + self.dropout1(attn_out)

        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer_norm2(hidden_states)
            out_hist, _ = self.cross_attn_hist(
                normed_hidden_states, kv_states=encoder_hidden_states,
                mask=encoder_mask)
            if self.enable_global_ca and encoder_hidden_states_sim is not None:
                out_subseq, _ = self.cross_attn_sim(
                    normed_hidden_states, kv_states=encoder_hidden_states_sim,
                    mask=encoder_mask_sim)
                gate = self.sigmoid(self.gate_layer(torch.cat(
                    [normed_hidden_states, out_subseq], dim=-1)))
                if encoder_mask_sim is not None:
                    # An all-masked row marks an empty long-history sample.
                    # Keep its history context at full strength instead of
                    # multiplying it by (1 - gate) with an empty global branch.
                    mask_rows = encoder_mask_sim.reshape(
                        encoder_mask_sim.shape[0], -1)
                    visible = mask_rows > -1e8
                    if mask_rows.is_floating_point():
                        # FP16 callers may use finite dtype.min instead of -1e9.
                        visible = visible & (mask_rows > torch.finfo(
                            mask_rows.dtype).min)
                    has_context = visible.any(dim=1)
                    gate = gate * has_context[:, None, None].to(
                        device=gate.device, dtype=gate.dtype)
                # The paper's gate applies to every position, including BOS -> SID1.
                fused_out = (1.0 - gate) * out_hist + gate * out_subseq
                layer_gate_value = gate.mean()
            else:
                fused_out = out_hist
                layer_gate_value = hidden_states.new_zeros(())
            hidden_states = hidden_states + self.dropout2(fused_out)

        normed_hidden_states = self.layer_norm3(hidden_states)
        ff_out = self.ff_out(self.ff_dropout(self.act(
            self.ff_in(normed_hidden_states))))
        hidden_states = hidden_states + self.dropout3(ff_out)
        return hidden_states, position_bias, layer_gate_value


def configure_paper_decoder(raw_model, enable_global_ca=True):
    """Replace decoder blocks before constructing the optimizer.

    Strict state loading checks that this adapter has the original parameter
    schema. Copying tensors afterwards preserves each parameter/buffer's exact
    dtype, device, values, and freezing policy, including mixed-dtype modules.
    Creation inside ``fork_rng`` preserves the CPU initialization stream used by
    the subsequence modules constructed next. No new parameter is introduced.
    """
    replacements = []
    for old_block in raw_model.decoder_blocks:
        with torch.random.fork_rng(devices=[]):
            new_block = PaperDriftT5Block(
                raw_model.config, is_decoder=True,
                has_relative_bias=old_block.self_attn.has_relative_bias,
                enable_global_ca=enable_global_ca)
        new_block.load_state_dict(old_block.state_dict(), strict=True)

        reference = next(old_block.parameters())
        new_block.to(device=reference.device, dtype=reference.dtype)
        old_parameters = dict(old_block.named_parameters())
        with torch.no_grad():
            for name, parameter in new_block.named_parameters():
                old_parameter = old_parameters[name]
                parameter.data = old_parameter.detach().clone()
                parameter.requires_grad_(old_parameter.requires_grad)

        new_submodules = dict(new_block.named_modules())
        for name, old_buffer in old_block.named_buffers():
            parent_name, _, buffer_name = name.rpartition('.')
            new_submodules[parent_name]._buffers[buffer_name] = (
                old_buffer.detach().clone())
        for name, old_submodule in old_block.named_modules():
            new_submodules[name].training = old_submodule.training
        replacements.append(new_block)

    raw_model.decoder_blocks = nn.ModuleList(replacements)
    raw_model.decoder_blocks.training = raw_model.training
    return raw_model


class PaperSoftPartition(SoftPartition):
    """Padding-safe partition with the base module's parameters and interface.

    The inherited MLP, kappa function, local JSD, and fixed stage centers are
    reused. This forward follows the reviewed base forward explicitly because
    masking its returned boundary scores would be too late: kappa, coordinates,
    memberships, and Gumbel assignments have already been computed by then.
    ``d0`` retains the caller's original semantics; this adapter does not add a
    learned user drift score. Future base-formula changes need the same review.
    """

    def forward(self, x_long, m_long, c1_long, d0, sid1_size,
                history_ts_long=None):
        cfg = self.cfg
        B, L_h, D = x_long.shape
        device = x_long.device
        valid = m_long.bool()
        # A boundary exists only when both adjacent interactions are real.
        # This also makes the first real interaction zero after left padding.
        boundary_valid = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            valid[:, 1:] & valid[:, :-1],
        ], dim=1)

        cos_pair = F.cosine_similarity(
            x_long[:, 1:, :], x_long[:, :-1, :], dim=-1, eps=1e-8)
        cos_pair = torch.cat([
            torch.zeros(B, 1, device=device), 1.0 - cos_pair,
        ], dim=1)

        if getattr(cfg, 'use_real_timestamp', False) and history_ts_long is not None:
            dt = torch.zeros(B, L_h, device=device)
            dt[:, 1:] = (history_ts_long[:, 1:] - history_ts_long[:, :-1]).clamp(
                min=getattr(cfg, 'ts_min_dt_floor', 1e-5))
            dt = dt * boundary_valid.float()
            log_dt = torch.log1p(dt)
        else:
            log_dt = torch.full((B, L_h), math.log(2.0), device=device)

        jsd_local = self._local_jsd(c1_long, valid, sid1_size)
        diff = torch.cat([
            torch.zeros(B, 1, D, device=device),
            x_long[:, 1:, :] - x_long[:, :-1, :],
        ], dim=1)
        d0_exp = d0.view(B, 1).expand(B, L_h)
        phi = torch.cat([
            cos_pair.unsqueeze(-1), log_dt.unsqueeze(-1),
            jsd_local.unsqueeze(-1), diff, d0_exp.unsqueeze(-1),
        ], dim=-1)
        b = torch.sigmoid(self.mlp_b(phi).squeeze(-1)) * boundary_valid.float()

        L_eff = valid.float().sum(dim=1)
        boundary_count = boundary_valid.float().sum(dim=1).clamp_min(1.0)
        b_bar = b.sum(dim=1) / boundary_count
        # Population variance over real positions, including the first b=0.
        # This preserves the original unpadded variance definition and removes
        # only PAD positions; boundary mean still averages valid adjacent pairs.
        valid_count = L_eff.clamp_min(1.0)
        b_position_mean = b.sum(dim=1) / valid_count
        v_u = 4.0 * (((b - b_position_mean.unsqueeze(1)) ** 2)
                     * valid.float()).sum(dim=1) / valid_count
        kappa = self.kappa_u(d0, L_eff, b_bar, v_u)

        S_ut = b.cumsum(dim=1)
        S_uL = S_ut[:, -1:].clamp_min(cfg.eps)
        if cfg.coord_normalize == 'raw':
            r_ut = 1.0 + S_ut
        elif cfg.coord_normalize == 'K':
            r_ut = 1.0 + (cfg.K - 1) * S_ut / S_uL
        else:
            r_ut = 1.0 + (kappa.unsqueeze(1) - 1.0) * S_ut / S_uL

        k_idx = torch.arange(1, self.K + 1, device=device, dtype=torch.float32)
        m_uk = torch.sigmoid(
            (kappa.unsqueeze(1) - k_idx.unsqueeze(0) + 0.5) / cfg.tau_m)
        diff2 = (r_ut.unsqueeze(-1) - self.mu_k.view(1, 1, -1)) ** 2
        ex = torch.exp(-diff2 / cfg.tau_a) * m_uk.unsqueeze(1)
        a_utk = ex / (ex.sum(dim=-1, keepdim=True) + cfg.eps)
        a_utk = a_utk * valid.float().unsqueeze(-1)

        out = {
            'b': b, 'S_ut': S_ut, 'r_ut': r_ut, 'kappa': kappa,
            'm_uk': m_uk, 'a_utk': a_utk,
        }
        if cfg.use_gumbel:
            logits_a = torch.log(a_utk + cfg.eps)
            if self.training:
                noise = -torch.log(-torch.log(
                    torch.rand_like(logits_a) + cfg.eps) + cfg.eps)
                logits_a = logits_a + noise
            # Evaluation retains temperature sharpening and omits random noise.
            y = F.softmax(logits_a / cfg.gumbel_tau, dim=-1)
            out['y_utk'] = y * valid.float().unsqueeze(-1)
        else:
            out['y_utk'] = a_utk
        return out


def build_subsequence_attention_bias(m_uk, dtype):
    """Attend all K slots; an empty long history masks every slot."""
    if m_uk.ndim != 2 or m_uk.size(1) == 0:
        raise ValueError("m_uk must have shape (batch, positive K)")
    has_context = (m_uk > 0).any(1, keepdim=True)
    masked_value = max(-1e9, torch.finfo(dtype).min)
    bias = torch.zeros_like(m_uk, dtype=dtype)
    bias = bias.masked_fill(~has_context, masked_value)
    return bias[:, None, None, :]


def _pool(hidden, valid):
    weight = valid.float().unsqueeze(-1)
    return (hidden * weight).sum(1) / (weight.sum(1) + 1e-8)


class HukCrossAttn(nn.Module):
    """Residual cross-attention inserted between encoder self-attention and FFN."""

    def __init__(self, config):
        super().__init__()
        self.layer_norm = gl.T5LayerNorm(config["d_model"])
        self.cross_attn = gl.T5Attention(
            config["d_model"], config["num_heads"], config["d_kv"],
            config["dropout_rate"], has_relative_bias=False)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, hidden_states, memory, memory_mask):
        output, _ = self.cross_attn(
            self.layer_norm(hidden_states), kv_states=memory, mask=memory_mask)
        return hidden_states + self.dropout(output)


class SubseqRepresentation(nn.Module):
    """Item-sum embeddings, adaptive partition, and linear subsequence encoding."""

    def __init__(self, cfg: DriftConfig, shared: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.shared = shared
        self.partition = PaperSoftPartition(cfg, cfg.d_model)
        self.intra_enc = IntraSubsequenceEncoder(cfg, cfg.d_model)
        self.struct = StructureConstraints(cfg)

    def forward(self, history_token_ids, history_ts_long=None):
        cfg = self.cfg
        x_full, m_full = per_item_embed(
            self.shared, history_token_ids, target_depth=cfg.target_depth,
            pad_token_id=cfg.pad_token_id, aggregate="sum")
        tokens = history_token_ids.reshape(
            history_token_ids.size(0), -1, cfg.target_depth)
        first = tokens[:, :, 0]
        c1_full = (first - 1).clamp(min=0, max=cfg.codebook[0] - 1)
        c1_full = torch.where(first == cfg.pad_token_id,
                              torch.full_like(c1_full, -1), c1_full)
        x_long, m_long, _, _ = split_recent_long(
            x_full, m_full, cfg.recent_window_r)
        long_length = x_long.size(1)
        if x_full.size(1) <= cfg.recent_window_r:
            # The split helper emits one masked dummy, never a recent item.
            c1_long = torch.full_like(c1_full[:, :1], -1)
            ts_long = (None if history_ts_long is None
                       else history_ts_long.new_zeros(history_token_ids.size(0), 1))
        else:
            c1_long = c1_full[:, :long_length].contiguous()
            ts_long = (None if history_ts_long is None
                       else history_ts_long[:, :long_length].contiguous())
        d0 = x_full.new_zeros(x_full.size(0))
        part = self.partition(
            x_long, m_long, c1_long, d0, cfg.codebook[0],
            history_ts_long=ts_long)
        h_uk, _ = self.intra_enc(
            x_long, m_long, part["y_utk"], d0, history_ts_long=ts_long)
        # Empty long histories must not supply learned evidence or attenuate
        # the decoder's history path through an otherwise nonzero gate.
        has_long = m_long.bool().any(1)
        h_uk = h_uk.masked_fill(~has_long[:, None, None], 0.0)
        m_uk = part["m_uk"].masked_fill(~has_long[:, None], 0.0)
        return {
            "h_uk": h_uk, "m_uk": m_uk, "a_utk": part["a_utk"],
            "kappa": part["kappa"],
            "x_long": x_long, "m_long": m_long, "c1_long": c1_long,
            "sid1_size": cfg.codebook[0], "timestamps_long": ts_long,
        }


class RevisedDrift(nn.Module):
    """One module owns the backbone and every branch sharing its SID table.

    Iterating ``parameters()`` deduplicates the shared embedding. The instance
    disables the GLASS retrieval/projector and SIDTier prefix; only the revised
    DRIFT decoder supplies the global cross-attention path.
    """

    def __init__(self, model_config, drift_cfg: DriftConfig,
                 lambda_rec=0.5, lambda_subrec=0.5, lambda_sub=0.05,
                 sub_lam_t=0.5, sub_lam_c=0.5, rec_tau=0.07):
        super().__init__()
        model_cfg = dict(model_config)
        model_cfg.update(use_gate=False, use_sidtier=False)
        if drift_cfg.d_model != model_cfg["d_model"]:
            raise ValueError("DRIFT and backbone d_model must agree")
        if drift_cfg.target_depth != 3:
            raise ValueError("Revised DRIFT-TIGER requires three SID levels")
        if rec_tau <= 0:
            raise ValueError("rec_tau must be positive")
        # Match the reviewed baseline initialization without consuming the RNG
        # stream used to initialize the additional DRIFT modules next.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(drift_cfg.seed)
            self.raw_model = gl.GLASS(model_cfg)
        configure_paper_decoder(self.raw_model, enable_global_ca=True)
        self.cfg = drift_cfg
        self.subseq = SubseqRepresentation(self.cfg, self.raw_model.shared)
        self.struct = StructureConstraints(self.cfg)
        self.huk_cross_modules = nn.ModuleList([
            HukCrossAttn(model_cfg) for _ in range(model_cfg["num_layers"])])
        self.loss_rec_inbatch = loss_rec_inbatch
        self.loss_subseq_structure = loss_subseq_structure
        self.lambda_rec = float(lambda_rec)
        self.lambda_subrec = float(lambda_subrec)
        self.lambda_sub = float(lambda_sub)
        self.sub_lam_t = float(sub_lam_t)
        self.sub_lam_c = float(sub_lam_c)
        self.rec_tau = float(rec_tau)
        self.enable_encoder_subsequence_ca = True

    def encode(self, batch):
        raw = self.raw_model
        sub = self.subseq(batch["history"], batch.get("timestamps"))
        recent_token_len = self.cfg.recent_window_r * self.cfg.target_depth
        recent = batch["history"][:, -recent_token_len:]
        valid = batch["attention_mask"][:, -recent_token_len:]
        hidden = raw.encoder_dropout(raw.shared(recent))
        mask = raw.get_extended_attention_mask(valid, hidden.dtype)
        huk_mask = build_subsequence_attention_bias(sub["m_uk"], hidden.dtype)
        position_bias = None
        for block, cross in zip(raw.encoder_blocks, self.huk_cross_modules):
            output, position_bias = block.self_attn(
                block.layer_norm1(hidden), mask=mask, position_bias=position_bias)
            hidden = hidden + block.dropout1(output)
            if self.enable_encoder_subsequence_ca:
                hidden = cross(hidden, sub["h_uk"], huk_mask)
            output = block.ff_out(block.ff_dropout(block.act(
                block.ff_in(block.layer_norm3(hidden)))))
            hidden = hidden + block.dropout3(output)
        hidden = raw.encoder_norm(hidden)
        return hidden, mask, valid, sub, huk_mask

    def loss_components(self, batch):
        hidden, mask, valid, sub, huk_mask = self.encode(batch)
        raw, labels = self.raw_model, batch["target"]
        logits, _ = raw.forward_decoder(
            raw._shift_right(labels), hidden, mask, sub["h_uk"], huk_mask)
        ce = F.cross_entropy(logits.reshape(-1, raw.vocab_size),
                             labels.reshape(-1), ignore_index=-100)
        rec = self.loss_rec_inbatch(
            _pool(hidden, valid.bool()), labels, raw.shared,
            tau=self.rec_tau, item_repr="first", item_mlp=None)
        subrec = self.loss_rec_inbatch(
            _pool(sub["h_uk"], sub["m_uk"] >= 0.5), labels, raw.shared,
            tau=self.rec_tau, item_repr="first", item_mlp=None)
        stats = self.struct(
            sub["x_long"], sub["m_long"], sub["c1_long"], sub["a_utk"],
            sub["sid1_size"], history_ts_long=sub["timestamps_long"])
        structure = self.loss_subseq_structure(
            stats, sub["m_uk"], lam_t=self.sub_lam_t, lam_c=self.sub_lam_c)
        total = (ce + self.lambda_rec * rec + self.lambda_subrec * subrec
                 + self.lambda_sub * structure)
        return {"total": total, "ce": ce, "rec": rec,
                "subrec": subrec, "structure": structure}

    def loss(self, batch):
        return self.loss_components(batch)["total"]

    def forward(self, batch):
        return self.loss(batch)

    @torch.no_grad()
    def generate(self, batch, beam_size=20, max_length=4):
        """Native full-prefix beam search; four calls return three SID levels."""
        if not 1 <= int(beam_size) <= self.raw_model.vocab_size:
            raise ValueError("beam_size must be within [1, vocab_size]")
        if int(max_length) < self.cfg.target_depth:
            raise ValueError("max_length must generate all three SID levels")
        hidden, mask, _, sub, huk_mask = self.encode(batch)
        raw = self.raw_model
        beam, length = int(beam_size), int(max_length)
        b, device = hidden.size(0), hidden.device
        ids = torch.full((b, 1), raw.bos_token_id, dtype=torch.long, device=device)
        logits, _ = raw.forward_decoder(ids, hidden, mask, sub["h_uk"], huk_mask)
        top_scores, top_ids = torch.topk(
            torch.softmax(logits[:, -1, :], dim=-1), beam, dim=-1)
        hidden = hidden.repeat_interleave(beam, 0)
        mask = mask.repeat_interleave(beam, 0)
        memory = sub["h_uk"].repeat_interleave(beam, 0)
        huk_mask = huk_mask.repeat_interleave(beam, 0)
        ids = torch.cat([ids.repeat_interleave(beam, 0), top_ids.reshape(-1, 1)], 1)
        scores = torch.log(top_scores + 1e-9).reshape(-1)
        # Preserve unrestricted search and the fourth decoder call, whose score
        # can affect the ranking of the three SID levels returned to evaluation.
        for _step in range(length - 1):
            logits, _ = raw.forward_decoder(ids, hidden, mask, memory, huk_mask)
            next_scores = scores[:, None] + F.log_softmax(logits[:, -1, :], -1)
            best_scores, best_indices = next_scores.reshape(b, -1).topk(beam, 1)
            prior = best_indices // raw.vocab_size
            token = best_indices % raw.vocab_size
            base = (torch.arange(b, device=device) * beam)[:, None]
            take = (base + prior).reshape(-1)
            scores = best_scores.reshape(-1)
            ids = torch.cat([ids[take], token.reshape(-1, 1)], 1)
            hidden, mask = hidden[take], mask[take]
            memory, huk_mask = memory[take], huk_mask[take]
        return ids.reshape(b, beam, -1)[:, :, 1:4]
