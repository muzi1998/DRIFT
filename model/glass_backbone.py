# -*- coding: utf-8 -*-
"""SID-based generative recommendation backbone (T5-style encoder-decoder).

This module provides the generative backbone on top of which our preference-shift
subsequence learning method is built, together with shared evaluation utilities.
It is imported (never executed directly) by the two main entry scripts
``train_drift_sidtier.py`` and ``train_drift_recent.py``.

Contents
--------
* ``T5LayerNorm`` / ``T5RelativePositionBias`` / ``T5Attention`` — transformer parts.
* ``DualAwareT5Block`` — an encoder block, or a decoder block with two
  cross-attention paths (encoder memory + auxiliary memory) fused by a learned gate.
* ``GLASS`` — the encoder-decoder model with an optional SID-tier prefix and beam
  search over the three SID levels.
* metric helpers (``calculate_pos_index`` / ``recall_at_k`` / ``ndcg_at_k``),
  ``evaluate`` and ``set_seed``.

The module-level constants below are defaults for the base configuration (KuaiRec
and ML-20M, codebook [256,256,256]). Entry scripts override them per dataset; e.g.
TAOBAO-MM uses codebook [64,128,128], which changes ``sidtier_dim`` and
``bos_token_id`` accordingly.
"""

import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import GenRecDataset          # noqa: F401  (re-exported to entries)
from dataloader import GenRecDataLoader    # noqa: F401  (re-exported to entries)

# ================= Configuration defaults =================
BEAM_SIZE = 20
MAX_LENGTH = 50            # model context length, in items
D_MODEL = 96               # hidden size
INFER_BATCH_SIZE = 128
TRAIN_FILE_PATH = 'train_sim_augmented.parquet'
VALID_FILE_PATH = 'valid_with_history.parquet'
TEST_FILE_PATH = 'test_with_history.parquet'
CODEBOOK = [256, 256, 256]  # per-level codebook sizes; TAOBAO-MM uses [64,128,128]

USE_GATE = True             # gated dual cross-attention in the decoder
USE_SIDTIER = True          # prepend the SID-tier feature to the encoder input
BATCH_SIZE = 512 * 4

# Derived from CODEBOOK so that changing the codebook cannot cause shape mismatch.
SIDTIER_DIM_DEFAULT = CODEBOOK[0] * 4          # e.g. 64 -> 256, 256 -> 1024
BOS_TOKEN_ID_DEFAULT = sum(CODEBOOK) + 1       # e.g. [64,128,128] -> 321

def print_log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ================= Model Layers =================

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.type_as(self.weight)


class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=128, num_buckets=32):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large) + ret

    def forward(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0) 
        return values


class T5Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, dropout_rate=0.1, has_relative_bias=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.inner_dim = num_heads * d_kv
        
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_bias = T5RelativePositionBias(num_heads)

    def forward(self, hidden_states, kv_states=None, mask=None, position_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = kv_states is not None
        kv_input = kv_states if is_cross else hidden_states
        
        q = self.q(hidden_states).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k(kv_input).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v(kv_input).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if self.has_relative_bias:
            position_bias = self.relative_bias(q.shape[2], k.shape[2])
        
        if position_bias is not None:
            scores += position_bias
            
        if mask is not None:
            scores += mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        output = self.o(context)
        
        return output, position_bias


class DualAwareT5Block(nn.Module):
    def __init__(self, config, is_decoder=False, has_relative_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        
        self.layer_norm1 = T5LayerNorm(config['d_model'])
        self.self_attn = T5Attention(config['d_model'], config['num_heads'], config['d_kv'], config['dropout_rate'], has_relative_bias)
        self.dropout1 = nn.Dropout(config['dropout_rate'])
        
        if is_decoder:
            self.layer_norm2 = T5LayerNorm(config['d_model'])
            self.cross_attn_hist = T5Attention(config['d_model'], config['num_heads'], config['d_kv'], config['dropout_rate'], has_relative_bias=False)
            
            self.cross_attn_sim = T5Attention(config['d_model'], config['num_heads'], config['d_kv'], config['dropout_rate'], has_relative_bias=False)
            
            self.gate_layer = nn.Linear(config['d_model'] * 2, config['d_model'])
            self.sigmoid = nn.Sigmoid()
            self.dropout2 = nn.Dropout(config['dropout_rate'])

        self.layer_norm3 = T5LayerNorm(config['d_model'])
        self.ff_in = nn.Linear(config['d_model'], config['d_ff'], bias=False)
        self.act = nn.ReLU() if config['feed_forward_proj'] == 'relu' else nn.GELU()
        self.ff_dropout = nn.Dropout(config['dropout_rate'])
        self.ff_out = nn.Linear(config['d_ff'], config['d_model'], bias=False)
        self.dropout3 = nn.Dropout(config['dropout_rate'])

    def forward(self, hidden_states, mask=None, 
                encoder_hidden_states=None, encoder_mask=None,       
                encoder_hidden_states_sim=None, encoder_mask_sim=None,
                position_bias=None):
        layer_gate_value = None
        
        normed_hidden_states = self.layer_norm1(hidden_states)
        attn_out, position_bias = self.self_attn(normed_hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = hidden_states + self.dropout1(attn_out)

        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer_norm2(hidden_states)
            
            out_hist, _ = self.cross_attn_hist(normed_hidden_states, kv_states=encoder_hidden_states, mask=encoder_mask)
            
            if encoder_hidden_states_sim is not None and USE_GATE:
                out_sim, _ = self.cross_attn_sim(
                    normed_hidden_states, 
                    kv_states=encoder_hidden_states_sim, 
                    mask=encoder_mask_sim
                )
                
                out_sim[:, 0, :] = 0.0
                gate_input = torch.cat([normed_hidden_states, out_sim], dim=-1)
                gate = self.sigmoid(self.gate_layer(gate_input))
                step_mask = torch.ones_like(gate)
                step_mask[:, 0, :] = 0.0 
            
                final_gate = gate * step_mask
                fused_out = out_hist * (1.0 - final_gate) + out_sim * final_gate
                
                if gate.size(1) > 1:
                    layer_gate_value = final_gate[:, 1:, :].mean()
                else:
                    layer_gate_value = torch.tensor(0.0, device=hidden_states.device)
            else:
                fused_out = out_hist
                layer_gate_value = torch.tensor(0.0, device=hidden_states.device)
            hidden_states = hidden_states + self.dropout2(fused_out)

        normed_hidden_states = self.layer_norm3(hidden_states)
        ff_out = self.ff_out(self.ff_dropout(self.act(self.ff_in(normed_hidden_states))))
        hidden_states = hidden_states + self.dropout3(ff_out)
        
        return hidden_states, position_bias, layer_gate_value



class GLASS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.bos_token_id = config['bos_token_id']
        
        self.use_gate = config.get('use_gate', False) 
        self.use_sidtier = config.get('use_sidtier', False)

        self.shared = nn.Embedding(config['vocab_size'], config['d_model'])

        if self.use_gate:
            print("✅ USE_GATE is True: Using Shared Embeddings (No external file loaded).")
            self.sim_embedding = self.shared  
            self.sim_projector = nn.Linear(self.d_model, self.d_model)
        else:
            self.sim_embedding = None
            self.sim_projector = None

        if self.use_sidtier:
            self.sidtier_dim = config.get('sidtier_dim', SIDTIER_DIM_DEFAULT)
            self.sidtier_mlp = nn.Sequential(
                nn.Linear(self.sidtier_dim, self.d_model),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(self.d_model, self.d_model)
            )
            print(f"✅ USE_SIDTIER is True: SIDTier MLP initialized (dim: {self.sidtier_dim} -> {self.d_model})")
            print(f"✅ Strategy Update: SIDTier will be CONCATENATED with Encoder History Sequence.")

        self.encoder_blocks = nn.ModuleList()
        for i in range(config['num_layers']):
            self.encoder_blocks.append(DualAwareT5Block(config, is_decoder=False, has_relative_bias=(i==0)))
        self.encoder_norm = T5LayerNorm(config['d_model'])
        self.encoder_dropout = nn.Dropout(config['dropout_rate'])

        self.decoder_blocks = nn.ModuleList()
        for i in range(config['num_decoder_layers']):
            self.decoder_blocks.append(DualAwareT5Block(config, is_decoder=True, has_relative_bias=(i==0)))
        self.decoder_norm = T5LayerNorm(config['d_model'])
        self.decoder_dropout = nn.Dropout(config['dropout_rate'])

        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.lm_head.weight = self.shared.weight

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Total parameters: {total_params}\n"
            f"Trainable parameters: {trainable_params}\n"
            f"Frozen parameters: {total_params - trainable_params}"
        )

    def get_extended_attention_mask(self, mask, dtype):
        if mask is None: return None
        # mask: (Batch, SeqLen) -> (Batch, 1, 1, SeqLen)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -1e9
        return extended_mask.to(dtype=dtype)

    def get_causal_mask(self, seq_len, dtype, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -1e9, diagonal=1)
        return mask[None, None, :, :]
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config['bos_token_id'] 
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, decoder_start_token_id)
        return shifted_input_ids

    def forward_encoder(self, input_ids, attention_mask, sidtier=None):
        # 1) Embed the interaction history tokens.
        hidden_states = self.shared(input_ids) # (Batch, SeqLen, D)

        # 2) Optionally prepend the projected SID-tier feature as one extra token.
        if self.use_sidtier and sidtier is not None:
            sidtier_emb = self.sidtier_mlp(sidtier) # (Batch, D)
            sidtier_emb = sidtier_emb.unsqueeze(1)  # (Batch, 1, D)
            
            # Concatenate: [SIDTier, History]
            hidden_states = torch.cat([sidtier_emb, hidden_states], dim=1) # (Batch, 1 + SeqLen, D)
            
            # Extend the mask with a leading 1 for the SID-tier token.
            batch_size = attention_mask.size(0)
            sid_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([sid_mask, attention_mask], dim=1) # (Batch, 1 + SeqLen)

        # 3) Standard encoder stack.
        hidden_states = self.encoder_dropout(hidden_states)
        extended_mask = self.get_extended_attention_mask(attention_mask, hidden_states.dtype)
        
        position_bias = None
        for block in self.encoder_blocks:
            hidden_states, position_bias, _ = block(hidden_states, mask=extended_mask, position_bias=position_bias)
            
        hidden_states = self.encoder_norm(hidden_states)
        return hidden_states, extended_mask

    def forward_decoder(self, decoder_input_ids, 
                        encoder_hidden_states, encoder_mask, 
                        sim_hidden_states=None, sim_mask=None,
                        step_idx=None):
        
        hidden_states = self.shared(decoder_input_ids)
        hidden_states = self.decoder_dropout(hidden_states)
        
        seq_len = decoder_input_ids.shape[1]
        causal_mask = self.get_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        
        position_bias = None
        all_gates = []
        for block in self.decoder_blocks:
            hidden_states, position_bias, gate_val = block(
                hidden_states, 
                mask=causal_mask, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_mask=encoder_mask,
                encoder_hidden_states_sim=sim_hidden_states, 
                encoder_mask_sim=sim_mask, 
                position_bias=position_bias,
            )
            if gate_val is not None: all_gates.append(gate_val)
        
        hidden_states = self.decoder_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        avg_gate = torch.stack(all_gates).mean() if all_gates else torch.tensor(0.0)
        return logits, avg_gate

    def forward(self, input_ids, attention_mask, sim_input_ids, sim_attention_mask, labels=None, sidtier=None):
        # 1) Encoder forward; the SID-tier token is concatenated inside.
        encoder_hidden_states, encoder_mask = self.forward_encoder(input_ids, attention_mask, sidtier=sidtier)
        
        # 2) Build the auxiliary memory from retrieved similar items (gate only).
        sim_hidden_states = None
        sim_mask = None

        if self.use_gate and self.sim_embedding is not None:
             # Retrieved-item embeddings are used as a frozen lookup.
             with torch.no_grad():
                sim_emb = self.sim_embedding(sim_input_ids)
             sim_hidden_states = self.sim_projector(sim_emb)
             sim_mask = self.get_extended_attention_mask(sim_attention_mask, encoder_hidden_states.dtype)
        
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)
            logits, avg_gate = self.forward_decoder(
                decoder_input_ids, 
                encoder_hidden_states, encoder_mask,
                sim_hidden_states, sim_mask
            )
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            return loss, logits, avg_gate
        else:
            return None, None, None

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, all_sim_input_ids, all_sim_attention_mask, beam_size=20, max_length=4, sidtier=None):
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # 1) Encoder forward (SID-tier token concatenated here).
        encoder_hidden_states, encoder_mask = self.forward_encoder(input_ids, attention_mask, sidtier=sidtier)
        
        curr_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        # First decoding step; auxiliary memory is not available yet.
        logits, _ = self.forward_decoder(curr_ids, encoder_hidden_states, encoder_mask, None, None)
        next_token_logits = logits[:, -1, :] 

        topk_scores, topk_sids = torch.topk(torch.softmax(next_token_logits, dim=-1), beam_size, dim=-1) 

        real_sim_hidden = None
        real_sim_mask_extended = None

        # 2) Build the auxiliary memory (only when the gate is enabled).
        # The SID-tier signal is already inside encoder_hidden_states.
        if self.use_gate:
            B, N, FlatLen = all_sim_input_ids.shape
            
            OFFSET = 1
            group_indices = topk_sids - OFFSET 
            group_indices = torch.clamp(group_indices, 0, N - 1)

            K = beam_size
            
            gather_idx = group_indices.view(B, K, 1).expand(-1, -1, FlatLen)
            selected_sim_codes = torch.gather(all_sim_input_ids, 1, gather_idx)
            selected_sim_mask = torch.gather(all_sim_attention_mask, 1, gather_idx)

            selected_sim_codes_flat = selected_sim_codes.view(-1, FlatLen)
            selected_sim_mask_flat = selected_sim_mask.view(-1, FlatLen)

            sim_emb = self.shared(selected_sim_codes_flat)
            
            if self.sim_projector is not None:
                real_sim_hidden = self.sim_projector(sim_emb)
            else:
                real_sim_hidden = sim_emb
            
            real_sim_mask_extended = self.get_extended_attention_mask(selected_sim_mask_flat, encoder_hidden_states.dtype)

        # 3) Expand tensors along the beam dimension.
        K = beam_size
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(K, dim=0)
        encoder_mask = encoder_mask.repeat_interleave(K, dim=0)
        
        if real_sim_hidden is not None and real_sim_hidden.size(0) == batch_size:
            real_sim_hidden = real_sim_hidden.repeat_interleave(K, dim=0)
            real_sim_mask_extended = real_sim_mask_extended.repeat_interleave(K, dim=0)
        
        curr_ids = curr_ids.repeat_interleave(K, dim=0) 
        curr_ids = torch.cat([curr_ids, topk_sids.view(-1, 1)], dim=1) 
        
        beam_scores = torch.log(topk_scores + 1e-9).view(-1)
        
        for step in range(max_length - 1):
            logits, _ = self.forward_decoder(
                curr_ids,
                encoder_hidden_states, encoder_mask,
                real_sim_hidden, real_sim_mask_extended
            )
            
            next_token_logits = logits[:, -1, :]
            next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
            
            next_scores = beam_scores.unsqueeze(1) + next_token_logprobs
            next_scores = next_scores.view(batch_size, K * self.vocab_size)
            
            best_scores, best_indices = next_scores.topk(K, dim=1)
            
            prev_beam_indices = best_indices // self.vocab_size
            new_token_indices = best_indices % self.vocab_size
            
            batch_base = (torch.arange(batch_size, device=device) * K).unsqueeze(1)
            global_indices = (batch_base + prev_beam_indices).view(-1)
            
            beam_scores = best_scores.view(-1)
            
            curr_ids = curr_ids[global_indices]
            curr_ids = torch.cat([curr_ids, new_token_indices.view(-1, 1)], dim=1)
            
            encoder_hidden_states = encoder_hidden_states[global_indices]
            encoder_mask = encoder_mask[global_indices]
            
            if self.use_gate and real_sim_hidden is not None:
                real_sim_hidden = real_sim_hidden[global_indices]
                real_sim_mask_extended = real_sim_mask_extended[global_indices]
            
        return curr_ids.view(batch_size, K, -1)


# ================= Training & Utils (Unchanged) =================

def calculate_pos_index(preds, labels, maxk=None):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    actual_beam = preds.shape[1]
    pos_index = torch.zeros((preds.shape[0], actual_beam), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(actual_beam):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index

def recall_at_k(pos_index, k):
    safe_k = min(k, pos_index.shape[1])
    return pos_index[:, :safe_k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
    safe_k = min(k, pos_index.shape[1])
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :safe_k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_gate = 0.0
    acc_stats = {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}, 2: {'correct': 0, 'total': 0}}
    
    for batch in tqdm(train_loader, desc="  Training", leave=False):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        
        sim_input_ids = batch['sim_input_ids'].to(device)
        sim_attention_mask = batch['sim_attention_mask'].to(device)
        
        sidtier = batch.get('sidtier', None)
        if sidtier is not None:
            sidtier = sidtier.to(device)

        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            sim_input_ids=sim_input_ids, 
            sim_attention_mask=sim_attention_mask, 
            labels=labels,
            sidtier=sidtier
        )
        
        loss, logits, gate_val = outputs

        if loss.dim() > 0: loss = loss.mean()
        if gate_val.dim() > 0: gate_val = gate_val.mean() 

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gate += gate_val.item() 
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)            
            seq_len = labels.shape[1]
            for i in range(3):
                if i < seq_len:
                    cur_labels = labels[:, i]
                    cur_preds = preds[:, i]
                    mask = (cur_labels != -100)
                    acc_stats[i]['correct'] += ((cur_preds == cur_labels) & mask).sum().item()
                    acc_stats[i]['total'] += mask.sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_gate = total_gate / len(train_loader) 
    
    metrics = {f'acc_{i}': (acc_stats[i]['correct']/acc_stats[i]['total'] if acc_stats[i]['total']>0 else 0) for i in range(3)}
    metrics['avg_gate'] = avg_gate 
    
    return avg_loss, metrics

def evaluate(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    exact_matches = {'pos_1': 0, 'pos_1_2': 0, 'pos_1_2_3': 0, 'total_samples': 0}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="  Evaluating", leave=False):
            input_ids = batch['history'].to(device)
            if input_ids.dim() == 3:
                 input_ids = input_ids.view(input_ids.size(0), -1)
            
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            all_sim_input_ids = batch['sim_input_ids'].to(device)
            sidtier = batch.get('sidtier', None)
            if sidtier is not None:
                sidtier = sidtier.to(device)
            
            if 'sim_attention_mask' in batch:
                all_sim_attention_mask = batch['sim_attention_mask'].to(device)
                if all_sim_attention_mask.dim() == 2:
                    all_sim_attention_mask = all_sim_attention_mask.unsqueeze(1)
            else:
                if all_sim_input_ids.dim() == 3: 
                    b, n, flat_len = all_sim_input_ids.shape
                    all_sim_attention_mask = (all_sim_input_ids != 0).long()
                else: 
                     all_sim_attention_mask = (all_sim_input_ids != 0).long()

            if isinstance(model, nn.DataParallel):
                preds = model.module.generate(
                    input_ids, attention_mask, 
                    all_sim_input_ids, all_sim_attention_mask, 
                    beam_size, sidtier=sidtier
                )
            else:
                preds = model.generate(
                    input_ids, attention_mask, 
                    all_sim_input_ids, all_sim_attention_mask, 
                    beam_size, sidtier=sidtier
                )
            
            preds = preds[:, :, 1:4] 
            labels = labels[:, 0:3]
            pos_index = calculate_pos_index(preds, labels, maxk=preds.shape[1])
            
            for k in topk_list:
                recalls['Recall@' + str(k)].append(recall_at_k(pos_index, k).mean().item())
                ndcgs['NDCG@' + str(k)].append(ndcg_at_k(pos_index, k).mean().item())

            top1_pred = preds[:, 0, :] 
            batch_size_curr = labels.size(0)
            exact_matches['total_samples'] += batch_size_curr
            
            top1_pred_list = top1_pred.cpu().tolist()
            labels_list = labels.cpu().tolist()

            for i in range(batch_size_curr):
                cur_pred = top1_pred_list[i]
                cur_label = labels_list[i]
                if len(cur_pred) >= 1 and cur_pred[0] == cur_label[0]:
                    exact_matches['pos_1'] += 1
                    if len(cur_pred) >= 2 and cur_pred[1] == cur_label[1]:
                        exact_matches['pos_1_2'] += 1
                        if len(cur_pred) >= 3 and cur_pred[2] == cur_label[2]:
                            exact_matches['pos_1_2_3'] += 1

    avg_recalls = {k: sum(v)/len(v) for k,v in recalls.items()}
    avg_ndcgs = {k: sum(v)/len(v) for k,v in ndcgs.items()}
    
    total = exact_matches['total_samples'] if exact_matches['total_samples'] > 0 else 1
    prob_stats = {
        'P_1': exact_matches['pos_1'] / total,
        'P_1_2': exact_matches['pos_1_2'] / total,
        'P_1_2_3': exact_matches['pos_1_2_3'] / total
    }

    return avg_recalls, avg_ndcgs, prob_stats

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
