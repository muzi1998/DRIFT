"""
hstu_drift_common.py — Shared HSTU C v4/v5 model components for ML-20M sliding training.

Contains: HSTU model (separate output_embedding), SubseqRepresentation, cross-attn fusion,
output gate, split_recent_long, drift forward pass, L_sub losses, sampled_softmax_loss.

Extracted from run_hstu_drift_c_v5_ml20m_aligned.py (models identical).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def model_stats(model):
    return {'total_params': int(sum(p.numel() for p in model.parameters()))}


def sampled_softmax_loss(hidden, target_index, output_weight, num_negatives, temperature):
    temperature = max(float(temperature), 1e-8)
    if num_negatives <= 0:
        logits = (hidden @ output_weight.t()) / temperature
        return F.cross_entropy(logits, target_index)
    num_classes = output_weight.size(0)
    neg_index = torch.randint(0, num_classes, (target_index.size(0), num_negatives), device=target_index.device)
    target_expanded = target_index.unsqueeze(1)
    collision_mask = neg_index.eq(target_expanded)
    while collision_mask.any():
        neg_index[collision_mask] = torch.randint(0, num_classes, (int(collision_mask.sum().item()),), device=target_index.device)
        collision_mask = neg_index.eq(target_expanded)
    pos_weight = output_weight[target_index]
    neg_weight = output_weight[neg_index]
    pos_logits = (hidden * pos_weight).sum(dim=-1, keepdim=True) / temperature
    neg_logits = torch.einsum('bd,bnd->bn', hidden, neg_weight) / temperature
    sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
    sampled_target = torch.zeros(target_index.size(0), dtype=torch.long, device=target_index.device)
    return F.cross_entropy(sampled_logits, sampled_target)


# ==================== HSTU Model (修复 1 separate output_emb + 修复 8 optional softmax) ====================

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))

    def forward(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.weight.device)
        rel = positions[:, None] - positions[None, :]
        return self.weight[rel + self.weight.size(0) // 2]


class HSTUBlockV4(nn.Module):
    def __init__(self, hidden_dim, linear_dim, attention_dim, num_heads, max_len, dropout, attn_dropout, use_softmax=False):
        super().__init__()
        self.hidden_dim, self.linear_dim, self.attention_dim, self.num_heads = hidden_dim, linear_dim, attention_dim, num_heads
        self.use_softmax = use_softmax
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.uvqk = nn.Linear(hidden_dim, num_heads * (2 * linear_dim + 2 * attention_dim), bias=False)
        self.rel_bias = RelativePositionalBias(max_len)
        self.norm_attn = nn.LayerNorm(num_heads * linear_dim)
        self.output = nn.Linear(num_heads * linear_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        nn.init.normal_(self.uvqk.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, padding_mask):
        B, L, _ = x.shape
        normed_x = self.norm_input(x)
        uvqk = self.uvqk(normed_x)
        ss = [self.num_heads * self.linear_dim] * 2 + [self.num_heads * self.attention_dim] * 2
        u, v, q, k = torch.split(uvqk, ss, dim=-1)
        u = F.silu(u).view(B, L, self.num_heads, self.linear_dim)
        v = F.silu(v).view(B, L, self.num_heads, self.linear_dim)
        q = F.silu(q).view(B, L, self.num_heads, self.attention_dim)
        k = F.silu(k).view(B, L, self.num_heads, self.attention_dim)
        causal_mask = torch.tril(torch.ones((L, L), device=x.device, dtype=torch.bool))
        valid_key_mask = ~padding_mask
        attn_mask = causal_mask.unsqueeze(0) & valid_key_mask.unsqueeze(1)
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k)
        attn = attn + self.rel_bias(L).unsqueeze(0).unsqueeze(0)
        if self.use_softmax:
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
        else:
            attn = F.silu(attn)
            attn = attn / max(L, 1)
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), 0.0)
        attn = self.attn_dropout(attn)
        attn_output = torch.einsum('bhnm,bmhd->bnhd', attn, v).reshape(B, L, self.num_heads * self.linear_dim)
        gated = u.reshape(B, L, self.num_heads * self.linear_dim) * self.norm_attn(attn_output)
        out = self.output(self.dropout(gated)) + x
        return out.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTUV4(nn.Module):
    def __init__(self, num_items, max_len, hidden_dim, linear_dim, attention_dim,
                 num_blocks, num_heads, dropout, attn_dropout, use_softmax=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.output_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HSTUBlockV4(hidden_dim, linear_dim, attention_dim, num_heads, max_len,
                        dropout, attn_dropout, use_softmax=use_softmax)
            for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.output_embedding.weight[0].zero_()

    def output_weight(self):
        return self.output_embedding.weight[1:]


class SubseqRepresentationV3(nn.Module):
    def __init__(self, hidden_dim: int, K_max: int = 20, max_long_len: int = 950):
        super().__init__()
        self.hidden_dim, self.K_max = hidden_dim, K_max
        in_dim = 1 + 1 + hidden_dim + 1
        self.mlp_b = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mu_k", torch.arange(1, K_max + 1, dtype=torch.float32))
        self.tau_a, self.tau_m, self.K_min = 1.0, 1.0, 1
        self.theta_d = nn.Parameter(torch.tensor(2.0))
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_in = nn.Linear(hidden_dim, hidden_dim)
        self.w_in = nn.Linear(hidden_dim, K_max, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(0.1))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.intra_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.intra_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_long, mask_long, d_u0, log_dt=None):
        B, L, D = x_long.shape
        device = x_long.device
        kappa = self.K_min + (self.K_max - self.K_min) * torch.sigmoid(self.theta_d * d_u0)
        cos_pair = F.cosine_similarity(x_long[:, 1:, :], x_long[:, :-1, :], dim=-1, eps=1e-6)
        cos_pair = torch.cat([torch.zeros(B, 1, device=device), 1.0 - cos_pair], dim=1)
        if log_dt is not None:
            log_dt_long = log_dt[:, -L:]
        else:
            log_dt_long = torch.full((B, L), math.log(2.0), device=device)
        diff = torch.cat([torch.zeros(B, 1, D, device=device), x_long[:, 1:, :] - x_long[:, :-1, :]], dim=1)
        d0_exp = d_u0.unsqueeze(1).expand(B, L).unsqueeze(-1)
        phi = torch.cat([cos_pair.unsqueeze(-1), log_dt_long.unsqueeze(-1), diff, d0_exp], dim=-1)
        b = torch.sigmoid(self.mlp_b(phi).squeeze(-1)) * mask_long.float()
        b[:, 0] = 0.0
        S_ut = b.cumsum(dim=1)
        S_uL = S_ut[:, -1:].clamp_min(1e-6)
        r_ut = 1.0 + (kappa.unsqueeze(1) - 1.0) * S_ut / S_uL
        k_idx = torch.arange(1, self.K_max + 1, device=device, dtype=torch.float32)
        m_uk = torch.sigmoid((kappa.unsqueeze(1) - k_idx.unsqueeze(0) + 0.5) / self.tau_m)
        diff2 = (r_ut.unsqueeze(-1) - self.mu_k.view(1, 1, -1)) ** 2
        ex = torch.exp(-diff2 / self.tau_a) * m_uk.unsqueeze(1)
        a_utk = ex / (ex.sum(dim=-1, keepdim=True) + 1e-8)
        a_utk = a_utk * mask_long.float().unsqueeze(-1)
        v_t = self.W_v(x_long)
        h_in = torch.tanh(self.W_in(x_long))
        r_in = self.w_in(h_in)
        lam = F.softmax(r_in + torch.log(a_utk.clamp_min(1e-8)), dim=1)
        h_uk = torch.einsum("bld,blk->bkd", v_t, lam)
        h_uk = self.layer_norm(h_uk)
        h_uk = h_uk + self.ffn(h_uk)
        h_uk = self.ffn_norm(h_uk)
        m_uk_bool = m_uk >= 0.5
        huk_key_padding = ~m_uk_bool
        attn_out, _ = self.intra_attn(h_uk, h_uk, h_uk, key_padding_mask=huk_key_padding, need_weights=False)
        h_uk = h_uk + attn_out
        h_uk = self.intra_norm(h_uk)
        return h_uk, m_uk_bool, a_utk, x_long


class HukCrossAttnV2(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, layerscale_init=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(layerscale_init * torch.ones(hidden_dim))

    def forward(self, hidden_states, h_uk, huk_mask=None):
        normed = self.layer_norm(hidden_states)
        ca_out, _ = self.cross_attn(normed, h_uk, h_uk, key_padding_mask=huk_mask, need_weights=False)
        return hidden_states + self.gamma * self.dropout(ca_out)


class OutputGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, final_h, h_uk, m_uk_bool):
        mask = m_uk_bool.float().unsqueeze(-1)
        h_uk_pooled = (h_uk * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        gate = torch.sigmoid(self.W_g(final_h))
        return (1.0 - gate) * final_h + gate * h_uk_pooled


def split_recent_long(seq, recent_r):
    r = min(recent_r, seq.size(1))
    if seq.size(1) <= r:
        seq_recent, seq_long = seq, seq[:, :1]
        mask_recent, mask_long = seq.ne(0), seq_long.ne(0)
    else:
        seq_long, seq_recent = seq[:, :-r], seq[:, -r:]
        mask_long, mask_recent = seq_long.ne(0), seq_recent.ne(0)
    return seq_recent, seq_long, mask_recent, mask_long


def forward_hstu_drift_c(model, subseq_enc, huk_modules, output_gate, seq, d_u0, log_dt, recent_r):
    """HSTU C v4/v5 forward: long段子序列表示 + recent段主任务 + 门控融合。"""
    B = seq.size(0)
    seq_recent, seq_long, mask_recent, mask_long = split_recent_long(seq, recent_r)
    x_long = model.item_embedding(seq_long) * math.sqrt(model.hidden_dim)
    log_dt_long = log_dt[:, :seq_long.size(1)]
    h_uk, m_uk_bool, a_utk, x_long_out = subseq_enc(x_long, mask_long, d_u0, log_dt_long)
    huk_mask = ~m_uk_bool
    L_r = seq_recent.size(1)
    pos = torch.arange(L_r, device=seq.device).unsqueeze(0).expand(B, L_r)
    x = model.item_embedding(seq_recent) * math.sqrt(model.hidden_dim)
    x = x + model.pos_embedding(pos)
    padding_mask = ~mask_recent
    x = model.emb_dropout(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    for i, blk in enumerate(model.blocks):
        x = blk(x, padding_mask=padding_mask)
        x = huk_modules[i](x, h_uk, huk_mask)
    x = model.final_norm(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    final_h = F.normalize(x[:, -1, :], p=2, dim=-1, eps=1e-6)
    fused = output_gate(final_h, h_uk, m_uk_bool)
    return F.normalize(fused, p=2, dim=-1, eps=1e-6), a_utk, x_long_out, log_dt_long


# ==================== L_sub losses (v5) ====================

def compute_l_time_v4(a_utk, log_dt_long, eps=1e-8):
    B, L, K = a_utk.shape
    if L < 2:
        return a_utk.new_zeros(())
    a_lag = a_utk[:, :-1, :]
    a_now = a_utk[:, 1:, :]
    pair = a_lag * a_now
    ld = log_dt_long[:, 1:].unsqueeze(-1)
    num = (pair * ld).sum(dim=1)
    den = pair.sum(dim=1).clamp_min(eps)
    return (num / den).mean()


def compute_l_structure_v4(x_long, a_utk, eps=1e-8):
    B, L, D = x_long.shape
    K = a_utk.size(2)
    weights = a_utk.sum(dim=1).clamp_min(eps)
    centroid = torch.einsum("bld,blk->bkd", x_long, a_utk) / weights.unsqueeze(-1)
    x_n = F.normalize(x_long, dim=-1, eps=eps)
    c_n = F.normalize(centroid, dim=-1, eps=eps)
    cos_tk = torch.einsum("bld,bkd->blk", x_n, c_n)
    num = (a_utk * (1.0 - cos_tk)).sum(dim=1)
    den = a_utk.sum(dim=1).clamp_min(eps)
    return (num / den).mean()


def compute_l_sub_v5(a_utk, x_long, log_dt_long, lam_t=0.5, lam_c=0.5):
    l_time = compute_l_time_v4(a_utk, log_dt_long)
    l_struct = compute_l_structure_v4(x_long, a_utk)
    return lam_t * l_time + lam_c * l_struct, l_time, l_struct
