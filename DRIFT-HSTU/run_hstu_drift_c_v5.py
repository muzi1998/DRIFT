"""
run_hstu_drift_c_v5.py — HSTU C v5: v4 + L_sub 辅助损失
========================================================
基于 HSTU C v4 (test=0.0858)，引入 L_sub 更新子序列表示：

L_total = L_main + λ_sub · L_sub
L_sub = λ_t · L_time + λ_c · L_structure

L_time: 子序列内相邻物品时间间隔加权平均（用真实 log_dt）
  → 鼓励时间相近的物品分到同一子序列
L_structure: 子序列内物品与质心的加权 1-cos 距离
  → x_long 不 detach（修复 1 已分离 output_emb，item_emb 更新不影响打分）

v5 相比 v4 的改动：
  1. forward 返回 a_utk, x_long（供 L_sub 计算）
  2. 新增 compute_l_time_v4（接收真实 log_dt）
  3. 新增 compute_l_structure_v4（x_long 不 detach）
  4. loss 加 λ_sub · L_sub
  5. 新增 --lambda_sub, --sub_lam_t, --sub_lam_c 参数
"""
from __future__ import annotations
import argparse, math, time, os
from pathlib import Path
from hstu_paths import resolve_args, portable_path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from stage0_common import build_glass_aligned_splits, GlassSplit
from run_hstu_kuairec import sampled_softmax_loss, set_seed, model_stats

DRIFT_MODE = 'c_v5'


# ==================== 数据加载（带时间戳）====================

def load_kuairec_sequences_with_ts(data_dir: str) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray], int]:
    """读取 KuaiRec parquet，返回 sequences + timestamps + num_items。"""
    dfs = []
    for name in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, f'{name}.parquet')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    unique_videos = sorted(df['video_id'].unique())
    vid_map = {v: i + 1 for i, v in enumerate(unique_videos)}
    sequences, timestamps = {}, {}
    for uid, group in df.groupby('user_id'):
        sequences[int(uid)] = [vid_map[v] for v in group['video_id'].values]
        timestamps[int(uid)] = group['timestamp'].values.astype(np.float32)
    return sequences, timestamps, len(unique_videos)


def compute_log_dt(ts_array: np.ndarray, max_len: int) -> np.ndarray:
    """从时间戳数组计算 log(1+|Δt|)，对齐到 max_len。"""
    dt = np.diff(ts_array, prepend=ts_array[0] if len(ts_array) > 0 else 0.0)
    log_dt = np.log1p(np.abs(dt).clip(min=0).astype(np.float32))
    arr = np.full(max_len, math.log(2.0), dtype=np.float32)
    if len(log_dt) > 0:
        arr[-len(log_dt):] = log_dt[-max_len:]
    return arr


# ==================== HSTU 模型（修复 1 + 修复 8）====================

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))
    def forward(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.weight.device)
        rel = positions[:, None] - positions[None, :]
        return self.weight[rel + self.weight.size(0) // 2]


class HSTUBlockV4(nn.Module):
    """HSTU Block，支持 softmax 归一化（修复 8，可选）。"""
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
        valid_key_mask = ~padding_mask  # padding_mask: True=padding → ~=True=valid
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
    """HSTU with separate output_embedding (修复 1) + optional softmax attention (修复 8)。"""
    def __init__(self, num_items, max_len, hidden_dim, linear_dim, attention_dim,
                 num_blocks, num_heads, dropout, attn_dropout, use_softmax=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.output_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)  # 修复 1: 独立
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HSTUBlockV4(hidden_dim, linear_dim, attention_dim, num_heads, max_len,
                        dropout, attn_dropout, use_softmax=use_softmax)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)  # 独立初始化
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.output_embedding.weight[0].zero_()

    def output_weight(self):
        return self.output_embedding.weight[1:]  # 修复 1: 不共享，不归一化


# ==================== SubseqRepresentationV3（修复 3 + 修复 4）====================

class SubseqRepresentationV3(nn.Module):
    """v3: FFN 非线性 + 子序列间 attention + 真实时间戳。"""
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
        # 修复 3: FFN 非线性
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        # 修复 3: 子序列间 attention (K×K, 非常轻量)
        self.intra_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.intra_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_long, mask_long, d_u0, log_dt=None):
        B, L, D = x_long.shape
        device = x_long.device
        kappa = self.K_min + (self.K_max - self.K_min) * torch.sigmoid(self.theta_d * d_u0)

        cos_pair = F.cosine_similarity(x_long[:, 1:, :], x_long[:, :-1, :], dim=-1, eps=1e-6)
        cos_pair = torch.cat([torch.zeros(B, 1, device=device), 1.0 - cos_pair], dim=1)
        # 修复 4: 使用真实 log_dt
        if log_dt is not None:
            log_dt_long = log_dt[:, -L:]  # 对齐 long 段
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

        # 修复 3: FFN 非线性变换
        h_uk = h_uk + self.ffn(h_uk)
        h_uk = self.ffn_norm(h_uk)

        # 修复 3: 子序列间 attention (让 K 个子序列表示互相交互)
        m_uk_bool = m_uk >= 0.5
        huk_key_padding = ~m_uk_bool  # True=invalid for key_padding_mask
        attn_out, _ = self.intra_attn(h_uk, h_uk, h_uk, key_padding_mask=huk_key_padding, need_weights=False)
        h_uk = h_uk + attn_out
        h_uk = self.intra_norm(h_uk)

        return h_uk, m_uk_bool, a_utk, x_long


# ==================== 融合组件 ====================

class HukCrossAttnV2(nn.Module):
    """Cross-Attention with LayerScale (from drift_subseq_v2)."""
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
    """输出层门控融合 (from drift_subseq)."""
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


# ==================== L_sub 辅助损失（v5 新增）====================

def compute_l_time_v4(a_utk, log_dt_long, eps=1e-8):
    """L_time: 子序列内相邻物品的时间间隔加权平均（用真实 log_dt）。

    对每个子序列 k: Σ_t a_{t,k}·a_{t-1,k}·log(1+Δt_t) / Σ_t a_{t,k}·a_{t-1,k}
    鼓励时间相近的物品分到同一子序列。

    Args:
        a_utk:       (B, L, K) 软分配权重
        log_dt_long: (B, L) 真实 log(1+|Δt|)
    """
    B, L, K = a_utk.shape
    if L < 2:
        return a_utk.new_zeros(())
    a_lag = a_utk[:, :-1, :]  # (B, L-1, K)
    a_now = a_utk[:, 1:, :]   # (B, L-1, K)
    pair = a_lag * a_now       # (B, L-1, K)

    ld = log_dt_long[:, 1:].unsqueeze(-1)  # (B, L-1, 1) — 真实时间间隔

    num = (pair * ld).sum(dim=1)        # (B, K)
    den = pair.sum(dim=1).clamp_min(eps)  # (B, K)
    return (num / den).mean()


def compute_l_structure_v4(x_long, a_utk, eps=1e-8):
    """L_structure: 子序列内物品与质心的加权 1-cos 距离。

    x_long 不 detach — v4 已分离 output_embedding，item_embedding 更新不影响打分矩阵。
    L_structure 的梯度回传到 item_embedding，鼓励子序列内物品语义紧凑。

    Args:
        x_long: (B, L, D) — 不 detach
        a_utk:  (B, L, K)
    """
    B, L, D = x_long.shape
    K = a_utk.size(2)

    weights = a_utk.sum(dim=1).clamp_min(eps)        # (B, K)
    centroid = torch.einsum("bld,blk->bkd", x_long, a_utk) / weights.unsqueeze(-1)  # (B, K, D)

    x_n = F.normalize(x_long, dim=-1, eps=eps)       # (B, L, D)
    c_n = F.normalize(centroid, dim=-1, eps=eps)     # (B, K, D)
    cos_tk = torch.einsum("bld,bkd->blk", x_n, c_n)  # (B, L, K)

    num = (a_utk * (1.0 - cos_tk)).sum(dim=1)        # (B, K)
    den = a_utk.sum(dim=1).clamp_min(eps)            # (B, K)
    return (num / den).mean()


def compute_l_sub_v5(a_utk, x_long, log_dt_long, lam_t=0.5, lam_c=0.5):
    """L_sub = λ_t · L_time + λ_c · L_structure"""
    l_time = compute_l_time_v4(a_utk, log_dt_long)
    l_struct = compute_l_structure_v4(x_long, a_utk)
    return lam_t * l_time + lam_c * l_struct, l_time, l_struct


# ==================== Dataset（带时间戳）====================

class DriftDatasetV4(Dataset):
    """带时间戳的 Dataset。"""
    def __init__(self, splits, timestamps, max_len, d_u_cache):
        self.users = [u for u, s in splits.items() if len(s.train_seq) >= 1]
        self.splits, self.timestamps, self.max_len, self.d_u_cache = splits, timestamps, max_len, d_u_cache
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        u = self.users[idx]; s = self.splits[u]
        full = s.train_seq + [s.train_target]; full = full[-self.max_len:]
        seq = np.zeros(self.max_len, dtype=np.int64)
        seq[self.max_len - len(full):] = np.array(full)
        # 时间戳
        ts = self.timestamps.get(u, np.array([0.0]))
        log_dt = compute_log_dt(ts[-len(full):], self.max_len)
        return (torch.from_numpy(seq), torch.tensor(full[-1], dtype=torch.long),
                torch.tensor(self.d_u_cache.get(u, 0.0), dtype=torch.float32),
                torch.from_numpy(log_dt))


def forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seq, d_u0, log_dt, recent_r):
    """方式 C v4: A + B + 5 项修复。"""
    B = seq.size(0)
    seq_recent, seq_long, mask_recent, mask_long = split_recent_long(seq, recent_r)

    # 子序列表示（修复 3: FFN+attn, 修复 4: 真实 log_dt）
    x_long = model.item_embedding(seq_long) * math.sqrt(model.hidden_dim)
    log_dt_long = log_dt[:, :seq_long.size(1)]  # long 段的 log_dt
    h_uk, m_uk_bool, a_utk, x_long_out = subseq_enc(x_long, mask_long, d_u0, log_dt_long)
    huk_mask = ~m_uk_bool

    # 近期物品前向 (mask 修正: padding_mask=True 表示 padding)
    L_r = seq_recent.size(1)
    pos = torch.arange(L_r, device=seq.device).unsqueeze(0).expand(B, L_r)
    x = model.item_embedding(seq_recent) * math.sqrt(model.hidden_dim)
    x = x + model.pos_embedding(pos)
    padding_mask = ~mask_recent  # True=padding (正确语义!)
    x = model.emb_dropout(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    for i, blk in enumerate(model.blocks):
        x = blk(x, padding_mask=padding_mask)    # Self-Attn (正确 mask)
        x = huk_modules[i](x, h_uk, huk_mask)     # CrossAttn(h_uk)
    x = model.final_norm(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    final_h = F.normalize(x[:, -1, :], p=2, dim=-1, eps=1e-6)

    # 输出层门控融合
    fused = output_gate(final_h, h_uk, m_uk_bool)
    return F.normalize(fused, p=2, dim=-1, eps=1e-6), a_utk, x_long_out, log_dt_long


def evaluate(model, subseq_enc, huk_modules, output_gate, splits, timestamps, users, max_len, bs, device, mode, topk_list, d_u_cache, recent_r):
    model.eval(); subseq_enc.eval(); huk_modules.eval(); output_gate.eval()
    hd = {k: 0.0 for k in topk_list}; nd = {k: 0.0 for k in topk_list}; tot = 0
    with torch.inference_mode():
        for st in range(0, len(users), bs):
            bu = users[st:st+bs]; seqs, targets, log_dts = [], [], []
            for u in bu:
                sp = splits[u]; hist = sp.val_history
                tgt = sp.val_target if mode == 'val' else sp.test_target
                a = np.zeros(max_len, dtype=np.int64); h = hist[-max_len:]; a[max_len-len(h):] = np.array(h)
                seqs.append(a); targets.append(tgt)
                ts = timestamps.get(u, np.array([0.0]))
                log_dts.append(compute_log_dt(ts[-len(h):], max_len))
            seqs = torch.from_numpy(np.stack(seqs)).to(device)
            tgt_t = torch.tensor(targets, dtype=torch.long).to(device)
            d_u0 = torch.tensor([d_u_cache.get(u, 0.0) for u in bu], dtype=torch.float32, device=device)
            log_dt = torch.from_numpy(np.stack(log_dts)).to(device)
            h, _, _, _ = forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seqs, d_u0, log_dt, recent_r)
            scores = h @ model.output_weight().t()
            for ri, u in enumerate(bu):
                hist = splits[u].val_history; tgt = int(tgt_t[ri].item())
                mi = [i for i in hist if int(i) != tgt]
                if mi: scores[ri, torch.tensor(mi, device=device, dtype=torch.long)-1] = -1e9
            _, tp = torch.topk(scores, k=max(topk_list), dim=1); tp = tp + 1
            for ri in range(len(bu)):
                tgt = int(tgt_t[ri].item()); pl = tp[ri].tolist()
                for k in topk_list:
                    if tgt in pl[:k]: hd[k] += 1; nd[k] += 1/math.log2(pl[:k].index(tgt)+2)
                tot += 1
    if tot == 0: return {f'Recall@{k}': 0 for k in topk_list}, {f'NDCG@{k}': 0 for k in topk_list}
    return {f'Recall@{k}': hd[k]/tot for k in topk_list}, {f'NDCG@{k}': nd[k]/tot for k in topk_list}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--kuairec_data_dir', type=str, required=True)
    p.add_argument('--drift_cache_dir', type=str, default='outputs/hstu/drift_cache')
    p.add_argument('--max_len', type=int, default=1000)
    p.add_argument('--recent_r', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=96)
    p.add_argument('--linear_dim', type=int, default=16)
    p.add_argument('--attention_dim', type=int, default=16)
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--K_max', type=int, default=20)  # 修复 6: 4→20
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--optimizer', type=str, default='adamw')
    p.add_argument('--adam_beta1', type=float, default=0.9)
    p.add_argument('--adam_beta2', type=float, default=0.98)
    p.add_argument('--adam_eps', type=float, default=1e-8)
    p.add_argument('--num_negatives', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=20260521)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--early_stop_patience', type=int, default=50)
    p.add_argument('--selection_metric', type=str, default='NDCG@10')
    p.add_argument('--load_stage0_emb', action='store_true', default=False)
    p.add_argument('--use_softmax_attn', action='store_true', default=False)  # 修复 8
    # v5: L_sub 参数
    p.add_argument('--lambda_sub', type=float, default=0.05)
    p.add_argument('--sub_lam_t', type=float, default=0.5)
    p.add_argument('--sub_lam_c', type=float, default=0.5)
    p.add_argument('--output_dir', type=str, default='outputs/hstu',
                   help='run artifacts, relative to the repository root')
    args = resolve_args(p.parse_args())
    if not 1 <= args.recent_r < args.max_len or args.K_max < 1:
        p.error('Require 1 <= recent_r < max_len and K_max >= 1')
    return args


def main():
    args = parse_args(); set_seed(args.seed)
    sd = Path(args.output_dir)
    ld = sd/'logs'; ld.mkdir(parents=True, exist_ok=True); cd = sd/'ckpt'; cd.mkdir(parents=True, exist_ok=True)
    lp = str(ld/f'HSTU_kuairec_maxlen{args.max_len}_R{args.recent_r}_K{args.K_max}_seed{args.seed}_drift_c_v5.log')
    sp = str(cd/f'HSTU_kuairec_maxlen{args.max_len}_R{args.recent_r}_K{args.K_max}_seed{args.seed}_drift_c_v5.pth')
    lf = open(lp, 'a', encoding='utf-8')
    def log(m):
        ts = time.strftime('%Y-%m-%d %H:%M:%S'); l = f'[{ts}] {m}'; print(l, flush=True); lf.write(l+'\n'); lf.flush()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Loading data (with timestamps)...')
    seqs, timestamps, ni = load_kuairec_sequences_with_ts(args.kuairec_data_dir)
    splits = build_glass_aligned_splits(seqs, args.max_len)
    eu = sorted([u for u, s in splits.items() if len(s.train_seq) >= 1])
    d_u_cache = {}
    for s2 in ['train', 'valid', 'test']:
        df = pd.read_parquet(f'{args.drift_cache_dir}/d_u_init_{s2}.parquet')
        d_u_cache.update({int(r['user']): float(r['d_u_init']) for _, r in df.iterrows()})
    log(f'  users={len(seqs)}, items={ni}, d_u={len(d_u_cache)}, ts_users={len(timestamps)}')
    ds = DriftDatasetV4(splits, timestamps, args.max_len, d_u_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=str(device).startswith('cuda'))
    model = HSTUV4(ni, args.recent_r, args.hidden_dim, args.linear_dim, args.attention_dim,
                   args.num_blocks, args.num_heads, args.dropout, args.attn_dropout,
                   use_softmax=args.use_softmax_attn).to(device)
    subseq_enc = SubseqRepresentationV3(args.hidden_dim, K_max=args.K_max, max_long_len=args.max_len-args.recent_r).to(device)
    huk_modules = nn.ModuleList([HukCrossAttnV2(args.hidden_dim, num_heads=4, dropout=0.1) for _ in range(args.num_blocks)]).to(device)
    output_gate = OutputGate(args.hidden_dim).to(device)
    if args.load_stage0_emb:
        p = os.path.join(args.drift_cache_dir, 'stage0_state.pt')
        if os.path.exists(p):
            model.item_embedding.weight.data.copy_(torch.load(p, map_location=device, weights_only=False)['shared_emb'])
            log('  Loaded stage0 emb (item_embedding only, output_embedding independent)')
    params = list(model.parameters()) + list(subseq_enc.parameters()) + list(huk_modules.parameters()) + list(output_gate.parameters())
    oc = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    opt = oc(params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    tk = [1, 3, 5, 10, 20]
    log(f'mode=C_v5(L_sub), recent_r={args.recent_r}, K_max={args.K_max}, softmax_attn={args.use_softmax_attn}, λ_sub={args.lambda_sub}')
    log(f'  fixes: [1]separate_emb [3]ffn+intra_attn [4]real_timestamp [6]K_max={args.K_max} [8]softmax={args.use_softmax_attn}')
    log(f'  L_sub: λ_t={args.sub_lam_t} λ_c={args.sub_lam_c} (x_long NOT detached)')
    log(f'stats: {model_stats(model)}')
    bv, bs2, be, ni2 = -1.0, None, None, 0; t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train(); subseq_enc.train(); huk_modules.train(); output_gate.train(); el, sc = 0.0, 0
        for step, (seq, tgt, du, log_dt) in enumerate(loader, 1):
            seq, tgt, du, log_dt = seq.to(device), tgt.to(device), du.to(device), log_dt.to(device)
            h, a_utk, x_long, log_dt_long = forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seq, du, log_dt, args.recent_r)
            l_main = sampled_softmax_loss(h, tgt-1, model.output_weight(), int(args.num_negatives), float(args.temperature))
            l_sub, l_time, l_struct = compute_l_sub_v5(a_utk, x_long, log_dt_long, args.sub_lam_t, args.sub_lam_c)
            loss = l_main + args.lambda_sub * l_sub
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            el += float(loss.item()); sc += 1
            if step % args.log_every == 0: log(f'  ep={ep} step={step} loss={loss.item():.4f} main={l_main.item():.4f} sub={l_sub.item():.4f}(t={l_time.item():.4f} c={l_struct.item():.4f})')
        vr, vn = None, None; stop = False
        if ep % max(1, args.eval_every) == 0 or ep == args.epochs:
            vr, vn = evaluate(model, subseq_enc, huk_modules, output_gate, splits, timestamps, eu, args.max_len, args.eval_batch_size, device, 'val', tk, d_u_cache, args.recent_r)
            vm = {**vr, **vn}; vs = vm[args.selection_metric]
            if vs > bv:
                bv, be = vs, ep
                bs2 = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       'subseq': {k: v.detach().cpu().clone() for k, v in subseq_enc.state_dict().items()},
                       'huk': {k: v.detach().cpu().clone() for k, v in huk_modules.state_dict().items()},
                       'gate': {k: v.detach().cpu().clone() for k, v in output_gate.state_dict().items()}}
                torch.save(bs2, sp); log(f'  >> best {args.selection_metric}={vs:.4f} saved'); ni2 = 0
            else:
                ni2 += 1
                if args.early_stop_patience > 0 and ni2 >= args.early_stop_patience: stop = True
        vs2 = '' if vr is None else ' | val ' + ' '.join(f'{k}={v:.4f}' for k, v in {**vr, **vn}.items())
        log(f'ep={ep}/{args.epochs} loss={el/max(sc,1):.4f}{vs2}')
        if stop: log(f'Early stop at {ep}'); break
    if bs2:
        model.load_state_dict(bs2['model']); subseq_enc.load_state_dict(bs2['subseq'])
        huk_modules.load_state_dict(bs2['huk']); output_gate.load_state_dict(bs2['gate'])
    tr, tn = evaluate(model, subseq_enc, huk_modules, output_gate, splits, timestamps, eu, args.max_len, args.eval_batch_size, device, 'test', tk, d_u_cache, args.recent_r)
    log(f'=== Results === best_epoch={be}')
    log('test: ' + ' '.join(f'{k}={v:.4f}' for k, v in {**tr, **tn}.items()))
    log(f'total_time={time.time()-t0:.1f}s'); lf.close()

if __name__ == '__main__': main()
