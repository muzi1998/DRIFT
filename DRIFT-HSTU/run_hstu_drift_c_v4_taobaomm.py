"""
run_hstu_drift_c_v4_taobaomm.py — HSTU C v4 Taobao MM 变体（无时间戳）
=====================================================================
基于 run_hstu_drift_c_v4.py，移除所有时间戳相关组件：

时间戳移除清单（共 6 处）：
  1. 删除 load_kuairec_sequences_with_ts → 改为 load_taobao_mm_splits（从 GLASS parquet 读取 flat_id）
  2. 删除 compute_log_dt 函数
  3. DriftDatasetV4: 删除 timestamps/log_dt，__getitem__ 只返回 (seq, tgt, d_u0)
  4. SubseqRepresentationV3.forward: 删除 log_dt 参数，phi 特征从 [cos, log_dt, diff, d0] 改为 [cos, diff, d0]
     → in_dim 从 1+1+D+1 改为 1+D+1
  5. forward_hstu_drift_c_v4: 删除 log_dt 传参
  6. evaluate: 删除 timestamps/log_dt 计算

用户偏好偏移处理：
  - Taobao MM 无 stage0 drift cache → d_u0 = 0（全局常数）
  - kappa = K_min + (K_max - K_min) * sigmoid(theta_d * 0) = (1+K_max)/2 ≈ 10.5
  - 子序列划分仍可学习（基于 cos 相似度 + embedding diff），但无 per-user 自适应

保留的 v4 修复：
  [1] 分离 input/output embedding
  [3] FFN 非线性 + 子序列间 attention
  [6] K_max=20
  [8] 可选 softmax attention

对齐 GLASS:
  - 数据: GLASS prepare_glass.py 输出 → flat_id 转换
  - 超参: lr=1e-4, AdamW, wd=0.01, bs=512, epochs=100, early_stop=50, NDCG@20
  - 评估: 不排除 seen-item
  - item embedding 随机初始化
"""
from __future__ import annotations
import argparse, math, time, os
from pathlib import Path
from hstu_paths import resolve_args, portable_path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DRIFT_MODE = 'c_v4_taobaomm'


# ==================== 数据加载（无时间戳）====================

class RecSplit:
    def __init__(self, train_seq, train_target, val_history, val_target, test_target):
        self.train_seq = train_seq
        self.train_target = train_target
        self.val_history = val_history
        self.val_target = val_target
        self.test_target = test_target


def load_taobao_mm_splits(data_dir: str) -> Tuple[Dict[str, RecSplit], int]:
    """读取 prepare_taobao_mm_for_rec.py 输出的 parquet，构建 split。无时间戳。"""
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    valid_df = pd.read_parquet(os.path.join(data_dir, 'valid.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    splits = {}
    for _, row in train_df.iterrows():
        uid = str(row['user'])
        splits[uid] = RecSplit(list(row['seq']), int(row['target']), [], 0, 0)
    for _, row in valid_df.iterrows():
        uid = str(row['user'])
        if uid in splits:
            splits[uid].val_history = list(row['seq'])
            splits[uid].val_target = int(row['target'])
    for _, row in test_df.iterrows():
        uid = str(row['user'])
        if uid in splits:
            splits[uid].test_target = int(row['target'])
    # num_items = max flat_id across ALL seqs and targets
    all_max = 0
    for df in [train_df, valid_df, test_df]:
        for s in df['seq']:
            m = max(s)
            if m > all_max:
                all_max = m
        t_max = int(df['target'].max())
        if t_max > all_max:
            all_max = t_max
    return splits, int(all_max)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def model_stats(model):
    return {'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


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


# ==================== HSTU 模型（修复 1 + 修复 8）====================

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))
    def forward(self, seq_len):
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
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.output_embedding.weight[0].zero_()

    def output_weight(self):
        return self.output_embedding.weight[1:]


# ==================== SubseqRepresentationV3（移除时间戳）====================

class SubseqRepresentationV3NoTS(nn.Module):
    """v3 无时间戳版: FFN 非线性 + 子序列间 attention，phi 特征去掉 log_dt。

    原版 phi = [cos_pair(1), log_dt(1), diff(D), d0(1)]  → in_dim = D+3
    本版 phi = [cos_pair(1), diff(D), d0(1)]              → in_dim = D+2
    """
    def __init__(self, hidden_dim, K_max=20, max_long_len=950):
        super().__init__()
        self.hidden_dim, self.K_max = hidden_dim, K_max
        # in_dim: cos(1) + diff(D) + d0(1) = D + 2（移除了 log_dt 的 1 维）
        in_dim = 1 + hidden_dim + 1
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
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.intra_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.intra_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_long, mask_long, d_u0):
        """无 log_dt 参数。"""
        B, L, D = x_long.shape
        device = x_long.device
        kappa = self.K_min + (self.K_max - self.K_min) * torch.sigmoid(self.theta_d * d_u0)

        cos_pair = F.cosine_similarity(x_long[:, 1:, :], x_long[:, :-1, :], dim=-1, eps=1e-6)
        cos_pair = torch.cat([torch.zeros(B, 1, device=device), 1.0 - cos_pair], dim=1)
        # 无 log_dt：phi = [cos_pair, diff, d0]
        diff = torch.cat([torch.zeros(B, 1, D, device=device), x_long[:, 1:, :] - x_long[:, :-1, :]], dim=1)
        d0_exp = d_u0.unsqueeze(1).expand(B, L).unsqueeze(-1)
        phi = torch.cat([cos_pair.unsqueeze(-1), diff, d0_exp], dim=-1)
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


# ==================== 融合组件（不变）====================

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


# ==================== Dataset（无时间戳）====================

class DriftDatasetV4NoTS(Dataset):
    """无时间戳的 Dataset。返回 (seq, tgt, d_u0)。"""
    def __init__(self, splits, max_len, d_u_cache):
        self.users = [u for u, s in splits.items() if len(s.train_seq) >= 1]
        self.splits, self.max_len, self.d_u_cache = splits, max_len, d_u_cache
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        u = self.users[idx]; s = self.splits[u]
        full = s.train_seq + [s.train_target]; full = full[-self.max_len:]
        seq = np.zeros(self.max_len, dtype=np.int64)
        seq[self.max_len - len(full):] = np.array(full)
        return (torch.from_numpy(seq),
                torch.tensor(full[-1], dtype=torch.long),
                torch.tensor(self.d_u_cache.get(u, 0.0), dtype=torch.float32))


# ==================== Forward（无时间戳）====================

def forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seq, d_u0, recent_r):
    """方式 C v4 无时间戳版。"""
    B = seq.size(0)
    seq_recent, seq_long, mask_recent, mask_long = split_recent_long(seq, recent_r)

    # 子序列表示（无 log_dt）
    x_long = model.item_embedding(seq_long) * math.sqrt(model.hidden_dim)
    h_uk, m_uk_bool, a_utk, x_long_out = subseq_enc(x_long, mask_long, d_u0)
    huk_mask = ~m_uk_bool

    # 近期物品前向
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
    return F.normalize(fused, p=2, dim=-1, eps=1e-6)


# ==================== Evaluation（无时间戳）====================

def evaluate(model, subseq_enc, huk_modules, output_gate, splits, users, max_len, bs, device, mode, topk_list, d_u_cache, recent_r):
    model.eval(); subseq_enc.eval(); huk_modules.eval(); output_gate.eval()
    hd = {k: 0.0 for k in topk_list}; nd = {k: 0.0 for k in topk_list}; tot = 0
    with torch.inference_mode():
        for st in range(0, len(users), bs):
            bu = users[st:st+bs]; seqs, targets = [], []
            for u in bu:
                sp = splits[u]; hist = sp.val_history
                tgt = sp.val_target if mode == 'val' else sp.test_target
                a = np.zeros(max_len, dtype=np.int64); h = hist[-max_len:]; a[max_len-len(h):] = np.array(h)
                seqs.append(a); targets.append(tgt)
            seqs = torch.from_numpy(np.stack(seqs)).to(device)
            tgt_t = torch.tensor(targets, dtype=torch.long).to(device)
            d_u0 = torch.tensor([d_u_cache.get(u, 0.0) for u in bu], dtype=torch.float32, device=device)
            h = forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seqs, d_u0, recent_r)
            scores = h @ model.output_weight().t()
            # 不排除 seen-item（与 GLASS 对齐）
            _, tp = torch.topk(scores, k=max(topk_list), dim=1); tp = tp + 1
            for ri in range(len(bu)):
                tgt = int(tgt_t[ri].item()); pl = tp[ri].tolist()
                for k in topk_list:
                    if tgt in pl[:k]: hd[k] += 1; nd[k] += 1/math.log2(pl[:k].index(tgt)+2)
                tot += 1
    if tot == 0: return {f'Recall@{k}': 0 for k in topk_list}, {f'NDCG@{k}': 0 for k in topk_list}
    return {f'Recall@{k}': hd[k]/tot for k in topk_list}, {f'NDCG@{k}': nd[k]/tot for k in topk_list}


# ==================== Main ====================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--max_len', type=int, default=1000)
    p.add_argument('--recent_r', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=96)
    p.add_argument('--linear_dim', type=int, default=16)
    p.add_argument('--attention_dim', type=int, default=16)
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--K_max', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--optimizer', type=str, default='adamw')
    p.add_argument('--adam_beta1', type=float, default=0.9)
    p.add_argument('--adam_beta2', type=float, default=0.98)
    p.add_argument('--adam_eps', type=float, default=1e-8)
    p.add_argument('--num_negatives', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--early_stop_patience', type=int, default=50)
    p.add_argument('--selection_metric', type=str, default='NDCG@20')
    p.add_argument('--use_softmax_attn', action='store_true', default=False)
    p.add_argument('--output_dir', type=str, default='outputs/hstu',
                   help='run artifacts, relative to the repository root')
    args = resolve_args(p.parse_args())
    if not 1 <= args.recent_r < args.max_len or args.K_max < 1:
        p.error('Require 1 <= recent_r < max_len and K_max >= 1')
    return args


def main():
    args = parse_args(); set_seed(args.seed)
    sd = Path(args.output_dir)
    ld = sd/'logs'; ld.mkdir(parents=True, exist_ok=True)
    cd = sd/'ckpt'; cd.mkdir(parents=True, exist_ok=True)
    lp = str(ld/f'HSTU_taobaomm_maxlen{args.max_len}_R{args.recent_r}_K{args.K_max}_seed{args.seed}_drift_c_v4.log')
    sp = str(cd/f'HSTU_taobaomm_maxlen{args.max_len}_R{args.recent_r}_K{args.K_max}_seed{args.seed}_drift_c_v4.pth')
    lf = open(lp, 'a', encoding='utf-8')
    def log(m):
        ts = time.strftime('%Y-%m-%d %H:%M:%S'); l = f'[{ts}] {m}'
        print(l, flush=True); lf.write(l+'\n'); lf.flush()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Loading Taobao MM data (no timestamps) from {portable_path(args.data_dir)} ...')
    splits, ni = load_taobao_mm_splits(args.data_dir)
    eu = sorted([u for u, s in splits.items() if len(s.train_seq) >= 1 and len(s.val_history) >= 1])
    d_u_cache = {}  # 无 stage0 → d_u0=0 for all users
    log(f'  users={len(splits)}, items={ni}, eval_users={len(eu)}, d_u=0 (no stage0)')

    ds = DriftDatasetV4NoTS(splits, args.max_len, d_u_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=str(device).startswith('cuda'))
    model = HSTUV4(ni, args.recent_r, args.hidden_dim, args.linear_dim, args.attention_dim,
                   args.num_blocks, args.num_heads, args.dropout, args.attn_dropout,
                   use_softmax=args.use_softmax_attn).to(device)
    subseq_enc = SubseqRepresentationV3NoTS(args.hidden_dim, K_max=args.K_max,
                                             max_long_len=args.max_len-args.recent_r).to(device)
    huk_modules = nn.ModuleList([HukCrossAttnV2(args.hidden_dim, num_heads=4, dropout=0.1) for _ in range(args.num_blocks)]).to(device)
    output_gate = OutputGate(args.hidden_dim).to(device)

    params = list(model.parameters()) + list(subseq_enc.parameters()) + list(huk_modules.parameters()) + list(output_gate.parameters())
    oc = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    opt = oc(params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    tk = [1, 3, 5, 10, 20]
    log(f'mode=C_v4_taobaomm(no_ts), recent_r={args.recent_r}, K_max={args.K_max}, softmax_attn={args.use_softmax_attn}')
    log(f'  fixes: [1]separate_emb [3]ffn+intra_attn [6]K_max={args.K_max} [8]softmax={args.use_softmax_attn}')
    log(f'  removed: [4]real_timestamp (TaobaoMM has no ts), d_u0=0 (no stage0)')
    log(f'  phi: [cos_pair, diff, d0] (in_dim=D+2, was D+3)')
    log(f'stats: model={model_stats(model)} subseq={model_stats(subseq_enc)}')
    log(f'Config: max_len={args.max_len}, bs={args.batch_size}, lr={args.lr}, wd={args.weight_decay}, seed={args.seed}')

    bv, bs2, be, ni2 = -1.0, None, None, 0; t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train(); subseq_enc.train(); huk_modules.train(); output_gate.train(); el, sc = 0.0, 0
        for step, (seq, tgt, du) in enumerate(loader, 1):
            seq, tgt, du = seq.to(device), tgt.to(device), du.to(device)
            h = forward_hstu_drift_c_v4(model, subseq_enc, huk_modules, output_gate, seq, du, args.recent_r)
            loss = sampled_softmax_loss(h, tgt-1, model.output_weight(), int(args.num_negatives), float(args.temperature))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            el += float(loss.item()); sc += 1
            if step % args.log_every == 0: log(f'  ep={ep} step={step} loss={loss.item():.4f}')
        vr, vn = None, None; stop = False
        if ep % max(1, args.eval_every) == 0 or ep == args.epochs:
            vr, vn = evaluate(model, subseq_enc, huk_modules, output_gate, splits, eu,
                              args.max_len, args.eval_batch_size, device, 'val', tk, d_u_cache, args.recent_r)
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
    tr, tn = evaluate(model, subseq_enc, huk_modules, output_gate, splits, eu,
                      args.max_len, args.eval_batch_size, device, 'test', tk, d_u_cache, args.recent_r)
    log(f'=== Results === best_epoch={be}')
    log('test: ' + ' '.join(f'{k}={v:.4f}' for k, v in {**tr, **tn}.items()))
    log(f'total_time={time.time()-t0:.1f}s'); lf.close()

if __name__ == '__main__': main()
