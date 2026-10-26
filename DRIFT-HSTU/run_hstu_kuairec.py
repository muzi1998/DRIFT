"""
HSTU baseline aligned with GLASS on KuaiRec.

Data splitting, evaluation inputs, and metrics are aligned with GLASS's
prepare_glass_kuairec_subseq_shift_degree_seq_len.py (step2_create_basic_datasets).

Key alignments:
  - Reads raw KuaiRec parquet (train/valid/test), uses video_id as item ID
  - Leave-one-out: seq[:-2] as history, seq[-2] as val, seq[-1] as test
  - Test history = val history (NOT train+[val]), matching GLASS
  - NDCG formula equivalent (0-based rank+2 vs 1-based rank+1)
  - K values: {1,3,5,10,20}, selection metric: NDCG@10, early_stop patience: 50
  - hidden_dim=96 (aligned with GLASS d_model=96)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from hstu_paths import resolve_args, resolve_path, portable_path, portable_config
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HSTU baseline on KuaiRec (GLASS-aligned)')
    # ── data ──
    parser.add_argument('--kuairec_data_dir', type=str, required=True,
                        help='Directory containing train.parquet, valid.parquet, test.parquet')
    parser.add_argument('--max_len', type=int, default=50,
                        help='Maximum sequence length (truncated from the end)')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='Minimum sequence length to keep (after removing last 2 for val/test)')
    # ── model ──
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--linear_dim', type=int, default=16)
    parser.add_argument('--attention_dim', type=int, default=16)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    # ── training ──
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.98)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--num_negatives', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20260521)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--selection_metric', type=str, default='NDCG@10',
                        choices=['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20',
                                 'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20'])
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--save_path', type=str, default='',
                        help='Checkpoint save path. If empty, auto-generated as ckpt/HSTU_kuairec_maxlen{max_len}.pth')
    parser.add_argument('--log_path', type=str, default='',
                        help='Log file path. If empty, auto-generated as logs/HSTU_kuairec_maxlen{max_len}.log')
    return resolve_args(parser.parse_args())


# ==================== Data Loading ====================

def load_kuairec_sequences(data_dir: str) -> Tuple[Dict[int, List[int]], int]:
    """
    Read raw KuaiRec parquet files, build user → [video_id] sequences.
    Merges train + valid + test parquets, sorts by (user_id, timestamp).
    Re-indexes video_id to 1-based (0 = padding).
    """
    dfs = []
    for name in ['train', 'valid', 'test']:
        path = Path(data_dir) / f'{name}.parquet'
        if path.exists():
            dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    unique_videos = sorted(df['video_id'].unique())
    vid_map = {v: i + 1 for i, v in enumerate(unique_videos)}

    sequences: Dict[int, List[int]] = {}
    for uid, group in df.groupby('user_id'):
        sequences[int(uid)] = [vid_map[v] for v in group['video_id'].values]
    return sequences, len(unique_videos)


# ==================== GLASS-aligned Split ====================

@dataclass
class GlassSplit:
    """Mirrors GLASS step2_create_basic_datasets splitting logic."""
    train_seq: List[int]      # history[:-1], used for training input
    train_target: int         # history[-1], training target
    val_history: List[int]    # history (truncated), val/test eval input
    val_target: int           # seq[-2]
    test_target: int          # seq[-1]
    full: set                 # all items in seq (for masking)


def build_glass_aligned_splits(
    user_sequences: Dict[int, List[int]],
    max_len: int,
    min_seq_len: int,
) -> Dict[int, GlassSplit]:
    """
    GLASS-aligned leave-one-out:
      history_all = seq[:-2]
      history = history_all[-max_len:]
      train: history[:-1] → predict history[-1]
      val:   history → predict seq[-2]
      test:  history → predict seq[-1]   (same history as val, no val_target!)
    """
    splits: Dict[int, GlassSplit] = {}
    for user, seq in user_sequences.items():
        if len(seq) < min_seq_len + 2:
            continue
        history_all = seq[:-2]
        history = history_all[-max_len:]
        if len(history) < min_seq_len:
            continue
        if len(history) < 2:
            continue
        splits[user] = GlassSplit(
            train_seq=history[:-1],
            train_target=history[-1],
            val_history=history,
            val_target=seq[-2],
            test_target=seq[-1],
            full=set(seq),
        )
    return splits


# ==================== Training Dataset ====================

class HSTUTrainDataset(Dataset):
    def __init__(self, splits: Dict[int, GlassSplit], max_len: int) -> None:
        self.users = [u for u, s in splits.items() if len(s.train_seq) >= 1]
        self.splits = splits
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        train = self.splits[user].train_seq
        target = self.splits[user].train_target
        # Use last max_len-1 items as input, predict the target
        history = train[-(self.max_len - 1):] if self.max_len > 1 else train[-1:]
        seq = np.zeros(self.max_len, dtype=np.int64)
        seq[self.max_len - len(history):] = np.array(history, dtype=np.int64)
        return torch.from_numpy(seq), torch.tensor(target, dtype=torch.long)


# ==================== HSTU Model ====================

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.weight.device)
        rel = positions[:, None] - positions[None, :]
        rel = rel + self.max_seq_len - 1
        return self.weight[rel]


class HSTUBlock(nn.Module):
    def __init__(self, hidden_dim, linear_dim, attention_dim, num_heads, max_len, dropout, attn_dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.uvqk = nn.Linear(hidden_dim, num_heads * (2 * linear_dim + 2 * attention_dim), bias=False)
        self.rel_bias = RelativePositionalBias(max_len)
        self.norm_attn = nn.LayerNorm(num_heads * linear_dim)
        self.output = nn.Linear(num_heads * linear_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.uvqk.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, padding_mask):
        batch_size, seq_len, _ = x.shape
        normed_x = self.norm_input(x)
        uvqk = self.uvqk(normed_x)
        split_sizes = [
            self.num_heads * self.linear_dim,
            self.num_heads * self.linear_dim,
            self.num_heads * self.attention_dim,
            self.num_heads * self.attention_dim,
        ]
        u, v, q, k = torch.split(uvqk, split_sizes, dim=-1)
        u = F.silu(u); v = F.silu(v); q = F.silu(q); k = F.silu(k)
        u = u.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        valid_key_mask = ~padding_mask
        attn_mask = causal_mask.unsqueeze(0) & valid_key_mask.unsqueeze(1)
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k)
        attn = F.silu(attn + self.rel_bias(seq_len).unsqueeze(0).unsqueeze(0))
        attn = attn / max(seq_len, 1)
        attn = attn.masked_fill(~attn_mask.unsqueeze(1), 0.0)
        attn = self.attn_dropout(attn)
        attn_output = torch.einsum('bhnm,bmhd->bnhd', attn, v).reshape(batch_size, seq_len, self.num_heads * self.linear_dim)
        gated = u.reshape(batch_size, seq_len, self.num_heads * self.linear_dim) * self.norm_attn(attn_output)
        out = self.output(self.dropout(gated)) + x
        return out.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTU(nn.Module):
    def __init__(self, num_items, max_len, hidden_dim, linear_dim, attention_dim, num_blocks, num_heads, dropout, attn_dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HSTUBlock(hidden_dim, linear_dim, attention_dim, num_heads, max_len, dropout, attn_dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()

    def output_weight(self):
        return F.normalize(self.item_embedding.weight[1:], p=2, dim=-1, eps=1e-6)

    def forward(self, seq):
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_embedding(seq) * math.sqrt(self.hidden_dim)
        x = x + self.pos_embedding(positions)
        padding_mask = seq.eq(0)
        x = self.emb_dropout(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def predict_scores(self, seq):
        hidden = self.forward(seq)
        final_hidden = F.normalize(hidden[:, -1, :], p=2, dim=-1, eps=1e-6)
        return final_hidden @ self.output_weight().t()


# ==================== Loss ====================

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


# ==================== Evaluation ====================

def build_eval_sequences(splits, users, max_len, mode):
    """Build eval sequences. Both val and test use val_history (GLASS-aligned)."""
    seqs, targets = [], []
    for user in users:
        split = splits[user]
        if mode == 'val':
            history = split.val_history
            target = split.val_target
        elif mode == 'test':
            history = split.val_history  # GLASS: test uses same history as val (no val_target)
            target = split.test_target
        else:
            raise ValueError(f'Unknown eval mode: {mode}')
        arr = np.zeros(max_len, dtype=np.int64)
        history = history[-max_len:]
        arr[max_len - len(history):] = np.array(history, dtype=np.int64)
        seqs.append(arr)
        targets.append(target)
    return torch.from_numpy(np.stack(seqs)), torch.tensor(targets, dtype=torch.long)


def evaluate_exact(model, splits, users, max_len, eval_batch_size, device, mode, topk_list):
    model.eval()
    hit_dict = {k: 0.0 for k in topk_list}
    ndcg_dict = {k: 0.0 for k in topk_list}
    total = 0
    with torch.inference_mode():
        for start in range(0, len(users), eval_batch_size):
            batch_users = users[start:start + eval_batch_size]
            seqs, targets = build_eval_sequences(splits, batch_users, max_len=max_len, mode=mode)
            seqs = seqs.to(device)
            targets = targets.to(device)
            scores = model.predict_scores(seqs)
            # Mask out history items (exclude seen items from candidates)
            for row_idx, user in enumerate(batch_users):
                history = splits[user].val_history
                target = int(targets[row_idx].item())
                mask_items = [item for item in history if int(item) != target]
                if mask_items:
                    hist_tensor = torch.tensor(mask_items, device=device, dtype=torch.long) - 1
                    scores[row_idx, hist_tensor] = -1e9
            max_k = max(topk_list)
            _, topk_pred = torch.topk(scores, k=max_k, dim=1)
            topk_pred = topk_pred + 1  # back to 1-based item ID
            for row_idx in range(len(batch_users)):
                target = int(targets[row_idx].item())
                pred_list = topk_pred[row_idx].tolist()
                for k in topk_list:
                    if target in pred_list[:k]:
                        hit_dict[k] += 1.0
                        ndcg_dict[k] += 1.0 / math.log2(pred_list[:k].index(target) + 2.0)
                total += 1
    if total == 0:
        return {f'Recall@{k}': 0.0 for k in topk_list}, {f'NDCG@{k}': 0.0 for k in topk_list}
    recalls = {f'Recall@{k}': hit_dict[k] / total for k in topk_list}
    ndcgs = {f'NDCG@{k}': ndcg_dict[k] / total for k in topk_list}
    return recalls, ndcgs


def model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': int(total_params), 'trainable_params': int(trainable_params)}


# ==================== Main ====================

def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    # ── auto log path ──
    if not args.log_path:
        script_dir = resolve_path('outputs/hstu')
        log_dir = script_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        args.log_path = str(log_dir / f'HSTU_kuairec_maxlen{args.max_len}.log')
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log_path, 'a', encoding='utf-8')

    # ── auto save path ──
    if not args.save_path:
        script_dir = resolve_path('outputs/hstu')
        ckpt_dir = script_dir / 'ckpt'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        args.save_path = str(ckpt_dir / f'HSTU_kuairec_maxlen{args.max_len}.pth')
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        log_file.write(line + '\n')
        log_file.flush()

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith('cuda') else 'cpu')

    log(f'Loading KuaiRec data from {portable_path(args.kuairec_data_dir)} ...')
    user_sequences, num_items = load_kuairec_sequences(args.kuairec_data_dir)
    log(f'  num_users={len(user_sequences)}, num_items={num_items}')

    splits = build_glass_aligned_splits(user_sequences, args.max_len, args.min_seq_len)
    train_users = [u for u, s in splits.items() if len(s.train_seq) >= 1]
    eval_users = sorted(train_users)
    log(f'  splits: {len(splits)} users (train={len(train_users)}, eval={len(eval_users)})')

    dataset = HSTUTrainDataset(splits=splits, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=str(device).startswith('cuda'), drop_last=False)

    model = HSTU(
        num_items=num_items, max_len=args.max_len, hidden_dim=args.hidden_dim,
        linear_dim=args.linear_dim, attention_dim=args.attention_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
    ).to(device)

    optimizer_cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                              betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                              weight_decay=args.weight_decay)

    topk_list = [1, 3, 5, 10, 20]
    config_info = {
        'model': 'HSTU', 'kuairec_data_dir': portable_path(args.kuairec_data_dir),
        'max_len': args.max_len, 'hidden_dim': args.hidden_dim,
        'linear_dim': args.linear_dim, 'attention_dim': args.attention_dim,
        'num_blocks': args.num_blocks, 'num_heads': args.num_heads,
        'dropout': args.dropout, 'batch_size': args.batch_size,
        'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
        'selection_metric': args.selection_metric, 'early_stop_patience': args.early_stop_patience,
        'num_items': num_items, 'num_users': len(user_sequences),
    }
    log(f'Config: {json.dumps(config_info, ensure_ascii=False)}')
    log(f'Model stats: {model_stats(model)}')

    best_val = -1.0
    best_state = None
    best_epoch = None
    no_improve_evals = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        for step, batch in enumerate(loader, start=1):
            seq, target = [x.to(device, non_blocking=True) for x in batch]
            hidden = F.normalize(model(seq)[:, -1, :], p=2, dim=-1, eps=1e-6)
            loss = sampled_softmax_loss(hidden, target - 1, model.output_weight(),
                                        int(args.num_negatives), float(args.temperature))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            step_count += 1
            if step % args.log_every == 0:
                log(f'  epoch={epoch} step={step} loss={loss.item():.4f}')

        val_recalls, val_ndcgs = None, None
        should_stop = False
        if epoch % max(1, int(args.eval_every)) == 0 or epoch == args.epochs:
            val_recalls, val_ndcgs = evaluate_exact(
                model=model, splits=splits, users=eval_users, max_len=args.max_len,
                eval_batch_size=args.eval_batch_size, device=device, mode='val', topk_list=topk_list)
            val_metrics = {**val_recalls, **val_ndcgs}
            val_score = val_metrics[args.selection_metric]
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # Save best checkpoint (overwrite, only keep the best)
                torch.save(best_state, args.save_path)
                log(f'  >> New best {args.selection_metric}={val_score:.4f}, saved to {args.save_path}')
                no_improve_evals = 0
            else:
                no_improve_evals += 1
                if int(args.early_stop_patience) > 0 and no_improve_evals >= int(args.early_stop_patience):
                    should_stop = True

        epoch_sec = time.time() - epoch_start
        val_str = ''
        if val_recalls is not None:
            val_str = f' | val ' + ' '.join(f'{k}={v:.4f}' for k, v in {**val_recalls, **val_ndcgs}.items())
        log(f'epoch={epoch}/{args.epochs} loss={epoch_loss/max(step_count,1):.4f} time={epoch_sec:.1f}s{val_str}')

        if should_stop:
            log(f'Early stopping at epoch {epoch} (patience={args.early_stop_patience})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_recalls, test_ndcgs = evaluate_exact(
        model=model, splits=splits, users=eval_users, max_len=args.max_len,
        eval_batch_size=args.eval_batch_size, device=device, mode='test', topk_list=topk_list)
    total_sec = time.time() - training_start

    log(f'=== Results ===')
    log(f'best_epoch={best_epoch}')
    log(f'test: ' + ' '.join(f'{k}={v:.4f}' for k, v in {**test_recalls, **test_ndcgs}.items()))
    log(f'total_time={total_sec:.1f}s')

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_summary = {
            'config': config_info,
            'model_stats': model_stats(model),
            'best_epoch': best_epoch,
            'test_metrics': {**test_recalls, **test_ndcgs},
            'total_train_eval_sec': total_sec,
        }
        output_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'Saved JSON to {output_path}')

    log_file.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
