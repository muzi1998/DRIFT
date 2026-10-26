"""
train_stage0_ml20m.py — Stage 0 训练入口 (ML20M 版)
========================================================================
从 ratings.csv + movies.csv 加载数据，训练 Stage0Model，导出 d_u_init。

输出:
  ml_drift_cache/stage0_state.pt
  ml_drift_cache/d_u_init_{train,valid,test}.parquet

运行:
  python train_stage0_ml20m.py \
    --ratings_csv data/ml-20m/ratings.csv \
    --movies_csv data/ml-20m/movies.csv \
    --max_len 200 --hidden_dim 96 --recent_r 10 --epochs 8 --device cuda:0
"""
from __future__ import annotations

import argparse, csv, json, os, time
from pathlib import Path
from hstu_paths import resolve_args, resolve_path, portable_path, portable_config
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage0_model import Stage0Model
from stage0_losses import total_stage0_loss
from stage0_common import GlassSplit


def load_ml20m_sequences(ratings_csv: str) -> Tuple[Dict[int, List[int]], int]:
    """从 ratings.csv 加载 ML20M 序列，1-based item ID。"""
    df = pd.read_csv(ratings_csv, usecols=['userId', 'movieId', 'timestamp'])
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    unique_movies = sorted(df['movieId'].unique())
    movie_map = {m: i + 1 for i, m in enumerate(unique_movies)}
    sequences = {}
    for uid, group in df.groupby('userId'):
        sequences[int(uid)] = [movie_map[m] for m in group['movieId'].values]
    return sequences, len(unique_movies)


def load_ml20m_categories(movies_csv: str, num_items: int, movie_map: Dict[int, int]) -> Tuple[torch.Tensor, int]:
    """从 movies.csv 构建 item_id → category_id 映射。
    genre 的第一个作为 category（与 KuaiRec 的 first_level_category 一致）。
    """
    movie_genres = {}
    all_genres = set()
    with open(movies_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row['movieId'])
            genres_str = row['genres']
            if genres_str == '(no genres listed)':
                movie_genres[mid] = []
            else:
                genres = genres_str.split('|')
                movie_genres[mid] = genres
                all_genres.update(genres)

    genre_list = sorted(all_genres)
    genre_map = {g: i for i, g in enumerate(genre_list)}
    num_categories = len(genre_list)
    print(f"  genres: {num_categories} ({genre_list})")

    # 构建 category_map: (num_items + 1,) long, category_map[0]=0 (padding)
    category_map = torch.zeros(num_items + 1, dtype=torch.long)
    for orig_mid, mapped_id in movie_map.items():
        genres = movie_genres.get(orig_mid, [])
        if genres:
            category_map[mapped_id] = genre_map[genres[0]]
        else:
            category_map[mapped_id] = 0

    return category_map, num_categories


def build_glass_aligned_splits(sequences: Dict[int, List[int]], max_len: int, min_seq_len: int = 5) -> Dict[int, GlassSplit]:
    """构建 leave-one-out splits (同 stage0_common)."""
    splits = {}
    for user, seq in sequences.items():
        if len(seq) < min_seq_len:
            continue
        history = seq[:-2]
        splits[user] = GlassSplit(
            train_seq=history[:-1],
            train_target=history[-1],
            val_history=history,
            val_target=seq[-2],
            test_target=seq[-1],
        )
    return splits


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 0 ML20M: pretrain d_u model")
    ap.add_argument('--ratings_csv', type=str, required=True)
    ap.add_argument('--movies_csv', type=str, required=True)
    ap.add_argument('--max_len', type=int, default=200)
    ap.add_argument('--hidden_dim', type=int, default=96)
    ap.add_argument('--recent_r', type=int, default=10)
    ap.add_argument('--secondary_r', type=int, default=15)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--cache_dir', type=str, default='outputs/hstu/ml_drift_cache')
    return resolve_args(ap.parse_args())


class Stage0Dataset(Dataset):
    def __init__(self, splits: dict, max_len: int):
        self.users = list(splits.keys())
        self.splits = splits
        self.max_len = max_len
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        user = self.users[idx]
        s = self.splits[user]
        seq = s.train_seq + [s.train_target]
        arr = np.zeros(self.max_len, dtype=np.int64)
        seq = seq[-self.max_len:]
        arr[self.max_len - len(seq):] = np.array(seq, dtype=np.int64)
        return torch.from_numpy(arr), torch.tensor(user, dtype=torch.long)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"[Stage0-ML20M] Loading data from {portable_path(args.ratings_csv)} ...")
    sequences, num_items = load_ml20m_sequences(args.ratings_csv)
    print(f"  num_users={len(sequences)}, num_items={num_items}")

    # 重建 movie_map (与 load_ml20m_sequences 一致)
    df_all = pd.read_csv(args.ratings_csv, usecols=['movieId'])
    unique_movies = sorted(df_all['movieId'].unique())
    movie_map = {m: i + 1 for i, m in enumerate(unique_movies)}

    print(f"[Stage0-ML20M] Loading categories from {args.movies_csv} ...")
    category_map, num_categories = load_ml20m_categories(args.movies_csv, num_items, movie_map)

    print("[Stage0-ML20M] Building splits ...")
    splits = build_glass_aligned_splits(sequences, args.max_len)
    print(f"  splits: {len(splits)} users")

    dataset = Stage0Dataset(splits, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Stage0Model(
        num_items=num_items, d_model=args.hidden_dim,
        num_categories=num_categories,
        recent_window_r=args.recent_r, secondary_r=args.secondary_r,
    ).to(device)
    model.set_category_map(category_map.to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage0-ML20M] trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        n_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for item_ids, _user_ids in pbar:
            item_ids = item_ids.to(device)
            d_main, psi5_main, psi5_pred = model(item_ids, recent_window_r=args.recent_r)
            d_aux, _, _ = model(item_ids, recent_window_r=args.secondary_r)
            loss_dict = total_stage0_loss(d_main, d_aux, psi5_main, psi5_pred)
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sum_loss += float(loss.item())
            n_steps += 1
            if n_steps % args.log_every == 0:
                d_mu = float(d_main.mean().item())
                d_sd = float(d_main.std(unbiased=False).item()) if d_main.numel() > 1 else 0.0
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", d_mu=f"{d_mu:.3f}", d_sd=f"{d_sd:.3f}")
        n = max(n_steps, 1)
        print(f"[Stage0-ML20M][Ep {epoch}/{args.epochs}] total={sum_loss/n:.4f}")

    # 导出 state
    state = model.export_state()
    state_path = os.path.join(args.cache_dir, "stage0_state.pt")
    torch.save(state, state_path)
    print(f"[Stage0-ML20M] saved state -> {state_path}")

    # 导出 d_u parquet
    print("[Stage0-ML20M] Exporting d_u_init parquet ...")
    model.eval()
    for split_name in ["train", "valid", "test"]:
        records = []
        for user in tqdm(splits.keys(), desc=f"  Export {split_name}"):
            s = splits[user]
            if split_name == "train":
                seq = s.train_seq + [s.train_target]
            else:
                seq = s.val_history
            arr = np.zeros(args.max_len, dtype=np.int64)
            seq_t = seq[-args.max_len:]
            arr[args.max_len - len(seq_t):] = np.array(seq_t, dtype=np.int64)
            item_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                d_u, psi5, _ = model(item_tensor, recent_window_r=args.recent_r)
            records.append({
                "user": int(user), "d_u_init": float(d_u.item()),
                "psi0_1": float(psi5[0, 0].item()), "psi0_2": float(psi5[0, 1].item()),
                "psi0_3": float(psi5[0, 2].item()), "psi0_4": float(psi5[0, 3].item()),
                "psi0_5": float(psi5[0, 4].item()),
            })
        df = pd.DataFrame(records)
        out_path = os.path.join(args.cache_dir, f"d_u_init_{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(df)} rows -> {out_path}")
        print(f"    d_u stats: mean={df['d_u_init'].mean():.4f}, std={df['d_u_init'].std():.4f}")

    print("[Stage0-ML20M] DONE")


if __name__ == "__main__":
    main()
