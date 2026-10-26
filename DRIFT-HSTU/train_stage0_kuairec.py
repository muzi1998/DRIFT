"""
train_stage0_kuairec.py — Stage 0 训练入口
========================================================================
预训练用户偏移度量 d_u 模型，8 epoch，5 项自监督损失。

输出:
  drift_cache/stage0_state.pt
  drift_cache/d_u_init_{train,valid,test}.parquet  (auto export)

运行示例:
  python train_stage0_kuairec.py \
    --kuairec_data_dir data/kuairec_processed \
    --category_csv data/kuairec_caption_category.csv \
    --max_len 50 --hidden_dim 96 --recent_r 10 --epochs 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from hstu_paths import resolve_args, resolve_path, portable_path, portable_config

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage0_common import load_kuairec_sequences, load_item_categories, build_glass_aligned_splits, GlassSplit
from stage0_model import Stage0Model
from stage0_losses import total_stage0_loss


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 0: pretrain d_u model")
    ap.add_argument('--kuairec_data_dir', type=str, required=True)
    ap.add_argument('--category_csv', type=str, required=True,
                    help='Path to kuairec_caption_category.csv')
    ap.add_argument('--max_len', type=int, default=50)
    ap.add_argument('--hidden_dim', type=int, default=96)
    ap.add_argument('--recent_r', type=int, default=10, help='主窗口 r')
    ap.add_argument('--secondary_r', type=int, default=15, help='扰动窗口 r\'')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--cache_dir', type=str, default='outputs/hstu/drift_cache')
    return resolve_args(ap.parse_args())


class Stage0Dataset(Dataset):
    def __init__(self, splits: dict, max_len: int):
        self.users = list(splits.keys())
        self.splits = splits
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        s = self.splits[user]
        # 用 train_seq + train_target 作为完整历史
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

    print(f"[Stage0] Loading data from {portable_path(args.kuairec_data_dir)} ...")
    sequences, num_items = load_kuairec_sequences(args.kuairec_data_dir)
    print(f"  num_users={len(sequences)}, num_items={num_items}")

    # 加载类别映射
    print(f"[Stage0] Loading categories from {portable_path(args.category_csv)} ...")
    # 需要重建 vid_map（与 load_kuairec_sequences 一致）
    dfs = []
    for name in ['train', 'valid', 'test']:
        path = os.path.join(args.kuairec_data_dir, f'{name}.parquet')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
    df_all = pd.concat(dfs, ignore_index=True).sort_values(['user_id', 'timestamp'])
    unique_videos = sorted(df_all['video_id'].unique())
    vid_map = {v: i + 1 for i, v in enumerate(unique_videos)}

    category_map, num_categories = load_item_categories(args.category_csv, num_items, vid_map)
    print(f"  num_categories={num_categories}")

    # 构建 splits
    splits = build_glass_aligned_splits(sequences, args.max_len)
    dataset = Stage0Dataset(splits, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 模型
    model = Stage0Model(
        num_items=num_items, d_model=args.hidden_dim,
        num_categories=num_categories,
        recent_window_r=args.recent_r, secondary_r=args.secondary_r,
    ).to(device)
    model.set_category_map(category_map.to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage0] trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_align = sum_smooth = sum_spread = sum_robust = sum_recon = 0.0
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
            sum_align += float(loss_dict["align"].item())
            sum_smooth += float(loss_dict["smooth"].item())
            sum_spread += float(loss_dict["spread"].item())
            sum_robust += float(loss_dict["robust"].item())
            sum_recon += float(loss_dict["recon"].item())
            n_steps += 1

            if n_steps % args.log_every == 0:
                d_mu = float(d_main.mean().item())
                d_sd = float(d_main.std(unbiased=False).item()) if d_main.numel() > 1 else 0.0
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", d_mu=f"{d_mu:.3f}", d_sd=f"{d_sd:.3f}")

        n = max(n_steps, 1)
        print(f"[Stage0][Ep {epoch}/{args.epochs}] "
              f"total={sum_loss/n:.4f} align={sum_align/n:.4f} smooth={sum_smooth/n:.4f} "
              f"spread={sum_spread/n:.4f} robust={sum_robust/n:.4f} recon={sum_recon/n:.4f}")

    # 导出 state
    state = model.export_state()
    state_path = os.path.join(args.cache_dir, "stage0_state.pt")
    torch.save(state, state_path)
    print(f"[Stage0] saved state -> {state_path}")

    # 自动导出 d_u parquet
    print("[Stage0] Exporting d_u_init parquet ...")
    from export_d_u_init_kuairec import export_d_u_init
    export_d_u_init(
        state_path=state_path,
        splits=splits,
        max_len=args.max_len,
        device=device,
        cache_dir=args.cache_dir,
    )
    print("[Stage0] DONE")


if __name__ == "__main__":
    main()
