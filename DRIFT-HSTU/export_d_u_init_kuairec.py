"""
export_d_u_init_kuairec.py — 导出 d_u parquet
========================================================================
加载 stage0_state.pt，扫 train/valid/test splits，对每用户计算 d_u + ψ，
导出 parquet 供 Stage 1 查表注入。

输出:
  drift_cache/d_u_init_train.parquet  columns=[user, d_u_init, psi0_1..5]
  drift_cache/d_u_init_valid.parquet
  drift_cache/d_u_init_test.parquet
"""
from __future__ import annotations

import os
from pathlib import Path
from hstu_paths import resolve_args, resolve_path, portable_path, portable_config
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stage0_model import Stage0Model
from stage0_common import GlassSplit


def export_d_u_init(
    state_path: str,
    splits: Dict[int, GlassSplit],
    max_len: int,
    device: torch.device,
    cache_dir: str,
    recent_r: int = 10,
):
    """导出 d_u_init_{train,valid,test}.parquet"""
    state = torch.load(state_path, map_location=device)
    cfg = state["config"]
    model = Stage0Model(
        num_items=cfg["num_items"], d_model=cfg["d_model"],
        num_categories=cfg["num_categories"],
        recent_window_r=cfg["recent_window_r"],
        secondary_r=cfg["secondary_r"],
    ).to(device)
    model.load_export_state(state)
    model.eval()

    os.makedirs(cache_dir, exist_ok=True)

    for split_name in ["train", "valid", "test"]:
        records = []
        for user in tqdm(splits.keys(), desc=f"  Export {split_name}"):
            s = splits[user]
            if split_name == "train":
                seq = s.train_seq + [s.train_target]
            else:  # valid / test 用 val_history
                seq = s.val_history

            arr = np.zeros(max_len, dtype=np.int64)
            seq_trimmed = seq[-max_len:]
            arr[max_len - len(seq_trimmed):] = np.array(seq_trimmed, dtype=np.int64)
            item_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

            with torch.no_grad():
                d_u, psi5, _ = model(item_tensor, recent_window_r=recent_r)

            records.append({
                "user": int(user),
                "d_u_init": float(d_u.item()),
                "psi0_1": float(psi5[0, 0].item()),
                "psi0_2": float(psi5[0, 1].item()),
                "psi0_3": float(psi5[0, 2].item()),
                "psi0_4": float(psi5[0, 3].item()),
                "psi0_5": float(psi5[0, 4].item()),
            })

        df = pd.DataFrame(records)
        out_path = os.path.join(cache_dir, f"d_u_init_{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    import argparse
    from stage0_common import load_kuairec_sequences, build_glass_aligned_splits

    ap = argparse.ArgumentParser()
    ap.add_argument("--state_path", type=str, required=True)
    ap.add_argument("--kuairec_data_dir", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--recent_r", type=int, default=10)
    ap.add_argument("--cache_dir", type=str, default="outputs/hstu/drift_cache")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = resolve_args(ap.parse_args())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sequences, _ = load_kuairec_sequences(args.kuairec_data_dir)
    splits = build_glass_aligned_splits(sequences, args.max_len)
    export_d_u_init(args.state_path, splits, args.max_len, device, args.cache_dir, args.recent_r)
