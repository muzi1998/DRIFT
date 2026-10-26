"""
stage0_common.py — 共用工具函数
========================================================================
数据加载、类别映射、mask 工具等，供 Stage0 / Stage1 共用。
"""
from __future__ import annotations

import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ==================== Data Loading ====================

def load_kuairec_sequences(data_dir: str) -> Tuple[Dict[int, List[int]], int]:
    """读取 KuaiRec 原始 parquet，构建 user → [video_id] 序列。
    合并 train + valid + test，按 (user_id, timestamp) 排序。
    video_id 重索引为 1-based（0 = padding）。
    """
    dfs = []
    for name in ['train', 'valid', 'test']:
        path = pd.io.common.os.path.join(data_dir, f'{name}.parquet')
        import os
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    unique_videos = sorted(df['video_id'].unique())
    vid_map = {v: i + 1 for i, v in enumerate(unique_videos)}

    sequences: Dict[int, List[int]] = {}
    for uid, group in df.groupby('user_id'):
        sequences[int(uid)] = [vid_map[v] for v in group['video_id'].values]
    return sequences, len(unique_videos)


def load_item_categories(csv_path: str, num_items: int,
                         vid_map: Dict[int, int] = None) -> torch.Tensor:
    """读取 kuairec_caption_category.csv，构建 item_id → category_id 映射。

    Args:
        csv_path: kuairec_caption_category.csv 路径
        num_items: 物品总数（1-based，0=padding）
        vid_map: 原始 video_id → 重索引 item_id 的映射（可选，若 None 则直接用 video_id）

    Returns:
        category_map: (num_items + 1,) long tensor, category_map[0]=0 (padding)
    """
    video_to_cat: Dict[int, int] = {}
    cat_set: set = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_vid = row.get('video_id', '')
            raw_cat = row.get('first_level_category_id', '')
            if not raw_vid or not raw_cat:
                continue
            try:
                vid = int(raw_vid)
                cat = int(raw_cat)
            except (ValueError, TypeError):
                continue
            video_to_cat[vid] = cat
            cat_set.add(cat)

    # 重映射 category 到 0-based 连续
    cat_list = sorted(cat_set)
    cat_remap = {c: i for i, c in enumerate(cat_list)}
    num_categories = len(cat_list)

    category_map = torch.zeros(num_items + 1, dtype=torch.long)
    for vid, cat in video_to_cat.items():
        if vid_map is not None:
            item_id = vid_map.get(vid, 0)
        else:
            item_id = vid
        if 0 < item_id <= num_items:
            category_map[item_id] = cat_remap[cat]

    return category_map, num_categories


# ==================== Mask Utilities ====================

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x: (B, L, D), mask: (B, L) bool → (B, D)"""
    m = mask.float().unsqueeze(-1)
    s = (x * m).sum(dim=1)
    n = m.sum(dim=1).clamp_min(eps)
    return s / n


def masked_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x: (B, L, D), mask: (B, L) bool → (B,) std over L, averaged over D"""
    m = mask.float().unsqueeze(-1)  # (B, L, 1)
    n = m.sum(dim=1).clamp_min(1.0)  # (B, 1)
    mean = (x * m).sum(dim=1) / n    # (B, D)
    var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / n  # (B, D)
    return var.mean(dim=-1).clamp_min(eps * eps).sqrt()


def local_change_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """mean_t (1 - cos(x_t, x_{t-1})), 仅相邻有效位置计入。返回 (B,)."""
    if x.size(1) < 2:
        return x.new_zeros(x.size(0))
    cos = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1, eps=eps)
    m_pair = (mask[:, 1:] & mask[:, :-1]).float()
    change = (1.0 - cos) * m_pair
    denom = m_pair.sum(dim=1).clamp_min(1.0)
    return (change.sum(dim=1) / denom).clamp(0.0, 2.0)


# ==================== GLASS-aligned Split ====================

from dataclasses import dataclass


@dataclass
class GlassSplit:
    train_seq: list
    train_target: int
    val_history: list
    val_target: int
    test_target: int


def build_glass_aligned_splits(sequences: Dict[int, List[int]],
                                max_len: int, min_seq_len: int = 5) -> Dict[int, GlassSplit]:
    """GLASS-aligned leave-one-out:
      history_all = seq[:-2], history = history_all[-max_len:]
      train: history[:-1] → predict history[-1]
      val:   history → predict seq[-2]
      test:  history → predict seq[-1]  (same history as val)
    """
    splits: Dict[int, GlassSplit] = {}
    for user, seq in sequences.items():
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
        )
    return splits
