"""
stage0_psi_features.py — ψ 特征计算（无语义 ID 版）
========================================================================
5 维 ψ 特征，ψ₂ 用真实物品类别分布 JSD（替代 GLASS 的一级 SID 分布）。

ψ₁ = 1 - cos(p_r, p_h)                    近期/历史均值向量偏移
ψ₂ = JSD(P_R_category, P_H_category)      物品类别分布偏移
ψ₃ = Std(W_x·x)|_R - Std(W_x·x)|_H       投影后波动差异
ψ₄ = 0 (无时间戳) 或 log(1+Δt_R) - log(1+Δt_H)
ψ₅ = mean(1-cos(x_t,x_{t-1}))|_R - ...|_H 局部变化率差异
"""
from __future__ import annotations
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage0_common import masked_mean, masked_std, local_change_mean


def _jsd(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """JSD over last dim. P, Q: (B, C) → (B,)"""
    M = 0.5 * (P + Q)
    kl_pm = (P * (P.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(dim=-1)
    kl_qm = (Q * (Q.clamp_min(eps).log() - M.clamp_min(eps).log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _smooth_category_hist(category_ids: torch.Tensor, mask: torch.Tensor,
                          num_categories: int, eps: float = 1e-3) -> torch.Tensor:
    """平滑经验分布。category_ids: (B, L), mask: (B, L) → (B, num_categories)"""
    safe = category_ids.clamp(min=0, max=num_categories - 1)
    oh = F.one_hot(safe, num_classes=num_categories).float()
    oh = oh * mask.float().unsqueeze(-1)
    counts = oh.sum(dim=1)
    denom = mask.float().sum(dim=1, keepdim=True) + eps * num_categories
    return (counts + eps) / denom


def compute_psi5(
    x_full: torch.Tensor,            # (B, L, D) item embeddings
    m_full: torch.Tensor,            # (B, L) bool mask
    cat_full: torch.Tensor,          # (B, L) long category IDs
    recent_window_r: int,
    num_categories: int,
    proj_var: nn.Linear,             # W_x 投影矩阵
    eps: float = 1e-8,
    jsd_smooth_eps: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算 5 维 R_u vs H_u 对比特征 ψ。

    Returns:
        psi5: (B, 5)
        meta: dict (调试用)
    """
    B, L, D = x_full.shape
    r = min(int(recent_window_r), L)

    # 拆分 R_u / H_u
    x_r, m_r = x_full[:, -r:], m_full[:, -r:]
    x_h, m_h = x_full[:, :-r], m_full[:, :-r]
    cat_r = cat_full[:, -r:]
    cat_h = cat_full[:, :-r]

    # ψ₁: 1 - cos(mean_R, mean_H)
    p_r = masked_mean(x_r, m_r, eps)
    p_h = masked_mean(x_h, m_h, eps)
    psi_1 = (1.0 - F.cosine_similarity(p_r, p_h, dim=-1, eps=eps)).clamp(0.0, 2.0)

    # ψ₂: JSD(P_R_category, P_H_category)
    P_R = _smooth_category_hist(cat_r, m_r, num_categories, jsd_smooth_eps)
    P_H = _smooth_category_hist(cat_h, m_h, num_categories, jsd_smooth_eps)
    psi_2 = _jsd(P_R, P_H)

    # ψ₃: Std(W_x·x)|_R - Std(W_x·x)|_H
    proj = proj_var(x_full)
    std_R = masked_std(proj[:, -r:], m_r, eps)
    std_H = masked_std(proj[:, :-r], m_h, eps)
    psi_3 = std_R - std_H

    # ψ₄: 无时间戳 → 0
    psi_4 = x_full.new_zeros(B)

    # ψ₅: 局部变化率差异
    chg_R = local_change_mean(x_r, m_r, eps)
    chg_H = local_change_mean(x_h, m_h, eps)
    psi_5 = chg_R - chg_H

    psi5 = torch.stack([psi_1, psi_2, psi_3, psi_4, psi_5], dim=-1)

    meta = {"p_r": p_r, "p_h": p_h, "P_R": P_R, "P_H": P_H,
            "std_R": std_R, "std_H": std_H}
    return psi5, meta
