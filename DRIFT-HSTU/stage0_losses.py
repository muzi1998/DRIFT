"""
stage0_losses.py — 5 项自监督损失（与 GLASS losses.py 对齐）
========================================================================
L_align  = MSE(d_u, sigmoid(mean(z_score(ψ))).detach())
L_smooth = MSE(d_u(r), d_u(r'))
L_spread = max(0, τ_std - Std(d_u))²
L_robust = batch 内 hinge 排序
L_recon  = MSE(Dec(d_u), ψ.detach())

L_stage0 = α·L_align + β·L_smooth + γ·L_spread + δ·L_robust + ε·L_recon
默认: α=1.0, β=1.0, γ=0.1, δ=1.0, ε=0.02
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as F


def _z_score_batch(psi5: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """psi5: (B, 5) → z-score 后的 ψ̃"""
    mu = psi5.mean(dim=0, keepdim=True)
    sigma = psi5.std(dim=0, keepdim=True).clamp_min(eps)
    return (psi5 - mu) / sigma


def loss_align(d_u: torch.Tensor, psi5: torch.Tensor) -> torch.Tensor:
    s_tilde = torch.sigmoid(_z_score_batch(psi5).mean(dim=-1)).detach()
    return F.mse_loss(d_u, s_tilde)


def loss_smooth(d_u_main: torch.Tensor, d_u_aux: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(d_u_main, d_u_aux)


def loss_spread(d_u: torch.Tensor, std_target: float = 0.15) -> torch.Tensor:
    if d_u.numel() < 2:
        return d_u.new_zeros(())
    cur_std = d_u.std(unbiased=False)
    return torch.clamp(std_target - cur_std, min=0.0).pow(2)


def loss_robust(d_u: torch.Tensor, psi5: torch.Tensor,
                margin: float = 0.05) -> torch.Tensor:
    B = d_u.size(0)
    if B < 2:
        return d_u.new_zeros(())
    s = _z_score_batch(psi5).mean(dim=-1)
    idx = torch.randperm(B, device=d_u.device)
    sign = torch.sign(s - s[idx])
    diff = sign * (d_u - d_u[idx])
    return F.relu(margin - diff).mean()


def loss_recon(psi5_pred: torch.Tensor, psi5_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(psi5_pred, psi5_target.detach())


def total_stage0_loss(
    d_u_main: torch.Tensor,
    d_u_aux: torch.Tensor,
    psi5_main: torch.Tensor,
    psi5_pred: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    delta: float = 1.0,
    eps_w: float = 0.02,
    margin: float = 0.05,
    std_target: float = 0.15,
) -> Dict[str, torch.Tensor]:
    L_a = loss_align(d_u_main, psi5_main)
    L_sm = loss_smooth(d_u_main, d_u_aux)
    L_sp = loss_spread(d_u_main, std_target=std_target)
    L_r = loss_robust(d_u_main, psi5_main, margin=margin)
    L_rc = loss_recon(psi5_pred, psi5_main)
    total = alpha * L_a + beta * L_sm + gamma * L_sp + delta * L_r + eps_w * L_rc
    return {"total": total, "align": L_a.detach(), "smooth": L_sm.detach(),
            "spread": L_sp.detach(), "robust": L_r.detach(), "recon": L_rc.detach()}
