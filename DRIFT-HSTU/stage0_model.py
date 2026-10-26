"""
stage0_model.py — Stage 0 模型定义
========================================================================
Stage0Model: 预训练用户偏移度量 d_u（无语义 ID 版）

结构:
  - item_embedding: nn.Embedding(num_items+1, d_model)
  - proj_var (W_x): nn.Linear(d_model, d_model, bias=False)  — ψ₃ 投影
  - mlp_d0: Linear(5,64)→ReLU→Linear(64,1)                  — d_u = sigmoid(MLP_d(ψ))
  - decoder: Linear(1,32)→ReLU→Linear(32,5)                 — L_recon 用
  - category_map: (num_items+1,) long buffer                 — item_id → category_id
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from stage0_psi_features import compute_psi5


class Stage0Model(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int = 96,
        num_categories: int = 41,
        recent_window_r: int = 10,
        secondary_r: int = 15,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.num_categories = num_categories
        self.recent_window_r = int(recent_window_r)
        self.secondary_r = int(secondary_r)
        self.pad_token_id = pad_token_id

        # 物品 embedding
        self.shared_emb = nn.Embedding(num_items + 1, d_model, padding_idx=pad_token_id)
        nn.init.normal_(self.shared_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.shared_emb.weight[pad_token_id].zero_()

        # ψ₃ 投影矩阵
        self.proj_var = nn.Linear(d_model, d_model, bias=False)

        # MLP_d: ψ(5) → d_u(1)
        self.mlp_d0 = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # decoder (L_recon)
        self.decoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 5)
        )

        # item_id → category_id 映射（register_buffer）
        self.register_buffer("category_map", torch.zeros(num_items + 1, dtype=torch.long))

    def set_category_map(self, category_map: torch.Tensor):
        """加载 item_id → category_id 映射"""
        self.category_map.copy_(category_map.to(self.category_map.device))

    def forward(
        self,
        item_ids: torch.Tensor,          # (B, L) long, 0=padding
        recent_window_r: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向: 输出 d_u, psi5, psi5_pred.

        Returns:
            d_u: (B,) ∈ [0,1]
            psi5: (B, 5)
            psi5_pred: (B, 5) decoder(d_u)
        """
        r = int(recent_window_r) if recent_window_r is not None else self.recent_window_r
        mask = item_ids.ne(self.pad_token_id)  # (B, L)

        # ① 物品表示
        x = self.shared_emb(item_ids)  # (B, L, D)

        # ② category IDs
        cat = self.category_map[item_ids]  # (B, L)

        # ③ 计算 ψ (5 维)
        psi5, _ = compute_psi5(
            x_full=x, m_full=mask, cat_full=cat,
            recent_window_r=r, num_categories=self.num_categories,
            proj_var=self.proj_var,
        )

        # ④ d_u
        d_u = torch.sigmoid(self.mlp_d0(psi5).squeeze(-1))  # (B,)

        # ⑤ recon
        psi5_pred = self.decoder(d_u.unsqueeze(-1))  # (B, 5)

        return d_u, psi5, psi5_pred

    def export_state(self) -> Dict:
        return {
            "shared_emb": self.shared_emb.weight.detach().cpu().clone(),
            "proj_var": self.proj_var.state_dict(),
            "mlp_d0": self.mlp_d0.state_dict(),
            "decoder": self.decoder.state_dict(),
            "category_map": self.category_map.detach().cpu().clone(),
            "config": {
                "num_items": self.num_items,
                "d_model": self.d_model,
                "num_categories": self.num_categories,
                "recent_window_r": self.recent_window_r,
                "secondary_r": self.secondary_r,
                "pad_token_id": self.pad_token_id,
            },
        }

    def load_export_state(self, state: Dict):
        with torch.no_grad():
            self.shared_emb.weight.copy_(state["shared_emb"].to(self.shared_emb.weight.device))
        self.proj_var.load_state_dict(state["proj_var"])
        self.mlp_d0.load_state_dict(state["mlp_d0"])
        self.decoder.load_state_dict(state["decoder"])
        if "category_map" in state:
            self.category_map.copy_(state["category_map"].to(self.category_map.device))
