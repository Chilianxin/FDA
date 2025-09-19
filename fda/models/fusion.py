from __future__ import annotations

import torch
import torch.nn as nn


class CrossGating(nn.Module):
    def __init__(self, inter_dim: int, intra_dim: int, hidden: int = 64):
        super().__init__()
        self.g_inter_to_intra = nn.Sequential(nn.Linear(inter_dim, hidden), nn.GELU(), nn.Linear(hidden, intra_dim))
        self.g_intra_to_inter = nn.Sequential(nn.Linear(intra_dim, hidden), nn.GELU(), nn.Linear(hidden, inter_dim))

    def forward(self, h_inter: torch.Tensor, h_intra: torch.Tensor):
        # elementwise gates
        g_i2a = torch.sigmoid(self.g_inter_to_intra(h_inter))
        g_a2i = torch.sigmoid(self.g_intra_to_inter(h_intra))
        h_intra_mod = h_intra * g_i2a
        h_inter_mod = h_inter * g_a2i
        return h_inter_mod, h_intra_mod


class Fusion(nn.Module):
    def __init__(self, inter_dim: int, intra_dim: int, out_dim: int):
        super().__init__()
        self.cross = CrossGating(inter_dim, intra_dim)
        self.proj = nn.Linear(inter_dim + intra_dim, out_dim)

    def forward(self, h_inter: torch.Tensor, h_intra: torch.Tensor):
        hi, ha = self.cross(h_inter, h_intra)
        h = torch.cat([hi, ha], dim=-1)
        return self.proj(h)

