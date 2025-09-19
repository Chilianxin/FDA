from __future__ import annotations

import torch
import torch.nn as nn


class DilatedTCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, kernel: int = 3):
        super().__init__()
        mods = []
        d = 1
        c = in_dim
        for _ in range(layers):
            mods += [
                nn.Conv1d(c, hidden, kernel, padding=d*(kernel-1), dilation=d),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 1),
                nn.GELU(),
            ]
            c = hidden
            d *= 2
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.net(x).mean(dim=-1)


class GatingNet(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden: int = 64, temperature: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, num_experts)
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x) / self.temperature
        return torch.softmax(logits, dim=-1)


class TCNMoE(nn.Module):
    def __init__(self, in_feat: int, cond_dim: int, num_experts: int = 3, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.experts = nn.ModuleList([
            DilatedTCN(in_feat, hidden, layers),
            DilatedTCN(in_feat, hidden, layers),
            DilatedTCN(in_feat, hidden, layers),
        ])
        self.gate = GatingNet(cond_dim, num_experts)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x_seq: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, C, T]; cond: [B, D]
        expert_outs = [e(x_seq) for e in self.experts]  # each [B, H]
        H = torch.stack(expert_outs, dim=1)  # [B, E, H]
        gate = self.gate(cond).unsqueeze(-1)  # [B, E, 1]
        h = (H * gate).sum(dim=1)  # [B, H]
        return self.out(h)

