from __future__ import annotations

import torch
import torch.nn as nn


class RegimeNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_regimes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_regimes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

