from __future__ import annotations

import torch
import torch.nn as nn


class ImpactNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.kappa = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        self.alpha = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, feat: torch.Tensor):
        kappa = self.kappa(feat).squeeze(-1)
        alpha = 0.6 + 0.4 * self.alpha(feat).squeeze(-1)  # map to [0.6,1.0]
        beta = self.beta(feat).squeeze(-1)
        return kappa, alpha, beta


def impact_cost(q: torch.Tensor, sigma: torch.Tensor, ADV: torch.Tensor, kappa: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # q, sigma, ADV, kappa, alpha, beta : [N]
    adv_ratio = (q.abs() / (ADV + 1e-8)).clamp(min=0.0)
    temp = kappa * sigma * (adv_ratio ** alpha)
    perm = beta * q.abs()
    return temp + perm

