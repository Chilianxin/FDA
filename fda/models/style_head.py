from __future__ import annotations

import torch
import torch.nn as nn


class StyleHead(nn.Module):
    def __init__(self, in_dim: int, num_styles: int = 8, use_anchor: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_styles, bias=False)
        self.use_anchor = use_anchor
        # register anchor indices: 0-momentum,1-value,2-volatility,3-liquidity
        self.num_styles = num_styles

    def forward(self, h_inter: torch.Tensor) -> torch.Tensor:
        # E: [N, K]
        E = self.linear(h_inter)
        return E

    @staticmethod
    def orthogonality_loss(W: torch.Tensor) -> torch.Tensor:
        # W shape: [K, D] or [D, K]? Our linear has weight [K, D]
        M = W @ W.t()
        I = torch.eye(M.size(0), device=M.device)
        return ((M - I) ** 2).mean()

    @staticmethod
    def sparsity_loss(E: torch.Tensor) -> torch.Tensor:
        return E.abs().mean()

