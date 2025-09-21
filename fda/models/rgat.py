from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, adj_indices: torch.Tensor, adj_weights: torch.Tensor):
        # x: [N, D]; adj_indices: [E, 2]; adj_weights: [E]
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim
        h = self.W(x).view(N, H, D)  # [N, H, D]
        src = adj_indices[:, 0]
        dst = adj_indices[:, 1]
        hs = h[src]  # [E, H, D]
        hd = h[dst]
        e = (hs * self.a_src).sum(-1) + (hd * self.a_dst).sum(-1)  # [E, H]
        e = torch.tanh(e) * adj_weights.unsqueeze(-1)
        # softmax per destination node
        alpha = torch.zeros_like(e)
        for i in range(N):
            mask = (dst == i)
            if mask.any():
                logits = e[mask]
                alpha[mask] = torch.softmax(logits, dim=0)
        # message passing
        m = alpha.unsqueeze(-1) * hs  # [E, H, D]
        out = torch.zeros(N, H, D, device=x.device)
        out.index_add_(0, dst, m)  # aggregate to dst
        out = out.view(N, H * D)
        return self.act(self.dropout(out))


class StackedRGAT(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods.append(SimpleRGATLayer(d, hid, num_heads=heads, dropout=dropout))
            d = hid * heads
        self.layers = nn.ModuleList(mods)
        self.out_dim = d

    def forward(self, x: torch.Tensor, adj_indices: torch.Tensor, adj_weights: torch.Tensor):
        h = x
        for layer in self.layers:
            h = layer(h, adj_indices, adj_weights)
        return h

