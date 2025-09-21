from __future__ import annotations

import torch
import torch.nn as nn

from .tcn_moe import TCNMoE
from .rgat import StackedRGAT
from .fusion import Fusion
from .style_head import StyleHead


class Predictor(nn.Module):
    def __init__(self, seq_in_feat: int, cond_dim: int, node_feat_dim: int, hidden: int = 128, num_styles: int = 8):
        super().__init__()
        # intra-stock encoder (sequence)
        self.intra = TCNMoE(in_feat=seq_in_feat, cond_dim=cond_dim, hidden=hidden)
        # inter-stock encoder (graph)
        self.rgat = StackedRGAT(in_dim=node_feat_dim, hid=hidden//4, heads=4, layers=2)
        self.fusion = Fusion(inter_dim=self.rgat.out_dim, intra_dim=hidden, out_dim=hidden)
        self.mu = nn.Linear(hidden, 1)
        self.q_heads = nn.Linear(hidden, 3)  # q10, q50, q90
        self.uncertainty = nn.Linear(hidden, 1)
        self.style = StyleHead(in_dim=self.rgat.out_dim, num_styles=num_styles)

    def forward(self, x_seq: torch.Tensor, x_node: torch.Tensor, cond: torch.Tensor, adj_indices: torch.Tensor, adj_weights: torch.Tensor):
        # x_seq: [B*N, C, T] per node sequences batched; x_node: [N, F]; cond: [B*N, D]
        # For simplicity, assume B=1 in this minimal version
        h_intra = self.intra(x_seq, cond)              # [N, H]
        h_inter = self.rgat(x_node, adj_indices, adj_weights)  # [N, H_g]
        E = self.style(h_inter)                        # [N, K]
        h = self.fusion(h_inter, h_intra)              # [N, H]
        mu = self.mu(h).squeeze(-1)
        qs = self.q_heads(h)                           # [N, 3]
        u = torch.softplus(self.uncertainty(h)).squeeze(-1)
        return { 'mu': mu, 'quantiles': qs, 'uncertainty': u, 'styles': E, 'h_inter': h_inter, 'h_intra': h_intra, 'h': h }

