from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import Fusion
from .rgat import StackedRGAT
from .sat_fan import SATFAN
from .style_head import StyleHead


class Predictor(nn.Module):
    """
    Dual-stream Alpha predictor.

    Args:
        seq_in_feat: feature dimension per timestep for SAT-FAN (price/volume features after stacking)
        cond_dim: optional conditioning dimension per stock (placeholder for future extensions)
        node_feat_dim: per-node static/graph features consumed by RGAT
    """

    def __init__(self, seq_in_feat: int, cond_dim: int, node_feat_dim: int, hidden: int = 128, num_styles: int = 8):
        super().__init__()
        self.seq_in_feat = seq_in_feat
        self.cond_dim = cond_dim
        self.intra = SATFAN(in_feat=seq_in_feat, hidden=hidden)
        self.rgat = StackedRGAT(in_dim=node_feat_dim, hid=hidden // 4, heads=4, layers=2)
        self.fusion = Fusion(inter_dim=self.rgat.out_dim, intra_dim=hidden, out_dim=hidden)
        self.mu = nn.Linear(hidden, 1)
        self.q_heads = nn.Linear(hidden, 3)
        self.uncertainty = nn.Linear(hidden, 1)
        self.style = StyleHead(in_dim=self.rgat.out_dim, num_styles=num_styles)

    def _reshape_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Accept [B, N, T, F] or [N, T, F] and reshape to SAT-FAN format [B*N, F, T].
        """
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
        assert x_seq.dim() == 4, f"x_seq must be [B,N,T,F], got {x_seq.shape}"
        B, N, T, F = x_seq.shape
        x = x_seq.reshape(B * N, T, F).permute(0, 2, 1).contiguous()
        return x, B, N

    @staticmethod
    def _prepare_node_features(x_node: torch.Tensor, B: int) -> torch.Tensor:
        """
        Ensure node features follow [B, N, F].
        """
        if x_node.dim() == 2:
            x_node = x_node.unsqueeze(0).expand(B, -1, -1)
        return x_node

    @staticmethod
    def _select_adj(adj: torch.Tensor, b: int) -> torch.Tensor:
        """
        Allow either shared adjacency [E,2] or per-batch [B,E,2].
        """
        if adj.dim() == 3:
            return adj[b]
        return adj

    def forward(
        self,
        x_seq: torch.Tensor,
        x_node: torch.Tensor,
        cond: torch.Tensor | None,
        adj_indices: torch.Tensor,
        adj_weights: torch.Tensor,
    ) -> dict:
        """
        Args:
            x_seq: [B, N, T, F] time-series tensor
            x_node: [B, N, node_feat_dim] or [N, node_feat_dim]
            cond: optional conditioning tensor [B, N, cond_dim] (kept for API parity)
            adj_indices/weights: graph structure (shared or per-batch)
        """
        seq_flat, B, N = self._reshape_seq(x_seq)
        if cond is not None and cond.dim() == 3:
            cond_flat = cond.reshape(B * N, cond.size(-1))
        else:
            cond_flat = None
        h_intra = self.intra(seq_flat, cond_flat)  # [B*N, H]
        h_intra = h_intra.view(B, N, -1)

        x_node = self._prepare_node_features(x_node, B)
        h_inter_list = []
        for b in range(B):
            adj_idx_b = self._select_adj(adj_indices, b)
            adj_w_b = self._select_adj(adj_weights, b)
            h_inter_b = self.rgat(x_node[b], adj_idx_b, adj_w_b)
            h_inter_list.append(h_inter_b)
        h_inter = torch.stack(h_inter_list, dim=0)  # [B, N, Hg]

        styles = self.style(h_inter)
        fused = self.fusion(h_inter, h_intra)
        mu = self.mu(fused).squeeze(-1)
        qs = self.q_heads(fused)
        sigma = F.softplus(self.uncertainty(fused)).squeeze(-1)

        rl_state = {
            'alpha_mu': mu,
            'alpha_q': qs,
            'alpha_uncertainty': sigma,
            'styles': styles,
            'h_intra': h_intra,
            'h_inter': h_inter,
        }
        return {
            'mu': mu,
            'quantiles': qs,
            'uncertainty': sigma,
            'styles': styles,
            'h_inter': h_inter,
            'h_intra': h_intra,
            'h': fused,
            'rl_state': rl_state,
        }

