from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class StockModel(nn.Module):
    """
    Simple GAT-based model operating on per-timestep stock node features.

    Expects x: [num_nodes, in_dim], edge_index: [2, E], edge_weight: [E] (optional)
    Returns predictions per node: [num_nodes, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        heads: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.gat1 = GATv2Conv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout_p)
        self.act = nn.ELU()
        self.gat2 = GATv2Conv(in_channels=hidden_dim * heads, out_channels=hidden_dim, heads=1, dropout=dropout_p)
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.gat2(h, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        out = self.readout(h)
        return out

