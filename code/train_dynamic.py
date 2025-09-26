from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from code.graph_utils import moralize_graph, to_edge_index
from code.ngc_builder import NGCConfig, build_dynamic_graphs
from code.stock_model import StockModel


class TimeWindowDataset(Dataset):
    """
    A minimal dataset producing (timestamp, x, y) per step.

    x: node features [N, F]
    y: targets [N, 1]
    Assumes features_df and targets_df share the same DatetimeIndex and columns (stocks).
    """

    def __init__(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> None:
        assert isinstance(features_df.index, pd.DatetimeIndex)
        assert features_df.index.equals(targets_df.index)
        assert list(features_df.columns) == list(targets_df.columns)
        self.features_df = features_df
        self.targets_df = targets_df
        self.timestamps = features_df.index.to_list()

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int):
        ts = self.timestamps[idx]
        x = torch.tensor(self.features_df.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.targets_df.iloc[idx].values, dtype=torch.float32)
        # shape to [N, F] and [N, 1]
        x = x.view(-1, 1)  # if single feature per node
        y = y.view(-1, 1)
        return ts, x, y


def select_graph_for_timestamp(graphs: Dict[pd.Timestamp, np.ndarray], ts: pd.Timestamp) -> np.ndarray:
    """
    Select the latest graph whose key <= ts.
    """
    keys = sorted(graphs.keys())
    chosen = None
    for k in keys:
        if k <= ts:
            chosen = k
        else:
            break
    if chosen is None:
        # fallback to earliest
        return graphs[keys[0]]
    return graphs[chosen]


def train(
    prices: pd.DataFrame,
    window_size: int = 252,
    step_size: int = 21,
    ngc_cfg: NGCConfig = NGCConfig(),
    hidden_dim: int = 64,
    heads: int = 4,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    prices = prices.sort_index()
    # Build dynamic causal graphs using close prices (or returns)
    graphs = build_dynamic_graphs(prices, window_size=window_size, step_size=step_size, ngc_config=ngc_cfg)

    # Example supervised target: next-day return; features: same-day return
    returns = prices.pct_change().dropna()
    features_df = returns.iloc[:-1]
    targets_df = returns.iloc[1:]

    dataset = TimeWindowDataset(features_df, targets_df)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    num_nodes = prices.shape[1]
    model = StockModel(in_dim=1, hidden_dim=hidden_dim, out_dim=1, heads=heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (ts,), (x,), (y,) in loader:
            ts = ts[0]
            x = x[0].to(device)  # [N, 1]
            y = y[0].to(device)  # [N, 1]

            # Select and moralize the directed graph for this timestamp
            directed_adj = select_graph_for_timestamp(graphs, ts)
            undirected_adj = moralize_graph(directed_adj)
            edge_index, edge_weight = to_edge_index(undirected_adj, threshold=0.0)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            pred = model(x, edge_index=edge_index, edge_weight=edge_weight)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV file with datetime index and stock columns")
    parser.add_argument("--window_size", type=int, default=252)
    parser.add_argument("--step_size", type=int, default=21)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()

    ngc_cfg = NGCConfig(device=args.device)
    print("NGC config:", asdict(ngc_cfg))
    train(
        prices=df,
        window_size=args.window_size,
        step_size=args.step_size,
        ngc_cfg=ngc_cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()

