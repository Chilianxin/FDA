from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch

from fda.data.dataset import load_panel
from fda.graphs.ngc_builder import NGCConfig, rolling_dynamic_ngc
from fda.models.predictor import Predictor


def _dummy_batch(N: int, seq_channels: int = 8, T: int = 32, cond_dim: int = 16):
    x_seq = torch.randn(N, seq_channels, T)
    x_node = torch.randn(N, 64)
    cond = torch.randn(N, cond_dim)
    return x_seq, x_node, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--data_csv', type=str, default='')
    ap.add_argument('--window_size', type=int, default=252)
    ap.add_argument('--step_size', type=int, default=21)
    ap.add_argument('--max_nodes', type=int, default=64)
    args = ap.parse_args()

    print('Stage A: DL encoders with dynamic NGC graph (skeleton)')

    if not args.data_csv:
        print('No data_csv provided; skipping dynamic graph build (demo only).')
        return

    df = load_panel(args.data_csv)
    # Restrict to the latest N tickers for demo speed
    last_date = df['trade_date'].max()
    tickers = (
        df[df['trade_date'] == last_date]['ts_code']
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if args.max_nodes and args.max_nodes < len(tickers):
        tickers = tickers[: args.max_nodes]
    df = df[df['ts_code'].isin(tickers)].copy()

    # Build rolling NGC-based dynamic graphs
    ngc_cfg = NGCConfig(max_lag=5, hidden=128, depth=2, lmbda_group=1e-3, epochs=50, batch_size=256)
    graphs: Dict[str, Dict[str, object]] = rolling_dynamic_ngc(
        df,
        window_size=args.window_size,
        step_size=args.step_size,
        cfg=ngc_cfg,
        node_order=tickers,
        top_k_per_col=15,
    )
    if not graphs:
        print('No dynamic graphs built (insufficient data).')
        return

    # Prepare model (dimensions are placeholders for demo)
    seq_in_feat = 8
    cond_dim = 16
    node_feat_dim = 64
    model = Predictor(seq_in_feat=seq_in_feat, cond_dim=cond_dim, node_feat_dim=node_feat_dim)

    # Simple loop over dynamic windows
    for epoch in range(args.epochs):
        for end_date, g in graphs.items():
            adj_indices = g['adj_indices']  # [E,2]
            adj_weights = g['adj_weights']  # [E]
            nodes = g['nodes']
            N = len(nodes)
            # dummy features for demonstration
            x_seq, x_node, cond = _dummy_batch(N, seq_channels=seq_in_feat, T=32, cond_dim=cond_dim)
            out = model(x_seq, x_node, cond, adj_indices, adj_weights)
            mu = out['mu']
            print(f"Epoch {epoch+1} @ {end_date}: N={N}, edges={adj_indices.size(0)}; mu.mean={mu.mean().item():.4f}")


if __name__ == '__main__':
    main()

