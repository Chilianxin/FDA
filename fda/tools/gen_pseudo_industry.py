from __future__ import annotations

import argparse
import os
import pandas as pd
from datetime import datetime

from fda.data.dataset import load_panel
from fda.graphs.build_graph import build_fused_graph
from fda.graphs.communities import detect_communities, align_labels, smooth_labels, merge_small


def to_week(date_str: str) -> str:
    d = datetime.strptime(date_str, '%Y%m%d')
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"


def main():
    ap = argparse.ArgumentParser(description='Generate weekly pseudo-industry mapping via graph communities')
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--window', type=int, default=120)
    ap.add_argument('--topk', type=int, default=15)
    ap.add_argument('--resolution', type=float, default=1.0)
    ap.add_argument('--min_size', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_panel(args.input)

    weeks = sorted({to_week(d) for d in df['trade_date'].unique()})
    last_date_by_week = {}
    for date in sorted(df['trade_date'].unique()):
        w = to_week(date)
        last_date_by_week[w] = date

    prev = None
    history = []
    snapshots = []
    for w in weeks:
        end_date = last_date_by_week[w]
        try:
            S, nodes = build_fused_graph(df, end_date=end_date, window=args.window, top_k=args.topk)
        except ValueError:
            continue
        labs = detect_communities(S, resolution=args.resolution)
        labs = align_labels(prev, labs) if prev is not None else labs
        labs = merge_small(labs, min_size=args.min_size)
        history.append(labs)
        if len(history) >= 2:
            labs = smooth_labels(history, window=min(4, len(history)))
        prev = labs.copy()
        mapping = pd.DataFrame({'ts_code': nodes, 'pseudo_industry': labs, 'trade_week': w, 'trade_date': end_date})
        snapshots.append(mapping)
        mapping.to_csv(os.path.join(args.outdir, f'pseudo_industry_{w}.csv'), index=False)

    if snapshots:
        all_map = pd.concat(snapshots, ignore_index=True)
        all_map.to_parquet(os.path.join(args.outdir, 'pseudo_industry_latest.parquet'), index=False)
        latest_week = max(all_map['trade_week'])
        all_map[all_map['trade_week'] == latest_week].to_csv(os.path.join(args.outdir, 'pseudo_industry_latest.csv'), index=False)


if __name__ == '__main__':
    main()

