import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

from .features import engineer_features
from .graph import build_multi_relation_graph
from .community import leiden_communities, align_labels, smooth_membership, merge_small_clusters


def parse_args():
    ap = argparse.ArgumentParser(description="Generate weekly pseudo-industry mapping via graph communities")
    ap.add_argument('--input', required=True, help='Input CSV with columns: ts_code,trade_date,open,high,low,close,pct_chg,vol,amount,turnover_rate,volume_ratio,pe,pb,circ_mv,市场融资融券余额变化率')
    ap.add_argument('--outdir', required=True, help='Output directory for per-week mapping CSVs and a latest parquet')
    ap.add_argument('--window', type=int, default=120, help='Rolling window (days) for graph construction')
    ap.add_argument('--topk', type=int, default=15, help='Top-K neighbors to keep per node')
    ap.add_argument('--resolution', type=float, default=1.0, help='Leiden resolution parameter')
    ap.add_argument('--min_size', type=int, default=5, help='Minimum cluster size to keep; smaller will be merged')
    return ap.parse_args()


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_week(date_str: str) -> str:
    # ISO week label: YYYY-Www
    d = datetime.strptime(date_str, '%Y%m%d')
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"


def main():
    args = parse_args()
    ensure_outdir(args.outdir)
    df = pd.read_csv(args.input, dtype={'ts_code': str, 'trade_date': str})
    df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

    # Feature engineering
    df_feat = engineer_features(df)

    # Weekly endpoints
    weeks = sorted({to_week(d) for d in df_feat['trade_date'].unique()})
    # Build a mapping from week label to last trade_date in that week
    last_date_by_week = {}
    for date in sorted(df_feat['trade_date'].unique()):
        w = to_week(date)
        last_date_by_week[w] = date

    prev_labels = None
    history = []
    all_snapshots = []

    for w in weeks:
        end_date = last_date_by_week[w]
        try:
            S, nodes = build_multi_relation_graph(df_feat, end_date=end_date, window=args.window, top_k=args.topk)
        except ValueError:
            continue
        labs = leiden_communities(S, resolution=args.resolution)
        labs = align_labels(prev_labels, labs) if prev_labels is not None else labs
        labs = merge_small_clusters(labs, min_size=args.min_size)
        history.append(labs)
        if len(history) >= 2:
            labs = smooth_membership(history, window=min(4, len(history)))
        prev_labels = labs.copy()

        mapping = pd.DataFrame({'ts_code': nodes, 'pseudo_industry': labs})
        mapping['trade_week'] = w
        mapping['trade_date'] = end_date
        all_snapshots.append(mapping)
        out_csv = os.path.join(args.outdir, f"pseudo_industry_{w}.csv")
        mapping.to_csv(out_csv, index=False)

    if all_snapshots:
        out_all = pd.concat(all_snapshots, ignore_index=True)
        out_parquet = os.path.join(args.outdir, 'pseudo_industry_latest.parquet')
        out_all.to_parquet(out_parquet, index=False)
        # Also write latest week alias
        latest_week = max(out_all['trade_week'])
        latest_map = out_all[out_all['trade_week'] == latest_week]
        latest_map.to_csv(os.path.join(args.outdir, 'pseudo_industry_latest.csv'), index=False)


if __name__ == '__main__':
    main()

