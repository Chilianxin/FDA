## Pseudo-Industry generation via graph communities

Generate weekly pseudo-industry mapping from HS300-like CSV input.

Example:

```bash
pip install -r requirements.txt
bash scripts/generate_pseudo_industry.sh data/hs300.csv outputs/pseudo_industry
```

This will produce per-week CSVs and `pseudo_industry_latest.parquet/csv` in the output directory.

# FAD: Financial Alpha with Differentiable execution

End-to-end research codebase for building pseudo-industry mappings via graph communities and preparing inputs for a large-capacity, risk-averse, impact-aware deep reinforcement learning strategy.

This repo currently includes a complete, runnable pipeline to generate weekly pseudo-industry labels (graph communities) from pure numerical A-share data (HS300-like). These labels can be used as “industry buckets” for portfolio constraints, cross-impact modeling, and risk budgeting in downstream training.

Planned next steps (not yet included here): deep models (TCN-MoE + Relational GAT), market regime module, differentiable execution/cost head, and a constrained RL agent (PPO/SAC with CVaR constraint).

## Repository Structure

```
code/
  __init__.py
  features.py                 # Feature engineering (returns, volatility, ADV, Amihud, cross-sectional z-scores)
  graph.py                    # Multi-relation fused graph construction + top-K sparsification
  community.py                # Leiden/Louvain communities, weekly alignment, smoothing, small-cluster merge
  cli_pseudo_industry.py      # CLI to generate weekly pseudo-industry mapping artifacts
configs/
  config.yaml                 # (placeholder)
  logging.yaml                # (placeholder)
docs/
  DATA.md, EXPERIMENTS.md, MODEL_DESIGN.md, USAGE.md   # (placeholders for future expansion)
scripts/
  generate_pseudo_industry.sh # One-liner to run the CLI
  train.sh, evaluate.sh       # (placeholders)
requirements.txt              # Python dependencies
```

## Data Schema

Input CSV required columns:
- ts_code, trade_date (YYYYMMDD)
- open, high, low, close
- pct_chg (percentage, e.g., 1.23 for +1.23%)
- vol, amount
- turnover_rate, volume_ratio
- pe, pb, circ_mv
- 市场融资融券余额变化率 (optional, used downstream; not required for community detection)

Assumptions:
- Daily frequency, HS300 constituents or similar liquid universe.
- `trade_date` is sortable string (YYYYMMDD).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- Community detection prefers `python-igraph` + `leidenalg`. If unavailable, it falls back to `python-louvain` via NetworkX.
- `pyarrow` is used to save a Parquet artifact of all snapshots.

## Feature Engineering (code/features.py)

- compute_returns: builds log-returns from `pct_chg`.
- compute_volatility: Parkinson volatility proxy from OHLC.
- compute_liquidity_features: ADV (60d rolling mean of amount), Amihud proxy (E[|r|/amount]).
- add_static_features: `log_circ_mv`, cross-sectional z-scores for `pe/pb` per day.
- compute_cross_sectional_zscores: per-day z-scores for liquidity/size proxies.
- engineer_features: orchestrates the full default pipeline.

## Graph Construction (code/graph.py)

`build_multi_relation_graph(df, end_date, window=120, top_k=15, weights=(0.25,0.25,0.15,0.25,0.10))`
- Selects a rolling window (default 120d) ending at `end_date`.
- Builds and fuses multiple similarities (all within [0,1]):
  - Correlation (neutralized to a pseudo-index) / partial similarity
  - Tail co-movement (bottom 20% co-exceedance)
  - Lead-lag similarity (max cross-correlation with lags 1–3)
  - Liquidity & valuation similarity (RBF over z-scored turnover/volume_ratio/Amihud/log_circ_mv/pe/pb)
  - Impact co-occurrence (volume_ratio high & negative return co-occurrence in recent 60d)
- Symmetrizes, top-K sparsifies, and rescales to [0,1]. Returns `(S, nodes)`.

## Community Detection (code/community.py)

- leiden_communities: Leiden (preferred) or Louvain fallback over weighted, undirected graph.
- align_labels: greedy overlap matching to align weekly labels to previous week.
- smooth_membership: 4-week majority voting (configurable) to stabilize labels.
- merge_small_clusters: merges clusters smaller than a threshold (default 5) into nearest large cluster.

## CLI: Pseudo-Industry Generation (code/cli_pseudo_industry.py)

Generate weekly pseudo-industry labels from an input CSV, save per-week CSVs, plus `latest` artifacts.

```bash
python -m code.cli_pseudo_industry \
  --input /absolute/path/to/hs300.csv \
  --outdir /absolute/path/to/outputs/pseudo_industry \
  --window 120 --topk 15 --resolution 1.0 --min_size 5

# or convenience script
bash scripts/generate_pseudo_industry.sh /absolute/path/to/hs300.csv /absolute/path/to/outputs/pseudo_industry
```

Outputs per run:
- Per-week CSV: `pseudo_industry_YYYY-Www.csv` with columns `ts_code, pseudo_industry, trade_week, trade_date`
- Latest snapshot CSV/Parquet: `pseudo_industry_latest.csv` and `pseudo_industry_latest.parquet`

## How to Use the Pseudo-Industry in Training

1) Join labels back to your daily panel:
```python
import pandas as pd
panel = pd.read_csv('your_panel.csv', dtype={'ts_code': str, 'trade_date': str})
labels = pd.read_csv('outputs/pseudo_industry/pseudo_industry_latest.csv', dtype={'ts_code': str})
panel = panel.merge(labels[['ts_code','pseudo_industry']], on='ts_code', how='left')
```

2) Constraints with pseudo-industry buckets k:
- Cluster weight cap: ∑_{i in k} w_i ≤ w̄_k
- Cluster turnover cap: ∑_{i in k} |q_i| ≤ τ_k
- Cluster ADV cap: ∑_{i in k} |q_i| ≤ ρ_k · ADV_k (ADV_k is cluster ADV)

3) Cross-impact (optional): increase cross-impact γ_{ij} for same-cluster pairs to penalize crowding/chain impact.

4) Reporting: track cluster concentration, violations, and their effect on post-cost performance and CVaR.

## Roadmap for Full Training (planned)

The following components are part of the intended full system and will be added incrementally:

- Prediction module (DL):
  - Stock-intra: multi-scale TCN-MoE (short/mid/long), regime-aware gating (volatility/liquidity/valuation/two-financing).
  - Stock-inter: multi-relation, directed Relational GAT over the same graph used for pseudo-industry.
  - Cross-view coupling: GAT→MoE and MoE→GAT gating; scale alignment.
  - Style head: learn style exposures E for constraints/budgeting.

- Market module:
  - Regime classifier from index/breadth/liquidity features.
  - Differentiable execution & impact head (single-name and cross-name impact).

- RL module (constrained):
  - State includes predictions, style exposures, market regime, impact params, account state.
  - Action includes target weights or trade vector and multi-day execution schedule (SoftSchedule).
  - Constraints: cluster caps (pseudo-industry), ADV caps, CVaR/Drawdown, turnover, style budgets.
  - Algorithm: PPO/SAC with Lagrangian multipliers and distributional/quantile critics.

Expected training flow:
```
Stage A: Pretrain DL encoders (self-supervised + supervised ranking/quantile losses)
Stage B: Pretrain differentiable portfolio + execution head (post-cost objective + CVaR regularization)
Stage C: Constrained RL fine-tuning in rolling simulation (T+1, limit-up/down soft fills, ADV caps)
```

The `scripts/train.sh` and `scripts/evaluate.sh` are placeholders to be wired to the training entrypoints once these modules are added. Until then, you can already use the generated pseudo-industry labels to enforce realistic cluster constraints in your own training loop.

## Repro & Evaluation Tips

- Use Purged/Embargo walk-forward splits (e.g., 12m train / 3m val / 3m test; embargo ≈ 20d).
- Report costed performance (net of impact and fees), CVaR(10%), Max Drawdown, turnover, ADV violation rate, and capacity curve vs AUM.
- Perform ablations: no cluster constraints, no cross-impact boost, different top-K, different resolution, different windows (60/120/180d).

## License

See `LICENSE` in the repository.
