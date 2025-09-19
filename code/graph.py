from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _pivot_returns(df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    R = window_df.pivot(index='trade_date', columns='ts_code', values='r')
    R = R.sort_index().fillna(0.0)
    return R


def _weighted_index_return(window_df: pd.DataFrame) -> pd.Series:
    # pseudo-index: circ_mv-weighted average return per day
    if 'circ_mv' in window_df.columns:
        w = window_df.groupby('trade_date')['circ_mv'].transform(lambda s: s / (s.sum() + 1e-12))
        return (window_df['r'] * w).groupby(window_df['trade_date']).sum()
    else:
        return window_df.groupby('trade_date')['r'].mean()


def _neutralize_to_index(R: pd.DataFrame, idx_r: pd.Series) -> pd.DataFrame:
    idx = idx_r.reindex(R.index).fillna(0.0).values[:, None]
    # simple OLS beta per column
    X = np.c_[np.ones(len(idx)), idx]
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ R.values
    fitted = X @ betas
    resid = R.values - fitted
    return pd.DataFrame(resid, index=R.index, columns=R.columns)


def _corr_partial_similarity(Rn: pd.DataFrame) -> np.ndarray:
    # Pearson correlation as baseline similarity
    S_corr = np.corrcoef(Rn.values.T)
    S_corr = np.nan_to_num(S_corr, nan=0.0)
    # clip negatives for similarity
    S_corr = np.clip(S_corr, 0.0, 1.0)
    return S_corr


def _tail_co_movement(R: pd.DataFrame, q: float = 0.2) -> np.ndarray:
    # indicator of being in bottom q-quantile per stock
    thresh = R.quantile(q, axis=0)
    I = (R.lt(thresh, axis=1)).astype(float)
    Tail = (I.T @ I) / max(1, len(R))
    Tail = Tail.values
    Tail = Tail / (Tail.max() + 1e-12)
    return Tail


def _lead_lag_similarity(R: pd.DataFrame, max_lag: int = 3) -> np.ndarray:
    # Symmetric similarity by maximum cross-correlation over small lags
    X = R.values
    n = X.shape[1]
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = X[:, i]
        for j in range(i + 1, n):
            xj = X[:, j]
            cc_max = 0.0
            for L in range(1, max_lag + 1):
                if L < len(X):
                    cc1 = np.corrcoef(xi[L:], xj[:-L])[0, 1]
                    cc2 = np.corrcoef(xi[:-L], xj[L:])[0, 1]
                    cc_max = max(cc_max, np.nan_to_num(cc1, nan=0.0), np.nan_to_num(cc2, nan=0.0))
            S[i, j] = S[j, i] = max(0.0, cc_max)
    return S


def _rbf_similarity(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / max(1, X.shape[1])
    # pairwise squared distance
    G = X @ X.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    S = np.exp(-gamma * np.clip(sq, 0.0, None))
    np.fill_diagonal(S, 0.0)
    return S


def build_multi_relation_graph(
    df: pd.DataFrame,
    end_date: str,
    window: int = 120,
    top_k: int = 15,
    weights: Tuple[float, float, float, float, float] = (0.25, 0.25, 0.15, 0.25, 0.10),
) -> Tuple[np.ndarray, list]:
    """Build fused similarity matrix S and aligned node list for a given window ending at end_date.

    Returns (S, nodes) where S is (N,N) similarity in [0,1], nodes is list of ts_code ordered by columns.
    """
    # select rolling window
    mask = (df['trade_date'] <= end_date)
    dates = np.sort(df.loc[mask, 'trade_date'].unique())
    if len(dates) == 0:
        raise ValueError(f"No data up to {end_date}")
    start_cut = dates[max(0, len(dates) - window)]
    wdf = df[(df['trade_date'] >= start_cut) & (df['trade_date'] <= end_date)].copy()

    # pivot returns and neutralize to pseudo index
    R = _pivot_returns(df, wdf)
    idx_r = _weighted_index_return(wdf)
    Rn = _neutralize_to_index(R, idx_r)

    # aligned nodes
    nodes = list(Rn.columns)
    # similarity components
    S_corr = _corr_partial_similarity(Rn)
    S_tail = _tail_co_movement(R)

    # features for liquidity/valuation similarity
    fcols = ['turnover_rate_xz', 'volume_ratio_xz', 'amihud_xz', 'log_circ_mv_xz', 'pe_z', 'pb_z']
    feats = wdf.sort_values('trade_date').groupby('ts_code')[fcols].last().reindex(nodes)
    feats = feats.fillna(feats.median())
    S_liqval = _rbf_similarity(feats.values)

    S_leadlag = _lead_lag_similarity(R)

    # impact co-move
    w60 = wdf[wdf['trade_date'] >= np.sort(wdf['trade_date'].unique())[-min(60, len(wdf['trade_date'].unique()))]]
    vr = w60.groupby('ts_code')['volume_ratio'].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
    flag = (vr > 1.0) & (w60['r'] < w60.groupby('ts_code')['r'].transform(lambda s: s.quantile(0.2)))
    # build co-occurrence
    dates60 = w60.loc[flag, 'trade_date']
    co = pd.crosstab(w60.loc[flag, 'trade_date'], w60.loc[flag, 'ts_code']).T
    co = co.values @ co.values.T
    if co.size == 0:
        S_impact = np.zeros_like(S_corr)
    else:
        S_impact = co.astype(float)
        S_impact = S_impact / (S_impact.max() + 1e-12)
        S_impact = S_impact[: len(nodes), : len(nodes)]

    w1, w2, w3, w4, w5 = weights
    S = w1 * S_corr + w2 * S_tail + w3 * S_leadlag + w4 * S_liqval + w5 * S_impact
    S = (S + S.T) / 2.0
    S = np.clip(S, 0.0, None)

    # sparsify by top-k
    if top_k is not None and top_k > 0:
        kth = np.partition(S, -top_k, axis=1)[:, -top_k][:, None]
        mask_topk = S >= kth
        S = S * mask_topk
    # normalize to [0,1]
    if S.max() > 0:
        S = S / S.max()
    np.fill_diagonal(S, 0.0)
    return S, nodes

