from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def _pivot_returns(window_df: pd.DataFrame) -> pd.DataFrame:
    R = window_df.pivot(index='trade_date', columns='ts_code', values='r').sort_index().fillna(0.0)
    return R


def _pseudo_index_return(window_df: pd.DataFrame) -> pd.Series:
    if 'circ_mv' in window_df.columns:
        w = window_df.groupby('trade_date')['circ_mv'].transform(lambda s: s / (s.sum() + 1e-12))
        return (window_df['r'] * w).groupby(window_df['trade_date']).sum()
    return window_df.groupby('trade_date')['r'].mean()


def _neutralize_to_index(R: pd.DataFrame, idx_r: pd.Series) -> pd.DataFrame:
    idx = idx_r.reindex(R.index).fillna(0.0).values[:, None]
    X = np.c_[np.ones(len(idx)), idx]
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ R.values
    fitted = X @ betas
    resid = R.values - fitted
    return pd.DataFrame(resid, index=R.index, columns=R.columns)


def _corr_similarity(Rn: pd.DataFrame) -> np.ndarray:
    S = np.corrcoef(Rn.values.T)
    S = np.nan_to_num(S, nan=0.0)
    return np.clip(S, 0.0, 1.0)


def _tail_comove(R: pd.DataFrame, q: float = 0.2) -> np.ndarray:
    thr = R.quantile(q, axis=0)
    I = (R.lt(thr, axis=1)).astype(float)
    T = (I.T @ I) / max(1, len(R))
    T = T.values
    return T / (T.max() + 1e-12)


def _leadlag(R: pd.DataFrame, max_lag: int = 3) -> np.ndarray:
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


def _rbf(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / max(1, X.shape[1])
    G = X @ X.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    S = np.exp(-gamma * np.clip(sq, 0.0, None))
    np.fill_diagonal(S, 0.0)
    return S


def build_fused_graph(df: pd.DataFrame, end_date: str, window: int = 120, top_k: int = 15,
                      weights=(0.25,0.25,0.15,0.25,0.10)) -> Tuple[np.ndarray, list]:
    mask = (df['trade_date'] <= end_date)
    dates = np.sort(df.loc[mask, 'trade_date'].unique())
    if len(dates) == 0:
        raise ValueError(f"No data up to {end_date}")
    start_cut = dates[max(0, len(dates) - window)]
    wdf = df[(df['trade_date'] >= start_cut) & (df['trade_date'] <= end_date)].copy()

    R = _pivot_returns(wdf)
    idx_r = _pseudo_index_return(wdf)
    Rn = _neutralize_to_index(R, idx_r)
    nodes = list(Rn.columns)

    S_corr = _corr_similarity(Rn)
    S_tail = _tail_comove(R)

    fcols = ['turnover_rate_xz','volume_ratio_xz','amihud_xz','log_circ_mv_xz','pe_z','pb_z']
    feats = wdf.sort_values('trade_date').groupby('ts_code')[fcols].last().reindex(nodes)
    feats = feats.fillna(feats.median())
    S_liqval = _rbf(feats.values)

    S_leadlag = _leadlag(R)

    w60 = wdf.tail(min(60, len(wdf)))
    vr = w60.groupby('ts_code')['volume_ratio'].transform(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-8))
    flag = (vr > 1.0) & (w60['r'] < w60.groupby('ts_code')['r'].transform(lambda s: s.quantile(0.2)))
    co = pd.crosstab(w60.loc[flag,'trade_date'], w60.loc[flag,'ts_code']).T
    if co.shape[0] == 0:
        S_impact = np.zeros_like(S_corr)
    else:
        C = co.values @ co.values.T
        S_impact = C / (C.max() + 1e-12)
        S_impact = S_impact[:len(nodes), :len(nodes)]

    w1,w2,w3,w4,w5 = weights
    S = w1*S_corr + w2*S_tail + w3*S_leadlag + w4*S_liqval + w5*S_impact
    S = (S + S.T) / 2.0
    S = np.clip(S, 0.0, None)
    # Top-K sparsify
    if top_k and top_k > 0:
        kth = np.partition(S, -top_k, axis=1)[:, -top_k][:, None]
        S = S * (S >= kth)
    if S.max() > 0:
        S = S / S.max()
    np.fill_diagonal(S, 0.0)
    return S, nodes

