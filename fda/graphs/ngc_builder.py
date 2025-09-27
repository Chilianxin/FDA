from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 2, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_lagged_design(R: pd.DataFrame, max_lag: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a design matrix from panel returns with lags.

    Inputs
    - R: DataFrame indexed by trade_date, columns are tickers; values are returns
    - max_lag: number of past lags to include per stock

    Returns
    - X: [T - max_lag, N * max_lag] design matrix, row t contains [x_i(t-1..t-max_lag)] for all i
    - groups: list mapping each column block to stock index i: (stock_index, lag_index)
    """
    Rv = R.values.astype(np.float32)
    T, N = Rv.shape
    if T <= max_lag:
        raise ValueError("Not enough time steps for specified max_lag")
    X_list = []
    groups: List[Tuple[int, int]] = []
    for lag in range(1, max_lag + 1):
        X_list.append(Rv[max_lag - lag: T - lag, :])
        for i in range(N):
            groups.append((i, lag))
    X = np.concatenate(X_list, axis=1)  # [T-max_lag, N*max_lag]
    return X, groups


def _group_lasso_penalty(model: nn.Module, groups: List[Tuple[int, int]], num_stocks: int, max_lag: int) -> torch.Tensor:
    """
    Group Lasso over input groups per stock across all lags.

    We assume the first Linear layer of MLP maps from [N*max_lag] -> hidden (or -> 1 for depth=1).
    For Group Lasso, we aggregate the Frobenius norm of the weight sub-matrix corresponding to each stock
    across all its lags.
    """
    # Find the first Linear layer
    first_linear: Optional[nn.Linear] = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    W: torch.Tensor = first_linear.weight  # [out_dim, in_dim]
    in_dim = W.shape[1]
    expected_in = num_stocks * max_lag
    if in_dim != expected_in:
        # If input dimension does not match expected, skip group lasso
        return torch.tensor(0.0, device=W.device)
    # Reshape to [out_dim, num_stocks, max_lag]
    Wg = W.view(W.shape[0], num_stocks, max_lag)
    # Group norm per stock: Frobenius over (out_dim, max_lag)
    penalty = 0.0
    for i in range(num_stocks):
        Wi = Wg[:, i, :]  # [out_dim, max_lag]
        penalty = penalty + torch.norm(Wi, p=2)
    return penalty


@dataclass
class NGCConfig:
    max_lag: int = 5
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    lmbda_group: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NGC_Builder:
    """
    Neural Granger Causality builder for a panel of stocks.

    For each target stock j, fit an MLP to predict r_j(t) from lagged returns of all stocks.
    Apply Group Lasso over groups defined by source stock i (across all lags), encouraging sparsity at stock level.
    The learned non-zero groups indicate i -> j Granger causality.
    """

    def __init__(self, config: Optional[NGCConfig] = None):
        self.cfg = config or NGCConfig()

    def _fit_single_target(self, R: pd.DataFrame, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train an MLP to predict target stock using all lagged inputs.

        Returns
        - coef_importance: [N] importance score per source stock for this target
        - y_pred: predictions for alignment/debug (ignored by caller)
        """
        cfg = self.cfg
        X_all, groups = _build_lagged_design(R, cfg.max_lag)
        y_full = R.values.astype(np.float32)[cfg.max_lag :, target_idx]
        T = X_all.shape[0]
        N = R.shape[1]

        # train/val split (last 20% as validation)
        val_len = max(1, int(0.2 * T))
        train_idx = np.arange(0, T - val_len)
        val_idx = np.arange(T - val_len, T)

        X_train = torch.from_numpy(X_all[train_idx])
        y_train = torch.from_numpy(y_full[train_idx]).unsqueeze(-1)
        X_val = torch.from_numpy(X_all[val_idx])
        y_val = torch.from_numpy(y_full[val_idx]).unsqueeze(-1)

        in_dim = X_all.shape[1]
        model = _MLP(in_dim=in_dim, hidden=cfg.hidden, depth=cfg.depth, out_dim=1, dropout=cfg.dropout).to(cfg.device)
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_val = math.inf
        best_state = None
        patience = cfg.patience
        no_improve = 0

        def iterate_minibatches(X: torch.Tensor, y: torch.Tensor, bs: int):
            num = X.shape[0]
            idx = torch.randperm(num)
            for s in range(0, num, bs):
                batch = idx[s : s + bs]
                yield X[batch].to(cfg.device), y[batch].to(cfg.device)

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in iterate_minibatches(X_train, y_train, cfg.batch_size):
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                mse = nn.functional.mse_loss(pred, yb)
                gl = _group_lasso_penalty(model, groups, num_stocks=N, max_lag=cfg.max_lag)
                loss = mse + cfg.lmbda_group * gl
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)

            # validation
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val.to(cfg.device))
                val_mse = nn.functional.mse_loss(pred_val, y_val.to(cfg.device)).item()
            if val_mse < best_val - 1e-8:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # derive group importances from first layer weights after training
        first_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            coef_importance = np.zeros(N, dtype=np.float32)
        else:
            W = first_linear.weight.detach().cpu().numpy()  # [out_dim, in_dim]
            W = W.reshape(W.shape[0], N, cfg.max_lag)       # [out_dim, N, L]
            # Frobenius norm over (out_dim, L)
            gi = np.linalg.norm(W, axis=(0, 2))
            coef_importance = gi.astype(np.float32)

        # predictions for entire range (optional)
        with torch.no_grad():
            X_full = torch.from_numpy(X_all).to(cfg.device)
            y_hat = model(X_full).squeeze(-1).detach().cpu().numpy()
        return coef_importance, y_hat

    def build_causal_graph(
        self,
        window_df: pd.DataFrame,
        max_nodes: Optional[int] = None,
        normalize: bool = True,
        top_k_per_col: Optional[int] = 15,
        node_order: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build directed adjacency A (N, N) where A[i,j] measures i -> j causality strength.

        Inputs
        - window_df: panel window with columns ['trade_date','ts_code','r'] (requires R pivotable)
        - max_nodes: optionally limit number of stocks for speed (take first N by sorted symbol)
        - normalize: scale each column to [0,1] by its max (if > 0)
        - top_k_per_col: keep top-K incoming causes per target j
        - node_order: if provided, enforce this order of nodes (intersecting available stocks)
        """
        req = {'trade_date', 'ts_code', 'r'}
        if not req.issubset(set(window_df.columns)):
            missing = sorted(list(req - set(window_df.columns)))
            raise ValueError(f"window_df missing columns: {missing}")

        R = window_df.pivot(index='trade_date', columns='ts_code', values='r').sort_index().fillna(0.0)
        if node_order is not None:
            cols = [c for c in node_order if c in R.columns]
            R = R.reindex(columns=cols)
        else:
            R = R.reindex(columns=sorted(R.columns))

        if max_nodes is not None and max_nodes > 0:
            R = R.iloc[:, : max_nodes]

        nodes = list(R.columns)
        N = len(nodes)
        A = np.zeros((N, N), dtype=np.float32)

        for j in range(N):
            imp_j, _ = self._fit_single_target(R, j)
            # zero out self-causality for clarity
            imp_j[j] = 0.0
            A[:, j] = imp_j

        # normalize columns to [0,1]
        if normalize:
            for j in range(N):
                col = A[:, j]
                m = col.max()
                if m > 0:
                    A[:, j] = col / m

        # keep top-K incoming per column
        if top_k_per_col is not None and top_k_per_col > 0:
            for j in range(N):
                col = A[:, j]
                if (col > 0).any():
                    kth = np.partition(col, -min(top_k_per_col, N - 1))[-min(top_k_per_col, N - 1)]
                    A[:, j] = col * (col >= kth)
        return A, nodes


def moralize_graph(A: np.ndarray) -> np.ndarray:
    """
    Moralize a directed adjacency matrix A (N,N) into an undirected matrix M (N,N).
    Steps:
    1) Connect parents: for each node j, fully connect all i,k where A[i,j]>0 and A[k,j]>0
    2) Drop directions: make edges undirected by symmetrizing and taking max weight
    """
    N = A.shape[0]
    M = np.zeros_like(A, dtype=np.float32)
    # Step 1: connect parents
    for j in range(N):
        parents = np.where(A[:, j] > 0)[0]
        for p_idx in range(len(parents)):
            i = parents[p_idx]
            for q_idx in range(p_idx + 1, len(parents)):
                k = parents[q_idx]
                w = float(min(1.0, max(A[i, j], A[k, j])))
                M[i, k] = max(M[i, k], w)
                M[k, i] = max(M[k, i], w)
    # Step 2: drop direction by symmetrizing original A into M as well
    S = np.maximum(A, A.T)
    M = np.maximum(M, S)
    np.fill_diagonal(M, 0.0)
    return M


def to_edge_index(S: np.ndarray, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert undirected similarity/adjacency matrix S (N,N) into edge_index and edge_weight tensors.
    Returns
    - edge_index: [E, 2] with undirected edges represented once per direction (i->j and j->i)
    - edge_weight: [E]
    """
    assert S.ndim == 2 and S.shape[0] == S.shape[1]
    N = S.shape[0]
    rows, cols = np.where(S > threshold)
    # build undirected by duplicating both directions
    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        w = float(S[i, j])
        if w <= threshold:
            continue
        src.append(i); dst.append(j); weights.append(w)
        src.append(j); dst.append(i); weights.append(w)
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def rolling_dynamic_ngc(
    df: pd.DataFrame,
    window_size: int = 252,
    step_size: int = 21,
    cfg: Optional[NGCConfig] = None,
    node_order: Optional[List[str]] = None,
    top_k_per_col: int = 15,
) -> Dict[str, Dict[str, object]]:
    """
    Build dynamic causal graphs over rolling windows.

    Returns a dict keyed by window end_date (str):
      {
        'A': directed adjacency (N,N),
        'M': moralized undirected adjacency (N,N),
        'edge_index': LongTensor [E,2],
        'edge_weight': FloatTensor [E],
        'nodes': List[str]
      }
    """
    cfg = cfg or NGCConfig()
    dates = np.sort(df['trade_date'].unique())
    out: Dict[str, Dict[str, object]] = {}
    builder = NGC_Builder(cfg)
    idx = 0
    while True:
        start = idx
        end = idx + window_size
        if end > len(dates):
            break
        end_date = dates[end - 1]
        wdf = df[(df['trade_date'] >= dates[start]) & (df['trade_date'] <= end_date)][['trade_date','ts_code','r']].copy()
        A, nodes = builder.build_causal_graph(wdf, node_order=node_order, top_k_per_col=top_k_per_col)
        M = moralize_graph(A)
        ei, ew = to_edge_index(M, threshold=0.0)  # [2,E]
        adj_idx = ei.t().contiguous()             # [E,2] for existing RGAT
        out[str(end_date)] = {
            'A': A,
            'M': M,
            'edge_index': ei,
            'edge_weight': ew,
            'adj_indices': adj_idx,
            'adj_weights': ew,
            'nodes': nodes,
        }
        idx += step_size
    return out

