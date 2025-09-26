from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_extractor(x)
        out = self.regressor(hidden)
        return out


def compute_group_lasso_penalty(
    first_layer_weight: torch.Tensor,
    group_size: int,
    num_groups: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Group Lasso (l2,1) penalty over groups of input features.

    The penalty is computed on the first layer weight matrix W with shape
    [hidden_dim, input_dim]. The input features are organized as groups
    of contiguous features, each group corresponding to all lags of one stock.

    For group g, extract W[:, g*group_size : (g+1)*group_size] and compute
    its Frobenius norm. Sum over groups.
    """
    if first_layer_weight.dim() != 2:
        raise ValueError("Expected first layer weight to be 2D [hidden_dim, input_dim]")

    hidden_dim, input_dim = first_layer_weight.shape
    if num_groups * group_size != input_dim:
        raise ValueError(
            f"num_groups * group_size must equal input_dim, got {num_groups} * {group_size} != {input_dim}"
        )

    penalty = first_layer_weight.new_tensor(0.0)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        block = first_layer_weight[:, start:end]
        # Frobenius norm of block
        group_norm = torch.sqrt(torch.sum(block * block) + epsilon)
        penalty = penalty + group_norm
    return penalty


@dataclass
class NGCConfig:
    lags: int = 5
    hidden_dim: int = 64
    dropout_p: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    weight_decay: float = 0.0
    group_lasso_alpha: float = 1e-3
    early_stop_patience: int = 20
    threshold_percentile: float = 80.0  # edges above this percentile are kept (per target)
    device: str = "cpu"


class NGC_Builder:
    """
    Neural Granger Causality builder using an MLP with Group Lasso over input groups.

    Given a panel of stock time series (T x N), it trains one predictor per target stock
    to forecast its next value from the lagged values of all stocks. Group Lasso forces
    sparsity at the stock-group level, revealing Granger causal parents.
    """

    def __init__(self, config: Optional[NGCConfig] = None) -> None:
        self.config = config or NGCConfig()

    @staticmethod
    def build_design_matrix(values: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build lagged design matrix X and aligned targets Y for next-step prediction.

        values: array of shape [T, N]
        returns X of shape [T - lags, N * lags] and Y of shape [T - lags, N]
        where row t contains lags [t-lags, t) in ascending time order for each stock,
        flattened stock-major (grouped by stock, each group has size lags).
        """
        if values.ndim != 2:
            raise ValueError("values must be 2D [time, stocks]")
        num_time, num_stocks = values.shape
        if num_time <= lags:
            raise ValueError("Not enough time steps to build lagged matrix")

        design_rows: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        for t in range(lags, num_time):
            window = values[t - lags : t, :]  # [lags, N]
            # group by stock: for each stock, take its lags in time order
            grouped = []
            for stock_index in range(num_stocks):
                grouped.append(window[:, stock_index])  # [lags]
            row = np.concatenate(grouped, axis=0)  # [N*lags]
            design_rows.append(row)
            targets.append(values[t, :])  # next-step values for all stocks

        X = np.stack(design_rows, axis=0)
        Y = np.stack(targets, axis=0)
        return X, Y

    def _train_single_target(
        self,
        design_X: np.ndarray,
        target_y: np.ndarray,
        num_stocks: int,
        lags: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Train an MLP for one target with Group Lasso penalty on input groups.
        Returns (group_norms, best_val_loss).
        """
        device = torch.device(self.config.device)

        # Standardize X and y for stability
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_std = x_scaler.fit_transform(design_X)
        y_std = y_scaler.fit_transform(target_y.reshape(-1, 1)).reshape(-1)

        X_tensor = torch.from_numpy(X_std).float()
        y_tensor = torch.from_numpy(y_std).float().unsqueeze(-1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

        model = SimpleMLP(
            input_dim=design_X.shape[1],
            hidden_dim=self.config.hidden_dim,
            dropout_p=self.config.dropout_p,
            output_dim=1,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        mse_loss = nn.MSELoss()

        best_val_loss = math.inf
        epochs_no_improve = 0

        # Simple holdout split: last 10% as validation
        num_samples = X_tensor.shape[0]
        split_index = int(num_samples * 0.9)
        train_dataset = TensorDataset(X_tensor[:split_index], y_tensor[:split_index])
        val_dataset = TensorDataset(X_tensor[split_index:], y_tensor[split_index:])
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        for epoch in range(self.config.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_X)
                loss = mse_loss(preds, batch_y)

                # Group Lasso penalty on first layer weights
                first_layer = None
                for module in model.feature_extractor:
                    if isinstance(module, nn.Linear):
                        first_layer = module
                        break
                if first_layer is None:
                    raise RuntimeError("First Linear layer not found in feature_extractor")

                penalty = compute_group_lasso_penalty(
                    first_layer_weight=first_layer.weight,
                    group_size=lags,
                    num_groups=num_stocks,
                )
                total_loss = loss + self.config.group_lasso_alpha * penalty

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                for val_X, val_y in val_loader:
                    val_X = val_X.to(device)
                    val_y = val_y.to(device)
                    val_pred = model(val_X)
                    val_losses.append(mse_loss(val_pred, val_y).item())
                mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
            if mean_val + 1e-8 < best_val_loss:
                best_val_loss = mean_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stop_patience:
                    break

        # Extract group norms from trained first layer
        first_layer = None
        for module in model.feature_extractor:
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            raise RuntimeError("First Linear layer not found in feature_extractor")

        with torch.no_grad():
            group_norms = []
            for group_index in range(num_stocks):
                start = group_index * lags
                end = start + lags
                block = first_layer.weight[:, start:end]
                group_norm = torch.linalg.matrix_norm(block, ord="fro").item()
                group_norms.append(group_norm)
        return np.asarray(group_norms, dtype=np.float32), float(best_val_loss)

    def build_adjacency(
        self,
        values: np.ndarray,
        stock_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build directed adjacency matrix A (N x N) where A[i, j] indicates strength of
        Granger causality from stock i (cause) to stock j (effect).

        Returns (A, thresholds) where thresholds[j] is the keep threshold used for target j.
        """
        if values.ndim != 2:
            raise ValueError("values must be 2D [time, stocks]")
        num_time, num_stocks = values.shape
        lags = self.config.lags
        X, Y = self.build_design_matrix(values, lags)

        adjacency = np.zeros((num_stocks, num_stocks), dtype=np.float32)
        thresholds = np.zeros((num_stocks,), dtype=np.float32)

        for target_index in range(num_stocks):
            target_y = Y[:, target_index]
            group_norms, _ = self._train_single_target(
                design_X=X,
                target_y=target_y,
                num_stocks=num_stocks,
                lags=lags,
            )
            # Normalize group norms to [0,1] per target for comparability
            if np.max(group_norms) > 0:
                normalized = group_norms / (np.max(group_norms) + 1e-12)
            else:
                normalized = group_norms
            perc = np.percentile(normalized, self.config.threshold_percentile)
            thresholds[target_index] = float(perc)
            adjacency[:, target_index] = (normalized >= perc).astype(np.float32) * normalized

        return adjacency, thresholds


def build_dynamic_graphs(
    values: pd.DataFrame,
    window_size: int,
    step_size: int,
    ngc_config: Optional[NGCConfig] = None,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Build a dictionary of directed adjacency matrices over rolling windows.

    values: DataFrame indexed by datetime with stock columns.
    window_size: number of rows per window (e.g., 252 trading days)
    step_size: stride between windows (e.g., 21 trading days)

    Returns: {window_end_timestamp: adjacency_matrix}
    """
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("values must have a DatetimeIndex")
    if values.shape[0] < window_size:
        raise ValueError("Not enough rows for the first window")

    builder = NGC_Builder(config=ngc_config)
    graphs: Dict[pd.Timestamp, np.ndarray] = {}

    start_idx = 0
    while start_idx + window_size <= values.shape[0]:
        end_idx = start_idx + window_size
        window_df = values.iloc[start_idx:end_idx]
        adjacency, _ = builder.build_adjacency(window_df.values)
        window_end_ts = window_df.index[-1]
        graphs[window_end_ts] = adjacency
        start_idx += step_size
    return graphs

