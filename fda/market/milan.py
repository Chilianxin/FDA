from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -------- Regime (HMM) --------


class MarketRegimeDetector:
    def __init__(self, n_components: int = 3, random_state: int = 42):
        # Delayed import to keep dependency optional at import time
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, covariance_type='full', random_state=random_state)

    def train(self, index_returns: np.ndarray) -> None:
        # index_returns: [T] or [T,1]
        x = np.asarray(index_returns).reshape(-1, 1).astype(float)
        self.model.fit(x)

    def predict_proba(self, latest_returns: np.ndarray) -> np.ndarray:
        # latest_returns: [T] used for posterior smoothing; returns last-step state probabilities
        x = np.asarray(latest_returns).reshape(-1, 1).astype(float)
        logprob, post = self.model.score_samples(x)  # post: [T, K]
        return post[-1]


# -------- Micro Liquidity Probe --------


@dataclass
class ProbeConfig:
    vol_window: int = 60
    rv_window: int = 10


def micro_liquidity_probe(
    vol: np.ndarray,
    amount: np.ndarray,
    returns: np.ndarray,
    cfg: Optional[ProbeConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-asset impact potential: volume/amount Z-score and realized volatility.

    Inputs are panel arrays aligned by time: [T, N]
    Returns (impact_potential, mask)
    - impact_potential: [T, N, 2] features: [z_volume, rv_10d]
    - mask: [T, N] valid indicators
    """
    cfg = cfg or ProbeConfig()
    T, N = vol.shape
    # rolling z-score of volume
    z_vol = np.zeros_like(vol, dtype=float)
    for t in range(T):
        s = max(0, t - cfg.vol_window + 1)
        base = vol[s : t + 1]
        m = base.mean(axis=0)
        sd = base.std(axis=0, ddof=0) + 1e-8
        z_vol[t] = (vol[t] - m) / sd
    # realized volatility over returns
    rv = np.zeros_like(returns, dtype=float)
    for t in range(T):
        s = max(0, t - cfg.rv_window + 1)
        base = returns[s : t + 1]
        rv[t] = base.std(axis=0, ddof=0)
    feat = np.stack([z_vol, rv], axis=-1)  # [T, N, 2]
    mask = np.isfinite(feat).all(axis=-1)
    feat[~np.isfinite(feat)] = 0.0
    return feat, mask


# -------- Impact Propagation Transformer --------


class ImpactPropagationTransformer(nn.Module):
    def __init__(self, stock_dim: int, macro_dim: int, probe_dim: int, intent_dim: int, model_dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb_stock = nn.Linear(stock_dim, model_dim)
        self.emb_macro = nn.Linear(macro_dim, model_dim)
        self.emb_probe = nn.Linear(probe_dim, model_dim)
        self.emb_intent = nn.Linear(intent_dim, model_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_state_vector: torch.Tensor,
        impact_potential_vector: torch.Tensor,
        trading_intention_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
          - stock_features: [B, N, F_s]
          - macro_state_vector: [B, F_m] (broadcast per stock)
          - impact_potential_vector: [B, N, F_p]
          - trading_intention_vector: [B, N, F_i]
        Returns:
          - encoded: [B, N, D]
        """
        B, N, _ = stock_features.shape
        e_s = self.emb_stock(stock_features)
        e_m = self.emb_macro(macro_state_vector).unsqueeze(1).expand(B, N, -1)
        e_p = self.emb_probe(impact_potential_vector)
        e_i = self.emb_intent(trading_intention_vector)
        x = e_s + e_m + e_p + e_i  # [B, N, D]
        h = self.encoder(x)
        return h


class ImpactPredictionHead(nn.Module):
    def __init__(self, model_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2), nn.GELU(), nn.Linear(model_dim * 2, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D] -> costs: [B, N]
        out = self.mlp(h).squeeze(-1)
        return out


class MILAN(nn.Module):
    def __init__(self, stock_dim: int, macro_dim: int, probe_dim: int, intent_dim: int, model_dim: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = ImpactPropagationTransformer(stock_dim, macro_dim, probe_dim, intent_dim, model_dim, layers, heads, dropout)
        self.head = ImpactPredictionHead(model_dim)

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_state_vector: torch.Tensor,
        impact_potential_vector: torch.Tensor,
        trading_intention_vector: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(stock_features, macro_state_vector, impact_potential_vector, trading_intention_vector)
        costs = self.head(h)
        return costs


class MacroProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, max(out_dim, 8)), nn.GELU(), nn.Linear(max(out_dim, 8), out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineReturnHead(nn.Module):
    def __init__(self, stock_dim: int, macro_dim: int, probe_dim: int, hidden: int = 128, return_scale: float = 0.05):
        super().__init__()
        self.return_scale = return_scale
        in_dim = stock_dim + macro_dim + probe_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, stock_features: torch.Tensor, macro_state_vector: torch.Tensor, impact_potential_vector: torch.Tensor) -> torch.Tensor:
        # stock_features: [B, N, F_s], macro_state_vector: [B, F_m], impact_potential_vector: [B, N, F_p]
        B, N, _ = stock_features.shape
        m = macro_state_vector.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([stock_features, m, impact_potential_vector], dim=-1)
        r = self.mlp(x).squeeze(-1)
        # squash to a reasonable daily return range
        return torch.tanh(r) * self.return_scale


class MarketModel(nn.Module):
    """
    Wrapper to produce RL-ready market state with late fusion fields.

    forward(...) returns dict with keys:
      - z_macro: [B, Dz]
      - regime_probs: [B, K] or None if not provided
      - risk_metrics: dict with aggregates (e.g., rv_mean, zvol_mean) as [B]
      - impact_costs: [B, N]
    """

    def __init__(self, stock_dim: int, macro_dim: int, probe_dim: int, intent_dim: int, model_dim: int = 128, macro_z_dim: int = 16, layers: int = 2, heads: int = 4, dropout: float = 0.1, impact_to_return_scale: float = 1.0, baseline_return_scale: float = 0.05):
        super().__init__()
        self.milan = MILAN(stock_dim, macro_dim, probe_dim, intent_dim, model_dim, layers, heads, dropout)
        self.macro_proj = MacroProjector(macro_dim, out_dim=macro_z_dim)
        self.baseline = BaselineReturnHead(stock_dim, macro_dim, probe_dim, hidden=model_dim, return_scale=baseline_return_scale)
        self.impact_to_return_scale = impact_to_return_scale

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_state_vector: torch.Tensor,
        impact_potential_vector: torch.Tensor,
        trading_intention_vector: torch.Tensor,
        regime_probs: torch.Tensor | None = None,
    ) -> dict:
        z_macro = self.macro_proj(macro_state_vector)
        impact_costs = self.milan(stock_features, macro_state_vector, impact_potential_vector, trading_intention_vector)  # [B, N]
        # baseline expected (normal) return per asset
        r_norm = self.baseline(stock_features, macro_state_vector, impact_potential_vector)  # [B, N]
        # derive direction from intention's first channel if available
        if trading_intention_vector.dim() == 3 and trading_intention_vector.size(-1) > 0:
            intent_dir = torch.tanh(trading_intention_vector[..., 0])  # [-1,1]
        else:
            intent_dir = torch.zeros_like(impact_costs)
        # map costs to expected return impact (negative sign for cost)
        r_impact = - self.impact_to_return_scale * impact_costs * intent_dir
        r_total = r_norm + r_impact
        # simple aggregates as risk metrics
        rv_mean = impact_potential_vector[..., 1].mean(dim=1)  # [B]
        zvol_mean = impact_potential_vector[..., 0].mean(dim=1)  # [B]
        out = {
            'z_macro': z_macro,
            'regime_probs': regime_probs,
            'risk_metrics': {
                'rv_mean': rv_mean,
                'zvol_mean': zvol_mean,
            },
            'impact_costs': impact_costs,
            'r_norm': r_norm,
            'r_impact': r_impact,
            'r_total_pred': r_total,
        }
        return out

