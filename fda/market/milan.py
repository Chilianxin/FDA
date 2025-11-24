from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .regime import GaussianHMMTeacher, RegimeNet


# -------- Regime (HMM) --------


class MarketRegimeDetector:
    """
    Offline HMM teacher that produces pseudo labels for RegimeNet.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.teacher = GaussianHMMTeacher(n_components=n_components, random_state=random_state)

    def train(self, index_returns: np.ndarray) -> None:
        self.teacher.fit(index_returns)

    def predict_proba(self, latest_returns: np.ndarray) -> np.ndarray:
        post = self.teacher.posterior(latest_returns)
        if post is None:
            raise RuntimeError("MarketRegimeDetector must be trained before calling predict_proba.")
        return post.numpy()

    @property
    def model(self) -> GaussianHMMTeacher:
        return self.teacher


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


# -------- Impact Propagation Transformer (Macro-aware) --------


class MicroEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImpactPropagationTransformer(nn.Module):
    """
    Macro-aware transformer encoder that performs cross-attention between per-stock features and regime probabilities.
    """

    def __init__(
        self,
        stock_dim: int,
        macro_dim: int,
        probe_dim: int,
        intent_dim: int,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stock_proj = nn.Linear(stock_dim + probe_dim + intent_dim, model_dim)
        self.macro_proj = nn.Linear(macro_dim, model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_probs: torch.Tensor,
        micro_latent: torch.Tensor,
        trading_intention_vector: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = stock_features.shape
        inputs = torch.cat([stock_features, micro_latent, trading_intention_vector], dim=-1)
        x = self.stock_proj(inputs)
        macro_token = self.macro_proj(macro_probs).unsqueeze(1)  # [B,1,D]
        attn_out, _ = self.cross_attn(x, macro_token, macro_token)
        x = self.norm(x + attn_out)
        h = self.encoder(x)
        return h


class ImpactPredictionHead(nn.Module):
    """
    Produces a multiplicative correction factor γ \in [min_scale, max_scale] for physics-informed costs.
    """

    def __init__(self, model_dim: int = 128, min_scale: float = 0.5, max_scale: float = 2.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(h)
        return self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(raw)


class MILAN(nn.Module):
    def __init__(
        self,
        stock_dim: int,
        macro_dim: int,
        probe_dim: int,
        intent_dim: int,
        model_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ImpactPropagationTransformer(
            stock_dim=stock_dim,
            macro_dim=macro_dim,
            probe_dim=probe_dim,
            intent_dim=intent_dim,
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            dropout=dropout,
        )
        self.head = ImpactPredictionHead(model_dim)

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_probs: torch.Tensor,
        micro_latent: torch.Tensor,
        trading_intention_vector: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(stock_features, macro_probs, micro_latent, trading_intention_vector)
        gamma = self.head(h)
        return gamma


class MarketModel(nn.Module):
    """
    Physics-informed cost module with HMM teacher-student and differentiable macro perception.
    """

    def __init__(
        self,
        stock_dim: int,
        macro_in_dim: int,
        probe_dim: int,
        intent_dim: int,
        num_regimes: int = 3,
        model_dim: int = 128,
        micro_latent_dim: int = 32,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        kappa: float = 1.0,
    ):
        super().__init__()
        self.regime_student = RegimeNet(macro_in_dim, hidden=model_dim, num_regimes=num_regimes, dropout=dropout)
        self.micro_encoder = MicroEncoder(probe_dim, hidden=model_dim // 2, latent_dim=micro_latent_dim, dropout=dropout)
        self.milan = MILAN(
            stock_dim=stock_dim,
            macro_dim=num_regimes,
            probe_dim=micro_latent_dim,
            intent_dim=intent_dim,
            model_dim=model_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        self.kappa = kappa
        self.eps = 1e-8

    def _physics_base_cost(self, sigma: torch.Tensor, adv: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(self.eps)
        adv = adv.clamp_min(self.eps)
        intent_abs = intent.abs()
        return self.kappa * sigma * torch.sqrt(intent_abs / adv)

    def forward(
        self,
        stock_features: torch.Tensor,
        macro_features: torch.Tensor,
        impact_potential_vector: torch.Tensor,
        trading_intention_vector: torch.Tensor,
        adv: torch.Tensor,
        realized_vol: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            stock_features: [B, N, F_s]
            macro_features: [B, F_macro]
            impact_potential_vector: [B, N, F_probe]
            trading_intention_vector: [B, N, F_intent]
            adv: [B, N]
            realized_vol: [B, N]
            teacher_probs: optional teacher regime probabilities for KD
        """
        macro_out = self.regime_student(macro_features)
        z_macro = macro_out['probs']
        micro_latent = self.micro_encoder(impact_potential_vector)
        if trading_intention_vector.dim() == 2:
            trading_intention_vector = trading_intention_vector.unsqueeze(-1)
        gamma = self.milan(stock_features, z_macro, micro_latent, trading_intention_vector).squeeze(-1)
        base_cost = self._physics_base_cost(realized_vol, adv, trading_intention_vector.squeeze(-1))
        impact_costs = base_cost * gamma
        kd_loss = RegimeNet.kd_loss(macro_out['logits'], teacher_probs) if teacher_probs is not None else macro_out['logits'].sum() * 0.0
        risk_metrics = {
            'rv_mean': impact_potential_vector[..., 1].mean(dim=1),
            'zvol_mean': impact_potential_vector[..., 0].mean(dim=1),
            'intent_l2': trading_intention_vector.squeeze(-1).pow(2).mean(dim=1),
        }
        return {
            'z_macro': z_macro,
            'regime_logits': macro_out['logits'],
            'regime_kd_loss': kd_loss,
            'risk_metrics': risk_metrics,
            'impact_costs': impact_costs,
            'base_cost': base_cost,
            'gamma': gamma,
            'micro_latent': micro_latent,
            'micro_stats': impact_potential_vector,
        }

