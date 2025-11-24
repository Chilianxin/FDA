from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeNet(nn.Module):
    """
    Student network that distills HMM teacher posteriors into a differentiable macro-state representation.
    """

    def __init__(self, in_dim: int, hidden: int = 64, num_regimes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_regimes = num_regimes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_regimes),
        )

    def forward(self, macro_features: torch.Tensor) -> dict:
        logits = self.mlp(macro_features)
        probs = torch.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}

    @staticmethod
    def kd_loss(student_log_probs: torch.Tensor, teacher_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        KL divergence between teacher distribution and student logits.
        """
        if teacher_probs is None:
            return student_log_probs.new_tensor(0.0)
        teacher = (teacher_probs / temperature).clamp_min(1e-8)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        student = (student_log_probs / temperature).log_softmax(dim=-1)
        return F.kl_div(student, teacher, reduction='batchmean') * (temperature ** 2)


@dataclass
class GaussianHMMTeacher:
    """
    Thin wrapper over hmmlearn GaussianHMM to keep dependency optional.
    """

    n_components: int = 3
    random_state: int = 42

    def __post_init__(self):
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state,
        )
        self._is_trained = False

    def fit(self, index_returns: torch.Tensor | 'np.ndarray') -> None:
        import numpy as np

        data = torch.as_tensor(index_returns, dtype=torch.float32).cpu().numpy()
        data = data.reshape(-1, 1).astype(np.float64)
        self.model.fit(data)
        self._is_trained = True

    def posterior(self, recent_returns: torch.Tensor | 'np.ndarray') -> Optional[torch.Tensor]:
        if not self._is_trained:
            return None
        import numpy as np

        data = torch.as_tensor(recent_returns, dtype=torch.float32).cpu().numpy()
        data = data.reshape(-1, 1).astype(np.float64)
        _, post = self.model.score_samples(data)
        return torch.from_numpy(post[-1]).float()

