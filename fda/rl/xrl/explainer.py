from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class XRLExplainer:
    def __init__(self, q_network, background_states: np.ndarray, framework: str = 'torch'):
        import shap  # lazy import; raises if missing
        self.framework = framework
        self.q = q_network
        self.background = background_states.astype(np.float32)
        if framework == 'torch':
            self.explainer = shap.DeepExplainer(self.q, torch.as_tensor(self.background))
        else:
            self.explainer = shap.KernelExplainer(lambda x: self.q(torch.as_tensor(x, dtype=torch.float32)).detach().numpy(), self.background)

    def explain_decision(self, state: np.ndarray) -> np.ndarray:
        import shap
        x = state.astype(np.float32).reshape(1, -1)
        shap_values = self.explainer.shap_values(torch.as_tensor(x) if self.framework == 'torch' else x)
        # shap returns list per output for some backends; unify to [A, F]
        if isinstance(shap_values, list):
            arr = np.stack([np.array(sv).reshape(1, -1) for sv in shap_values], axis=0)  # [A, 1, F]
            return arr[:, 0, :]
        else:
            return np.array(shap_values)  # [A, F]

    @staticmethod
    def format_explanation(shap_for_action: np.ndarray, feature_names: List[str], top_k: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        idx_sorted = np.argsort(np.abs(shap_for_action))[::-1]
        pos = []
        neg = []
        for i in idx_sorted:
            val = shap_for_action[i]
            name = feature_names[i] if i < len(feature_names) else f'f{i}'
            if val >= 0 and len(pos) < top_k:
                pos.append((name, float(val)))
            elif val < 0 and len(neg) < top_k:
                neg.append((name, float(val)))
            if len(pos) >= top_k and len(neg) >= top_k:
                break
        return pos, neg

