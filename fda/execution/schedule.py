from __future__ import annotations

import torch


def soft_schedule(q: torch.Tensor, H: int = 5, logits: torch.Tensor | None = None) -> torch.Tensor:
    # q: [N]; return per-day schedule [H, N]
    if logits is None:
        logits = torch.zeros(H, device=q.device)
    w = torch.softmax(logits, dim=0)  # [H]
    return w.unsqueeze(1) * q.unsqueeze(0)

