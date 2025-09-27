from __future__ import annotations

from collections import deque, namedtuple
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = namedtuple('Transition', ['s', 'a', 'r', 'sp', 'd'])


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, sp, d):
        self.buf.append(Transition(s, a, r, sp, d))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buf, batch_size)
        s = torch.stack([torch.as_tensor(b.s) for b in batch])
        a = torch.as_tensor([b.a for b in batch], dtype=torch.long)
        r = torch.as_tensor([b.r for b in batch], dtype=torch.float32)
        sp = torch.stack([torch.as_tensor(b.sp) for b in batch])
        d = torch.as_tensor([b.d for b in batch], dtype=torch.float32)
        return Transition(s, a, r, sp, d)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, tau: float = 0.005):
        self.q = QNetwork(state_dim, action_dim)
        self.targ = QNetwork(state_dim, action_dim)
        self.targ.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.buffer = ReplayBuffer()

    def act(self, s: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.q(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q, dim=-1).item())

    def update(self, batch_size: int = 64):
        if len(self.buffer) < batch_size:
            return None
        tr = self.buffer.sample(batch_size)
        q = self.q(tr.s)
        q_sa = q.gather(1, tr.a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.targ(tr.sp).max(dim=1).values
            y = tr.r + self.gamma * (1.0 - tr.d) * q_next
        loss = nn.functional.mse_loss(q_sa, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        # soft update
        for tp, p in zip(self.targ.parameters(), self.q.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
        return loss.item()

