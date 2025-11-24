from __future__ import annotations

from collections import deque, namedtuple
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        value = self.value_head(feats)
        adv = self.adv_head(feats)
        return value + adv - adv.mean(dim=1, keepdim=True)


Transition = namedtuple('Transition', ['s', 'a', 'r', 'sp', 'd'])


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, sp, d):
        self.buf.append(Transition(s, a, r, sp, d))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buf, batch_size)
        return Transition(
            torch.stack([torch.as_tensor(b.s) for b in batch]),
            torch.as_tensor([b.a for b in batch], dtype=torch.long),
            torch.as_tensor([b.r for b in batch], dtype=torch.float32),
            torch.stack([torch.as_tensor(b.sp) for b in batch]),
            torch.as_tensor([b.d for b in batch], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q = QNetwork(state_dim, action_dim, hidden=hidden).to(self.device)
        self.targ = QNetwork(state_dim, action_dim, hidden=hidden).to(self.device)
        self.targ.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.buffer = ReplayBuffer()

    def act(self, s: torch.Tensor, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.q(s.to(self.device).unsqueeze(0))
        return int(torch.argmax(q, dim=-1).item())

    def td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        q_values = self.q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_online = self.q(next_states)
            next_actions = torch.argmax(next_q_online, dim=-1)
            next_q_target = self.targ(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * next_q_target
        loss = nn.functional.smooth_l1_loss(q_values, target)
        return loss

    def update(self, batch_size: int = 64) -> Optional[torch.Tensor]:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        loss = self.td_loss(batch.s, batch.a, batch.r, batch.sp, batch.d)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._soft_update()
        return loss.detach()

    def _soft_update(self):
        for targ_p, p in zip(self.targ.parameters(), self.q.parameters()):
            targ_p.data.copy_(targ_p.data * (1 - self.tau) + p.data * self.tau)

