from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, n_assets: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_assets)
        )

    def forward(self, x):
        logits = self.net(x)
        w = torch.softmax(logits, dim=-1)
        return w


class Critic(nn.Module):
    def __init__(self, n_assets: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(n_assets, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.v(x).squeeze(-1)


class PPOLagrangian:
    def __init__(self, n_assets: int, lr: float = 3e-4):
        self.actor = Actor(n_assets)
        self.critic = Critic(n_assets)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def step(self, obs, returns, advantages):
        # minimal supervision: obs is current weights; advantages computed externally
        w = self.actor(obs)
        v = self.critic(obs)
        # dummy PPO objective (placeholder): maximize advantages * log pi(w)
        logp = torch.sum(torch.log(w + 1e-8) * w, dim=-1)
        actor_loss = -(advantages * logp).mean()
        critic_loss = ((returns - v) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {'loss': loss.item(), 'actor': actor_loss.item(), 'critic': critic_loss.item()}

