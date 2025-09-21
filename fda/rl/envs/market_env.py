from __future__ import annotations

import numpy as np
import torch


class MarketEnv:
    def __init__(self, prices: np.ndarray, returns: np.ndarray, adv: np.ndarray, pseudo_industry: np.ndarray,
                 impact_params: dict, adv_cap: float = 0.1):
        self.prices = prices  # [T, N]
        self.returns = returns  # [T, N]
        self.adv = adv  # [T, N]
        self.cluster = pseudo_industry  # [N]
        self.adv_cap = adv_cap
        self.impact_params = impact_params
        self.T, self.N = returns.shape
        self.t = 0
        self.position = np.zeros(self.N)
        self.cash = 1.0

    def reset(self):
        self.t = 0
        self.position[:] = 0.0
        self.cash = 1.0
        return self._obs()

    def step(self, target_w: np.ndarray):
        # enforce simplex
        target_w = np.maximum(target_w, 0.0)
        if target_w.sum() > 0:
            target_w = target_w / target_w.sum()
        # turnover vector q (weight change)
        q = target_w - self.position
        # ADV cap (soft clip)
        cap = self.adv_cap * np.maximum(self.adv[self.t], 1e-8)
        q_dollar = q  # assume wealth=1
        q_dollar = np.clip(q_dollar, -cap, cap)
        # impact cost proxy (temporary + permanent)
        sigma = np.sqrt(np.maximum(1e-8, (self.returns[max(0,self.t-20):self.t+1]**2).mean(axis=0)))
        kappa = self.impact_params['kappa']
        alpha = self.impact_params['alpha']
        beta = self.impact_params['beta']
        temp = kappa * sigma * (np.abs(q_dollar) / (np.maximum(1e-8, self.adv[self.t]))) ** alpha
        perm = beta * np.abs(q_dollar)
        cost = (temp + perm).sum()
        # apply trade
        self.position = target_w
        # next day return
        r = (self.position * self.returns[self.t]).sum()
        pnl = r - cost
        self.cash *= (1.0 + pnl)
        self.t += 1
        done = self.t >= self.T - 1
        return self._obs(), pnl, done, {'cost': cost}

    def _obs(self):
        return {'t': self.t, 'position': self.position.copy(), 'cash': self.cash}

