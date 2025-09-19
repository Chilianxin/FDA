from __future__ import annotations

import argparse
import numpy as np
import torch

from fda.rl.envs.market_env import MarketEnv
from fda.rl.algo.ppo_lagrangian import PPOLagrangian


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--N', type=int, default=50)
    args = ap.parse_args()

    # minimal simulation with random returns and adv
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, size=(args.T, args.N))
    prices = np.cumprod(1 + returns, axis=0)
    adv = rng.uniform(1e-3, 1e-2, size=(args.T, args.N))
    pseudo_industry = rng.integers(0, 10, size=(args.N,))
    impact_params = {'kappa': 1e-3, 'alpha': 0.7, 'beta': 1e-4}
    env = MarketEnv(prices, returns, adv, pseudo_industry, impact_params, adv_cap=0.1)

    agent = PPOLagrangian(n_assets=args.N, lr=3e-4)

    obs = env.reset()
    for t in range(args.T - 1):
        # simple state: current weights
        s = torch.tensor(obs['position'], dtype=torch.float32)
        w = agent.actor(s).detach().numpy()
        obs_next, pnl, done, info = env.step(w)
        # dummy returns/advantages
        ret = torch.tensor([pnl], dtype=torch.float32)
        adv = ret.clone()
        agent.step(s.unsqueeze(0), ret, adv)
        obs = obs_next
        if done:
            break
    print('Stage C minimal RL run finished. Final cash:', obs['cash'])


if __name__ == '__main__':
    main()

