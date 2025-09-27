from __future__ import annotations

import argparse
import numpy as np
import torch

from fda.rl.envs.market_env import MarketEnv
from fda.rl.algo.dqn import DQNAgent
from fda.rl.xrl.explainer import XRLExplainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    # minimal simulation with random returns and adv
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, size=(args.T, args.N))
    prices = np.cumprod(1 + returns, axis=0)
    adv = rng.uniform(1e-3, 1e-2, size=(args.T, args.N))
    pseudo_industry = rng.integers(0, 3, size=(args.N,))
    impact_params = {'kappa': 1e-3, 'alpha': 0.7, 'beta': 1e-4}
    env = MarketEnv(prices, returns, adv, pseudo_industry, impact_params, adv_cap=0.1)

    # discrete action space: choose among N one-hot target assets for simplicity
    action_dim = args.N
    state_dim = args.N  # use current weights as toy state
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    # background samples for SHAP (random states)
    bg = np.eye(state_dim, dtype=np.float32)[: min(50, state_dim)]
    explainer = XRLExplainer(agent.q, bg)

    obs = env.reset()
    for t in range(args.T - 1):
        s = obs['position'].astype(np.float32)
        a = agent.act(s, eps=args.eps)
        # convert discrete action to target weights: one-hot
        w = np.zeros(args.N, dtype=np.float32); w[a] = 1.0
        obs_next, pnl, done, info = env.step(w)
        r = pnl
        sp = obs_next['position'].astype(np.float32)
        agent.buffer.push(s, a, r, sp, float(done))
        loss = agent.update(batch_size=args.batch)

        # XRL explanation (demo): explain current decision
        shap_vals = explainer.explain_decision(s)
        # select action's shap values for potential logging/printing
        shap_a = shap_vals[a]

        obs = obs_next
        if done:
            break
    print('Stage C DQN+XRL minimal run finished. Final cash:', obs['cash'])


if __name__ == '__main__':
    main()

