from __future__ import annotations

import argparse
import numpy as np
import torch

from fda.rl.envs.market_env import MarketEnv
from fda.rl.algo.dqn import DQNAgent
from fda.rl.xrl.explainer import XRLExplainer
from fda.models.predictor import Predictor
from fda.market.milan import MarketModel


def _build_adj_indices_weights(N: int):
    # fully-connected directed graph without self-loops
    src = []
    dst = []
    w = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src.append(i); dst.append(j); w.append(1.0)
    adj_indices = torch.tensor(list(zip(src, dst)), dtype=torch.long)
    adj_weights = torch.tensor(w, dtype=torch.float32)
    return adj_indices, adj_weights


def _rolling_feat(x: np.ndarray, t: int, window: int, reducer: str) -> np.ndarray:
    s = max(0, t - window + 1)
    seg = x[s: t + 1]
    if seg.size == 0:
        return np.zeros(x.shape[1], dtype=np.float32)
    if reducer == 'mean':
        return seg.mean(axis=0).astype(np.float32)
    if reducer == 'std':
        return seg.std(axis=0, ddof=0).astype(np.float32)
    if reducer == 'sum':
        return seg.sum(axis=0).astype(np.float32)
    return seg[-1].astype(np.float32)


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

    # Predictor and MarketModel setup
    seq_in_feat = 1
    seq_T = 32
    cond_dim = 1
    node_feat_dim = 64
    predictor = Predictor(seq_in_feat=seq_in_feat, cond_dim=cond_dim, node_feat_dim=node_feat_dim)

    stock_dim = 4
    macro_dim = 3
    probe_dim = 2
    intent_dim = 1
    market_model = MarketModel(stock_dim=stock_dim, macro_dim=macro_dim, probe_dim=probe_dim, intent_dim=intent_dim)

    # adjacency for RGAT
    adj_indices, adj_weights = _build_adj_indices_weights(args.N)

    # build initial state (t=0) to size agent
    obs = env.reset()
    t = 0
    # predictor inputs at t
    r_hist = returns[max(0, t - (seq_T - 1)): t + 1]
    if r_hist.shape[0] < seq_T:
        pad = np.zeros((seq_T - r_hist.shape[0], args.N), dtype=np.float32)
        r_hist = np.concatenate([pad, r_hist], axis=0)
    x_seq = torch.from_numpy(r_hist.T[:, None, :].astype(np.float32))  # [N, 1, T]
    x_node = torch.zeros(args.N, node_feat_dim, dtype=torch.float32)   # placeholder
    cond = torch.zeros(args.N, cond_dim, dtype=torch.float32)
    pred_out = predictor(x_seq, x_node, cond, adj_indices, adj_weights)
    rl_state = pred_out['rl_state']
    # market inputs at t
    stock_features = np.stack([
        _rolling_feat(returns, t, 5, 'mean'),
        _rolling_feat(returns, t, 10, 'std'),
        adv[t].astype(np.float32),
        _rolling_feat(prices, t, 5, 'mean'),
    ], axis=-1).astype(np.float32)  # [N, 4]
    macro_state = np.array([
        returns[: t + 1].mean() if t >= 0 else 0.0,
        returns[: t + 1].std(ddof=0) if t >= 0 else 0.0,
        float(t) / max(1, args.T - 1),
    ], dtype=np.float32)  # [3]
    # impact potential: z-adv (window 60), rv10
    # compute z-adv by simple normalization over window
    s = max(0, t - 59)
    base_adv = adv[s: t + 1]
    mu_adv = base_adv.mean(axis=0)
    sd_adv = base_adv.std(axis=0, ddof=0) + 1e-8
    z_adv = ((adv[t] - mu_adv) / sd_adv).astype(np.float32)
    rv10 = _rolling_feat(returns, t, 10, 'std')
    impact_potential = np.stack([z_adv, rv10], axis=-1)  # [N,2]
    # trading intention from current position to one-hot (placeholder until action chosen)
    trading_intention = np.zeros((args.N, 1), dtype=np.float32)
    m_out = market_model(
        torch.from_numpy(stock_features[None, ...]),
        torch.from_numpy(macro_state[None, ...]),
        torch.from_numpy(impact_potential[None, ...]),
        torch.from_numpy(trading_intention[None, ...]),
    )
    # assemble observation vector for DQN (late fusion)
    def _flatten_state(pred: dict, market: dict) -> np.ndarray:
        feats = [
            pred['alpha_mu'].detach().numpy(),
            pred['alpha_q'].detach().numpy().reshape(-1),
            pred['alpha_uncertainty'].detach().numpy(),
            pred['styles'].detach().numpy().reshape(-1),
            market['z_macro'].detach().numpy().reshape(-1),
            market['risk_metrics']['rv_mean'].detach().numpy(),
            market['risk_metrics']['zvol_mean'].detach().numpy(),
            market['r_norm'].detach().numpy().reshape(-1),
            market['r_impact'].detach().numpy().reshape(-1),
            market['impact_costs'].detach().numpy().reshape(-1),
        ]
        return np.concatenate([f.reshape(-1) for f in feats], axis=0).astype(np.float32)

    s0 = _flatten_state(rl_state, m_out)

    # DQN agent sized by fused state
    action_dim = args.N
    state_dim = int(s0.shape[0])
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    # SHAP background: sample unit vectors of state space (small demo)
    bg = np.eye(state_dim, dtype=np.float32)[: min(50, state_dim)]
    explainer = XRLExplainer(agent.q, bg)

    obs = env.reset()
    feature_names = [f'f{i}' for i in range(state_dim)]
    for t in range(args.T - 1):
        # predictor inputs at t
        r_hist = returns[max(0, t - (seq_T - 1)): t + 1]
        if r_hist.shape[0] < seq_T:
            pad = np.zeros((seq_T - r_hist.shape[0], args.N), dtype=np.float32)
            r_hist = np.concatenate([pad, r_hist], axis=0)
        x_seq = torch.from_numpy(r_hist.T[:, None, :].astype(np.float32))
        x_node = torch.zeros(args.N, node_feat_dim, dtype=torch.float32)
        cond = torch.zeros(args.N, cond_dim, dtype=torch.float32)
        pred_out = predictor(x_seq, x_node, cond, adj_indices, adj_weights)
        rl_state = pred_out['rl_state']

        # market inputs at t
        stock_features = np.stack([
            _rolling_feat(returns, t, 5, 'mean'),
            _rolling_feat(returns, t, 10, 'std'),
            adv[t].astype(np.float32),
            _rolling_feat(prices, t, 5, 'mean'),
        ], axis=-1).astype(np.float32)
        macro_state = np.array([
            returns[max(0, t - 20): t + 1].mean(),
            returns[max(0, t - 20): t + 1].std(ddof=0),
            float(t) / max(1, args.T - 1),
        ], dtype=np.float32)
        s60 = max(0, t - 59)
        base_adv = adv[s60: t + 1]
        mu_adv = base_adv.mean(axis=0)
        sd_adv = base_adv.std(axis=0, ddof=0) + 1e-8
        z_adv = ((adv[t] - mu_adv) / sd_adv).astype(np.float32)
        rv10 = _rolling_feat(returns, t, 10, 'std')
        impact_potential = np.stack([z_adv, rv10], axis=-1)

        # agent action
        # build fused observation for acting
        market_out_for_act = market_model(
            torch.from_numpy(stock_features[None, ...]),
            torch.from_numpy(macro_state[None, ...]),
            torch.from_numpy(impact_potential[None, ...]),
            torch.from_numpy(np.zeros((1, args.N, intent_dim), dtype=np.float32)),
        )
        s_vec = _flatten_state(rl_state, market_out_for_act)
        a = agent.act(s_vec, eps=args.eps)

        # convert action to weights (one-hot) and intention vector
        w = np.zeros(args.N, dtype=np.float32); w[a] = 1.0
        dq = w - obs['position'].astype(np.float32)
        adv_ratio = np.abs(dq) / (np.maximum(adv[t], 1e-8))
        intent = np.sign(dq) * adv_ratio

        # get market outputs with intention (costs become action-sensitive)
        market_out = market_model(
            torch.from_numpy(stock_features[None, ...]),
            torch.from_numpy(macro_state[None, ...]),
            torch.from_numpy(impact_potential[None, ...]),
            torch.from_numpy(intent[None, :, None].astype(np.float32)),
        )

        # env step to get raw return and env cost
        obs_next, pnl_with_env_cost, done, info = env.step(w)
        pnl_raw = pnl_with_env_cost + float(info.get('cost', 0.0))
        milan_cost = market_out['impact_costs'].detach().numpy().reshape(-1).sum().item()
        r = pnl_raw - milan_cost

        # next fused observation for learning target (use zero intention again)
        market_out_next = market_model(
            torch.from_numpy(stock_features[None, ...]),
            torch.from_numpy(macro_state[None, ...]),
            torch.from_numpy(impact_potential[None, ...]),
            torch.from_numpy(np.zeros((1, args.N, intent_dim), dtype=np.float32)),
        )
        sp_vec = _flatten_state(rl_state, market_out_next)

        agent.buffer.push(s_vec, a, r, sp_vec, float(done))
        loss = agent.update(batch_size=args.batch)

        # XRL explanation (demo): explain current decision
        shap_vals = explainer.explain_decision(s_vec)  # [A, F]
        shap_a = shap_vals[a]
        idx = np.argsort(np.abs(shap_a))[::-1][:3]
        drivers = [(feature_names[i], float(shap_a[i])) for i in idx]
        print(f"t={t} a={a} pnl_raw={pnl_raw:.6f} milan_cost={milan_cost:.6f} r={r:.6f} drivers={drivers}")

        obs = obs_next
        if done:
            break
    print('Stage C DQN+XRL (integrated) run finished. Final cash:', obs['cash'])


if __name__ == '__main__':
    main()

