from __future__ import annotations

import argparse
import numpy as np
import torch

from fda.graphs.ngc_builder import NGCConfig, rolling_dynamic_ngc
from fda.market.milan import MarketModel
from fda.models.predictor import Predictor
from fda.rl.algo.dqn import DQNAgent
from fda.rl.envs.market_env import MarketEnv
from fda.rl.xrl.explainer import XRLExplainer


def _rolling_feat(x: np.ndarray, t: int, window: int, reducer: str) -> np.ndarray:
    start = max(0, t - window + 1)
    seg = x[start : t + 1]
    if seg.size == 0:
        return np.zeros(x.shape[1], dtype=np.float32)
    if reducer == 'mean':
        return seg.mean(axis=0).astype(np.float32)
    if reducer == 'std':
        return seg.std(axis=0, ddof=0).astype(np.float32)
    if reducer == 'sum':
        return seg.sum(axis=0).astype(np.float32)
    return seg[-1].astype(np.float32)


def _pad_seq(panel: np.ndarray, t: int, seq_len: int) -> np.ndarray:
    window = panel[max(0, t - (seq_len - 1)) : t + 1]
    if window.shape[0] < seq_len:
        pad = np.zeros((seq_len - window.shape[0], panel.shape[1]), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window.astype(np.float32)


def _assemble_state(alpha_dict: dict, market_dict: dict, position: np.ndarray) -> torch.Tensor:
    feats = [
        alpha_dict['alpha_mu'].reshape(-1),
        alpha_dict['alpha_q'].reshape(-1),
        alpha_dict['alpha_uncertainty'].reshape(-1),
        alpha_dict['styles'].reshape(-1),
        market_dict['z_macro'].reshape(-1),
        market_dict['risk_metrics']['rv_mean'].reshape(-1),
        market_dict['risk_metrics']['zvol_mean'].reshape(-1),
        market_dict['risk_metrics']['intent_l2'].reshape(-1),
        market_dict['impact_costs'].reshape(-1),
        market_dict['base_cost'].reshape(-1),
        market_dict['gamma'].reshape(-1),
        market_dict['micro_latent'].reshape(-1),
        market_dict['micro_stats'].reshape(-1),
        torch.from_numpy(position.astype(np.float32)),
    ]
    return torch.cat([f.view(-1) for f in feats], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--replay_coef', type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, size=(args.T, args.N))
    prices = np.cumprod(1 + returns, axis=0)
    adv = rng.uniform(1e-3, 1e-2, size=(args.T, args.N))
    pseudo_industry = rng.integers(0, 3, size=(args.N,))
    impact_params = {'kappa': 1e-3, 'alpha': 0.7, 'beta': 1e-4}
    env = MarketEnv(prices, returns, adv, pseudo_industry, impact_params, adv_cap=0.1)

    seq_T = 32
    seq_feat = 1
    cond_dim = 1
    node_feat_dim = 64
    predictor = Predictor(seq_in_feat=seq_feat, cond_dim=cond_dim, node_feat_dim=node_feat_dim)

    stock_dim = 4
    macro_in_dim = 3
    probe_dim = 2
    intent_dim = 1
    num_regimes = 3
    market_model = MarketModel(
        stock_dim=stock_dim,
        macro_in_dim=macro_in_dim,
        probe_dim=probe_dim,
        intent_dim=intent_dim,
        num_regimes=num_regimes,
        model_dim=128,
        micro_latent_dim=32,
        layers=2,
        heads=4,
        dropout=0.1,
        kappa=impact_params['kappa'],
    )

    # Build dynamic causal graphs
    ts_codes = [f"S{i:03d}" for i in range(args.N)]
    dates = [f"D{t:03d}" for t in range(args.T)]
    rows = [(dates[ti], ts_codes[ni], float(returns[ti, ni])) for ti in range(args.T) for ni in range(args.N)]
    import pandas as pd

    df_ret = pd.DataFrame(rows, columns=['trade_date', 'ts_code', 'r'])
    ngc_cfg = NGCConfig(max_lag=5, epochs=20, batch_size=128)
    graph_dict = rolling_dynamic_ngc(df_ret, window_size=seq_T, step_size=1, cfg=ngc_cfg, node_order=ts_codes, top_k_per_col=15)
    graph_dates = sorted(graph_dict.keys())

    def _adj_for_t(t_idx: int):
        key_idx = max(0, min(len(graph_dates) - 1, t_idx))
        g = graph_dict[graph_dates[key_idx]]
        return g['adj_indices'], g['adj_weights']

    # Helper to run predictor
    def run_predictor(t_idx: int):
        seq_np = _pad_seq(returns, t_idx, seq_T)
        x_seq = torch.from_numpy(seq_np.transpose(1, 0)[:, :, None]).unsqueeze(0)
        x_node = torch.zeros(1, args.N, node_feat_dim, dtype=torch.float32)
        cur_adj_idx, cur_adj_w = _adj_for_t(t_idx)
        return predictor(x_seq, x_node, None, cur_adj_idx, cur_adj_w)

    # Helper to build market tensors
    def build_market_tensors(t_idx: int):
        stock_features = np.stack(
            [
                _rolling_feat(returns, t_idx, 5, 'mean'),
                _rolling_feat(returns, t_idx, 10, 'std'),
                adv[t_idx].astype(np.float32),
                _rolling_feat(prices, t_idx, 5, 'mean'),
            ],
            axis=-1,
        ).astype(np.float32)
        macro_state = np.array(
            [
                returns[max(0, t_idx - 20) : t_idx + 1].mean(),
                returns[max(0, t_idx - 20) : t_idx + 1].std(ddof=0),
                float(t_idx) / max(1, args.T - 1),
            ],
            dtype=np.float32,
        )
        s60 = max(0, t_idx - 59)
        base_adv = adv[s60 : t_idx + 1]
        mu_adv = base_adv.mean(axis=0)
        sd_adv = base_adv.std(axis=0, ddof=0) + 1e-8
        z_adv = ((adv[t_idx] - mu_adv) / sd_adv).astype(np.float32)
        rv10 = _rolling_feat(returns, t_idx, 10, 'std')
        impact_potential = np.stack([z_adv, rv10], axis=-1).astype(np.float32)
        adv_tensor = torch.from_numpy(adv[t_idx].astype(np.float32)).unsqueeze(0)
        sigma_tensor = torch.from_numpy(rv10.astype(np.float32)).unsqueeze(0)
        tensors = {
            'stock': torch.from_numpy(stock_features).unsqueeze(0),
            'macro': torch.from_numpy(macro_state).unsqueeze(0),
            'micro': torch.from_numpy(impact_potential).unsqueeze(0),
            'adv': adv_tensor,
            'sigma': sigma_tensor,
        }
        return tensors

    def zero_intention():
        return torch.zeros(1, args.N, intent_dim, dtype=torch.float32)

    def assemble_state_from_cache(pred_out: dict, market_out: dict, position: np.ndarray):
        rl_state = {
            'alpha_mu': pred_out['rl_state']['alpha_mu'][0],
            'alpha_q': pred_out['rl_state']['alpha_q'][0],
            'alpha_uncertainty': pred_out['rl_state']['alpha_uncertainty'][0],
            'styles': pred_out['rl_state']['styles'][0],
        }
        market_state = {
            'z_macro': market_out['z_macro'][0],
            'risk_metrics': {
                'rv_mean': market_out['risk_metrics']['rv_mean'].unsqueeze(-1),
                'zvol_mean': market_out['risk_metrics']['zvol_mean'].unsqueeze(-1),
                'intent_l2': market_out['risk_metrics']['intent_l2'].unsqueeze(-1),
            },
            'impact_costs': market_out['impact_costs'][0],
            'base_cost': market_out['base_cost'][0],
            'gamma': market_out['gamma'][0],
            'micro_latent': market_out['micro_latent'][0],
            'micro_stats': market_out['micro_stats'][0],
        }
        return _assemble_state(rl_state, market_state, position)

    obs = env.reset()
    t = 0
    pred_cache = run_predictor(t)
    market_inputs = build_market_tensors(t)
    market_cache = market_model(
        market_inputs['stock'],
        market_inputs['macro'],
        market_inputs['micro'],
        zero_intention(),
        market_inputs['adv'],
        market_inputs['sigma'],
    )
    state_tensor = assemble_state_from_cache(pred_cache, market_cache, obs['position'])
    state_dim = state_tensor.numel()
    action_dim = args.N
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    opt_pred = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    opt_mkt = torch.optim.Adam(market_model.parameters(), lr=1e-3)
    bg = np.eye(state_dim, dtype=np.float32)[: min(50, state_dim)]
    explainer = XRLExplainer(agent.q, bg)
    feature_names = [f"f{i}" for i in range(state_dim)]

    while t < args.T - 1:
        state_tensor = assemble_state_from_cache(pred_cache, market_cache, obs['position']).requires_grad_(True)
        action = agent.act(state_tensor.detach(), eps=args.eps)
        weights = np.zeros(args.N, dtype=np.float32)
        weights[action] = 1.0
        dq = weights - obs['position'].astype(np.float32)
        adv_ratio = np.abs(dq) / np.maximum(adv[t], 1e-8)
        intent = np.sign(dq) * adv_ratio
        intent_tensor = torch.from_numpy(intent[None, :, None].astype(np.float32))

        market_intent = market_model(
            market_inputs['stock'],
            market_inputs['macro'],
            market_inputs['micro'],
            intent_tensor,
            market_inputs['adv'],
            market_inputs['sigma'],
        )

        obs_next, gross, done, info = env.step(weights)
        predicted_cost = market_intent['impact_costs'].sum().item()
        reward = float(gross) - predicted_cost

        # cost supervision proxy
        start_lookback = max(0, t - 20)
        sigma_vec = np.sqrt(np.maximum(1e-8, (returns[start_lookback : t + 1] ** 2).mean(axis=0)))
        adv_vec = np.maximum(adv[t], 1e-8)
        temp_vec = impact_params['kappa'] * sigma_vec * (np.abs(dq) / adv_vec) ** impact_params['alpha']
        perm_vec = impact_params['beta'] * np.abs(dq)
        cost_vec = temp_vec + perm_vec

        if not done:
            t_next = t + 1
            pred_next = run_predictor(t_next)
            market_inputs_next = build_market_tensors(t_next)
            market_next = market_model(
                market_inputs_next['stock'],
                market_inputs_next['macro'],
                market_inputs_next['micro'],
                zero_intention(),
                market_inputs_next['adv'],
                market_inputs_next['sigma'],
            )
            next_state_tensor = assemble_state_from_cache(pred_next, market_next, obs_next['position'])
        else:
            next_state_tensor = torch.zeros_like(state_tensor)
            pred_next = None
            market_next = None
            market_inputs_next = None

        agent.buffer.push(state_tensor.detach(), action, reward, next_state_tensor.detach(), float(done))

        cur_loss = agent.td_loss(
            state_tensor.unsqueeze(0),
            torch.tensor([action]),
            torch.tensor([reward], dtype=torch.float32),
            next_state_tensor.unsqueeze(0),
            torch.tensor([float(done)], dtype=torch.float32),
        )
        replay_loss = None
        if len(agent.buffer) >= args.batch:
            batch = agent.buffer.sample(args.batch)
            replay_loss = agent.td_loss(batch.s, batch.a, batch.r, batch.sp, batch.d)
        loss_pred = torch.nn.functional.mse_loss(pred_cache['mu'][0], torch.from_numpy(returns[t].astype(np.float32)))
        impact_pred = market_intent['impact_costs'].squeeze(0)
        loss_cost = torch.nn.functional.mse_loss(impact_pred, torch.from_numpy(cost_vec.astype(np.float32)))
        kd_loss = market_intent['regime_kd_loss']
        total_market_loss = loss_cost + 0.1 * kd_loss

        total_loss = cur_loss + total_market_loss + loss_pred
        if replay_loss is not None:
            total_loss = total_loss + args.replay_coef * replay_loss

        agent.opt.zero_grad()
        opt_pred.zero_grad()
        opt_mkt.zero_grad()
        total_loss.backward()
        agent.opt.step()
        opt_pred.step()
        opt_mkt.step()
        agent._soft_update()

        shap_vals = explainer.explain_decision(state_tensor.detach().numpy())
        shap_a = shap_vals[action]
        idx = np.argsort(np.abs(shap_a))[::-1][:3]
        drivers = [(feature_names[i], float(shap_a[i])) for i in idx]
        print(f"t={t} a={action} gross={gross:.6f} pred_cost={predicted_cost:.6f} reward={reward:.6f} drivers={drivers}")

        obs = obs_next
        t += 1
        if not done:
            pred_cache = pred_next
            market_cache = market_next
            market_inputs = market_inputs_next
        else:
            break

    print(f"Stage C loop finished. Final cash: {obs['cash']:.6f}")


if __name__ == '__main__':
    main()

