## Preliminary and Methodology for Dynamic-Causality & XRL Trading System (DCX)

本章参照预研文献中“Preliminary 与 Methodology”的组织方式，结合本仓库的模型结构（预测 + 市场 → RL）、以及上一章给出的细化设计，完整给出符号体系、MDP 形式化与交易成本设定，并逐步阐述各模块（股票内/股票间/融合/市场/强化学习）的方法与数据流，配以公式与实现要点。

### 1. Preliminary

#### 1.1 市场与数据记号
```text
Universe: Σ (|Σ| = N)
Time index: t = 1..T, lookback L, prediction horizon d
Prices: c_u[t]  (close), Returns: r_u[t] = log( c_u[t] / c_u[t-1] ) or plain (c_{t}-c_{t-1})/c_{t-1}
Features: x_u[t] ∈ R^F  (price/volume/amount/liquidity/valuation/derived tech factors)
Panel: X = { x_u[1..L] }_{u∈Σ}
Graph at time t: G_t = (V, E_t, W_t)
ADV: ADV_u[t], realized vol: σ_u[t], activity z-score: z_u[t]
```

#### 1.2 预测目标与监督信号
```text
Target (d-day forward return):
y_u[t] = ( c_u[t+d] - c_u[t] ) / c_u[t]
or  y_u[t] = log( c_u[t+d] / c_u[t] )
```
模型输出点预测 μ_u[t] 及分位数/不确定性，用于后续组合与 RL 决策。

#### 1.3 MDP 形式化（环境、状态、动作、转移、奖励）
```text
State s_t:
  Late-fusion of predictor-side rl_state (μ, q, u, h_intra, h_inter, styles)
  and market-side (z_macro, risk_metrics, r_norm, r_impact, impact_costs).
Action a_t:
  Target weights w_t ∈ Δ^N  (or discrete proxy). Trade vector Δw_t = w_t - w_{t-1}.
Transition:
  Positions update with executed trade. Prices evolve exogenously.
Reward R_t (impact-aware):
  R_t = (w_{t-1} · r_t) - cost_t
```

#### 1.4 交易成本与价格冲击（与 MarketModel 对齐）
```text
Single-asset impact proxy per step:
temp_u[t] = κ_u[t] · σ_u[t] · ( |Δw_u[t]| / ADV_u[t] )^{α_u[t]}
perm_u[t] = β_u[t] · |Δw_u[t]|
cost_t = Σ_u ( temp_u[t] + perm_u[t] )
```
MarketModel 输出 impact_costs 用作奖励端硬扣减，并提供回报分解 r_norm, r_impact 以利审计与决策参考。

---

### 2. Methodology

#### 2.1 总览：预测 + 市场 → 强化学习
```text
Data → (Intra: SAT-FAN) & (Inter: NGC→RGAT) → Fusion → (μ, q, u, styles)
Data → MarketModel (HMM + Probe + MILAN + Return Decomposition)
                      → { z_macro, risk_metrics, impact_costs, r_norm, r_impact }
State s_t = late_fusion( predictor outputs, market outputs )
Policy (DQN) acts: a_t ⇒ Δw_t ⇒ intent_t ⇒ MarketModel updates costs
Reward uses impact-aware cost; SHAP provides per-decision attribution
```

#### 2.2 股票内关系模块（SAT-FAN，Why→Example→Method）
• Why：单股序列具有多尺度与非平稳结构，固定窗口或单尺度难以兼顾短/中/长依赖；同时价格—成交量—流动性具备时频互补信息。

• Example：某标的呈“放量上行—横盘—缩量回撤”三段循环；成交活跃与收益的同步/滞后变化，需要不同时间尺度与频域信息协同建模。

• Method（按数据流）
```text
1) Feature Enhancement (time-frequency):
   tilde_x_u[t] = (x_u[t] - mu_u) / (sigma_u + eps)
   f_u = TopK( | rFFT( x_u[channel] ) | ), broadcast over time
   bar_x_u[t] = concat( tilde_x_u[t], f_u )

2) Multi-scale TCN experts (short/mid/long):
   H^(s) = TCN_s( bar_x_u[1:L] ), dilation 1,2,4,...  receptive fields R_s ↑

3) Temporal attention (intra-expert):
   score_s[t] = w_s^T tanh( W_s h_s[t] ),
   alpha_s[t] = softmax_t( score_s[t] ),
   c_s = Σ_t alpha_s[t] · h_s[t]

4) Expert fusion attention (inter-expert):
   H_intra = Attn_fuse( [c_short, c_mid, c_long] )   →  h_intra
```

#### 2.3 股票间关系模块（NGC→RGAT，Why→Example→Method）
• Why：股票间传导具有非线性、非对称且时变；静态相关或固定邻接无法捕捉当期主导关系与方向性。

• Example：同业白酒股在“政策/风格切换期”可能改变领先—滞后关系，静态图会误导信息聚合。

• Method（按数据流）
```text
1) NGC causal discovery with Group Lasso:
   For target j, build X_lag ∈ R^{T×(N·L_lag)} from all stocks' lagged returns.
   Minimize:  MSE( MLP(X_lag; Θ), y^(j) ) + λ_grp · Σ_i || W[:, G_i] ||_F
   → column-group sparsity per source stock i across lags ⇒ A_ij (i→j)

2) Moralization & sparse graph:
   A → (connect parents, drop direction) → M → (edge_index, edge_weight), adj_indices

3) RGAT sparse message passing:
   h_i^(h) = W^(h) · x_i
   e_ij^(h) = a_src^(h)^T h_i^(h) + a_dst^(h)^T h_j^(h)
   α_ij^(h) = softmax_over_i( e_ij^(h) | dst=j )
   m_ij^(h) = α_ij^(h) · h_i^(h)
   z_j^(h)  = Σ_{i∈N(j)} m_ij^(h)
   h_inter_j = Concat_h( z_j^(h) )
```

#### 2.4 融合模块（Cross-Gating + Heads）
• Why：股票内/间视角互补但噪声结构不同，直接拼接会放大偏差；需要跨视角抑噪与双向门控。

• Method
```text
g_intra<-inter = σ( W_i2a · h_inter )
g_inter<-intra = σ( W_a2i · h_intra )
tilde_h_intra = h_intra ⊙ g_intra<-inter
tilde_h_inter = h_inter ⊙ g_inter<-intra
h = φ( [tilde_h_inter, tilde_h_intra] )
Outputs: μ, quantiles(q10/q50/q90), uncertainty u, styles E
```

#### 2.5 市场模块（HMM + Probe + MILAN + Return Decomposition）
• Why：在容量敏感场景，需区分“正常涨跌”与“资金冲击增量”，并前瞻估算交易成本，指导策略在“收益—成本”间取舍。

• Method（按数据流）
```text
1) Macro Regime (HMM) & MacroProjector → z_macro, regime probs
2) Micro Liquidity Probe → activity z (volume/amount z-score), rv (realized vol)
3) Impact propagation (MILAN):
   Encodes [stock_features, z_macro, impact_potential, intent]
   TransformerEncoder → ImpactPredictionHead → impact_costs
4) Return decomposition:
   r_norm  = BaselineReturnHead( stock_features, macro, probe )
   r_impact = -scale · impact_costs ⊙ sign(intent)
   r_total  = r_norm + r_impact
```

#### 2.6 强化学习（DQN + XRL）
• Why：将“预测 α”与“成本/冲击”以决策最优化的方式统一，避免高 α 但高冲击的不可执行方案。

• Example：在高 z 活跃期相同幅度的调仓带来更高冲击；策略若不内生惩罚将劣化。

• Method
```text
State s_t = vec( rl_state_t  ||  z_macro, risk_metrics, r_norm, r_impact, impact_costs )
Action a_t: target weights w_t (or discrete proxy)
Intent: intent_t = sign(Δw_t) ⊙ ( |Δw_t| / ADV_t )
Reward: R_t = (w_{t-1} · r_t) - Σ_i [ κ_i,t σ_i,t ( |Δw_i,t|/ADV_i,t )^{α_i,t} + β_i,t |Δw_i,t| ]
Q-learning: y_t = R_t + γ max_a Q_θ(s_{t+1}, a);  L_DQN = ( y_t - Q_θ(s_t, a_t) )^2
XRL: SHAP on Q_θ(s_t, a_t) → top-K positive/negative drivers (decision memo)
```

#### 2.7 训练策略与端到端联动
```text
• Dynamic graphs: use rolling_dynamic_ngc windows at each step for RGAT
• Auxiliary supervised losses:
  L_pred = MSE( μ, r_{t+1} ) + Σ_{q∈{0.1,0.5,0.9}} Pinball(q)
  L_mkt  = MSE( r_total, r ) + α · MSE( impact_costs, cost_proxy )
• Structure regularizers:
  R_group = Σ_i || W[:, G_i] ||_F   (NGC group lasso)
  R_style = || W_style W_style^T - I ||_F^2  +  λ_sp ||E||_1
• Joint objective (alternate or end-to-end):
  min  λ1·L_pred + λ2·L_mkt + λ3·L_DQN + λ4·R_group + λ5·R_style
```

---

### 3. Summary
本文在“预测 + 市场 → 强化学习”的统一框架下，将动态因果图（NGC）与时频多尺度表征（SAT-FAN）结合，通过 RGAT 捕捉时变传导；市场侧以 HMM/Probe/MILAN 前瞻估算冲击与回报分解；策略以 impact-aware 奖励优化并由 SHAP 提供可解释性。该体系在容量受限情形兼顾预测精度、执行成本与可审计性，便于工程落地与扩展。

