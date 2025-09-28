## 第 3 章 基于动态因果图与自适应时频注意的股票回报率预测方法

本章在统一的“数据→（预测×市场）→强化学习→执行”框架下，提出一种面向容量敏感交易的可解释预测与决策体系：以动态格兰杰因果（NGC）学习时变股票间结构；以自适应时频注意网络（SAT-FAN）学习股票内多尺度时序表征；以市场冲击前瞻网络（MILAN）建模宏观情境与大额交易冲击；以 DQN+XRL（SHAP）实现可解释的策略学习与审计。在保持端到端联动训练的同时，体系显式区分“正常涨跌”与“资金冲击增量”，以成本感知奖励引导策略降低自致冲击。

### 3.1 股票回报率预测问题及相关定义

#### 3.1.1 符号说明
为统一表述，采用如下记号：斜体英文字母表示标量（如 x, d），加粗斜体小写表示向量（如 x），加粗斜体大写表示矩阵（如 X），希腊字母表示超参数（如 λ）。令 Σ 表示股票集合，N=|Σ|。

#### 3.1.2 问题定义
对任意股票 u∈Σ，在时间步 τ∈[1,L] 观测到特征向量 x(u)τ∈R^F。本文预测未来 d 日收益，定义为
 y(u)τ = (c(u)τ+d − c(u)τ)/c(u)τ ，
其中 c(u)τ 为收盘价。设 X 为刻画全体股票、时序与图结构的综合输入，Θ 为模型参数，预测函数 f(·;Θ) 输出点预测 ŷ(u)τ 及不确定性等辅助量：{ŷ(u)}u∈Σ = f(X;Θ)。

### 3.2 模型总体设计与构建（Stock-DCX：Dynamic Causality × Time-Frequency Attention × XRL）

模型由三条关键路径与融合/决策层构成（如图 3.1 所示，见 assets/fda_architecture_ltr.svg）：
1) 股票间路径（Inter-Stock）：基于 NGC 的动态图，驱动 RGAT 学习时变的非线性传导关系；
2) 股票内路径（Intra-Stock）：SAT-FAN 以时频增强与多尺度 TCN 专家建模单股时序，并以注意力自适应选点；
3) 市场路径（Macro/Micro-Impact）：MILAN 汇聚宏观 Regime、微观流动性与交易意图，前瞻估计冲击成本与回报分解；
4) 融合与策略：预测路径输出 mu/quantiles/uncertainty/styles；市场路径输出 z_macro/impact decomposition。两路状态晚融合后输入策略；以 impact-aware 奖励驱动学习，并以 SHAP 进行决策归因。

#### 3.2.1 股票间关系聚合（动态因果图 NGC → RGAT）
NGC（fda/graphs/ngc_builder.py）对每个目标股票 j 构造滞后设计矩阵 X_lag 并训练 MLP 预测 r(j)，在首层施加“以来源股票为组”的组套索（Group Lasso），得到因果强度 A_ij。为与 GAT 兼容，先将有向 A 道德化为无向 M，再转为稀疏 edge_index/edge_weight。滚动窗口生成随时间变化的图字典 {G_t}。

RGAT（fda/models/rgat.py）使用 adj_indices/adj_weights 进行稀疏注意力聚合，得到股票间表示 h_inter。该路径解决了静态相关图无法刻画非线性与时变传导的痛点。

#### 3.2.2 股票内关系聚合（SAT-FAN：自适应时频注意）
SAT-FAN（fda/models/sat_fan.py）包含：
- 特征增强：时域 z-score 与 FFT 频谱 Top-K 幅值拼接；
- 多尺度 TCN 专家：短/中/长三路，指数膨胀卷积形成固定但互补的感受野；
- 专家内时间注意力：在各自窗口内自适应挑选关键时刻，生成上下文向量；
- 专家间融合注意力：以自注意力融合三位专家，得到 h_intra。

#### 3.2.3 融合与预测头
融合层（fda/models/fusion.py）采用 Cross-Gating 双向门控并拼接投影，得到融合表示 h。预测头产生：点预测 mu、分位数 q10/q50/q90、不确定性 uncertainty；风格矩阵 styles（fda/models/style_head.py）。

#### 3.2.4 市场冲击前瞻与回报分解（MILAN）
MILAN（fda/market/milan.py）包含：
- 宏观情境：MarketRegimeDetector(HMM) 识别 Regime；MacroProjector 输出低维 z_macro；
- 微观探测：micro_liquidity_probe 计算成交活跃异常 z 与短期实现波动率 rv；
- 冲击传播：ImpactPropagationTransformer 编码“股票特征 + 宏观 + 冲击潜力 + 交易意图”，ImpactPredictionHead 预测逐标的预期冲击成本 impact_costs；
- 回报分解：BaselineReturnHead 预测正常涨跌 r_norm，配合意图得到冲击增量 r_impact，输出 r_total_pred = r_norm + r_impact。

#### 3.2.5 训练目标与正则
监督端：回归损失 L_pred=MSE(mu, r_{t+1}) 或分位数损失；市场损失 L_mkt=MSE(r_total_pred, r)+α·MSE(impact_costs, cost_proxy)。结构正则：风格正交/稀疏、NGC 组套索稀疏、Regime 平滑等。

### 3.3 强化学习与可解释性
策略学习采用 DQN（fda/rl/algo/dqn.py）；训练入口 fda/training/train_rl.py。晚融合状态为 s_t=[rl_state, z_macro, risk_metrics, r_norm, r_impact, impact_costs]；动作映射为交易意图 intent=sign(Δw)·|Δw|/ADV；奖励 R_t=PnL−∑_i cost_i。可解释性通过 XRLExplainer(SHAP) 对所选动作做事后归因，输出正/负 Top-K 驱动特征。

### 3.4 实验与结果分析（建议设置）
数据：CSI300/HSI/NDX100；指标：IC、RankIC、RoR、ASR；基线：MLP/LSTM/GRU/ALSTM/Transformer/StockMixer/MASTER/图模型/RWKV 等；消融：Inter/Intra/无融合/无 NGC/无道德化/无 FFT/无专家内注意/无 MILAN/无 XRL；超参与鲁棒性实验按章节建议配置。

### 3.5 本章小结
本章提出 Stock-DCX：以 NGC 提供时变结构先验、SAT-FAN 提供股票内多尺度注意表征、MILAN 实现冲击成本与回报分解，并以 impact-aware 奖励与 SHAP 可解释策略形成闭环。体系在容量受限场景兼顾精度、成本与合规解释，具备良好可落地性。

