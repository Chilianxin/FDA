## 第三章 模型总体架构与方法

本章系统化阐述本项目的端到端交易智能体架构，涵盖数据预处理、动态因果图构建、深度学习预测模块（股票内/股票间与融合）、市场冲击前瞻模块、强化学习与可解释性（XRL），以及训练联动与执行闭环。设计目标是在容量受限与价格冲击显著的场景下，学习在成本感知前提下的最优交易策略。

### 3.1 问题表述与记号
设资产集合大小为 \(N\)，交易日索引为 \(t=1,\dots,T\)。记价格收益向量为 \(\mathbf{r}_t\in\mathbb{R}^N\)，每资产的时间序列特征为 \(\mathbf{x}^i_{t-L+1:t}\in\mathbb{R}^{C\times L}\)，ADV/流动性等微观量为 \(\ell^i_t\)。动态图以 \(\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t,\mathbf{W}_t)\) 表示，且由 NGC 构造的有向因果邻接 \(\mathbf{A}_t\) 及其道德化无向邻接 \(\mathbf{M}_t\)。

预测模块目标：学习映射 \(f_\theta\) 使得点预测 \(\hat{\mu}_t\approx \mathbb{E}[\mathbf{r}_{t+1}\mid \cdot]\)，并输出不确定性与风格表征。市场模块目标：学习映射 \(g_\phi\) 分解观测回报为 \(\mathbf{r}_t\approx \underbrace{\mathbf{r}^{\text{norm}}_t}_{\text{正常涨跌}} + \underbrace{\mathbf{r}^{\text{impact}}_t}_{\text{资金冲击}}\)，并给出逐标的冲击成本 \(\mathbf{c}_t\)。强化学习目标：在 impact-aware 奖励下优化策略 \(\pi\)。

### 3.2 设计目标与原则
- **容量友好**：显式建模冲击成本与跨标的传导，减少自致冲击。
- **动态感知**：通过 NGC 与宏观/微观探测捕捉时变关系。
- **可解释性**：基于 SHAP 的 XRL，对每次决策给出可审计归因。
- **工程可用**：模块解耦、接口清晰，便于替换/扩展与高效训练。

### 3.3 数据与预处理
- 数据载入与校验：`fda/data/dataset.py::load_panel`；派生 `r, hl_vol, ADV, Amihud, z-score` 等。
- 时间对齐与切分：滚动/走访式窗口，避免信息泄露；图与模型按时间戳严格对齐。

### 3.4 动态因果图（NGC）
- 位置：`fda/graphs/ngc_builder.py`
- 方法：对每目标 j，构造滞后设计矩阵，经 MLP 预测 \(r^j_t\)，并在首层施加**组套索**（以来源股票为组）得到因果强度 \(A_{ij}\)。
- 图处理：有向 \(\mathbf{A}\) → 道德化 → 无向 \(\mathbf{M}\) → 稀疏表示 `edge_index/edge_weight` 与 `adj_indices/adj_weights`。
- 动态：`rolling_dynamic_ngc(df, window, step)` 生成按窗口终点索引的动态图字典。

### 3.5 预测模块（股票内/间与融合）
- 接口：`fda/models/predictor.py::Predictor`；输出 `mu, quantiles, uncertainty, styles, h_intra, h_inter, rl_state`。
- 股票内 SAT-FAN（`fda/models/sat_fan.py`）：
  - 特征增强：时域标准化 + FFT 频谱 Top-K 幅值拼接。
  - 多尺度 TCN 专家：短/中/长三路（指数膨胀卷积），感受野固定；
  - 专家内时间注意力：在各自窗口内自适应聚焦关键时刻，输出上下文向量；
  - 专家间融合注意力：自注意力融合三个尺度，得 \(\mathbf{h}^{\text{intra}}\)。
- 股票间 RGAT（`fda/models/rgat.py`）：基于 `adj_indices/adj_weights` 的稀疏图注意力传递，得 \(\mathbf{h}^{\text{inter}}\)。
- 融合与风格：`fda/models/fusion.py`（Cross-Gating + 拼接投影）得融合表征 \(\mathbf{h}\)；`fda/models/style_head.py` 得风格矩阵 \(\mathbf{E}\)。

### 3.6 市场模块（宏观情境 + 冲击前瞻）
- 位置：`fda/market/milan.py`
- 宏观：`MarketRegimeDetector(HMM)` 输出 `regime_probs`；`MacroProjector` 输出 \(\mathbf{z}_{\text{macro}}\)。
- 微观：`micro_liquidity_probe` 产出成交活跃异常 z 与 \(rv\) 作为冲击潜力。
- 冲击传播：`ImpactPropagationTransformer` 以“股票特征 + 宏观 + 冲击潜力 + 交易意图”编码，`ImpactPredictionHead` 预测 \(\mathbf{c}_t\)。
- 回报分解：`BaselineReturnHead` 预测 \(\mathbf{r}^{\text{norm}}_t\)；按意图方向得到 \(\mathbf{r}^{\text{impact}}_t\)，并输出 \(\mathbf{r}^{\text{total}}_t=\mathbf{r}^{\text{norm}}_t+\mathbf{r}^{\text{impact}}_t\)。
- 接口：`MarketModel.forward → { z_macro, regime_probs, risk_metrics, impact_costs, r_norm, r_impact, r_total_pred }`。

### 3.7 强化学习与可解释性（XRL）
- 算法：`DQNAgent`（`fda/rl/algo/dqn.py`）；训练入口 `fda/training/train_rl.py`。
- 状态（晚融合）：\(\mathbf{s}_t=[\text{rl_state},\; z_{\text{macro}},\; \text{risk},\; r^{\text{norm}},\; r^{\text{impact}},\; \mathbf{c}_t]\)。
- 动作与意图：策略输出权重/离散选择，经 \(\Delta \mathbf{w}_t\) 构造 \(\mathbf{intent}_t=\operatorname{sign}(\Delta\mathbf{w}_t)\cdot |\Delta\mathbf{w}_t|/\text{ADV}_t\)。
- 奖励（成本硬接线）：\(R_t=(\mathbf{w}_{t-1}\odot \mathbf{r}_t)\mathbf{1}-\sum_i c^i_t\)。
- 可解释性：`XRLExplainer(SHAP)` 对所选动作 Q 值归因，输出正/负 Top-K 驱动特征，形成“决策备忘录”。

### 3.8 端到端联动与训练
- 动态图接入：按时间步从 `rolling_dynamic_ngc` 选择窗口图输送至 RGAT。
- 联动优化（示例实现在 `training/train_rl.py`）：
  - 预测侧：以实现收益监督 \(\hat{\mu}\)；
  - 市场侧：以实现收益监督 \(r^{\text{total}}\)，以环境成本代理监督 \(\mathbf{c}\)；
  - RL：以 impact-aware 奖励更新策略；同时对 `Predictor/MarketModel` 进行辅助更新，形成联合学习闭环。

### 3.9 在线推理与执行闭环
数据→预处理→NGC→`Predictor`/`MarketModel`→晚融合状态→策略→动作→意图→`MarketModel` 评估成本→执行/结算。监控冲击成本/容量曲线、Regime 稳定性与解释日志一致性。

### 3.10 小结
本章提出以“动态因果 + 时频多尺度 + 冲击前瞻 + 可解释 RL”为核心的端到端结构：NGC 支撑 RGAT 的时变关系学习；SAT-FAN 在股票内通过多尺度注意力自适应选点；市场模块以 HMM/MILAN/回报分解量化宏微观情境与交易冲击；RL 以 impact-aware 奖励优化执行；XRL 提升可解释性与合规可审计性。该体系在保持可扩展性的同时，为容量敏感的交易决策提供系统性的、可解释的学习框架。

