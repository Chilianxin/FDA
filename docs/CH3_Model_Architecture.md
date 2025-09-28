## 第三章 模型总体架构与方法

本章系统化阐述本项目的端到端交易智能体架构，涵盖数据预处理、动态因果图构建、深度学习预测模块（股票内/股票间与融合）、市场冲击前瞻模块、强化学习与可解释性（XRL），以及训练联动与执行闭环。设计目标是在容量受限与价格冲击显著的场景下，学习在成本感知前提下的最优交易策略。

### 3.1 设计目标与原则
- **容量友好**：显式建模冲击成本与跨标的传导，减少自致冲击。
- **动态感知**：通过动态格兰杰因果（NGC）与宏观/微观探测捕捉时变关系。
- **可解释性**：引入基于 SHAP 的 XRL，对每次决策给出可审计的特征归因。
- **工程可用**：模块解耦、接口清晰，便于替换/扩展与高效训练。

### 3.2 数据与预处理
- 数据载入与校验：`fda/data/dataset.py::load_panel`
  - 必备字段：价格/量/额、估值、广度、`pct_chg` 等；标准化日期与代码。
  - 派生特征：对数收益 `r`、`hl_vol`、`ADV`、`Amihud`、横截面 z 分数（流动性/估值）。
- 训练切分：滚动/走访式窗口，避免未来信息泄露；在图与模型构建时按时间戳严格对齐。

### 3.3 动态因果图（NGC）
- 位置：`fda/graphs/ngc_builder.py`
- 方法：对每个目标股票，以全体股票的滞后收益构造设计矩阵，经 MLP 预测，并在首层施加**组套索（Group Lasso）**，以“来源股票”为组实现组稀疏，得到 i→j 的非线性格兰杰因果强度。
- 图处理：
  - 有向图 `A` → 道德化（连接共同父节点并去向）→ 无向图 `M`。
  - 转换为稀疏表示：`edge_index([2,E])` 与 `edge_weight([E])`；同时提供 `adj_indices([E,2])` 供纯 PyTorch RGAT 使用。
- 动态：`rolling_dynamic_ngc(df, window, step)` 以窗口终点为键输出动态图字典，随时间滚动更新。

### 3.4 预测模块（股票内/间与融合）
- 总体接口：`fda/models/predictor.py::Predictor`
  - 输出：`mu`（点预测）、`quantiles(q10/q50/q90)`、`uncertainty`、`styles`、中间表示 `h_intra/h_inter`，以及供 RL 使用的 `rl_state`（晚融合字段）。

- 股票内（Intra）— SAT-FAN：`fda/models/sat_fan.py`
  - 特征增强：时域标准化 + FFT 频谱 Top-K 幅值拼接。
  - 多尺度 TCN 专家：短/中/长三路（指数膨胀卷积），固定感受野但由**时间注意力**在窗口内自适应选点。
  - 专家间融合注意力：以自注意力融合三位专家的上下文向量，得到股票内表征 `h_intra`。

- 股票间（Inter）— RGAT：`fda/models/rgat.py`
  - 使用 `adj_indices/adj_weights` 进行稀疏图注意力传递，得到 `h_inter`。
  - 图由 NGC 动态生成，捕捉时变的非线性因果结构。

- 融合与风格：`fda/models/fusion.py`、`fda/models/style_head.py`
  - Cross-Gating 双向门控（GAT→TCN 与 TCN→GAT）并拼接投影，得到融合表示 `h`。
  - 风格头 `StyleHead` 提取风格矩阵 `E`，可作风格约束与解释分析。

### 3.5 市场模块（宏观情境 + 冲击前瞻）
- 位置：`fda/market/milan.py`
- 组件：
  - 宏观感知：`MarketRegimeDetector(HMM)` 识别隐含市场状态；`MacroProjector` 输出 `z_macro`。
  - 微观流动性探测：`micro_liquidity_probe` 计算成交活跃异常 z 分数与短期实现波动率 `rv`，形成冲击潜力。
  - 冲击传播：`ImpactPropagationTransformer`（Transformer 编码）对“股票特征 + 宏观 + 冲击潜力 + 交易意图”进行建模。
  - 成本/回报头：`ImpactPredictionHead` 预测逐标的预期冲击成本 `impact_costs`；`BaselineReturnHead` 预测正常涨跌 `r_norm`；结合交易意图得到 `r_impact`，并输出 `r_total_pred = r_norm + r_impact`。
- 对外统一接口：`MarketModel.forward(...) → { z_macro, regime_probs, risk_metrics, impact_costs, r_norm, r_impact, r_total_pred }`。

### 3.6 强化学习与可解释性（XRL）
- RL 算法：默认提供 `DQNAgent`（`fda/rl/algo/dqn.py`）作为可解释性基线；训练入口 `fda/training/train_rl.py`。
- 状态（晚融合）：
  - 预测侧：`rl_state = { alpha_mu, alpha_q, alpha_uncertainty, styles, h_intra, h_inter }`
  - 市场侧：`{ z_macro, regime_probs, risk_metrics, r_norm, r_impact, impact_costs }`
  - 拼接后作为 DQN 的输入特征。
- 奖励（硬接线成本）：`reward = PnL − impact_costs`（可扩展 cross-impact/执行日程）。
- 可解释性：`XRLExplainer(SHAP)` 对所选动作的 Q 值做事后归因，输出正/负 Top-K 驱动特征，生成“决策备忘录”。

### 3.7 端到端联动与训练策略
- 动态图接入：训练循环按时间步从 `rolling_dynamic_ngc` 选取对应窗口的图输入 RGAT。
- 联动优化（示例实现，见 `fda/training/train_rl.py`）：
  - 预测侧：用实现收益监督 `mu`（MSE/Huber/分位数损失可选）。
  - 市场侧：用实现收益监督 `r_total_pred`，用环境成本代理监督 `impact_costs`。
  - RL 同步：DQN 按 impact-aware 奖励更新策略；同时对 `Predictor/MarketModel` 进行辅助更新，形成联合学习闭环。
- 稳定性建议：
  - 先以监督/自监督预训练 `SATFAN/RGAT/MarketModel`，再与 RL 交替/联合微调。
  - 对状态做标准化与降维投影，控制输入尺度与方差。

### 3.8 在线推理与执行闭环
- 在线数据进入预处理 → NGC 动态图更新（可异步） → `Predictor` 与 `MarketModel` 前向 → 晚融合状态进入策略 → 策略输出动作 → 构造交易意图 → `MarketModel` 评估成本 → 执行与结算。
- 关键监控：冲击成本与容量曲线、策略在不同 Regime 下的稳定性、解释日志命中率与一致性。

### 3.9 安全、合规与可解释审计
- 输出“决策备忘录”：记录每笔决策的 SHAP 归因、关键状态特征、预期成本与 Regime，以便审计与风控复核。
- 数据与模型变更需留存版本，复现实验以哈希/时间戳追踪；训练/推理日志持久化。

### 3.10 小结
本章提出了以“动态因果 + 时频多尺度 + 冲击前瞻 + 可解释 RL”为核心的端到端结构：
1) NGC 提供时变的非线性股票间结构，使得 RGAT 能在动态图上学习传导关系；
2) SAT-FAN 在股票内通过多尺度与注意力自适应挑选关键时刻；
3) 市场模块以 HMM/MILAN/回报分解量化宏微观情境与交易冲击；
4) RL 以 impact-aware 奖励驱动策略在容量约束下优化执行；
5) XRL 提升可解释性与合规可审计性。该体系在保持可扩展性的同时，为容量敏感的交易决策提供了系统性的、可解释的学习框架。

