## 第 3 章 基于动态因果图与自适应时频注意的股票回报率预测方法

本章在统一的“数据→（预测×市场）→强化学习→执行”框架下，提出一种面向容量敏感交易的可解释预测与决策体系：以动态格兰杰因果（NGC）学习时变股票间结构；以自适应时频注意网络（SAT-FAN）学习股票内多尺度时序表征；以市场冲击前瞻网络（MILAN）建模宏观情境与大额交易冲击；以 DQN+XRL（SHAP）实现可解释的策略学习与审计。在保持端到端联动训练的同时，体系显式区分“正常涨跌”与“资金冲击增量”，以成本感知奖励引导策略降低自致冲击，其整体框架如图 3.1 所示（见 assets/fda_architecture_ltr.svg）。

### 3.1 股票回报率预测问题及相关定义

#### 3.1.1 符号说明
为便于后续描述，本文采用如下记号：斜体英文字母表示标量（如 x 与 d）；加粗斜体小写表示向量（如 x）；加粗斜体大写表示矩阵（如 X）；希腊字母表示超参数（如 λ）。令 Σ 表示股票集合，N=|Σ|。令 G_t=(V,E_t,W_t) 表示 t 时刻的图（节点为股票）。

#### 3.1.2 问题定义
对任一股票 u∈Σ，在时间步 τ∈[1,L] 可观测特征向量 x(u)_τ∈R^F。本文预测未来 d 日收益（回报率）：
 y(u)_τ = (c(u)_{τ+d} − c(u)_τ)/c(u)_τ ，
其中 c(u)_τ 为收盘价。记 X 为刻画全体股票、时序与图结构的综合输入，Θ 为模型参数，预测函数 f(·;Θ) 输出点预测 \hat{y}(u)_τ 及不确定性等：{\hat{y}(u)}_{u∈Σ} = f(X;Θ)。

### 3.2 基于动态因果与时频注意的模型设计与构建（Stock-DCX）

本节从股票间关系聚合、股票内关系聚合、跨时间聚合与融合、市场冲击前瞻与回报分解四个方面展开，给出与现有实现一一对应的模块化描述。

#### 3.2.1 股票间关系聚合（NGC → RGAT）
（1）动态因果图构建 NGC（fda/graphs/ngc_builder.py）。对每个目标股票 j，构造滞后设计矩阵 X_lag（包含所有股票的过去 L_lag 个收益），以 MLP 预测 r^{(j)}，并在首层权重施加“以来源股票为组”的组套索（Group Lasso）正则，得到因果强度 A_{ij}（i→j）。随后：
- 道德化（Moralization）：将有向图 A 连接共同父节点并去除方向，得到无向图 M；
- 稀疏化与标准化：转为 edge_index([2,E]) 与 edge_weight([E])，并保留 adj_indices([E,2]) 以兼容纯 PyTorch RGAT；
- 动态化：rolling_dynamic_ngc(df, window, step) 以窗口终点为键生成图字典 {G_t}，捕捉时变传导结构。

（2）关系图注意 RGAT（fda/models/rgat.py）。用 adj_indices/adj_weights 进行稀疏注意力消息传递：
- 线性变换 W 将节点特征投影为多头表示；
- 以源/宿节点打分并做宿节点归一 softmax；
- 信息聚合得到股票间表征 h_inter；
该路径显式利用 NGC 提供的结构先验，克服静态相关或完全连接图的失真问题。

#### 3.2.2 股票内关系聚合（SAT-FAN）
（1）特征增强（FeatureEnhancer）。对每股时序做 z-score 标准化，并对指定通道做 rFFT，取频谱幅值 Top-K 幅值并与时域特征拼接，引入频域先验。

（2）多尺度 TCN 专家。构建短/中/长三路 Dilated TCN（指数膨胀 1,2,4,…），以固定但互补的感受野覆盖近中远期模式。

（3）专家内时间注意力（TemporalAttention）。对每路 TCN 的输出 [B,H,T] 做线性打分与 softmax，沿时间加权汇聚为上下文向量 [B,H]，实现“在固定窗口内自适应选点”。

（4）专家间融合注意力（ExpertFusionAttention）。以三路上下文向量组成 [B,3,H]，用自注意/多头注意做加权融合，得到 h_intra。

#### 3.2.3 跨时间相关性与融合预测头
（1）跨视角融合（Fusion）。以 Cross-Gating 双向门控（Inter→Intra 与 Intra→Inter）抑制噪声与不一致维度，拼接投影得到融合表示 h。

（2）预测头与风格。线性层输出 mu 与 quantiles(q10/q50/q90)，softplus 输出 uncertainty；StyleHead 线性映射得到风格矩阵 styles，便于做风格约束与解释。

### 3.3 市场冲击前瞻（MILAN）与回报分解

（1）宏观情境。MarketRegimeDetector(HMM) 基于指数收益序列拟合隐状态，输出 regime_probs；MacroProjector 将宏观特征压缩为低维 z_macro。

（2）微观流动性探测（Probe）。按 60 日窗口计算成交活跃异常 z（量/额 z-score）与 10 日实现波动率 rv，组成冲击潜力向量。

（3）冲击传播与成本预测（MILAN）。ImpactPropagationTransformer 将“股票特征 + z_macro + 冲击潜力 + 交易意图”嵌入统一空间，经 TransformerEncoder 聚合传导，ImpactPredictionHead 输出逐标的 impact_costs。

（4）回报分解。BaselineReturnHead 预测 r_norm；结合交易意图（方向与规模）得到冲击增量 r_impact，进而得到 r_total_pred = r_norm + r_impact，用于与实现收益对齐监督。

### 3.4 强化学习（DQN）与可解释性（XRL）

（1）状态（晚融合）。将预测侧 rl_state = {alpha_mu, alpha_q, alpha_uncertainty, styles, h_intra, h_inter} 与市场侧 {z_macro, regime_probs, risk_metrics, r_norm, r_impact, impact_costs} 并行拼接为策略输入。

（2）动作与交易意图。策略输出目标权重/离散动作，经 Δw 构造 intent = sign(Δw)·|Δw|/ADV，使 MILAN 的成本/冲击回报对动作敏感。

（3）奖励（成本硬接线）。reward = PnL − ∑ impact_costs，可扩展纳入 cross-impact 与执行日程；环境返回成本细项用于审计与对账。

（4）可解释性（XRL）。XRLExplainer(SHAP) 对所选动作 Q 值做事后归因，输出正/负 Top-K 驱动特征，形成“决策备忘录”。

### 3.5 实验与结果分析

#### 3.5.1 数据集
建议在 CSI300、HSI、NDX100 上进行评测：涵盖 A 股大型蓝筹、港股多行业与美股高成长科技股，体现不同波动与结构特征。数据包含 2018–2022（示例）期间的成分股，特征包含价格/量/额、估值、流动性与派生技术因子；标签为未来 d 日收益。

#### 3.5.2 实验超参数与环境设置
超参示例：NGC 滞后阶数 L_lag∈{3,5,7}、组套索系数 λ_group∈{1e-4,1e-3,1e-2}；SAT-FAN FFT Top-K∈{4,8,16}、TCN 层数/膨胀；RGAT 头数/层数；MILAN 层数/头数；impact-to-return 比例；DQN 学习率与 ε 调度。使用早停与权重衰减以防过拟合。

#### 3.5.3 评估指标
与参考一致：IC、RankIC、RoR、ASR。IC/RankIC∈[−1,1] 越大越好；RoR 越高越好；ASR 以 252 交易日年化，越高越好。

#### 3.5.4 基准模型与对比
基线包含：MLP、LSTM、GRU、ALSTM、Transformer、StockMixer、MASTER、图模型（GCN/GAT）、RWKV/MATCC 等；本模型为 Stock-DCX（NGC+SAT-FAN+RGAT+MILAN+DQN+XRL）。

#### 3.5.5 消融实验
设置 Inter only、Intra only、Inter+Intra（无 Cross-Gating）、无 NGC（或静态相关图）、无道德化、无 FFT 频谱增强、无专家内注意、无 MILAN/无回报分解、无 XRL，评估各模块贡献。

#### 3.5.6 超参数敏感性
分析 NGC 的 L_lag/λ_group、SAT-FAN 的 Top-K/层数/膨胀、RGAT 头数、MILAN 头数/层数与 impact-to-return 比例对 IC/RankIC 的影响，寻找精度-复杂度平衡点。

#### 3.5.7 鲁棒性实验
通过扩张资产池、注入噪声、改变窗口/步长与 Regime 切换事件等，观察性能稳定性；记录解释一致性（决策备忘录）与成本预测偏差。

### 3.6 本章小结
本章提出的 Stock-DCX 以“动态因果 + 时频多尺度 + 冲击前瞻 + 可解释 RL”为核心：NGC 为 RGAT 提供时变结构先验；SAT-FAN 在股票内以注意力自适应选点；MILAN 将宏观/微观/意图融合实现成本与回报分解；策略以 impact-aware 奖励优化执行并以 SHAP 解释。该体系在容量受限场景兼顾预测精度、执行成本与合规可解释性，具备良好的工程可落地性与扩展潜力。

