## 第 3 章 基于动态因果图与自适应时频注意的股票回报率预测方法

本章先定义符号与问题，再概述整体结构与模块功能，随后沿数据流路径，逐一详细说明深度学习部分的“股票内关系模块（为何→举例→方法与公式）”“股票间关系模块（为何→举例→方法与公式）”与“融合模块”，最后介绍强化学习部分（为何→举例→方法、状态/环境/动作与公式）并给出全模型的损失/奖励函数。

### 3.1 股票回报率预测问题及相关定义

#### 3.1.1 符号说明
斜体英文字母表示标量（x,d），加粗斜体小写表示向量（\(\boldsymbol{x}\)），加粗斜体大写表示矩阵（\(\boldsymbol{X}\)），希腊字母为超参数（\(\lambda\)）。令 \(\varSigma\) 为股票集合，\(N=|\varSigma|\)。记 \(c^{(u)}_\tau\) 为股票 \(u\) 在 \(\tau\) 的收盘价，\(\boldsymbol{x}^{(u)}_\tau\in\mathbb{R}^{F}\) 为该时刻的特征向量。记动态图 \(\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t,\boldsymbol{W}_t)\)。

#### 3.1.2 问题定义
目标为未来 \(d\) 日回报预测：
\[y^{(u)}_\tau = \frac{c^{(u)}_{\tau+d}-c^{(u)}_{\tau}}{c^{(u)}_{\tau}},\quad u\in\varSigma,\; \tau\in[1,L].\]
给定全体股票历史特征与图结构 \(\boldsymbol{X}\)，学习映射 \(f(\cdot;\Theta)\) 输出点预测与不确定性：\(\{\hat{y}^{(u)}\}_{u\in\varSigma}=f(\boldsymbol{X};\Theta)\)。

### 3.2 整体结构与模块功能（总览）
数据经预处理后分两路进入深度学习预测部分：（1）股票内路径用 SAT-FAN 自适应提取多尺度时频模式；（2）股票间路径用 NGC 生成动态图并驱动 RGAT 学习时变传导；随后经“跨视角融合模块”耦合内外部信息并输出 \(\mu\)、分位数与不确定性；并行地，市场模块（HMM+Probe+MILAN）产生宏观/冲击成本与回报分解（可作为状态与成本）；最终，两路状态晚融合进入强化学习，策略输出动作，动作映射成交易意图再反馈给市场模块评估冲击，形成闭环（见 assets/fda_architecture_ltr.svg）。

### 3.3 深度学习部分

#### 3.3.1 股票内关系模块（SAT-FAN）
（为何）单股价格序列存在多尺度、非平稳与异步特征：短期噪声与跳变、周月度节律与结构性变化并存。若仅用固定窗口或单尺度编码，易错过跨尺度的关键信号。

（举例）如中兴通讯在特定 2 个月区间内“放量上涨—高位震荡—缩量回调”三段模式交替。成交量与收益的联动、短期波动与中期节律共现，单一尺度难以同时覆盖。

（方法与公式，沿数据流）
1) 时频特征增强。对输入 \(\boldsymbol{x}^{(u)}_{1:L}\) 做时间标准化与频域扩展：
\[\tilde{\boldsymbol{x}}^{(u)}_{\tau} = \frac{\boldsymbol{x}^{(u)}_{\tau} - \boldsymbol{\mu}^{(u)}}{\boldsymbol{\sigma}^{(u)}+\varepsilon},\quad \boldsymbol{f}^{(u)}=\operatorname{TopK}\big(|\operatorname{rFFT}(x^{(u)}_{ch})|\big),\]
将 \(\boldsymbol{f}^{(u)}\) 按时间广播并与 \(\tilde{\boldsymbol{x}}^{(u)}_{\tau}\) 拼接得 \(\bar{\boldsymbol{x}}^{(u)}_{\tau}\)。

2) 多尺度 TCN 专家。对 \(\bar{\boldsymbol{x}}^{(u)}_{1:L}\) 送入短/中/长三路 Dilated TCN：
\[\boldsymbol{H}^{(s)}=\operatorname{TCN}_s(\bar{\boldsymbol{x}}^{(u)}_{1:L}),\quad s\in\{\text{short,mid,long}\},\]
其中每层空洞卷积的膨胀率依次为 \(1,2,4,\dots\)，感受野 \(R_s\) 递增，覆盖不同时间跨度。

3) 专家内时间注意力。对每路输出 \(\boldsymbol{H}^{(s)}=[\boldsymbol{h}^{(s)}_1,\dots,\boldsymbol{h}^{(s)}_L]\)：
\[s^{(s)}_\tau = \boldsymbol{w}_s^\top\tanh(\boldsymbol{W}_s\boldsymbol{h}^{(s)}_\tau),\quad \alpha^{(s)}_\tau = \frac{e^{s^{(s)}_\tau}}{\sum_k e^{s^{(s)}_k}},\quad \boldsymbol{c}^{(s)}=\sum_{\tau=1}^L \alpha^{(s)}_\tau\,\boldsymbol{h}^{(s)}_\tau.\]

4) 专家间融合注意力。将 \([\boldsymbol{c}^{(\text{short})},\boldsymbol{c}^{(\text{mid})},\boldsymbol{c}^{(\text{long})}]\) 组成 \(\boldsymbol{C}\in\mathbb{R}^{3\times H}\)，以自注意/多头注意融合为 \(\boldsymbol{h}^{\text{intra}}\)。

#### 3.3.2 股票间关系模块（NGC→RGAT）
（为何）股票间存在行业/供给链/风格/资金等传导，且呈非线性与时变性。静态相关或固定邻接无法反映“当下”的非对称影响与因果方向。

（举例）如两只白酒股在某段时间高度联动，但政策扰动或资金风格切换会改变领先—滞后关系，单一静态图会误导信息聚合方向。

（方法与公式，沿数据流）
1) NGC 因果识别。对目标股票 \(j\) 构造滞后设计矩阵（全体股票过去 L_lag 个收益）：\(\boldsymbol{X}_{\text{lag}}\in\mathbb{R}^{T\times (N\cdot L_{lag})}\)。以 MLP 预测 \(\boldsymbol{y}^{(j)}\in\mathbb{R}^{T}\)：
\[\min_{\Theta_{\text{MLP}}}\; \underbrace{\|\operatorname{MLP}(\boldsymbol{X}_{\text{lag}};\Theta_{\text{MLP}})-\boldsymbol{y}^{(j)}\|_2^2}_{\text{MSE}} + \lambda_{\text{grp}}\sum_{i=1}^N\Big\|\boldsymbol{W}_{:,\mathcal{G}_i}\Big\|_F,\]
其中 \(\mathcal{G}_i\) 为来源股票 \(i\) 的“跨滞后”列组，组范数促使整支股票被整体选择/剔除，得到因果强度 \(A_{ij}\)。

2) 道德化与稀疏图。\(\boldsymbol{A}\) 经道德化得到无向 \(\boldsymbol{M}\)，再转为 \(\text{edge\_index},\text{edge\_weight}\)（并给出 \(\text{adj\_indices}\)）。

3) RGAT 聚合。对节点特征 \(\boldsymbol{x}_i\) 与边集 \(\mathcal{E}\)：
\[\boldsymbol{h}_i^{(h)}=\boldsymbol{W}^{(h)}\boldsymbol{x}_i,\quad e_{ij}^{(h)} = (\boldsymbol{a}^{(h)}_{\text{src}})^\top\boldsymbol{h}_i^{(h)} + (\boldsymbol{a}^{(h)}_{\text{dst}})^\top\boldsymbol{h}_j^{(h)},\]
对固定宿节点 \(j\) 做 masked-softmax 得 \(\alpha_{ij}^{(h)}\)，消息为 \(\boldsymbol{m}_{ij}^{(h)}=\alpha_{ij}^{(h)}\boldsymbol{h}_i^{(h)}\)，聚合 \(\boldsymbol{z}_j^{(h)}=\sum_{i\in\mathcal{N}(j)}\boldsymbol{m}_{ij}^{(h)}\) 并拼接各头得 \(\boldsymbol{h}^{\text{inter}}_j=\big\Vert_h \boldsymbol{z}^{(h)}_j\)。

#### 3.3.3 融合模块（Cross-Gating + 预测头）
（为何）股票内与股票间视角互补且噪声结构不同，直接拼接会造成维度/尺度不匹配与噪声放大。

（方法与公式）交叉门控抑噪并融合：
\[\boldsymbol{g}_{\text{intra} \leftarrow \text{inter}}=\sigma(\boldsymbol{W}_{i2a}\boldsymbol{h}^{\text{inter}}),\quad \boldsymbol{g}_{\text{inter} \leftarrow \text{intra}}=\sigma(\boldsymbol{W}_{a2i}\boldsymbol{h}^{\text{intra}}),\]
\[\tilde{\boldsymbol{h}}^{\text{intra}}=\boldsymbol{h}^{\text{intra}}\odot\boldsymbol{g}_{\text{intra} \leftarrow \text{inter}},\quad \tilde{\boldsymbol{h}}^{\text{inter}}=\boldsymbol{h}^{\text{inter}}\odot\boldsymbol{g}_{\text{inter} \leftarrow \text{intra}},\]
拼接投影 \(\boldsymbol{h}=\phi\big([\tilde{\boldsymbol{h}}^{\text{inter}},\tilde{\boldsymbol{h}}^{\text{intra}}]\big)\)。预测头输出 \(\mu\)、分位数 \(q_{0.1},q_{0.5},q_{0.9}\) 与不确定性 \(u\)。

### 3.4 强化学习部分（为何→举例→方法与公式）
（为何）在容量受限与冲击显著的场景，单纯最大化预测收益容易导致“高换手—高冲击—高成本”。策略需在“收益—成本—风险”间权衡。

（举例）在流动性脆弱期（高 \(rv\)、高 z-活跃），同等幅度的调仓会产生更高滑点与永久冲击；策略若不感知成本，将在实盘中显著劣化。

（方法与公式）
1) 状态（晚融合）：
\[\boldsymbol{s}_t=\operatorname{vec}\big(\underbrace{\text{rl\_state}_t}_{\mu,\,q,\,u,\,\boldsymbol{h}^{\text{intra}},\,\boldsymbol{h}^{\text{inter}},\,\text{styles}},\; \underbrace{\boldsymbol{z}_{\text{macro}},\,\text{risk},\,\boldsymbol{r}^{\text{norm}},\,\boldsymbol{r}^{\text{impact}},\,\boldsymbol{c}}_{\text{市场侧}}\big).\]

2) 动作与环境。令动作为目标权重 \(\boldsymbol{w}_t\)（或离散选择），交易向量 \(\Delta\boldsymbol{w}_t=\boldsymbol{w}_t-\boldsymbol{w}_{t-1}\)。意图向量：
\[\boldsymbol{\text{intent}}_t=\operatorname{sign}(\Delta\boldsymbol{w}_t)\odot\frac{|\Delta\boldsymbol{w}_t|}{\operatorname{ADV}_t}.\]

3) 奖励（影响感知）：
\[R_t=\underbrace{\boldsymbol{w}_{t-1}^\top\boldsymbol{r}_t}_{\text{PnL}}\;-\;\underbrace{\sum_i \kappa^i_t\,\sigma^i_t\Big(\frac{|\Delta w^i_t|}{\operatorname{ADV}^i_t}\Big)^{\alpha^i_t}+\beta^i_t|\Delta w^i_t|}_{\text{impact costs}}.\]

4) DQN 更新。以 Q 网络 \(Q_\theta\) 学习：
\[y_t=R_t+\gamma\max_a Q_\theta(\boldsymbol{s}_{t+1},a),\quad \mathcal{L}_{\text{DQN}}=\big(y_t-Q_\theta(\boldsymbol{s}_t,a_t)\big)^2.\]
解释性：对 \(Q_\theta(\boldsymbol{s}_t, a_t)\) 以 SHAP 归因，输出正/负 Top-K 驱动特征（决策备忘录）。

### 3.5 全模型的损失/奖励函数（联合目标）
监督与正则：
\[\mathcal{L}_{\text{pred}}=\operatorname{MSE}(\mu, r_{t+1}) + \sum_{q\in\{0.1,0.5,0.9\}} \operatorname{Pinball}(q),\]
\[\mathcal{L}_{\text{mkt}}=\operatorname{MSE}(\boldsymbol{r}^{\text{total}},\boldsymbol{r})+\alpha\,\operatorname{MSE}(\boldsymbol{c},\boldsymbol{c}^{\text{proxy}}),\]
\[\mathcal{R}_{\text{group}}=\sum_i\|\boldsymbol{W}_{:,\mathcal{G}_i}\|_F,\quad \mathcal{R}_{\text{style}}=\|\boldsymbol{W}_{\text{style}}\boldsymbol{W}_{\text{style}}^\top-\boldsymbol{I}\|_F^2+\lambda_{\text{sp}}\|\boldsymbol{E}\|_1.\]
强化学习：\(\mathcal{L}_{\text{DQN}}\) 如上，reward 为 \(R_t\)。

联合优化（可交替/端到端）：
\[\min\; \lambda_1\mathcal{L}_{\text{pred}}+\lambda_2\mathcal{L}_{\text{mkt}}+\lambda_3\mathcal{L}_{\text{DQN}}+\lambda_4\mathcal{R}_{\text{group}}+\lambda_5\mathcal{R}_{\text{style}}.\]

### 3.6 小结
本章给出了从符号与问题定义开始，沿数据流对股票内（SAT-FAN）、股票间（NGC→RGAT）与融合模块的完整推导，并阐明了为何与如何引入强化学习及其状态/环境/动作/奖励的数理表述。联合目标在“预测精度—成本约束—可解释性”之间建立可优化的权衡，为容量敏感的交易提供系统化、可审计的学习框架。

