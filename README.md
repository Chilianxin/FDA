# FDA

## 动态格兰杰因果驱动的股票关系模块

本仓库已集成一个基于神经网络格兰杰因果（Neural Granger Causality, NGC）的动态图构建流程，用以替换静态的相关性图。该流程能够在滚动窗口上学习市场中时变的非线性因果关系，并将其道德化（Moralization）为无向图，以便与 GAT 模型配合训练。

### 组件概要

- `code/ngc_builder.py`: 提供 `NGC_Builder` 与 `build_dynamic_graphs`。
  - 使用 MLP 进行单目标预测，并引入组套索（Group Lasso）惩罚以做组级稀疏化，从而发现格兰杰原因。
  - 支持滚动窗口生成时变有向因果图（邻接矩阵）。
- `code/graph_utils.py`: 图工具函数。
  - `moralize_graph`: 将有向因果图进行“父节点连边 + 去方向”得到无向图。
  - `to_edge_index`: 将无向邻接矩阵转换为 `torch_geometric` 需要的 `edge_index` 与 `edge_weight`。
- `code/stock_model.py`: 基于 `torch_geometric` 的 `StockModel`（GATv2）。
- `code/train_dynamic.py`: 训练脚本，按时间戳选择对应的动态图，进行前向与优化。
- `scripts/setup_env.sh`: 环境初始化脚本（创建 venv、安装 PyTorch、PyG 与可选 NGC 库）。
- `scripts/train.sh`: 调用训练脚本的入口。

### 安装与环境

1) 安装 Python 3.10+。若在 Linux 环境，可直接执行：

```bash
bash scripts/setup_env.sh
```

完成后通过如下命令激活虚拟环境：

```bash
source .venv/bin/activate
```

若需手工安装依赖，核心依赖包括：`torch`、`torch_geometric`、`numpy`、`pandas`、`scikit-learn`。

可选：集成开源 NGC 参考实现（如 `Neural-GC`、`mlcausality`），脚本中已包含尝试安装。

### 数据准备

训练脚本期望一个 CSV：索引为日期（可解析为 Datetime），列为股票代码，数值为收盘价（或价格序列）。

示例：`data/prices.csv`

```
date,AAPL,MSFT,GOOG
2018-01-02,172.26,85.95,1053.40
2018-01-03,172.23,86.35,1082.48
...
```

### 运行训练

```bash
source .venv/bin/activate
bash scripts/train.sh data/prices.csv
```

或直接：

```bash
python -m code.train_dynamic --csv data/prices.csv --window_size 252 --step_size 21 --device cpu
```

关键流程：
- 滚动窗口上调用 `NGC_Builder` 生成有向因果图。
- 对应时间点选取最近窗口的图，并进行道德化与转换为 `edge_index`。
- 使用 `StockModel`（GAT）进行前向与优化，目标默认为“下一日收益率”预测（可根据需要替换特征与目标）。

### 可配置项

`NGCConfig`（见 `code/ngc_builder.py`）支持如下常见超参：`lags`、`hidden_dim`、`group_lasso_alpha`、`threshold_percentile`、`epochs`、`early_stop_patience` 等。

### 注意事项与扩展

- 组套索惩罚施加在第一层权重上，按“单支股票的所有滞后”为一组进行惩罚。
- `threshold_percentile` 控制每个目标节点保留的父集合规模，可按验证集或业务先验调整。
- 若需 GPU，请在安装 PyTorch / PyG 时选择对应 CUDA 版本，并将 `device` 设为 `cuda`。
- 可将价格转收益率、做标准化、增加更多时序特征等，以提升稳定性与效果。
