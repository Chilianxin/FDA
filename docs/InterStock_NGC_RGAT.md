# 股票间关系模块（NGC→RGAT）数学表达（Markdown）

## 1. 数据与收益设定

```text
给定窗口内收益矩阵 R ∈ R^{L×N} （列为股票，行为时间）。
```

## 2. NGC 构图（组套索 MLP）

```text
构造滞后设计（滞后阶数 Lg）：
  X_lag ∈ R^{(L−Lg)×(N·Lg)}
  对每个目标股票 j 有 y^(j) ∈ R^{(L−Lg)}

组套索 MLP（首层按“来源股票 i 跨全部滞后”成组）：
  min_{Θ_j}  || MLP(X_lag; Θ_j) − y^(j) ||_2^2  +  λ · Σ_{i=1..N} || W_j[:, G_i] ||_F

因果强度（列归一化，去自环）：
  A_{ij} = norm_i( || W_j[:, G_i] ||_F ),   A_{jj} = 0

（可选）每列保留 Top-K：
  A_{·j} ← KeepTopK( A_{·j}, K )
```

## 3. 道德化与无向邻接（供 RGAT 使用）

```text
对称化：
  S = max( A, A^T )

父节点完备（Moralization）：
  Parents(j) = { i | A_{ij} > 0 }
  任意 i,k ∈ Parents(j):
    M_{ik} = max( S_{ik}, A_{ij}, A_{kj} ),   M_{ki} = M_{ik}

最终无向邻接（对角线为 0 且裁剪到 [0,1]）：
  M = clip( M, 0, 1 ),  M_{ii} = 0

稀疏边与权重（供图注意力）：
  E = { (i,k) | M_{ik} > 0 },   w_{ik} = M_{ik}
```

## 4. RGAT 聚合（M, X → h_inter）

```text
节点特征： X ∈ R^{N×F}
多头 h = 1..H，单头维度 d

线性投影：
  H^(h) = X · W^(h)       (W^(h) ∈ R^{F×d}, H^(h) ∈ R^{N×d})

带边权的打分（仅对 (i,j)∈E）：
  e_ij^(h) = a_src^(h)^T H_i^(h) + a_dst^(h)^T H_j^(h)
  e_ij^(h) ← e_ij^(h) · w_{ij}

按目的节点 j 的 softmax 归一：
  α_ij^(h) = softmax_{i∈N(j)}( e_ij^(h) )

消息与聚合：
  m_ij^(h) = α_ij^(h) · H_i^(h)
  Z_j^(h)  = Σ_{i∈N(j)} m_ij^(h)

多头拼接与激活：
  h_inter_j = σ( Concat_h( Z_j^(h) ) )   ∈ R^{H·d}
```

---

以上流程给出“原始收益 → NGC 因果图 → 道德化无向邻接 → RGAT 聚合 → h_inter”的完整数学表达，供绘制结构图或作为实现对照。

