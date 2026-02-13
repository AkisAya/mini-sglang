# 张量并行权重切分原理

## 概述

张量并行（Tensor Parallelism, TP）通过在多个GPU间分割模型权重和计算，实现高效的分布式推理。mini-sglang 中的 `_shard_state_dict` 函数根据不同层的计算特性采用不同的切分策略。

## 一、数学基础

### 矩阵乘法的并行化

对于矩阵乘法 $Y = X \cdot W$，其中：
- $X \in \mathbb{R}^{b \times n}$ 表示输入（batch, hidden_dim）
- $W \in \mathbb{R}^{n \times m}$ 表示权重矩阵
- $Y \in \mathbb{R}^{b \times m}$ 表示输出

可以按两种方式分割 $W$：

#### Row-wise 切分（沿行切分，DIM_0）
$$W = \begin{bmatrix} W_0 \\ W_1 \\ \vdots \\ W_{p-1} \end{bmatrix}, \quad Y = X \cdot W = \begin{bmatrix} X \cdot W_0 \\ X \cdot W_1 \\ \vdots \\ X \cdot W_{p-1} \end{bmatrix}$$

- 各GPU独立计算输出的不同部分
- 无需通信，高效并行

#### Column-wise 切分（沿列切分，DIM_1）
$$W = [W_0 \mid W_1 \mid \cdots \mid W_{p-1}], \quad Y = X \cdot W = [X \cdot W_0 \mid X \cdot W_1 \mid \cdots \mid X \cdot W_{p-1}]$$

- 每个GPU处理输入的一部分
- 需要 AllReduce 汇聚结果

---

## 二、DIM_0 切分（Q、K、V、Gate、Up 投影）

### 计算流程

**Self-Attention 中的 Q、K、V 投影：**

```
Input: X ∈ ℝ^(batch, seq_len, d_model)

W_q ∈ ℝ^(d_model, d_model)  →  切分为 W_q^0, W_q^1, ..., W_q^(n-1)

GPU_i 计算: Q_i = X @ W_q^i  ∈ ℝ^(batch, seq_len, d_model/n)
```

### 为什么使用 DIM_0 切分

1. **独立性强**：每个GPU的Q、K、V是输出的独立分片，无数据依赖
   - GPU_0 生成Q的前1024维
   - GPU_1 生成Q的中间1024维
   - ...
   - 各部分可以独立进行Attention计算

2. **通信高效**：
   - 计算前：无需通信（每个GPU都有完整输入X）
   - 计算后：无需通信（各自保有独立结果）

3. **FFN 中的 Gate 和 Up 投影** 同理：
   ```
   Gate_i = X @ W_gate^i  ∈ ℝ^(batch, seq_len, ffn_dim/n)
   Up_i   = X @ W_up^i    ∈ ℝ^(batch, seq_len, ffn_dim/n)
   Gate_Up_i = [Gate_i, Up_i]  # 后续合并
   ```

### 数学验证

切分前后计算等价性：
$$Q = X \cdot W_Q, \quad Q_i = X \cdot W_Q^i$$
$$Q = [Q_0 \mid Q_1 \mid \cdots \mid Q_{n-1}]$$

其中 $Q_i$ 是 $Q$ 的第 $i$ 个列分块，对应权重的行分块。

---

## 三、DIM_1 切分（O、Down 投影）

### 计算流程

**Attention 中的输出投影和 FFN 中的下投影：**

```
Input: Attention_Out ∈ ℝ^(batch, seq_len, d_model)

W_o ∈ ℝ^(d_model, d_model)  →  切分为 [W_o^0 | W_o^1 | ... | W_o^(n-1)]

GPU_i 计算: Y_i = Attention_Out @ W_o^i  ∈ ℝ^(batch, seq_len, d_model/n)
```

### 为什么使用 DIM_1 切分

1. **输入数据受限**：Attention输出和FFN中间层是上一层的结果，需要聚合
   
2. **输出维度切分**：
   - GPU_0 计算输出的前1024维
   - GPU_1 计算输出的中间1024维
   - ...
   - 最后需要 AllReduce 汇聚

3. **计算流程**：
   ```
   Attention_Out ∈ ℝ^(batch, seq_len, 4096)
   分割为: [Attn_0 | Attn_1 | ... | Attn_{n-1}]  (沿seq_len维)
   
   或完整输入，但权重沿列分割:
   Y_i = Attention_Out @ W_o^i
   
   最后: Y = AllReduce([Y_0, Y_1, ..., Y_{n-1}])
   ```

### 数学验证

$$Y = X \cdot W_O = X \cdot [W_O^0 \mid W_O^1 \mid \cdots \mid W_O^{n-1}]$$
$$= [X \cdot W_O^0 \mid X \cdot W_O^1 \mid \cdots \mid X \cdot W_O^{n-1}]$$

使用 AllReduce 实现最终的行级聚合。

---

## 四、词表切分（Embedding 和 LM Head）

### 计算流程

**Embedding层：**
```
Token_IDs: [batch_size, seq_len]  ∈ {0, 1, ..., vocab_size-1}

W_embed ∈ ℝ^(vocab_size, d_model)  →  按行切分

GPU_i: W_embed_i ∈ ℝ^(vocab_size/n, d_model)

对于token_id，使用 gather 操作获取对应embedding
```

**LM Head 层：**
```
Hidden: [batch_size, seq_len, d_model]

W_lm_head ∈ ℝ^(vocab_size, d_model)  →  按行切分

Logits_i = Hidden @ W_lm_head_i^T  ∈ ℝ^(batch_size, seq_len, vocab_size/n)

最后通过 AllGather 汇聚所有logits进行 argmax 找最大值token
```

### 为什么行切分（与 DIM_0 等价）

1. **词表维度的特殊性**：
   ```
   索引访问: embedding[token_id] 
   需要按行切分，使得token可以分配到某个GPU
   
   vocab_start_idx = rank * (vocab_size / n_gpu)
   vocab_end_idx = min((rank + 1) * (vocab_size / n_gpu), vocab_size)
   
   分配规则：token_id 属于 vocab_start_idx ~ vocab_end_idx 时，
   由对应GPU处理
   ```

2. **通信模式**：
   - **Embedding前向**：AllGather（汇聚所有embedding）
   - **LM Head前向**：AllGather（汇聚所有logits）
   - **LM Head反向**（若有梯度）：AllReduce（汇聚梯度）

### 数学验证

```
原始: logits = hidden @ W_lm_head^T  ∈ ℝ^(batch, seq_len, vocab_size)

切分后:
logits_i = hidden @ W_lm_head_i^T  ∈ ℝ^(batch, seq_len, vocab_size/n)

最终: logits = [logits_0 | logits_1 | ... | logits_{n-1}]  via AllGather
```

---

## 五、通信模式总结

| 层类型 | 切分方式 | 前向通信 | 后向通信 | 目的 |
|--------|--------|--------|--------|------|
| Q/K/V/Gate/Up | DIM_0（行切） | 无 | AllReduce | 输出并行 |
| O/Down | DIM_1（列切） | AllReduce | AllGather | 输入汇聚 |
| Embedding | 行切 | AllGather | AllReduce | Token分散 |
| LM Head | 行切 | AllGather | AllReduce | Logits分散 |

---

## 六、代码实现对应关系

```python
# DIM_0 切分（沿dim=0切分，取当前rank的分片）
value.chunk(n, dim=0)[r]
# 等价于：value[r*chunk_size:(r+1)*chunk_size, :]

# DIM_1 切分（沿dim=1切分，取当前rank的分片）
value.chunk(n, dim=1)[r]
# 等价于：value[:, r*chunk_size:(r+1)*chunk_size]

# 词表切分（行切，非均匀分割处理余数）
vocab_start_idx = r * num_embeddings_per_partition
vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
value[vocab_start_idx:vocab_end_idx, :]
```

---

## 七、切分后的推理流程示例（2 GPU）

```
输入: batch=[1, 4096]

┌─────────────────────────────────────────────┐
│  Q_proj 权重 [4096, 4096] → DIM_0 切分     │
├─────────────────────────────────────────────┤
│ GPU_0: W_q^0 [4096, 2048]                   │
│ GPU_1: W_q^1 [4096, 2048]                   │
│                                              │
│ GPU_0: Q_0 = input @ W_q^0  [1, 2048]       │
│ GPU_1: Q_1 = input @ W_q^1  [1, 2048]       │
│ → Q = [Q_0, Q_1]  [1, 4096] (concat dim=1) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Attention 计算（分布式）                   │
│  scores = Q @ K^T / √d                      │
│  每个GPU处理 2048 维的attention              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  O_proj 权重 [4096, 4096] → DIM_1 切分     │
├─────────────────────────────────────────────┤
│ GPU_0: W_o^0 [4096, 2048]                   │
│ GPU_1: W_o^1 [4096, 2048]                   │
│                                              │
│ GPU_0: Y_0 = attn_out @ W_o^0  [1, 2048]    │
│ GPU_1: Y_1 = attn_out @ W_o^1  [1, 2048]    │
│ → AllReduce: Y = Y_0 + Y_1  [1, 4096]       │
└─────────────────────────────────────────────┘
```

---

## 八、性能分析

### 计算与通信的权衡

1. **DIM_0 切分**：
   - 计算量：$\frac{1}{n}$（均衡分摊）
   - 通信：0（最优）
   - **用途**：用于产生独立输出分片的层

2. **DIM_1 切分**：
   - 计算量：$\frac{1}{n}$（均衡分摊）
   - 通信：AllReduce，成本为 $O(\frac{m}{n})$（$m$ 为输出维度）
   - **用途**：用于需要汇聚结果的层

3. **总体吞吐**：
   - 理想情况（无通信瓶颈）：$n$ 倍加速
   - 实际情况：受 AllReduce 操作影响，约 $0.7n \sim 0.9n$ 倍加速

### 何时应该使用张量并行

- GPU间带宽充足（NVLink、高速互联）
- 模型相对较小，不需要流水线并行
- 需要低延迟推理（对应单样本推理）

---

## 参考

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- Efficient Large-Scale Language Model Training on GPU Clusters
- mini-sglang 源码：`python/minisgl/models/weight.py`
