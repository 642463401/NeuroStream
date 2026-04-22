# NeuroStream 数学公式汇总

本文档整理了 NeuroStream 框架中涉及的所有核心数学公式。

---

## 1. 记忆系统

### 1.1 唤醒评分 (Wake-up Score)

$$
S_{wake}(q, m) = \cos(q, m) + \ln(1 + n_{access}) + \alpha_{reward} \cdot r
$$

- $q$: 查询向量
- $m$: 记忆向量
- $n_{access}$: 记忆被访问次数
- $\alpha_{reward}$: reward 权重 (默认 0.5)
- $r$: 记忆的 reward 评分 $\in [-1, 1]$

> 来源: `memory/pool.py:83-86`

### 1.2 余弦相似度 (Cosine Similarity)

$$
\cos(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}
$$

向量预先经过 L2 归一化 (`faiss.normalize_L2`)，因此内积即为余弦相似度。

> 来源: `memory/pool.py`, `memory/tiered.py`, `consolidation/time_integral.py`

### 1.3 指数衰减 (Exponential Decay)

$$
I(t) = I_0 \cdot e^{-\lambda \cdot \Delta t}
$$

- $I_0$: 初始记忆强度
- $\lambda$: 衰减率 (默认 0.01)
- $\Delta t = t_{now} - t_{created}$

> 来源: `memory/pool.py:95-101`

### 1.4 时间积分固化 (Time-Integral Consolidation)

当新记忆与已有记忆相似度 $\geq \theta_{merge}$ (默认 0.8) 时，合并而非新增:

$$
I_{merged} = I_{old} \cdot e^{-\lambda \cdot \max(\Delta t, 0)} + I_{new}
$$

> 来源: `consolidation/time_integral.py:73-80`

### 1.5 反馈累积 (Feedback Accumulation)

$$
r_{new} = 0.7 \cdot r_{old} + 0.3 \cdot s_{feedback}
$$

指数移动平均方式整合新反馈信号到记忆的 reward 评分中。

> 来源: `runtime/learning.py:31-34`

---

## 2. 分层存储

### 2.1 Warm 层向量检索

$$
\text{scores} = V \cdot q^T, \quad \text{top-}k = \text{argpartition}(\text{scores}, -k)
$$

Hot 层使用 FAISS (sub-ms), Warm 层使用 NumPy 矩阵乘法 (~ms)。

> 来源: `memory/tiered.py:68-78`

### 2.2 晋升条件 (Promotion)

$$
\text{Warm} \rightarrow \text{Hot}: \quad I \geq \theta_{promotion}
$$

- $\theta_{promotion}$: 晋升阈值 (默认 0.7)

> 来源: `memory/tiered.py:212-213`

### 2.3 冷库归档条件 (Archival)

$$
\text{Warm} \rightarrow \text{Cold}: \quad \text{age} > T_{demotion} \;\land\; n_{access} = 0
$$

- $T_{demotion}$: 降级时间窗口 (默认 300s)

> 来源: `memory/tiered.py:297-300`

---

## 3. 影子权重系统

### 3.1 残差投射 (Residual Projection)

$$
f(x) = x + \text{MLP}(x)
$$

$$
\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

其中 $W_2, b_2$ 初始化为 **零**，使得初始行为为恒等映射: $f(x) = x + 0 = x$。

> 来源: `shadow/projector.py:28-44`

### 3.2 对比学习损失 — InfoNCE (Contrastive Loss)

$$
\text{logits}_{ij} = \frac{z_i \cdot z_j}{\tau}
$$

$$
\text{labels}_{ij} = \text{softmax}\left(\frac{\text{sim}_{ij}}{\tau}\right)
$$

$$
\mathcal{L}_{contrast} = -\frac{1}{N} \sum_{i} \sum_{j \neq i} \text{labels}_{ij} \cdot \log \text{softmax}(\text{logits}_{ij})
$$

- $z_i = \text{normalize}(f(x_i))$: L2 归一化后的投射向量
- $\text{sim}_{ij}$: 原始记忆向量间的余弦相似度
- $\tau$: 温度参数 (默认 0.07)

> 来源: `shadow/objectives.py:11-61`

### 3.3 奖励加权对比损失 (Reward-Weighted Contrastive Loss)

$$
w_i = \max(0.1, \; 1 + r_i)
$$

$$
\mathcal{L}_{weighted} = \frac{\sum_i \mathcal{L}_i \cdot w_i}{\sum_i w_i}
$$

高 reward 的记忆获得更大训练权重，低 reward 的记忆被抑制但不会完全消失 (最小权重 0.1)。

> 来源: `shadow/objectives.py:64-110`

### 3.4 记忆强度评分 (Batch Construction)

$$
\text{score}(m) = I_m \cdot \max(0.1, \; 1 + r_m)
$$

按此评分降序排列，取 top-$B$ 构建训练 batch。

> 来源: `shadow/gradient.py:38-42`

### 3.5 影子权重渐近性质 (Asymptotic Decay)

当 $t \to \infty$ 且无新记忆输入时:

1. 记忆强度 $I(t) \to 0$（指数衰减）
2. 训练信号（梯度）$\to 0$（因 batch 评分 $\propto I_m$）
3. AdamW 的 weight decay 项主导: $\theta_{t+1} \approx (1 - \lambda_{wd} \cdot \eta) \cdot \theta_t$
4. 因此残差分支参数 $\theta \to 0$，投射器回归恒等映射: $f(x) \to x$

$$
\lim_{t \to \infty} \theta(t) = 0 \quad \Longrightarrow \quad \lim_{t \to \infty} f(x) = x + \text{MLP}_\theta(x) = x
$$

> 来源: `shadow/manager.py` (AdamW with `shadow_weight_decay=0.01`)

### 3.6 跨进程 EMA 权重同步

$$
\theta_{local} \leftarrow (1 - \alpha) \cdot \theta_{local} + \alpha \cdot \theta_{shadow}
$$

- $\alpha$: EMA 系数 (默认 0.01)
- 约需 $\sim 1/\alpha = 100$ 次拉取才能完全收敛到新权重

> 来源: `shadow/sync.py:51-64`

---

## 4. 抗灾难性遗忘

### 4.1 弹性权重固化 — EWC (Elastic Weight Consolidation)

$$
\mathcal{L}_{EWC} = \lambda \sum_i F_i \cdot (\theta_i - \theta_i^{*})^2
$$

- $\theta_i^{*}$: 锚点参数 (固化时刻的快照)
- $F_i$: Fisher 信息矩阵对角线元素 (参数重要性)
- $\lambda$: 惩罚系数 (默认 1000.0)

> 来源: `forgetting/ewc.py:31-44`

### 4.2 Fisher 信息估计 (Diagonal Fisher)

$$
\hat{F}_i = \frac{1}{N} \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2
$$

使用 $N$ 个伪样本 (默认 64) 估计每个参数的重要性。

> 来源: `forgetting/ewc.py:53-87`

### 4.3 蓄水池采样 — Experience Replay (Reservoir Sampling)

对于第 $t$ 个到达的记忆:

$$
P(\text{replace}) = \begin{cases}
1 & \text{if } |\text{buffer}| < K \\
\frac{K}{t} & \text{otherwise, replace random position}
\end{cases}
$$

- $K$: 缓冲区大小 (默认 256)

保证每条历史记忆被保留的概率相等，无论到达顺序。

> 来源: `forgetting/replay.py:36-46`

---

## 5. Transformer 解码器

### 5.1 缩放点积注意力 (Scaled Dot-Product Attention)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- $d_k$: 每个注意力头的维度
- 自注意力使用因果掩码 (下三角矩阵)，确保自回归性质

> 来源: `transformer/model.py:36-42`

### 5.2 记忆交叉注意力 (Memory Cross-Attention)

$$
\text{CrossAttn}(Q_{tokens}, K_{mem}, V_{mem}) = \text{softmax}\left(\frac{Q_{tokens} K_{mem}^T}{\sqrt{d_k}} + M_{mask}\right) V_{mem}
$$

- $Q_{tokens}$: 来自 token 序列
- $K_{mem}, V_{mem}$: 来自检索到的记忆向量
- $M_{mask}$: 记忆填充掩码 (无效位置设为 $-\infty$)

> 来源: `transformer/model.py:46-93`

### 5.3 层归一化 (Layer Normalization)

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

采用 Pre-LN 架构: $x = x + \text{Attn}(\text{LN}(x))$

> 来源: `transformer/model.py`

### 5.4 奖励加权交叉熵损失 (Reward-Weighted Cross-Entropy)

$$
\mathcal{L}_{token} = \text{CE}(\hat{y}_t, y_t) = -\log P(y_t | y_{<t}, \text{mem})
$$

$$
\mathcal{L}_{sample}^{(i)} = \frac{\sum_t \mathcal{L}_{token}^{(i,t)} \cdot \mathbb{1}[y_t \neq \text{PAD}]}{\sum_t \mathbb{1}[y_t \neq \text{PAD}]}
$$

$$
w_i = \max(0.1, \; 1 + r_i)
$$

$$
\mathcal{L}_{total} = \frac{\sum_i \mathcal{L}_{sample}^{(i)} \cdot w_i}{\sum_i w_i}
$$

Query 部分标签设为 $-100$ (ignore), 仅对 response 部分计算损失。

> 来源: `transformer/train.py:119-133`

---

## 6. 文本生成采样

### 6.1 温度缩放 (Temperature Scaling)

$$
\text{logits}' = \frac{\text{logits}}{T}
$$

- $T > 1$: 更均匀 (更随机)
- $T < 1$: 更尖锐 (更确定)
- 默认 $T = 0.8$

> 来源: `transformer/generate.py:39-40`

### 6.2 Top-K 采样

$$
P'(w) = \begin{cases}
P(w) & \text{if } w \in \text{top-}K \\
0 & \text{otherwise}
\end{cases}
$$

保留概率最高的 $K$ 个 token (默认 $K=50$)，其余设为 $-\infty$。

> 来源: `transformer/generate.py:42-45`

### 6.3 核采样 — Top-P / Nucleus Sampling

$$
V_p = \arg\min_V \left\{ \sum_{w \in V} P(w) \geq p \right\}
$$

$$
P'(w) = \begin{cases}
P(w) & \text{if } w \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

从累积概率刚好超过 $p$ (默认 0.9) 的最小 token 集合中采样。

> 来源: `transformer/generate.py:48-54`

---

## 7. 编码器

### 7.1 L2 归一化

$$
\hat{v} = \frac{v}{\|v\| + \epsilon}, \quad \epsilon = 10^{-8}
$$

所有编码器输出统一归一化到单位球面上。

> 来源: `encoder/projection.py:74-79`

### 7.2 正交初始化 (Orthogonal Initialization)

投射层权重矩阵 $W$ 初始化为正交矩阵: $W^T W = I$

保证初始投射不会丢失信息或产生数值问题。

> 来源: `encoder/projection.py:40`

---

## 8. 正则化

### 8.1 梯度裁剪 (Gradient Clipping)

$$
g' = \frac{g}{\max\left(1, \; \frac{\|g\|}{g_{max}}\right)}
$$

- $g_{max} = 1.0$

> 来源: `shadow/manager.py:83-84`, `transformer/train.py:137-138`

### 8.2 权重衰减 (Weight Decay — AdamW)

$$
\theta_{t+1} = \theta_t - \eta \left(\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda_{wd} \cdot \theta_t\right)
$$

- $\lambda_{wd} = 0.01$
- $\beta_1 = 0.9, \; \beta_2 = 0.95$ (Transformer), $\beta_2 = 0.999$ (Shadow)

> 来源: `transformer/train.py:79-83`, `shadow/manager.py`

---

## 9. 参数默认值速查

| 符号 | 参数 | 默认值 | 公式出现位置 |
|------|------|--------|------------|
| $\lambda_{decay}$ | `decay_rate` | 0.01 | 指数衰减 |
| $\theta_{merge}$ | `merge_similarity_threshold` | 0.8 | 时间积分固化 |
| $\theta_{consolidation}$ | `consolidation_threshold` | 0.3 | 固化最低强度 |
| $\theta_{prune}$ | `prune_threshold` | 0.1 | 剪枝最低强度 |
| $\theta_{promotion}$ | `promotion_threshold` | 0.7 | Warm→Hot 晋升 |
| $T_{demotion}$ | `demotion_interval_sec` | 300s | Warm→Cold 降级 |
| $\alpha_{reward}$ | `reward_weight` | 0.5 | 唤醒评分 |
| $\alpha_{EMA}$ | `shadow_ema_alpha` | 0.01 | 跨进程权重同步 |
| $\lambda_{wd}$ | `shadow_weight_decay` | 0.01 | 影子权重衰减 (AdamW) |
| $\tau_{contrast}$ | temperature | 0.07 | 对比学习 |
| $\lambda_{EWC}$ | `ewc_lambda` | 1000.0 | EWC 惩罚 |
| $K_{reservoir}$ | `replay_buffer_size` | 256 | 蓄水池采样 |
| $T_{decode}$ | `decoder_temperature` | 0.8 | 生成采样 |
| $K_{topk}$ | `decoder_top_k` | 50 | Top-K 采样 |
| $p_{nucleus}$ | `decoder_top_p` | 0.9 | 核采样 |
| $g_{max}$ | max_grad_norm | 1.0 | 梯度裁剪 |
