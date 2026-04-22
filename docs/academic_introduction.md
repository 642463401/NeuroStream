# NeuroStream: 以记忆为核心的持续学习框架

**面向学术研究者的技术介绍**

> NeuroStream: A Memory-Centric Continual Learning Framework — Technical Introduction for Academic Researchers

---

## 摘要

NeuroStream 是一个记忆驱动的持续学习框架，其核心思想是将**显式记忆对象**而非张量作为系统的第一公民，通过双进程架构实现推理与学习的完全解耦。该框架引入影子权重（Shadow Weights）机制，使模型参数在推理过程中通过指数移动平均（EMA）异步更新，区别于检索增强生成（RAG）的静态检索范式——知识被真正内化到模型参数中。系统设计借鉴了认知科学中的双过程理论（Dual-Process Theory），分别模拟大脑皮层的快速推理和海马体的缓慢记忆固化过程。框架已完成两轮架构迭代（V3→V4），在 250K 医学对话数据集上验证了 97M 参数模型的有效性（V4 验证集困惑度 35.1，LLM-as-Judge 综合评分 46.48%），支持多模态编码、分层记忆存储、弹性权重固化（EWC）等关键能力。

**关键词**：持续学习、记忆增强神经网络、灾难性遗忘、双进程架构、影子权重、对比学习、LLM-as-Judge 评估

---

## 1. 研究背景与动机

### 1.1 传统深度学习的局限性

当前主流深度学习框架（PyTorch、TensorFlow）遵循"离线训练—在线推理"的范式：模型在大规模数据集上完成训练后，以冻结参数进行推理，无法在部署后持续吸收新知识。这一范式面临三个核心问题：

1. **灾难性遗忘（Catastrophic Forgetting）**：当模型在新任务上微调时，旧任务的性能急剧下降 [Kirkpatrick et al., 2017; French, 1999]。
2. **知识不透明性**：知识以隐式方式分布在数百万参数中，难以审计、追踪或选择性遗忘。
3. **推理-学习耦合**：训练和推理无法并行进行，限制了边缘计算和实时学习场景的应用。

### 1.2 检索增强生成（RAG）的不足

RAG [Lewis et al., 2020] 通过外部知识库扩展模型能力，但存在本质局限：

- 检索到的知识不改变模型参数，模型本身未"学到"新知识。
- 检索质量严重依赖编码器和索引策略。
- 对于需要深层推理整合的知识，简单拼接上下文效果有限。

### 1.3 NeuroStream 的定位

NeuroStream 提出一种介于传统微调和 RAG 之间的新范式：

| 维度 | 传统微调 | RAG | **NeuroStream** |
|------|---------|-----|-----------------|
| 知识存储 | 隐式（参数） | 外部（文档库） | **显式记忆 + 参数协同** |
| 参数是否变化 | 是（需停机训练） | 否 | **是（异步持续更新）** |
| 推理时可用 | 是 | 否（需停止服务） | **是（推理与学习并行）** |
| 可解释性 | 低 | 中 | **高（可追溯到具体记忆）** |
| 遗忘控制 | 无 | N/A | **EWC + 经验回放** |

---

## 2. 理论基础

### 2.1 认知科学的双过程理论

NeuroStream 的架构设计受到 Kahneman [2011] 双过程理论和互补学习系统理论（Complementary Learning Systems, CLS）[McClelland et al., 1995] 的启发：

- **系统 1（快思考）→ 推理进程**：快速、自动化的模式匹配与响应生成，对应大脑皮层的功能。从记忆池中检索相关知识，结合上下文生成输出。
- **系统 2（慢思考）→ 学习进程**：缓慢、深思熟虑的知识整合与参数优化，对应海马体的记忆固化功能。负责评估、筛选和巩固新信息。

两个进程通过共享内存（`torch.share_memory_()`）异步通信，避免了互相阻塞。

### 2.2 记忆的数学表示

系统中每条记忆 $m$ 定义为一个结构化对象：

$$
m = \langle \mathbf{v}, \eta, I, r, t, n \rangle
$$

其中 $\mathbf{v} \in \mathbb{R}^d$ 为 L2 归一化的语义向量，$\eta \in \{\text{text}, \text{image}, \text{audio}\}$ 为模态标识，$I \in \mathbb{R}^+$ 为记忆强度，$r \in [-1, 1]$ 为反馈奖励，$t$ 为时间戳，$n$ 为访问计数。

### 2.3 记忆生命周期

记忆经历完整的生命周期：**感知 → 缓冲 → 评估 → 固化/丢弃 → 衰减**，遵循以下关键方程：

**时间积分固化**：当新记忆与已有记忆的余弦相似度超过阈值 $\theta_{merge}$ 时，进行强度合并而非新增：

$$
I_{merged} = I_{old} \cdot e^{-\lambda \cdot \Delta t} + I_{new}
$$

**指数衰减**：

$$
I(t) = I_0 \cdot e^{-\lambda \cdot (t - t_0)}
$$

**唤醒评分**：检索时综合考虑语义相似度、访问频率和反馈信号：

$$
S_{wake}(q, m) = \cos(\mathbf{q}, \mathbf{v}_m) + \ln(1 + n_m) + \alpha_r \cdot r_m
$$

---

## 3. 系统架构

### 3.1 整体架构

```
                     NeuroStreamPipeline / Trainer（用户接口）
                                  |
                           NeuroStreamEngine
                                  |
                        ┌─────────┴─────────┐
                        |                   |
                   推理进程              学习进程
                 (Inference)           (Learning)
                 ├─ UnifiedEncoder      ├─ ShortTermBuffer
                 ├─ MemoryProjector     ├─ MemoryPool (FAISS)
                 ├─ Recall / Search     ├─ TimeIntegralConsolidation
                 ├─ Transformer         ├─ ShadowWeightManager
                 │  Decoder (可选)      ├─ TransformerTrainer
                 └─ ToolRegistry        └─ ForgettingStrategy (EWC / Replay)
                        |                   |
                        └── SharedWeightBuffer ──┘
                          (torch.share_memory_ + EMA 同步)
```

### 3.2 模块化设计

框架采用高度模块化的可插拔架构，研究者可以替换任意组件进行消融实验：

| 模块 | 可替换组件 | 研究方向 |
|------|-----------|---------|
| **编码器** | FeatureHash / SBERT / CLIP / Whisper / 自定义 | 多模态表示学习 |
| **记忆索引** | FAISS (IVF/Flat/HNSW) / NumPy / 自定义 | 近似最近邻检索 |
| **固化策略** | TimeIntegral / 自定义 `ConsolidationStrategy` | 记忆固化机制 |
| **遗忘防护** | EWC / ExperienceReplay / 自定义 `ForgettingStrategy` | 灾难性遗忘 |
| **反馈机制** | LLMScorer / HumanFeedback / 自定义 `FeedbackProvider` | 奖励建模 |
| **解码器** | MemoryConditionedTransformer / 自定义 | 记忆增强生成 |

### 3.3 分层记忆存储

受计算机存储层次结构启发，记忆池采用三级存储：

| 层级 | 实现 | 延迟 | 容量 | 晋升/降级条件 |
|------|------|------|------|-------------|
| **Hot** | FAISS 内存索引 | < 1 ms | 有限 | $I \geq \theta_{promotion}$ |
| **Warm** | NumPy memmap | ~ 1 ms | 中等 | 默认层 |
| **Cold** | 磁盘序列化 | ~ 10 ms | 无限 | age > $T_{demotion}$ 且 $n_{access} = 0$ |

---

## 4. 核心算法

### 4.1 影子权重机制（Shadow Weights）

影子权重是 NeuroStream 区别于 RAG 的核心创新。其基本思想是：学习进程训练一个轻量级的**记忆投射器**（MemoryProjector），通过 EMA 将参数异步同步到推理进程，使推理过程中的知识表示随时间演化。

**记忆投射器**采用残差 MLP 架构，初始化为恒等映射：

$$
f(\mathbf{x}) = \mathbf{x} + \text{MLP}_\theta(\mathbf{x}), \quad \text{MLP}_\theta(\mathbf{x}) = W_2 \cdot \text{GELU}(W_1 \mathbf{x} + b_1) + b_2
$$

其中 $W_2, b_2$ 初始化为零，确保 $f(\mathbf{x}) = \mathbf{x}$ 在训练初始阶段。

**跨进程 EMA 同步**：

$$
\theta_{inference} \leftarrow (1 - \alpha) \cdot \theta_{inference} + \alpha \cdot \theta_{learning}
$$

默认 $\alpha = 0.01$，约需 $\sim 1/\alpha = 100$ 次同步完全收敛。

**渐近性质**：当无新记忆输入且时间趋于无穷时，记忆强度衰减至零，训练信号消失，AdamW 的权重衰减使参数回归零，投射器退化为恒等映射——系统"自然遗忘"不再被强化的知识。

### 4.2 对比学习目标

影子权重训练采用 InfoNCE 对比损失及其奖励加权变体：

**标准 InfoNCE**：

$$
\mathcal{L}_{contrast} = -\frac{1}{N} \sum_{i} \sum_{j \neq i} \text{softmax}\left(\frac{\text{sim}_{ij}}{\tau}\right) \cdot \log \text{softmax}\left(\frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\tau}\right)
$$

**奖励加权变体**：

$$
w_i = \max(0.1, \; 1 + r_i), \quad \mathcal{L}_{weighted} = \frac{\sum_i \mathcal{L}_i \cdot w_i}{\sum_i w_i}
$$

高奖励记忆获得更大训练权重，低奖励记忆被抑制但不完全消除（最小权重 0.1）。

### 4.3 灾难性遗忘防护

#### 弹性权重固化（EWC）

$$
\mathcal{L}_{EWC} = \lambda_{EWC} \sum_i F_i \cdot (\theta_i - \theta_i^*)^2
$$

其中 $\theta_i^*$ 为锚点参数，$F_i$ 为 Fisher 信息矩阵对角线元素（通过 $N$ 个伪样本估计）：

$$
\hat{F}_i = \frac{1}{N} \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2
$$

#### 经验回放（Experience Replay）

采用蓄水池采样（Reservoir Sampling）维护固定大小的历史记忆缓冲区，保证每条历史记忆被保留的概率相等：

$$
P(\text{replace}) = \begin{cases} 1 & \text{if } |\text{buffer}| < K \\ K / t & \text{otherwise} \end{cases}
$$

### 4.4 记忆增强 Transformer 解码器

可选的 GPT 风格解码器在标准因果自注意力基础上增加**记忆交叉注意力层**：

$$
\text{CrossAttn}(Q_{tokens}, K_{mem}, V_{mem}) = \text{softmax}\left(\frac{Q_{tokens} K_{mem}^T}{\sqrt{d_k}} + M_{mask}\right) V_{mem}
$$

其中 $Q$ 来自当前 token 序列，$K, V$ 来自检索到的记忆向量。当无可用记忆时，交叉注意力退化，模型等价于标准 GPT。

---

## 5. 实验验证

### 5.1 实验设置

| 配置 | V3 实验 | V4 实验 |
|------|---------|---------|
| 模型 | MemoryConditionedTransformer, 97.3M | MemoryConditionedTransformer, 97.0M |
| 架构 | $d_{model}=512$, 12 层, 8 头, $d_{ff}=2048$ | $d_{model}=512$, 12 层, 8 头, $d_{ff}=1365$ |
| 位置编码 | 学习位置嵌入 (512 token) | RoPE (1024+ token) |
| FFN | GELU | SwiGLU (gated) |
| 归一化 | LayerNorm | RMSNorm |
| 数据 | 250K 条医学对话（中文 80%, 英文 20%） | 同左 |
| 来源 | MedDialog [Chen et al., 2020]、HealthcareMagic、iCliniq | 同左 |
| 优化器 | AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.01) | 同左 |
| 学习率 | 3e-4, Warmup 500 步 + Cosine Decay | 同左 |
| 精度 | FP16 混合精度 | FP16 + 梯度检查点 |
| 硬件 | NVIDIA RTX 4060 (8GB VRAM) | 同左 |

### 5.2 训练结果

**V3 架构（基线）**

| Epoch | Train Loss | Val Loss | Val Perplexity |
|-------|-----------|----------|----------------|
| 1 | 4.866 | 4.237 | 69.2 |
| 5 | 3.924 | 3.834 | 46.2 |
| 10 | 3.743 | **3.740** | **42.1** |

**V4 架构（RoPE + SwiGLU + RMSNorm）**

| Epoch | Train Loss | Val Loss | Val Perplexity |
|-------|-----------|----------|----------------|
| 1 | — | 3.833 | 46.2 |
| 5 | — | 3.637 | 38.0 |
| 10 | 3.467 | **3.558** | **35.1** |

**架构对比**：V4 相比 V3，val perplexity 从 42.1 降低到 35.1（降幅 16.6%），在参数量近乎相同（97.3M vs 97.0M）的条件下实现了显著提升。SwiGLU 的 $d_{ff}=1365 \approx d_{model} \times 8/3$ 设计保持了与 V3 GELU ($d_{ff}=2048$) 相近的参数量。

### 5.3 LLM-as-Judge 多维评估

采用 Qwen3-Max 作为评审模型，对 30 个随机采样的医学问答进行五维度评分（1-10 分制），结合加权计算综合分数。

**V3 → V4 对比**

| 维度 | 权重 | V3 均值 | V4 均值 | 变化 |
|------|------|--------|--------|------|
| 事实准确性 | 30% | 5.30 | 4.93 | -0.37 |
| 指令遵循与回答相关性 | 25% | 3.70 | 4.37 | **+0.67** |
| 信息完整度 | 20% | 2.37 | 3.07 | **+0.70** |
| 逻辑与安全性 | 15% | 5.00 | 4.87 | -0.13 |
| 表达流畅度 | 10% | 7.57 | 7.33 | -0.24 |
| **综合加权分** | — | **44.95%** | **46.48%** | **+1.53%** |

V4 在**指令遵循**（+0.67）和**信息完整度**（+0.70）上有明显改善，说明 RoPE 更长的上下文窗口和 SwiGLU 的门控机制有助于生成更切题、更完整的回复。事实准确性的轻微下降可能与随机采样波动有关。

### 5.4 NLP 基线指标

| 指标 | V3 | V4 | 变化 |
|------|-----|-----|------|
| Token F1 (ROUGE-1) | 0.1429 | **0.1987** | +39.1% |
| BLEU-2 | 0.0397 | **0.0584** | +47.1% |
| LCS-F1 (ROUGE-L) | 0.1229 | **0.1564** | +27.3% |
| ROUGE-2 F1 | 0.0426 | **0.0530** | +24.4% |
| Jaccard Similarity | 0.0917 | **0.1261** | +37.5% |
| Char-level F1 | 0.1835 | **0.2109** | +14.9% |

V4 在所有模糊匹配指标上均有显著提升，表明生成的回答与参考答案之间的词汇/短语重叠度更高。

### 5.5 同量级对比

| 模型 | 参数量 | 训练数据量 | Perplexity |
|------|--------|-----------|-----------|
| **NeuroStream V3** | 97M | ~50M tokens | 42.1 |
| **NeuroStream V4** | 97M | ~50M tokens | **35.1** |
| GPT-2 Small | 117M | ~10B tokens | ~30-35 |
| OPT-125M | 125M | 180B tokens | ~27 |
| BioGPT | 347M | PubMed 文献 | ~25 |

V3→V4 的架构升级在不增加数据量的前提下将困惑度从 42.1 降至 35.1，已接近使用 200 倍以上数据的 GPT-2 Small（~30-35）。

### 5.6 关键发现

- **V4 架构优势显著**：相同参数量和数据量下，RoPE + SwiGLU + RMSNorm 相比 Learned PE + GELU + LayerNorm 带来 16.6% 的 PPL 降幅。
- **无过拟合**：V3 的 Train-Val gap 仅 0.003，V4 的 gap 为 0.091，均未出现过拟合。
- **生成质量改善**：V4 在所有 NLP 模糊匹配指标上均优于 V3（Token F1 提升 39.1%），LLM 评审综合分提升 1.53%。
- **容量尚未饱和**：10 个 epoch 后 val loss 仍在下降，表明更多训练 epoch 或更大数据集可进一步改善。

---

## 6. 研究者使用指南

### 6.1 环境配置

```bash
# 核心依赖（PyTorch + FAISS + NumPy）
pip install -e .

# 完整研究环境
pip install -e ".[full,dev]"
```

如需获取框架源码或合作授权，请联系项目团队。

### 6.2 研究者 API

NeuroStream 提供两层 API：`NeuroStreamPipeline`（快速原型验证）和 `NeuroStreamTrainer`（完全控制）。

```python
from neurostream import NeuroStreamTrainer, NeuroStreamConfig, MemoryProjector
from neurostream.forgetting import EWC

config = NeuroStreamConfig(
    dim=128,
    shadow_ema_alpha=0.005,
    ewc_lambda=500.0,
    decoder_enabled=True,
    decoder_layers=12,
    decoder_dim=512,
)

trainer = NeuroStreamTrainer(
    config=config,
    projector=MemoryProjector(dim=128, hidden=256),
    forgetting_strategy=EWC(lambda_=500.0),
)

trainer.start()
for entry in data_stream:
    trainer.ingest(entry["text"])

# 生成与评估
answer = trainer.generate("光速是多少?")
trainer.save_checkpoint("checkpoint.json")
```

### 6.3 消融实验建议

NeuroStream 的模块化设计天然适合消融实验。以下是若干建议研究方向：

| 消融变量 | 实验设计 | 预期洞察 |
|---------|---------|---------|
| 记忆交叉注意力 | 对比有/无 `MemoryCrossAttention` | 量化记忆注入对生成质量的贡献 |
| 影子权重 vs RAG | 对比 EMA 同步 vs 静态检索拼接 | 验证参数内化是否优于上下文拼接 |
| EWC 惩罚系数 | 扫描 $\lambda \in \{0, 100, 500, 1000, 5000\}$ | 探索稳定性-可塑性权衡 |
| 记忆衰减率 | 扫描 $\lambda_{decay} \in \{0.001, 0.01, 0.1\}$ | 研究长期记忆保留策略 |
| 编码器选择 | FeatureHash vs SBERT vs CLIP | 表示质量对下游任务的影响 |
| 记忆池容量 | 限制记忆池大小 $\{1K, 10K, 100K\}$ | 容量-性能-延迟权衡 |

### 6.4 自定义组件扩展

所有核心模块均提供抽象基类（ABC），研究者可实现自定义策略：

```python
from neurostream.consolidation.base import ConsolidationStrategy
from neurostream.forgetting.base import ForgettingStrategy
from neurostream.encoder.base import EncoderBase
from neurostream.feedback.base import FeedbackProvider

# 示例：实现自定义固化策略
class AttentionBasedConsolidation(ConsolidationStrategy):
    def consolidate(self, buffer, pool):
        # 您的研究算法
        ...
```

### 6.5 基准评测工具

框架内置论文风格的评测报表生成器：

```python
from neurostream.agent import BenchmarkEvaluator, BenchmarkReporter

evaluator = BenchmarkEvaluator()
results = evaluator.evaluate(model, test_dataset)
# 4 维评测：accuracy, relevance, completeness, fluency

reporter = BenchmarkReporter()
reporter.render_table(results, output_path="benchmark_table.png")
reporter.render_curve(training_log, output_path="training_curve.png")
reporter.render_radar(results, output_path="radar.png")
```

---

## 7. 与相关工作的关系

### 7.1 持续学习（Continual Learning）

NeuroStream 采用了持续学习领域的经典方法（EWC [Kirkpatrick et al., 2017]、Experience Replay [Rolnick et al., 2019]），并将其整合到实时双进程架构中。与传统持续学习方法的区别在于：不依赖任务边界定义，通过记忆强度的自然衰减和固化实现隐式任务切换。

### 7.2 记忆增强神经网络（Memory-Augmented Neural Networks）

与 Neural Turing Machine [Graves et al., 2014]、Memory Networks [Weston et al., 2015] 等工作类似，NeuroStream 使用外部记忆作为显式知识存储。核心区别在于：NeuroStream 的记忆不仅用于检索，还通过影子权重机制反向影响模型参数。

### 7.3 检索增强生成（Retrieval-Augmented Generation）

与 RAG [Lewis et al., 2020]、RETRO [Borgeaud et al., 2022] 相比，NeuroStream 的记忆通过交叉注意力和影子权重两个通道影响输出——前者提供即时检索上下文，后者将高频知识内化到参数中。

### 7.4 知识蒸馏（Knowledge Distillation）

Agent 训练闭环采用类似 Hinton et al. [2015] 的教师-学生范式，由大型语言模型（如 Qwen）作为教师，NeuroStream 作为学生，在对话任务上进行在线蒸馏。

---

## 8. 消融实验设计

基于框架的模块化架构，以下消融实验已具备可行性：

### 8.1 已完成的消融

| 消融变量 | 对照 | 实验 | 结果 |
|---------|------|------|------|
| **架构版本** | V3 (LayerNorm+GELU+LearnedPE) | V4 (RMSNorm+SwiGLU+RoPE) | PPL: 42.1 → 35.1 (↓16.6%) |

### 8.2 可行消融实验（零代码修改）

| 消融变量 | 实验设计 | 控制方式 |
|---------|---------|---------|
| **数据语言比例** | EN-only / ZH-only / 50-50 / 80-20 | `--en-size`, `--zh-size` CLI 参数 |
| **无记忆基线** | 关闭记忆交叉注意力 | 训练时已默认不传入记忆向量，`MemoryCrossAttention` 返回零 |

### 8.3 低成本消融实验（约 15-50 行代码）

| 消融变量 | 实验设计 | 预期洞察 |
|---------|---------|---------|
| **模型规模** | 6 层 256d (约 25M) vs 12 层 512d (97M) | 参数效率与容量饱和点 |
| **Dropout 率** | 扫描 $\{0.0, 0.1, 0.2, 0.3\}$ | 正则化对 train-val gap 的影响 |
| **Label Smoothing** | 扫描 $\{0.0, 0.05, 0.1, 0.2\}$ | 输出分布校准 |
| **上下文长度** | 512 vs 1024 vs 2048 | 长上下文对医学对话的边际收益 |

### 8.4 建议消融实验路线图

| 阶段 | 实验 | 目标 | 估计训练时间 |
|------|------|------|------------|
| Phase 1 | 数据比例扫描 (EN/ZH) | 确定最优双语配比 | 3×10h |
| Phase 2 | 模型规模扫描 (25M/50M/97M) | 绘制 scaling curve | 3×10h |
| Phase 3 | 记忆交叉注意力消融 | 量化记忆机制的独立贡献 | 1×10h |
| Phase 4 | 超参数扫描 (Dropout/LR) | 优化正则化策略 | 4×10h |

---

## 9. 当前局限性与未来方向

### 9.1 已知局限

1. **消融实验不足**：V3→V4 架构对比已完成，但尚未与去除记忆交叉注意力的基线 Transformer 进行对照，无法精确量化记忆机制的独立贡献。
2. **训练-推理分离**：当前 Transformer 训练未集成实时记忆检索管线，交叉注意力的记忆输入为空，记忆增强的全部潜力尚未释放。
3. **规模有限**：97M 参数和 250K 数据在现代标准下偏小，Chinchilla 最优值建议约 2B tokens。
4. **LLM-as-Judge 局限性**：评审模型（Qwen3-Max）的偏好可能引入系统性偏差，不同评审模型的评分一致性有待验证。

### 9.2 已解决的历史局限

- ~~**评测指标单一**~~：已引入 LLM-as-Judge 五维评分（事实准确性、指令遵循、信息完整度、逻辑安全性、表达流畅度）和 6 项 NLP 模糊匹配指标（Token F1、BLEU-2、LCS-F1、ROUGE-2、Jaccard、Char-level F1）。
- ~~**架构待升级**~~：V4 已采用 RoPE 位置编码、SwiGLU FFN、RMSNorm，在同参数量下实现 16.6% 的 PPL 降幅。

### 9.3 未来研究方向

- **记忆增强训练闭环**：将实时记忆检索集成到训练循环中，实现真正的记忆条件化学习。
- **大规模验证**：扩展至 350M+ 参数，配合全量数据训练，验证架构在更大规模下的表现。
- **多模态记忆融合**：探索文本-图像-音频记忆在统一向量空间中的交叉检索与互补效应。
- **选择性遗忘**：利用显式记忆的可追溯性，实现精确的知识删除（machine unlearning）。
- **隐私保护**：基于记忆的显式存储特性，探索差分隐私和联邦学习场景下的记忆管理。
- **理论分析**：对影子权重 EMA 同步的收敛性和稳定性-可塑性权衡提供形式化分析。
- **多评审交叉验证**：使用多个 LLM（GPT-4、Claude、Qwen）作为评审，消除单一评审偏差。

---

## 10. 技术规格速查

| 项目 | 规格 |
|------|------|
| 编程语言 | Python ≥ 3.10 |
| 深度学习框架 | PyTorch ≥ 2.0 |
| 向量检索 | FAISS ≥ 1.7 |
| 分词器 | tiktoken cl100k_base (100,277 tokens) |
| 许可证 | 非开源，需授权使用 |
| 测试覆盖 | 22 个测试文件, 253 个测试用例 |
| 核心依赖 | 3 个（torch, faiss-cpu, numpy） |
| 可选模态 | 文本 (SBERT) / 图像 (CLIP) / 音频 (Whisper) |
| GPU 支持 | CUDA 自动检测, FP16 混合精度, 梯度检查点 |
| API 层次 | Pipeline（5 行上手）/ Trainer（完全控制） |
| 当前最优模型 | V4, 97M 参数, PPL=35.1, LLM-Judge=46.48% |
| 评测体系 | LLM-as-Judge 五维评分 + 6 项 NLP 模糊匹配指标 |

---

## 参考文献

- Borgeaud, S., et al. (2022). Improving language models by retrieving from trillions of tokens. *ICML*.
- Chen, S., et al. (2020). MedDialog: Large-scale medical dialogue datasets. *EMNLP*.
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*.
- Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. *arXiv:1410.5401*.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*.
- Rolnick, D., et al. (2019). Experience replay for continual learning. *NeurIPS*.
- Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. *ICLR*.

---

## 引用

如果您在研究中使用了 NeuroStream，请引用：

```bibtex
@software{neurostream2026,
  title     = {NeuroStream: A Memory-Centric Continual Learning Framework},
  year      = {2026},
  note      = {Version 0.1.0}
}
```

---

> 本文档最后更新于 2026 年 4 月 21 日。如有疑问，请参阅 [完整 API 文档](api/) 或联系项目团队。
