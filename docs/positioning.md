# NeuroStream 项目定位

> **一句话定位**:NeuroStream 是**面向边缘端 + 大模型补充层的记忆型持续学习框架**,以"边推理边学习 + 显式记忆 + 双系统"为核心差异化,**不与通用 LLM 正面竞争**,而是补充 SOTA LLM 难以解决的隐私、状态化、个性化场景。
>
> 文档目的:对外宣传、合作交流、招募协作者时的统一话术基础。

---

## 1. 项目不做什么(同样重要)

明确不做这些,可以避免被错位评判:

| 不做 | 为什么 |
|---|---|
| ❌ **替代 ChatGPT / Claude 这类通用对话 LLM** | 它们在"通用知识 + 大规模预训练"上有十亿美元级护城河,正面对抗无胜算 |
| ❌ **在 MMLU / HumanEval 等通用 benchmark 上打榜** | NeuroStream 不是为这些 benchmark 设计的 |
| ❌ **追求 100B+ 参数的"大模型"路线** | 这条路是大厂专属,个人 / 小团队进不去也没必要进 |
| ❌ **声称"自研神经网络算法"** | V4 仍基于 Transformer 主干,本项目的创新在系统组合 + 持续学习管道,而非底层算法 |
| ❌ **取代 RAG** | RAG 适合 "我有海量已知文档要检索" 场景,与 NS 的 "我要让模型自己持续学习内化" 是互补不是替代 |

---

## 2. 项目做什么

### 核心定位:边缘端 + 大模型双端适配的记忆型持续学习框架

**两种用法,同一套架构**:

```
┌─────────────────────────────────────────────────────────────┐
│  用法 A · 边缘端独立部署                                      │
│  ─────────────────────────                                  │
│  97M 参数 · fp16 ≈ 200MB · 移动 GPU / 中端 CPU 可跑          │
│  完全本地运行 · 不依赖云端 · 数据隐私零外流                  │
│  适用:个人助理 / 设备端 AI / 隐私敏感场景                   │
│                                                              │
│  典型场景:                                                  │
│    - 医疗设备的本地诊断辅助(数据不出院区)                  │
│    - 个人设备上的长期记忆 AI 助手                            │
│    - 离线 / 弱网络环境的智能服务                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  用法 B · 大模型补充层(协作)                                │
│  ─────────────────────────                                  │
│  作为 GPT-4 / Claude / Qwen 等 SOTA LLM 的:                  │
│    · 长期记忆层(跨会话状态保留)                            │
│    · 个性化层(每个用户独立的持续学习实例)                  │
│    · 隐私缓冲层(敏感信息在本地处理,不送云端)              │
│                                                              │
│  典型集成:                                                  │
│    主对话 = SOTA LLM(强通用能力)                          │
│    个性化记忆 + 状态管理 = NeuroStream(轻量、本地、可学)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心技术差异化(对外解释一句话版)

> "在持续学习场景下,我们用 Memory + Cross-Attention + 双进程异步训练,让小模型(97M)做到大模型(347M+ BioGPT)级别的领域表现,**且无需大规模预训练**。"

### 三个具体差异点

#### 差异 1 · 不是 RAG,是参数化记忆 + 检索协同

| 维度 | 传统 RAG | NeuroStream |
|---|---|---|
| 知识在哪里 | 完全在外部文档库 | **同时在记忆池 + 模型参数中** |
| 知识进入方式 | 检索后拼到 prompt 里 | **通过 Cross-Attention 内化进模型** |
| 模型本身会变吗 | ❌ 不变 | ✅ 影子权重 EMA 持续更新 |
| 是否需要预训练大模型 | ✅ 强依赖 | ❌ **from-scratch 也能工作**(V4 PPL 8.5 from scratch) |

#### 差异 2 · 边推理边学习,非"训练-部署"两阶段

```
传统 LLM:                  NeuroStream:
[训练 100T tokens]          [推理 + 学习同时在跑]
    ↓                           ↑↓
[部署到推理]                ConversationBuffer
    ↓                           ↓
推理(权重冻结)            学习进程持续训练
                                ↓
                            EMA 同步回推理进程
```

工程实现:推理 / 学习两个独立进程,通过 `SharedWeightBuffer` 的 `torch.share_memory_()` 做 EMA 同步。**推理永不阻塞,学习永不停止**。

#### 差异 3 · 生物启发的双系统架构

| 系统 | 角色 | 实现 |
|---|---|---|
| **皮层**(慢、统计、密集) | 长期参数化记忆 | Transformer 主干 + Cross-Attention |
| **海马体**(快、片段、稀疏) | 短期 episodic 记忆 | FAISS Hot/Warm/Cold 三层 + 时间衰减 |

设计灵感来自神经科学的 **Complementary Learning Systems**(McClelland 1995,30 年理论背书)。

---

## 4. 硬数据(对外可量化的优势证据)

详见 [`experiments/v4_phase1_ablation.md`](experiments/v4_phase1_ablation.md)。

| 指标 | 数据 | 说明 |
|---|---|---|
| **架构有效性** | V3 PPL 35.10 → **V4 PPL 8.54** | 同 97M / 同 250K / 同 from-scratch,纯架构改造,**4.11× 提升** |
| **数据效率** | 250K 对话 vs BioGPT 数十亿 tokens | 数据量少 4 个数量级,临床表现接近 |
| **参数效率** | 97M vs BioGPT 347M | 参数少 3.5×,可部署到边缘端 |
| **In-domain shortcut 健康度** | 反事实记忆注入 0/5 命中 | Cross-Attn 不被错记忆误导,架构是健康的 |
| **持续学习能力** | 在线 ingest + teach 不阻塞推理 | 真正的 streaming learning,非批量微调 |

---

## 5. 当前阶段与路线(对外坦诚)

### 已完成(2026-05)
- ✅ 完整框架开源(neurostream/ 包,253 测试通过)
- ✅ V4 Phase 1 训练验证(95M 在 250K 医学对话上跑通 from-scratch)
- ✅ 双进程异步训练 + 影子权重 EMA 同步 + 抗灾难性遗忘
- ✅ In-domain shortcut 健康诊断

### 进行中(2026 Q2-Q3)
- 🔄 Phase 2 持续学习实验(测真实"无遗忘"能力)
- 🔄 Continual learning benchmark 接入(Permuted MNIST / Split CIFAR)
- 🔄 严格组件 ablation(消融 cross-attn / shadow / EWC 各自贡献)
- 🔄 OpenAI 兼容 API 服务化(`/v1/chat/completions` 端点)

### 未来(2026 Q4 起)
- ⏳ 自研架构探索 — V4 仍是 Transformer 主干,**自研架构是 v0.3+ 阶段的研究方向**
- ⏳ 候选替代家族:State Space Models(Mamba)/ Modern Hopfield / DNC / CLS-inspired
- ⏳ 多 GPU 训练支持

---

## 6. 对外答疑标准话术

### Q: 这跟 ChatGPT / Claude 比怎么样?
> **不在一个赛道**。ChatGPT 是通用 LLM,有云端、大规模预训练、万亿参数能力。NeuroStream 是边缘端可部署的持续学习框架,目标是**互补**,不是替代。我们提供 GPT-4 不愿/不能做的:本地、隐私、状态化、个性化、持续学习。

### Q: 这不就是 RAG 吗?
> **不是**。RAG 是把检索结果拼到 prompt 里,模型本身不变。NeuroStream 通过 Cross-Attention 把记忆**内化进模型参数**,影子权重 EMA 让模型在线持续演化。**RAG 是查表,NeuroStream 是学习**。

### Q: 你自研了 Transformer 吗?
> **没有**,我也不会这么声称。**V4 仍基于 Transformer 主干**(GPT-style Decoder + RoPE + SwiGLU + RMSNorm 是 LLaMA 系列的现代实践)。NeuroStream 的创新在 **系统组合**(Memory + Cross-Attn + 双进程 + 持续学习管道),不在底层算法。**自研架构是后续 v0.3+ 阶段的研究方向**,目前还在打基础。

### Q: 为什么用 97M 这么小的模型?
> 因为目标是**边缘可部署**。97M fp16 ≈ 200MB,移动 GPU / 中端 CPU 可跑。即使在这个规模下,V4 通过架构改造在 from-scratch 训练上把 perplexity 从 V3 的 35.1 降到 8.54,**证明小模型 + 好架构 + 持续学习 ≈ 大模型微调**。这是项目的核心赌注。

### Q: 跟 Mamba / RWKV 这些替代架构有什么关系?
> **互补关系**。Mamba / RWKV 是替代 Transformer 主干的方向,NeuroStream 当前还基于 Transformer。未来 v0.3+ 会研究把主干换成 SSM 类的可能性 — 因为它们的"状态化"哲学与 NeuroStream 的"持续学习"哲学天然契合。但**这是长期方向**,不是短期承诺。

### Q: 你们的数据怎么来的?隐私问题?
> Phase 1 用公开医学对话数据集(MedDialog / HealthcareMagic),具体数据来源详见 [PROGRESS.md](../PROGRESS.md)。
> 项目本身的核心价值正是**让 AI 能在用户数据上本地持续学习,无需上传任何东西** — 这才是 NeuroStream 解决的真问题。

---

## 7. 合作与协作

### 当前欢迎的协作类型

- **持续学习 / 神经科学背景**的研究者:特别是 Complementary Learning Systems / 灾难性遗忘 / Episodic Memory 方向
- **边缘部署 / 量化压缩**专家:把 97M 进一步压到 50M 以下,支持更低端设备
- **替代架构爱好者**:对 Mamba / SSM / Modern Hopfield / DNC 等 post-Transformer 方向感兴趣的人
- **特定垂直领域**(医疗 / 法律 / 教育):提供领域数据 + 用户场景的合作

### 不适合的协作

- 期望快速商业化变现(目前是研究项目,还没产品化)
- 期望对标 GPT-4 通用能力(明确不做这个)
- 期望用大规模预训练路线(我们走小模型 + 架构创新路线)

---

## 8. 项目元信息

| 项 | 值 |
|---|---|
| 项目名 | NeuroStream |
| 当前版本 | v0.2.x(Phase 1 完成,Phase 2 就绪) |
| 仓库结构 | `main` 完整源码 / `public` 仅文档(精简发布) |
| License | 专有(All Rights Reserved,详见 [LICENSE](../LICENSE)) |
| 开发语言 | Python ≥ 3.10 + PyTorch ≥ 2.0 + CUDA(Blackwell+ 用 cu128) |
| 硬件要求 | 训练:RTX 5060+(8GB);推理:可降到 CPU |
| 主要协作渠道 | GitHub Issues / Discussions(待开放) |

---

## 9. 相关文档

- [`README.md`](../README.md) — 项目总览 + 快速上手
- [`PROGRESS.md`](../PROGRESS.md) — 详细进度 + 已修 bug
- [`architecture.md`](architecture.md) — 系统架构详解
- [`experiments/v4_phase1_ablation.md`](experiments/v4_phase1_ablation.md) — V3→V4 架构有效性 ablation(对外宣传的硬数据来源)
- [`experiments/medical_v3.md`](experiments/medical_v3.md) — V3 实验报告(对照组)
- [`learning/00_knowledge_map.md`](learning/00_knowledge_map.md) — 系统学习路径(开发者补 AI 基础用)

---

**致读者**:本文档持续更新。如需引用项目最新数据,请参考 [`PROGRESS.md`](../PROGRESS.md) 的"最后更新"日期。如对项目有疑问、合作意向或学术讨论,欢迎通过仓库 Issues 联系。
