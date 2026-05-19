# NeuroStream 知识地图

> 整套学习路径的"目录页"。看完这一页就知道:学什么、按什么顺序学、什么时候去看 [01_systematic.md](01_systematic.md) 还是 [02_project_codex.md](02_project_codex.md)。

---

## 两份教材怎么配合

```
┌─────────────────────────────────────────────────────────────┐
│   01_systematic.md  ←─  通用理论(B 站 + 论文)              │
│                          学完能去任何 AI 项目工作            │
│                                                              │
│              ↕  互相印证                                    │
│                                                              │
│   02_project_codex.md  ←─  对着 NeuroStream 代码精讲         │
│                              学完能完全掌控本项目            │
└─────────────────────────────────────────────────────────────┘
```

**推荐用法**:学一个 systematic 模块,立刻打开对应的 codex 章节看项目里怎么落地。理论 + 代码同时进入大脑,远比单读快。

---

## 完整知识依赖图

```
                          ┌─────────────────────────┐
                          │     L0  数学基础         │
                          │  线代 · 微积分 · 概率    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   L1  神经网络基础       │
                          │  MLP · 反向传播 · 优化器 │
                          └─────┬───────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │  L2 序列  │    │ L2 训练  │     │ L2 表征  │
        │   建模    │    │   工程   │     │   学习   │
        │ Token/LM │     │ AdamW/   │     │ Contrast │
        │  Causal  │    │ fp16/EMA │     │  InfoNCE │
        └────┬─────┘     └────┬─────┘     └────┬─────┘
             │                │                │
             └────────┬───────┴────────┬───────┘
                      ▼                ▼
            ┌──────────────────┐  ┌──────────────────┐
            │  L3 Transformer  │  │  L3 持续学习      │
            │  Self-Attn / FFN │  │  EWC / Replay    │
            │  RoPE / RMSNorm  │  │  Catastrophic    │
            │  Multi-Head      │  │  Forgetting      │
            └────────┬─────────┘  └─────────┬────────┘
                     │                      │
                     │   ┌──────────────┐   │
                     ├──▶│ L3 向量检索  │◀──┤
                     │   │ FAISS / ANN  │   │
                     │   │ Cosine Sim   │   │
                     │   └──────┬───────┘   │
                     │          │           │
                     └──────────┼───────────┘
                                ▼
                  ┌──────────────────────────────┐
                  │      L4  NeuroStream 整合     │
                  │  双进程 · 影子权重 · 边推边学  │
                  │  生物隐喻 · Memory + 参数协同 │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │  L5 Agent / 工具系统 (可选)   │
                  │  Tool Use · MCP · 蒸馏        │
                  └──────────────────────────────┘
```

---

## 13 个模块速查

每个模块的详细内容见 [01_systematic.md](01_systematic.md);每个模块对应的项目代码精讲见 [02_project_codex.md](02_project_codex.md)。

| # | 层级 | 模块 | 系统理论 | 项目代码 | 难度 |
|---|---|---|---|---|---|
| 1 | L0 | 线性代数与张量 | 系-§1 | 代-§1 (`types.py`) | ★☆☆ |
| 2 | L0 | 微积分与反向传播 | 系-§2 | — | ★★☆ |
| 3 | L0 | 概率与信息论 | 系-§3 | 代-§5 (`train.py:loss`) | ★★☆ |
| 4 | L1 | 神经网络构件 | 系-§4 | 代-§4 (`shadow/projector.py`) | ★★☆ |
| 5 | L1 | 训练循环与优化 | 系-§5 | 代-§5 (`transformer/train.py`) | ★★★ |
| 6 | L2 | 词元化与语言建模 | 系-§6 | 代-§5 (`tokenizer.py`) | ★★☆ |
| 7 | L3 | Transformer 架构 | 系-§7 | 代-§5 (`transformer/model.py`) | ★★★★ |
| 8 | L3 | 自回归生成 | 系-§8 | 代-§5 (`transformer/generate.py`) | ★★★ |
| 9 | L3 | 对比学习 | 系-§9 | 代-§4 (`shadow/objectives.py`) | ★★★ |
| 10 | L3 | 向量检索与记忆 | 系-§10 | 代-§3 (`memory/`) | ★★★ |
| 11 | L3 | 持续学习与抗遗忘 | 系-§11 | 代-§6 (`forgetting/`) | ★★★★ |
| 12 | L4 | 多进程异步系统 | 系-§12 | 代-§7 (`runtime/`) | ★★★★ |
| 13 | L5 | Agent / 工具(可选) | 系-§13 | 代-§9 (`agent/`, `tools/`) | ★★★ |

---

## 三阶段路径(预估 3-4 个月,每周 10-15h)

```
Week  1  2  3  4  5  6  7  8  9  10 11 12 13 14
       ├──A──┤├──────B──────┤├──────C──────┤├D┤

A (3w):  数学补漏 + 神经网络基础(L0 + L1)
         → 看 3Blue1Brown,跑通项目 5 行 demo

B (5w):  Transformer + 训练工程(L2 + L3 主干)
         → 看 Karpathy「Let's build GPT」+ 沐神 Transformer 精读
         → 读 transformer/model.py 全文

C (5w):  NeuroStream 差异化(L3 检索/持续学习/对比 + L4 双进程)
         → 读 forgetting/ + shadow/ + memory/ + runtime/
         → 自己改一个小特性,重训

D (1w):  Agent / 工具系统(可选)
         → 如果计划接入 OpenAI 兼容 API,加这层
```

---

## 学习节奏建议

### 每日

- 30 分钟视频(B 站,1.5 倍速)
- 30 分钟读项目代码(对照笔记标记)
- 30 分钟自己复现一个小片段(纸上或代码)

### 每周

- 周末把本周所学整合一篇笔记(标记到 [01_systematic.md](01_systematic.md) 对应章节末)
- 一个"自测问题"(在 [01_systematic.md](01_systematic.md) 每章末的"自测清单"里挑一个,写出完整答案)

### 每月

- 跑一次完整训练(`python train.py`)
- 修改一个小参数(如 `decoder_dropout`),观察 val_loss 变化
- 用纸把整个数据流(从 `engine.ingest` 到 `decoder_trainer.train_step`)画出来

---

## 阶段验收(里程碑)

| 阶段末 | 你应该能 |
|---|---|
| A 末(第 3 周) | 解释 `shadow/projector.py` 全部 50 行代码,讲清"残差 MLP 为何 zero-init" |
| B 末(第 8 周) | 不看代码画出 `MemoryConditionedTransformer` 单层的数据流(self-attn / cross-attn / FFN) |
| C 末(第 13 周) | 给陌生人讲清"NeuroStream 与 RAG 的本质区别在哪三层" |
| D 末(第 14 周) | 给项目加一个 OpenAI 兼容 `/v1/chat/completions` API |

---

## 主资源池(B 站)

> 给关键词,自己用 B 站搜索框搜出最新版本。BV 号有时会失效,UP 主名 + 课程名是最稳的。

### 必看(任何路径都不能跳)

| UP 主 / 频道 | 关键词 | 用途 |
|---|---|---|
| **3Blue1Brown 中文官方** | 线性代数的本质 / 神经网络 | 数学直觉 |
| **跟李沐学AI** | 动手学深度学习 / 论文精读 | 系统课 + 论文精讲 |
| **Andrej Karpathy 中文搬运** | Let's build GPT / makemore / tokenizer | 手搓代码 |
| **跟李宏毅老师学** | 2024 机器学习 / 生成式 AI | 通俗讲解(适合复盘) |

### 单项进阶

| 主题 | 推荐 UP 主 / 搜索词 |
|---|---|
| Transformer 论文精读 | 「跟李沐学AI」Transformer / Attention Is All You Need |
| GPT / BERT 系列 | 「跟李沐学AI」GPT 1/2/3 精读 / BERT 精读 |
| 持续学习 EWC | 搜「弹性权重固化 EWC」/「灾难性遗忘」 |
| 对比学习 InfoNCE | 「跟李沐学AI」SimCLR / MoCo / CLIP 精读 |
| 向量检索 FAISS | 搜「FAISS 原理」/「ANN 近似最近邻」 |
| RoPE 旋转位置编码 | 搜「RoPE 苏剑林」/「旋转位置编码」 |
| SwiGLU / RMSNorm | 搜「LLaMA 架构详解」 |
| 多进程 PyTorch | 搜「PyTorch DDP」/「multiprocessing share_memory」 |

### 论文精读(在 B 站找精讲再去读原文)

| 论文 | 重要程度 | 项目对应 |
|---|---|---|
| Attention Is All You Need (2017) | ★★★★★ | `transformer/model.py` |
| GPT-2 / GPT-3 论文 | ★★★★ | 整体架构 |
| LLaMA / LLaMA-2 | ★★★★ | RoPE + SwiGLU + RMSNorm |
| RoFormer (RoPE) | ★★★★ | `transformer/model.py` 位置编码 |
| EWC (Kirkpatrick 2017) | ★★★ | `forgetting/ewc.py` |
| SimCLR / InfoNCE (CPC) | ★★★ | `shadow/objectives.py` |
| CLIP (2021) | ★★★ | `encoder/image.py` |
| FAISS (2017) | ★★ | `memory/pool.py` |

---

## 下一步

打开 [01_systematic.md](01_systematic.md) 从 §1 开始。或者如果手头宽松,直接跳到 [02_project_codex.md](02_project_codex.md) 的 §1(读 `types.py`),看不懂的概念再回 systematic 查。

两份文档双向链接,可以随意跳。
