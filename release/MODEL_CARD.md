# NeuroStream V4 Phase 1 — Sanitized Model Weights

> **Release tag**:`v0.2.0-phase1`
> **File**:`snapshot_best_sanitized.pt`
> **Size**:566.1 MB (593,587,033 bytes)
> **SHA256**:`3e546499746443642951ddc0e589a10fdf0a188e911eb2ef9b7e00aa268e4323`
> **License**:**Proprietary — All Rights Reserved**(展示用途,见 [LICENSE](../LICENSE))

---

## 1. 是什么 / What is this?

NeuroStream V4 Phase 1 训练得到的 best snapshot 的**脱敏版本**。

- **包含**:97M 参数的 Memory-Conditioned Transformer **模型权重(fp32)** + 训练步数信息 + 架构元数据
- **不包含**:训练时累积的 224,928 条 memories(已移除);optimizer / scheduler / scaler 状态(已移除)

发布目的:展示 NeuroStream 框架的训练成果(V4 PPL 8.5),供学术讨论与同行评估。**不构成可直接使用的推理服务**,因为运行所需的 `neurostream/` 源码(双进程引擎、影子权重 EMA 同步、Memory-Conditioned cross-attention 实现等)未开源。

---

## 2. 训练背景

| 项 | 值 |
|---|---|
| 模型类型 | MemoryConditionedTransformer(GPT-style + Cross-Attention) |
| 参数量(可训练) | 97M |
| 参数量(含 buffer / lm_head) | 148.4M(state_dict 全部) |
| 层数 / dim / heads / FFN | 12 / 512 / 8 / 1366(SwiGLU) |
| 位置编码 | RoPE(Rotary Position Embedding) |
| 归一化 | RMSNorm |
| 序列长度 | 512 tokens |
| Tokenizer | tiktoken cl100k_base(词表 100,277) |
| 训练数据 | 250K 医学对话(50K 英文 + 200K 中文) |
| **预训练** | ❌ **无**,完全 from-scratch |
| 训练框架 | NeuroStream 双进程持续学习(memory + shadow EMA + cross-attn 全在线参训) |
| 抗灾难性遗忘 | EWC(λ=500) |
| 训练步数 | 42,689 decoder steps |
| Best 触发步 | step **40,176**(val_loss 早停 patience=5) |
| **Best val_loss** | **2.1453**(cross-entropy,token 级) |
| **Best perplexity** | **≈ 8.5** |
| 训练时长 | 22.99 hours |
| 硬件 | NVIDIA RTX 5060 8GB(Blackwell sm_120, CUDA 12.8) |

### 与 V3 baseline 的对比

| | V3(纯 GPT-style) | V4(本模型) | 提升 |
|---|---|---|---|
| 参数 | 97M | 97M | 同 |
| 数据 | 250K 医学对话 | 250K 医学对话 | 同 |
| 预训练 | ❌ from-scratch | ❌ from-scratch | 同 |
| 训练框架 | 标准监督 LM | NeuroStream 双进程 + memory + cross-attn | **唯一变量** |
| Val PPL | 42.1 | **8.5** | **5× 提升,纯架构贡献** |

详见 [docs/experiments/v4_phase1_ablation.md](../docs/experiments/v4_phase1_ablation.md)。

---

## 3. 评估指标(qwen3-max LLM Judge, 30 样本)

| 维度 | V3 | V4(本模型) | Δ |
|---|---|---|---|
| factual_accuracy | 4.93 | 5.43 | **+10.1%** |
| relevance | 4.03 | 4.10 | +1.7% |
| completeness | 2.87 | 2.60 | -9.4% |
| **logic_safety** | 4.77 | **5.83** | **+22.2%** ← 最大改善 |
| fluency | 7.50 | 7.40 | -1.3% |
| **Overall Weighted** | 0.4527 | **0.4790** | **+5.8%** |

详细评估见 `output/phase1_v2/eval_report.json`(随 [docs](../docs/experiments/v4_phase1_ablation.md) 同步发布)。

---

## 4. 脱敏处理细节

### 4.1 被移除的内容

| 字段 | 原 snapshot | 脱敏后 | 原因 |
|---|---|---|---|
| `memories` | 224,928 条记忆(向量 + 时间戳 + 强度 + 元数据) | `[]`(空列表) | **训练数据指纹**:记忆向量经反向解码可能恢复出原始医学对话片段,涉及训练数据隐私 |
| `decoder_state.scaler` | fp16 GradScaler 状态 | 删除 | 没源码用不上,无需暴露 |
| `decoder_state.scheduler` | 学习率调度状态 | 删除 | 同上 |
| `decoder_state.optimizer` | AdamW momentum / variance | 删除(原 snapshot 已无,避免 PyTorch zip 2GB 限制) | 同上 |

### 4.2 保留的内容

- `decoder_state.model`(完整 fp32 模型权重)— 用于展示 NeuroStream 架构产物
- `decoder_state.step_count = 40178`(训练步数,展示训练量)
- `metadata`(脱敏标记 + 架构信息 + 训练 step)

### 4.3 验证脱敏

```python
import torch
ckpt = torch.load("snapshot_best_sanitized.pt", map_location="cpu", weights_only=False)
assert len(ckpt["memories"]) == 0, "脱敏失败:memories 未清空"
assert ckpt["metadata"]["tag"] == "sanitized"
print(f"Step: {ckpt['decoder_state']['step_count']}")
print(f"State keys: {list(ckpt['decoder_state']['model'].keys())[:5]}")
```

---

## 5. 如何使用(以及为什么不能直接用)

### 5.1 你可以做什么

- ✅ 加载 `state_dict` 检查参数分布、权重统计、可视化注意力模式
- ✅ 用作研究目的的 fingerprint(证明特定时刻的 NeuroStream 实例存在)
- ✅ 学术对比的硬数据(同 97M / 同 250K / 同 from-scratch 下取得 PPL 8.5 的模型权重)

### 5.2 你不能做什么

- ❌ 直接加载到生产推理服务 —— 缺 `neurostream/runtime/`(双进程引擎)、`neurostream/shadow/`(影子权重)、`neurostream/transformer/` 等核心运行时
- ❌ 继续训练 —— 缺 `neurostream/runtime/learning.py`(学习进程)、`train.py`(训练入口)
- ❌ 复现实验 —— 同上

这是**主动设计选择**:NeuroStream 的架构创新点(双进程异步训练、影子权重 EMA 跨进程同步、Memory-Conditioned cross-attention 实现)属于 `Proprietary` 范围,仅存于私有仓库。

### 5.3 学术/合作访问

如需在学术或合作场景下完整复现,请联系版权方,详见 [README.md](../README.md) 末尾。

---

## 6. 局限性与风险

1. **仅医学领域** — 训练数据 100% 医学对话,在通用领域(法律 / 编程 / 日常)上无法泛化(预期行为,非缺陷)
2. **非 SOTA 通用 LLM** — 97M 参数远不足以替代 GPT-4 / Claude 类通用模型,NeuroStream 也不以此为目标(详见 [docs/positioning.md](../docs/positioning.md))
3. **诊疗安全性有限** — 即使 LLM judge logic_safety 5.83/10,**模型输出不应作为实际医疗建议**,仅供学术展示
4. **In-domain shortcut 可能存在** — 已用 `diagnose_shortcut_v2.py` 跑过 3 项 in-domain 测试(C2.2 / C2.3 / C2.4),反事实记忆注入危险词 0/5 命中,架构在 in-domain 是健康的;但**任何 250K 单域 from-scratch 模型在跨领域上都会失败**

---

## 7. 引用

```bibtex
@misc{neurostream_v4_phase1_weights,
  title  = {NeuroStream V4 Phase 1 — Sanitized Model Weights},
  author = {Huang, Zhipeng},
  year   = {2026},
  month  = {5},
  url    = {https://github.com/642463401/NeuroStream/releases/tag/v0.2.0-phase1},
  note   = {97M params, from-scratch on 250K medical dialogues, val PPL 8.5, sanitized (memories removed)}
}
```

---

## 8. 相关资源

- [README.md](../README.md) — 项目总览
- [docs/positioning.md](../docs/positioning.md) — 项目定位:边缘 + 大模型双端适配
- [docs/experiments/v4_phase1_ablation.md](../docs/experiments/v4_phase1_ablation.md) — V3 vs V4 完整 ablation 报告
- [docs/architecture.md](../docs/architecture.md) — 系统架构
- [LICENSE](../LICENSE) — 专有许可证(All Rights Reserved)
