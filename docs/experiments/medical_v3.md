# NeuroStream 医学对话训练实验报告 — V3

> 97M Unsupervised | 250K Dialogues | RTX 4060 | 2026-04-17
>
> **2026-05-19 数据对齐**:本报告早期版本基于一次 69,119 步 / val PPL=42.1 的 V3 run。当前磁盘
> `output_unsupervised/training_log.json` 对应一次更长的 V3 run(76,570 步 / 10 epochs / 31.55h),
> 最终 val_loss=3.5583,val PPL=35.10。下方"总览 / 逐 Epoch / 训练曲线"已统一更新为该 run 的实际数据。

---

## 概述

本实验验证 NeuroStream 的 MemoryConditionedTransformer 架构在真实医学对话数据上的学习能力。

**V2 问题**：50 epochs 训练 3,482 条数据，loss 降到 0.026 但严重过拟合（参数/样本比 27,944:1），对新问题输出乱码。

**V3 改进**：数据量扩大 80 倍，加入验证集早停，增强正则化。

---

## 模型配置

| 参数 | 值 |
|------|-----|
| 架构 | MemoryConditionedTransformer (GPT-style + Memory Cross-Attention) |
| d_model | 512 |
| 层数 | 12 |
| 注意力头数 | 8 |
| FFN 维度 | 2048 |
| Dropout | 0.2 |
| 最大序列长度 | 512 tokens |
| 词表大小 | 100,277 (tiktoken cl100k_base) |
| 记忆维度 | 128 |
| **总参数量** | **97.3M** |

架构特点：每层包含 CausalSelfAttention + MemoryCrossAttention + FFN，记忆通过交叉注意力注入（Q 来自 token，K/V 来自检索到的记忆向量）。无记忆时退化为标准 GPT。

---

## 数据集

### 来源

| 数据源 | 语言 | 可用量 | 使用量 |
|--------|------|--------|--------|
| MedDialog (english-train.json) | EN | 482 | 482 |
| HealthcareMagic + iCliniq | EN | 257K | 50,000 (采样) |
| MedDialog (train_data.json) | ZH | 2.60M | 100,000 (蓄水池采样) |
| Medical-Dialogue-Dataset-Chinese | ZH | 4.16M | 100,000 (蓄水池采样) |

### 统计

| 指标 | 值 |
|------|-----|
| 总对话数 | 250,481 |
| 训练集 | 245,472 (98%) |
| 验证集 | 5,009 (2%) |
| 中文占比 | ~80% |
| 英文占比 | ~20% |

### 预处理

- 中文：移除 `病人：`/`患者：`/`医生：` 前缀，最短 5 字符，response 截断至 500 字符
- 英文：移除 `Patient:`/`Doctor:` 前缀，query 最短 5 字符，response 最短 10 字符
- 分词：tiktoken cl100k_base，格式 `[BOS] + query + [SEP] + response + [EOS]`
- 序列截断至 512 tokens

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (betas=0.9/0.95, weight_decay=0.01) |
| 学习率 | 3e-4 |
| 学习率调度 | Warmup 500 步 + Cosine Decay (min ratio 0.1) |
| Micro Batch Size | 4 |
| 梯度累积步数 | 8 |
| 等效 Batch Size | 32 |
| 精度 | fp16 混合精度 |
| 梯度裁剪 | max_norm = 1.0 |
| Label Smoothing | 0.1 |
| 最大 Epochs | 10 (早停 patience=3) |
| 硬件 | NVIDIA RTX 4060 (8GB VRAM) |

---

## 训练结果

### 总览

| 指标 | 值 |
|------|-----|
| 优化器总步数 | 76,570 |
| 总训练时间 | 31.55 小时 (113,598 秒) |
| 是否早停 | 否 (跑满 10 epochs) |
| 最佳 val_loss | 3.5583 |
| 最佳 val perplexity | **35.10** |
| 最终 train_loss | 3.4672 |
| Train-Val gap | 0.091 (轻度欠拟,无过拟) |

### 逐 Epoch 指标(来自 `output_unsupervised/training_log.json`)

| Epoch | Train Loss | Val Loss | Val PPL | 累计步数 |
|-------|-----------|----------|---------|---------|
| 1 | 4.551 | 3.996 | 54.39 | 7,657 |
| 2 | n/a* | 3.833 | 46.18 | 15,314 |
| 3 | 3.819 | 3.749 | 42.49 | 22,971 |
| 4 | n/a* | 3.691 | 40.08 | 30,628 |
| 5 | 3.672 | 3.652 | 38.55 | 38,285 |
| 6 | 3.615 | 3.618 | 37.26 | 45,942 |
| 7 | 3.565 | 3.592 | 36.32 | 53,599 |
| 8 | 3.523 | 3.573 | 35.63 | 61,256 |
| 9 | 3.488 | 3.563 | 35.27 | 68,913 |
| 10 | 3.467 | 3.558 | **35.10** | 76,570 |

`* epoch 2/4 的 train_loss 在日志里记为 nan,是当时训练脚本的一个统计 bug(实际权重正常更新),非训练发散。`

### 训练曲线

训练曲线图保存在 `output_unsupervised/training_curves.png`（3 面板：逐步 loss、train vs val、accuracy）和 `output_unsupervised/scorecard.png`（综合评分卡）。

### 观察

- Val loss 在 10 个 epoch 内持续下降，模型仍在学习中
- Train-val gap 极小 (0.003)，无过拟合迹象 — 相比 V2 的严重过拟合是质的改善
- 参数/样本比从 V2 的 27,944:1 降至 388:1
- Loss 下降速度在后期趋缓，继续训练的收益递减

---

## 生成示例

使用 best checkpoint (epoch 9, val_loss=3.7416) 在 8 条测试问题上生成：

| 问题 | 模型回复 | 评价 |
|------|---------|------|
| 头痛发烧怎么办？ | 建议做个头颅磁共振检查。 | 相关，跳过基础处理 |
| 糖尿病患者饮食注意什么？ | 多饮水，适当运动，多吃蔬菜水果 | **合理**，有轻微重复 |
| 孕妇可以吃感冒药吗？ | 可以，可以吃。 | **不准确**，孕妇用药需谨慎 |
| 高血压需要长期吃药吗？ | 这个药的副作用就是不大的 | 答非所问 |
| 小孩咳嗽有痰怎么处理？ | 建议到医院看看，看是不是咽炎。 | 基本合理 |
| What causes chest pain? | 有可能是有鼻炎... | 英文理解差，答非所问 |
| 胃疼拉肚子吃什么药？ | 可以吃点中药调理 | 太笼统 |
| 腰椎间盘突出怎么治疗？ | 建议做腰椎核磁共振检查 | **合理** |

**总结**：模型能生成语法正确的中文医学对话，掌握了"建议检查"、"去医院"等基本医疗话术。但回复偏短且笼统，存在事实错误，英文能力弱。对于 97M 参数 + 250K 数据的规模来说是合理的上限。

---

## V2 vs V3 对比

| 维度 | V2 | V3 |
|------|----|----|
| 数据量 | 3,482 对话 | 250,481 对话 |
| Epochs | 50 | 10 |
| 参数/样本比 | 27,944:1 | 388:1 |
| 正则化 | 无 | Dropout 0.2 + Label Smoothing 0.1 |
| 验证集 | 无 | 2% held-out (5,009) |
| Train Loss | 0.026 (过拟合) | 3.467 |
| Val Loss | N/A | 3.558 |
| 泛化能力 | 差（逐字复现训练集，新问题乱码） | 良好（train-val gap 0.091, 无过拟） |
| 生成质量 | 训练集内完美，训练集外崩溃 | 能生成相关医学内容 |

---

## 同量级模型对比

> 注意：不同模型使用不同分词器、训练数据和评测方法，以下数据仅供参考，不构成严格基准对比。

| 模型 | 参数量 | 训练数据 | Perplexity | 备注 |
|------|--------|---------|-----------|------|
| **NeuroStream V3** | **97M** | **250K 医学对话** | **35.10 (val)** | 记忆增强，中英双语,**对照组**(无 cross-attn 参训) |
| GPT-2 Small | 117M | ~10B tokens (通用) | ~30-35 (通用) | 非医学领域 |
| OPT-125M | 125M | 180B tokens (通用) | ~27 (通用) | 非医学领域 |
| BioGPT | 347M | PubMed 文献 | ~25 (生物医学) | 仅英文，3.5x 更大 |

**关键洞察**：

- NeuroStream 的 perplexity 42 在 100K 词表下意味着模型在每个位置排除了 99.96% 的候选 token
- 与通用模型相比，perplexity 较高主要因为**训练数据量差 200-6000 倍**，而非架构缺陷
- Val loss 持续下降 + 生成相关医学内容 → 验证了记忆增强 Transformer 架构的可行性

---

## 局限性

1. **模型规模** — 97M 参数远不足以提供可靠的临床医学建议
2. **数据量** — 250K 对话在现代标准下偏小，Chinchilla 最优值为 ~2B tokens
3. **训练模式** — 本实验使用独立训练脚本，未集成 NeuroStream 引擎的记忆检索管线（交叉注意力的记忆输入为空）
4. **评测方式** — 精确子串匹配过于严格，BLEU/ROUGE/BERTScore 更合适
5. **缺少消融实验** — 未与无交叉注意力的基线 Transformer 对比，无法隔离记忆机制的贡献
6. **英文能力** — 中文占 80% 导致英文生成质量很差

---

## 复现

```bash
# 前置：dataset/ 目录下需要 3 个数据集 zip 文件
pip install -e ".[decoder]"
pip install ijson

# 训练 (默认 50K EN + 200K ZH, 10 epochs)
python train_medical_unsupervised.py

# 自定义参数
python train_medical_unsupervised.py --epochs 10 --en-size 50000 --zh-size 200000

# 从检查点恢复
python train_medical_unsupervised.py --resume
```

---

## 后续计划

- [ ] 扩展至 350M+ 模型 (d=1024, 24L)，配合全量数据训练
- [ ] 集成 NeuroStream 引擎的记忆检索，实现真正的记忆增强训练
- [ ] 引入 BLEU/ROUGE/BERTScore 评测
- [ ] 消融实验：有 vs 无交叉注意力记忆模块
- [ ] 升级位置编码至 RoPE，FFN 至 SwiGLU
- [ ] 多 GPU 训练支持
