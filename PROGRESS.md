# NeuroStream 项目进度

> 最后更新: 2026-05-09
> 当前定位: **持续学习训练框架**（方案 A 落地中）
> Phase 1 状态: ✅ 训练通路打通，模型可生成连贯医学回答，待 Phase 2 测持续学习

---

## 一、项目定位

NeuroStream 是一个**以记忆为核心、边推理边学习**的训练框架。
严格遵循 `docs/architecture.md` 中的设计：

- **双进程**：`NeuroStreamEngine` 同时跑推理进程 + 学习进程，互不阻塞
- **记忆池**：FAISS Hot/Warm/Cold 分层 + 时间衰减 + 固化 + 剪枝
- **影子权重**：`MemoryProjector` + `SharedWeightBuffer` 跨进程 EMA 同步
- **记忆增强 Transformer**：每层 cross-attention 读取记忆 K/V
- **抗灾难性遗忘**：EWC / Experience Replay
- **生物隐喻**：皮层（推理）+ 海马体（固化）

**新增架构原则（2026-05-09）**：
> AI 的输出**永不进入训练管道**。知识只能通过应用层显式 `engine.ingest()` /
> `engine.teach()` 注入。`engine.generate()` 是纯只读推理，不写记忆池、不写
> conversation_buffer。

理由：模型训练自己的输出是退化的自蒸馏，会让 loss 假性骤降到 0.04 以下、
量化指标失真，且违背"知识来自外部"的契约。

---

## 二、Phase 1 训练成果（2026-05-09）

### 训练配置

| 参数 | 值 |
|---|---|
| 数据 | 250K 医学对话 (50K EN + 200K ZH, `dataset/medical_cleaned.json`) |
| 模型 | 95M 参数（d=512, 12L, 8H, ff=1366, SwiGLU+RoPE+RMSNorm） |
| 双进程 | shadow=on, decoder=on, cross-attn enabled |
| 抗遗忘 | EWC λ=500 |
| 内存 | ConversationBuffer 600K, Memory pool 220K+ |

### 两轮训练快照

| 文件 | 大小 | 说明 |
|---|---|---|
| `output/phase1/snapshot_final.pt` | 874 MB | 首轮训练，3785 decoder steps，220K memories |
| `output/phase1_more/snapshot_final.pt` | 880 MB | resume 续训 4h，~5000 decoder steps，225K memories |

### 模型质量（基于 `output/phase1_more` eval）

软指标：
```
Empty rate:       0.0%      ← 从不空回复
Strict match:     0.0%      ← substring match 在开放问答上必然 0
Char overlap:     13.8%     ← 含医学相关字符
Bigram Jaccard:   1.1%      ← 短语级重合（不同医生措辞自由度大）
Length ratio:     66.7%     ← 长度匹配
```

实际生成示例（`training_log.json` 解码后）：
```
Q: 高血压 150/98，老公磁共振...
A: 建议您来院就诊，带上所有资料。              ← 标准分诊建议

Q: 小儿关节弹响膜囊，下肢异常 5 天
A: 您好，建议您先到医院检查一下，必要时做个膝关节 CT 检查。  ← 专业且对症

Q: 关于肺癌
A: 可以手术，但是不需要做，但是要看你的病情。  ← 合理临床判断
```

**结论**：模型已能产出合理医学回答，从初始 11.5 loss 降到 ~2.7 (ppl ~15)。
strict_match=0% 是开放医学问答的本质局限（参考答案唯一），不代表模型差。

### Windows 终端显示注意

PowerShell 默认 GBK，看到的"口字码"只是字节渲染错位。看真实输出：
```powershell
chcp 65001  # 切 UTF-8
# 或直接用 VSCode 打开 output/phase1_more/training_log.json
```

---

## 三、本次冲刺修掉的关键 bug

按发现顺序，每条都对应一个生产事故：

| # | 现象 | 根因 | 修复位置 |
|---|---|---|---|
| 1 | `train.py` 在 `public` 分支看不到 `neurostream/` 源码 | `public` 是精简发布分支，不含源码 | 切到 `main` 分支开发 |
| 2 | cross-attn 训练时见不到记忆（`memory_vectors=zeros(0,dim)`） | `engine.teach()` 没把 query_vec 传给 learning_worker | `engine.py:teach()` + `learning.py` conversation_q 消费段 |
| 3 | 4000 ingests 只有 26-54 条进 pool | 主进程 push 完立刻 snapshot，learning_worker 没消化完队列 | `train.py:wait_drain()` |
| 4 | `wait_drain` 卡 600s 不返回 | Windows `mp.Queue.qsize()` 子进程消费后不递减 | 只检查 `conversation_q`，不信 `memory_q` |
| 5 | 92% 训练数据被静默丢弃 | `ConversationBuffer = deque(maxlen=5000)`，主进程 push 比 decoder 消费快 13x | `transformer/config.py` 加 `conversation_buffer_size=600000` |
| 6 | `--epochs N` 实际只训 1/9 epoch | 主进程 push 完立刻退出，bonus 只 120s | `train.py:wait_drain()` 加 `train_min` 参数 |
| 7 | eval 时 pool 5 分钟从 23K 崩到 23 条 | `decay_rate=0.01` × 300s 累积衰减 → 大批 intensity 跌破阈值被剪 | `train.py:build_config` decay_rate 0.01→0.001 |
| 8 | snapshot 后 pool 一次性 179K→21 | `pool.decay()` 用 `dt = now - mem.timestamp`，snapshot 期间 dt 累积成数百秒，单次 decay 一口气把累积秒数全算 | `pool.py:decay()` 加 `max_dt=5` 钳制 + 显式 `dt=` 参数；`learning.py` 传 `dt=learning_interval_sec` |
| 9 | inference 输出乱码（`reelsbutt narrator...`） | resume 时 decoder_state 只灌进 learning_worker 的 trainer，inference 端的 `self._decoder` 还是随机初始化；EMA α=0.005 太慢 | `engine.py:start()` 同步加载到 `self._decoder`；`inference.py` 加每 5s 主动 pull |
| 10 | eval-only loss 假性掉到 0.04 | inference 把生成的 (query, AI回答) 推回 `conversation_q` 训练，模型在拟合自己的输出 | `inference.py:_handle_generate()` 移除 `conversation_q.put` + `memory_q.put` |
| 11 | strict substring match 永远 0% | 开放医学问答只有一个参考答案 | `train.py:_score_one()` 加 char_overlap / bigram_jaccard / len_ratio / empty_rate |

---

## 四、单一训练入口（`train.py`）

```bash
# 持续学习（默认）
python train.py

# 从快照恢复继续训练 N 分钟
python train.py --resume output/phase1/snapshot_final.pt --train-min 240

# 只评估，不训练
python train.py --resume output/phase1/snapshot_final.pt --eval-only

# Phase 2 持续学习（要求 --resume）
python train.py --resume output/phase1/snapshot_final.pt \
                --phase2-data dataset/medical_phase2.json --output output/phase2

# 蒸馏模式（委托给 AgentLoop，工作流之外用得少）
python train.py --distill --api-key $env:DASHSCOPE_KEY
```

主要参数（默认值）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--memory-dim` | 128 | 记忆向量维度 |
| `--d-model` | 512 | Transformer 维度 |
| `--n-layers` | 12 | 层数 |
| `--n-heads` | 8 | 注意力头数 |
| `--d-ff` | 1366 | SwiGLU FFN（≈ d×8/3） |
| `--seq-len` | 512 | 序列长度 |
| `--lr` | 3e-4 | 学习率 |
| `--batch-size` | 4 | micro batch |
| `--grad-accum` | 8 | 累积步数（有效 batch=32） |
| `--forgetting` | ewc | none / ewc / replay |
| `--max-hours` | 23 | 最长总时间 |
| `--snapshot-hours` | 2 | 快照间隔 |
| `--epochs` | 1 | 数据 push 遍数（不是训练轮次） |
| `--train-min` | 120 | push 完后让 decoder 继续训练的分钟数 |
| `--eval-only` | False | 跳过训练只跑 eval |
| `--eval-n` | 30 | eval 采样数 |
| `--phase2-data` | None | Phase 2 数据集路径，要求 `--resume` |

---

## 五、Phase 2 准备工作

### 数据集生成脚本：`prepare_phase2_data.py`

```bash
# 默认：50K EN + 200K ZH，与 phase 1 不重叠
python prepare_phase2_data.py

# 推荐做对照实验：纯英文 50K，测中文遗忘
python prepare_phase2_data.py --target-en 50000 --target-zh 0 \
  --output dataset/medical_phase2_en_only.json

# 用更大的 6.5M 中文池
python prepare_phase2_data.py --use-zh-extra
```

原始数据池容量（已验证）：

| 来源 | 总量 | Phase 1 占用 | Phase 2 可用 |
|---|---|---|---|
| `Medical-Dialogue-Dataset-English.zip` | 181K | 74K | **107K** |
| `MedDialog_processed.zip` 中文 | 2.44M | 165K | **2.27M** |
| `Medical-Dialogue-Dataset-Chinese.zip` | ~6.5M | 部分 | 数百万 |

去重保证：用 phase1 的 query MD5 哈希集合（242,503 个）filter 原始流。

---

## 六、运行约束

- GPU: RTX 5060 8GB（**注意：不是 Ti**，比之前以为的略弱）
- 系统 RAM 无 ECC，长跑 >24h 可能死机 → `--max-hours` 默认 23h
- ConversationBuffer 600K 条 ≈ 2.4GB CPU RAM
- Memory pool 200K+ 条 ≈ 100MB CPU RAM (FAISS index)
- 单 epoch 真实训练时间：理论 64 min，实测 4h 训了 1200 步 (~5%)，瓶颈在 learning_worker 大池 consolidate 周期

---

## 七、下一步路线

1. **Phase 2 持续学习实验**（P0 — 立即可做）
   ```bash
   python prepare_phase2_data.py --target-en 50000 --target-zh 200000
   python train.py --resume output/phase1_more/snapshot_final.pt \
                   --phase2-data dataset/medical_phase2.json \
                   --train-min 120 --output output/phase2
   ```
   关注点：phase 2 训完后用同一 eval 集衡量
   - char_overlap 是否提升（说明持续学习有收益）
   - 找回 phase1 val 集做"遗忘率"对比

2. **预训练 backbone 接入**（P1，可选）
   `Cloud-5090/checkpoint_latest.pt` 是 1.2B token 的双语 backbone（`no_cross_attn=True`）。
   需要写权重迁移脚本：self-attn/FFN/embed → 新带 cross-attn 模型，cross-attn 层 zero-init。
   预期能让 char_overlap 从 13.8% 跳到 25%+。

3. **训练吞吐优化**（P1）
   当前 4h 只训 1200 步，瓶颈在大 pool 的 consolidate/decay 周期。可选方案：
   - `consolidate` 改为 batch 模式（一次处理 100 条而非 1 条）
   - `decay` 跨多个 cycle 才执行一次
   - 单独的 `MemoryPool` 操作进程

4. **更靠谱的 eval**（P2）
   - 接入 LLM-as-judge（DashScope）做语义评分
   - 加 ROUGE-L / BLEU / 中文 BERTScore

5. **REST API + 多 GPU**（P3，README Roadmap 已列）

---

## 八、`neurostream/` 包能力（核心，不变）

| 模块 | 内容 |
|---|---|
| `types.py` | Memory / Modality / TierLevel |
| `config.py` | `NeuroStreamConfig`（**新增 `decoder_buffer_size`**） |
| `encoder/` | FeatureHash / SBERT / CLIP / Whisper / UnifiedEncoder |
| `memory/pool.py` | **`decay()` 加显式 `dt` + `max_dt` 钳制** |
| `consolidation/` | `TimeIntegralStrategy` |
| `shadow/` | `MemoryProjector` / `SharedWeightBuffer` / `ShadowWeightManager` |
| `forgetting/` | `EWC` / `ExperienceReplay` / `NoOpStrategy` |
| `runtime/engine.py` | **`start()` 加载 decoder_state 到 inference 端** |
| `runtime/learning.py` | **加 IDLE_DECAY_PAUSE_SEC + cycle-based decay 调用** |
| `runtime/inference.py` | **5s 周期主动 pull decoder + 移除 generate→training 回灌** |
| `transformer/config.py` | **新增 `conversation_buffer_size: int = 600_000`** |
| `transformer/train.py` | **`ConversationBuffer` 用 config 的 `conversation_buffer_size`** |
| `tools/` | Tool ABC + Calculator/PythonExec/HTTP + MCP |
| `agent/` | `AgentLoop` + `TeacherLLM` + `BenchmarkEvaluator` + `BenchmarkReporter` |
| `api/` | `NeuroStreamPipeline` / `NeuroStreamTrainer` |

`train.py` 仅是这些能力的薄包装层，主要职责是配置 + 数据流转 + 报告。

---

## 九、分支策略

- `main` — 完整源码（开发分支，当前在用）
- `public` — 仅文档/配置（对外发布分支）
- 工作流：在 main 干活，验证后 `git checkout public && git checkout main -- <files>`
  把对外可见的文件选择性合并过去。

**未提交变更**（截至 2026-05-09）：
- `train.py`, `prepare_phase2_data.py`（新建）
- `neurostream/runtime/engine.py`, `learning.py`, `inference.py`
- `neurostream/memory/pool.py`
- `neurostream/transformer/config.py`, `train.py`
- `neurostream/config.py`
- `PROGRESS.md`（本文件）

建议在 phase 2 实验完成后一并 commit。

---

## 十、关键工件路径速查

```
dataset/
  medical_cleaned.json         Phase 1 训练数据 (250K)
  medical_phase2.json          [待生成] Phase 2 数据
  medical_cleaned_meta.json    Phase 1 数据元信息

output/
  pilot/, pilot2/, pilot3/     冒烟测试历史
  phase1/                      Phase 1 第一轮 (3785 steps)
    snapshot_final.pt
    training_log.json
    scorecard.png
  phase1_more/                 Phase 1 续训 (5000 steps, 当前最新)
    snapshot_final.pt          ← 最新可用模型
    training_log.json
    scorecard.png
  phase1_eval/                 Phase 1 第一次 eval-only (修复前，无效)
  phase1_eval2/                Phase 1 第二次 eval-only (修复后)

Cloud-5090/
  checkpoint_latest.pt         1.2B token backbone（no_cross_attn）— 待迁移
  training_log.json
```
