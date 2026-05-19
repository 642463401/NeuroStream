# NeuroStream 专项教材

> 按"读代码顺序"组织。每章贴关键代码 + 设计意图 + 自测问题 + 实操练习。
> 看不懂某个概念,跳到 [01_systematic.md](01_systematic.md) 对应 § 补理论。

**导航**:[§1 数据结构](#1-数据结构与配置) · [§2 编码器](#2-编码器体系) · [§3 记忆池](#3-记忆池与分层存储) · [§4 影子权重](#4-影子权重核心差异化) · [§5 Transformer](#5-memory-conditioned-transformer) · [§6 持续学习](#6-持续学习与抗遗忘) · [§7 双进程运行时](#7-双进程运行时) · [§8 训练入口](#8-训练入口-trainpy) · [§9 Agent 与工具](#9-agent--工具系统-可选)

---

## §1 数据结构与配置

**关联系统理论**:无(纯工程)

### 1.1 `Memory` —— 记忆的最小单位

文件:`neurostream/types.py`

```python
@dataclass
class Memory:
    vector: Tensor                    # 概念向量
    modality: Modality = TEXT         # 模态(text/image/audio)
    intensity: float = 1.0            # 强度,随时间衰减
    timestamp: float = ...            # 创建时间
    access_count: int = 0             # 被检索次数(影响重要性)
    tier: TierLevel = HOT             # 当前分层(Hot/Warm/Cold)
    reward: float = 0.0               # 反馈评分 [-1, 1]
    correction: Optional[str] = None  # 错误时的正确版本
```

**设计精髓**:`Memory` 不是 "RAG chunk",**它是一等公民**。整个项目的数据结构都围绕 `List[Memory]` 展开。理解这点比看任何代码更重要。

### 1.2 `NeuroStreamConfig` —— 30+ 超参数集中地

文件:`neurostream/config.py`

按功能分组:记忆 / 影子权重 / 抗遗忘 / 反馈 / Transformer decoder / 工具。每个字段都有默认值。

### 自测

1. `Memory.reward` 字段从 [-1, 1],它在哪里被读取并影响训练?(提示:`shadow/objectives.py` 和 `transformer/train.py`)
2. `TierLevel` 三层为什么是 Hot/Warm/Cold 不是 Hot/Cold 两层?

### 实操

打开 IPython:
```python
from neurostream import Memory, Modality
import torch
m = Memory(vector=torch.randn(128), modality=Modality.TEXT, intensity=0.8)
print(m)
```

---

## §2 编码器体系

**关联系统理论**:[01_systematic.md §1](01_systematic.md#1-线性代数与张量) + [§13](01_systematic.md#13-agent--工具系统)

文件:
- `neurostream/encoder/base.py` — `EncoderBase` ABC
- `neurostream/encoder/text.py` — **`FeatureHashEncoder`(零依赖,从这看起)**
- `neurostream/encoder/sbert.py` — SBERT 语义编码
- `neurostream/encoder/image.py` — CLIP 图像
- `neurostream/encoder/audio.py` — Whisper 音频
- `neurostream/encoder/unified.py` — 多模态路由器

### 设计精髓

**ABC 抽象类 + 注册表 + 工厂方法**(`UnifiedEncoder.default()` / `with_sbert()`)。这是个**软件工程模式**比 ML 理论更重要的章节。

### 关键问题

1. `FeatureHashEncoder` 为什么是零依赖?(n-gram + 哈希)
2. `ProjectedEncoder` 怎么把 SBERT 的 384 维降到项目里的 128 维?
3. **懒加载**(`__getattr__`)和 **pickle 安全**(`__getstate__`)在 Windows spawn 多进程下为什么必要?

### 实操

```python
from neurostream.encoder import UnifiedEncoder
enc = UnifiedEncoder.default(dim=128)
v = enc.encode("猫是哺乳动物", modality="text")
print(v.shape, v.norm())  # torch.Size([128]), tensor(1.0)
```
**注意 norm 是 1** —— 编码器输出永远 L2 归一化(`base.py` 强制)。

---

## §3 记忆池与分层存储

**关联系统理论**:[01_systematic.md §10](01_systematic.md#10-向量检索与记忆系统)

文件:
- `neurostream/memory/buffer.py` — `ShortTermBuffer`(线程安全的队列)
- `neurostream/memory/pool.py` — **`MemoryPool`(核心)**
- `neurostream/memory/tiered.py` — `TieredMemoryPool`(Hot/Warm/Cold)
- `neurostream/consolidation/time_integral.py` — 时间积分固化策略

### 数据流

```
encoder.encode(text)
   ↓
ShortTermBuffer.push(memory)          # 线程安全,先入临时缓冲
   ↓
pool.consolidate(buffer, threshold)   # 周期性固化:合并/丢弃/入池
   ├─ merge: 高相似度 → 强度叠加 + 时间衰减
   ├─ discard: 强度 < threshold 直接扔
   └─ add: 新记忆进 FAISS index
   ↓
pool.search(query, top_k)             # 检索:cos_sim + log(1+access_count)
   ↓
pool.decay(dt) + pool.prune()         # 周期性衰减 + 剪枝
```

### 关键代码

`pool.search` 的打分公式:

```python
score = cosine_similarity(query, memory) + log(1 + access_count) + reward_weight * reward
```

`pool.decay(dt, max_dt=5)`(2026-05 关键修复,见 [PROGRESS.md](../../PROGRESS.md) bug #8):

```python
def decay(self, dt: float, max_dt: float = 5.0):
    dt = min(dt, max_dt)  # 钳制,避免 snapshot 期间累积秒数一次性吃光池子
    for m in self.memories:
        m.intensity *= math.exp(-self.decay_rate * dt)
```

### 关键问题

1. `MemoryPool` 用的是 `IndexFlatL2`(精确)还是 `IndexIVFFlat`(近似)?为什么?
2. `consolidate` 里"合并"和"丢弃"是怎么决策的?三个阈值各管什么?
3. `pool.decay` 为什么必须钳制 `max_dt=5`?(PROGRESS 里的真实事故)

### 实操

```python
from neurostream.memory.pool import MemoryPool
from neurostream import Memory
import torch

pool = MemoryPool(dim=128, decay_rate=0.01)
for _ in range(100):
    pool.add(Memory(vector=torch.randn(128) / 11.3))  # 大致归一化
q = torch.randn(128) / 11.3
results = pool.search(q, top_k=5)
print(len(results), results[0].intensity)
```

---

## §4 影子权重(核心差异化)

**关联系统理论**:[§4](01_systematic.md#4-神经网络构件) + [§5 EMA](01_systematic.md#5-训练循环与优化) + [§9 对比学习](01_systematic.md#9-对比学习与表征)

> **这是 NeuroStream 与所有 RAG 系统的根本区别**。RAG 是把检索结果拼到 prompt 里;NeuroStream 真的让一部分模型参数(`MemoryProjector`)持续在变。

文件:
- `neurostream/shadow/projector.py` — **`MemoryProjector`(50 行,从这看起)**
- `neurostream/shadow/objectives.py` — InfoNCE / Reward-Weighted ContrastiveLoss
- `neurostream/shadow/gradient.py` — `MemoryGradientComputer`(构造对比 batch)
- `neurostream/shadow/sync.py` — `SharedWeightBuffer`(跨进程共享内存)
- `neurostream/shadow/manager.py` — `ShadowWeightManager`(训练循环)

### 4.1 MemoryProjector —— 残差 MLP + Zero-Init(必读 50 行)

```python
class MemoryProjector(nn.Module):
    def __init__(self, dim=128, hidden=256, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)  # zero-init!
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)  # 残差

    def project_and_normalize(self, x):
        out = self.forward(x)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)
```

**核心设计**:
- **残差**:`y = x + MLP(x)` → 让梯度能直通,深网络可训(参考 ResNet)
- **Zero-init 最后一层**:`MLP(x) = 0` at day 0 → 整个 projector 是恒等映射,**不影响现有逻辑**
- **训练过程**:对比学习慢慢学到一个比恒等映射更好的投射

### 4.2 InfoNCE 对比学习

`shadow/objectives.py:ContrastiveLoss.forward`:
```python
proj_norm = F.normalize(projected, dim=1)             # L2 归一化
logits = proj_norm @ proj_norm.T / self.temperature   # (N, N) 相似度
logits_off = logits.masked_select(~eye).view(N, N-1)  # 去对角线
labels = F.softmax(gt_off / temperature, dim=1)       # 软标签
loss = -(labels * F.log_softmax(logits_off, dim=1)).sum(dim=1).mean()
```

**关键细节**:**软标签**(用真实相似度的 softmax)而不是 one-hot —— 比标准 InfoNCE 更稳。

### 4.3 跨进程同步 SharedWeightBuffer

`shadow/sync.py`:基于 `torch.share_memory_()`,**近乎无锁**。学习进程 `push()`,推理进程 `pull(ema_alpha=0.01)`:

```python
θ ← (1-α)·θ + α·θ_shadow
```

α=0.005-0.01 → 大约 100-200 次 pull 才完全收敛 → 推理行为**平滑演化,不会突变**。

### 关键问题(每题都要会)

1. 残差 + zero-init 在数学上意味着 day-0 模型行为是什么?
2. InfoNCE 里 `temperature` 太小(0.01)/ 太大(1.0)各会怎样?
3. `MemoryGradientComputer` 是怎么从 `List[Memory]` 构造一个 batch 的?(看 `shadow/gradient.py`)
4. 为什么用 EMA 而不是直接覆盖?(PROGRESS bug #9:覆盖太突然导致输出乱码)

### 实操

读完 `projector.py` 50 行,**关闭文件**,在纸上默写出来。验收标准:能写出 `__init__` + `forward` + `project_and_normalize`。

---

## §5 Memory-Conditioned Transformer

**关联系统理论**:[§7](01_systematic.md#7-transformer-架构) + [§5 训练工程](01_systematic.md#5-训练循环与优化) + [§8 生成](01_systematic.md#8-自回归生成)

> 这是项目最重的章节,占整个 NeuroStream 模型参数的 99%(95M / 95.06M)。

文件:
- `neurostream/transformer/tokenizer.py` — tiktoken cl100k_base 封装
- `neurostream/transformer/config.py` — `TransformerConfig`
- `neurostream/transformer/model.py` — **核心模型**
- `neurostream/transformer/train.py` — `TransformerTrainer`
- `neurostream/transformer/generate.py` — 自回归生成

### 5.1 model.py 结构总览

```
neurostream/transformer/model.py
├─ RMSNorm                       # 比 LayerNorm 少减均值,等效 + 略快
├─ RotaryEmbedding               # RoPE 位置编码:在 Q/K 上旋转
├─ _rotate_half / apply_rope     # 旋转算子
├─ SwiGLUFFN                     # SwiGLU 激活的 FFN(参数量 ≈ 2.67·d²)
├─ CausalSelfAttention           # 因果自注意力(Q=K=V=token)
├─ MemoryCrossAttention          # 记忆交叉注意力(Q=token, K/V=memory)
├─ DecoderBlock                  # self-attn → cross-attn → FFN
└─ MemoryConditionedTransformer  # 堆 N 层 DecoderBlock
```

### 5.2 单层 DecoderBlock 数据流

```
x  ──→ RMSNorm ──→ CausalSelfAttention(x) ──┐
       │                                     │
       └─────────── residual ──────────────→ + ──→ x'
                                                  │
x' ──→ RMSNorm ──→ MemoryCrossAttention(x', mem) ─┐
       │                                          │
       └────────── residual ──────────────────────→ + ──→ x''
                                                       │
x'' ──→ RMSNorm ──→ SwiGLUFFN(x'') ──┐
        │                            │
        └──── residual ──────────────→ + ──→ output
```

**Pre-LN 结构**(LayerNorm 在 sublayer 内、residual 之前),12 层堆叠。

### 5.3 RoPE 关键代码

`model.py:apply_rope`:
```python
def apply_rope(q, k, cos, sin):
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
```

**为什么 RoPE 比 absolute / learned 好**:
- 编码"相对位置"而非绝对位置(`q_i · k_j` 的内积只跟 `i-j` 有关)
- 外推性好(训练时只见 512,推理可达 2048)
- LLaMA / Qwen / Mistral 全部用 RoPE

### 5.4 训练 train.py:train_step

文件:`neurostream/transformer/train.py:130-222`

关键技术:
- **Reward-weighted loss**:`(per_sample_loss * weights).sum() / weights.sum()`,weight = `1 + reward`
- **梯度累积**(grad_accum=8):micro batch 内 `scaled_loss = weighted_loss / accum_steps`
- **fp16 混合精度**:`autocast` + `GradScaler`
- **clip_grad_norm_(1.0)**:防爆炸
- **scheduler**:warmup 500 步 + cosine decay

### 5.5 evaluate(2026-05 新增,早停用)

`train.py:225-285`:teacher-forcing forward + per-token cross-entropy。**与 train_step 区别**:
- `model.eval() + torch.no_grad()`
- 不加 reward 权重(纯均匀)
- 按 token 数归一化(而非样本数,避免长短样本权重失衡)
- 返回 `{val_loss, val_perplexity, n_samples, n_tokens}`

### 5.6 生成 generate.py

支持 `temperature` / `top_k` / `top_p` 三种采样,顺序:
1. logits / temperature
2. top-k 过滤(保留前 k)
3. top-p 过滤(累积概率到 p)
4. softmax → multinomial 采样

### 关键问题(共 10 题)

1. RMSNorm 比 LayerNorm 简化在哪?数学上为什么等价(几乎)?
2. RoPE 怎么把位置信息编码进 attention?(在 Q/K 而非 input)
3. SwiGLU 比 GELU 多出来的"门控"在做什么?
4. `CausalSelfAttention` 里因果掩码是怎么实现的?(`tril` 矩阵)
5. `MemoryCrossAttention` 与 `CausalSelfAttention` 三点不同?(无掩码,Q≠K/V,memory_dim≠d_model)
6. `MemoryConditionedTransformer.forward(input_ids, memory_vectors, memory_mask)` 三个输入的形状?
7. `gradient_accumulation_steps=8` 在 `train_step` 里如何体现?
8. 为什么训练时 `model.train()`,evaluate 时 `model.eval()`?(Dropout / BN 行为不同)
9. `decoder_buffer_size=600_000` 是什么的容量?(`ConversationBuffer`)
10. tiktoken cl100k_base 的特殊 token 是哪些?

### 实操

1. 用 IPython 实例化最小模型:
```python
from neurostream.transformer.config import TransformerConfig
from neurostream.transformer.model import MemoryConditionedTransformer
import torch
cfg = TransformerConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128, max_seq_len=32, memory_dim=64)
m = MemoryConditionedTransformer(cfg)
ids = torch.randint(0, cfg.vocab_size, (1, 16))
mem = torch.randn(1, 3, 64)
mask = torch.ones(1, 3, dtype=torch.bool)
out = m(ids, mem, mask)
print(out.shape)  # (1, 16, vocab_size)
```

2. 把 `memory_mask` 全置 False(`mask = torch.zeros(...)`),看输出是否不同。理解 mask 的作用。

---

## §6 持续学习与抗遗忘

**关联系统理论**:[01_systematic.md §11](01_systematic.md#11-持续学习与抗灾难性遗忘)

文件:
- `neurostream/forgetting/base.py` — `ForgettingStrategy` ABC + `NoOpStrategy`
- `neurostream/forgetting/ewc.py` — **`EWC`(Elastic Weight Consolidation)**
- `neurostream/forgetting/replay.py` — `ExperienceReplay`(蓄水池采样)

### 6.1 EWC 数学

`L_total = L_new + λ · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²`

- `θ*ᵢ`:旧任务训练后的"锚点"参数
- `Fᵢ`:对角 Fisher 信息(参数对似然的敏感度)
- `λ`:正则强度(项目默认 500)

### 6.2 关键代码

`forgetting/ewc.py:EWC._compute_fisher`:用历史 batch 的梯度平方近似 Fisher 对角元。
`forgetting/ewc.py:EWC.penalty`:返回二次项 loss,加到主 loss 上。

### 6.3 Experience Replay 蓄水池采样

`forgetting/replay.py`:
```python
# 蓄水池采样保证流式数据下每条历史样本被保留概率相等
if len(buffer) < k:
    buffer.append(item)
else:
    j = random.randint(0, count - 1)
    if j < k:
        buffer[j] = item
```

### 关键问题

1. `λ=500` 太大 / 太小各会怎样?
2. Fisher 对角近似的精度损失能接受吗?(实证上 OK)
3. EWC 和 Experience Replay 能同时用吗?(可以,叠加效果通常更好)
4. 蓄水级采样在第 `count=10000, k=512` 时,新样本被纳入的概率是?(512/10000)

### 实操

读 `forgetting/replay.py` 全文(很短),手算:数据流来到第 1000 条时,第 1 条还在 buffer 的概率是多少?(`512/1000`)

---

## §7 双进程运行时

**关联系统理论**:[01_systematic.md §12](01_systematic.md#12-多进程异步训练系统)

> 项目最复杂的工程章节。但**只有这里**让"边推理边学习"成为可能。

文件:
- `neurostream/runtime/channels.py` — `ChannelSet`(8 个 Queue)
- `neurostream/runtime/engine.py` — `NeuroStreamEngine`(生命周期管理)
- `neurostream/runtime/inference.py` — 推理进程主循环
- `neurostream/runtime/learning.py` — 学习进程主循环
- `neurostream/shadow/sync.py` — `SharedWeightBuffer`(共享内存)

### 7.1 进程拓扑

```
                Main 进程
              (engine.ingest)
                    │
                    ▼ input_q
┌─────────────────────────────────────────────────┐
│                                                  │
│  推理进程 (Inference)         学习进程 (Learning) │
│  ─ 接收输入                   ─ ShortTermBuffer  │
│  ─ encoder.encode             ─ MemoryPool       │
│  ─ projector投射 + L2         ─ consolidate      │
│  ─ search_req → 学习          ─ decay + prune    │
│  ─ Memory 入队 ─ memory_q ──→ ─ ShadowMgr 训练   │
│  ─ EMA pull 影子权重          ─ TransformerTrainer│
│                               ─ push_to_buffer    │
└──────────┬────────────────────┴──────────────────┘
           │
           ▼
     SharedWeightBuffer  (torch.share_memory_)
        (EMA 同步)
```

### 7.2 八条 Queue(每条都是契约)

| Queue | 方向 | 内容 |
|---|---|---|
| `input_q` | Main → 推理 | 原始输入 dict / None(毒丸) |
| `memory_q` | 推理 → 学习 | 新 `Memory` 对象 |
| `search_req_q` | 推理 → 学习 | `(query_vec, top_k)` |
| `search_resp_q` | 学习 → 推理 | `List[Memory]` |
| `conversation_q` | Main → 学习 | decoder 训练数据 |
| `generate_req_q` / `resp_q` | Main → 推理 → Main | 生成请求/响应 |
| `feedback_q` | Main → 学习 | reward 信号 |
| `snapshot_req_q` / `resp_q` | Main → 学习 → Main | 非破坏快照 |
| **`eval_req_q` / `resp_q`** | Main → 学习 → Main | val_loss 评估(早停用,2026-05 新增) |
| `metrics_q` | 学习 → Main | 训练指标 |

### 7.3 学习进程主循环(`learning.py`)

```python
while not stop_event.is_set():
    # 1. 收集新记忆到 ShortTermBuffer
    drain(channels.memory_q) → buffer

    # 2. 响应检索请求
    drain(channels.search_req_q) → pool.search → search_resp_q

    # 3. 处理反馈
    drain(channels.feedback_q) → 更新 memory.reward

    # 4. 处理快照/评估请求
    handle snapshot_req_q
    handle eval_req_q  # 2026-05 新增

    # 5. 收集对话训练数据
    drain(channels.conversation_q) → conversation_buffer

    # 6. 周期性固化 + 衰减 + 剪枝
    if now - last_learn >= 1.0:
        pool.consolidate(buffer)
        pool.decay(dt=1.0)
        pool.prune()

    # 7. 周期性影子权重训练
    if now - last_shadow >= 10.0:
        shadow_mgr.train_step(pool.memories)
        shadow_mgr.push_to_buffer(shadow_buffer)

    # 8. 周期性 Transformer 训练
    if (in_active_group or now - last_decoder >= interval):
        decoder_trainer.train_step(n_steps=1)
        if 组内剩余 == 0:
            decoder_trainer.push_to_buffer(decoder_buffer)

    time.sleep(0.05)
```

### 7.4 必须知道的几个"坑"(全在 PROGRESS.md 里)

| Bug | 现象 | 修复 |
|---|---|---|
| #2 | cross-attn 训练时见不到记忆 | `engine.teach()` 传 query_vec 给 learning_worker |
| #4 | `wait_drain` 卡住 | Windows `mp.Queue.qsize()` 在子进程消费后不递减 |
| #5 | 92% 训练数据被静默丢弃 | `ConversationBuffer` 5K → 600K |
| #8 | snapshot 后 pool 一次性 179K→21 | `pool.decay(max_dt=5)` 钳制 |
| #9 | 输出乱码 | inference 端 resume 加载 decoder state + 主动 EMA pull |
| #10 | loss 假性掉 0.04 | 移除 inference→training 回灌 |

### 关键问题(共 8 题)

1. `mp.set_start_method("spawn", force=True)` 在 Windows 是必须的,为什么?
2. 为什么训练在 GPU,push_to_buffer 时却要拷贝到 CPU 共享内存?
3. `SharedWeightBuffer.push(model)` 是同步还是异步?(同步)
4. 主进程发"毒丸"(`None`)给 input_q,推理进程怎么响应?
5. `decoder_train_steps_per_update=16` 这一组 16 步内,推理端能看到训练好的权重吗?(看不到,push 才同步)
6. EMA α=0.005,推理端从收到 push 到完全收敛新权重需要多少次 pull?(约 200)
7. `mp.Queue.qsize()` 在 Windows 不可靠,为什么 `wait_drain` 仍然能用?(检查 conversation_q 而非 memory_q + 多次 stable 判定)
8. 8 个 Queue 里哪个是双向的?哪个是单向的?

### 实操

不跑训练,只启动 engine 看进程初始化:
```python
from neurostream import NeuroStreamPipeline
with NeuroStreamPipeline(dim=128) as pipe:
    pipe.ingest("猫是哺乳动物")
    import time; time.sleep(3)
    # 看 stderr 输出 [Learning] consolidated N, pool=...
```

---

## §8 训练入口 train.py

**关联系统理论**:[§5 训练工程](01_systematic.md#5-训练循环与优化) + [§11 持续学习](01_systematic.md#11-持续学习与抗灾难性遗忘)

文件:`train.py`(项目唯一训练入口)

### 8.1 流程图

```
main()
├─ argparse 解析参数
├─ load_dataset(en, zh)                       # 加载 250K 医学对话
├─ split_train_val                            # 245K / 5K
├─ tokenize_val_records(val_data)             # 主进程预 tokenize val 集
├─ build_config(args) → NeuroStreamConfig
├─ NeuroStreamTrainer(config).engine.start()  # spawn 2 个进程
└─ run_continual(engine, train_data, val_records, args):
    ├─ 阶段 1: PUSH
    │   for item in train_data:
    │       engine.ingest(query)              # → memory_q
    │       engine.teach(query, response)     # → conversation_q
    ├─ 阶段 2: DRAIN
    │   wait_drain(conversation_q, 600s)
    └─ 阶段 3: TRAIN-UNTIL-STOP
        loop:
            cur_step = collect_latest_step(metrics_q)
            if cur_step >= next_eval_step:
                metrics = engine.evaluate_val_loss(val_records[:200])
                if val_loss < best:
                    save snapshot_best.pt
                    patience = 0
                else:
                    patience += 1
                    if cur_step >= target and patience >= 5: break
                    if patience >= 10: break
            sleep(10)
    └─ save snapshot_final.pt
    └─ run_eval(generate 30 条 + 软指标)
    └─ render_reports(scorecard.png + val_loss_curve.png)
```

### 8.2 关键参数(2026-05 早停改造后)

| 参数 | 默认 | 含义 |
|---|---|---|
| `--target-epochs` | 1.0 | 等量 epoch 目标(数据量/effective_batch) |
| `--min-steps` | 2000 | 兜底最小步数 |
| `--early-stop` | True | 默认开启,`--no-early-stop` 关 |
| `--early-stop-eval-every` | 500 | 每 N step 评估一次 |
| `--early-stop-patience` | 5 | 连续 N 次没改善 |
| `--early-stop-min-delta` | 0.001 | 改善阈值 |
| `--early-stop-eval-n` | 200 | val 采样数 |
| `--max-hours` | 168 | 兜底硬上限(7 天) |
| `--decoder-interval` | 0.0 | 组间间隔(GPU 吞吐调优) |
| `--decoder-steps-per-group` | 16 | 组内连续步数 |

### 8.3 早停停止判定(`train_until_stop`)

```python
if val_loss < best - min_delta:
    patience = 0
    save("snapshot_best.pt")
else:
    patience += 1

if cur_step >= target and patience >= 5:      # 双条件
    stop_reason = "early-stop"
if patience >= 10:                              # hard limit
    stop_reason = "early-stop hard limit"
if elapsed > max_hours * 3600:
    stop_reason = "max-hours"
if interrupted:                                 # SIGINT
    stop_reason = "interrupted"
```

### 关键问题

1. `--target-epochs=1.0` + 250K 样本 + effective_batch=32,target_step 是多少?(7813)
2. 为什么 push 阶段完成后才启动早停(而不是 push 期间)?(防止 buffer 没填满时 val_loss 不稳)
3. `wait_drain` 只看 conversation_q,为什么不看 memory_q?(Windows mp.Queue bug)
4. SIGINT 触发后会发生什么序列动作?

### 实操

跑一个最小 pilot(2k EN + 8k ZH,几分钟跑完,验证流程):
```powershell
python train.py --en-size 2000 --zh-size 8000 `
                --early-stop-eval-every 100 `
                --early-stop-patience 3 `
                --output output/pilot_codex_test
```
看 `training_log.json` 里 `early_stop.val_history`,理解每个字段含义。

---

## §9 Agent / 工具系统 (可选)

**关联系统理论**:[01_systematic.md §13](01_systematic.md#13-agent--工具系统)

> Phase 1/2 不依赖。规划上 OpenAI 兼容 API 时再深入。

文件:
- `neurostream/tools/base.py` — `Tool` ABC + `ToolRegistry`
- `neurostream/tools/builtin/` — Calculator / PythonExec / HTTPRequest
- `neurostream/tools/mcp/` — MCP 协议(JSON-RPC 2.0)
- `neurostream/agent/teacher.py` — `TeacherLLM`(DashScope)
- `neurostream/agent/evaluator.py` — `BenchmarkEvaluator`(4 维评测)
- `neurostream/agent/report.py` — `BenchmarkReporter`(Matplotlib)
- `neurostream/agent/loop.py` — `AgentLoop`(Teacher→Student 蒸馏)

### 9.1 Tool 调用流程

```
LLM 输出含 <tool:calculator>{"expression": "1+2"}</tool>
   ↓
generate.py 解析特殊标记
   ↓
ToolRegistry.execute("calculator", {"expression": "1+2"})
   ↓
返回 ToolResult(output="3.0")
   ↓
回灌 generation context 继续生成
```

### 9.2 MCP 协议

JSON-RPC 2.0 over stdio,Anthropic 推的开放标准,目的是让 LLM 能用任何外部工具(GitHub / Slack / 数据库 / ...)。

### 9.3 蒸馏 AgentLoop

```
Teacher LLM (Qwen3-Max via DashScope)
   ↓ 给定 query 生成参考 answer
Student (NeuroStream's decoder)
   ↓ 用 (query, answer) 做 reward-weighted 训练
   ↓ 评测打分
BenchmarkReporter
   ↓ 输出论文风格表格 / 雷达图
```

### 关键问题

1. Tool 输出是怎么进 token 流的?(特殊标记 + tokenizer)
2. MCP 与 OpenAI function calling 协议层差异?
3. 蒸馏 vs 直接训练(在 medical_cleaned.json 上),收益差在哪?

### 实操

```python
from neurostream import NeuroStreamPipeline
pipe = NeuroStreamPipeline(dim=128, tools_enabled=True)
result = pipe.call_tool("calculator", {"expression": "sqrt(144) + pi"})
print(result.output)  # "15.141592653589793"
```

---

## 通关检验

学完 9 章,这些任务你应该都能独立完成:

| # | 任务 | 涉及 |
|---|---|---|
| 1 | 给 `MemoryProjector` 加一层(`dim → hidden → hidden → dim`)并重训 | §4 |
| 2 | 修改 `TransformerConfig`,把 `n_layers` 从 12 改成 6,跑通训练 | §5 |
| 3 | 实现一个简单 `ForgettingStrategy.SmartReplay`(按 reward 加权采样而非均匀) | §6 |
| 4 | 给 `ChannelSet` 加一条新 Queue(比如 `health_check_q`),改 learning_worker 响应 | §7 |
| 5 | 给 `train.py` 加一个 `--early-stop-metric` 参数,可选 `val_loss` / `char_overlap` | §8 |
| 6 | 用 FastAPI 写一个 `/v1/chat/completions` 端点,调 `engine.generate()` | §9 |

完成任意 3 个,你就完全掌控这个项目。

---

## 推荐配套阅读顺序

1. [PROGRESS.md](../../PROGRESS.md) —— 真实事故录,11 个 bug 等于 11 节"反面教材课"
2. [docs/architecture.md](../architecture.md) —— 系统架构鸟瞰图(已经熟悉的概念,作为索引)
3. [docs/math_formulas.md](../math_formulas.md) —— 25 个核心公式
4. [docs/experiments/medical_v3.md](../experiments/medical_v3.md) —— V3 实验报告,看真实指标

---

**回到顶部**:[§1](#1-数据结构与配置) · 跳转 [01_systematic.md](01_systematic.md) · [00_knowledge_map.md](00_knowledge_map.md)
