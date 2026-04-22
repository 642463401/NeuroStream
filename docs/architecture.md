# Architecture

NeuroStream 的核心理念：**让 AI 系统像人脑一样持续学习、选择性遗忘、在记忆中演化**。

---

## 系统总览

```
                          NeuroStreamPipeline / Trainer
                                    │
                          NeuroStreamEngine
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
            推理进程 (Inference)              学习进程 (Learning)
            ├── UnifiedEncoder               ├── ShortTermBuffer
            ├── MemoryProjector              ├── MemoryPool / TieredMemoryPool
            └── 唤醒检索                      ├── ConsolidationStrategy
                                             ├── ShadowWeightManager
                                             └── ForgettingStrategy
                    │                               │
                    └──── SharedWeightBuffer ────────┘
                          (跨进程 EMA 同步)
```

---

## 双进程架构

NeuroStream 将推理和学习分离到两个独立进程，互不阻塞。

### 推理进程

负责实时处理输入数据，不做任何训练：

1. 从 `input_q` 接收原始数据
2. `UnifiedEncoder` 编码为向量
3. `MemoryProjector` 投射 + L2 归一化
4. 向学习进程发起检索请求（`search_req_q`）
5. 构造 `Memory` 对象，推入 `memory_q`
6. 周期性 EMA 拉取影子权重，使投射行为平滑演化

### 学习进程

独占记忆池，执行所有慢速操作：

1. 从 `memory_q` 收集新记忆到 `ShortTermBuffer`
2. 响应检索请求（`search_req_q` → `search_resp_q`）
3. 周期性执行固化 → 衰减 → 剪枝
4. 影子权重对比学习训练
5. `push()` 新权重到 `SharedWeightBuffer`

### 进程间通信

6 条类型化 Queue 组成 `ChannelSet`：

```
input_q:       主进程 → 推理    原始输入 / None (毒丸)
memory_q:      推理 → 学习      新 Memory
search_req_q:  推理 → 学习      (query_vec, top_k)
search_resp_q: 学习 → 推理      List[Memory]
result_q:      学习 → 主进程    最终记忆池
metrics_q:     学习 → 主进程    训练指标
```

生命周期：`start()` spawn 两个进程 → `ingest()` 推数据 → `shutdown()` 毒丸关闭推理 → `stop_event` 关闭学习 → 收集结果。

---

## 记忆系统

### 记忆表示

每条记忆 (`Memory`) 包含：

- **vector** — 概念向量 (Tensor)
- **modality** — 来源模态
- **intensity** — 强度，随时间衰减
- **timestamp** — 创建时间戳
- **access_count** — 被检索命中的次数

### 短期缓冲 → 长期固化

```
输入 → ShortTermBuffer (线程安全) → flush() → ConsolidationStrategy → MemoryPool
```

`TimeIntegralStrategy` 实现三级漏斗：

1. **合并**：与池中高相似度记忆合并（余弦 >= `merge_threshold`），强度叠加 + 时间衰减
2. **丢弃**：强度低于 `intensity_threshold` 的弱记忆直接丢弃
3. **固化**：通过筛选的记忆加入长期池

### 唤醒检索

```
score = cosine_similarity(query, memory) + log(1 + access_count)
```

频繁被检索的记忆得到加成，模拟"越用越强"的唤醒效应。

### 时间衰减与剪枝

```
intensity *= exp(-decay_rate * dt)      # 全局指数衰减
prune(threshold) → 移除 intensity < threshold 的记忆
```

### 分层存储 (TieredMemoryPool)

三级分层：Hot / Warm / Cold。

| 路径 | 条件 |
|------|------|
| 新记忆 → Hot | 始终 |
| Hot → Warm | Hot 超容量，最弱记忆降级 |
| Warm → Hot | 检索命中且强度 >= `promotion_threshold` |
| Warm → Cold | 未访问超过 `demotion_interval_sec` |
| Cold → 搜索 | 仅在 Hot + Warm 结果不足时 |

检索优先搜索 Hot 层，逐级扩展，热门记忆自动晋升。

---

## 影子权重系统

这是 NeuroStream 的核心差异化：模型权重随记忆演化。

### MemoryProjector

残差 MLP：`x + MLP(x)`，最后一层 zero-init → 初始行为为恒等映射。

```
input(dim) → Linear(dim, hidden) → GELU → Linear(hidden, dim, zero-init) → + input
```

### 训练流程

```
高强度记忆 → MemoryGradientComputer → 对比学习 batch
                                          │
shadow_model → forward → InfoNCE loss ──────┤
                                          + ForgettingStrategy.penalty
                                          │
                                    backward + clip grad + optimizer.step
                                          │
                                    push → SharedWeightBuffer
```

1. `MemoryGradientComputer` 按强度选 top-N 记忆，计算两两余弦相似度作为 ground truth
2. `ContrastiveLoss` 计算 InfoNCE 损失
3. 加上抗遗忘惩罚 (EWC / NoOp)
4. 反向传播 + 梯度裁剪 + 优化器更新
5. 写入 `SharedWeightBuffer`

### 跨进程同步

```
学习进程: train → push(shadow_model) → version++
推理进程: pull(projector, ema_alpha=0.01) → θ = (1-α)θ + αθ_shadow
```

`SharedWeightBuffer` 基于 `torch.share_memory_()`，近乎无锁。EMA alpha=0.01 意味着约 100 次 pull 才完全收敛到新权重，推理行为平滑演化，不会突变。

---

## 编码器体系

```
EncoderBase (ABC)
├── FeatureHashEncoder         — 零依赖，特征哈希
└── ProjectedEncoder (ABC)     — 维度适配基类
    ├── SBERTEncoder           — sentence-transformers
    ├── CLIPImageEncoder       — open_clip
    └── WhisperAudioEncoder    — openai-whisper

UnifiedEncoder                 — 多模态路由器
├── register(modality, encoder)
├── encode(data, modality) → Tensor(dim,)
└── 工厂: default() / with_sbert() / with_pretrained() / full_multimodal()
```

### 维度适配

预训练模型输出维度各不相同 (384 / 512 / 768 等)。`ProjectedEncoder` 统一处理：

- `native_dim == target_dim` → 直接输出
- `native_dim != target_dim` → `nn.Linear(native, target, bias=False)` + 正交初始化

所有输出都经过 L2 归一化 + `detach().cpu()`，保证多进程安全。

### 惰性加载

预训练模型按需加载：
- `__getattr__` 惰性导入，`import neurostream` 不触发重量级依赖
- `__getstate__` 剥离模型对象，pickle 安全 (Windows `spawn`)
- 子进程首次 `encode()` 时 `_get_model()` 加载模型

---

## 抗灾难性遗忘

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| `NoOpStrategy` | 无惩罚 | 快速原型 / 短期任务 |
| `EWC` | `λ * Σ F_i * (θ_i - anchor_i)²` | 知识累积、防止旧知识丢失 |
| `ExperienceReplay` | 蓄水池采样旧记忆混入训练 | 数据分布变化、流式学习 |

EWC 通过 Fisher 信息矩阵衡量参数重要性，对重要参数施加弹性约束。Experience Replay 通过数据混合防止遗忘，蓄水池采样保证每条历史记忆被保留概率相等。

两种策略可通过 `NeuroStreamConfig.forgetting_strategy` 配置，也可自定义 `ForgettingStrategy` 子类。

---

## 设计原则

1. **推理学习分离** — 推理不等训练，训练不阻推理
2. **零配置到全控** — `NeuroStreamPipeline` 三行启动，`NeuroStreamTrainer` 完全可控
3. **平滑演化** — EMA 同步确保行为渐变，不会因一次训练步骤突变
4. **生物启发** — 衰减、唤醒、固化机制模拟人脑记忆特性
5. **可扩展** — 编码器、固化策略、遗忘策略均为 ABC，支持自定义实现
6. **零重依赖** — 默认安装仅需 PyTorch + FAISS，预训练编码器按需安装
