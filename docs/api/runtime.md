# Runtime

## `NeuroStreamEngine`

双进程运行时引擎。管理推理进程和学习进程的完整生命周期。

```python
from neurostream.runtime import NeuroStreamEngine
from neurostream import NeuroStreamConfig

config = NeuroStreamConfig(dim=128)
engine = NeuroStreamEngine(config=config)
engine.start()
engine.ingest("some text", modality="text")
metrics = engine.get_metrics()
engine.shutdown(save_path="memories.json")
```

### 构造参数

| 参数 | 说明 |
|------|------|
| `config` | `NeuroStreamConfig`，默认使用全部默认值 |
| `encoder` | `UnifiedEncoder`，默认 `UnifiedEncoder.default(dim)` |
| `projector` | `MemoryProjector`，默认自动创建 |
| `consolidation` | `ConsolidationStrategy`，默认 `TimeIntegralStrategy` |
| `forgetting` | `ForgettingStrategy`，默认 `None` |

### 方法

| 方法 | 说明 |
|------|------|
| `start()` | 启动推理 + 学习两个进程 |
| `ingest(data, modality, **kwargs)` | 喂入单条数据 |
| `ingest_batch(items)` | 批量喂入 `List[Dict]` |
| `get_metrics()` | 获取训练指标 `List[Dict]` |
| `shutdown(save_path, timeout)` | 优雅关闭，可选保存记忆池 |
| `running` | 是否运行中 |

## 双进程架构

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│ 推理进程 (Inference)         │     │ 学习进程 (Learning)       │
│                             │     │                          │
│ input_q → Encode → Project  │     │ memory_q → Buffer        │
│         → Search → Memory   │────→│         → Consolidate    │
│         → output            │     │         → Decay / Prune  │
│                             │←────│         → Shadow Train   │
│ EMA pull shadow weights     │     │         → Push weights   │
└─────────────────────────────┘     └──────────────────────────┘
```

### 通信通道 (`ChannelSet`)

| 通道 | 方向 | 数据 |
|------|------|------|
| `input_q` | 外部 → 推理 | 原始输入 |
| `memory_q` | 推理 → 学习 | 新 Memory |
| `search_req_q` | 推理 → 学习 | (query_vec, top_k) |
| `search_resp_q` | 学习 → 推理 | `List[Memory]` |
| `result_q` | 学习 → 外部 | 最终记忆池 |
| `metrics_q` | 学习 → 外部 | 训练指标 |

### 生命周期

1. `start()` — spawn 两个进程
2. `ingest()` — 数据推入 `input_q`
3. 推理进程编码 + 投射 + 检索
4. 学习进程固化 + 训练影子权重
5. `shutdown()` — 毒丸关闭推理 → stop_event 关闭学习 → 收集结果
