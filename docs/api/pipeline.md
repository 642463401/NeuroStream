# Pipeline & Trainer

## `NeuroStreamPipeline`

开发者级极简 API，零配置启动。

```python
from neurostream import NeuroStreamPipeline

with NeuroStreamPipeline(dim=128) as pipe:
    pipe.ingest("用户反馈积极")
    pipe.ingest_many(["文本1", "文本2", "文本3"])
    pipe.wait(3.0)
    pipe.shutdown(save_path="memories.json")
```

### 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 向量维度 |
| `shadow` | `True` | 是否启用影子权重 |
| `**config_overrides` | | 传递给 `NeuroStreamConfig` 的其他参数 |

### 方法

| 方法 | 说明 |
|------|------|
| `ingest(text, modality, **metadata)` | 喂入一条数据 |
| `ingest_many(texts, modality)` | 批量喂入 |
| `wait(seconds)` | 等待学习进程处理 |
| `shutdown(save_path)` | 关闭并可选保存 |

支持 context manager (`with` 语句)。

---

## `NeuroStreamTrainer`

研究者级完全可控接口。

```python
from neurostream import NeuroStreamTrainer, NeuroStreamConfig, MemoryProjector
from neurostream.forgetting import EWC

config = NeuroStreamConfig(dim=128, shadow_lr=1e-3)
trainer = NeuroStreamTrainer(
    config=config,
    projector=MemoryProjector(dim=128, hidden=256),
    forgetting_strategy=EWC(lambda_=500.0),
)
trainer.start()
trainer.ingest("text data")
metrics = trainer.get_metrics()
trainer.save_checkpoint("ckpt.json")
```

### 构造参数

| 参数 | 说明 |
|------|------|
| `config` | `NeuroStreamConfig` |
| `encoder` | `UnifiedEncoder`，可选 |
| `projector` | `MemoryProjector`，可选 |
| `consolidation_strategy` | `ConsolidationStrategy`，可选 |
| `forgetting_strategy` | `ForgettingStrategy`，可选 |

### 方法

| 方法 | 说明 |
|------|------|
| `start()` | 启动引擎 |
| `shutdown()` | 关闭引擎 |
| `ingest(data, modality, **kwargs)` | 喂入数据 |
| `ingest_batch(items)` | 批量喂入 |
| `get_metrics()` | 获取新训练指标 |
| `save_checkpoint(path)` | 保存记忆池快照 |
| `config` | 当前配置 |
| `total_shadow_steps` | 累计影子权重训练步数 |

---

## `TrainerCallback`

训练事件回调协议 (Protocol)。

```python
from neurostream.api.callbacks import TrainerCallback

class MyCallback:
    def on_ingest(self, data: dict) -> None: ...
    def on_consolidation(self, new_count: int, pool_size: int) -> None: ...
    def on_shadow_step(self, metrics: dict) -> None: ...
    def on_shutdown(self, final_pool: list) -> None: ...
```

### `PrintCallback`

内置的调试回调，打印固化和训练事件到 stdout。

```python
from neurostream.api.callbacks import PrintCallback
cb = PrintCallback()
```
