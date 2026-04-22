# Types

## `Modality`

模态枚举。

```python
from neurostream import Modality

Modality.TEXT   # "text"
Modality.IMAGE  # "image"
Modality.AUDIO  # "audio"
```

## `TierLevel`

存储层级枚举。

```python
from neurostream import TierLevel

TierLevel.HOT   # "hot"  — FAISS 内存索引, sub-ms
TierLevel.WARM  # "warm" — numpy 矩阵, ~ms
TierLevel.COLD  # "cold" — 磁盘 JSON, ~10ms
```

## `Memory`

记忆单元 — 记忆池中的最小存储单位。

```python
from neurostream import Memory
import torch

mem = Memory(
    vector=torch.randn(128),     # (dim,) 概念向量
    modality=Modality.TEXT,       # 模态，默认 TEXT
    intensity=1.0,               # 强度，时间衰减
    timestamp=time.time(),       # 创建时间
    access_count=0,              # 被检索次数
    tier=TierLevel.HOT,          # 存储层级
    metadata={"source": "api"},  # 自定义元数据
    source_hash=None,            # 可选去重哈希
)
```

### 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector` | `Tensor` | (必填) | 概念向量，L2 归一化 |
| `modality` | `Modality` | `TEXT` | 模态标识 |
| `intensity` | `float` | `1.0` | 记忆强度，随时间衰减 |
| `timestamp` | `float` | `time.time()` | 创建/更新时间戳 |
| `access_count` | `int` | `0` | 被检索命中次数 |
| `tier` | `TierLevel` | `HOT` | 当前存储层级 |
| `metadata` | `Dict[str, Any]` | `{}` | 任意键值对 |
| `source_hash` | `Optional[str]` | `None` | 可选，用于去重 |
