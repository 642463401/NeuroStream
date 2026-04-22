# Memory

## `ShortTermBuffer`

线程安全的短期记忆缓冲区。推理进程写入，学习进程批量消费。

```python
from neurostream.memory import ShortTermBuffer

buf = ShortTermBuffer()
buf.push(memory)
items = buf.flush()  # 原子性返回所有并清空
```

| 方法/属性 | 说明 |
|----------|------|
| `push(memory)` | 线程安全写入一条 Memory |
| `flush()` | 原子性取出全部并清空，返回 `List[Memory]` |
| `empty` | 是否为空 |
| `__len__()` | 当前缓冲大小 |

## `MemoryPool`

长期记忆池 — FAISS 向量索引 + 时间衰减 + 剪枝。

```python
from neurostream.memory import MemoryPool

pool = MemoryPool(dim=128, decay_rate=0.01)
pool.add(memory)
results = pool.search(query_vector, top_k=5)
pool.decay()
pool.prune(threshold=0.1)
pool.save("pool.json")
```

### 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 向量维度 |
| `decay_rate` | `0.01` | 衰减率 |
| `index` | `None` | IndexBackend 实例 (默认 FaissBackend) |
| `consolidation` | `None` | ConsolidationStrategy 实例 |

### 方法

| 方法 | 说明 |
|------|------|
| `add(memory)` | 添加一条记忆 |
| `search(query_vector, top_k)` | 唤醒机制检索: `score = cosine + log(1 + access_count)` |
| `decay()` | 全局衰减: `intensity *= exp(-rate * dt)` |
| `prune(threshold)` | 移除 `intensity < threshold` 的记忆，返回移除数 |
| `consolidate(buffer, threshold)` | 从缓冲区固化记忆，委托 ConsolidationStrategy |
| `save(path)` | JSON 持久化 |
| `MemoryPool.load(path, dim)` | 从 JSON 恢复 |
| `MemoryPool.from_config(config)` | 工厂方法 |
| `size` | 记忆数量 |
| `memories` | 记忆列表副本 |

## `TieredMemoryPool`

三级分层记忆池: Hot / Warm / Cold。

```python
from neurostream.memory import TieredMemoryPool
from neurostream import NeuroStreamConfig

config = NeuroStreamConfig(dim=128, hot_capacity=1000)
pool = TieredMemoryPool(config=config)
pool.add(memory)          # → Hot
results = pool.search(query, top_k=10)  # Hot → Warm → Cold
stats = pool.maintenance()  # 降级 + 归档
pool.save("tiered_data/")
```

### 分层规则

| 路径 | 条件 |
|------|------|
| 新记忆 → Hot | 始终 |
| Hot → Warm | Hot 超容量时，最弱记忆降级 |
| Warm → Hot | 被检索命中且 `intensity >= promotion_threshold` |
| Warm → Cold | 未访问超过 `demotion_interval_sec` |
| Cold → 搜索 | 仅在 Hot + Warm 结果不足 top_k 时 |

### 方法

| 方法 | 说明 |
|------|------|
| `add(memory)` | 加入 Hot，自动降级 |
| `search(query, top_k)` | 跨层检索，自动晋升 |
| `consolidate(buffer, threshold)` | 委托 Hot 层 |
| `decay()` / `prune()` | 委托 Hot 层 |
| `maintenance()` | 执行降级 + 归档，返回统计 |
| `save(directory)` | 保存三层到目录 |
| `TieredMemoryPool.load(directory, config)` | 恢复 |
| `total_size` | 三层总记忆数 |
| `hot` / `warm` / `cold` | 直接访问各层 |
