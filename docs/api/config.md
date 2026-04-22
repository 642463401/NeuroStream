# Config

## `NeuroStreamConfig`

框架全局配置，所有组件从此读取参数。

```python
from neurostream import NeuroStreamConfig

config = NeuroStreamConfig(dim=256, shadow_lr=1e-3)
```

### 参数一览

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **向量维度** | | | |
| `dim` | `int` | `128` | 全局向量维度，所有编码器/索引/投射器共享 |
| **记忆池** | | | |
| `decay_rate` | `float` | `0.01` | 时间衰减率，`intensity *= exp(-rate * dt)` |
| `consolidation_threshold` | `float` | `0.3` | 固化强度阈值，低于此值不入池 |
| `prune_threshold` | `float` | `0.1` | 剪枝阈值，低于此值被移除 |
| `merge_similarity_threshold` | `float` | `0.8` | 合并相似度阈值，高于此值合并记忆 |
| **分层存储** | | | |
| `hot_capacity` | `int` | `10,000` | Hot 层最大容量 (FAISS) |
| `warm_capacity` | `int` | `100,000` | Warm 层最大容量 (numpy) |
| `promotion_threshold` | `float` | `0.7` | Warm → Hot 晋升强度阈值 |
| `demotion_interval_sec` | `float` | `300.0` | Warm 中未访问超过此时间则归档到 Cold |
| **影子权重** | | | |
| `shadow_enabled` | `bool` | `True` | 是否启用影子权重学习 |
| `shadow_ema_alpha` | `float` | `0.01` | EMA 拉取系数，~100 次收敛 |
| `shadow_lr` | `float` | `1e-4` | 影子模型学习率 |
| `shadow_update_interval_sec` | `float` | `5.0` | 训练间隔（秒） |
| `shadow_batch_size` | `int` | `32` | 对比学习 batch 大小 |
| **抗遗忘** | | | |
| `forgetting_strategy` | `str` | `"none"` | `"ewc"` / `"replay"` / `"none"` |
| `ewc_lambda` | `float` | `1000.0` | EWC 惩罚系数 |
| `replay_buffer_size` | `int` | `256` | 经验回放缓冲区大小 |
| **运行时** | | | |
| `learning_interval_sec` | `float` | `1.0` | 学习进程固化间隔 |
| `inference_search_top_k` | `int` | `5` | 推理时检索记忆数 |
