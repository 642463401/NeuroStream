# Forgetting

抗灾难性遗忘策略 — 防止学习新知识时遗忘旧知识。

## `ForgettingStrategy` (ABC)

```python
from neurostream.forgetting.base import ForgettingStrategy

class MyStrategy(ForgettingStrategy):
    def compute_penalty(self, model) -> Tensor:
        """加到 loss 上的参数惩罚"""
        ...
    def update_anchor(self, model) -> None:
        """固化后冻结参考点"""
        ...
    def on_new_memories(self, memories) -> None:
        """新记忆入池时的钩子 (可选)"""
        pass
```

## `NoOpStrategy`

什么都不做，返回 0 惩罚。

```python
from neurostream.forgetting import NoOpStrategy
noop = NoOpStrategy()
```

## `EWC`

弹性权重固化 (Elastic Weight Consolidation)。

```python
from neurostream.forgetting import EWC

ewc = EWC(lambda_=1000.0, n_samples=64)
```

### 原理

```
penalty = λ * Σ_i F_i * (θ_i - anchor_i)²
```

- `F_i`: Fisher 信息矩阵对角元素 (衡量参数重要性)
- `anchor_i`: 固化时刻的参数值
- `λ`: 惩罚强度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_` | `1000.0` | 惩罚系数，越大越保守 |
| `n_samples` | `64` | Fisher 估计采样数 |

| 方法 | 说明 |
|------|------|
| `compute_penalty(model)` | 计算 EWC 惩罚 (可微) |
| `update_anchor(model)` | 保存当前权重为锚点 + 估计 Fisher |

## `ExperienceReplay`

经验回放 — 蓄水池采样旧记忆混入训练 batch。

```python
from neurostream.forgetting import ExperienceReplay

replay = ExperienceReplay(buffer_size=256)
replay.on_new_memories(new_mems)   # 蓄水池采样
old_mems = replay.sample(16)      # 随机采样混入训练
```

### 原理

不加参数惩罚，而是通过数据混合防止遗忘。蓄水池采样保证每条历史记忆被保留的概率相等。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `buffer_size` | `256` | 回放缓冲区大小 |

| 方法 | 说明 |
|------|------|
| `on_new_memories(memories)` | 蓄水池采样更新缓冲区 |
| `sample(n)` | 随机采样 n 条 |
| `compute_penalty(model)` | 始终返回 0 |
| `size` | 当前缓冲区大小 |
