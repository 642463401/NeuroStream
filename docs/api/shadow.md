# Shadow Weights

影子权重系统是 NeuroStream 的核心差异化 — 让模型权重真正随记忆演化。

## `MemoryProjector`

残差 MLP 投射器。

```python
from neurostream import MemoryProjector

proj = MemoryProjector(dim=128, hidden=256)
refined = proj(raw_vector)                    # x + MLP(x)
normalized = proj.project_and_normalize(raw)  # + L2 归一化
```

### 架构

```
input (dim) → Linear(dim, hidden) → GELU → Linear(hidden, dim, zero-init) → + input
```

**zero-init**: 最后一层权重/偏置初始化为 0，初始行为 = 恒等映射。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 输入/输出维度 |
| `hidden` | `256` | 隐藏层维度 |

| 方法 | 说明 |
|------|------|
| `forward(x)` | `x + MLP(x)` 残差投射 |
| `project_and_normalize(x)` | forward + L2 归一化 |

## `SharedWeightBuffer`

跨进程权重同步，基于 `torch.share_memory_()`。

```python
from neurostream.shadow import SharedWeightBuffer

buffer = SharedWeightBuffer(projector)  # 创建共享内存
buffer.push(shadow_model)              # 学习进程写入
buffer.pull(projector, ema_alpha=0.01) # 推理进程 EMA 拉取
```

### 同步协议

```
学习进程: train → push() → version++
推理进程: pull(ema_alpha=0.01) → θ = (1-α)θ + αθ_shadow
```

`alpha=0.01` → 约 100 次 pull 收敛到新权重 → 推理行为平滑演化，无突变。

| 方法 | 说明 |
|------|------|
| `push(model)` | 写入新权重，返回版本号 |
| `pull(model, ema_alpha)` | EMA 拉取，返回是否更新 |
| `get_version()` | 当前版本号 |

## `ShadowWeightManager`

完整的影子权重训练循环。

```python
from neurostream.shadow import ShadowWeightManager

mgr = ShadowWeightManager(projector, config, forgetting=ewc)
metrics = mgr.train_step(memories)   # 对比学习 + 抗遗忘
mgr.push_to_buffer(shared_buffer)    # 同步到推理进程
```

### 训练流程

1. 从高强度记忆构造对比学习 batch (`MemoryGradientComputer`)
2. 前向 shadow_model → InfoNCE loss (`ContrastiveLoss`)
3. \+ 抗遗忘惩罚 (EWC / NoOp)
4. 反向传播 + 梯度裁剪 + optimizer.step()
5. push 到 SharedWeightBuffer

| 方法 | 说明 |
|------|------|
| `train_step(memories)` | 单步训练，返回 `{"loss", "penalty", "grad_norm", "step"}` |
| `on_consolidation_done(memories)` | 更新遗忘锚点 + 回放缓冲 |
| `push_to_buffer(buffer)` | 写入共享内存 |
| `step_count` | 累计训练步数 |

## `ContrastiveLoss`

InfoNCE 对比学习损失。

```python
from neurostream.shadow import ContrastiveLoss

loss_fn = ContrastiveLoss(temperature=0.07)
loss = loss_fn(projected_vectors, similarity_matrix)
```

## `MemoryGradientComputer`

从记忆池构造对比学习训练 batch。

```python
from neurostream.shadow import MemoryGradientComputer

gc = MemoryGradientComputer(batch_size=32)
vectors, sim_matrix = gc.build_batch(memories)
```

按强度排序取 top-N 记忆，计算两两 FAISS 余弦相似度作为 ground truth。
