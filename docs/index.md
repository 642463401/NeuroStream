# NeuroStream

**以记忆为核心的 AI 训练框架** — 边推理边学习，推理与学习完全解耦。

## 核心特性

- **Memory 是一等公民** — 不是 Tensor + autograd，而是 Memory + MemoryPool
- **影子权重** — 模型权重真正在变化，不是 RAG 式检索
- **双进程架构** — 推理进程（快/只读）+ 学习进程（慢/写入），互不阻塞
- **可插拔编码器** — FeatureHash（零依赖）/ SBERT / CLIP / Whisper，一行切换
- **抗灾难性遗忘** — EWC 弹性权重固化 + 经验回放
- **分层存储** — Hot (FAISS, sub-ms) → Warm (numpy, ~ms) → Cold (磁盘) 自动晋升降级

## 安装

```bash
pip install -e .

# 可选: 预训练编码器
pip install -e ".[sbert]"     # sentence-transformers
pip install -e ".[clip]"      # CLIP 图像编码
pip install -e ".[whisper]"   # Whisper 音频编码
pip install -e ".[pretrained]" # 全部
```

## 快速开始

```python
from neurostream import NeuroStreamPipeline

with NeuroStreamPipeline(dim=128) as pipe:
    pipe.ingest_many(["用户反馈非常积极", "产品发布成功", "服务器负载正常"])
    pipe.wait(3.0)
    pipe.shutdown(save_path="memories.json")
```

更多示例见 [Quickstart](quickstart.md)。

## 文档导航

| 文档 | 内容 |
|------|------|
| [Quickstart](quickstart.md) | 5 分钟上手（开发者 + 研究者） |
| [Architecture](architecture.md) | 系统架构与设计哲学 |
| **API Reference** | |
| [Config](api/config.md) | NeuroStreamConfig 全部参数 |
| [Types](api/types.md) | Memory / Modality / TierLevel |
| [Encoder](api/encoder.md) | 编码器体系 |
| [Memory](api/memory.md) | 记忆池 + 分层存储 |
| [Shadow](api/shadow.md) | 影子权重系统 |
| [Forgetting](api/forgetting.md) | 抗灾难性遗忘 |
| [Runtime](api/runtime.md) | 双进程引擎 |
| [Pipeline & Trainer](api/pipeline.md) | 用户 API |

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- FAISS (faiss-cpu >= 1.7)
