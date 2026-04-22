# Changelog

本项目的所有重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/)。

---

## [0.1.0] - 2026-04-10

首次公开发布。包含完整的记忆增强框架、双进程运行时、Transformer 解码器、工具系统和 Agent 训练闭环。

### 核心框架

- **包结构** — `neurostream/` Python 包，`pip install -e .` 即可使用
- **类型系统** — `Memory` / `Modality` / `TierLevel` 核心类型定义
- **配置中心** — `NeuroStreamConfig`，30+ 可调参数，按功能分组
- **用户 API** — `NeuroStreamPipeline`（开发者，5 行上手）+ `NeuroStreamTrainer`（研究者，完全可控）
- **回调协议** — `TrainingCallback` ABC，支持 `on_consolidation` / `on_shadow_update` / `on_decay` 等钩子

### 编码器系统

- **可插拔架构** — `EncoderBase` ABC + `UnifiedEncoder` 注册表工厂
- **FeatureHashEncoder** — n-gram 特征哈希，零外部依赖
- **SBERTEncoder** — sentence-transformers 语义编码
- **CLIPImageEncoder** — open_clip 图像编码
- **WhisperAudioEncoder** — openai-whisper 音频编码
- **ProjectedEncoder** — 维度适配基类，支持任意预训练编码器降维
- **懒加载** — 所有可选依赖按需导入，未安装时零影响

### 记忆管理

- **ShortTermBuffer** — 线程安全的短期缓冲区
- **MemoryPool** — FAISS 向量索引 + 时间衰减 + 依赖注入
- **TieredMemoryPool** — Hot (FAISS, sub-ms) / Warm (NumPy, ~ms) / Cold (磁盘) 三层存储，自动晋升降级
- **TimeIntegralConsolidation** — 时间积分固化策略 (S_t = S_{t-1} * decay + I_new)

### 影子权重系统

- **MemoryProjector** — 残差 MLP，zero-init，确保初始输出为零
- **SharedWeightBuffer** — `torch.share_memory_()` 跨进程共享，EMA 拉取
- **ContrastiveLoss / RewardWeightedContrastiveLoss** — InfoNCE 对比学习 + reward 加权
- **MemoryGradientComputer** — 构造对比学习 batch，计算记忆投影梯度
- **ShadowWeightManager** — 完整训练循环，集成遗忘策略

### 抗灾难性遗忘

- **EWC** — 弹性权重固化，对角 Fisher 信息矩阵惩罚
- **ExperienceReplay** — 经验回放，蓄水池采样，混合训练

### 反馈机制

- **Memory.reward** — [-1, 1] 评分字段，影响检索排序和训练权重
- **FeedbackProvider** ABC — 人工反馈 + LLM 评分器 (DashScope/OpenAI)

### 双进程运行时

- **NeuroStreamEngine** — 生命周期管理，spawn 推理+学习进程
- **InferenceProcess** — 快速只读路径，EMA 权重拉取
- **LearningProcess** — 后台训练，固化/衰减/剪枝 + 影子权重 + Transformer 训练
- **ChannelSet** — 类型化 mp.Queue 封装 (memory_q / search_q / feedback_q / snapshot_q 等)

### Transformer 解码器

- **MemoryConditionedTransformer** — GPT-style Decoder + 记忆交叉注意力
- **CausalSelfAttention** — 因果自注意力 + 因果掩码
- **MemoryCrossAttention** — Q 来自 token，K/V 来自记忆向量，支持维度适配
- **TransformerTrainer** — reward-weighted 训练，梯度累积，fp16，warmup + cosine decay
- **自回归生成** — top-k / top-p 采样，工具调用支持
- **Tokenizer** — tiktoken cl100k_base 封装，100,277 词表

### 工具系统

- **Tool ABC** + **ToolRegistry** — 可插拔工具注册
- **内置工具** — Calculator / PythonExec / HTTPRequest
- **MCP 协议** — JSON-RPC 2.0 客户端，支持外部工具服务器

### Agent 训练闭环

- **TeacherLLM** — DashScope API，LLM 蒸馏训练
- **AgentLoop** — Teacher → Student 蒸馏循环
- **BenchmarkEvaluator** — 4 维评测（准确性/相关性/完整性/流畅性）
- **BenchmarkReporter** — Matplotlib 论文风格报表（表格/曲线/雷达图）

### GPU 支持

- 自动设备检测 (`torch.cuda.is_available()`)
- 计算在 GPU，通信在 CPU
- fp16 混合精度训练

### 测试

- 22 个测试文件，253 个测试用例，~9 秒完成
- 覆盖：类型系统、编码器、记忆管理、影子权重、固化、遗忘、反馈、Transformer、工具、Agent
- 包含线程安全并发测试、梯度流验证、确定性校验

### 文档

- 11 篇 Markdown API 文档 (docs/)
- 25 个核心数学公式 (docs/math_formulas.md)
- 4 种快速上手场景 (docs/quickstart.md)
- 系统架构设计文档 (docs/architecture.md)

### 训练实验

- **V3 医学对话训练** — 97M 参数，250K 对话 (80% 中文)，10 epochs，23.8h on RTX 4060
- val_loss = 3.740, perplexity = 42.1, train-val gap < 0.004（良好泛化）
