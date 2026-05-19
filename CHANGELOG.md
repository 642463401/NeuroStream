# Changelog

本项目的所有重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/)。

---

## [0.2.0] - 2026-05-12

方案 A 持续学习训练框架落地。回归 README 原始定位：记忆 + 影子权重 + 双进程 +
Memory-Conditioned cross-attention 全部进入训练路径。早期 V3 GPT-style 监督预训练
脚本下线，仓库改为单一训练入口。

### 训练入口收敛

- 新增 `train.py` 作为唯一训练入口，覆盖 Phase 1 / Phase 2 持续学习与 `--eval-only` /
  `--distill` 子模式
- 下线 `train_medical.py` / `train_medical_unsupervised.py` / `run_ablation.py` /
  `api_server.py` / `chat.py` / `eval_medical_qa.py` / `clean_medical_data.py` /
  `test_pipeline.py` / `main.py` / `memory_pool.py` / `encoder.py` 等历史脚本
- 新增 `prepare_phase2_data.py`：用 phase1 query MD5 集合（24 万+）过滤原始流，
  生成不与 phase1 重叠的 phase2 数据

### 训练路径修复

- **cross-attn 实际见到记忆** — `engine.teach()` 把 query_vec 传给 learning_worker，
  避免 `memory_vectors=zeros(0, dim)` 的空向量训练
- **ConversationBuffer 600K** — `transformer/config.py` 新增 `conversation_buffer_size`
  字段（默认 600K），修复主进程 push 比 decoder 消费快 13× 导致 92% 数据被静默丢弃
- **`pool.decay()` dt 钳制** — 新增 `max_dt=5` 与显式 `dt` 参数，杜绝 snapshot/暂停
  期间累积秒数被一次性 decay 掉，pool 从 179K 一次性掉到 21 的事故
- **inference 同步 decoder 权重** — `engine.start()` 把 resume 的 decoder_state 灌进
  `self._decoder`；`inference.py` 加 5s 周期主动 pull，避免推理端始终用随机权重
- **生成不回灌训练** — `inference._handle_generate()` 移除把生成的 (query, AI 回答)
  推回 `conversation_q`/`memory_q` 的逻辑，杜绝模型拟合自己输出导致的 loss 假性骤降
- **`wait_drain()`** — `train.py` 新增 push-完后让 decoder 继续训练 N 分钟的接口，
  并只通过 `conversation_q` 判断队列空（绕开 Windows `mp.Queue.qsize()` 子进程消费
  后不递减的 bug）

### 评估改进

- `train.py:_score_one()` 在开放医学问答上引入 `char_overlap` / `bigram_jaccard` /
  `len_ratio` / `empty_rate`，替代必然为 0% 的 substring `strict_match`

### 配置

- `NeuroStreamConfig.decoder_buffer_size: int = 600_000`
- `TransformerConfig.conversation_buffer_size: int = 600_000`
- `from_neurostream_config()` 透传上述字段

### 当前可用快照

- `output/phase1/snapshot_final.pt` — 874 MB，3785 decoder steps，220K memories
- `output/phase1_more/snapshot_final.pt` — 880 MB，~5000 steps，225K memories，
  loss 11.5 → 2.7（perplexity ≈ 15）

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
