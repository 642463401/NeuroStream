# NeuroStream 项目进度报告

> 最后更新: 2026-04-10
> 版本: 0.6.0
> 状态: **Agent 闭环完成，支持 LLM Teacher 蒸馏训练 + 论文风格 Matplotlib 评测表**

---

## 一、项目定位

**以记忆为核心的 AI 训练框架** — 区别于所有现有 AI 框架。

- 核心哲学: Memory 是一等公民，不是 Tensor + autograd
- 核心能力: 边推理边学习，推理进程和学习进程完全解耦
- 核心差异: 影子权重（模型权重真正在变化，而非 RAG 式检索）
- 生物隐喻: 皮层（快思考/推理） + 海马体（慢思考/巩固）

---

## 二、架构总览

```
neurostream/                          Python 包，pip install -e . 可用
├── types.py                          Memory / Modality / TierLevel
├── config.py                         NeuroStreamConfig (30+ 参数)
│
├── encoder/                          可插拔多模态编码器
│   ├── base.py                       EncoderBase (ABC)
│   ├── projection.py                 ProjectedEncoder (维度适配基类)
│   ├── text.py                       FeatureHashEncoder (n-gram hash, 零依赖)
│   ├── sbert.py                      SBERTEncoder (sentence-transformers)
│   ├── image.py                      CLIPImageEncoder (open_clip)
│   ├── audio.py                      WhisperAudioEncoder (openai-whisper)
│   └── unified.py                    UnifiedEncoder (注册表 + 工厂方法)
│
├── memory/                           记忆管理
│   ├── index.py                      IndexBackend ABC + FaissBackend
│   ├── buffer.py                     ShortTermBuffer (线程安全)
│   ├── pool.py                       MemoryPool (FAISS + 依赖注入)
│   └── tiered.py                     TieredMemoryPool (Hot/Warm/Cold)
│
├── consolidation/                    可插拔固化策略
│   ├── base.py                       ConsolidationStrategy ABC
│   └── time_integral.py              S_t = S_{t-1} * decay + I_new
│
├── shadow/                           影子权重 (核心差异化)
│   ├── projector.py                  MemoryProjector (残差MLP, zero-init)
│   ├── sync.py                       SharedWeightBuffer (share_memory_)
│   ├── objectives.py                 ContrastiveLoss + RewardWeightedContrastiveLoss
│   ├── gradient.py                   MemoryGradientComputer (reward-weighted batch)
│   └── manager.py                    ShadowWeightManager
│
├── forgetting/                       抗灾难性遗忘
│   ├── base.py                       ForgettingStrategy ABC + NoOp
│   ├── ewc.py                        EWC (对角 Fisher 惩罚)
│   └── replay.py                     ExperienceReplay (蓄水池采样)
│
├── runtime/                          双进程运行时
│   ├── channels.py                   ChannelSet (6条类型化Queue)
│   ├── inference.py                  推理进程
│   ├── learning.py                   学习进程
│   └── engine.py                     NeuroStreamEngine
│
├── feedback/                         反馈系统
│   ├── base.py                       FeedbackProvider ABC + FeedbackResult
│   ├── llm_scorer.py                 LLMScorer (OpenAI 兼容 API)
│   └── human.py                      HumanFeedback (直接透传)
│
├── transformer/                      Transformer 解码器 (NeuroStream 推理核心)
│   ├── config.py                     TransformerConfig
│   ├── tokenizer.py                  Tokenizer (tiktoken cl100k_base)
│   ├── model.py                      MemoryConditionedTransformer (~32M params)
│   ├── generate.py                   自回归生成 (top-k/top-p/temperature)
│   └── train.py                      TransformerTrainer + ConversationBuffer
│
├── tools/                            工具系统 (Phase 13)
│   ├── base.py                       Tool ABC + ToolResult + ToolCall
│   ├── registry.py                   ToolRegistry (注册/执行/超时)
│   ├── builtin/                      内置工具
│   │   ├── calculator.py             CalculatorTool (AST 安全计算)
│   │   ├── python_exec.py            PythonExecTool (subprocess 沙盒)
│   │   └── http_request.py           HttpRequestTool (urllib)
│   └── mcp/                          MCP 协议
│       ├── client.py                 MCPClient (JSON-RPC 2.0 over stdio)
│       └── bridge.py                 MCPTool (MCP → Tool 桥接)
│
├── agent/                            Agent 闭环训练 (Phase 14)
│   ├── config.py                     AgentLoopConfig / TrainingStep / TrainingLog
│   ├── teacher.py                    TeacherLLM (DashScope qwen3-max)
│   ├── evaluator.py                  BenchmarkEvaluator (4维评测)
│   ├── report.py                     BenchmarkReporter (Matplotlib 论文风格)
│   └── loop.py                       AgentLoop (Teacher→Student 闭环)
│
├── api/                              用户接口
│   ├── trainer.py                    NeuroStreamTrainer (研究者)
│   ├── pipeline.py                   NeuroStreamPipeline (开发者)
│   └── callbacks.py                  TrainerCallback 协议
│
└── utils/                            工具 (预留)

docs/                                 Markdown 文档集 (11 篇)
├── index.md                          项目总览 + 快速开始
├── quickstart.md                     5 分钟上手 (4 种场景)
├── architecture.md                   系统架构 + 设计哲学
└── api/                              API 参考 (8 篇)
    ├── config.md                     NeuroStreamConfig 全参数
    ├── types.md                      Memory / Modality / TierLevel
    ├── encoder.md                    编码器体系
    ├── memory.md                     记忆管理
    ├── shadow.md                     影子权重
    ├── forgetting.md                 抗灾难性遗忘
    ├── runtime.md                    双进程运行时
    └── pipeline.md                   Pipeline / Trainer / Callbacks

tests/                                pytest 单元测试 (253 tests)
├── test_types.py                     Memory / Modality / TierLevel
├── test_config.py                    NeuroStreamConfig
├── test_callbacks.py                 PrintCallback
├── test_encoder_text.py              FeatureHashEncoder
├── test_encoder_projection.py        ProjectedEncoder (via Mock)
├── test_encoder_unified.py           UnifiedEncoder 注册/工厂
├── test_encoder_sbert.py             SBERTEncoder (条件)
├── test_memory_buffer.py             ShortTermBuffer 线程安全
├── test_memory_index.py              FaissBackend
├── test_memory_pool.py               MemoryPool 衰减/剪枝/持久化
├── test_memory_tiered.py             Hot/Warm/Cold 分层逻辑
├── test_consolidation.py             TimeIntegralStrategy
├── test_shadow_projector.py          MemoryProjector
├── test_shadow_objectives.py         ContrastiveLoss + RewardWeighted
├── test_shadow_gradient.py           MemoryGradientComputer (3-tuple)
├── test_shadow_manager.py            ShadowWeightManager
├── test_forgetting.py                NoOp / EWC / Replay
├── test_feedback.py                  反馈机制完整测试 (27 tests)
├── test_transformer.py               Transformer Decoder 完整测试 (33 tests)
├── test_tools.py                     工具系统完整测试 (48 tests)
├── test_agent.py                     Agent 闭环完整测试 (49 tests)
└── test_shadow_e2e.py                跨进程影子权重 E2E
```

---

## 三、Phase 完成状态

| Phase | 内容 | 状态 | 关键验证数据 |
|-------|------|------|-------------|
| 1 | 包结构 + 可插拔抽象层 | done | 所有 import 通过 |
| 2 | 影子权重跨进程同步 | done | 33次push, 30次pull, weight_drift=0.056 |
| 3 | 抗灾难性遗忘 (EWC+Replay) | done | EWC penalty + Replay 混合训练验证 |
| 4 | Runtime 双进程编排 | done | Engine start/ingest/shutdown 生命周期 |
| 5 | 用户 API | done | quickstart.py: 10文本→10记忆, 0报错 |
| 6 | 分层记忆 Hot/Warm/Cold | done | 20记忆→Hot=10,Warm=10; 归档后Cold=5 |
| 7 | 真实编码器集成 | done | ProjectedEncoder 维度适配 + pickle 往返 + 向后兼容 |
| 8 | 单元测试覆盖 | done | 96 tests ALL PASSED in 2.69s, 覆盖 17 个模块 |
| 9 | API 文档 (Markdown) | done | 11 篇文档: index + quickstart + 8 API 参考 + architecture |
| 10 | GPU/CUDA 支持 | done | 自动设备检测, 13 个文件改动, 96 tests 全部通过 |
| 11 | 反馈机制 | done | reward 加权搜索/训练, FeedbackProvider, 123 tests 全部通过 |
| 12 | Transformer Decoder | done | 记忆增强 GPT-style 生成, ~32M params, 156 tests 全部通过 |
| 13 | 工具系统 | done | Tool ABC + Registry + 3内置 + MCP Client, 204 tests 全部通过 |
| 14 | Agent 闭环 | done | LLM Teacher 蒸馏 + 4维基准评测 + Matplotlib 报表, 253 tests 全部通过 |

---

## 四、核心技术实现细节

### 4.1 影子权重同步协议
- **MemoryProjector**: 残差 MLP (~65K 参数), zero-init → day-0 行为 = 恒等
- **SharedWeightBuffer**: torch.share_memory_() 跨进程, 近乎无锁
- **EMA Pull**: alpha=0.01, 约100次pull收敛, 行为平滑演化无突变
- **训练目标**: InfoNCE 对比学习, 从记忆池高强度记忆构造正负对

### 4.2 反馈机制 (Phase 11)
- **Memory.reward**: [-1, 1] 评分, 0=无反馈, 正=好, 负=差
- **Memory.correction**: 纠正内容 (错误记忆的正确版本)
- **search() 融合 reward**: `wake_score = cosine + log(1+access_count) + reward_weight * reward`
- **build_batch() 排序**: `score = intensity * max(0.1, 1 + reward)`, 高 reward 优先入训练 batch
- **RewardWeightedContrastiveLoss**: InfoNCE + per-sample reward 加权, reward=-1 → weight=0.1 (抑制), reward=1 → weight=2 (强化)
- **FeedbackProvider**: 可插拔策略 — LLMScorer (调 OpenAI 兼容 API 自动评分) + HumanFeedback (直接透传)
- **EMA 更新**: `new_reward = 0.7 * old + 0.3 * score`, 平滑更新避免震荡
- **反馈通道**: feedback_q → learning worker → 更新 Memory.reward

### 4.3 筛选漏斗
- 短期缓冲 → 时间积分 (S_t = S_{t-1} * decay + I_new) → 合并相似 → 剪枝弱记忆 → 固化
- 唤醒权重 = 向量余弦相似度 + log(1 + access_count) + reward_weight * reward

### 4.4 分层存储
- Hot: FAISS IndexFlatIP, 内存, sub-ms, 容量上限
- Warm: numpy 矩阵暴力搜索, 内存, ~ms
- Cold: 磁盘 JSON, 全扫描, 仅在Hot+Warm不足时触发
- 自动晋升(Warm→Hot) / 降级(Hot→Warm) / 归档(Warm→Cold)

### 4.5 抗灾难性遗忘
- EWC: 对角 Fisher 信息矩阵约束重要权重偏移
- ExperienceReplay: 蓄水池采样旧记忆混入训练batch
- 两者可通过配置切换, 或自定义 ForgettingStrategy

### 4.6 预训练编码器集成
- **ProjectedEncoder**: 维度适配基类, nn.Linear(native_dim, target_dim, bias=False) + L2 归一化
- 当 native_dim == target_dim 时零开销 (无投射层)
- orthogonal 初始化投射矩阵, 保持嵌入空间距离结构
- **SBERTEncoder**: 包装 sentence-transformers, all-MiniLM-L6-v2 (384维) 等
- **CLIPImageEncoder**: 包装 open_clip, ViT-B-32 (512维) 等, 接受 PIL.Image 或文件路径
- **WhisperAudioEncoder**: 包装 openai-whisper, encoder hidden states mean-pool 为固定向量
- 所有编码器: 懒加载 (首次 encode 时才加载模型) + __getstate__ pickle 安全 (Windows mp.spawn)
- 工厂方法: `UnifiedEncoder.with_sbert()` / `.with_pretrained()` / `.full_multimodal()`
- 可选依赖: `pip install neurostream[sbert]` / `[clip]` / `[whisper]` / `[pretrained]`

### 4.7 工具系统 (Phase 13)
- **Tool ABC**: 所有工具继承 — name/description/parameters/execute
- **ToolResult**: output + success + error; **ToolCall**: name + arguments
- **ToolRegistry**: register/get/execute 带超时保护 (ThreadPoolExecutor)
- **CalculatorTool**: AST 白名单安全计算, 无 eval/exec, 支持 abs/sqrt/log/sin/cos/pi/e
- **PythonExecTool**: subprocess 隔离执行 Python 代码, 带超时
- **HttpRequestTool**: urllib GET/POST, 响应截断 4KB
- **MCPClient**: JSON-RPC 2.0 over stdio, connect→initialize→tools/list→tools/call→disconnect
- **MCPTool**: 将单个 MCP server 工具适配为 Tool 接口
- **Tokenizer**: TOOL_CALL_ID=100261, TOOL_RESULT_ID=100262, decode 自动过滤
- **generate()**: 检测 TOOL_CALL token → 继续采样工具名+参数 → 执行 → 注入 TOOL_RESULT + 结果 → 继续生成
- **config**: tools_enabled=False (默认关闭), tool_timeout_sec=5.0, max_tool_calls_per_generation=5
- **Engine**: call_tool() 显式调用, register_tool() 注册自定义, register_mcp() 连接 MCP server
- **无新外部依赖**: urllib + subprocess + ast + concurrent.futures 均为标准库

### 4.8 Agent 闭环训练 (Phase 14)
- **TeacherLLM**: 封装 DashScope API (qwen3-max), generate/evaluate/generate_qa_pairs
- **AgentLoop**: Teacher generate → Student generate → Teacher evaluate → feedback → ingest → train
- **BenchmarkEvaluator**: 4维评测 — Knowledge QA / Math Reasoning / Tool Use / Memory Recall
- **BenchmarkReporter**: Matplotlib 论文风格图表
  - `render_table()`: 颜色编码对比表 (红→黄→绿), 蓝色表头, 加粗 Overall 行
  - `render_training_curves()`: 双面板 — Teacher 评分曲线 + 各类别 Accuracy 进展
  - `render_radar_chart()`: 极坐标多维能力可视化
- **AgentLoopConfig**: API 配置 + 训练循环参数 + 评测参数
- **TrainingLog**: 步骤记录 + 评测结果 + Engine 指标
- **数据流**: Teacher 生成参考 → Student 尝试 → Teacher 评分 → Feedback 更新 reward → Ingest 注入记忆
- **容错**: Student/Teacher 超时/异常自动降级, 评分 JSON 解析失败 → 默认 0 分
- **依赖**: dashscope>=1.0, matplotlib>=3.5 (可选 `pip install neurostream[agent]`)

### 4.9 Transformer Decoder (Phase 12)
- **MemoryConditionedTransformer**: GPT-style causal decoder + memory cross-attention
- **架构**: 每层 DecoderBlock = CausalSelfAttention + MemoryCrossAttention + FFN (pre-norm residual)
- **记忆注入**: K/V 投射 memory_dim→d_model，每个 token 位置 attend 所有记忆
- **Weight tying**: LM head 与 token embedding 共享权重
- **默认规模**: 6 层 / 256 dim / 4 heads / 1024 FFN = ~32M params
- **Tokenizer**: tiktoken cl100k_base (100K vocab), 支持中英文, BOS/EOS/PAD/SEP 特殊 token
- **生成**: top-k + top-p + temperature 采样, greedy (temp=0) 确定性
- **训练**: reward-weighted CE loss, query 位置 mask 为 -100
- **ConversationBuffer**: deque 环形缓冲, reward 加权采样
- **TransformerTrainer**: 类似 ShadowWeightManager 模式, AdamW (betas=0.9/0.95, wd=0.01)
- **运行时集成**: generate_req_q/generate_resp_q 请求-响应, conversation_q 训练数据通道
- **SharedWeightBuffer**: 同 MemoryProjector 模式, EMA pull (alpha=0.005)

### 4.9 GPU/CUDA 支持
- **设计原则**: 计算在 GPU，通信在 CPU — 模型 forward/backward 在 GPU，输出 `.detach().cpu()` 进入队列/记忆池
- **NeuroStreamConfig**: `device="auto"` → `resolve_device()` 自动检测 CUDA
- **MemoryProjector**: `__init__(device=...)`, forward 自动移动输入到模型设备
- **SharedWeightBuffer**: 缓冲区始终 CPU (`share_memory_()` 要求), push 自动 GPU→CPU, pull 自动 CPU→GPU
- **ShadowWeightManager**: shadow_model 在 GPU 训练, vectors/sim_matrix 自动迁移
- **EWC/NoOp/Replay**: penalty 和 Fisher 张量跟随 model device
- **ProjectedEncoder/SBERT/CLIP/Whisper**: 投射层和预训练模型在指定 device 上加载
- **向后兼容**: 所有改动默认 CPU, 无 GPU 时零影响, 96 个现有测试全部通过

### 4.11 单元测试
- **21 个测试文件, 253 个测试用例**, 覆盖所有可独立测试的模块
- 测试范围: types / config / encoder (text, projection, unified, sbert) / memory (buffer, index, pool, tiered) / consolidation / shadow (projector, objectives, gradient, manager) / forgetting (NoOp, EWC, Replay) / callbacks / feedback (base, human, llm_scorer, end-to-end)
- Phase 11: test_feedback.py (27 tests) — reward / search / loss / FeedbackProvider / LLMScorer / E2E
- Phase 12: test_transformer.py (33 tests) — tokenizer / model / causal masking / generation / training / weight sync
- Phase 13: test_tools.py (48 tests) — Tool ABC / Calculator / PythonExec / HTTP / Registry / MCP mock / Tokenizer tokens / Generation / Config / E2E
- Phase 14: test_agent.py (49 tests) — config / TeacherLLM (mock DashScope) / BenchmarkEvaluator / AgentLoop / BenchmarkReporter (Matplotlib) / imports
- 线程安全测试: ShortTermBuffer 并发 push/flush (10线程×100条)
- 确定性验证: FeatureHashEncoder 相同输入→相同向量
- 梯度流验证: MemoryProjector + ContrastiveLoss + RewardWeightedContrastiveLoss + EWC penalty
- multiprocessing 相关模块 (runtime workers, engine, pipeline, trainer) 由已有 E2E 测试覆盖

---

## 五、用户 API 示例

### 开发者 (5 行上手)
```python
from neurostream import NeuroStreamPipeline

with NeuroStreamPipeline(dim=128) as pipe:
    pipe.ingest_many(["text1", "text2", "text3"])
    pipe.wait(3.0)
    pipe.shutdown(save_path="pool.json")
```

### 研究者 (完全可控)
```python
from neurostream import NeuroStreamTrainer, NeuroStreamConfig, MemoryProjector
from neurostream.forgetting import EWC

config = NeuroStreamConfig(dim=128, shadow_ema_alpha=0.005, ewc_lambda=500.0)
trainer = NeuroStreamTrainer(
    config=config,
    projector=MemoryProjector(dim=128, hidden=256),
    forgetting_strategy=EWC(lambda_=500.0),
)
trainer.start()
for entry in data_stream:
    trainer.ingest(entry["text"])
trainer.save_checkpoint("checkpoint.json")
```

### 自定义编码器
```python
from neurostream.encoder import EncoderBase, UnifiedEncoder

class SBERTEncoder(EncoderBase):
    dim = 384
    modality = "text"
    def encode(self, text): ...

encoder = UnifiedEncoder(dim=384).register(SBERTEncoder())
```

### 预训练编码器 (一行切换)
```python
from neurostream import NeuroStreamPipeline

# 零依赖默认 (FeatureHashEncoder)
with NeuroStreamPipeline(dim=128) as pipe: ...

# SBERT 语义编码
from neurostream.encoder import UnifiedEncoder
encoder = UnifiedEncoder.with_sbert(dim=128)
# 传入 NeuroStreamTrainer 或 NeuroStreamEngine

# 全模态 (text + image + audio)
encoder = UnifiedEncoder.full_multimodal(dim=256)
```

---

## 六、环境信息

- Python: 3.11.2 (.venv)
- PyTorch: 2.11.0 (CPU)
- FAISS: 1.13.2 (CPU)
- OS: Windows
- 安装方式: `pip install -e .`

---

## 七、距离"可发布"的剩余工作

| 项目 | 优先级 | 工作量 | 说明 |
|------|--------|--------|------|
| ~~GPU 支持~~ | ~~done~~ | ~~—~~ | ~~Phase 10 已完成~~ |
| ~~反馈机制~~ | ~~done~~ | ~~—~~ | ~~Phase 11 已完成~~ |
| ~~Transformer Decoder~~ | ~~done~~ | ~~—~~ | ~~Phase 12 已完成~~ |
| ~~工具系统~~ | ~~done~~ | ~~—~~ | ~~Phase 13 已完成~~ |
| ~~Agent 闭环~~ | ~~done~~ | ~~—~~ | ~~Phase 14 已完成~~ |
| CI/CD + PyPI 发布 | P1 | 0.5天 | GitHub Actions |
| REST API 层 | P1 | 1-2天 | FastAPI 封装 |
| 分布式训练 | P2 | 2-3天 | 多 GPU / 多节点 |

**v0.6: Agent 闭环完成。支持 LLM Teacher 蒸馏训练 (DashScope qwen3-max) + 论文风格 Matplotlib 评测表。**
