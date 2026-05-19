# 系统性学习:AI / 深度学习 / Transformer

> 本文档是通用 AI 知识体系。每个 § 学完,去 [02_project_codex.md](02_project_codex.md) 对应章节看 NeuroStream 怎么落地。
> B 站资源给的是 **UP 主名 + 搜索关键词**,不给 BV 号(易失效)。在 B 站搜索框输入即可。

**导航**:[§1](#1-线性代数与张量) · [§2](#2-微积分与反向传播) · [§3](#3-概率与信息论) · [§4](#4-神经网络构件) · [§5](#5-训练循环与优化) · [§6](#6-词元化与语言建模) · [§7](#7-transformer-架构) · [§8](#8-自回归生成) · [§9](#9-对比学习与表征) · [§10](#10-向量检索与记忆系统) · [§11](#11-持续学习与抗灾难性遗忘) · [§12](#12-多进程异步训练系统) · [§13](#13-agent--工具系统)

---

## §1 线性代数与张量

**为什么这是地基**:Transformer 的 attention 公式 `softmax(QK^T/√d)V` 就是矩阵乘法;模型参数都是张量;每个 `.shape` 都涉及维度推理。看不懂矩阵乘法 = 看不懂任何深度学习代码。

### 必须搞懂的 8 个概念

1. **向量** — 数列,有方向 + 大小
2. **点积**(dot product)— `a·b = Σ aᵢbᵢ`,衡量两向量"对齐度"
3. **余弦相似度** — `cos(a,b) = a·b / (|a||b|)`,归一化后的点积,值域 [-1, 1]
4. **矩阵乘法** — `(M×K) @ (K×N) = (M×N)`,K 必须对齐
5. **L2 范数** — `|x| = √Σxᵢ²`,向量"长度"
6. **张量**(Tensor)— 多维数组,PyTorch 核心数据结构
7. **广播**(broadcasting)— 形状不同时如何"自动"对齐
8. **`einsum`** — 通用张量收缩,看懂 attention 的关键

### 关键问题(自测)

- 为什么 attention 公式里要除以 `√d_k`?
- L2 归一化后,点积和余弦相似度的关系是?
- `tensor.shape = (B, T, V)` 这三个维度分别是什么?

### B 站资源

- **3Blue1Brown 中文官方** — 搜「线性代数的本质」(15 集,精华前 7 集,4 小时)
- **跟李沐学AI** — 搜「数据操作 数据预处理」(d2l 教程前几集)
- **小破站** — 搜「PyTorch 张量」(入门视频很多,挑一个 1h 内的)

### 验收

打开 PyTorch shell,不查文档完成:
```python
x = torch.randn(4, 8, 16)   # (B=4, T=8, D=16)
y = torch.randn(16, 32)
z = x @ y                    # z.shape = ?  → (4, 8, 32)
```
能解释最后一步怎么算出来的。

---

## §2 微积分与反向传播

**为什么重要**:训练就是"求梯度 → 更新参数"。`loss.backward()` 一行背后是整张计算图的链式法则。

### 必须搞懂的 7 个概念

1. **导数** — 函数变化率
2. **偏导数** — 多元函数对单个变量的导数
3. **梯度**(gradient)— 所有偏导组成的向量,**指向最快上升方向**
4. **链式法则** — `(f∘g)'(x) = f'(g(x))·g'(x)`,反向传播的数学基础
5. **计算图**(computation graph)— 张量操作组成的 DAG
6. **自动微分**(autograd)— PyTorch 自动构建计算图 + 反向求梯度
7. **梯度爆炸 / 消失** — 深网络中的两大经典问题,引出 ResNet / LayerNorm

### 关键问题

- `loss.backward()` 之后,参数是如何被更新的?
- 为什么要做梯度裁剪(`clip_grad_norm_`)?
- `with torch.no_grad():` 块里做的操作,梯度还会被记录吗?

### B 站资源

- **3Blue1Brown 中文官方** — 搜「微积分的本质」(可选,只看前 4 集即可)
- **3Blue1Brown 中文官方** — 搜「神经网络」第 3 集"反向传播的本质"(**必看**)
- **Andrej Karpathy 中文搬运** — 搜「Zero to Hero micrograd」(**手搓 autograd,2 小时,神课**)

### 验收

能在纸上画出 `y = (x + 2) * 3 → loss = y²` 的计算图,并手算 `dloss/dx`。

---

## §3 概率与信息论

**为什么重要**:语言模型输出是概率分布(softmax);损失函数是交叉熵;`perplexity = exp(loss)`。

### 必须搞懂的 7 个概念

1. **softmax** — 把任意实数向量映射到概率分布:`softmax(x)ᵢ = eˣⁱ / Σ eˣʲ`
2. **概率分布** — 一组非负数,和为 1
3. **熵**(entropy)— `H(p) = -Σ p log p`,分布的"不确定度"
4. **交叉熵**(cross-entropy)— `H(p,q) = -Σ p log q`,衡量用 q 表示 p 的代价
5. **KL 散度** — `KL(p||q) = H(p,q) - H(p)`,两分布的"距离"(非对称)
6. **最大似然估计**(MLE)— 选参数使观测数据概率最大,等价于最小化交叉熵
7. **Perplexity** — `ppl = exp(cross_entropy)`,语言模型评估标配,"平均不确定 token 数"

### 关键问题

- 训练日志显示 `val_loss=2.16, ppl=8.7`,两个数的关系是?
- 为什么分类任务用交叉熵而不是 MSE?
- `label_smoothing=0.1` 在数学上做了什么?

### B 站资源

- **跟李沐学AI** — 搜「softmax 回归」(d2l 课)
- **跟李沐学AI** — 搜「信息论 交叉熵」
- 搜「KL 散度 通俗讲解」

### 验收

不看公式手算:`p = [0.7, 0.2, 0.1]`,真实标签是 0,交叉熵损失 = ?(答:`-log(0.7) ≈ 0.357`)

---

## §4 神经网络构件

**为什么重要**:Transformer 是堆出来的乐高,先认识每个积木块。

### 必须搞懂的 8 个概念

1. **全连接层 `nn.Linear(in, out)`** — 矩阵乘 + 偏置:`y = xW + b`
2. **激活函数**:ReLU / GELU / **SwiGLU**(NeuroStream 用)
3. **MLP**(多层感知机)— Linear + 激活 + Linear,FFN 的本体
4. **残差连接**(residual)— `y = x + F(x)`,**让深网络可训**
5. **归一化**:BatchNorm / LayerNorm / **RMSNorm**(NeuroStream 用)
6. **Dropout** — 训练时随机置零,防过拟合
7. **Embedding 层** — `nn.Embedding(vocab, dim)`,把 token id 映射到向量
8. **参数初始化** — Xavier / Kaiming / **zero-init**(NeuroStream 影子权重用)

### 关键问题

- 为什么残差连接 `y = x + F(x)` 让深网络可训?(梯度直通)
- LayerNorm 和 BatchNorm 区别?为什么 Transformer 用前者?
- SwiGLU 比 GELU 强在哪?(门控机制)
- "zero-init 最后一层"意味着初始模型在做什么?(恒等映射)

### B 站资源

- **跟李沐学AI** — 搜「多层感知机 d2l」
- **跟李沐学AI** — 搜「批量归一化」(讲清楚 BN vs LN)
- 搜「SwiGLU 激活函数」
- 搜「RMSNorm LLaMA」

### 验收

读懂 `neurostream/shadow/projector.py` 全部 50 行,讲清:
- `Linear → GELU → Linear` 的形状变换
- 为什么最后一层 zero-init

---

## §5 训练循环与优化

**为什么重要**:`loss.backward() + optimizer.step()` 之外,还有 10 个工程实践决定能不能训出来。

### 必须搞懂的 10 个概念

1. **SGD** — 最基础,`θ ← θ - lr·∇L`
2. **Momentum** — 加"惯性",抗震荡
3. **Adam / AdamW** — 自适应学习率 + 解耦权重衰减(**项目默认**)
4. **学习率调度**:warmup + cosine decay(项目用的组合)
5. **梯度累积**(grad accumulation)— 小显存模拟大 batch:`batch=4, accum=8` ≈ `batch=32`
6. **混合精度训练**(fp16 + GradScaler)— 显存减半 + 速度提升
7. **梯度裁剪** — `clip_grad_norm_(params, max_norm=1.0)` 防爆炸
8. **权重衰减**(weight decay)— L2 正则的另一种表述
9. **Label smoothing** — 让 one-hot 标签变软,提高泛化
10. **EMA**(指数滑动平均)— `θ_ema ← (1-α)·θ_ema + α·θ`,**项目影子权重的核心**

### 关键问题

- 为什么有 warmup?(开始时梯度不稳)
- `batch_size=4, grad_accum=8`,实际等效 batch 是?optimizer 每多少次 forward 才 step 一次?
- fp16 训练为什么需要 GradScaler?
- AdamW 比 Adam 改进在哪?(weight decay 实现位置)
- EMA 的 α=0.01 意味着多少步才能"完全收敛"到新值?

### B 站资源

- **跟李沐学AI** — 搜「优化算法 d2l」(Adam / AdamW 系列)
- **跟李沐学AI** — 搜「混合精度训练」
- 搜「学习率 warmup cosine」
- 搜「梯度累积 显存」

### 验收

能讲清 `transformer/train.py:train_step` 函数里:
- `accum_loop` 为什么除以 `accum_steps`
- `scaler.unscale_ → clip_grad_norm_ → scaler.step` 这三步顺序为什么不能换

---

## §6 词元化与语言建模

**为什么重要**:模型不认字,只认 token id。tokenization 决定模型能看多远、词表多大、参数多冗余。

### 必须搞懂的 6 个概念

1. **Tokenizer**:把字符串 → token id 序列
2. **词表**(vocabulary)— 全部可识别 token,NeuroStream 用 tiktoken cl100k_base 的 **100,277 词表**
3. **BPE**(Byte-Pair Encoding)— 把高频字节对合并成新 token,平衡词表大小 / 序列长度
4. **特殊 token**:BOS(begin)/ EOS(end)/ PAD / SEP
5. **因果语言建模**(Causal LM)— 给定前 n 个 token 预测第 n+1 个,`P(xₙ₊₁ | x₁..ₙ)`
6. **Teacher Forcing** — 训练时用真实 token 作为下一步输入(而非模型自己生成的)

### 关键问题

- BPE 和 char-level / word-level 分词各有什么优劣?
- 中文在 tiktoken 里是如何被切分的?(字节级 BPE)
- Teacher forcing 训练 vs 自回归推理,行为为什么不同?(exposure bias)

### B 站资源

- **Andrej Karpathy 中文搬运** — 搜「Let's build the GPT Tokenizer」(**2 小时,神课**)
- **跟李沐学AI** — 搜「seq2seq」(理解 LM 训练 paradigm)
- 搜「BPE 算法」

### 验收

在 Python shell 里跑:
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("Hello 你好"))  # 看看怎么切
```
能解释为什么"你好"被切成 6 个 token(中文字节级)。

---

## §7 Transformer 架构

**这是项目最核心理论**。NeuroStream 的 `MemoryConditionedTransformer` 就是 GPT + 多一层 cross-attention。

### 必须搞懂的 12 个概念

1. **Self-Attention** — `Attention(Q,K,V) = softmax(QK^T/√d)V`
2. **Q / K / V 的角色** — Query 问、Key 答、Value 取
3. **多头注意力**(Multi-Head)— 并行多套 Q/K/V,捕捉不同关系
4. **因果掩码**(Causal Mask)— 上三角 -inf,确保位置 i 看不到 i+1
5. **位置编码**:
   - Absolute(Vaswani 原版,sin/cos)
   - Learned(GPT-2)
   - **RoPE**(旋转位置编码,NeuroStream + LLaMA 用)
6. **FFN** — Linear → 激活 → Linear,通常 `4d` 隐藏维(SwiGLU 用 `≈ 8d/3` 保参数量)
7. **LayerNorm 位置**:Pre-LN(`x + Attn(LN(x))`)vs Post-LN(原版,难训)
8. **残差连接**:每个 sublayer 都要 `x + sublayer(x)`
9. **KV cache**:推理时缓存历史 K/V,避免重复计算
10. **Cross-Attention**:Q 来自 decoder,K/V 来自外部(NeuroStream 来自记忆)
11. **Encoder-Decoder vs Decoder-Only**:T5 vs GPT
12. **梯度检查点**(gradient checkpointing)— 用计算换显存

### 关键问题(这层每题都要会)

- 为什么 `QK^T / √d_k`?(防 softmax 饱和)
- 多头注意力的"头"在做什么?如果只用 1 头会怎样?
- 为什么 decoder 需要因果掩码,encoder 不需要?
- RoPE 怎么把位置信息编码进 attention?(在 Q/K 上旋转)
- Pre-LN 为什么比 Post-LN 好训?
- `MemoryCrossAttention` 与 `CausalSelfAttention` 三层不同点?

### B 站资源(必看)

- **跟李沐学AI** — 搜「Transformer 论文逐段精读」(**必看,2 小时**)
- **Andrej Karpathy 中文搬运** — 搜「Let's build GPT from scratch」(**手搓 GPT,3 小时,神课**)
- **跟李沐学AI** — 搜「Attention is all you need」
- **跟李宏毅老师学** — 搜「李宏毅 Transformer」(直觉解释)
- 搜「RoPE 旋转位置编码 苏剑林」(中文最佳讲解)
- 搜「LLaMA 架构详解」(SwiGLU / RMSNorm / RoPE 一站搞定)

### 验收

不看代码,在纸上画出一个 Transformer block 的内部数据流(Pre-LN 版),标出每个张量的 `(B, T, D)` 维度。

---

## §8 自回归生成

**Transformer 训练好后,怎么"说话"**。

### 必须搞懂的 7 个概念

1. **Greedy** — 每步选概率最大的 token,确定性 + 无趣
2. **Temperature** — softmax 前除以 T,T 越小越尖锐,T→0 = greedy
3. **Top-k 采样** — 只在 top-k 候选里采样
4. **Top-p / nucleus 采样** — 累积概率到 p 为止
5. **Beam search** — 维护 k 个候选,适合翻译,**不适合开放生成**
6. **Repetition penalty** — 惩罚已出现 token
7. **KV cache** — 缓存历史 K/V,加速生成

### 关键问题

- temperature=0.5 和 temperature=2.0 行为差异?
- top-k=40 与 top-p=0.9 同时设置,先做哪个?
- 为什么对话模型不用 beam search?
- KV cache 节省了什么计算?

### B 站资源

- 搜「LLM 文本生成 采样」
- 搜「HuggingFace generate 函数」
- **跟李沐学AI** — 搜「束搜索」

### 验收

讲清 `transformer/generate.py` 里 `temperature → top_k → top_p` 三步采样的顺序与各自作用。

---

## §9 对比学习与表征

**为什么重要**:NeuroStream 的影子权重训练是 InfoNCE 对比学习。不懂这层,看不懂 `shadow/objectives.py`。

### 必须搞懂的 6 个概念

1. **表征学习**(representation learning)— 学到一个"好用"的向量空间
2. **对比学习** — 让相似样本靠近、不相似的远离,**无需显式标签**
3. **InfoNCE 损失** — `L = -log[exp(sim(q,k+)/τ) / Σ exp(sim(q,k)/τ)]`
4. **温度 τ** — 控制"困难度",τ 小越严苛
5. **L2 归一化 + 余弦相似度** — 对比学习的标配组合
6. **SimCLR / MoCo / CLIP**(代表方法)

### 关键问题

- InfoNCE 与交叉熵的关系?(本质都是交叉熵)
- 为什么 InfoNCE 训练需要 L2 归一化?
- 温度 τ=0.07 vs τ=1.0 行为差别?
- Reward-weighted InfoNCE 怎么把 reward 接进 loss?

### B 站资源

- **跟李沐学AI** — 搜「SimCLR 论文精读」
- **跟李沐学AI** — 搜「MoCo 论文精读」
- **跟李沐学AI** — 搜「CLIP 论文精读」
- 搜「对比学习 InfoNCE」

### 验收

读懂 `shadow/objectives.py:ContrastiveLoss`,讲清:
- batch 内哪些被当正样本、哪些是负样本
- `MemoryGradientComputer` 如何构造对比对

---

## §10 向量检索与记忆系统

**NeuroStream 的"记忆池"就是带衰减/合并/分层的向量数据库**。

### 必须搞懂的 8 个概念

1. **向量数据库**:把"语义相似"变成"几何近邻"
2. **精确最近邻**(brute force)vs **近似最近邻**(ANN)
3. **FAISS** — Facebook 开源的 ANN 库,**项目用 FlatL2 / IVF**
4. **索引算法**:HNSW(图)/ IVF(倒排)/ PQ(乘积量化)
5. **检索打分**:`score = cos(q, k) + log(1 + access_count)`(项目里"越用越强")
6. **时间衰减** — `intensity *= exp(-decay_rate * dt)`,模拟遗忘
7. **分层存储**:Hot / Warm / Cold(类比 CPU cache)
8. **RAG vs 参数化记忆**(NeuroStream 是后者,**核心差异化**)

### 关键问题

- RAG 和 NeuroStream 影子权重的"知识注入"在哪里不同?
- FAISS `IndexFlatL2` 和 `IndexIVFFlat` 时间复杂度对比?
- 为什么记忆要分 Hot/Warm/Cold,而不是一个大池子?
- `pool.decay(dt=5)` 加 `max_dt` 钳制是为了防什么?(PROGRESS bug #8)

### B 站资源

- 搜「FAISS 向量检索」
- 搜「ANN 近似最近邻」
- 搜「HNSW 算法」(进阶)
- 搜「RAG 检索增强生成」(对照学习)

### 验收

跑一个最小 FAISS demo:
```python
import faiss, numpy as np
idx = faiss.IndexFlatL2(128)
idx.add(np.random.randn(1000, 128).astype('float32'))
D, I = idx.search(np.random.randn(5, 128).astype('float32'), k=10)
```
能解释 D / I 各代表什么。

---

## §11 持续学习与抗灾难性遗忘

**NeuroStream 的核心目标之一**:边推理边学习,不能学新忘旧。

### 必须搞懂的 6 个概念

1. **灾难性遗忘**(Catastrophic Forgetting)— 学新任务时把旧任务能力丢光
2. **EWC**(Elastic Weight Consolidation)— 用 Fisher 信息矩阵保护重要参数
   - `L_total = L_new + λ·Σ Fᵢ·(θᵢ - θ*ᵢ)²`
3. **Fisher 信息矩阵** — 衡量参数对似然的敏感度(对角近似)
4. **Experience Replay** — 蓄水池采样旧数据混入训练
5. **蓄水池采样**(Reservoir Sampling)— 流式数据中均匀采样 k 个,O(n) 时间 / O(k) 空间
6. **概念漂移**(Concept Drift)— 数据分布随时间变化

### 关键问题

- EWC 中 λ 太大 / 太小各会怎样?
- Fisher 信息矩阵为什么用对角近似?(全矩阵太大,n×n)
- 蓄水池采样的核心算法是什么?(第 i 个元素以 k/i 概率保留)
- Phase 2 持续学习实验要观察什么指标?(遗忘率 + 新任务收益)

### B 站资源

- 搜「灾难性遗忘 EWC」
- 搜「持续学习 Continual Learning」
- 搜「Fisher 信息」(数学背景)
- **跟李沐学AI** — 搜「持续学习」(可能有,搜搜看)

### 验收

读懂 `forgetting/ewc.py`,讲清:
- `_compute_fisher` 怎么算 Fisher 信息
- `penalty` 函数里那个二次项的物理含义

---

## §12 多进程异步训练系统

**NeuroStream 的工程核心**:推理学习分离,跨进程权重同步。

### 必须搞懂的 8 个概念

1. **进程 vs 线程** — Python GIL,CPU-bound 必须用多进程
2. **`multiprocessing` `spawn` vs `fork`** — Windows 只有 spawn,序列化成本更高
3. **进程间通信**:Queue / Pipe / Shared Memory
4. **PyTorch `torch.share_memory_()`** — 把张量放共享内存,跨进程零拷贝读
5. **Producer-Consumer 模式**:推理是生产者(memory),学习是消费者
6. **毒丸模式**(poison pill)— 用 `None` 通知 worker 退出
7. **EMA 跨进程同步** — 学习进程 push 新权重,推理进程 EMA pull
8. **死锁陷阱**:队列满了 put 阻塞、qsize() 在 Windows 不可靠等

### 关键问题

- 为什么 `mp.set_start_method("spawn")`?(Linux 默认 fork 更快,Windows 没 fork)
- `share_memory_` 后多个进程写同一张量会怎样?(竞态)
- EMA α=0.005 vs α=0.05,推理端权重多久能追上学习端?
- Windows 上 `mp.Queue.qsize()` 为什么不可靠?(PROGRESS bug #4)

### B 站资源

- 搜「Python 多进程 multiprocessing」
- 搜「PyTorch 多 GPU DDP」(进阶但有参考价值)
- 搜「共享内存 Python」
- 搜「Producer Consumer 模式」

### 验收

读懂 `runtime/engine.py` 的 `start()` 和 `runtime/learning.py` 的主循环,在纸上画出 6 个 Queue 的数据流向。

---

## §13 Agent / 工具系统

**NeuroStream 已实现**,但 Phase 1/2 暂不依赖。打算上 OpenAI 兼容 API 时再深入。

### 必须搞懂的 6 个概念

1. **Tool Use / Function Calling** — LLM 输出"调用 X 工具,参数 Y",外部执行后回灌
2. **MCP**(Model Context Protocol)— Anthropic 提出的工具协议,JSON-RPC 2.0
3. **LLM 蒸馏**(Distillation)— Teacher LLM(大,慢)生成数据训练 Student(小,快)
4. **基准评估**(Benchmark)— 4 维(准确/相关/完整/流畅)等
5. **LLM-as-Judge** — 让 LLM 给 LLM 打分
6. **强化学习反馈**(RLHF / DPO)— 用 reward 信号继续训(NeuroStream 用简化版 reward-weighted)

### 关键问题

- Tool use 是怎么在 token 流里"被发现"的?(特殊标记 + 解析)
- MCP 协议为什么用 JSON-RPC 而非 REST?(stdio + 流式)
- 蒸馏与 RAG 的区别?
- LLM-as-judge 的偏见问题?

### B 站资源

- 搜「Function Calling LLM」
- 搜「MCP Model Context Protocol」(较新)
- 搜「RLHF」(关注 OpenAI / Anthropic 风格教程)
- **跟李沐学AI** — 搜「InstructGPT」(RLHF 经典)

### 验收

读懂 `tools/base.py:Tool` ABC 和 `agent/teacher.py:TeacherLLM`,讲清"蒸馏流程"完整一轮的数据走向。

---

## 跨章节自测清单

学完所有 §,这些问题你都该秒答:

| # | 问题 | 涉及 § |
|---|---|---|
| 1 | `val_loss=2.16, ppl=8.7` 的关系? | §3 |
| 2 | 残差 MLP zero-init 后初始行为是什么? | §4 |
| 3 | `batch=4, grad_accum=8, lr=3e-4` 实际等效 batch 是多少? | §5 |
| 4 | tiktoken cl100k_base 词表大小? | §6 |
| 5 | `softmax(QK^T/√d)V` 每个符号的含义和形状? | §7 |
| 6 | top-k=40 + top-p=0.9 + temp=0.5 三者顺序? | §8 |
| 7 | InfoNCE 损失公式? | §9 |
| 8 | FAISS `IndexFlatL2` 时间复杂度? | §10 |
| 9 | EWC penalty 公式? | §11 |
| 10 | `mp.set_start_method("spawn", force=True)` 在 Windows 为什么必要? | §12 |
| 11 | MCP 协议底层用什么? | §13 |
| 12 | NeuroStream 与 RAG 的三层本质差异? | §10+§12 |

---

**下一步**:打开 [02_project_codex.md](02_project_codex.md),按"读代码顺序"系统看一遍 NeuroStream。
