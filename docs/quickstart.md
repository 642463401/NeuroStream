# Quickstart

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

只需要 `torch` 和 `tiktoken`，无其他依赖。

## 2. 放置权重文件

将 `checkpoint_best.pt` 放到项目根目录下：

```
NeuroStream/
├── neurostream/
├── output_unsupervised/
│   └── checkpoint_best.pt    ← 放这里
├── chat.py
└── requirements.txt
```

## 3. 开始对话

```bash
python chat.py
```

```
Device: cuda
Loading checkpoint: output_unsupervised/checkpoint_best.pt
Model: 97.2M params | d=512, 12L, 8H

NeuroStream Medical QA
Type your question, or 'quit' to exit.

You: 头痛怎么办？
AI: 头痛的原因很多，建议先排除器质性病变...

You: What causes a fever?
AI: Fever is typically caused by infections...
```

支持中英文医学问答。输入 `quit` 退出。

## 可选参数

```bash
# CPU 推理（无显卡时）
python chat.py --cpu

# 调节生成参数
python chat.py --temperature 0.7 --top-k 50 --max-tokens 512

# 指定其他 checkpoint
python chat.py --checkpoint path/to/other_checkpoint.pt
```

## 最小打包清单

只需以下文件即可运行对话：

```
NeuroStream/
├── neurostream/
│   └── transformer/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── generate.py
│       └── tokenizer.py
├── output_unsupervised/
│   └── checkpoint_best.pt       # ~1.3GB 权重
├── chat.py
└── requirements.txt
```

## 进阶用法

### Python API 调用

```python
import torch
from neurostream.transformer.tokenizer import Tokenizer
from neurostream.transformer.generate import generate, GenerationConfig
from chat import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("output_unsupervised/checkpoint_best.pt", device)
tokenizer = Tokenizer()

query = "糖尿病的早期症状有哪些？"
input_ids = tokenizer.encode_conversation(query)
input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

output = generate(model, input_tensor, tokenizer=tokenizer,
                  gen_config=GenerationConfig(temperature=0.5, max_new_tokens=256))
print(tokenizer.decode(output[0].tolist()))
```

### 训练自己的模型

```bash
# 准备数据后训练
python train_medical_unsupervised.py --epochs 10

# 评测
python eval_medical_qa.py --checkpoint output_unsupervised/checkpoint_best.pt

# 消融实验
python run_ablation.py --list
python run_ablation.py --phase 1
```
