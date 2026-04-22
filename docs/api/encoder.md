# Encoder

## 编码器体系

```
EncoderBase (ABC)
├── FeatureHashEncoder     — 字符 n-gram 哈希 (零依赖)
├── ProjectedEncoder (ABC) — 维度适配基类
│   ├── SBERTEncoder       — sentence-transformers
│   ├── CLIPImageEncoder   — open_clip
│   └── WhisperAudioEncoder — openai-whisper
└── UnifiedEncoder         — 注册表模式分发
```

## `EncoderBase` (ABC)

所有编码器的基类。

```python
from neurostream.encoder import EncoderBase

class MyEncoder(EncoderBase):
    @property
    def dim(self) -> int: ...
    @property
    def modality(self) -> str: ...
    def encode(self, data) -> Tensor: ...
    def encode_batch(self, data: List) -> Tensor: ...  # 可选覆盖
```

**约定**: `encode()` 返回的向量必须 L2 归一化，shape 为 `(dim,)`。

## `FeatureHashEncoder`

零依赖的确定性文本编码器。相同文本永远产生相同向量。

```python
from neurostream.encoder import FeatureHashEncoder

enc = FeatureHashEncoder(dim=128)
vec = enc.encode("hello world")  # shape=(128,), norm=1.0
```

| 属性/方法 | 说明 |
|----------|------|
| `dim` | 输出维度 |
| `modality` | `"text"` |
| `encode(text)` | 字符 bigram MD5 哈希 → L2 归一化向量 |
| `encode_batch(texts)` | 批量编码 |

## `ProjectedEncoder` (ABC)

预训练编码器的维度适配基类。当模型原生维度 != `dim` 时，自动插入 `nn.Linear` 投射层。

```python
from neurostream.encoder.projection import ProjectedEncoder

class MyPretrainedEncoder(ProjectedEncoder):
    @property
    def _native_dim(self) -> int: return 384
    @property
    def _modality(self) -> str: return "text"
    def _encode_native(self, data) -> Tensor: ...
```

| 属性/方法 | 说明 |
|----------|------|
| `_native_dim` | (抽象) 预训练模型原生维度 |
| `_modality` | (抽象) 模态标识 |
| `_encode_native(data)` | (抽象) 原始编码，返回 `(native_dim,)` |
| `dim` | 投射后输出维度 |
| `encode(data)` | 编码 + 投射 + L2 归一化 + detach + cpu |
| `_projection` | `nn.Linear` 或 `None` (dims 相等时为 None) |

## `SBERTEncoder`

sentence-transformers 文本编码器。

```bash
pip install neurostream[sbert]
```

```python
from neurostream.encoder import SBERTEncoder

enc = SBERTEncoder(dim=128, model_name="all-MiniLM-L6-v2")
vec = enc.encode("语义丰富的文本")  # shape=(128,)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 输出维度 (自动投射) |
| `model_name` | `"all-MiniLM-L6-v2"` | HuggingFace 模型名 |

特性: 懒加载模型 + pickle 安全 (跨进程兼容)。

## `CLIPImageEncoder`

CLIP 图像编码器。

```bash
pip install neurostream[clip]
```

```python
from neurostream.encoder import CLIPImageEncoder

enc = CLIPImageEncoder(dim=128, model_name="ViT-B-32")
vec = enc.encode("photo.jpg")     # 文件路径
vec = enc.encode(pil_image)        # PIL.Image
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 输出维度 |
| `model_name` | `"ViT-B-32"` | CLIP 模型名 |
| `pretrained` | `"openai"` | 预训练权重源 |

## `WhisperAudioEncoder`

Whisper 音频编码器。

```bash
pip install neurostream[whisper]
```

```python
from neurostream.encoder import WhisperAudioEncoder

enc = WhisperAudioEncoder(dim=128, model_name="base")
vec = enc.encode("audio.wav")          # 文件路径
vec = enc.encode(numpy_array_16khz)    # numpy 数组
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | `128` | 输出维度 |
| `model_name` | `"base"` | `tiny`/`base`/`small`/`medium`/`large` |

## `UnifiedEncoder`

多模态编码器注册表，按 modality 分发。

```python
from neurostream.encoder import UnifiedEncoder

# 注册自定义编码器
encoder = UnifiedEncoder(dim=128)
encoder.register(my_text_encoder).register(my_image_encoder)
vec = encoder.encode("hello", modality="text")
```

### 工厂方法

| 方法 | 说明 |
|------|------|
| `UnifiedEncoder.default(dim)` | FeatureHashEncoder (零依赖) |
| `UnifiedEncoder.with_sbert(dim, model_name)` | SBERT 文本编码 |
| `UnifiedEncoder.with_pretrained(dim, text, image, audio, ...)` | 按需组合 |
| `UnifiedEncoder.full_multimodal(dim)` | 全部三模态 |

### 方法

| 方法 | 说明 |
|------|------|
| `register(encoder)` | 注册编码器，返回 self (可链式调用) |
| `encode(data, modality)` | 按模态分发编码 |
| `encode_batch(data, modality)` | 批量编码 |
