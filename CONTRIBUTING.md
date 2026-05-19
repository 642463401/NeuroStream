# Contributing to NeuroStream / 贡献指南

> ⚠️ **专有项目说明 / Proprietary Notice**
>
> NeuroStream 在 [LICENSE](LICENSE) 下为**专有软件，保留所有权利**。仓库虽公开
> 可见，但不接受外部 fork/复制/再分发，也不对外开放 PR 流程。本文件仅供经版权方
> **书面授权**的协作者参考其内部开发约定。
>
> NeuroStream is proprietary software under [LICENSE](LICENSE). The repository
> is publicly visible for display only — it is **not** an open-source project,
> and external pull requests are not accepted. This document is intended for
> collaborators who have received **prior written authorization** from the
> copyright holder.

---

## 开发环境搭建 / Development Setup

### 前置要求

- Python >= 3.10
- (可选) CUDA 兼容 GPU

### 搭建步骤

```bash
git clone https://github.com/<your-username>/NeuroStream.git
cd NeuroStream
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 完整安装 (含测试依赖)
pip install -e ".[full,dev]"

# 运行测试
pytest tests/ -v
# 253 passed in ~9s
```

### Windows 注意事项

- 多进程代码必须在 `if __name__ == '__main__':` 保护下运行
- 控制台编码：设置 `PYTHONIOENCODING=utf-8` 或在代码中调用 `sys.stdout.reconfigure(encoding='utf-8')`
- 路径使用 `os.path.join()` 或 `pathlib.Path`，不要硬编码 `/`

---

## 代码规范 / Code Style

### 通用规范

- 类型注解：所有公共 API 必须有类型注解
- 文档字符串：中文或英文均可，与周围代码保持一致
- 导入顺序：标准库 → 第三方 → 本地，各组之间空一行
- 行宽：无硬性限制，保持可读性即可

### 项目约定

- **ABC 模式** — 新模块遵循抽象基类模式（参考 `encoder/base.py`、`forgetting/base.py`）
- **懒加载** — 可选依赖在函数内部导入，未安装时不影响核心功能
- **L2 归一化** — 所有编码器输出必须 L2 归一化
- **零初始化** — 新增的投射层使用 zero-init，确保初始输出为零（参考 `shadow/projector.py`）
- **跨进程通信** — 仅通过 `mp.Queue` 和 `SharedWeightBuffer`，不直接共享 Python 对象

### 安全

- **禁止硬编码 API Key** — 使用环境变量 `os.environ.get("DASHSCOPE_API_KEY")`
- **禁止提交大文件** — 模型权重 (`.pt`)、数据集 (`.zip`) 不入版本控制

---

## 测试要求 / Testing

- 所有 PR 必须通过 `pytest tests/ -v`
- 新功能必须附带测试，放在 `tests/` 目录
- 测试命名：`test_<module>.py`，函数名 `test_<行为描述>`
- 目标：测试套件在 CPU 上 30 秒内完成
- Mock 外部依赖（DashScope、sentence-transformers 等），不依赖网络

---

## 提交 PR / Pull Request Process

1. Fork 本仓库
2. 从 `main` 创建特性分支 (`git checkout -b feature/your-feature`)
3. 编写代码 + 测试
4. 运行完整测试套件
5. 提交 PR，描述包含：
   - **What** — 改了什么
   - **Why** — 为什么改（链接 Issue）
   - **Testing** — 如何测试
6. 等待 Review

---

## 模块开发指南 / Module Guide

### 添加新编码器

```python
# neurostream/encoder/your_encoder.py
from .base import EncoderBase
from ..types import Modality

class YourEncoder(EncoderBase):
    @property
    def dim(self) -> int:
        return 768

    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def encode(self, raw_input) -> Tensor:
        # 实现编码逻辑
        # 输出必须 L2 归一化
        vec = ...
        return vec / vec.norm()
```

注册到 `UnifiedEncoder`，在 `__init__.py` 添加懒导入，在 `pyproject.toml` 添加可选依赖。

### 添加新遗忘策略

```python
# neurostream/forgetting/your_strategy.py
from .base import ForgettingStrategy

class YourStrategy(ForgettingStrategy):
    def compute_penalty(self, model) -> Tensor:
        # 返回正则化惩罚项
        ...

    def update_anchor(self, model, memories) -> None:
        # 更新锚点参数
        ...
```

### 添加新工具

```python
# neurostream/tools/builtin/your_tool.py
from ..base import Tool, ToolResult

class YourTool(Tool):
    @property
    def name(self) -> str:
        return "your_tool"

    @property
    def description(self) -> str:
        return "Tool description"

    @property
    def parameters(self) -> dict:
        return {"param1": {"type": "string", "description": "..."}}

    def execute(self, **kwargs) -> ToolResult:
        # 实现工具逻辑
        ...
```

---

## 架构概览

系统采用双进程模型：

- **推理进程** — 快速只读路径，处理输入 → 编码 → 检索记忆 → 生成输出
- **学习进程** — 后台训练，固化/衰减/剪枝 + 影子权重更新 + Transformer 训练
- **跨进程同步** — 通过 `SharedWeightBuffer`（`torch.share_memory_`）+ EMA 拉取

详见 [docs/architecture.md](docs/architecture.md)。

---

## Issues

- **Bug 报告** — 包含 Python 版本、操作系统、完整 traceback
- **功能请求** — 描述使用场景，而非具体实现方案
- **问题讨论** — 使用 Discussions 标签
