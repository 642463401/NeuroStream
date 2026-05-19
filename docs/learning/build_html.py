"""把 docs/learning/ 下的 3 个 md 合并成单文件 HTML(带 TOC + 中文友好样式)

用法:
    python docs/learning/build_html.py
    → 生成 docs/learning/learning_path.html

依赖: markdown, pygments (已在主环境装好)
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
OUT = ROOT / "learning_path.html"

SOURCES = [
    ("00_knowledge_map.md", "知识地图"),
    ("01_systematic.md", "系统性学习"),
    ("02_project_codex.md", "NeuroStream 专项教材"),
]

CSS = """
:root {
    --bg: #fdfdfc;
    --fg: #2c2c2c;
    --muted: #666;
    --accent: #2563eb;
    --code-bg: #f5f5f4;
    --border: #e5e5e5;
    --table-header: #f1f5f9;
    --hover: #eef2ff;
}
* { box-sizing: border-box; }
html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", "Source Han Sans CN", "Noto Sans CJK SC",
                 Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.7;
}
.container {
    display: grid;
    grid-template-columns: 280px 1fr;
    min-height: 100vh;
}
nav.sidebar {
    background: #fafaf9;
    border-right: 1px solid var(--border);
    padding: 1.5rem 1.2rem;
    overflow-y: auto;
    position: sticky;
    top: 0;
    height: 100vh;
}
nav.sidebar h2 {
    font-size: 1rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
nav.sidebar ul {
    list-style: none;
    padding: 0;
    margin: 0 0 1.5rem 0;
}
nav.sidebar li {
    margin: 0.2rem 0;
}
nav.sidebar a {
    color: var(--fg);
    text-decoration: none;
    display: block;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.4;
}
nav.sidebar a:hover {
    background: var(--hover);
    color: var(--accent);
}
nav.sidebar a.h3 {
    padding-left: 1.5rem;
    font-size: 0.85rem;
    color: var(--muted);
}
nav.sidebar a.h2 {
    font-weight: 600;
    color: var(--fg);
}
main {
    padding: 2.5rem 3rem;
    max-width: 920px;
    overflow-x: hidden;
}
.doc-section {
    padding-bottom: 3rem;
    margin-bottom: 3rem;
    border-bottom: 2px solid var(--border);
}
.doc-section:last-child { border-bottom: none; }
h1 {
    font-size: 2.2rem;
    margin: 0 0 1rem 0;
    color: var(--accent);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.5rem;
}
h2 {
    font-size: 1.7rem;
    margin: 2.5rem 0 1rem 0;
    color: #1e3a8a;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}
h3 {
    font-size: 1.3rem;
    margin: 1.8rem 0 0.8rem 0;
    color: #1e40af;
}
h4 {
    font-size: 1.1rem;
    margin: 1.5rem 0 0.6rem 0;
    color: #374151;
}
p { margin: 0.8rem 0; }
strong { color: #111; }
em { color: #555; }
a {
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px dashed var(--accent);
}
a:hover {
    border-bottom-style: solid;
}
code {
    font-family: "JetBrains Mono", "Cascadia Code", Consolas,
                 "Source Code Pro", monospace;
    font-size: 0.88em;
    background: var(--code-bg);
    padding: 0.15em 0.4em;
    border-radius: 3px;
    color: #be185d;
}
pre {
    background: #1f2937;
    color: #e5e7eb;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    overflow-x: auto;
    line-height: 1.5;
    font-size: 0.88em;
    margin: 1rem 0;
}
pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
}
table {
    border-collapse: collapse;
    margin: 1.2rem 0;
    width: 100%;
    font-size: 0.92em;
}
table th, table td {
    border: 1px solid var(--border);
    padding: 0.5rem 0.8rem;
    text-align: left;
    vertical-align: top;
}
table th {
    background: var(--table-header);
    font-weight: 600;
    color: #1e3a8a;
}
table tr:nth-child(even) td {
    background: #fafaf9;
}
blockquote {
    border-left: 4px solid var(--accent);
    margin: 1rem 0;
    padding: 0.5rem 1rem;
    background: #eff6ff;
    color: #1e3a8a;
}
ul, ol {
    padding-left: 1.5rem;
}
li { margin: 0.3rem 0; }
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}
.top-bar {
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
    color: #fff;
    padding: 1.2rem 3rem;
    grid-column: 1 / -1;
}
.top-bar h1 {
    color: #fff;
    border: none;
    padding: 0;
    margin: 0 0 0.3rem 0;
    font-size: 1.6rem;
}
.top-bar p {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}
/* 打印样式 */
@media print {
    .container { grid-template-columns: 1fr; }
    nav.sidebar { display: none; }
    main { padding: 1rem; max-width: 100%; }
    .top-bar { padding: 0.5rem 1rem; }
    pre { font-size: 0.75em; page-break-inside: avoid; }
    table { font-size: 0.8em; page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
}
/* 响应式 */
@media (max-width: 900px) {
    .container { grid-template-columns: 1fr; }
    nav.sidebar { display: none; }
    main { padding: 1.5rem; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroStream 学习路径合集</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<div class="top-bar">
<h1>NeuroStream 学习路径合集</h1>
<p>知识地图 · 系统性 AI 理论 · 项目专项教材 — 三合一阅读视图</p>
</div>
<nav class="sidebar">
<h2>目录</h2>
{toc}
</nav>
<main>
{content}
</main>
</div>
</body>
</html>
"""


def slugify(text: str) -> str:
    """生成锚点 id:中文保留,英文小写,空格变 -,去掉标点"""
    text = re.sub(r"[`*_\[\]()§#]", "", text)
    text = text.strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^\w一-鿿\-]", "", text)
    return text.lower()


def extract_headings(html: str, section_prefix: str) -> list[tuple[int, str, str]]:
    """从渲染的 HTML 提取 h2/h3 标题,生成 [(level, text, anchor_id), ...]"""
    headings: list[tuple[int, str, str]] = []
    # markdown 库的 toc 扩展会自动加 id,但我们自己也需要拦截一份用作侧栏
    pattern = re.compile(r'<h([23])[^>]*>(.*?)</h\1>', re.DOTALL)
    for m in pattern.finditer(html):
        level = int(m.group(1))
        raw_text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not raw_text:
            continue
        anchor = f"{section_prefix}-{slugify(raw_text)}"
        headings.append((level, raw_text, anchor))
    return headings


def inject_anchors(html: str, section_prefix: str) -> str:
    """给 h2/h3 注入 id 属性,与侧栏 anchor 对齐"""
    def replace(m: re.Match) -> str:
        level = m.group(1)
        attrs = m.group(2) or ""
        content = m.group(3)
        raw_text = re.sub(r"<[^>]+>", "", content).strip()
        if not raw_text:
            return m.group(0)
        anchor = f"{section_prefix}-{slugify(raw_text)}"
        if "id=" in attrs:
            attrs = re.sub(r'id="[^"]*"', f'id="{anchor}"', attrs)
        else:
            attrs = f' id="{anchor}"' + attrs
        return f"<h{level}{attrs}>{content}</h{level}>"

    return re.sub(
        r"<h([23])([^>]*)>(.*?)</h\1>",
        replace,
        html,
        flags=re.DOTALL,
    )


def rewrite_relative_md_links(html: str) -> str:
    """把 *.md 内部链接改成 # 锚点,避免点击跳出"""
    # 简化处理:把所有指向 .md 的链接去掉(让 anchor 自身工作)
    # 锚点链接(#xxx) 保留,跨文件 [xxx](yyy.md) 改为 # 段首
    def rewrite(m: re.Match) -> str:
        href = m.group(1)
        if href.endswith(".md"):
            target = Path(href).name.replace(".md", "")
            return f'href="#section-{target}"'
        if ".md#" in href:
            fname, anchor = href.split(".md#", 1)
            target = Path(fname).name
            return f'href="#section-{target}"'
        return m.group(0)

    return re.sub(r'href="([^"]+)"', rewrite, html)


def build() -> None:
    md = markdown.Markdown(extensions=[
        "extra",          # 表格 / fenced code / footnotes 等
        "codehilite",     # pygments 高亮
        "tables",
        "sane_lists",
    ], extension_configs={
        "codehilite": {
            "guess_lang": False,
            "noclasses": True,  # 内联 style 而非 class,无需额外 CSS
            "pygments_style": "friendly",
        },
    })

    sections_html: list[str] = []
    toc_blocks: list[str] = []

    for fname, title in SOURCES:
        path = ROOT / fname
        if not path.exists():
            print(f"[warn] {fname} not found, skipped")
            continue

        raw = path.read_text(encoding="utf-8")
        rendered = md.convert(raw)
        md.reset()

        section_id = f"section-{fname.replace('.md', '')}"
        rendered = inject_anchors(rendered, section_id)
        rendered = rewrite_relative_md_links(rendered)

        sections_html.append(
            f'<div class="doc-section" id="{section_id}">\n{rendered}\n</div>'
        )

        # 侧栏目录:每个 section 一个块
        headings = extract_headings(rendered, section_id)
        toc_block = [
            f'<a class="h2" href="#{section_id}">{title}</a>',
            "<ul>",
        ]
        for level, text, anchor in headings:
            css_class = "h2" if level == 2 else "h3"
            toc_block.append(f'<li><a class="{css_class}" href="#{anchor}">{text}</a></li>')
        toc_block.append("</ul>")
        toc_blocks.append("\n".join(toc_block))

    final = HTML_TEMPLATE.format(
        css=CSS,
        toc="\n".join(toc_blocks),
        content="\n".join(sections_html),
    )

    OUT.write_text(final, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"[ok] generated {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
