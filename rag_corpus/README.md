# RAG 资料库 / RAG Corpus

把 PDF、TXT、Markdown 放进对应子目录后，运行索引重建脚本即可被临床助手检索。

## 目录说明

| 目录 | 用途 | 示例 |
|------|------|------|
| `guidelines/` | 临床指南、专家共识 | ACC/AHA ACS、ESC |
| `screening/` | 无创筛查/影像模型论文与说明 | Li 2025 舌背 HSI+CAD |
| `methods/` | RAG、LLM 评测方法学文献 | JACC RAG 论文 |
| `papers/` | 其他参考文献 | 综述、说明书 |
| `inbox/` | **临时投放区**：不确定分类时先丢这里 | 任意新 PDF |

也支持在子目录里再建一层文件夹；索引时会递归扫描。

## 快速开始

```bash
# 1. 把文件放进上述目录（或 inbox）
cp your-paper.pdf rag_corpus/inbox/

# 2. 重建索引（首次或更新材料后）
python scripts/rebuild_rag_index.py

# 若无法访问 HuggingFace，可先用纯词法索引（与 hybrid 的 lexical 部分一致）：
python scripts/rebuild_rag_index.py --skip-embeddings

# 3. 启动应用
python app.py
```

## 可选：manifest.yaml

在 `rag_corpus/manifest.yaml` 里可为单个文件指定显示名称、检索权重或是否参与索引。
未列出的文件仍会自动发现并索引。

## 索引产物

- 生成目录：`rag_corpus/.rag_index/`（已 gitignore，勿手改）
- 旧版单文件缓存 `*.rag_cache.json` 仍可用于兼容；推荐以 corpus 索引为准

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_CORPUS_DIR` | `./rag_corpus` | 资料库根目录 |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `hybrid` / `embedding` / `lexical`（Vercel 建议 `lexical`） |
| `RAG_TOP_K` | `5` | 检索返回片段数（对齐 JACC 论文） |
| `RAG_SKIP_EMBEDDINGS` | - | `1` 时跳过向量；Vercel 部署默认开启 |
| `LLM_RAG_MODE` | `rag` | `rag` 正式环境；`no_rag` 仅离线评测 |
| `DEEPSEEK_TIMEOUT` | `300` | API 读取超时（秒）；综合判断 + V4-Pro 建议 300–600 |

## 依赖与部署

- 本地开发：`pip install -r requirements-dev.txt`（包含 PyMuPDF、sentence-transformers）。
- Vercel 生产：使用根目录的 `requirements.txt`（已剔除 PyMuPDF / sentence-transformers，强制 lexical 模式）。
- 详细 Vercel 部署步骤见 `docs/vercel-deploy.md`。
