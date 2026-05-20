# Vercel 部署指南

Vercel Lambda 单函数有 **500 MB** 体积上限，而 `sentence-transformers`（含 PyTorch）会占用约 2 GB。
因此本项目在生产环境改为 **「轻量依赖 + 词法检索 (lexical)」** 模式；嵌入向量仅在本地开发时使用。

## 1. 部署前本地准备

```bash
# 安装完整开发依赖（含 sentence-transformers / PyMuPDF）
pip install -r requirements-dev.txt

# 把 PDF / Markdown 放进 rag_corpus/ 各子目录后，构建轻量索引
python scripts/rebuild_rag_index.py --skip-embeddings --force

# 检查产物：应有 meta.json / chunks.json / documents.json
ls -lh rag_corpus/.rag_index/
```

> 不要提交 `rag_corpus/.rag_index/embeddings.npy`（已在 `.gitignore` / `.vercelignore` 中排除）。

## 2. Vercel 项目环境变量

在 Vercel Dashboard → Settings → Environment Variables 设置：

| 名称 | 值 | 说明 |
|------|----|------|
| `DEEPSEEK_API_KEY` | 你的密钥 | **必填**，机密 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` 或 `deepseek-v4-flash` | 可选 |
| `RAG_RETRIEVAL_MODE` | `lexical` | 已在 `vercel.json` 默认设置 |
| `RAG_SKIP_EMBEDDINGS` | `1` | 已在 `vercel.json` 默认设置 |
| `DEEPSEEK_TIMEOUT` | `180` | 综合判断较慢时可调大到 300 |

## 3. 部署

```bash
# 首次或更新结构后
vercel --prod

# 或通过 GitHub 集成：直接 push 到主分支
```

## 4. 产物大小估算

| 项 | 大小 |
|----|------|
| `requirements.txt` 依赖（生产） | ~150–200 MB |
| `rag_corpus/` PDF + 词法索引 | ~12 MB |
| 模型 `.dat` / `.dir` / `.bak` | <100 KB |
| 应用代码 + 模板 + prompt | <1 MB |

合计应 < 250 MB，可舒适放进 500 MB 限额。

## 5. 常见问题

### 部署后 AI 报告显示「未检索到」
- 没有把 `rag_corpus/.rag_index/meta.json` 与 `chunks.json` 一起提交。
- 解决：本地执行 `python scripts/rebuild_rag_index.py --skip-embeddings` 后再次部署。

### Lambda 超时
- DeepSeek-V4-Pro 综合判断生成可能 60–120 秒。
- Vercel Hobby 计划 Serverless 函数最长 10s，Pro 计划默认 60s（可加 `vercel.json` 中 `functions[].maxDuration` 调大）。
- 综合判断建议在 Pro 计划上使用，或前端改成异步轮询。

### 想在 Vercel 上也用向量检索
- 需要换更小的嵌入方案：例如调用 OpenAI / DeepSeek 的 embeddings API，
  在 `rag_index.py` 内替换 `SentenceTransformer.encode`。
