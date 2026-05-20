# LLM 离线评测

对齐 JACC 2025 RAG 论文思路：对比 **RAG** vs **no_rag**，用金标准核心建议做相似度评测。

## 准备金标准

复制 `gold_cases.sample.json` 为 `gold_cases.json`，由心内科填写 `expected_core_recommendation`。

## 运行

```bash
export DEEPSEEK_API_KEY=your_key   # 可选；仅评测检索时可不配置

# 仅检查 RAG 检索是否命中金标准相关片段
python eval/run_eval.py --retrieval-only

# 完整生成评测（需要 API）
LLM_RAG_MODE=rag python eval/run_eval.py
LLM_RAG_MODE=no_rag python eval/run_eval.py
```

## 指标

- `retrieval_hit`：金标准关键词是否出现在 Top-K 片段中
- 生成评测（可选）：对接 DeepEval GEval，阈值 0.5 计为正确
