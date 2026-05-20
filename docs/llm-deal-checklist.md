# LLM DEAL 清单（基于 JACC 2025 RAG 论文 Supplement）

| Section | Item | 本项目填写 |
|---------|------|------------|
| Model | LLM | DeepSeek API（默认 `deepseek-v4-pro`） |
| Model | 用途 | SVR 筛查输出 + 本地指南 RAG → 辅助报告 |
| Model | 参数 | 仅RAG: max_tokens=4096；综合判断: max_tokens=8192, top_k=5 |
| Model | RAG | `rag_corpus/` + hybrid 检索（embedding + lexical） |
| Prompt | 策略 | Zero-shot + 固定章节 schema（`prompts/clinical_assistant_v1.yaml`） |
| Prompt | 多步 | 检索 → 生成（两阶段）；无迭代 refine |
| Output | 评测 | `eval/run_eval.py` + 可选 DeepEval |
| Safety | 约束 | 仅本地片段、非诊断、依据不足明示 |
| Stochasticity | 控制 | 低温；生产建议固定模型版本 |

## 运维检查表

- [ ] 新材料放入 `rag_corpus/` 对应目录后运行 `python scripts/rebuild_rag_index.py`
- [ ] `python scripts/corpus_status.py` 确认 `index_built=true`
- [ ] 正式环境 `LLM_RAG_MODE=rag`，禁用 `no_rag`
- [ ] 更新 `eval/gold_cases.json` 并跑检索命中率
