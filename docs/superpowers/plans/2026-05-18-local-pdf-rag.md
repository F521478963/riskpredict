# Local PDF RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 DeepSeek 风险分析前，从本地 ACS 指南 PDF 中按 risk 等级检索相关片段，并把片段作为参考资料注入 prompt。

**Architecture:** 新增一个独立 `pdf_rag.py` 模块负责 PDF 文本抽取、分块、缓存和本地检索。`DeepSeekAnalyzer` 通过依赖注入接收 retriever，在 `analyze()` 内按 `risk['label_en']` 取回指南片段，并在 `_build_prompt()` 中加入“参考指南片段”约束。应用启动时在 `app.py` 配置固定 PDF 路径。

**Tech Stack:** Python 标准库、PyMuPDF (`fitz`)、现有 `unittest` 测试、现有 OpenAI 兼容 DeepSeek SDK。

---

### Task 1: Prompt Injection Test

**Files:**
- Modify: `tests/test_ai_analysis.py`
- Modify: `ai_analysis.py`

- [ ] **Step 1: Write failing test**

Add a test that injects a fake retriever into `DeepSeekAnalyzer`, runs high-risk analysis, and asserts the final DeepSeek prompt contains the retrieved guideline text and risk-specific query terms.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_ai_analysis.DeepSeekAnalyzerTest`
Expected: FAIL because `DeepSeekAnalyzer` does not accept or call a retriever yet.

- [ ] **Step 3: Implement minimal analyzer integration**

Add optional `guideline_retriever` to `DeepSeekAnalyzer.__init__`, call it from `analyze()`, and pass retrieved snippets into `_build_prompt()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_ai_analysis.DeepSeekAnalyzerTest`
Expected: PASS.

### Task 2: Local PDF Retriever

**Files:**
- Create: `pdf_rag.py`
- Create: `tests/test_pdf_rag.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing tests**

Test chunk ranking with in-memory text chunks: high-risk queries should prefer ACS management and invasive evaluation guidance; low-risk queries should prefer discharge/follow-up or lower-risk wording when present.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_pdf_rag`
Expected: FAIL because `pdf_rag.py` does not exist yet.

- [ ] **Step 3: Implement minimal retriever**

Implement `GuidelineRAGRetriever` with text normalization, chunking, cache metadata, PyMuPDF extraction, and deterministic keyword/TF-style scoring without external embedding services.

- [ ] **Step 4: Add dependency**

Add `PyMuPDF` to `requirements.txt`.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_pdf_rag tests.test_ai_analysis`
Expected: PASS.

### Task 3: App Wiring

**Files:**
- Modify: `app.py`
- Modify: `tests/test_ai_analysis.py`

- [ ] **Step 1: Wire PDF path**

Set the ACS PDF path in `app.py` and pass `GuidelineRAGRetriever(pdf_path)` into `DeepSeekAnalyzer`.

- [ ] **Step 2: Keep failure non-blocking**

If PDF parsing or retrieval fails, DeepSeek analysis should continue with an empty guideline context and include no fabricated references.

- [ ] **Step 3: Final verification**

Run: `python -m unittest`
Expected: PASS.
