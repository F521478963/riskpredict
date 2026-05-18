# Multi PDF RAG And Compact Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `2024 ESC(1).pdf` 加入本地 RAG 检索范围，并压缩页面指标输入区让一屏显示更多指标。

**Architecture:** 在 `pdf_rag.py` 新增组合检索器，复用现有单 PDF 检索器的独立缓存与评分逻辑，查询时合并两份指南结果并统一排序。`app.py` 配置 ACC/AHA 与 ESC 两个 PDF 路径，页面模板只调整 CSS，不改变表单字段含义。

**Tech Stack:** Python 标准库、PyMuPDF、Flask/Jinja、现有 `unittest`。

---

### Task 1: Multi PDF RAG

**Files:**
- Modify: `/Users/samuels/riskpredict/pdf_rag.py`
- Modify: `/Users/samuels/riskpredict/app.py`
- Test: `/Users/samuels/riskpredict/tests/test_pdf_rag.py`
- Test: `/Users/samuels/riskpredict/tests/test_app.py`

- [ ] Write failing tests for a combined retriever returning results from multiple child retrievers and app wiring including `2024 ESC(1).pdf`.
- [ ] Run `python3 -m unittest tests.test_pdf_rag tests.test_app` and confirm the new tests fail for missing implementation.
- [ ] Add `CombinedGuidelineRAGRetriever` that accepts retrievers, calls `search_for_risk()`, merges non-empty results, sorts by score, and keeps source metadata if provided.
- [ ] Update `app.py` to define `ESC_GUIDELINE_PDF_PATH` and pass both PDF retrievers through the combined retriever.
- [ ] Run `python3 -m unittest tests.test_pdf_rag tests.test_app` and confirm pass.

### Task 2: Compact Input Layout

**Files:**
- Modify: `/Users/samuels/riskpredict/templates/index.html`
- Test: `/Users/samuels/riskpredict/tests/test_app.py`

- [ ] Write failing page tests asserting tighter grid min width, smaller gaps, smaller group padding, and smaller number input padding.
- [ ] Run `python3 -m unittest tests.test_app.AppTest` and confirm fail.
- [ ] Adjust CSS: reduce `.card` padding, `.feature-grid` min column width and gap, `.feature-group` padding, label spacing/font sizes, and number input padding.
- [ ] Run `python3 -m unittest tests.test_app.AppTest` and confirm pass.

### Task 3: Verification

**Files:**
- No additional source changes.

- [ ] Run `python3 -m unittest discover -s tests -p 'test_*.py'`.
- [ ] Run a smoke script that queries the combined retriever against both PDFs and prints result count and sources.
