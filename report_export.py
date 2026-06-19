import re
from datetime import datetime, timezone


EVIDENCE_SECTION_TITLES = frozenset(
    {
        "风险等级",
        "RAG 要点",
        "RAG 要点与模型补充",
        "依据来源",
        "本地依据",
        "RAG 要点摘要",
        "模型补充评估",
    }
)
ASSESSMENT_SECTION_TITLES = frozenset(
    {
        "综合判断",
        "建议评估与管理措施",
        "模型与筛查局限性",
    }
)


def split_analysis_body(body):
    """将 LLM 正文拆为「判断依据」与「诊断评估与检查建议」两段 Markdown。"""
    body = (body or "").strip()
    if not body:
        return "", ""

    sections = _parse_markdown_sections(body)
    if not sections:
        return "", body

    evidence_parts = []
    assessment_parts = []
    for title, content in sections:
        block = f"### {title}\n{content}".strip()
        if _should_skip_assessment_section(title):
            continue
        if title in ASSESSMENT_SECTION_TITLES:
            assessment_parts.append(block)
        elif title in EVIDENCE_SECTION_TITLES:
            evidence_parts.append(block)
        else:
            assessment_parts.append(block)

    assessment_text = "\n\n".join(assessment_parts).strip()
    return "\n\n".join(evidence_parts).strip(), sanitize_assessment_markdown(assessment_text)


def build_analysis_report(result, risk, ai_analysis, judgment_mode, judgment_labels):
    ai_analysis = ai_analysis or {}
    mode = ai_analysis.get("judgment_mode") or judgment_mode or "combined"
    label = ai_analysis.get("judgment_label") or judgment_labels.get(mode, mode)
    risk = risk or {}
    body = ai_analysis.get("content") or ""
    error = ai_analysis.get("error")

    if error and not body:
        body = f"> ⚠️ {error}"

    _, assessment_body = split_analysis_body(body)
    if body.strip() and not assessment_body:
        assessment_body = sanitize_assessment_markdown(body.strip())

    meta = {
        "generated_at": _format_timestamp(),
        "prediction": result,
        "risk_zh": risk.get("label_zh"),
        "risk_en": risk.get("label_en"),
        "judgment_mode": mode,
        "judgment_label": label,
        "error": error,
    }

    assessment_report = _build_module_report(
        module_key="assessment",
        title="诊断评估与检查建议",
        title_en="Diagnostic Assessment & Recommendations",
        meta=meta,
        body=assessment_body,
    )

    return {
        "title": "AI风险分析报告",
        **meta,
        "body_markdown": body.strip(),
        "assessment_markdown": assessment_body,
        "assessment_report": assessment_report,
    }


def _build_module_report(module_key, title, title_en, meta, body):
    header_title = f"{title} / {title_en}"
    full_markdown = _build_full_markdown(
        title=header_title,
        meta=meta,
        body=body,
        empty_message="（本节暂无内容）",
    )
    export_slug = "assessment"
    return {
        "module": module_key,
        "title": title,
        "title_en": title_en,
        "generated_at": meta["generated_at"],
        "prediction": meta["prediction"],
        "risk_zh": meta["risk_zh"],
        "risk_en": meta["risk_en"],
        "judgment_mode": meta["judgment_mode"],
        "judgment_label": meta["judgment_label"],
        "body_markdown": body,
        "full_markdown": full_markdown,
        "export_slug": export_slug,
        "error": meta.get("error"),
    }


def _build_full_markdown(title, meta, body, empty_message=None):
    header_lines = [
        f"# {title}",
        "",
        f"- 生成时间 / Generated: {meta['generated_at']}",
    ]
    if meta.get("prediction") is not None:
        header_lines.append(
            f"- 预测值 / Predicted Value: `{meta['prediction']:.6f}`"
        )
    else:
        header_lines.append("- 预测值 / Predicted Value: N/A")

    header_lines.extend(
        [
            f"- 风险分级 / Risk Level: **{meta.get('risk_zh') or 'N/A'}** / {meta.get('risk_en') or 'N/A'}",
            f"- 分析模式 / Mode: **{meta.get('judgment_label') or 'N/A'}** (`{meta.get('judgment_mode') or 'N/A'}`)",
            "- 用途说明 / Purpose: 仅供医学研究参考，用于辅助形成更详细的检查建议；不用于实际临床治疗决策。",
            "",
            "---",
            "",
        ]
    )

    content = body.strip() if body else (empty_message or "")
    return "\n".join(header_lines) + content + ("\n" if content else "")


def _should_skip_assessment_section(title):
    skip_keywords = ("不确定性", "局限性", "医生复核要点")
    return any(keyword in title for keyword in skip_keywords)


def sanitize_assessment_markdown(text):
    """诊断评估模块：去除来源标注，并剔除不确定性相关小节。"""
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(r"\[片段\d+\]", "", text)
    text = re.sub(r"\[模型\]", "", text)
    text = re.sub(
        r"####\s*\d+\.\s*[^\n]*(?:不确定性|局限性|医生复核)[^\n]*\n.*?(?=\n####\s|\n###\s|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_markdown_sections(body):
    pattern = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    if not matches:
        return []

    sections = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections.append((title, content))
    return sections


def _format_timestamp():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
