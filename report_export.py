from datetime import datetime, timezone


def build_analysis_report(result, risk, ai_analysis, judgment_mode, judgment_labels):
    ai_analysis = ai_analysis or {}
    mode = ai_analysis.get("judgment_mode") or judgment_mode or "rag_only"
    label = ai_analysis.get("judgment_label") or judgment_labels.get(mode, mode)
    risk = risk or {}
    body = ai_analysis.get("content") or ""
    error = ai_analysis.get("error")

    header_lines = [
        "# AI 风险分析报告",
        "",
        f"- 生成时间 / Generated: {_format_timestamp()}",
        f"- 预测值 / Predicted Value: `{result:.6f}`" if result is not None else "- 预测值 / Predicted Value: N/A",
        f"- 风险分级 / Risk Level: **{risk.get('label_zh', 'N/A')}** / {risk.get('label_en', 'N/A')}",
        f"- 分析模式 / Mode: **{label}** (`{mode}`)",
        "",
        "---",
        "",
    ]

    if error and not body:
        body = f"> ⚠️ {error}"

    markdown = "\n".join(header_lines) + body.strip() + "\n"
    return {
        "title": "AI风险分析报告",
        "generated_at": _format_timestamp(),
        "prediction": result,
        "risk_zh": risk.get("label_zh"),
        "risk_en": risk.get("label_en"),
        "judgment_mode": mode,
        "judgment_label": label,
        "body_markdown": body.strip(),
        "full_markdown": markdown,
        "error": error,
    }


def _format_timestamp():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
