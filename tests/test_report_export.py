import unittest

from report_export import (
    build_analysis_report,
    sanitize_assessment_markdown,
    split_analysis_body,
)


class ReportExportTest(unittest.TestCase):
    def test_split_analysis_body_separates_evidence_and_assessment(self):
        body = """### 风险等级
预测值 0.42，高风险。

### RAG 要点
- 要点一 [片段1]

### 综合判断
#### 1. 总体结论
建议完善 ECG。

### 依据来源
[片段1] 指南。"""
        evidence, assessment = split_analysis_body(body)
        self.assertIn("### 风险等级", evidence)
        self.assertIn("### RAG 要点", evidence)
        self.assertIn("### 依据来源", evidence)
        self.assertIn("### 综合判断", assessment)
        self.assertNotIn("### 风险等级", assessment)

    def test_sanitize_assessment_removes_citations_and_uncertainty(self):
        raw = """### 综合判断

#### 1. 总体结论
建议完善 ECG [片段3]。

#### 4. 不确定性、局限性与医生复核要点
- RAG 覆盖不足。

#### 4. 评估结语
本次评估仅提示潜在冠脉病变可能性较高。"""

        cleaned = sanitize_assessment_markdown(raw)
        self.assertNotIn("[片段3]", cleaned)
        self.assertNotIn("不确定性", cleaned)
        self.assertIn("评估结语", cleaned)

    def test_build_analysis_report_includes_dual_modules(self):
        report = build_analysis_report(
            result=0.42,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
            ai_analysis={
                "content": (
                    "### 风险等级\n\n高风险。\n\n"
                    "### RAG 要点\n\n片段要点。\n\n"
                    "### 综合判断\n\n建议完善检查。\n\n"
                    "### 依据来源\n\n[片段1]。"
                ),
                "judgment_mode": "combined",
                "judgment_label": "综合判断",
            },
            judgment_mode="combined",
            judgment_labels={"combined": "综合判断", "rag_only": "仅RAG判断"},
        )

        self.assertIn("0.420000", report["full_markdown"])
        self.assertIn("### 综合判断", report["assessment_markdown"])
        self.assertIn("### 风险等级", report["evidence_markdown"])
        self.assertIn("仅供医学研究参考", report["evidence_report"]["full_markdown"])
        self.assertEqual(report["evidence_report"]["export_slug"], "evidence")
        self.assertEqual(report["assessment_report"]["export_slug"], "assessment")


if __name__ == "__main__":
    unittest.main()
