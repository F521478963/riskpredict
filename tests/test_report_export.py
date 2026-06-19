import unittest

from report_export import (
    build_analysis_report,
    sanitize_assessment_markdown,
    split_analysis_body,
)


class ReportExportTest(unittest.TestCase):
    def test_split_analysis_body_separates_evidence_and_assessment(self):
        body = """### Risk Level
Predicted value 0.42, high risk.

### RAG Key Points
- Point one [Snippet1]

### Clinical Assessment
#### 1. Overall Conclusion
Recommend ECG.

### Sources
[Snippet1] guideline."""
        evidence, assessment = split_analysis_body(body)
        self.assertIn("### Risk Level", evidence)
        self.assertIn("### RAG Key Points", evidence)
        self.assertIn("### Sources", evidence)
        self.assertIn("### Clinical Assessment", assessment)
        self.assertNotIn("### Risk Level", assessment)

    def test_sanitize_assessment_removes_citations_and_uncertainty(self):
        raw = """### Clinical Assessment

#### 1. Overall Conclusion
Recommend ECG [Snippet3].

#### 4. Uncertainties, Limitations & Physician Review
- RAG coverage is limited.

#### 4. Assessment Closing Statement
This assessment only suggests a higher likelihood of underlying coronary artery disease."""

        cleaned = sanitize_assessment_markdown(raw)
        self.assertNotIn("[Snippet3]", cleaned)
        self.assertNotIn("Uncertainties", cleaned)
        self.assertIn("Assessment Closing Statement", cleaned)

    def test_build_analysis_report_includes_assessment_module(self):
        report = build_analysis_report(
            result=0.42,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
            ai_analysis={
                "content": (
                    "### Risk Level\n\nHigh risk.\n\n"
                    "### RAG Key Points\n\nSnippet point.\n\n"
                    "### Clinical Assessment\n\nRecommend additional testing.\n\n"
                    "### Sources\n\n[Snippet1]."
                ),
                "judgment_mode": "combined",
                "judgment_label": "Combined Analysis",
            },
            judgment_mode="combined",
            judgment_labels={"combined": "Combined Analysis", "rag_only": "RAG-Only"},
        )

        self.assertIn("### Clinical Assessment", report["assessment_markdown"])
        self.assertNotIn("### Risk Level", report["assessment_markdown"])
        self.assertIn("For medical research reference only", report["assessment_report"]["full_markdown"])
        self.assertEqual(report["assessment_report"]["export_slug"], "assessment")
        self.assertNotIn("evidence_report", report)


if __name__ == "__main__":
    unittest.main()
