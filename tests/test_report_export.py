import unittest

from report_export import build_analysis_report


class ReportExportTest(unittest.TestCase):
    def test_build_analysis_report_includes_metadata_and_body(self):
        report = build_analysis_report(
            result=0.42,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
            ai_analysis={
                "content": "### 综合判断\n\n建议完善检查。",
                "judgment_mode": "combined",
                "judgment_label": "综合判断",
            },
            judgment_mode="combined",
            judgment_labels={"combined": "综合判断", "rag_only": "仅RAG判断"},
        )

        self.assertIn("# AI 风险分析报告", report["full_markdown"])
        self.assertIn("0.420000", report["full_markdown"])
        self.assertIn("综合判断", report["full_markdown"])
        self.assertIn("### 综合判断", report["body_markdown"])


if __name__ == "__main__":
    unittest.main()
