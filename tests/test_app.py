import unittest
from unittest.mock import patch

from app import (
    RAG_CORPUS_DIR,
    app,
    ai_analyzer,
    classify_risk,
    FEATURE_FIELDS,
    FEATURE_GROUPS,
    rag_corpus_store,
)


class AppTest(unittest.TestCase):
    def test_classify_risk_uses_point_eight_threshold(self):
        self.assertEqual(classify_risk(0.81)["label_en"], "Low Risk")
        self.assertEqual(classify_risk(0.8)["label_en"], "Low Risk")
        self.assertEqual(classify_risk(0.2)["label_en"], "High Risk")

    def test_manual_form_prediction_displays_result(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("预测结果 / Prediction Result".encode("utf-8"), response.data)
        self.assertIn("Predicted Value".encode("utf-8"), response.data)
        self.assertIn("风险分级 / Risk Level".encode("utf-8"), response.data)

    def test_manual_form_prediction_displays_ai_analysis_when_available(self):
        client = app.test_client()
        data = {"mode": "manual", "ai_judgment_mode": "rag_only"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "### 风险等级\n\nAI 风险分析报告 / AI Risk Analysis Report",
                "error": None,
                "judgment_mode": "rag_only",
                "judgment_label": "仅RAG判断",
            }
            response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        analyze.assert_called_once()
        self.assertEqual(
            analyze.call_args.kwargs.get("judgment_mode"),
            "rag_only",
        )
        self.assertIn("AI 辅助分析".encode("utf-8"), response.data)
        self.assertIn(b"analysis-report-data", response.data)
        self.assertIn(b'data-action="export-md"', response.data)
        self.assertIn("研究用途声明".encode("utf-8"), response.data)

    def test_homepage_contains_all_judgment_submit_buttons(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("0提示词".encode("utf-8"), response.data)
        self.assertIn("简易提示词".encode("utf-8"), response.data)
        self.assertIn("仅RAG判断".encode("utf-8"), response.data)
        self.assertIn("综合判断".encode("utf-8"), response.data)
        self.assertIn(b'id="ai_judgment_mode"', response.data)
        self.assertIn(b"setJudgmentMode", response.data)
        self.assertIn(b"submit-zero-prompt", response.data)
        self.assertIn(b"submit-simple-prompt", response.data)
        self.assertIn(b"submit-combined", response.data)

    def test_manual_form_passes_zero_prompt_judgment_mode(self):
        client = app.test_client()
        data = {"mode": "manual", "ai_judgment_mode": "zero_prompt"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {"content": "baseline", "error": None}
            client.post("/", data=data)

        self.assertEqual(analyze.call_args.kwargs.get("judgment_mode"), "zero_prompt")

    def test_manual_form_passes_combined_judgment_mode(self):
        client = app.test_client()
        data = {"mode": "manual", "ai_judgment_mode": "combined"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "综合报告",
                "error": None,
                "judgment_label": "综合判断",
            }
            client.post("/", data=data)

        self.assertEqual(analyze.call_args.kwargs.get("judgment_mode"), "combined")

    def test_homepage_groups_feature_inputs_by_feature_type(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(FEATURE_GROUPS), 4)
        self.assertIn("面部纹理指标（均值与方差）".encode("utf-8"), response.data)
        self.assertIn(b"face_texture", response.data)
        for group in FEATURE_GROUPS:
            self.assertIn(group["title_zh"].encode("utf-8"), response.data)

    def test_homepage_contains_ai_loading_result_area(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="analysis-status"', response.data)
        self.assertIn(b'class="loading-spinner"', response.data)
        self.assertIn("正在生成 AI 辅助分析".encode("utf-8"), response.data)
        self.assertIn(b"showAnalysisLoading", response.data)
        self.assertIn(b"initAnalysisModules", response.data)
        self.assertIn("判断依据".encode("utf-8"), response.data)
        self.assertIn("诊断评估与检查建议".encode("utf-8"), response.data)

    def test_manual_form_renders_markdown_reader_when_analysis_exists(self):
        client = app.test_client()
        data = {"mode": "manual", "ai_judgment_mode": "rag_only"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "### 风险等级\n\n测试内容",
                "error": None,
                "judgment_mode": "rag_only",
                "judgment_label": "仅RAG判断",
            }
            response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="evidence-markdown-reader"', response.data)
        self.assertIn(b'id="assessment-markdown-reader"', response.data)
        self.assertIn(b"analysis-report-data", response.data)
        self.assertIn("\\u98ce\\u9669\\u7b49\\u7ea7".encode("utf-8"), response.data)

    def test_homepage_contains_fill_test_data_button(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="fill-test-data-button"', response.data)
        self.assertIn("填入低风险参数".encode("utf-8"), response.data)
        self.assertIn(b"fillLowRiskData", response.data)
        self.assertIn(b"LOW_RISK_FEATURE_VALUES", response.data)

    def test_homepage_contains_reset_test_data_button(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="reset-test-data-button"', response.data)
        self.assertIn("重置参数".encode("utf-8"), response.data)
        self.assertIn(b"resetTestData", response.data)
        self.assertIn(b'id=\"fill-test-data-button\"', response.data)

    def test_homepage_keeps_parameter_buttons_in_floating_toolbar(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="floating-actions"', response.data)
        self.assertIn(b"position: fixed", response.data)
        self.assertIn(b"right: 24px", response.data)
        self.assertIn(b"bottom: 24px", response.data)

    def test_homepage_contains_high_risk_parameter_button(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="fill-high-risk-data-button"', response.data)
        self.assertIn("填入高风险参数".encode("utf-8"), response.data)
        self.assertIn(b"HIGH_RISK_FEATURE_VALUES", response.data)
        self.assertIn(b"fillHighRiskData", response.data)
        self.assertIn(b"177.508639", response.data)
        self.assertIn(b"0.847371", response.data)

    def test_app_wires_rag_corpus_store(self):
        self.assertTrue(RAG_CORPUS_DIR.endswith("rag_corpus"))
        status = rag_corpus_store.status()
        self.assertGreaterEqual(status["document_count"], 1)
        categories = {doc["category"] for doc in status["documents"]}
        self.assertIn("guidelines", categories)

    def test_homepage_uses_compact_feature_input_layout(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b'grid-template-areas:\n        "face_spectrum left_ear_texture right_ear_texture"\n        "face_texture face_texture face_texture";',
            response.data,
        )
        self.assertIn(b"grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))", response.data)
        self.assertIn(b"gap: 10px", response.data)
        self.assertIn(b"padding: 8px 10px", response.data)
        self.assertIn(b"font-size: 12px", response.data)


if __name__ == "__main__":
    unittest.main()
