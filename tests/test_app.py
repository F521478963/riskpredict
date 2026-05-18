import unittest
from unittest.mock import patch

from app import (
    ACS_GUIDELINE_PDF_PATH,
    CONSENSUS_GUIDELINE_PDF_PATH,
    ESC_GUIDELINE_PDF_PATH,
    app,
    ai_analyzer,
    classify_risk,
    FEATURE_FIELDS,
    FEATURE_GROUPS,
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
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "AI 风险分析报告 / AI Risk Analysis Report",
                "error": None,
            }
            response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI 风险分析 / AI Risk Analysis".encode("utf-8"), response.data)
        self.assertIn("AI 风险分析报告 / AI Risk Analysis Report".encode("utf-8"), response.data)

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
        self.assertIn("正在生成 AI 风险分析".encode("utf-8"), response.data)
        self.assertIn(b"showAnalysisLoading", response.data)

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

    def test_app_wires_both_local_guideline_pdfs_into_rag(self):
        self.assertTrue(ACS_GUIDELINE_PDF_PATH.endswith("Guidelines.pdf"))
        self.assertTrue(ESC_GUIDELINE_PDF_PATH.endswith("2024 ESC(1).pdf"))
        self.assertTrue(CONSENSUS_GUIDELINE_PDF_PATH.endswith("冠状动脉功能学临床应用专家共识(1).pdf"))
        source_names = [
            name for name, _retriever in ai_analyzer.guideline_retriever.retrievers
        ]

        self.assertEqual(
            source_names,
            [
                "2025 ACC/AHA ACS Guideline",
                "2024 ESC CCS Guideline",
                "冠状动脉功能学临床应用专家共识",
            ],
        )

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
