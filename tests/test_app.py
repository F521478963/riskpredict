import unittest
from unittest.mock import patch

from app import (
    RAG_CORPUS_DIR,
    app,
    ai_analyzer,
    classify_branch_qfr,
    classify_risk,
    FEATURE_FIELDS,
    FEATURE_GROUPS,
    rag_corpus_store,
)


from risk_config import BRANCH_QFR_THRESHOLDS, RISK_THRESHOLD


class AppTest(unittest.TestCase):
    def test_classify_risk_uses_ridge_threshold(self):
        self.assertEqual(classify_risk(RISK_THRESHOLD + 0.01)["label_en"], "High Risk")
        self.assertEqual(classify_risk(RISK_THRESHOLD)["label_en"], "High Risk")
        self.assertEqual(classify_risk(RISK_THRESHOLD - 0.01)["label_en"], "Low Risk")

    def test_classify_branch_qfr_uses_branch_thresholds(self):
        lad_threshold = BRANCH_QFR_THRESHOLDS["lad"]
        self.assertEqual(
            classify_branch_qfr(lad_threshold + 0.01, lad_threshold)["label_en"],
            "Normal",
        )
        self.assertEqual(
            classify_branch_qfr(lad_threshold - 0.01, lad_threshold)["label_en"],
            "Attention",
        )

    def test_feature_fields_include_branch_features(self):
        self.assertEqual(len(FEATURE_FIELDS), 33)

    def test_manual_form_prediction_displays_branch_qfr_panel(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Branch QFR Panel", response.data)
        self.assertIn(b"LAD (Left Anterior Descending)", response.data)
        self.assertIn(b"LCX (Left Circumflex)", response.data)
        self.assertIn(b"RCA (Right Coronary Artery)", response.data)

    def test_manual_form_prediction_displays_result(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)
        self.assertIn(b"Predicted Value", response.data)
        self.assertIn(b"Risk Level", response.data)

    def test_manual_form_prediction_displays_shap_panel(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "0.5"

        response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SHAP Interpretability", response.data)
        self.assertIn(b'id="shap-panel"', response.data)
        self.assertIn(b"Overall Screening (Ridge-RF)", response.data)
        self.assertIn(b"shap-table", response.data)
        self.assertIn(b"Push Up", response.data)

    def test_manual_form_prediction_displays_ai_analysis_when_available(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "### Combined Analysis\n\nAI Risk Analysis Report",
                "error": None,
                "judgment_mode": "combined",
                "judgment_label": "Combined Analysis",
            }
            response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        analyze.assert_called_once()
        self.assertEqual(
            analyze.call_args.kwargs.get("judgment_mode"),
            "combined",
        )
        self.assertIn(b"AI-Assisted Analysis", response.data)
        self.assertIn(b"analysis-report-data", response.data)
        self.assertNotIn(b'data-action="export-md"', response.data)
        self.assertIn(b"Diagnostic Assessment", response.data)

    def test_homepage_contains_combined_submit_button(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Combined Analysis", response.data)
        self.assertIn(b"submit-combined", response.data)
        self.assertNotIn("0提示词".encode("utf-8"), response.data)
        self.assertNotIn("简易提示词".encode("utf-8"), response.data)
        self.assertNotIn("仅RAG判断".encode("utf-8"), response.data)
        self.assertNotIn(b"ai_judgment_mode", response.data)
        self.assertNotIn(b"setJudgmentMode", response.data)

    def test_manual_form_always_uses_combined_judgment_mode(self):
        client = app.test_client()
        data = {"mode": "manual", "ai_judgment_mode": "zero_prompt"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {"content": "Combined report", "error": None}
            client.post("/", data=data)

        self.assertEqual(analyze.call_args.kwargs.get("judgment_mode"), "combined")

    def test_homepage_groups_feature_inputs_by_feature_type(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(FEATURE_GROUPS), 4)
        self.assertIn(b"Face Texture Features", response.data)
        self.assertIn(b"face_texture", response.data)
        for group in FEATURE_GROUPS:
            title = group["title_en"].replace("&", "&amp;").encode("utf-8")
            self.assertIn(title, response.data)

    def test_homepage_contains_ai_loading_result_area(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="analysis-status"', response.data)
        self.assertIn(b'id="analysis-progress-bar"', response.data)
        self.assertIn(b'id="analysis-complete-toast"', response.data)
        self.assertIn(b'class="loading-spinner"', response.data)
        self.assertIn(b"Running Combined Analysis", response.data)
        self.assertIn(b"setAnalysisInProgress", response.data)
        self.assertIn(b"initAnalysisCompleteToast", response.data)
        self.assertIn(b"initAnalysisModules", response.data)
        self.assertNotIn("判断依据".encode("utf-8"), response.data)
        self.assertIn(b"diagnostic assessment", response.data)

    def test_manual_form_renders_markdown_reader_when_analysis_exists(self):
        client = app.test_client()
        data = {"mode": "manual"}
        for field in FEATURE_FIELDS:
            data[field["name"]] = "1.0"

        with patch("app.ai_analyzer.analyze") as analyze:
            analyze.return_value = {
                "content": "### Combined Analysis\n\nSample content",
                "error": None,
                "judgment_mode": "combined",
                "judgment_label": "Combined Analysis",
            }
            response = client.post("/", data=data)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'id="evidence-markdown-reader"', response.data)
        self.assertIn(b'id="assessment-markdown-reader"', response.data)
        self.assertIn(b"analysis-report-data", response.data)
        self.assertIn(b"Combined Analysis", response.data)

    def test_homepage_contains_reset_test_data_button(self):
        response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="reset-test-data-button"', response.data)
        self.assertIn(b"Reset Data", response.data)
        self.assertIn(b"resetTestData", response.data)

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
