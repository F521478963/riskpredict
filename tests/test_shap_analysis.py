import unittest
from pathlib import Path

from shap_analysis import CLASSIFICATION_SPEC, REGRESSION_SPECS, analyze_model


class ShapAnalysisTest(unittest.TestCase):
    def test_classification_shap_outputs(self):
        output_dir = Path("20260610_most_powerful/shap_results_test")
        report = analyze_model(CLASSIFICATION_SPEC, output_dir, max_background=50)

        self.assertEqual(report["n_features"], 15)
        self.assertEqual(report["n_samples"], 240)
        self.assertIn("top_features", report)
        self.assertGreater(len(report["top_features"]), 0)

        model_dir = output_dir / "overall"
        expected_files = [
            "figure5_summary.png",
            "figure5_importance.png",
            "figure5_composite.png",
            "figure5_shap_values.csv",
            "figure5_feature_ranking.csv",
            "figure5_report.json",
        ]
        for filename in expected_files:
            self.assertTrue((model_dir / filename).exists(), filename)

    def test_regression_specs_match_branch_models(self):
        self.assertEqual([spec.model_id for spec in REGRESSION_SPECS], ["lad", "lcx", "rca"])


if __name__ == "__main__":
    unittest.main()
