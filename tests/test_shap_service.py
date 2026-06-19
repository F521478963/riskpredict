import unittest

from model_registry import FEATURE_SPECS, OVERALL_FEATURE_NAMES
from shap_service import ShapRuntime, build_shap_runtime


class ShapServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = build_shap_runtime()

    def _sample_feature_map(self) -> dict[str, float]:
        return {column: 0.5 for column, *_rest in FEATURE_SPECS}

    def test_explain_all_returns_four_models(self):
        results = self.runtime.explain_all(self._sample_feature_map())
        self.assertEqual(len(results), 4)
        self.assertEqual([item.model_id for item in results], ["overall", "lad", "lcx", "rca"])

    def test_overall_model_outputs_all_selected_features(self):
        result = self.runtime.explain_model("overall", self._sample_feature_map())
        self.assertEqual(result.feature_count, len(OVERALL_FEATURE_NAMES))
        self.assertEqual(len(result.features), len(OVERALL_FEATURE_NAMES))
        self.assertGreater(result.max_abs_shap, 0)

    def test_feature_payload_contains_value_and_shap(self):
        result = self.runtime.explain_model("lad", self._sample_feature_map())
        first = result.features[0]
        payload = first.to_dict()
        self.assertIn("value", payload)
        self.assertIn("shap_value", payload)
        self.assertIn("direction", payload)
        self.assertIn(payload["direction"], {"positive", "negative"})

    def test_to_dict_is_json_ready(self):
        result = self.runtime.explain_model("overall", self._sample_feature_map())
        payload = result.to_dict()
        self.assertEqual(payload["model_id"], "overall")
        self.assertEqual(len(payload["features"]), len(OVERALL_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
