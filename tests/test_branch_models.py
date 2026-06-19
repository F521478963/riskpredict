import unittest

from model_registry import load_branch_services, predict_branch_qfr, resolve_branch_qfr_panel
from risk_config import BRANCH_QFR_THRESHOLDS, RISK_THRESHOLD


class BranchModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.services = load_branch_services()

    def test_predict_branch_qfr_returns_absolute_value(self):
        feature_map = {name: 1.0 for name in self.services["lad"].feature_names}
        qfr = predict_branch_qfr(self.services["lad"], feature_map)
        self.assertGreaterEqual(qfr, 0.0)

    def test_all_branch_models_are_loaded(self):
        self.assertEqual(set(self.services.keys()), {"lad", "lcx", "rca"})

    def test_resolve_branch_qfr_panel_skips_low_risk_screening(self):
        values = {"lad": 0.9, "lcx": 0.9, "rca": 0.9}
        resolved = resolve_branch_qfr_panel(values, RISK_THRESHOLD - 0.01)
        self.assertEqual(resolved, values)

    def test_resolve_branch_qfr_panel_preserves_existing_attention(self):
        values = {"lad": 0.75, "lcx": 0.9, "rca": 0.9}
        resolved = resolve_branch_qfr_panel(values, RISK_THRESHOLD + 0.1)
        self.assertEqual(resolved, values)

    def test_resolve_branch_qfr_panel_scales_high_risk_all_normal(self):
        values = {"lad": 0.85, "lcx": 0.88, "rca": 0.91}
        resolved = resolve_branch_qfr_panel(values, RISK_THRESHOLD + 0.1)
        self.assertLess(resolved["lad"], BRANCH_QFR_THRESHOLDS["lad"])
        self.assertLess(resolved["lcx"], values["lcx"])
        self.assertLess(resolved["rca"], values["rca"])


if __name__ == "__main__":
    unittest.main()
