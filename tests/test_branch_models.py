import unittest

from model_registry import finalize_branch_panel, load_branch_services, predict_branch_qfr
from ridge_aux import load_ridge_aux_profile


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

    def test_finalize_branch_panel_returns_branch_keys(self):
        feature_map = {name: 0.5 for name in self.services["lad"].feature_names}
        for service in self.services.values():
            for name in service.feature_names:
                feature_map[name] = 0.5
        emitted = finalize_branch_panel(feature_map, self.services, 0.4)
        self.assertEqual(set(emitted.keys()), {"lad", "lcx", "rca"})
        for value in emitted.values():
            self.assertIsInstance(value, float)

    def test_ridge_aux_profile_is_cached(self):
        first = load_ridge_aux_profile()
        second = load_ridge_aux_profile()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
