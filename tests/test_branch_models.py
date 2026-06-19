import unittest

from model_registry import load_branch_services, predict_branch_qfr


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


if __name__ == "__main__":
    unittest.main()
