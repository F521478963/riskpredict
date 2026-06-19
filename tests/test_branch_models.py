import unittest
from unittest.mock import patch

from model_registry import finalize_branch_panel, load_branch_services, predict_branch_qfr
from output_align import PanelSink
from ridge_aux import RidgeAuxProfile, load_ridge_aux_profile


class BranchModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.services = load_branch_services()
        cls.profile = load_ridge_aux_profile()

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

    def _emit_with_readings(self, readings, reference_score):
        sink = PanelSink()
        sink._profile = self.profile
        with patch.object(sink, "collect", return_value=readings):
            return sink.emit({}, {}, reference_score)

    def test_panel_emit_reconciles_below_floor_readings(self):
        readings = {"lad": 0.65, "lcx": 0.72, "rca": 0.58}
        emitted = self._emit_with_readings(readings, 0.35)
        for value in emitted.values():
            self.assertGreaterEqual(value, self.profile.branch_floor)

    def test_panel_emit_preserves_aligned_readings(self):
        readings = {"lad": 0.85, "lcx": 0.82, "rca": 0.91}
        emitted = self._emit_with_readings(readings, 0.35)
        self.assertEqual(emitted, readings)

    def test_panel_emit_skips_when_reference_above_gate(self):
        readings = {"lad": 0.65, "lcx": 0.72, "rca": 0.58}
        emitted = self._emit_with_readings(readings, 0.65)
        self.assertEqual(emitted, readings)


if __name__ == "__main__":
    unittest.main()
