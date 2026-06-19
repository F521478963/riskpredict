import numpy as np
import pandas as pd
import unittest

from model_service import FeatureShapeError, PredictionService


class FakeScaler:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = np.asarray(values)
        return self.seen + 1


class FakeModel:
    n_features_in_ = 2

    def __init__(self):
        self.seen = None

    def predict(self, values):
        self.seen = np.asarray(values)
        return self.seen.sum(axis=1)


class PredictionServiceTest(unittest.TestCase):
    def test_predict_uses_all_columns_when_uploaded_file_matches_selected_features(self):
        scaler = FakeScaler()
        model = FakeModel()
        service = PredictionService(model=model, scaler=scaler, feature_indexes=[0, 2])
        frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])

        result = service.predict_frame(frame)

        self.assertEqual(scaler.seen.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(model.seen.tolist(), [[2, 3], [4, 5]])
        self.assertEqual(result["Predicted"].tolist(), [5, 9])
        self.assertEqual(result[["a", "b"]].values.tolist(), [[1, 2], [3, 4]])

    def test_predict_selects_model_feature_indexes_when_uploaded_file_has_full_feature_set(self):
        scaler = FakeScaler()
        model = FakeModel()
        service = PredictionService(model=model, scaler=scaler, feature_indexes=[0, 2])
        frame = pd.DataFrame([[10, 20, 30], [40, 50, 60]], columns=["f0", "f1", "f2"])

        result = service.predict_frame(frame)

        self.assertEqual(scaler.seen.tolist(), [[10, 30], [40, 60]])
        self.assertEqual(result["Predicted"].tolist(), [42, 102])

    def test_predict_values_returns_one_prediction_from_manual_inputs(self):
        scaler = FakeScaler()
        model = FakeModel()
        service = PredictionService(model=model, scaler=scaler, feature_indexes=[0, 2])

        result = service.predict_values([1, 3])

        self.assertEqual(scaler.seen.tolist(), [[1, 3]])
        self.assertEqual(model.seen.tolist(), [[2, 4]])
        self.assertEqual(result, 6)

    def test_predict_rejects_files_with_insufficient_feature_columns(self):
        service = PredictionService(model=FakeModel(), scaler=FakeScaler(), feature_indexes=[0, 2])
        frame = pd.DataFrame([[1]], columns=["only_one"])

        with self.assertRaises(FeatureShapeError) as context:
            service.predict_frame(frame)

        self.assertIn("需要 2 列", str(context.exception))

    def test_from_shelve_without_comb_x_uses_sequential_feature_indexes(self):
        import shelve
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "ridge_model")
            with shelve.open(base) as save:
                save["clf"] = FakeModel()
                save["ss"] = FakeScaler()

            service = PredictionService.from_shelve(base + ".dat")
            self.assertEqual(service.feature_indexes, [0, 1])

    def test_predict_selects_columns_by_feature_names(self):
        scaler = FakeScaler()
        model = FakeModel()
        service = PredictionService(
            model=model,
            scaler=scaler,
            feature_indexes=[0, 1],
            feature_names=["alpha", "beta"],
        )
        frame = pd.DataFrame([[10, 20, 30]], columns=["beta", "alpha", "gamma"])

        result = service.predict_frame(frame)

        self.assertEqual(scaler.seen.tolist(), [[20, 10]])
        self.assertEqual(result["Predicted"].tolist(), [32.0])


if __name__ == "__main__":
    unittest.main()
