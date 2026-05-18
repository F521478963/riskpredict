import os
import shelve
import warnings
from io import BytesIO

import numpy as np
import pandas as pd


class FeatureShapeError(ValueError):
    pass


class PredictionService:
    def __init__(self, model, scaler, feature_indexes):
        self.model = model
        self.scaler = scaler
        self.feature_indexes = list(feature_indexes)

    @classmethod
    def from_shelve(cls, model_dat_path):
        model_base_path = os.path.splitext(model_dat_path)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with shelve.open(model_base_path, flag="r") as save:
                return cls(
                    model=save["clf"],
                    scaler=save["ss"],
                    feature_indexes=save["comb_x"],
                )

    def predict_excel(self, excel_file):
        frame = pd.read_excel(excel_file)
        result = self.predict_frame(frame)
        output = BytesIO()
        result.to_excel(output, index=False)
        output.seek(0)
        return output

    def predict_values(self, values):
        if len(values) != len(self.feature_indexes):
            raise FeatureShapeError(
                f"指标数量不匹配：需要 {len(self.feature_indexes)} 个，当前 {len(values)} 个。"
            )

        frame = pd.DataFrame([values])
        result = self.predict_frame(frame)
        return float(result["Predicted"].iloc[0])

    def predict_frame(self, frame):
        if frame.empty:
            raise FeatureShapeError("上传的 Excel 没有可预测的数据。")

        feature_frame = self._select_feature_frame(frame)
        values = feature_frame.to_numpy(dtype=float)
        scaled_values = self.scaler.transform(values)
        predictions = np.asarray(self.model.predict(scaled_values)).reshape(-1, 1)

        result = frame.copy()
        result.insert(0, "Predicted", predictions[:, 0])
        return result

    def _select_feature_frame(self, frame):
        expected_count = len(self.feature_indexes)
        column_count = len(frame.columns)
        required_full_count = max(self.feature_indexes) + 1

        if column_count == expected_count:
            return frame

        if column_count >= required_full_count:
            return frame.iloc[:, self.feature_indexes]

        raise FeatureShapeError(
            f"上传的 Excel 列数不匹配：当前 {column_count} 列。"
            f"如果上传完整特征表，至少需要 {required_full_count} 列；"
            f"如果只上传模型筛选后的特征，需要 {expected_count} 列。"
        )
