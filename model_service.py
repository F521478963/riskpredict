import os
import shelve
import warnings
from io import BytesIO

import numpy as np
import pandas as pd


class FeatureShapeError(ValueError):
    pass


class PredictionService:
    def __init__(self, model, scaler, feature_indexes, feature_names=None):
        self.model = model
        self.scaler = scaler
        self.feature_indexes = list(feature_indexes)
        self.feature_names = list(feature_names or [])

    @classmethod
    def from_shelve(cls, model_dat_path, feature_names=None):
        model_base_path = os.path.splitext(model_dat_path)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with shelve.open(model_base_path, flag="r") as save:
                model = save["clf"]
                scaler = save["ss"]
                if "comb_x" in save:
                    feature_indexes = save["comb_x"]
                else:
                    feature_count = getattr(model, "n_features_in_", None)
                    if feature_count is None:
                        raise FeatureShapeError("模型文件缺少 comb_x，且无法从 clf 推断特征数量。")
                    feature_indexes = list(range(feature_count))
                names = feature_names or save.get("feature_names")
                return cls(
                    model=model,
                    scaler=scaler,
                    feature_indexes=feature_indexes,
                    feature_names=names,
                )

    def predict_values_by_names(self, feature_map):
        if not self.feature_names:
            raise FeatureShapeError("当前模型未配置 feature_names，无法按列名预测。")

        missing = [name for name in self.feature_names if name not in feature_map]
        if missing:
            raise FeatureShapeError(
                f"缺少 {len(missing)} 个模型特征，例如: {missing[:3]}"
            )

        values = [feature_map[name] for name in self.feature_names]
        return self.predict_values(values)

    def predict_frame_with_branches(self, frame, branch_specs, branch_services):
        if frame.empty:
            raise FeatureShapeError("上传的 Excel 没有可预测的数据。")

        overall = self.predict_frame(frame)
        for spec in branch_specs:
            service = branch_services[spec["id"]]
            column_name = f"{spec['id'].upper()}_QFR"
            try:
                feature_frame = service._select_feature_frame(frame)
                values = feature_frame.to_numpy(dtype=float)
                scaled_values = service.scaler.transform(values)
                raw_predictions = service.model.predict(scaled_values)
                overall[column_name] = np.abs(raw_predictions)
            except FeatureShapeError:
                overall[column_name] = np.nan

        return overall

    def predict_excel(self, excel_file, branch_specs=None, branch_services=None):
        frame = pd.read_excel(excel_file)
        if branch_specs and branch_services:
            result = self.predict_frame_with_branches(
                frame,
                branch_specs,
                branch_services,
            )
        else:
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

        if self.feature_names:
            missing = [name for name in self.feature_names if name not in frame.columns]
            if not missing:
                return frame[self.feature_names]

        if column_count == expected_count:
            return frame

        required_full_count = max(self.feature_indexes) + 1
        if column_count >= required_full_count:
            return frame.iloc[:, self.feature_indexes]

        raise FeatureShapeError(
            f"上传的 Excel 列数不匹配：当前 {column_count} 列。"
            f"如果上传完整特征表，至少需要 {required_full_count} 列；"
            f"如果只上传模型筛选后的 {expected_count} 个特征，需要 {expected_count} 列"
            f"{'（或包含模型特征列名）' if self.feature_names else ''}。"
        )
