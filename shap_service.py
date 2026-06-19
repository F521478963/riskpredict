"""Runtime SHAP explanations for web and batch prediction."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shap

from model_registry import (
    BRANCH_MODEL_SPECS,
    FEATURE_SPECS,
    MODEL_DIR,
    OVERALL_FEATURE_NAMES,
)
from shap_analysis import load_model_dataset, load_shelve_model

MAX_BACKGROUND = 100
BACKGROUND_SEED = 42

FEATURE_LABELS = {
    column: {"label_zh": label_zh, "label_en": label_en}
    for column, label_zh, label_en, _group in FEATURE_SPECS
}


@dataclass(frozen=True)
class RuntimeModelSpec:
    model_id: str
    model_file: str
    data_file: str
    feature_names: list[str]
    task: str
    title_zh: str
    title_en: str


RUNTIME_MODEL_SPECS = [
    RuntimeModelSpec(
        model_id="overall",
        model_file="y_Ridge.dat",
        data_file="y_Ridge.xlsx",
        feature_names=OVERALL_FEATURE_NAMES,
        task="classification",
        title_zh="整体筛查（Ridge-RF）",
        title_en="Overall Screening (Ridge-RF)",
    ),
    *[
        RuntimeModelSpec(
            model_id=branch["id"],
            model_file=branch["model_file"],
            data_file=branch["model_file"].replace(".dat", ".xlsx"),
            feature_names=branch["feature_names"],
            task="regression",
            title_zh=branch["label_zh"],
            title_en=branch["label_en"],
        )
        for branch in BRANCH_MODEL_SPECS
    ],
]


@dataclass
class ShapFeatureContribution:
    feature: str
    label_zh: str
    label_en: str
    value: float
    shap_value: float
    abs_shap: float
    direction: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShapExplanationResult:
    model_id: str
    title_zh: str
    title_en: str
    task: str
    base_value: float
    prediction: float
    feature_count: int
    max_abs_shap: float
    features: list[ShapFeatureContribution]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["features"] = [feature.to_dict() for feature in self.features]
        return payload


class _ModelExplainer:
    def __init__(self, spec: RuntimeModelSpec, model_dir: Path | None = None):
        self.spec = spec
        base = Path(model_dir or MODEL_DIR)
        model, scaler = load_shelve_model(base / spec.model_file)
        features, _labels, _columns = load_model_dataset(base / spec.data_file)
        aligned = features[spec.feature_names].to_numpy(dtype=float)
        scaled = scaler.transform(aligned)
        background = scaled
        if len(background) > MAX_BACKGROUND:
            rng = np.random.default_rng(BACKGROUND_SEED)
            indices = rng.choice(len(background), size=MAX_BACKGROUND, replace=False)
            background = background[indices]

        self.model = model
        self.scaler = scaler
        self.explainer = shap.LinearExplainer(model, background)

    def explain(self, feature_map: dict[str, float]) -> ShapExplanationResult:
        missing = [name for name in self.spec.feature_names if name not in feature_map]
        if missing:
            raise KeyError(f"缺少 SHAP 所需特征: {missing[:3]}")

        raw = np.array(
            [float(feature_map[name]) for name in self.spec.feature_names],
            dtype=float,
        ).reshape(1, -1)
        scaled = self.scaler.transform(raw)
        shap_values = np.asarray(self.explainer.shap_values(scaled)).reshape(-1)
        prediction = float(np.asarray(self.model.predict(scaled)).reshape(-1)[0])

        contributions: list[ShapFeatureContribution] = []
        for index, feature_name in enumerate(self.spec.feature_names):
            labels = FEATURE_LABELS.get(
                feature_name,
                {"label_zh": feature_name, "label_en": feature_name},
            )
            shap_value = float(shap_values[index])
            contributions.append(
                ShapFeatureContribution(
                    feature=feature_name,
                    label_zh=labels["label_zh"],
                    label_en=labels["label_en"],
                    value=float(raw[0, index]),
                    shap_value=shap_value,
                    abs_shap=abs(shap_value),
                    direction="positive" if shap_value >= 0 else "negative",
                )
            )

        contributions.sort(key=lambda item: item.abs_shap, reverse=True)
        max_abs = max((item.abs_shap for item in contributions), default=0.0)
        return ShapExplanationResult(
            model_id=self.spec.model_id,
            title_zh=self.spec.title_zh,
            title_en=self.spec.title_en,
            task=self.spec.task,
            base_value=float(self.explainer.expected_value),
            prediction=prediction,
            feature_count=len(contributions),
            max_abs_shap=max_abs,
            features=contributions,
        )


class ShapRuntime:
    def __init__(self, model_dir: Path | None = None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainers = {
                spec.model_id: _ModelExplainer(spec, model_dir=model_dir)
                for spec in RUNTIME_MODEL_SPECS
            }

    def explain_all(self, feature_map: dict[str, float]) -> list[ShapExplanationResult]:
        return [
            self._explainers[spec.model_id].explain(feature_map)
            for spec in RUNTIME_MODEL_SPECS
        ]

    def explain_model(self, model_id: str, feature_map: dict[str, float]) -> ShapExplanationResult:
        if model_id not in self._explainers:
            raise KeyError(f"未知 SHAP 模型: {model_id}")
        return self._explainers[model_id].explain(feature_map)


def build_shap_runtime(model_dir: Path | None = None) -> ShapRuntime:
    return ShapRuntime(model_dir=model_dir)
