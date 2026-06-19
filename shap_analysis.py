"""SHAP interpretability analysis for Ridge-RF prediction models."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from model_io import load_model_dataset, load_shelve_model
from model_registry import BRANCH_MODEL_SPECS, OVERALL_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "20260610_most_powerful"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "shap_results"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_file: str
    data_file: str
    feature_names: list[str]
    task: str  # "classification" | "regression"
    title_en: str
    figure_prefix: str


CLASSIFICATION_SPEC = ModelSpec(
    model_id="overall",
    model_file="y_Ridge.dat",
    data_file="y_Ridge.xlsx",
    feature_names=OVERALL_FEATURE_NAMES,
    task="classification",
    title_en="Overall Screening (Ridge-RF)",
    figure_prefix="figure5",
)

REGRESSION_SPECS = [
    ModelSpec(
        model_id=branch["id"],
        model_file=branch["model_file"],
        data_file=branch["model_file"].replace(".dat", ".xlsx"),
        feature_names=branch["feature_names"],
        task="regression",
        title_en=branch["label_en"],
        figure_prefix=f"supplementary_{branch['id']}",
    )
    for branch in BRANCH_MODEL_SPECS
]


def _display_name(feature_name: str) -> str:
    return feature_name


def build_explainer(model, scaler, background_scaled: np.ndarray) -> shap.LinearExplainer:
    return shap.LinearExplainer(model, background_scaled)


def compute_shap_values(
    model,
    scaler,
    features: pd.DataFrame,
    max_background: int = 100,
) -> tuple[np.ndarray, np.ndarray, shap.LinearExplainer]:
    x_raw = features.to_numpy(dtype=float)
    x_scaled = scaler.transform(x_raw)
    background = x_scaled
    if len(background) > max_background:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(background), size=max_background, replace=False)
        background = background[indices]
    explainer = build_explainer(model, scaler, background)
    shap_values = explainer.shap_values(x_scaled)
    return np.asarray(shap_values), x_raw, explainer


def pick_representative_indices(
    y: pd.Series,
    predictions: np.ndarray,
    task: str,
) -> tuple[int, int]:
    y_values = y.to_numpy(dtype=float)
    if task == "classification":
        pos = np.where(y_values == 1)[0]
        neg = np.where(y_values == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("分类任务缺少阳性或阴性样本，无法选择代表性个体。")
        abnormal_idx = int(pos[np.argmax(predictions[pos])])
        control_idx = int(neg[np.argmin(predictions[neg])])
        return abnormal_idx, control_idx

    pos = np.where(y_values <= 0.8)[0]
    neg = np.where(y_values > 0.8)[0]
    if len(pos) == 0:
        abnormal_idx = int(np.argmin(y_values))
    else:
        abnormal_idx = int(pos[np.argmin(y_values[pos])])
    if len(neg) == 0:
        control_idx = int(np.argmax(y_values))
    else:
        control_idx = int(neg[np.argmax(y_values[neg])])
    return abnormal_idx, control_idx


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def save_summary_plot(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    display_names = [_display_name(name) for name in feature_names]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=display_names,
        show=False,
        plot_size=None,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_importance_bar_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    ordered_names = ranking["feature"].tolist()
    ordered_values = ranking["mean_abs_shap"].to_numpy()

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(ordered_names))
    plt.barh(y_pos, ordered_values, color="#4C72B0")
    plt.yticks(y_pos, ordered_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return ranking.reset_index(drop=True)


def _make_explanation(
    shap_values: np.ndarray,
    base_value: float,
    sample_features: np.ndarray,
    feature_names: list[str],
    index: int,
) -> shap.Explanation:
    return shap.Explanation(
        values=shap_values[index],
        base_values=base_value,
        data=sample_features[index],
        feature_names=[_display_name(name) for name in feature_names],
    )


def save_local_plots(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    base_value: float,
    abnormal_idx: int,
    control_idx: int,
    output_dir: Path,
    prefix: str,
) -> dict[str, int]:
    cases = {
        "abnormal": abnormal_idx,
        "control": control_idx,
    }
    saved = {}
    for case_name, index in cases.items():
        explanation = _make_explanation(
            shap_values,
            base_value,
            features,
            feature_names,
            index,
        )
        waterfall_path = output_dir / f"{prefix}_waterfall_{case_name}.png"
        bar_path = output_dir / f"{prefix}_bar_{case_name}.png"

        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"{case_name.title()} case")
        plt.tight_layout()
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 6))
        shap.plots.bar(explanation, max_display=15, show=False)
        plt.title(f"{case_name.title()} case")
        plt.tight_layout()
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()

        saved[case_name] = int(index)
    return saved


def save_composite_figure(
    summary_path: Path,
    importance_path: Path,
    waterfall_abnormal_path: Path,
    waterfall_control_path: Path,
    bar_abnormal_path: Path,
    bar_control_path: Path,
    output_path: Path,
    title: str,
) -> None:
    image_paths = [
        summary_path,
        importance_path,
        waterfall_abnormal_path,
        waterfall_control_path,
        bar_abnormal_path,
        bar_control_path,
    ]
    labels = ["A", "B", "C", "D", "E", "F"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    for axis, image_path, label in zip(axes.flat, image_paths, labels):
        axis.imshow(plt.imread(image_path))
        axis.set_title(label, loc="left", fontweight="bold")
        axis.axis("off")
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def analyze_model(
    spec: ModelSpec,
    output_dir: Path,
    max_background: int = 100,
) -> dict:
    model_path = MODEL_DIR / spec.model_file
    data_path = MODEL_DIR / spec.data_file
    model, scaler = load_shelve_model(model_path)
    features, labels, data_columns = load_model_dataset(data_path)

    missing = [name for name in spec.feature_names if name not in features.columns]
    if missing:
        raise KeyError(f"{spec.model_id} 缺少特征列: {missing[:5]}")

    aligned = features[spec.feature_names]
    shap_values, raw_features, explainer = compute_shap_values(
        model,
        scaler,
        aligned,
        max_background=max_background,
    )
    predictions = np.asarray(model.predict(scaler.transform(raw_features))).reshape(-1)
    abnormal_idx, control_idx = pick_representative_indices(
        labels,
        predictions,
        spec.task,
    )

    model_output_dir = output_dir / spec.model_id
    model_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = model_output_dir / f"{spec.figure_prefix}_summary.png"
    importance_path = model_output_dir / f"{spec.figure_prefix}_importance.png"
    ranking = save_importance_bar_plot(
        shap_values,
        spec.feature_names,
        importance_path,
        title=f"SHAP Feature Importance - {spec.title_en}",
    )
    save_summary_plot(
        shap_values,
        raw_features,
        spec.feature_names,
        summary_path,
        title=f"SHAP Summary Plot - {spec.title_en}",
    )
    case_indices = save_local_plots(
        shap_values,
        raw_features,
        spec.feature_names,
        float(explainer.expected_value),
        abnormal_idx,
        control_idx,
        model_output_dir,
        spec.figure_prefix,
    )

    composite_path = model_output_dir / f"{spec.figure_prefix}_composite.png"
    save_composite_figure(
        summary_path,
        importance_path,
        model_output_dir / f"{spec.figure_prefix}_waterfall_abnormal.png",
        model_output_dir / f"{spec.figure_prefix}_waterfall_control.png",
        model_output_dir / f"{spec.figure_prefix}_bar_abnormal.png",
        model_output_dir / f"{spec.figure_prefix}_bar_control.png",
        composite_path,
        title=f"SHAP Interpretability - {spec.title_en}",
    )

    shap_values_path = model_output_dir / f"{spec.figure_prefix}_shap_values.csv"
    shap_frame = pd.DataFrame(shap_values, columns=spec.feature_names)
    shap_frame.insert(0, "sample_index", np.arange(len(shap_frame)))
    shap_frame.to_csv(shap_values_path, index=False)
    ranking.to_csv(model_output_dir / f"{spec.figure_prefix}_feature_ranking.csv", index=False)

    report = {
        "model_id": spec.model_id,
        "task": spec.task,
        "title_en": spec.title_en,
        "n_samples": int(len(aligned)),
        "n_features": len(spec.feature_names),
        "expected_value": float(explainer.expected_value),
        "representative_cases": {
            "abnormal": {
                "index": case_indices["abnormal"],
                "label": float(labels.iloc[case_indices["abnormal"]]),
                "prediction": float(predictions[case_indices["abnormal"]]),
            },
            "control": {
                "index": case_indices["control"],
                "label": float(labels.iloc[case_indices["control"]]),
                "prediction": float(predictions[case_indices["control"]]),
            },
        },
        "outputs": {
            "summary_plot": str(summary_path),
            "importance_plot": str(importance_path),
            "composite_plot": str(composite_path),
            "shap_values_csv": str(shap_values_path),
        },
        "top_features": ranking.head(5).to_dict(orient="records"),
    }
    with (model_output_dir / f"{spec.figure_prefix}_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def run_all(output_dir: Path | None = None, max_background: int = 100) -> dict:
    _configure_matplotlib()
    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        "classification": analyze_model(CLASSIFICATION_SPEC, target_dir, max_background),
        "regression": [],
    }
    for spec in REGRESSION_SPECS:
        reports["regression"].append(analyze_model(spec, target_dir, max_background))

    summary_path = target_dir / "shap_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    return reports
