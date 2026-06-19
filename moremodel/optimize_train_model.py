#!/usr/bin/env python3
"""
训练集拟合优化：提升「训练集训练 + 训练集评估」的样本内 AUC。

说明
----
- 目标：在 160 例训练数据上尽量提高拟合判别能力（样本内 AUC）。
- 相对原 LASSO+30 特征，改用更多特征 + 非线性模型（随机森林 / 梯度提升 / 软投票集成）。
- 样本内 AUC 可达 1.0 时，往往表示强过拟合；真正测试集仍需独立验证。

输出目录 trained_model/：
  best_model.joblib      — 标准化器 + 分类器 + 元数据
  feature_list.txt       — 入选特征名（测试集必须对齐）
  model_report.json      — AUC 对比与参数
  train_predictions.csv  — 训练集预测概率
  训练集优化说明.md

用法：
  python optimize_train_model.py
  python optimize_train_model.py --top-k 80 --model rf
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess_highdim_clinical import (
    OUTCOME_COL,
    dedupe_by_correlation,
    load_data,
    prefilter_by_outcome_corr,
    qc_features,
    select_by_fdr_and_mi,
    split_xy,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "test_data.xlsx"
OUTPUT_DIR = ROOT / "trained_model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练集 AUC 优化")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--prefilter-k", type=int, default=3000)
    p.add_argument("--corr-threshold", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=80)
    p.add_argument("--fdr-alpha", type=float, default=0.10)
    p.add_argument(
        "--model",
        choices=["auto", "rf", "gbm", "ensemble", "lasso"],
        default="auto",
        help="auto=在 rf/gbm/ensemble 中选训练 AUC 最高者",
    )
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def select_feature_matrix(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    prefilter_k: int,
    corr_threshold: float,
    top_k: int,
    fdr_alpha: float,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    X_qc, _ = qc_features(X, 0.30)
    X_pre, uni = prefilter_by_outcome_corr(y, X_qc, prefilter_k)
    X_dedup, _ = dedupe_by_correlation(X_pre, uni, corr_threshold)
    feats, _ = select_by_fdr_and_mi(
        y, X_dedup, uni, top_k, fdr_alpha, random_state
    )
    return X_dedup[feats], feats


def baseline_lasso_auc(
    y: pd.Series, X: pd.DataFrame, random_state: int
) -> float:
    Xf, _ = select_feature_matrix(
        y, X, prefilter_k=2000, corr_threshold=0.95, top_k=30, fdr_alpha=0.05, random_state=random_state
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    l1_ratios=(1.0,),
                    solver="saga",
                    Cs=np.logspace(-3, 1, 15),
                    cv=5,
                    scoring="roc_auc",
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=8000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(Xf, y)
    return float(roc_auc_score(y, pipe.predict_proba(Xf)[:, 1]))


def make_rf(rs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced",
        random_state=rs,
        n_jobs=-1,
    )


def make_gbm(rs: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        random_state=rs,
    )


def make_ensemble(rs: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("rf", make_rf(rs)),
            ("gbm", make_gbm(rs)),
        ],
        voting="soft",
        n_jobs=-1,
    )


def fit_and_auc(model, X_scaled: np.ndarray, y: pd.Series) -> float:
    model.fit(X_scaled, y)
    prob = model.predict_proba(X_scaled)[:, 1]
    return float(roc_auc_score(y, prob))


def pick_model(
    model_kind: str, X_scaled: np.ndarray, y: pd.Series, random_state: int
) -> tuple[object, str, float]:
    candidates: dict[str, object] = {
        "rf": make_rf(random_state),
        "gbm": make_gbm(random_state),
        "ensemble": make_ensemble(random_state),
    }
    if model_kind != "auto":
        m = candidates[model_kind]
        return m, model_kind, fit_and_auc(m, X_scaled, y)

    best_name, best_auc, best_model = "", -1.0, None
    for name, m in candidates.items():
        auc = fit_and_auc(m, X_scaled, y)
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, m
    assert best_model is not None
    return best_model, best_name, best_auc


def write_readme(
    path: Path,
    *,
    auc_base: float,
    auc_new: float,
    model_name: str,
    top_k: int,
    n_features: int,
) -> None:
    path.write_text(
        f"""# 训练集优化说明

## 结果对比（同一批 160 例上训练并评估）

| 方案 | 样本内 AUC |
|------|------------|
| 原方案：LASSO + 30 特征 | **{auc_base:.4f}** |
| 优化后：{model_name} + {top_k} 特征 | **{auc_new:.4f}** |

## 做了哪些改动？
1. 特征数由 30 增至 **{n_features}**（预筛 3000 → 去共线 → FDR/互信息入选）。
2. 分类器由稀疏 LASSO 改为 **{model_name}**（可拟合非线性边界）。
3. 共线性阈值放宽至 0.92，FDR α=0.10，保留更多与结局相关的纹理/光谱指标。

## 重要提示
- 样本内 AUC 很高（甚至 1.0）只说明模型**记住了训练集**，不等于能预测新病人。
- 你提供独立测试集后，请运行：

```bash
python predict_external_test.py --test 你的测试集.xlsx
```

- 测试集**不要**重新做特征筛选；必须使用本目录 `feature_list.txt` 中的同一批特征名。

## 文件
- `best_model.joblib`：预测用
- `feature_list.txt`：特征名列表
- `train_predictions.csv`：训练集每例预测概率
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, 0)
    y, X = split_xy(df)
    valid = y.isin([0, 1])
    y, X = y.loc[valid], X.loc[valid]

    print("计算基线 LASSO（30 特征）样本内 AUC …")
    auc_baseline = baseline_lasso_auc(y, X, args.random_state)
    print(f"  基线: {auc_baseline:.4f}")

    print(f"特征筛选 top_k={args.top_k} …")
    X_sel, feats = select_feature_matrix(
        y,
        X,
        prefilter_k=args.prefilter_k,
        corr_threshold=args.corr_threshold,
        top_k=args.top_k,
        fdr_alpha=args.fdr_alpha,
        random_state=args.random_state,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    if args.model == "lasso":
        pipe = Pipeline(
            [
                (
                    "clf",
                    LogisticRegressionCV(
                        l1_ratios=(1.0,),
                        solver="saga",
                        Cs=np.logspace(-4, 2, 25),
                        cv=5,
                        scoring="roc_auc",
                        class_weight="balanced",
                        random_state=args.random_state,
                        max_iter=10000,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_scaled, y)
        model = pipe
        model_name = "lasso_tuned"
        prob = pipe.predict_proba(X_scaled)[:, 1]
        auc_new = float(roc_auc_score(y, prob))
    else:
        model, model_name, auc_new = pick_model(
            args.model, X_scaled, y, args.random_state
        )
        prob = model.predict_proba(X_scaled)[:, 1]

    print(f"优化后 ({model_name}): 样本内 AUC = {auc_new:.4f}")

    meta = {
        "input": str(args.input.resolve()),
        "n_samples": int(len(y)),
        "outcome_class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "baseline_lasso_train_auc": auc_baseline,
        "optimized_train_auc": auc_new,
        "improvement": round(auc_new - auc_baseline, 4),
        "model_type": model_name,
        "prefilter_k": args.prefilter_k,
        "corr_threshold": args.corr_threshold,
        "top_k": args.top_k,
        "fdr_alpha": args.fdr_alpha,
        "n_selected_features": len(feats),
        "feature_names": feats,
    }

    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "feature_names": feats,
            "meta": meta,
            "selection_params": {
                "prefilter_k": args.prefilter_k,
                "corr_threshold": args.corr_threshold,
                "top_k": args.top_k,
                "fdr_alpha": args.fdr_alpha,
            },
        },
        args.output_dir / "best_model.joblib",
    )

    (args.output_dir / "feature_list.txt").write_text(
        "\n".join(feats) + "\n", encoding="utf-8"
    )
    with open(args.output_dir / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    pd.DataFrame({OUTCOME_COL: y.values, "pred_prob": prob}).to_csv(
        args.output_dir / "train_predictions.csv", index=False
    )

    write_readme(
        args.output_dir / "训练集优化说明.md",
        auc_base=auc_baseline,
        auc_new=auc_new,
        model_name=model_name,
        top_k=args.top_k,
        n_features=len(feats),
    )

    print(f"\n已保存至: {args.output_dir.resolve()}")
    print(f"AUC 提升: {auc_baseline:.4f} → {auc_new:.4f} (+{auc_new - auc_baseline:.4f})")


if __name__ == "__main__":
    main()
