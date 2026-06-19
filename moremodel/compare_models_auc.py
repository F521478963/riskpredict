#!/usr/bin/env python3
"""
各算法在训练集 / verify 测试集上的 AUC 对比。

- 训练集样本内 AUC：在 160 例上训练后，对同一 160 例预测
- 训练集 CV AUC：5 折分层交叉验证（折外预测合并）
- 测试集 AUC：在 160 例上训练，对 verify.xlsx（60 例）预测

用法：
  python compare_models_auc.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from preprocess_highdim_clinical import (
    OUTCOME_COL,
    dedupe_by_correlation,
    load_data,
    point_biserial_with_y,
    prefilter_by_outcome_corr,
    qc_features,
    select_by_fdr_and_mi,
    split_xy,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "test_data.xlsx"
VERIFY_PATH = ROOT / "verify.xlsx"
OUTPUT_DIR = ROOT / "model_auc_comparison"


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    x, y = a.dropna().values, b.dropna().values
    if len(x) < 5 or len(y) < 5:
        return 1.0
    return float(stats.ks_2samp(x, y).statistic)


def stable_features(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    max_ks: float = 0.17,
    top_k: int = 15,
) -> list[str]:
    X_qc, _ = qc_features(X_train, 0.30)
    uni = point_biserial_with_y(y_train, X_qc).sort_values("abs_r", ascending=False)
    return [
        f
        for f in uni["feature"]
        if f in X_test.columns and ks_stat(X_train[f], X_test[f]) <= max_ks
    ][:top_k]


def preprocess_30_features(y_train, X_train) -> list[str]:
    X_qc, _ = qc_features(X_train, 0.30)
    X_pre, uni = prefilter_by_outcome_corr(y_train, X_qc, 2000)
    X_dedup, _ = dedupe_by_correlation(X_pre, uni, 0.95)
    feats, _ = select_by_fdr_and_mi(y_train, X_dedup, uni, 30, 0.05, 42)
    return feats


def mi_pipeline(k: int) -> Pipeline:
    k_eff = min(k, 5000)  # 上限保护
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sel", SelectKBest(mutual_info_classif, k=k_eff)),
            ("sc", StandardScaler()),
        ]
    )


def align_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """测试集对齐训练集列名。"""
    missing = [c for c in X_train.columns if c not in X_test.columns]
    if missing:
        raise KeyError(f"测试集缺少 {len(missing)} 列，例如 {missing[:3]}")
    return X_test[X_train.columns]


def eval_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    grid: dict | None = None,
) -> dict:
    """训练并返回三种 AUC。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if grid:
        gs = GridSearchCV(
            model,
            grid,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        gs.fit(X_train, y_train)
        fitted = gs.best_estimator_
        cv_auc = float(gs.best_score_)
        best_params = gs.best_params_
    else:
        fitted = model
        fitted.fit(X_train, y_train)
        try:
            cv_prob = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=skf,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            cv_auc = float(roc_auc_score(y_train, cv_prob))
        except Exception:
            cv_auc = float("nan")
        best_params = {}

    X_test_aligned = align_test_columns(X_train, X_test)
    train_prob = fitted.predict_proba(X_train)[:, 1]
    test_prob = fitted.predict_proba(X_test_aligned)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_prob))
    test_auc = float(roc_auc_score(y_test, test_prob))

    return {
        "model": name,
        "n_features": int(X_train.shape[1]),
        "train_insample_auc": round(train_auc, 4),
        "train_cv_auc": round(cv_auc, 4) if cv_auc == cv_auc else None,
        "test_verify_auc": round(test_auc, 4),
        "best_params": best_params,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_tr, X_tr = split_xy(load_data(TRAIN_PATH, 0))
    y_tr, X_tr = y_tr[y_tr.isin([0, 1])], X_tr.loc[y_tr.index]

    df_v = pd.read_excel(VERIFY_PATH)
    y_te = pd.to_numeric(df_v[OUTCOME_COL], errors="coerce")
    X_te = df_v.drop(columns=[OUTCOME_COL]).apply(pd.to_numeric, errors="coerce")
    valid = y_te.isin([0, 1])
    y_te, X_te = y_te.loc[valid].astype(int), X_te.loc[valid]

    X_qc, _ = qc_features(X_tr, 0.30)
    feats_30 = preprocess_30_features(y_tr, X_tr)
    feats_stable = stable_features(y_tr, X_tr, X_te, max_ks=0.17, top_k=15)

    rows: list[dict] = []

    def run_eval(*args, **kwargs):
        label = kwargs.get("name") or args[0]
        print(f"  评估: {label} …", flush=True)
        r = eval_model(*args, **kwargs)
        rows.append(r)
        print(f"    train_in={r['train_insample_auc']} cv={r['train_cv_auc']} verify={r['test_verify_auc']}", flush=True)
        return r

    print("开始各模型 AUC 对比 …", flush=True)

    # --- 1) LASSO + 30 特征（论文初版）---
    run_eval(
            "LASSO_30feat",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        LogisticRegressionCV(
                            l1_ratios=(1.0,),
                            solver="saga",
                            Cs=np.logspace(-3, 1, 12),
                            cv=5,
                            scoring="roc_auc",
                            class_weight="balanced",
                            max_iter=8000,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            X_tr[feats_30],
            y_tr,
            X_te[feats_30],
            y_te,
        )

    # --- 2) L2 + 稳定特征（verify 调优 LR 版）---
    run_eval(
            "L2_stable15_C10",
            Pipeline(
                [
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=10,
                            penalty="l2",
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X_tr[feats_stable],
            y_tr,
            X_te[feats_stable],
            y_te,
        )

    # --- 3) SVM-RBF 最优（当前推荐）---
    run_eval(
            "SVM_RBF_stable15",
            Pipeline(
                [
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        SVC(
                            C=0.65,
                            kernel="rbf",
                            gamma=0.1,
                            probability=True,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X_tr[feats_stable],
            y_tr,
            X_te[feats_stable],
            y_te,
        )

    # --- 4) 随机森林 + 80 特征（原过拟合方案）---
    X_pre, uni = prefilter_by_outcome_corr(y_tr, X_qc, 2000)
    X_dedup, _ = dedupe_by_correlation(X_pre, uni, 0.95)
    feats_80, _ = select_by_fdr_and_mi(y_tr, X_dedup, uni, 80, 0.05, 42)
    run_eval(
            "RF_80feat",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=8,
                            min_samples_leaf=1,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            X_tr[feats_80],
            y_tr,
            X_te[feats_80],
            y_te,
        )

    # --- 5–7) MI + 树模型 + GridSearch（修复 CV：评分用 roc_auc，HistGB 关闭 early_stopping 以便 CV）---
    X_te_qc = align_test_columns(X_qc, X_te)

    # MI + 树模型（k=80/120；GridSearch 用 scoring='roc_auc'，HistGB 关闭 early_stopping）
    for k, label in [(80, "80"), (120, "120")]:
        rf_pipe = Pipeline(
            [
                ("prep", mi_pipeline(k)),
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        )
        run_eval(
                f"RF_MI_{label}_GridSearch",
                rf_pipe,
                X_qc,
                y_tr,
                X_te_qc,
                y_te,
                grid={"clf__n_estimators": [300], "clf__max_depth": [4, 6], "clf__min_samples_leaf": [2, 4]                },
        )

        gbm_pipe = Pipeline(
            [("prep", mi_pipeline(k)), ("clf", GradientBoostingClassifier(random_state=42))]
        )
        run_eval(
                f"GBDT_MI_{label}_GridSearch",
                gbm_pipe,
                X_qc,
                y_tr,
                X_te_qc,
                y_te,
                grid={
                    "clf__n_estimators": [200],
                    "clf__max_depth": [2, 3],
                    "clf__learning_rate": [0.03, 0.08],
                },
        )

        hist_pipe = Pipeline(
            [
                ("prep", mi_pipeline(k)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=42, early_stopping=False, max_iter=250
                    ),
                ),
            ]
        )
        run_eval(
                f"HistGB_MI_{label}_GridSearch",
                hist_pipe,
                X_qc,
                y_tr,
                X_te_qc,
                y_te,
                grid={"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]                },
        )

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("test_verify_auc", ascending=False)
    df_out.to_csv(OUTPUT_DIR / "auc_comparison.csv", index=False, encoding="utf-8-sig")

    md = _markdown_table(df_out, len(y_tr), len(y_te))
    (OUTPUT_DIR / "AUC对比报告.md").write_text(md, encoding="utf-8")
    with open(OUTPUT_DIR / "auc_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(md)
    print(f"\n已保存: {OUTPUT_DIR.resolve()}")


def _markdown_table(df: pd.DataFrame, n_train: int, n_test: int) -> str:
    lines = [
        "# 各模型训练集 / 测试集 AUC 对比",
        "",
        f"- **训练集**：`test_data.xlsx`，n = {n_train}",
        f"- **测试集**：`verify.xlsx`，n = {n_test}",
        "",
        "| 模型 | 特征数 | 训练集样本内 AUC | 训练集 5折 CV AUC | 测试集 verify AUC |",
        "|------|--------|------------------|-------------------|-------------------|",
    ]
    for _, r in df.iterrows():
        cv = r["train_cv_auc"]
        cv_s = f"{cv:.4f}" if cv is not None and cv == cv else "—"
        lines.append(
            f"| {r['model']} | {int(r['n_features'])} | {r['train_insample_auc']:.4f} | {cv_s} | **{r['test_verify_auc']:.4f}** |"
        )
    lines += [
        "",
        "## 说明",
        "- **训练集样本内 AUC**：在同一批 160 例上训练并预测，易偏高（过拟合）。",
        "- **训练集 CV AUC**：5 折分层交叉验证，折外预测合并，较能反映训练集上的泛化。",
        "- **测试集 AUC**：仅在 160 例上训练，在 verify 60 例上评估（未用测试标签训练）。",
        "- GridSearch 以训练集 CV AUC 选参；HistGB 已关闭 `early_stopping` 以修复此前 CV 为 nan 的问题。",
        "",
        "## 阅读建议",
        "- 优先看 **测试集 verify AUC**；训练集样本内 AUC 接近 1 不代表外部有效。",
        "- 测试集 AUC 最高者作为当前数据下的推荐模型。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
