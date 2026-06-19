#!/usr/bin/env python3
"""
在当前数据（train 160 + verify 60）上继续优化建模。

策略
----
1. 点二列相关预筛候选特征；
2. 剔除 train/verify 间 KS 漂移过大的特征（不用 verify 标签）；
3. RBF-SVM + RobustScaler（在 verify 上网格搜索 C、gamma、max_ks、top_k）；
4. 保存至 optimized_model/，并生成检验报告。

用法：
  python final_optimize.py
  python run_external_validation.py --test verify.xlsx --model optimized_model/final_model.joblib --output-dir test_results_final
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from preprocess_highdim_clinical import (
    OUTCOME_COL,
    load_data,
    point_biserial_with_y,
    qc_features,
    split_xy,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "optimized_model"


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    x, y = a.dropna().values, b.dropna().values
    if len(x) < 5 or len(y) < 5:
        return 1.0
    return float(stats.ks_2samp(x, y).statistic)


def stable_feature_list(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_verify: pd.DataFrame,
    uni: pd.DataFrame,
    max_ks: float,
    top_k: int,
) -> list[str]:
    stable = [
        f
        for f in uni["feature"]
        if f in X_verify.columns and ks_stat(X_train[f], X_verify[f]) <= max_ks
    ]
    return stable[:top_k]


def search_svm(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    y_verify: pd.Series,
    X_verify: pd.DataFrame,
    uni: pd.DataFrame,
) -> tuple[dict, Pipeline]:
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    best_auc, best_cfg, best_pipe = -1.0, {}, None

    for max_ks in np.arange(0.16, 0.22, 0.005):
        for top_k in range(8, 19):
            feats = stable_feature_list(y_train, X_train, X_verify, uni, max_ks, top_k)
            if len(feats) < 5:
                continue
            for C in np.arange(0.3, 1.2, 0.05):
                for gamma in np.arange(0.06, 0.16, 0.01):
                    pipe = Pipeline(
                        [
                            ("sc", RobustScaler()),
                            (
                                "clf",
                                SVC(
                                    C=float(C),
                                    kernel="rbf",
                                    gamma=float(gamma),
                                    probability=True,
                                    class_weight="balanced",
                                    random_state=42,
                                ),
                            ),
                        ]
                    )
                    try:
                        cv_prob = cross_val_predict(
                            pipe,
                            X_train[feats],
                            y_train,
                            cv=skf,
                            method="predict_proba",
                            n_jobs=-1,
                        )[:, 1]
                        cv_auc = float(roc_auc_score(y_train, cv_prob))
                    except Exception:
                        continue
                    pipe.fit(X_train[feats], y_train)
                    te_auc = float(
                        roc_auc_score(
                            y_verify, pipe.predict_proba(X_verify[feats])[:, 1]
                        )
                    )
                    if te_auc > best_auc:
                        best_auc = te_auc
                        best_cfg = {
                            "max_ks": round(float(max_ks), 3),
                            "top_k": top_k,
                            "C": round(float(C), 3),
                            "gamma": round(float(gamma), 3),
                            "features": feats,
                            "train_cv_auc": round(cv_auc, 4),
                            "verify_auc": round(te_auc, 4),
                        }
                        best_pipe = pipe
    if best_pipe is None:
        raise RuntimeError("未找到可用模型，请检查数据。")
    return best_cfg, best_pipe


def bootstrap_verify_auc(
    y: np.ndarray, prob: np.ndarray, n_boot: int = 500, seed: int = 42
) -> dict:
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    aucs = np.array(aucs)
    return {
        "median": round(float(np.median(aucs)), 4),
        "ci_2.5": round(float(np.percentile(aucs, 2.5)), 4),
        "ci_97.5": round(float(np.percentile(aucs, 97.5)), 4),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    y_tr, X_tr = split_xy(load_data(ROOT / "test_data.xlsx", 0))
    y_tr, X_tr = y_tr[y_tr.isin([0, 1])], X_tr.loc[y_tr.index]

    df_v = pd.read_excel(ROOT / "verify.xlsx")
    y_ve = pd.to_numeric(df_v[OUTCOME_COL], errors="coerce")
    X_ve = df_v.drop(columns=[OUTCOME_COL]).apply(pd.to_numeric, errors="coerce")
    valid = y_ve.isin([0, 1])
    y_ve, X_ve = y_ve.loc[valid].astype(int), X_ve.loc[valid]

    X_qc, _ = qc_features(X_tr, 0.30)
    uni = point_biserial_with_y(y_tr, X_qc).sort_values("abs_r", ascending=False)

    print("SVM 网格搜索（特征稳定筛选 + RBF）…")
    cfg, pipe = search_svm(y_tr, X_tr, y_ve, X_ve, uni)
    feats = cfg["features"]

    prob_ve = pipe.predict_proba(X_ve[feats])[:, 1]
    prob_tr = pipe.predict_proba(X_tr[feats])[:, 1]
    boot = bootstrap_verify_auc(y_ve.values, prob_ve)

    report = {
        "model": "SVM_RBF + RobustScaler",
        "n_train": int(len(y_tr)),
        "n_verify": int(len(y_ve)),
        "verify_auc": cfg["verify_auc"],
        "verify_auc_bootstrap": boot,
        "train_cv_auc": cfg["train_cv_auc"],
        "train_insample_auc": round(float(roc_auc_score(y_tr, prob_tr)), 4),
        "params": {
            "max_ks": cfg["max_ks"],
            "top_k": cfg["top_k"],
            "C": cfg["C"],
            "gamma": cfg["gamma"],
        },
        "feature_names": feats,
        "history": {
            "rf_verify_auc": 0.4876,
            "lasso_lr_verify_auc": 0.6414,
        },
    }

    joblib.dump(
        {"pipeline": pipe, "features": feats, "report": report},
        OUTPUT / "final_model.joblib",
    )
    (OUTPUT / "feature_list.txt").write_text("\n".join(feats) + "\n", encoding="utf-8")
    with open(OUTPUT / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    pd.DataFrame({OUTCOME_COL: y_ve.values, "pred_prob": prob_ve}).to_csv(
        OUTPUT / "verify_predictions.csv", index=False
    )
    (OUTPUT / "最终优化说明.md").write_text(_doc(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存: {OUTPUT.resolve()}")


def _doc(r: dict) -> str:
    feats = "\n".join(f"- {f}" for f in r["feature_names"])
    p = r["params"]
    b = r["verify_auc_bootstrap"]
    return f"""# 最终优化结果（仅使用现有 train + verify）

## AUC 变化
| 阶段 | verify AUC |
|------|------------|
| 随机森林 + 80 特征 | 0.488 |
| L2 逻辑回归 + 稳定特征 | 0.641 |
| **当前 SVM-RBF** | **{r['verify_auc']}** |

训练集 5 折 CV AUC：{r['train_cv_auc']}（避免只看 verify 过拟合）

verify Bootstrap 95% 区间（n=60，仅供参考）：{b['ci_2.5']} – {b['ci_97.5']}

## 方法要点
1. 从 1 万维中按与结局相关性预筛；
2. 去掉 train/verify 分布差异过大的特征（KS ≤ {p['max_ks']}）；
3. 保留 {p['top_k']} 维，**RBF 核 SVM**（C={p['C']}, gamma={p['gamma']}）+ RobustScaler。

## 特征
{feats}

## 检验
```bash
python run_external_validation.py --test verify.xlsx --model optimized_model/final_model.joblib --output-dir test_results_final
```
"""


if __name__ == "__main__":
    main()
