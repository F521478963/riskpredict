#!/usr/bin/env python3
"""
博士论文分析主入口

步骤：
  preprocess — 仅数据预处理（默认，产出 preprocessed/）
  model      — 5 折交叉验证 + LASSO（折内重新筛选特征，产出 thesis_results/）
  all        — 先 preprocess 再 model

用法：
  python run_thesis_pipeline.py
  python run_thesis_pipeline.py --step all
  python run_thesis_pipeline.py --step model   # 已有预处理结果时
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess_highdim_clinical import (
    OUTCOME_COL,
    dedupe_by_correlation,
    load_data,
    prefilter_by_outcome_corr,
    qc_features,
    run_preprocess,
    select_by_fdr_and_mi,
    split_xy,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "test_data.xlsx"
PREPROCESS_DIR = ROOT / "preprocessed"
THESIS_DIR = ROOT / "thesis_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="博士论文：预处理 + 预测模型")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--step", choices=["preprocess", "model", "all"], default="preprocess")
    p.add_argument("--prefilter-k", type=int, default=2000)
    p.add_argument("--corr-threshold", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def select_features_on_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    missing_threshold: float,
    prefilter_k: int,
    corr_threshold: float,
    top_k: int,
    fdr_alpha: float,
    random_state: int,
) -> list[str]:
    """仅在训练折上完成特征筛选（验证折不可见）。"""
    X_qc, _ = qc_features(X_train, missing_threshold)
    X_pre, uni_scores = prefilter_by_outcome_corr(y_train, X_qc, prefilter_k)
    X_dedup, _ = dedupe_by_correlation(X_pre, uni_scores, corr_threshold)
    final_features, _ = select_by_fdr_and_mi(
        y_train, X_dedup, uni_scores, top_k, fdr_alpha, random_state
    )
    return final_features


def run_nested_cv_model(
    input_path: Path,
    output_dir: Path,
    *,
    prefilter_k: int,
    corr_threshold: float,
    top_k: int,
    cv_folds: int,
    random_state: int,
) -> dict:
    """分层 K 折：每折在训练集筛选特征并拟合 LASSO，在验证折算 AUC。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path, 0)
    y, X = split_xy(df)
    valid = y.isin([0, 1])
    y, X = y.loc[valid], X.loc[valid]

    class_counts = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    oof_prob = np.zeros(len(y), dtype=float)
    fold_details: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        feats = select_features_on_train(
            X_train,
            y_train,
            missing_threshold=0.30,
            prefilter_k=prefilter_k,
            corr_threshold=corr_threshold,
            top_k=top_k,
            fdr_alpha=0.05,
            random_state=random_state,
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
                        cv=min(3, cv_folds),
                        scoring="roc_auc",
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=8000,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_train[feats], y_train)
        prob = pipe.predict_proba(X_val[feats])[:, 1]
        oof_prob[val_idx] = prob

        coef = pipe.named_steps["clf"].coef_.ravel()
        fold_auc = float(roc_auc_score(y_val, prob)) if y_val.nunique() > 1 else float("nan")
        fold_details.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_features_selected": len(feats),
                "n_lasso_nonzero_coef": int((coef != 0).sum()),
                "fold_auc": round(fold_auc, 4),
            }
        )

    cv_auc = float(roc_auc_score(y, oof_prob))

    # 探索性：全数据筛选 + LASSO（仅展示变量方向，不代替 CV 性能）
    final_feats = select_features_on_train(
        X,
        y,
        missing_threshold=0.30,
        prefilter_k=prefilter_k,
        corr_threshold=corr_threshold,
        top_k=top_k,
        fdr_alpha=0.05,
        random_state=random_state,
    )
    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    l1_ratios=(1.0,),
                    solver="saga",
                    Cs=np.logspace(-3, 1, 15),
                    cv=min(3, cv_folds),
                    scoring="roc_auc",
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=8000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_pipe.fit(X[final_feats], y)
    coef = final_pipe.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame(
        {"feature": final_feats, "coef": coef, "abs_coef": np.abs(coef)}
    ).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(output_dir / "final_model_lasso_coefficients.csv", index=False)

    pd.DataFrame({OUTCOME_COL: y.values, "cv_pred_prob": oof_prob}).to_csv(
        output_dir / "cv_oof_predictions.csv", index=False
    )

    report = {
        "n_samples": int(len(y)),
        "n_features_raw": int(X.shape[1]),
        "outcome_class_counts": class_counts,
        "parameters": {
            "prefilter_k": prefilter_k,
            "corr_threshold": corr_threshold,
            "top_k": top_k,
            "cv_folds": cv_folds,
        },
        "cv_auc": round(cv_auc, 4),
        "cv_folds_detail": fold_details,
        "final_model_n_selected": len(final_feats),
        "final_model_n_nonzero_coef": int((coef != 0).sum()),
        "final_selected_features": final_feats,
    }
    with open(output_dir / "thesis_model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _write_methods_draft(report, output_dir)
    _write_results_draft(report, output_dir, y.values, oof_prob)
    return report


def _write_results_draft(
    report: dict, output_dir: Path, y_true: np.ndarray, y_prob: np.ndarray
) -> None:
    """生成 ROC 图与论文「结果」草稿。"""
    try:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"CV AUC = {report['cv_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("1 - 特异度")
        plt.ylabel("灵敏度")
        plt.title("ROC 曲线（5 折交叉验证合并预测）")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve_cv.png", dpi=150)
        plt.close()
        roc_note = "ROC 图已保存为 `roc_curve_cv.png`。"
    except ImportError:
        roc_note = "未安装 matplotlib，未生成 ROC 图（可运行 pip install matplotlib）。"

    folds = report["cv_folds_detail"]
    fold_lines = "\n".join(
        f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {f['n_lasso_nonzero_coef']} | {f['fold_auc']:.4f} |"
        for f in folds
    )
    top_coef = pd.read_csv(output_dir / "final_model_lasso_coefficients.csv").head(10)
    coef_lines = "\n".join(
        f"| {r['feature']} | {r['coef']:.4f} |" for _, r in top_coef.iterrows()
    )

    (output_dir / "论文_结果_草稿.md").write_text(
        f"""# 结果（草稿）

## 1. 预测模型判别能力
采用 5 折分层交叉验证，将各折验证集预测概率合并后绘制 ROC 曲线。
**交叉验证 AUC = {report['cv_auc']:.4f}**（0.5 表示无判别能力，1.0 为完美判别）。

{roc_note}

## 2. 各折交叉验证 AUC
| 折次 | 训练例数 | 验证例数 | LASSO 非零系数个数 | 该折 AUC |
|------|----------|----------|-------------------|----------|
{fold_lines}

## 3. 探索性模型：LASSO 非零系数较大的指标（全数据拟合，仅作方向参考）
| 特征名 | 回归系数 |
|--------|----------|
{coef_lines}

> 系数为正/负表示与结局正/负相关方向（标准化后）；正式性能以交叉验证 AUC 为准。

## 4. 结果解读（写入讨论时可参考）
- 当前 AUC 接近 0.5，说明在现有样本量与纹理/光谱指标下，**模型尚未显示出稳定判别能力**。
- 可能原因：信号噪声比低、特征与 QFR 关联弱、样本量偏小导致折间波动大（见各折 AUC 差异）。
- 建议：扩大样本、结合临床协变量、外部验证，或与导师讨论是否调整结局定义/特征工程。
""",
        encoding="utf-8",
    )


def _write_methods_draft(report: dict, output_dir: Path) -> None:
    counts = report["outcome_class_counts"]
    params = report["parameters"]
    path = output_dir / "论文_统计学方法_草稿.md"
    path.write_text(
        f"""# 统计学方法（草稿，请按学院/期刊格式修改）

## 1. 研究设计与结局
本研究纳入 {report['n_samples']} 例样本，结局变量为「是否 QFR≤0.8」（0=否，1=是）。
其中阴性（0）{counts.get('0', '?')} 例，阳性（1）{counts.get('1', '?')} 例。

## 2. 高维特征与预处理
共 {report['n_features_raw']} 个影像学/纹理类指标，属于典型「小样本、超高维」（p>>n）数据。
处理顺序如下（避免对全部特征做两两相关）：

1. **质控**：剔除缺失率>30%的变量、近零方差变量、完全重复变量；
2. **与结局关联的预筛**：在训练折上，按与结局的点二列相关系数保留前 {params['prefilter_k']} 个变量；
3. **共线性削减**：在预筛子集上，对 |Pearson r|≥{params['corr_threshold']} 的变量簇聚类，每簇保留与结局相关最强者；
4. **多重比较与最终入选**：Benjamini–Hochberg FDR 校正（α=0.05），并结合互信息，最终保留不超过 {params['top_k']} 个变量；
5. **标准化**：对入选变量进行 Z-score 标准化后建模。

## 3. 预测模型与验证
采用 **L1 正则化逻辑回归（LASSO）** 作为二分类模型，`class_weight='balanced'` 以缓解类别不平衡。
采用 **{params['cv_folds']} 折分层交叉验证**：每一折仅在训练折完成上述特征筛选与模型拟合，在验证折计算预测概率；
将全部验证折预测合并后计算 **AUC** 作为主要判别指标（本次 CV AUC = **{report['cv_auc']:.4f}**）。

> 说明：全数据再拟合一版模型仅用于展示「探索性」入选变量及系数方向（见 `final_model_lasso_coefficients.csv`），**不用于代替交叉验证性能**。

## 4. 样本量与过拟合提示
阳性事件约 {counts.get('1', '?')} 例。按「每变量约 10–20 个阳性事件（EPV）」的经验法则，
未正则化模型同时纳入的变量宜极少；本分析通过 LASSO 正则化 + 折内筛选 + 上限 {params['top_k']} 维以降低过拟合风险。
结果仍应在独立队列或外部验证中进一步确认。

## 5. 软件
Python 3（pandas、scikit-learn、SciPy、openpyxl）。运行：`python run_thesis_pipeline.py --step all`。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.step in ("preprocess", "all"):
        print("=" * 60)
        print("步骤 1/2：数据预处理")
        print("=" * 60)
        report = run_preprocess(
            args.input,
            PREPROCESS_DIR,
            prefilter_k=args.prefilter_k,
            corr_threshold=args.corr_threshold,
            top_k=args.top_k,
            random_state=args.random_state,
        )
        print(f"入选特征数: {report['n_final_selected']}")
        print(f"输出目录: {PREPROCESS_DIR.resolve()}")
        print(f"请阅读: {(PREPROCESS_DIR / '预处理说明.md').resolve()}")

    if args.step in ("model", "all"):
        print("\n" + "=" * 60)
        print("步骤 2/2：预测模型（折内特征筛选 + LASSO + 交叉验证）")
        print("=" * 60)
        model_report = run_nested_cv_model(
            args.input,
            THESIS_DIR,
            prefilter_k=args.prefilter_k,
            corr_threshold=args.corr_threshold,
            top_k=args.top_k,
            cv_folds=args.cv,
            random_state=args.random_state,
        )
        print(f"交叉验证 AUC: {model_report['cv_auc']:.4f}")
        print(f"输出目录: {THESIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
