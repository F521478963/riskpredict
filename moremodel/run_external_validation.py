#!/usr/bin/env python3
"""
外部测试集完整检验（verify.xlsx 等）

用法：
  python run_external_validation.py --test verify.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from preprocess_highdim_clinical import OUTCOME_COL

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "trained_model" / "best_model.joblib"
DEFAULT_OUTPUT = ROOT / "test_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="外部测试集模型检验")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sheet", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    # 兼容两种保存格式：trained_model/ 与 tuned_model/
    if "pipeline" in bundle:
        pipe = bundle["pipeline"]
        feats = bundle["features"]
        meta = bundle.get("report", {})
    else:
        pipe = None
        meta = bundle.get("meta", {})
        feats = bundle["feature_names"]
        scaler = bundle["scaler"]
        model = bundle["model"]

    df = pd.read_excel(args.test, sheet_name=args.sheet)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"缺少 {len(missing)} 个特征列，例如: {missing[:5]}")

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    if pipe is not None:
        prob = pipe.predict_proba(X)[:, 1]
    else:
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[:, 1]

    out = pd.DataFrame({"pred_prob": prob})
    report: dict = {
        "test_file": str(args.test.resolve()),
        "model_file": str(args.model.resolve()),
        "n_test_samples": int(len(df)),
        "n_features_used": len(feats),
        "train_insample_auc_from_meta": meta.get("optimized_train_auc")
        or meta.get("train_insample_auc"),
        "baseline_train_auc_from_meta": meta.get("baseline_lasso_train_auc"),
        "model_type": meta.get("model_type") or meta.get("model"),
    }

    if OUTCOME_COL not in df.columns:
        out.to_csv(args.output_dir / "test_predictions.csv", index=False)
        print("测试集无结局列，仅保存预测概率。")
        return

    y = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    out[OUTCOME_COL] = y.values
    valid = y.isin([0, 1])
    y = y.loc[valid].astype(int)
    prob_v = prob[valid.values]
    out = out.loc[valid].copy()

    counts = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    auc = float(roc_auc_score(y, prob_v))
    fpr, tpr, thresholds = roc_curve(y, prob_v)
    youden_idx = int(np.argmax(tpr - fpr))
    thr_youden = float(thresholds[youden_idx])
    pred_y = (prob_v >= thr_youden).astype(int)
    pred_05 = (prob_v >= 0.5).astype(int)
    cm = confusion_matrix(y, pred_y)
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    report.update(
        {
            "outcome_distribution": counts,
            "external_test_auc": round(auc, 4),
            "average_precision": round(float(average_precision_score(y, prob_v)), 4),
            "brier_score": round(float(brier_score_loss(y, prob_v)), 4),
            "threshold_youden": round(thr_youden, 4),
            "accuracy_at_youden": round(float(accuracy_score(y, pred_y)), 4),
            "balanced_accuracy_at_youden": round(
                float(balanced_accuracy_score(y, pred_y)), 4
            ),
            "sensitivity_at_youden": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity_at_youden": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "confusion_matrix_youden": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "accuracy_at_0.5": round(float(accuracy_score(y, pred_05)), 4),
            "interpretation": (
                "AUC<0.5 表示判别方向可能弱于随机；"
                "训练集 AUC 高而测试集 AUC 低常见于过拟合或人群/测量差异。"
            ),
        }
    )

    out.to_csv(args.output_dir / "test_predictions.csv", index=False)
    with open(args.output_dir / "external_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ROC 图
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"测试集 AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"External validation ROC (AUC={auc:.3f})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(args.output_dir / "roc_external_test.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    train_auc = meta.get("optimized_train_auc") or meta.get("train_insample_auc", "—")
    md = f"""# 外部测试集检验报告

## 数据
- 测试文件：`{args.test.name}`
- 测试样本量：**{len(y)}** 例（结局有效）
- 结局分布：0 = {counts.get('0', '?')} 例，1 = {counts.get('1', '?')} 例
- 使用模型：`{meta.get('model_type', 'unknown')}`，特征数 **{len(feats)}**

## 主要结果

| 指标 | 训练集（样本内，优化后） | **外部测试集 verify** |
|------|-------------------------|------------------------|
| AUC | {train_auc} | **{auc:.4f}** |
| 准确率（Youden 阈值） | — | {report['accuracy_at_youden']:.4f} |
| 灵敏度 / 特异度 | — | {report['sensitivity_at_youden']:.4f} / {report['specificity_at_youden']:.4f} |

Youden 最佳截断值（在测试集上估计）：**{thr_youden:.4f}**

混淆矩阵（行=真实，列=预测；阈值=Youden）：

|  | 预测 0 | 预测 1 |
|--|--------|--------|
| 真实 0 | {tn} | {fp} |
| 真实 1 | {fn} | {tp} |

## 结论（供讨论参考）
- 训练集样本内 AUC 可达 1.0，但独立测试集 AUC 约为 **{auc:.4f}**，接近随机判别（0.5），提示**泛化能力不足**或训练/测试人群、采集条件存在差异。
- 论文中应如实报告外部验证 AUC，不宜仅报告训练集结果。
- 后续可考虑：扩大训练样本、重审特征工程、纳入临床协变量、在测试集上重新校准阈值等。

## 输出文件
- `test_predictions.csv`：预测概率
- `external_validation_report.json`：完整数值
- `roc_external_test.png`：ROC 曲线
"""
    (args.output_dir / "外部验证报告.md").write_text(md, encoding="utf-8")

    print("=" * 56)
    print("外部测试集检验完成")
    print("=" * 56)
    print(f"样本量: {len(y)}  结局: {counts}")
    print(f"外部测试 AUC: {auc:.4f}")
    print(f"训练集样本内 AUC（模型元数据）: {train_auc}")
    print(f"报告目录: {args.output_dir.resolve()}")
    print("=" * 56)


if __name__ == "__main__":
    main()
