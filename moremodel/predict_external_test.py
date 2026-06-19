#!/usr/bin/env python3
"""
用已保存的优化模型，对外部测试集预测并计算 AUC（若测试集含结局列）。

用法：
  python predict_external_test.py --test /path/to/test.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocess_highdim_clinical import OUTCOME_COL

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "trained_model" / "best_model.joblib"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="外部测试集预测")
    p.add_argument("--test", type=Path, required=True, help="测试集 Excel，列名需与训练集一致")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=ROOT / "test_results")
    p.add_argument("--sheet", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    feats: list[str] = bundle["feature_names"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    df = pd.read_excel(args.test, sheet_name=args.sheet)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(
            f"测试集缺少 {len(missing)} 个特征列，示例: {missing[:5]}"
        )

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"pred_prob": prob})
    test_auc = None
    if OUTCOME_COL in df.columns:
        y = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
        out[OUTCOME_COL] = y.values
        valid = y.isin([0, 1])
        if valid.sum() >= 10 and y[valid].nunique() == 2:
            test_auc = float(roc_auc_score(y[valid], prob[valid]))
            print(f"外部测试集 AUC = {test_auc:.4f}")
        else:
            print("测试集无有效结局列，仅输出预测概率。")
    else:
        print("测试集无结局列，仅输出预测概率。")

    out_path = args.output_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    report = {"test_file": str(args.test), "n_rows": len(df), "model": str(args.model)}
    if test_auc is not None:
        report["test_auc"] = test_auc
    with open(args.output_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
