#!/usr/bin/env python3
"""
高维临床数据预处理（p >> n）

适用：样本量 n 远小于特征数 p（如 n≈160, p>10000），二分类结局。

策略概要（不必先手工算完全部特征两两相关）：
1. 质控：去常数列、去完全重复列、可选剔除高缺失列
2. 冗余削减：在「与结局相关性」或方差筛选后的子集上，按 |r| 聚类/阈值去高度共线特征
3. 单变量筛选 + FDR 校正（控制多重比较假阳性）
4. 标准化，输出建模用矩阵

注意：特征筛选若用于最终模型评估，应放在交叉验证「内部」进行，避免信息泄漏；
本脚本产出的是「探索性/训练集」预处理结果，建模时请用 Pipeline + CV。

依赖：
    pip install pandas openpyxl numpy scipy scikit-learn

用法：
    python preprocess_highdim_clinical.py
    python preprocess_highdim_clinical.py --input test_data.xlsx --corr-threshold 0.95 --top-k 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

OUTCOME_COL = "是否QFR≤0.8（0非/1是）"
DEFAULT_INPUT = Path(__file__).resolve().parent / "test_data.xlsx"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "preprocessed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="高维临床数据预处理 (p >> n)")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sheet", default=0)
    p.add_argument("--missing-threshold", type=float, default=0.30)
    p.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="特征间 |Pearson r| 超过该值视为冗余，每簇保留 1 个",
    )
    p.add_argument(
        "--prefilter-k",
        type=int,
        default=2000,
        help="做相关聚类前先按与结局的 |点二列相关| 保留前 K 个特征（控制计算量）",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="FDR 校正后最终保留特征数上限（博士论文默认 30，与 EPV 经验法则折中）",
    )
    p.add_argument("--fdr-alpha", type=float, default=0.05, help="Benjamini-Hochberg FDR 水平")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_data(path: Path, sheet) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    if OUTCOME_COL not in df.columns:
        raise KeyError(f"未找到结局列 {OUTCOME_COL!r}")
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    y = df[OUTCOME_COL].astype(int)
    X = df.drop(columns=[OUTCOME_COL]).apply(pd.to_numeric, errors="coerce")
    return y, X


def qc_features(X: pd.DataFrame, missing_threshold: float) -> tuple[pd.DataFrame, dict]:
    log: dict = {"steps": []}

    n0 = X.shape[1]
    miss_rate = X.isna().mean()
    keep = miss_rate <= missing_threshold
    dropped_miss = (~keep).sum()
    X = X.loc[:, keep]
    log["steps"].append(f"剔除缺失率>{missing_threshold:.0%} 的特征: {dropped_miss}")

    var = X.var(axis=0, skipna=True)
    keep_var = var > 1e-12
    dropped_var = (~keep_var).sum()
    X = X.loc[:, keep_var]
    log["steps"].append(f"剔除近零方差特征: {dropped_var}")

    dup_mask = X.T.duplicated(keep="first")
    dropped_dup = int(dup_mask.sum())
    X = X.loc[:, ~dup_mask]
    log["steps"].append(f"剔除完全重复列: {dropped_dup}")

    log["n_features_raw"] = n0
    log["n_features_after_qc"] = X.shape[1]
    return X, log


def point_biserial_with_y(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """每个特征与二分类结局的点二列相关及 p 值。"""
    rows = []
    yv = y.values
    for col in X.columns:
        x = X[col].values
        mask = np.isfinite(x)
        if mask.sum() < 10:
            continue
        r, p = stats.pointbiserialr(yv[mask], x[mask])
        rows.append({"feature": col, "abs_r": abs(r), "r": r, "p_value": p})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """返回是否拒绝原假设的布尔掩码。"""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_idx = np.where(below)[0].max()
    cutoff = ranked[max_idx]
    return p <= cutoff


def prefilter_by_outcome_corr(
    y: pd.Series, X: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = point_biserial_with_y(y, X)
    top = scores.head(min(k, len(scores)))
    cols = top["feature"].tolist()
    return X[cols], scores


def dedupe_by_correlation(
    X: pd.DataFrame,
    scores: pd.DataFrame,
    corr_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    对高度相关特征做层次聚类，每簇保留与结局 |r| 最大的一个。
    仅在 prefilter 后的子矩阵上计算相关，避免 1 万维全 pairwise。
    """
    if X.shape[1] <= 1:
        return X, list(X.columns)

    imp = scores.set_index("feature")["abs_r"]
    corr_arr = X.corr(method="pearson").abs().to_numpy().copy()
    np.fill_diagonal(corr_arr, 0.0)
    dist = 1.0 - corr_arr
    np.clip(dist, 0.0, None, out=dist)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # 距离 1 - |r| <= 1 - threshold  => 同一簇
    cluster_id = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    selected: list[str] = []
    for cid in np.unique(cluster_id):
        members = X.columns[cluster_id == cid].tolist()
        best = max(members, key=lambda c: imp.get(c, 0.0))
        selected.append(best)
    return X[selected], selected


def select_by_fdr_and_mi(
    y: pd.Series,
    X: pd.DataFrame,
    univariate_scores: pd.DataFrame,
    top_k: int,
    fdr_alpha: float,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    sub = univariate_scores[univariate_scores["feature"].isin(X.columns)].copy()
    sub["fdr_significant"] = benjamini_hochberg(sub["p_value"].values, fdr_alpha)
    sig = sub.loc[sub["fdr_significant"], "feature"].tolist()

    mi = mutual_info_classif(
        X.values, y.values, random_state=random_state, discrete_features=False
    )
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_info": mi})
    mi_df = mi_df.sort_values("mutual_info", ascending=False)

    # 优先 FDR 显著；不足 top_k 时用互信息补足
    chosen = list(sig)
    if len(chosen) < top_k:
        for f in mi_df["feature"]:
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= top_k:
                break
    else:
        mi_order = {f: i for i, f in enumerate(mi_df["feature"])}
        chosen = sorted(chosen, key=lambda f: mi_order.get(f, 10**9))[:top_k]

    detail = sub.merge(mi_df, on="feature", how="right")
    detail["selected_final"] = detail["feature"].isin(chosen)
    return chosen, detail


def write_preprocess_readme(report: dict, output_dir: Path) -> None:
    """生成供论文「资料与方法」参考的中文说明。"""
    counts = report["outcome_class_counts"]
    feats = report["final_features"]
    feat_lines = "\n".join(f"  - {f}" for f in feats)
    text = f"""# 数据预处理说明（自动生成）

## 一、数据概况
- 原始文件：`{report['input']}`
- 样本量：**{report['n_samples']}** 例
- 原始特征数：**{report['n_features_raw']}** 个
- 结局「是否 QFR≤0.8」：0（未发病）**{counts.get('0', counts.get(0, '?'))}** 例，1（发病）**{counts.get('1', counts.get(1, '?'))}** 例

## 二、为何不能直接用 1 万多个指标建模？
本数据属于 **小样本、超高维（p>>n）**：特征数远大于样本数。若不做筛选，
模型会严重过拟合，论文审稿人也难以接受。因此采用「质控 → 与结局相关预筛 → 去共线 → 多重比较校正 → 标准化」的流程。

## 三、预处理步骤（写入论文时可略作润色）
1. **质控**：剔除缺失率 > 30% 的变量、近零方差变量、完全重复变量；
2. **与结局关联的预筛**：按与结局的点二列相关系数，保留前 **{report['prefilter_k']}** 个变量；
3. **共线性削减**：在预筛子集上，对 |Pearson r| ≥ **{report['corr_threshold']}** 的变量簇聚类，每簇保留与结局相关最强者（剩余 **{report['n_after_corr_dedup']}** 个）；
4. **多重比较与入选**：Benjamini–Hochberg FDR（α={report['fdr_alpha']}），结合互信息，最终保留 **{report['n_final_selected']}** 个变量；
5. **标准化**：对入选变量做 Z-score，得到 `X_preprocessed.csv`。

## 四、最终入选特征
{feat_lines}

## 五、输出文件说明
| 文件 | 含义 |
|------|------|
| `X_preprocessed.csv` | 标准化后的特征矩阵（**后续建模请用此文件 + y.csv**） |
| `X_selected_unscaled.csv` | 入选特征的原始数值（未标准化） |
| `y.csv` | 结局列 |
| `univariate_scores.csv` | 全部特征与结局的单变量相关与 p 值 |
| `selection_detail.csv` | 去共线后子集上的 FDR、互信息、是否入选 |
| `preprocess_report.json` | 机器可读的完整参数与结果 |

## 六、下一步：预测模型
在 `moremodel` 目录运行（交叉验证在每一折**内部**重新做特征筛选，避免信息泄漏）：

```bash
python run_thesis_pipeline.py --step model
```

或一键预处理 + 建模：

```bash
python run_thesis_pipeline.py --step all
```

## 七、重要提示
- 阳性事件约 {counts.get('1', counts.get(1, '?'))} 例；按 EPV≈10 的经验法则，未正则化模型同时纳入的变量宜 ≤ **{report['epv_rule_of_thumb_max_features']}** 个；本流程保留 {report['n_final_selected']} 个并建议建模时使用 **LASSO 正则化**。
- 当前 `X_preprocessed.csv` 的筛选是在**全数据**上完成的，适合探索性分析；**正式报告模型性能**时请使用 `run_thesis_pipeline.py` 的折内筛选结果。
"""
    (output_dir / "预处理说明.md").write_text(text, encoding="utf-8")


def run_preprocess(
    input_path: Path,
    output_dir: Path,
    *,
    sheet=0,
    missing_threshold: float = 0.30,
    corr_threshold: float = 0.95,
    prefilter_k: int = 2000,
    top_k: int = 30,
    fdr_alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """执行完整预处理并写入 output_dir，返回报告字典。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path, sheet)
    y, X = split_xy(df)

    valid = y.isin([0, 1])
    drop_note = None
    if (~valid).any():
        n_drop = int((~valid).sum())
        df = df.loc[valid].copy()
        y, X = split_xy(df)
        drop_note = f"剔除结局非 0/1 的样本: {n_drop}"

    class_counts = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    n_events = int(class_counts.get("1", 0))
    epv_hint = max(1, n_events // 10)

    X_qc, qc_log = qc_features(X, missing_threshold)
    X_pre, uni_scores = prefilter_by_outcome_corr(y, X_qc, prefilter_k)
    X_dedup, _ = dedupe_by_correlation(X_pre, uni_scores, corr_threshold)
    final_features, selection_detail = select_by_fdr_and_mi(
        y, X_dedup, uni_scores, top_k, fdr_alpha, random_state
    )

    X_final = X_dedup[final_features]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_final),
        columns=final_features,
        index=X_final.index,
    )

    X_scaled.to_csv(output_dir / "X_preprocessed.csv", index=False)
    X_final.to_csv(output_dir / "X_selected_unscaled.csv", index=False)
    y.to_csv(output_dir / "y.csv", index=False, header=[OUTCOME_COL])
    uni_scores.to_csv(output_dir / "univariate_scores.csv", index=False)
    selection_detail.to_csv(output_dir / "selection_detail.csv", index=False)
    (output_dir / "feature_list.txt").write_text(
        "\n".join(final_features) + "\n", encoding="utf-8"
    )

    report = {
        "input": str(input_path.resolve()),
        "n_samples": int(len(y)),
        "n_features_raw": int(X.shape[1]),
        "outcome_class_counts": class_counts,
        "qc": qc_log,
        "prefilter_k": prefilter_k,
        "n_after_corr_dedup": int(X_dedup.shape[1]),
        "n_final_selected": len(final_features),
        "final_features": final_features,
        "corr_threshold": corr_threshold,
        "fdr_alpha": fdr_alpha,
        "top_k_cap": top_k,
        "epv_rule_of_thumb_max_features": epv_hint,
        "notes": [
            drop_note,
            "p>>n：不宜在未正则化的逻辑回归中同时使用大量特征；"
            f"阳性事件 n1={n_events}，粗估 EPV≈10 时同时建模变量宜 ≤{epv_hint}。",
            "不必先对 1 万维做全量两两相关；应在 QC + 与结局相关预筛后，再处理共线性。",
            "建模评估请用 StratifiedKFold + Pipeline，在折内重复筛选/训练。",
        ],
    }
    report["notes"] = [n for n in report["notes"] if n]

    with open(output_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_preprocess_readme(report, output_dir)
    return report


def main() -> None:
    args = parse_args()
    report = run_preprocess(
        args.input,
        args.output_dir,
        sheet=args.sheet,
        missing_threshold=args.missing_threshold,
        corr_threshold=args.corr_threshold,
        prefilter_k=args.prefilter_k,
        top_k=args.top_k,
        fdr_alpha=args.fdr_alpha,
        random_state=args.random_state,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已写入: {args.output_dir.resolve()}")
    print(f"说明文档: {(args.output_dir / '预处理说明.md').resolve()}")


if __name__ == "__main__":
    main()
