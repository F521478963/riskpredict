#!/usr/bin/env python3
"""
特征筛选合理性验证：审计表、敏感性分析、嵌套 CV、逐维消融。

用法：
  python validate_feature_selection.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "test_data.xlsx"
VERIFY_PATH = ROOT / "verify.xlsx"
OUTPUT_DIR = ROOT / "feature_selection_validation"

# 与 SVM_RBF_stable15 / final_optimize 最优一致
DEFAULT_MAX_KS = 0.17
DEFAULT_TOP_K = 15
SVM_KW = dict(
    C=0.65,
    kernel="rbf",
    gamma=0.1,
    probability=True,
    class_weight="balanced",
    random_state=42,
)
FINAL_15 = [
    "right ear wenli variance183",
    "right ear wenli variance215",
    "face wenli variance896",
    "face wenli mean721",
    "right ear wenli variance1367",
    "face wenli variance792",
    "face guangpu variance106",
    "face guangpu variance107",
    "face guangpu variance108",
    "face guangpu variance109",
    "face wenli variance928",
    "face guangpu variance86",
    "face guangpu variance110",
    "face guangpu variance85",
    "face guangpu variance84",
]


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    x, y = a.dropna().values, b.dropna().values
    if len(x) < 5 or len(y) < 5:
        return 1.0
    return float(stats.ks_2samp(x, y).statistic)


def load_train_verify() -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    y_tr, X_tr = split_xy(load_data(TRAIN_PATH, 0))
    y_tr, X_tr = y_tr[y_tr.isin([0, 1])], X_tr.loc[y_tr.index]
    df_v = pd.read_excel(VERIFY_PATH)
    y_te = pd.to_numeric(df_v[OUTCOME_COL], errors="coerce")
    X_te = df_v.drop(columns=[OUTCOME_COL]).apply(pd.to_numeric, errors="coerce")
    valid = y_te.isin([0, 1])
    y_te, X_te = y_te.loc[valid].astype(int), X_te.loc[valid]
    return y_tr, X_tr, y_te, X_te


def rank_features(y_train: pd.Series, X_train: pd.DataFrame) -> pd.DataFrame:
    X_qc, _ = qc_features(X_train, 0.30)
    return point_biserial_with_y(y_train, X_qc).sort_values("abs_r", ascending=False)


def select_stable(
    uni: pd.DataFrame,
    X_train: pd.DataFrame,
    X_ref: pd.DataFrame,
    *,
    max_ks: float,
    top_k: int,
    use_ks: bool = True,
) -> list[str]:
    out: list[str] = []
    for f in uni["feature"]:
        if f not in X_ref.columns:
            continue
        if use_ks and ks_stat(X_train[f], X_ref[f]) > max_ks:
            continue
        out.append(f)
        if len(out) >= top_k:
            break
    return out


def svm_pipe() -> Pipeline:
    return Pipeline(
        [
            ("sc", RobustScaler()),
            ("clf", SVC(**SVM_KW)),
        ]
    )


def eval_svm(
    feats: list[str],
    y_tr: pd.Series,
    X_tr: pd.DataFrame,
    y_te: pd.Series,
    X_te: pd.DataFrame,
) -> dict:
    if len(feats) < 2:
        return {"n_features": len(feats), "train_cv_auc": None, "verify_auc": None}
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pipe = svm_pipe()
    try:
        cv_prob = cross_val_predict(
            pipe,
            X_tr[feats],
            y_tr,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        cv_auc = float(roc_auc_score(y_tr, cv_prob))
    except Exception:
        cv_auc = float("nan")
    pipe.fit(X_tr[feats], y_tr)
    te_auc = float(roc_auc_score(y_te, pipe.predict_proba(X_te[feats])[:, 1]))
    return {
        "n_features": len(feats),
        "train_cv_auc": round(cv_auc, 4) if cv_auc == cv_auc else None,
        "verify_auc": round(te_auc, 4),
    }


def build_audit(
    y_tr: pd.Series,
    X_tr: pd.DataFrame,
    y_te: pd.Series,
    X_te: pd.DataFrame,
    *,
    max_ks: float,
    top_k: int,
    audit_top_n: int = 50,
) -> pd.DataFrame:
    uni = rank_features(y_tr, X_tr)
    selected = select_stable(uni, X_tr, X_te, max_ks=max_ks, top_k=top_k, use_ks=True)
    selected_set = set(selected)

    rows = []
    passed_ks_rank: list[str] = []
    for rank, row in enumerate(uni.head(audit_top_n).itertuples(index=False), start=1):
        f = row.feature
        ks = ks_stat(X_tr[f], X_te[f]) if f in X_te.columns else float("nan")
        ks_ok = ks <= max_ks if ks == ks else False

        if f in selected_set:
            status = "入选最终模型"
        elif not ks_ok:
            status = "因 KS 超阈值未入选"
        elif len(passed_ks_rank) >= top_k:
            status = f"因 top_k={top_k} 截断未入选"
        else:
            status = "其他"

        if ks_ok:
            passed_ks_rank.append(f)

        rows.append(
            {
                "rank_by_abs_r": rank,
                "feature": f,
                "abs_r": round(float(row.abs_r), 4),
                "r": round(float(row.r), 4),
                "p_value": float(row.p_value),
                "ks_train_vs_verify": round(ks, 4) if ks == ks else None,
                "ks_pass": ks_ok,
                "in_final_15": f in selected_set,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def audit_final_features(
    uni: pd.DataFrame,
    selected: list[str],
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
) -> pd.DataFrame:
    """最终入选 15 维在全体特征中的 |r| 排名与 KS。"""
    rank_map = {row.feature: i + 1 for i, row in enumerate(uni.itertuples(index=False))}
    rows = []
    for f in selected:
        sub = uni[uni["feature"] == f]
        if sub.empty:
            rows.append(
                {
                    "feature": f,
                    "rank_by_abs_r": None,
                    "abs_r": None,
                    "ks_train_vs_verify": round(ks_stat(X_tr[f], X_te[f]), 4),
                    "in_top50_by_r": False,
                }
            )
            continue
        r0 = sub.iloc[0]
        rk = rank_map.get(f)
        rows.append(
            {
                "feature": f,
                "rank_by_abs_r": rk,
                "abs_r": round(float(r0["abs_r"]), 4),
                "ks_train_vs_verify": round(ks_stat(X_tr[f], X_te[f]), 4),
                "in_top50_by_r": rk is not None and rk <= 50,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_by_abs_r", na_position="last")


def run_sensitivity(
    y_tr: pd.Series,
    X_tr: pd.DataFrame,
    y_te: pd.Series,
    X_te: pd.DataFrame,
    uni: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for use_ks in (True, False):
        for top_k in (8, 12, 15, 20, 25, 30):
            for max_ks in (0.15, 0.17, 0.20, 0.25) if use_ks else (0.17,):
                feats = select_stable(
                    uni, X_tr, X_te, max_ks=max_ks, top_k=top_k, use_ks=use_ks
                )
                m = eval_svm(feats, y_tr, X_tr, y_te, X_te)
                rows.append(
                    {
                        "use_ks": use_ks,
                        "max_ks": max_ks if use_ks else None,
                        "top_k": top_k,
                        "n_selected": len(feats),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def run_nested_cv(
    y_tr: pd.Series,
    X_tr: pd.DataFrame,
    y_te: pd.Series,
    X_te: pd.DataFrame,
    *,
    max_ks: float,
    top_k: int,
    n_splits: int = 5,
) -> dict:
    """
    外层 5 折：每折仅在折内训练子集上 QC+排序+KS(折内 val 作分布参照)，
    在折外 val 上预测；verify 仅最终评估一次（固定流程，不参与选特征）。
    """
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    oof = np.full(len(y_tr), np.nan)
    fold_feats: list[list[str]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr)):
        y_in, X_in = y_tr.iloc[tr_idx], X_tr.iloc[tr_idx]
        y_va, X_va = y_tr.iloc[va_idx], X_tr.iloc[va_idx]
        uni = rank_features(y_in, X_in)
        feats = select_stable(uni, X_in, X_va, max_ks=max_ks, top_k=top_k, use_ks=True)
        fold_feats.append(feats)
        if len(feats) < 2:
            continue
        pipe = svm_pipe()
        pipe.fit(X_in[feats], y_in)
        oof[va_idx] = pipe.predict_proba(X_va[feats])[:, 1]

    valid = np.isfinite(oof)
    nested_train_auc = float(roc_auc_score(y_tr[valid], oof[valid])) if valid.sum() > 10 else None

    # verify：在全部 train 上按 train-vs-verify KS 选特征（与部署一致，但无 verify 标签参与）
    uni_full = rank_features(y_tr, X_tr)
    feats_full = select_stable(
        uni_full, X_tr, X_te, max_ks=max_ks, top_k=top_k, use_ks=True
    )
    m_verify = eval_svm(feats_full, y_tr, X_tr, y_te, X_te)

    # 特征稳定性：各折入选特征出现频率
    from collections import Counter

    cnt = Counter(f for fl in fold_feats for f in fl)
    stab = pd.DataFrame(
        [{"feature": f, "fold_frequency": c, "in_final_15": f in FINAL_15} for f, c in cnt.most_common()]
    )

    return {
        "nested_oof_auc_on_train": round(nested_train_auc, 4) if nested_train_auc else None,
        "verify_auc_same_pipeline_on_full_train": m_verify.get("verify_auc"),
        "train_cv_auc_full_train": m_verify.get("train_cv_auc"),
        "fold_feature_counts": [len(f) for f in fold_feats],
        "feature_stability": stab,
    }


def run_ablation(
    y_tr: pd.Series,
    X_tr: pd.DataFrame,
    y_te: pd.Series,
    X_te: pd.DataFrame,
    feats: list[str],
) -> pd.DataFrame:
    base = eval_svm(feats, y_tr, X_tr, y_te, X_te)
    rows = [
        {
            "dropped_feature": "(none, full model)",
            "n_features": len(feats),
            "train_cv_auc": base["train_cv_auc"],
            "verify_auc": base["verify_auc"],
            "verify_auc_drop": 0.0,
        }
    ]
    for drop in feats:
        sub = [f for f in feats if f != drop]
        m = eval_svm(sub, y_tr, X_tr, y_te, X_te)
        drop_val = (
            (base["verify_auc"] - m["verify_auc"])
            if base["verify_auc"] is not None and m["verify_auc"] is not None
            else None
        )
        rows.append(
            {
                "dropped_feature": drop,
                "n_features": len(sub),
                "train_cv_auc": m["train_cv_auc"],
                "verify_auc": m["verify_auc"],
                "verify_auc_drop": round(drop_val, 4) if drop_val is not None else None,
            }
        )
    return pd.DataFrame(rows)


def df_to_md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """简单 Markdown 表（不依赖 tabulate）。"""
    if df.empty:
        return "（无数据）"
    cols = cols or list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def region_summary(audit: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    def region(name: str) -> str:
        if name.startswith("right ear"):
            return "右耳"
        if name.startswith("left ear"):
            return "左耳"
        if name.startswith("face"):
            return "面部"
        return "其他"

    def ftype(name: str) -> str:
        if "guangpu" in name:
            return "光谱"
        if "wenli" in name:
            return "纹理"
        return "其他"

    rows = []
    for scope, names in [
        ("Top50_by_r", audit["feature"].tolist()),
        ("Final_15", selected),
    ]:
        for n in names:
            rows.append(
                {
                    "scope": scope,
                    "region": region(n),
                    "type": ftype(n),
                    "feature": n,
                }
            )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["scope", "region", "type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


def write_report(
    audit: pd.DataFrame,
    audit_final: pd.DataFrame,
    sensitivity: pd.DataFrame,
    nested: dict,
    ablation: pd.DataFrame,
    region: pd.DataFrame,
    baseline: dict,
) -> str:
    n_final = len(audit_final)
    in_top50 = int(audit_final["in_top50_by_r"].sum())
    ks_rejected = audit[(audit["rank_by_abs_r"] <= 30) & (~audit["ks_pass"])]
    rank_rejected = audit[
        (audit["ks_pass"]) & (~audit["in_final_15"]) & (audit["rank_by_abs_r"] <= 30)
    ]
    best_sens = sensitivity.sort_values("verify_auc", ascending=False).head(5)

    nested_auc = nested.get("nested_oof_auc_on_train")
    verify_deploy = nested.get("verify_auc_same_pipeline_on_full_train")
    stab_top = nested["feature_stability"].head(20)

    ab_top = ablation[ablation["dropped_feature"] != "(none, full model)"].sort_values(
        "verify_auc_drop", ascending=False
    ).head(5)

    lines = [
        "# 特征筛选验证报告",
        "",
        "## 1. 审计：高相关特征为何未入选（Top 50）",
        "",
        f"- 最终入选 **{n_final}** 个特征（max_ks={DEFAULT_MAX_KS}, top_k={DEFAULT_TOP_K}）",
        f"- 其中仅 **{in_top50}** 个落在「按 |r| 排序的 Top50」内，其余 **{n_final - in_top50}** 个排名更靠后但 KS 稳定（多为面部光谱方差）",
        f"- Top30 中因 **KS>阈值** 未入选：**{len(ks_rejected[ks_rejected['rank_by_abs_r']<=30])}** 个",
        f"- Top50 表内 KS 通过但未进最终 15：**{int(audit['ks_pass'].sum()) - in_top50}** 个（被排序更靠后但 KS 通过的 guangpu 等「占位」）",
        "",
        "详见 `feature_audit_top50.csv` 与 `feature_audit_final15_ranks.csv`。",
        "",
        "### Top30 内 KS 未通过示例（最多 5 个）",
        "",
    ]
    if len(ks_rejected):
        for _, r in ks_rejected.head(5).iterrows():
            lines.append(
                f"- #{int(r['rank_by_abs_r'])} `{r['feature']}` |r|={r['abs_r']}, KS={r['ks_train_vs_verify']}"
            )
    else:
        lines.append("- （无）")

    lines.extend(
        [
            "",
            "## 2. 敏感性分析（SVM 固定 C=0.65, gamma=0.1）",
            "",
            f"- 当前方案 verify AUC：**{baseline['verify_auc']}**（CV={baseline['train_cv_auc']}）",
            "",
            "### verify AUC 最高的 5 组参数",
            "",
            df_to_md_table(
                best_sens[
                    ["use_ks", "max_ks", "top_k", "n_selected", "train_cv_auc", "verify_auc"]
                ]
            ),
            "",
            "完整结果见 `sensitivity_grid.csv`。",
            "",
            "## 3. 嵌套交叉验证（仅用训练 160 例选特征）",
            "",
            f"- 训练集 OOF AUC（折内 KS，折外预测）：**{nested_auc}**",
            f"- 全训练集选特征 + verify 评估（与部署一致）：**{verify_deploy}**",
            f"- 全训练集 5 折 CV AUC：**{nested.get('train_cv_auc_full_train')}**",
            "",
            "各折入选特征数：" + str(nested.get("fold_feature_counts")),
            "",
            "### 折内入选频率 Top 特征",
            "",
            df_to_md_table(stab_top),
            "",
            "详见 `nested_cv_feature_stability.csv`。",
            "",
            "## 4. 逐维消融（去掉 1 个特征重训）",
            "",
            f"- 完整 15 维 verify AUC：**{baseline['verify_auc']}**",
            "",
            "### 去掉后 verify AUC 下降最多（该维贡献大）",
            "",
        ]
    )
    if len(ab_top):
        lines.append(
            df_to_md_table(ab_top[["dropped_feature", "verify_auc", "verify_auc_drop"]])
        )
    else:
        lines.append("- （无）")
    lines.extend(
        [
            "",
            "详见 `ablation_leave_one_out.csv`。",
            "",
            "## 5. 部位/类型分布",
            "",
            df_to_md_table(region),
            "",
            "## 6. 结论与建议",
            "",
        ]
    )

    # Auto conclusions
    sens_15 = sensitivity[
        (sensitivity["use_ks"] == True)
        & (sensitivity["max_ks"] == DEFAULT_MAX_KS)
        & (sensitivity["top_k"] == DEFAULT_TOP_K)
    ]
    no_ks_15 = sensitivity[
        (sensitivity["use_ks"] == False) & (sensitivity["top_k"] == DEFAULT_TOP_K)
    ]
    if len(sens_15) and len(no_ks_15):
        v_ks = sens_15.iloc[0]["verify_auc"]
        v_no = no_ks_15.iloc[0]["verify_auc"]
        lines.append(
            f"- **KS 筛选**：有 KS verify={v_ks}，无 KS verify={v_no}；"
            + ("KS 有助于当前 verify。" if v_ks and v_no and v_ks >= v_no else "去掉 KS 后 verify 更高，KS 可能过严。")
        )

    best_row = sensitivity.sort_values("verify_auc", ascending=False).iloc[0]
    if best_row["top_k"] != DEFAULT_TOP_K or best_row["use_ks"] != True:
        lines.append(
            f"- **维度**：网格最优为 top_k={int(best_row['top_k'])}, use_ks={best_row['use_ks']}, verify={best_row['verify_auc']}；"
            f"与当前 15 维方案对比见敏感性表。"
        )
    else:
        lines.append("- **维度**：当前 top_k=15 + KS 在敏感性网格中处于较优区间。")

    if nested_auc and verify_deploy:
        gap = verify_deploy - nested_auc
        lines.append(
            f"- **过拟合风险**：嵌套 OOF（{nested_auc}）与 verify（{verify_deploy}）相差 {gap:+.3f}；"
            + (
                "verify 明显高于折内泛化，需警惕 verify 调参/样本量偏小带来的乐观。"
                if gap > 0.08
                else "差距在可接受范围，但仍建议独立第三队列验证。"
            )
        )

    left_in_top50 = audit["feature"].str.startswith("left ear").sum()
    lines.append(
        f"- **左耳**：Top50 中左耳特征 {left_in_top50} 个，最终 15 维中 0 个；"
        "若临床认为左耳关键，应对照审计表中左耳特征的 |r| 与 KS。"
    )
    lines.append(
        "- **论文建议**：附 `feature_audit_top50.csv`；报告敏感性分析与嵌套 OOF；"
        "外部验证队列固定 15 特征、不再重筛。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("加载数据…")
    y_tr, X_tr, y_te, X_te = load_train_verify()
    uni = rank_features(y_tr, X_tr)

    print("1/4 特征审计 Top50…")
    audit = build_audit(
        y_tr, X_tr, y_te, X_te, max_ks=DEFAULT_MAX_KS, top_k=DEFAULT_TOP_K
    )
    audit.to_csv(OUTPUT_DIR / "feature_audit_top50.csv", index=False)
    feats_final = select_stable(
        uni, X_tr, X_te, max_ks=DEFAULT_MAX_KS, top_k=DEFAULT_TOP_K
    )
    audit_final = audit_final_features(uni, feats_final, X_tr, X_te)
    audit_final.to_csv(OUTPUT_DIR / "feature_audit_final15_ranks.csv", index=False)

    print("2/4 敏感性分析…")
    sensitivity = run_sensitivity(y_tr, X_tr, y_te, X_te, uni)
    sensitivity.to_csv(OUTPUT_DIR / "sensitivity_grid.csv", index=False)

    print("3/4 嵌套 CV…")
    nested = run_nested_cv(
        y_tr, X_tr, y_te, X_te, max_ks=DEFAULT_MAX_KS, top_k=DEFAULT_TOP_K
    )
    nested["feature_stability"].to_csv(
        OUTPUT_DIR / "nested_cv_feature_stability.csv", index=False
    )
    nested_json = {k: v for k, v in nested.items() if k != "feature_stability"}
    with open(OUTPUT_DIR / "nested_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(nested_json, f, ensure_ascii=False, indent=2)

    print("4/4 逐维消融…")
    ablation = run_ablation(y_tr, X_tr, y_te, X_te, feats_final)
    ablation.to_csv(OUTPUT_DIR / "ablation_leave_one_out.csv", index=False)

    baseline = eval_svm(feats_final, y_tr, X_tr, y_te, X_te)
    region = region_summary(audit, feats_final)
    region.to_csv(OUTPUT_DIR / "region_type_counts.csv", index=False)

    report = write_report(
        audit, audit_final, sensitivity, nested, ablation, region, baseline
    )
    (OUTPUT_DIR / "验证报告.md").write_text(report, encoding="utf-8")

    summary = {
        "baseline_15_stable": baseline,
        "audit_n_selected": int(audit["in_final_15"].sum()),
        "best_sensitivity": sensitivity.sort_values("verify_auc", ascending=False)
        .iloc[0]
        .to_dict(),
        "nested": nested_json,
    }
    with open(OUTPUT_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n完成。输出目录：{OUTPUT_DIR.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
