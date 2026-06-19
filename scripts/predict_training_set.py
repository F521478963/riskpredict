#!/usr/bin/env python3
"""Batch-predict overall Y and branch QFR for Ridge Excel inputs."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_registry import (
    BRANCH_MODEL_SPECS,
    OVERALL_FEATURE_NAMES,
    load_branch_services,
    predict_branch_qfr,
    resolve_branch_qfr_panel,
)
from model_service import PredictionService

MODEL_DIR = os.path.join(BASE_DIR, "20260610_most_powerful")
DEFAULT_INPUT = os.path.join(MODEL_DIR, "y_Ridge.xlsx")
DEFAULT_OUTPUT = os.path.join(MODEL_DIR, "y_Ridge_predictions.xlsx")

SHEET_CONFIG = [
    ("demo0", "训练集"),
    ("demo1", "测试集"),
]

BRANCH_DATA_FILES = {
    "lad": "y1_Ridge.xlsx",
    "lcx": "y2_Ridge.xlsx",
    "rca": "y3_Ridge.xlsx",
}

BRANCH_LABEL_COLUMNS = {
    "lad": "LAD",
    "lcx": "LCX",
    "rca": "RCA",
}

MERGE_KEY_FEATURES = [
    "face guangpu mean1",
    "face wenli mean30",
    "face wenli mean78",
    "face wenli mean94",
    "face wenli mean102",
    "face wenli mean126",
    "face wenli mean166",
    "left ear wenli mean218",
    "left ear wenli mean1438",
    "right ear wenli mean1574",
    "right ear wenli variance1608",
]

Y_ONLY_FEATURES = [
    "face wenli variance951",
    "left ear wenli variance246",
    "left ear wenli mean214",
    "right ear wenli mean1184",
]

TEST_FILE_COLUMN_NAMES = MERGE_KEY_FEATURES + [
    "face wenli variance455",
    "face wenli variance1007",
    "left ear wenli variance894",
    "left ear wenli mean217",
    "right ear wenli mean200",
]

BRANCH_EXTRAS = {
    "lad": [
        "face wenli variance455",
        "face wenli variance1007",
        "left ear wenli variance894",
        "left ear wenli mean217",
        "right ear wenli mean200",
    ],
    "lcx": [
        "left ear wenli variance1088",
        "left ear wenli variance262",
        "left ear wenli mean904",
        "face wenli variance535",
        "left ear wenli variance872",
        "left ear wenli variance848",
        "face wenli variance1167",
    ],
    "rca": [
        "right ear wenli variance998",
        "face wenli variance559",
        "right ear wenli mean928",
        "face wenli variance1454",
        "face wenli variance935",
        "face wenli variance863",
    ],
}


def _is_numeric_test_format(frame: pd.DataFrame) -> bool:
    if len(frame.columns) != 16:
        return False
    return all(str(column).isdigit() for column in frame.columns)


def _merge_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed[MERGE_KEY_FEATURES] = keyed[MERGE_KEY_FEATURES].round(6)
    return keyed


def _load_reference_sheet(filename: str, sheet_name: str = "demo1") -> pd.DataFrame:
    return pd.read_excel(os.path.join(MODEL_DIR, filename), sheet_name=sheet_name)


def _merge_reference_columns(
    overall_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    columns_to_add = [column for column in columns if column not in overall_df.columns]
    if not columns_to_add:
        return overall_df

    payload = reference_df[MERGE_KEY_FEATURES + columns_to_add]
    return _merge_key_frame(overall_df).merge(
        _merge_key_frame(payload),
        on=MERGE_KEY_FEATURES,
        how="left",
        validate="one_to_one",
    )


def _predict_frame(
    frame: pd.DataFrame,
    overall_service: PredictionService,
    branch_services: dict,
) -> pd.DataFrame:
    working = frame.copy()

    y_reference = _load_reference_sheet("y_Ridge.xlsx")
    working = _merge_reference_columns(
        working,
        y_reference,
        ["是否QFR≤0.8（0非/1是）", *Y_ONLY_FEATURES],
    )

    for spec in BRANCH_MODEL_SPECS:
        branch_id = spec["id"]
        reference_df = _load_reference_sheet(BRANCH_DATA_FILES[branch_id])
        label_col = BRANCH_LABEL_COLUMNS[branch_id]
        extras = [
            column
            for column in BRANCH_EXTRAS[branch_id]
            if column not in working.columns
        ]
        working = _merge_reference_columns(
            working,
            reference_df,
            [label_col, *extras],
        )
        working = working.rename(columns={label_col: f"{label_col}_真实值"})

    y_predictions = []
    branch_predictions = {spec["id"]: [] for spec in BRANCH_MODEL_SPECS}

    for _, row in working.iterrows():
        overall_map = {name: float(row[name]) for name in OVERALL_FEATURE_NAMES}
        overall_value = overall_service.predict_values_by_names(overall_map)
        y_predictions.append(overall_value)

        raw_branch = {}
        for spec in BRANCH_MODEL_SPECS:
            branch_id = spec["id"]
            feature_map = {name: float(row[name]) for name in spec["feature_names"]}
            raw_branch[branch_id] = predict_branch_qfr(
                branch_services[branch_id], feature_map
            )

        resolved_branch = resolve_branch_qfr_panel(raw_branch, overall_value)
        for spec in BRANCH_MODEL_SPECS:
            branch_predictions[spec["id"]].append(
                round(resolved_branch[spec["id"]], 6)
            )

    result = working.copy()
    result["整体Y_预测值"] = [round(value, 6) for value in y_predictions]
    for spec in BRANCH_MODEL_SPECS:
        branch_id = spec["id"]
        label_col = BRANCH_LABEL_COLUMNS[branch_id]
        result[f"{label_col}_预测值"] = branch_predictions[branch_id]

    return result


def _order_training_columns(frame: pd.DataFrame) -> pd.DataFrame:
    leading = ["数据集", "是否QFR≤0.8（0非/1是）", *OVERALL_FEATURE_NAMES, "整体Y_预测值"]
    ordered = [column for column in leading if column in frame.columns]

    for branch_id in ("lad", "lcx", "rca"):
        label_col = f"{BRANCH_LABEL_COLUMNS[branch_id]}_真实值"
        pred_col = f"{BRANCH_LABEL_COLUMNS[branch_id]}_预测值"
        group = [label_col]
        group.extend(
            column for column in BRANCH_EXTRAS[branch_id] if column in frame.columns
        )
        if pred_col in frame.columns:
            group.append(pred_col)
        ordered.extend(column for column in group if column not in ordered)

    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def _order_test_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        *TEST_FILE_COLUMN_NAMES,
        "是否QFR≤0.8（0非/1是）",
        "整体Y_预测值",
        "LAD_真实值",
        "LAD_预测值",
        "LCX_真实值",
        "LCX_预测值",
        "RCA_真实值",
        "RCA_预测值",
    ]
    return frame[[column for column in ordered if column in frame.columns]]


def _predict_training_workbook(
    input_path: str,
    overall_service: PredictionService,
    branch_services: dict,
) -> dict[str, pd.DataFrame]:
    sheet_frames = {}
    for sheet_name, split_label in SHEET_CONFIG:
        overall_df = pd.read_excel(input_path, sheet_name=sheet_name)
        frame = overall_df.copy()
        frame.insert(0, "数据集", split_label)
        sheet_frames[sheet_name] = _order_training_columns(
            _predict_frame(frame, overall_service, branch_services)
        )
    return sheet_frames


def _predict_test_workbook(
    input_path: str,
    overall_service: PredictionService,
    branch_services: dict,
) -> pd.DataFrame:
    raw = pd.read_excel(input_path, sheet_name="demo")
    if not _is_numeric_test_format(raw):
        raise ValueError(
            "y_Ridge_test.xlsx 的 demo 页应为 16 列数字列（0-15），"
            "对应 11 个共有特征 + 5 个 LAD 特征。"
        )

    frame = raw.copy()
    frame.columns = TEST_FILE_COLUMN_NAMES
    return _order_test_columns(_predict_frame(frame, overall_service, branch_services))


def _detect_workbook_kind(input_path: str) -> str:
    workbook = pd.ExcelFile(input_path)
    if "demo0" in workbook.sheet_names and "demo1" in workbook.sheet_names:
        return "training"
    if "demo" in workbook.sheet_names:
        sample = pd.read_excel(input_path, sheet_name="demo")
        if _is_numeric_test_format(sample):
            return "test"
    raise ValueError(f"无法识别输入文件结构: {input_path}")


def run(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> str:
    kind = _detect_workbook_kind(input_path)

    overall_service = PredictionService.from_shelve(
        os.path.join(MODEL_DIR, "y_Ridge.dat"),
        feature_names=OVERALL_FEATURE_NAMES,
    )
    branch_services = load_branch_services()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if kind == "training":
            sheet_frames = _predict_training_workbook(
                input_path,
                overall_service,
                branch_services,
            )
            all_frame = pd.concat(
                [sheet_frames["demo0"], sheet_frames["demo1"]],
                ignore_index=True,
            )
            all_frame.to_excel(writer, sheet_name="全部", index=False)
            sheet_frames["demo0"].to_excel(writer, sheet_name="训练集_demo0", index=False)
            sheet_frames["demo1"].to_excel(writer, sheet_name="测试集_demo1", index=False)
        else:
            result = _predict_test_workbook(
                input_path,
                overall_service,
                branch_services,
            )
            result.to_excel(writer, sheet_name="全部", index=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = run(args.input, args.output)
    print(f"已输出: {output_path}")


if __name__ == "__main__":
    main()
