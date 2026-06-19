"""Shared model and training-data loading for prediction and SHAP runtime."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import shelve


def load_shelve_model(model_path: Path):
    model_base = model_path.with_suffix("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with shelve.open(str(model_base), flag="r") as store:
            return store["clf"], store["ss"]


def load_model_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    train = pd.read_excel(data_path, sheet_name=0)
    test = pd.read_excel(data_path, sheet_name=1)
    outcome_col = train.columns[0]
    feature_cols = [col for col in train.columns[1:] if col in test.columns]
    frame = pd.concat([train, test], ignore_index=True)
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = frame[outcome_col].apply(pd.to_numeric, errors="coerce")
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), feature_cols
