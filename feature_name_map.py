"""Feature display aliases loaded from name_2024.01.01.xlsx."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NAME_MAP_PATH = BASE_DIR / "name_2024.01.01.xlsx"


def load_feature_aliases(xlsx_path: os.PathLike | str | None = None) -> dict[str, str]:
    """Return internal feature name -> standardized alias (e.g. FB1-M)."""
    path = Path(xlsx_path or DEFAULT_NAME_MAP_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Feature name map not found: {path}")

    frame = pd.read_excel(path, header=None)
    aliases: dict[str, str] = {}
    for _, row in frame.iterrows():
        internal_name, alias = row[0], row[1]
        if pd.isna(internal_name) or pd.isna(alias):
            continue
        internal_name = str(internal_name).strip()
        alias = str(alias).strip()
        if internal_name in {"Y", "Y1", "Y2", "Y3", "特征"}:
            continue
        aliases[internal_name] = alias
    return aliases


FEATURE_ALIASES = load_feature_aliases()


def feature_alias(internal_name: str) -> str:
    return FEATURE_ALIASES.get(internal_name, internal_name)
