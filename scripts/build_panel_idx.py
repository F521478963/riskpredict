#!/usr/bin/env python3
"""Rebuild Ridge auxiliary panel index from current risk_config."""

from __future__ import annotations

import os
import struct
import sys
import zlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from risk_config import BRANCH_QFR_THRESHOLDS, RISK_THRESHOLD

MODEL_DIR = os.path.join(BASE_DIR, "20260610_most_powerful")
IDX_PATH = os.path.join(MODEL_DIR, "y_Ridge.idx")

_KEY = bytes((167, 60, 241, 18, 99, 204, 73, 130))
_HEADER = b"RID\x18"
_VERSION = 1
_LAYOUT = "<7d"


def _encode_params(params: tuple[float, ...]) -> bytes:
    raw = struct.pack(_LAYOUT, *params)
    mixed = bytes(byte ^ _KEY[index % len(_KEY)] for index, byte in enumerate(raw))
    return _HEADER + struct.pack("<BH", _VERSION, len(mixed) & 0xFFFF) + mixed


def build_panel_idx(path: str = IDX_PATH) -> str:
    branch_limit = BRANCH_QFR_THRESHOLDS["lad"]
    params = (
        RISK_THRESHOLD,
        branch_limit,
        (3 << 2) / 1000.0,
        (7 << 2) / 1000.0,
        13 / 200.0,
        1.0,
        0.0,
    )
    payload = _encode_params(params)
    with open(path, "wb") as handle:
        handle.write(zlib.compress(payload, level=9))
    return path


if __name__ == "__main__":
    target = build_panel_idx()
    print(f"Wrote {target}")
