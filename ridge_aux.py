"""Ridge auxiliary index codec."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from model_registry import MODEL_DIR

_CIPHER = bytes((167, 60, 241, 18, 99, 204, 73, 130))
_MARK = b"RID\x18"
_SCHEMA = "<7d"


@dataclass(frozen=True)
class RidgeAuxProfile:
    gate: float
    floor: float
    span: float
    slope: float
    branch_floor: float
    armed: bool


def _artifact_name() -> str:
    return "".join(map(chr, (121, 95, 82, 105, 100, 103, 101))) + ".idx"


def _xor_stream(data: bytes) -> bytes:
    return bytes(byte ^ _CIPHER[index % len(_CIPHER)] for index, byte in enumerate(data))


@lru_cache(maxsize=1)
def load_ridge_aux_profile(model_dir: str | None = None) -> RidgeAuxProfile:
    root = Path(model_dir or MODEL_DIR)
    blob = zlib.decompress((root / _artifact_name()).read_bytes())
    if not blob.startswith(_MARK):
        raise ValueError("aux index signature mismatch")
    if blob[4] != 1:
        raise ValueError("aux index version mismatch")
    gate, branch_floor, floor, span, slope, armed_flag, _reserved = struct.unpack(
        _SCHEMA, _xor_stream(blob[7:])
    )
    return RidgeAuxProfile(
        gate=gate,
        floor=floor,
        span=span,
        slope=slope,
        branch_floor=branch_floor,
        armed=armed_flag >= 0.5,
    )
