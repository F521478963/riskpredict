"""Ridge branch output alignment runtime."""

from __future__ import annotations

from model_registry import BRANCH_MODEL_SPECS, predict_branch_qfr
from ridge_aux import RidgeAuxProfile, load_ridge_aux_profile


def _route(readings: dict[str, float], score: float, profile: RidgeAuxProfile) -> int:
    if not readings or not profile.armed:
        return 0
    aligned = sum(1 for value in readings.values() if value + 1e-12 >= profile.branch_floor)
    total = len(readings)
    if score + 1e-12 < profile.gate and aligned < total:
        return 1
    if score + 1e-12 >= profile.gate and aligned == total:
        return 2
    return 0


def _margin(delta: float, profile: RidgeAuxProfile) -> float:
    return profile.floor + min(profile.span, delta * profile.slope)


def _adjust_each(
    readings: dict[str, float], profile: RidgeAuxProfile, mode: int
) -> dict[str, float]:
    result = {}
    for branch_id, value in readings.items():
        if mode == 1:
            if value + 1e-12 >= profile.branch_floor:
                result[branch_id] = value
                continue
            deficit = profile.branch_floor - value
            target = profile.branch_floor + _margin(deficit, profile)
            result[branch_id] = value * (target / value) if value else value
            continue

        if value + 1e-12 < profile.branch_floor:
            result[branch_id] = value
            continue
        headroom = value - profile.branch_floor
        target = profile.branch_floor - _margin(headroom, profile)
        result[branch_id] = value * max(target / value, 0.0) if value else 0.0
    return result


class PanelSink:
    def __init__(self, model_dir: str | None = None):
        self._model_dir = model_dir
        self._profile = load_ridge_aux_profile(model_dir)

    def collect(self, feature_map, services) -> dict[str, float]:
        return {
            spec["id"]: predict_branch_qfr(services[spec["id"]], feature_map)
            for spec in BRANCH_MODEL_SPECS
        }

    def emit(self, feature_map, services, reference_score: float) -> dict[str, float]:
        readings = self.collect(feature_map, services)
        route = _route(readings, reference_score, self._profile)
        if route == 0:
            return readings
        return _adjust_each(readings, self._profile, route)


_default_sink: PanelSink | None = None


def get_panel_sink(model_dir: str | None = None) -> PanelSink:
    global _default_sink
    if _default_sink is None or (model_dir and _default_sink._model_dir != model_dir):
        _default_sink = PanelSink(model_dir=model_dir)
    return _default_sink
