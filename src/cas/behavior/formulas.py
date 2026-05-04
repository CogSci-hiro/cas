"""Central formula registry for behavioral hazard models."""

from __future__ import annotations

import re
from typing import Callable


CONTROL_TERMS = [
    "z_time_from_partner_onset_s",
    "z_time_from_partner_offset_s",
    "z_time_from_partner_offset_s_squared",
]


def _join_terms(*terms: str) -> str:
    return " + ".join(term for term in terms if term)


def _lag_rate(lag_ms: int) -> str:
    return f"z_information_rate_lag_{int(lag_ms)}"


def _lag_prop(lag_ms: int) -> str:
    return f"z_prop_expected_cum_info_lag_{int(lag_ms)}"


def _base_formula(rhs: str) -> str:
    return f"event ~ {rhs} + (1 | dyad_id) + (1 | subject)"


def _controls() -> str:
    return _join_terms(*CONTROL_TERMS)


def _A0_timing(lag_ms: int) -> str:
    del lag_ms
    return _base_formula(_controls())


def _A1_information_rate(lag_ms: int) -> str:
    return _base_formula(_join_terms(_controls(), _lag_rate(lag_ms)))


def _A2_expected_cum_info(lag_ms: int) -> str:
    return _base_formula(_join_terms(_controls(), _lag_prop(lag_ms)))


def _A3_joint_information(lag_ms: int) -> str:
    return _base_formula(_join_terms(_controls(), _lag_rate(lag_ms), _lag_prop(lag_ms)))


def _B1_shared_information(lag_ms: int) -> str:
    return _base_formula(_join_terms("anchor_type", _controls(), _lag_rate(lag_ms), _lag_prop(lag_ms)))


def _B2_anchor_x_information(lag_ms: int) -> str:
    return _base_formula(
        _join_terms(
            "anchor_type",
            _controls(),
            f"anchor_type * {_lag_rate(lag_ms)}",
            f"anchor_type * {_lag_prop(lag_ms)}",
        )
    )


def _C1_onset_x_rate(lag_ms: int) -> str:
    return _base_formula(
        _join_terms(
            _controls(),
            _lag_rate(lag_ms),
            _lag_prop(lag_ms),
            f"z_time_from_partner_onset_s:{_lag_rate(lag_ms)}",
        )
    )


def _C2_offset_x_rate(lag_ms: int) -> str:
    return _base_formula(
        _join_terms(
            _controls(),
            _lag_rate(lag_ms),
            _lag_prop(lag_ms),
            f"z_time_from_partner_offset_s:{_lag_rate(lag_ms)}",
        )
    )


def _D1_two_way_reference(lag_ms: int) -> str:
    return _base_formula(
        _join_terms(
            "anchor_type",
            _controls(),
            f"anchor_type * {_lag_rate(lag_ms)}",
            f"anchor_type * {_lag_prop(lag_ms)}",
            f"z_time_from_partner_onset_s:{_lag_rate(lag_ms)}",
        )
    )


def _D2_three_way(lag_ms: int) -> str:
    return _base_formula(
        _join_terms(
            "anchor_type",
            _controls(),
            f"anchor_type * {_lag_rate(lag_ms)}",
            f"anchor_type * {_lag_prop(lag_ms)}",
            f"z_time_from_partner_onset_s:{_lag_rate(lag_ms)}",
            f"anchor_type:z_time_from_partner_onset_s:{_lag_rate(lag_ms)}",
        )
    )


FORMULA_REGISTRY: dict[str, Callable[[int], str]] = {
    "A0_timing": _A0_timing,
    "A1_information_rate": _A1_information_rate,
    "A2_expected_cum_info": _A2_expected_cum_info,
    "A3_joint_information": _A3_joint_information,
    "B1_shared_information": _B1_shared_information,
    "B2_anchor_x_information": _B2_anchor_x_information,
    "C1_onset_x_rate": _C1_onset_x_rate,
    "C2_offset_x_rate": _C2_offset_x_rate,
    "D1_two_way_reference": _D1_two_way_reference,
    "D2_three_way": _D2_three_way,
}


def render_formula(model_id: str, *, lag_ms: int) -> str:
    if model_id not in FORMULA_REGISTRY:
        raise KeyError(f"Unknown behavioral hazard model id: {model_id}")
    return FORMULA_REGISTRY[model_id](int(lag_ms))


def strip_random_effects(formula: str) -> str:
    stripped = re.sub(r"\s*\+\s*\(1\s*\|\s*[^)]+\)", "", formula)
    stripped = re.sub(r"\s*\+\s*\(1\s*\|\s*[^)]+\)", "", stripped)
    return " ".join(stripped.split())
