"""Central formula registry for the behavioral hazard pipeline."""

from __future__ import annotations

import re
from typing import Callable


TIMING_TERMS = [
    "z_time_from_partner_onset_s",
    "z_time_from_partner_offset_s",
    "z_time_from_partner_offset_s_squared",
]
RANDOM_EFFECT_TERMS = ("(1 | dyad_id)", "(1 | subject)")


def _join_terms(*terms: str) -> str:
    return " + ".join(term for term in terms if term)


def timing_terms() -> str:
    return _join_terms(*TIMING_TERMS)


def lag_rate_term(lag_ms: int) -> str:
    return f"z_information_rate_lag_{int(lag_ms)}"


def lag_prop_term(lag_ms: int) -> str:
    return f"z_prop_expected_cum_info_lag_{int(lag_ms)}"


def _base_formula(rhs: str) -> str:
    return f"event ~ {rhs}"


def _M_0(lag_ms: int) -> str:
    del lag_ms
    return _base_formula(timing_terms())


def _M_1(lag_ms: int) -> str:
    return _base_formula(_join_terms(timing_terms(), lag_rate_term(lag_ms)))


def _M_2(lag_ms: int) -> str:
    return _base_formula(_join_terms(timing_terms(), lag_prop_term(lag_ms)))


def _M_3(lag_ms: int) -> str:
    return _base_formula(_join_terms(timing_terms(), lag_rate_term(lag_ms), lag_prop_term(lag_ms)))


def _M_4(lag_ms: int) -> str:
    rate = lag_rate_term(lag_ms)
    return _base_formula(
        _join_terms(
            timing_terms(),
            rate,
            lag_prop_term(lag_ms),
            f"z_time_from_partner_onset_s:{rate}",
            f"z_time_from_partner_offset_s:{rate}",
        )
    )


def _M_pooled_main(lag_ms: int) -> str:
    return _base_formula(_join_terms("anchor_type", timing_terms(), lag_rate_term(lag_ms), lag_prop_term(lag_ms)))


def _M_pooled_anchor_interaction(lag_ms: int) -> str:
    rate = lag_rate_term(lag_ms)
    prop = lag_prop_term(lag_ms)
    return _base_formula(
        _join_terms(
            "anchor_type",
            timing_terms(),
            rate,
            prop,
            "anchor_type:z_time_from_partner_onset_s",
            "anchor_type:z_time_from_partner_offset_s",
            "anchor_type:z_time_from_partner_offset_s_squared",
            f"z_time_from_partner_onset_s:{rate}",
            f"z_time_from_partner_offset_s:{rate}",
            f"anchor_type:{rate}",
            f"anchor_type:{prop}",
        )
    )


FORMULA_REGISTRY: dict[str, Callable[[int], str]] = {
    "M_0": _M_0,
    "M_1": _M_1,
    "M_2": _M_2,
    "M_3": _M_3,
    "M_4": _M_4,
    "M_pooled_main": _M_pooled_main,
    "M_pooled_anchor_interaction": _M_pooled_anchor_interaction,
}


def add_random_effects(formula: str) -> str:
    return _join_terms(formula, *RANDOM_EFFECT_TERMS)


def render_fixed_formula(model_id: str, *, lag_ms: int) -> str:
    if model_id not in FORMULA_REGISTRY:
        raise KeyError(f"Unknown behavioral hazard model id: {model_id}")
    return FORMULA_REGISTRY[model_id](int(lag_ms))


def render_formula(model_id: str, *, lag_ms: int, backend: str = "glm") -> str:
    fixed_formula = render_fixed_formula(model_id, lag_ms=lag_ms)
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "glm":
        return fixed_formula
    if normalized_backend == "glmm":
        return add_random_effects(fixed_formula)
    raise ValueError(f"Unsupported behavioral backend: {backend!r}")


def strip_random_effects(formula: str) -> str:
    stripped = re.sub(r"\s*\+\s*\(1\s*\|\s*[^)]+\)", "", formula)
    stripped = re.sub(r"\s*\+\s*\(1\s*\|\s*[^)]+\)", "", stripped)
    return " ".join(stripped.split())


def formula_metadata(model_id: str, *, lag_ms: int) -> dict[str, object]:
    formula_fixed = render_fixed_formula(model_id, lag_ms=lag_ms)
    return {
        "formula_fixed": formula_fixed,
        "formula_full": add_random_effects(formula_fixed),
        "random_effects": list(RANDOM_EFFECT_TERMS),
    }
