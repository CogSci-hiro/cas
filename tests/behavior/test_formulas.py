from __future__ import annotations

from pathlib import Path

from cas.behavior.config import load_behavior_hazard_config
from cas.behavior.formulas import FORMULA_REGISTRY, RANDOM_EFFECT_TERMS, render_fixed_formula, render_formula
from cas.behavior.pipeline import _build_comparison_specs, _build_model_specs


def test_behavior_formula_registry_contains_required_model_ids() -> None:
    required = {"M_0", "M_1", "M_2", "M_3", "M_4", "M_pooled_main", "M_pooled_anchor_interaction"}
    assert required <= set(FORMULA_REGISTRY)


def test_render_formula_uses_selected_lag_columns() -> None:
    formula = render_formula("M_3", lag_ms=300, backend="glm")
    assert "z_information_rate_lag_300" in formula
    assert "z_prop_expected_cum_info_lag_300" in formula


def test_pooled_anchor_interaction_formula_contains_anchor_interactions() -> None:
    formula = render_fixed_formula("M_pooled_anchor_interaction", lag_ms=250)
    assert "anchor_type:z_time_from_partner_onset_s" in formula
    assert "anchor_type:z_time_from_partner_offset_s" in formula
    assert "anchor_type:z_time_from_partner_offset_s_squared" in formula
    assert "anchor_type:z_information_rate_lag_250" in formula
    assert "anchor_type:z_prop_expected_cum_info_lag_250" in formula


def test_glmm_formulas_include_random_effects() -> None:
    formula = render_formula("M_0", lag_ms=0, backend="glmm")
    assert "(1 | dyad_id)" in formula
    assert "(1 | subject)" in formula


def test_glm_formulas_exclude_random_effect_terms() -> None:
    formula = render_formula("M_3", lag_ms=150, backend="glm")
    assert "(1 |" not in formula


def test_glmm_and_glm_share_same_fixed_effect_terms() -> None:
    fixed_formula = render_fixed_formula("M_pooled_anchor_interaction", lag_ms=200)
    assert render_formula("M_pooled_anchor_interaction", lag_ms=200, backend="glm") == fixed_formula
    assert render_formula("M_pooled_anchor_interaction", lag_ms=200, backend="glmm") == fixed_formula + " + " + " + ".join(RANDOM_EFFECT_TERMS)


def test_behavior_config_uses_shared_lag_defaults() -> None:
    config = load_behavior_hazard_config(Path("config/behavior/hazard.yaml"))
    assert config.candidate_lags_ms == [0, 50, 100, 150, 200, 250, 300, 400, 500]
    assert config.model_backend == "glm"
    assert config.lag_selection_criterion == "bic"


def test_build_model_specs_routes_shared_lag_to_all_models() -> None:
    class _Config:
        model_backend = "glm"

    specs = _build_model_specs(_Config(), selected_lag_ms=150)
    by_anchor_and_model = {(str(spec["anchor_subset"]), str(spec["model_id"])): spec for spec in specs}
    assert by_anchor_and_model[("fpp", "M_3")]["formula_fixed"] == render_fixed_formula("M_3", lag_ms=150)
    assert by_anchor_and_model[("spp", "M_4")]["formula_fixed"] == render_fixed_formula("M_4", lag_ms=150)
    assert by_anchor_and_model[("pooled", "M_pooled_anchor_interaction")]["formula_fixed"] == render_fixed_formula("M_pooled_anchor_interaction", lag_ms=150)


def test_build_model_specs_respect_backend_formula_mode() -> None:
    class _GlmConfig:
        model_backend = "glm"

    class _GlmmConfig:
        model_backend = "glmm"

    glm_specs = _build_model_specs(_GlmConfig(), selected_lag_ms=200)
    glmm_specs = _build_model_specs(_GlmmConfig(), selected_lag_ms=200)
    glm_formula = next(spec["formula_full"] for spec in glm_specs if spec["model_id"] == "M_3" and spec["anchor_subset"] == "fpp")
    glmm_formula = next(spec["formula_full"] for spec in glmm_specs if spec["model_id"] == "M_3" and spec["anchor_subset"] == "fpp")
    assert "(1 |" not in str(glm_formula)
    assert "(1 | dyad_id)" in str(glmm_formula)


def test_comparison_specs_cover_nested_sequence() -> None:
    comparison_ids = {str(spec["comparison_id"]) for spec in _build_comparison_specs()}
    assert "fpp__M_0__vs__M_1" in comparison_ids
    assert "fpp__M_3__vs__M_4" in comparison_ids
    assert "spp__M_0__vs__M_2" in comparison_ids
    assert "pooled__M_pooled_main__vs__M_pooled_anchor_interaction" in comparison_ids
