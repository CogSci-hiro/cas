from __future__ import annotations

from pathlib import Path

from cas.behavior.config import load_behavior_hazard_config
from cas.behavior.formulas import FORMULA_REGISTRY, render_formula
from cas.behavior.lags import select_behavior_lag
from cas.behavior.pipeline import _family_lags, _r_final_model_specs, _r_table_model_specs


def test_behavior_formula_registry_contains_required_model_ids() -> None:
    required = {
        "A0_timing",
        "A1_information_rate",
        "A2_expected_cum_info",
        "A3_joint_information",
        "B1_shared_information",
        "B2_anchor_x_information",
        "C1_onset_x_rate",
        "C2_offset_x_rate",
        "D1_two_way_reference",
        "D2_three_way",
    }
    assert required <= set(FORMULA_REGISTRY)


def test_render_formula_uses_selected_lag_columns() -> None:
    formula = render_formula("A3_joint_information", lag_ms=300)
    assert "z_information_rate_lag_300" in formula
    assert "z_prop_expected_cum_info_lag_300" in formula


def test_formulas_include_subject_random_intercept_request() -> None:
    assert "(1 | subject)" in render_formula("A0_timing", lag_ms=0)


def test_behavior_config_restores_dense_legacy_lag_grid() -> None:
    config = load_behavior_hazard_config(Path("config/behavior/hazard.yaml"))
    assert config.candidate_lags_ms == [0, 50, 100, 150, 200, 250, 300, 400, 500]


def test_lag_selection_uses_max_delta_log_likelihood(monkeypatch) -> None:
    class _Fit:
        def __init__(self, llf: float):
            self.log_likelihood = llf

    values = {"A0_timing": -10.0, "A3_joint_information_lag_0": -9.5, "A3_joint_information_lag_100": -8.0, "A3_joint_information_lag_200": -8.5}

    def _fake_fit(table, *, model_id: str, formula: str):  # noqa: ARG001
        return _Fit(values[model_id])

    monkeypatch.setattr("cas.behavior.lags.fit_formula_model", _fake_fit)
    selected = select_behavior_lag(fpp_table=None, candidate_lags_ms=[0, 100, 200])  # type: ignore[arg-type]
    assert selected.selected_lag_ms == 100
    assert bool(selected.scores.loc[selected.scores["candidate_lag_ms"] == 100, "selected"].iloc[0]) is True


def test_lag_selection_falls_back_to_first_candidate_when_all_scores_are_nonfinite(monkeypatch) -> None:
    class _Fit:
        def __init__(self, llf: float):
            self.log_likelihood = llf

    def _fake_fit(table, *, model_id: str, formula: str):  # noqa: ARG001
        return _Fit(float("nan"))

    monkeypatch.setattr("cas.behavior.lags.fit_formula_model", _fake_fit)
    selected = select_behavior_lag(fpp_table=None, candidate_lags_ms=[0, 100, 200])  # type: ignore[arg-type]
    assert selected.selected_lag_ms == 0
    assert "selection_fallback" in selected.payload
    assert bool(selected.scores.loc[selected.scores["candidate_lag_ms"] == 0, "selected"].iloc[0]) is True


def test_family_lags_fallback_to_legacy_single_lag_payload() -> None:
    assert _family_lags({"selected_lag_ms": 150}) == {"A": 150, "B": 150, "C": 150}


def test_family_lags_use_family_payload_when_available() -> None:
    assert _family_lags({"family_lags": {"A": 150, "B": 200, "C": 250}}) == {"A": 150, "B": 200, "C": 250}


def test_final_model_specs_route_family_specific_lags() -> None:
    specs = _r_final_model_specs(None, {"A": 150, "B": 200, "C": 250})
    by_model = {str(spec["model_id"]): spec for spec in specs}
    assert by_model["A3_joint_information"]["lag_ms"] == 150
    assert by_model["B2_anchor_x_information"]["lag_ms"] == 200
    assert by_model["C1_onset_x_rate"]["lag_ms"] == 250
    assert "lag_150" in str(by_model["A3_joint_information"]["formula"])
    assert "lag_200" in str(by_model["B2_anchor_x_information"]["formula"])
    assert "lag_250" in str(by_model["C1_onset_x_rate"]["formula"])


def test_table_specs_keep_c_family_reference_internal_but_public_comparisons_clean() -> None:
    _model_specs, comparison_specs = _r_table_model_specs(None, {"A": 150, "B": 200, "C": 250})
    c_specs = [spec for spec in comparison_specs if spec["family"] == "C"]
    assert len(c_specs) == 2
    for spec in c_specs:
        assert spec["reduced"] == "A3_joint_information__C_family_reference"
        assert spec["public_reduced"] == "A3_joint_information"
        assert spec["lag_ms"] == 250
