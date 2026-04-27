from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard.config import NeuralHazardConfig
from cas.hazard_behavior.neural_lowlevel import (
    FittedFormulaModel,
    fit_event_neural_comparisons,
    plot_neural_lowlevel_model_comparison,
)


class _DummyResult:
    def __init__(self, *, aic: float, bic: float, llf: float, n_params: int, converged: bool = True) -> None:
        self.aic = float(aic)
        self.bic = float(bic)
        self.llf = float(llf)
        self.converged = bool(converged)
        self.params = pd.Series(np.zeros(n_params, dtype=float))
        self.bse = pd.Series(np.ones(n_params, dtype=float))
        self.tvalues = pd.Series(np.zeros(n_params, dtype=float))
        self.pvalues = pd.Series(np.ones(n_params, dtype=float))

    def conf_int(self) -> pd.DataFrame:
        n_params = len(self.params)
        return pd.DataFrame({0: np.full(n_params, -0.1, dtype=float), 1: np.full(n_params, 0.1, dtype=float)})


def _base_table() -> pd.DataFrame:
    n_rows = 12
    table = pd.DataFrame(
        {
            "event_fpp": [0, 1] * 6,
            "event_spp": [0, 1] * 6,
            "episode_id": [f"ep-{idx}" for idx in range(n_rows)],
            "dyad_id": ["d1"] * n_rows,
            "participant_id": [f"p-{idx%2}" for idx in range(n_rows)],
            "time_from_partner_onset": np.linspace(0.05, 0.60, n_rows),
            "time_from_partner_offset": np.linspace(0.10, 0.65, n_rows),
            "z_information_rate_lag_150ms": np.linspace(-1.5, 1.5, n_rows),
            "z_prop_expected_cumulative_info_lag_700ms": np.linspace(1.2, -1.2, n_rows),
            "z_alpha_pc1": np.linspace(-0.5, 0.5, n_rows),
            "z_beta_pc1": np.linspace(0.3, -0.3, n_rows),
        }
    )
    table.loc[[1, 4], "z_alpha_pc1"] = np.nan
    table.loc[[2, 5, 8], "z_beta_pc1"] = np.nan
    return table


def _successful_stub(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    event_column: str,
) -> FittedFormulaModel:
    del formula
    is_parent = "_sample" in model_name
    result = _DummyResult(
        aic=100.0 if is_parent else 95.0,
        bic=110.0 if is_parent else 104.0,
        llf=-50.0 if is_parent else -46.0,
        n_params=4 if is_parent else 5,
    )
    return FittedFormulaModel(
        model_name=model_name,
        formula="",
        result=result,
        n_rows=int(len(riskset_table)),
        n_events=int(riskset_table[event_column].sum()),
        n_predictors=int(len(result.params)),
        converged=True,
        fit_warnings=[],
        error_message=None,
    )


def test_same_row_parent_refit_uses_child_complete_rows() -> None:
    table = _base_table()
    observed_sizes: dict[str, int] = {}

    def recording_stub(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        family = "alpha_beta" if "alpha_beta" in kwargs["model_name"] else ("alpha" if "_alpha_" in kwargs["model_name"] else "beta")
        observed_sizes[kwargs["model_name"]] = int(len(kwargs["riskset_table"]))
        return _successful_stub(**kwargs)

    comparison, _coeff, counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="fpp",
        neural_config=NeuralHazardConfig(enabled=True),
        fit_model_fn=recording_stub,
    )
    alpha_rows = counts["alpha"]["n_rows"]
    beta_rows = counts["beta"]["n_rows"]
    alpha_beta_rows = counts["alpha_beta"]["n_rows"]
    assert observed_sizes["M_behaviour_FPP_alpha_sample"] == alpha_rows
    assert observed_sizes["M_alpha_FPP"] == alpha_rows
    assert observed_sizes["M_behaviour_FPP_beta_sample"] == beta_rows
    assert observed_sizes["M_beta_FPP"] == beta_rows
    assert observed_sizes["M_behaviour_FPP_alpha_beta_sample"] == alpha_beta_rows
    assert observed_sizes["M_alpha_beta_FPP"] == alpha_beta_rows
    assert (comparison["n_rows_parent"] == comparison["n_rows_child"]).all()


def test_invalid_comparison_blocked_when_row_counts_differ() -> None:
    table = _base_table()

    def mismatched_stub(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        fitted = _successful_stub(**kwargs)
        if "_sample" in kwargs["model_name"]:
            return replace(fitted, n_rows=fitted.n_rows + 1, n_events=max(0, fitted.n_events - 1))
        return fitted

    comparison, _coeff, _counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="fpp",
        neural_config=NeuralHazardConfig(enabled=True),
        fit_model_fn=mismatched_stub,
    )
    assert (~comparison["same_rows"].astype(bool)).all()
    assert (~comparison["comparison_valid"].astype(bool)).all()
    assert comparison["delta_aic"].isna().all()
    assert comparison["delta_bic"].isna().all()


def test_delta_convention_child_minus_parent() -> None:
    table = _base_table()
    comparison, _coeff, _counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="fpp",
        neural_config=NeuralHazardConfig(enabled=True),
        fit_model_fn=_successful_stub,
    )
    assert (comparison["delta_aic"] == (comparison["child_aic"] - comparison["parent_aic"])).all()
    assert (comparison["delta_bic"] == (comparison["child_bic"] - comparison["parent_bic"])).all()


def test_fpp_valid_comparison_marks_valid_and_same_rows() -> None:
    table = _base_table()
    comparison, _coeff, _counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="fpp",
        neural_config=NeuralHazardConfig(enabled=True),
        fit_model_fn=_successful_stub,
    )
    assert comparison["same_rows"].astype(bool).all()
    assert comparison["comparison_valid"].astype(bool).all()
    assert (comparison["status"].astype(str) == "ok").all()


def test_spp_convergence_failure_row_recorded_without_crash() -> None:
    table = _base_table()

    def failing_spp_stub(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        if kwargs["model_name"].startswith("M_") and kwargs["model_name"].endswith("_SPP"):
            return FittedFormulaModel(
                model_name=kwargs["model_name"],
                formula=kwargs["formula"],
                result=None,
                n_rows=int(len(kwargs["riskset_table"])),
                n_events=int(kwargs["riskset_table"][kwargs["event_column"]].sum()),
                n_predictors=0,
                converged=False,
                fit_warnings=["overflow encountered in exp"],
                error_message="singular matrix",
            )
        return _successful_stub(**kwargs)

    comparison, _coeff, _counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="spp",
        neural_config=NeuralHazardConfig(enabled=True),
        fit_model_fn=failing_spp_stub,
    )
    assert (comparison["event_type"].astype(str) == "spp").all()
    assert (comparison["status"].astype(str) == "failed_convergence").all()
    assert (~comparison["comparison_valid"].astype(bool)).all()
    assert comparison["error_message"].astype(str).str.len().gt(0).all()


def test_plot_filters_invalid_rows(tmp_path: Path) -> None:
    comparison = pd.DataFrame(
        [
            {"event_type": "fpp", "neural_family": "alpha", "comparison_valid": True, "delta_bic": -10.0, "delta_aic": -8.0},
            {"event_type": "fpp", "neural_family": "beta", "comparison_valid": True, "delta_bic": -12.0, "delta_aic": -9.0},
            {"event_type": "fpp", "neural_family": "alpha_beta", "comparison_valid": True, "delta_bic": -13.0, "delta_aic": -9.5},
            {"event_type": "spp", "neural_family": "alpha", "comparison_valid": False, "delta_bic": -999.0, "delta_aic": -999.0},
        ]
    )
    plotted = plot_neural_lowlevel_model_comparison(comparison, tmp_path)
    assert (plotted["comparison_valid"].astype(bool)).all()
    assert (plotted["event_type"].astype(str) == "fpp").all()
    assert (tmp_path / "neural_lowlevel_model_comparison.png").exists()
