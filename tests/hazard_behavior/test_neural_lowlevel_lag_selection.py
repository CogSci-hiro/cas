from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.cli.commands.hazard_fpp_tde_hmm import _parse_neural_lag_grid_ms
from cas.hazard.config import NeuralHazardConfig
from cas.hazard_behavior.neural_lowlevel import (
    FittedFormulaModel,
    NeuralLagWindow,
    compute_neural_empirical_p_value,
    compute_neural_lag_bounds,
    fit_event_neural_comparisons,
    plot_neural_lowlevel_delta_bic_by_lag,
    select_best_neural_lags,
    select_neural_lag_mask,
)


class _DummyResult:
    def __init__(self, *, aic: float, bic: float, llf: float, n_params: int) -> None:
        self.aic = float(aic)
        self.bic = float(bic)
        self.llf = float(llf)
        self.converged = True
        self.params = pd.Series(np.zeros(n_params, dtype=float))
        self.bse = pd.Series(np.ones(n_params, dtype=float))
        self.tvalues = pd.Series(np.zeros(n_params, dtype=float))
        self.pvalues = pd.Series(np.ones(n_params, dtype=float))

    def conf_int(self) -> pd.DataFrame:
        n_params = len(self.params)
        return pd.DataFrame({0: np.full(n_params, -0.1, dtype=float), 1: np.full(n_params, 0.1, dtype=float)})


def _base_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_fpp": [0, 1, 0, 1, 0, 1],
            "episode_id": [f"ep-{idx}" for idx in range(6)],
            "dyad_id": ["d1"] * 6,
            "run": ["1"] * 6,
            "participant_id": [f"p-{idx%2}" for idx in range(6)],
            "participant_speaker_id": [f"d1_{idx%2}" for idx in range(6)],
            "time_from_partner_onset": np.linspace(0.05, 0.30, 6),
            "time_from_partner_offset": np.linspace(0.10, 0.35, 6),
            "bin_end": np.linspace(0.20, 0.45, 6),
            "z_information_rate_lag_150ms": np.linspace(-1.0, 1.0, 6),
            "z_prop_expected_cumulative_info_lag_700ms": np.linspace(1.0, -1.0, 6),
            "z_alpha_pc1": np.linspace(-0.5, 0.5, 6),
            "z_beta_pc1": np.linspace(0.6, -0.6, 6),
        }
    )


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
        aic=100.0 if is_parent else 90.0,
        bic=120.0 if is_parent else 80.0,
        llf=-50.0 if is_parent else -44.0,
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
        row_ids=tuple(str(value) for value in riskset_table.index),
    )


def test_lag_window_sample_selection() -> None:
    window_start, window_end = compute_neural_lag_bounds(1.0, lag_start_ms=100, lag_end_ms=500)
    assert window_start == 0.5
    assert window_end == 0.9
    times = np.array([0.49, 0.50, 0.70, 0.90, 0.91])
    mask = select_neural_lag_mask(times, bin_end=1.0, lag_start_ms=100, lag_end_ms=500)
    assert mask.tolist() == [False, True, True, True, False]


def test_causality_check_for_default_lag_grid() -> None:
    times = np.array([0.1, 0.3, 0.5, 0.7, 0.95])
    for lag_start_ms, lag_end_ms in NeuralHazardConfig().neural_lag_grid_ms:
        assert lag_start_ms > 0
        assert lag_end_ms > lag_start_ms
        mask = select_neural_lag_mask(times, bin_end=1.0, lag_start_ms=lag_start_ms, lag_end_ms=lag_end_ms)
        selected = times[mask]
        if selected.size:
            assert np.all(selected <= (1.0 - (lag_start_ms / 1000.0)))


def test_lag_grid_parser() -> None:
    assert _parse_neural_lag_grid_ms("50-250,100-300,100-500") == ((50, 250), (100, 300), (100, 500))


def test_same_row_comparison_checks_row_ids() -> None:
    table = _base_table()

    def row_id_mismatch_stub(**kwargs) -> FittedFormulaModel:  # type: ignore[no-untyped-def]
        fitted = _successful_stub(**kwargs)
        if "_sample" in kwargs["model_name"]:
            return fitted
        return FittedFormulaModel(
            model_name=fitted.model_name,
            formula=fitted.formula,
            result=fitted.result,
            n_rows=fitted.n_rows,
            n_events=fitted.n_events,
            n_predictors=fitted.n_predictors,
            converged=fitted.converged,
            fit_warnings=fitted.fit_warnings,
            error_message=fitted.error_message,
            row_ids=tuple(f"{row_id}_child" for row_id in fitted.row_ids),
        )

    comparison, _coeff, _counts = fit_event_neural_comparisons(
        riskset_table=table,
        event_type="fpp",
        neural_config=NeuralHazardConfig(enabled=True),
        lag_window=NeuralLagWindow(100, 500),
        fit_model_fn=row_id_mismatch_stub,
    )
    assert (~comparison["same_rows"].astype(bool)).all()
    assert (~comparison["comparison_valid"].astype(bool)).all()


def test_best_lag_selection() -> None:
    comparison = pd.DataFrame(
        [
            {"event_type": "fpp", "neural_family": "alpha", "lag_start_ms": 50, "lag_end_ms": 250, "lag_label": "50_250ms", "comparison_valid": True, "delta_bic": -10.0, "delta_aic": -8.0},
            {"event_type": "fpp", "neural_family": "alpha", "lag_start_ms": 100, "lag_end_ms": 500, "lag_label": "100_500ms", "comparison_valid": True, "delta_bic": -40.0, "delta_aic": -12.0},
            {"event_type": "fpp", "neural_family": "alpha", "lag_start_ms": 300, "lag_end_ms": 500, "lag_label": "300_500ms", "comparison_valid": True, "delta_bic": 5.0, "delta_aic": 2.0},
        ]
    )
    selected = select_best_neural_lags(comparison)
    assert selected.iloc[0]["lag_label"] == "100_500ms"


def test_null_empirical_p_value_uses_lower_tail() -> None:
    assert compute_neural_empirical_p_value(-10.0, np.array([-5.0, -15.0, 0.0])) == 0.5


def test_spp_failure_rows_do_not_block_fpp_selection() -> None:
    comparison = pd.DataFrame(
        [
            {"event_type": "fpp", "neural_family": "beta", "lag_start_ms": 100, "lag_end_ms": 500, "lag_label": "100_500ms", "status": "ok", "comparison_valid": True, "delta_bic": -12.0, "delta_aic": -9.0},
            {"event_type": "spp", "neural_family": "beta", "lag_start_ms": 100, "lag_end_ms": 500, "lag_label": "100_500ms", "status": "failed_convergence", "comparison_valid": False, "delta_bic": np.nan, "delta_aic": np.nan},
        ]
    )
    selected = select_best_neural_lags(comparison)
    assert selected["event_type"].astype(str).tolist() == ["fpp"]
    assert selected["lag_label"].astype(str).tolist() == ["100_500ms"]


def test_plot_filters_invalid_comparisons_by_default(tmp_path: Path) -> None:
    comparison = pd.DataFrame(
        [
            {"event_type": "fpp", "neural_family": "alpha", "lag_start_ms": 50, "lag_end_ms": 250, "lag_label": "50_250ms", "comparison_valid": True, "delta_bic": -10.0, "delta_aic": -7.0},
            {"event_type": "fpp", "neural_family": "beta", "lag_start_ms": 100, "lag_end_ms": 500, "lag_label": "100_500ms", "comparison_valid": True, "delta_bic": -14.0, "delta_aic": -10.0},
            {"event_type": "spp", "neural_family": "alpha", "lag_start_ms": 100, "lag_end_ms": 500, "lag_label": "100_500ms", "comparison_valid": False, "delta_bic": -999.0, "delta_aic": -999.0},
        ]
    )
    selected = select_best_neural_lags(comparison)
    plot_neural_lowlevel_delta_bic_by_lag(comparison, selected, tmp_path)
    assert (tmp_path / "neural_lowlevel_delta_bic_by_lag.png").exists()
