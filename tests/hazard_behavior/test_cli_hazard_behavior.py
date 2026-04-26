from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from cas.cli.main import main


def test_plot_behaviour_hazard_results_command_creates_minimal_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r_results_dir = tmp_path / "models"
    models_dir = r_results_dir / "lag_selection"
    predictions_dir = r_results_dir / "predictions"
    figures_dir = tmp_path / "figures"
    qc_dir = tmp_path / "qc_plots" / "lag_selection"
    r_results_dir.mkdir()
    models_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "predictor_family": ["information_rate", "prop_expected_cumulative_info"],
            "lag_ms": [0, 300],
            "delta_bic": [-2.0, -3.0],
        }
    ).to_csv(models_dir / "behaviour_timing_control_lag_selection.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["information_rate", "prop_expected"],
            "lag_ms": [100, 300],
            "child_BIC": [120.0, 118.0],
            "delta_BIC": [-4.0, -2.0],
            "beta": [0.2, 0.1],
            "conf_low": [0.1, 0.02],
            "conf_high": [0.3, 0.18],
            "odds_ratio": [1.22, 1.11],
            "odds_ratio_conf_low": [1.10, 1.02],
            "odds_ratio_conf_high": [1.35, 1.20],
            "converged": [True, True],
        }
    ).to_csv(r_results_dir / "r_glmm_information_rate_lag_sweep.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["prop_expected"],
            "lag_ms": [300],
            "child_BIC": [117.0],
            "delta_BIC": [-1.0],
            "beta": [0.15],
            "conf_low": [0.05],
            "conf_high": [0.25],
            "odds_ratio": [1.16],
            "odds_ratio_conf_low": [1.05],
            "odds_ratio_conf_high": [1.28],
            "converged": [True],
        }
    ).to_csv(r_results_dir / "r_glmm_prop_expected_lag_sweep.csv", index=False)
    (r_results_dir / "r_glmm_selected_behaviour_lags.json").write_text(
        '{"best_information_rate_lag_ms": 100, "best_prop_expected_lag_ms": 300}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "child_model": ["M_final_glmm"],
            "delta_BIC": [-4.0],
            "delta_AIC": [-5.0],
        }
    ).to_csv(r_results_dir / "r_glmm_final_behaviour_model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "predictor": ["z_information_rate_lag_100ms"] * 3,
            "predictor_value": [-1, 0, 1],
            "predicted_probability": [0.1, 0.15, 0.2],
            "conf_low": [0.08, 0.12, 0.17],
            "conf_high": [0.12, 0.18, 0.24],
            "fixed_effect_only": [True, True, True],
        }
    ).to_csv(predictions_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "plot-behaviour-hazard-results",
            "--r-results-dir",
            str(r_results_dir),
            "--timing-control-models-dir",
            str(models_dir),
            "--qc-output-dir",
            str(qc_dir),
            "--output-dir",
            str(figures_dir),
        ],
    )

    assert main() == 0
    assert {path.name for path in figures_dir.iterdir()} == {
        "behaviour_r_glmm_delta_bic_by_lag.png",
        "behaviour_r_glmm_coefficient_by_lag.png",
        "behaviour_r_glmm_odds_ratio_by_lag.png",
        "behaviour_r_glmm_final_model_comparison.png",
        "behaviour_r_glmm_final_predicted_hazard_information_rate.png",
        "data",
    }
    assert {path.name for path in qc_dir.iterdir()} == {"behaviour_pooled_delta_bic_by_lag.png"}
