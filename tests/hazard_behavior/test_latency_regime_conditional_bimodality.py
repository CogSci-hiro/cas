from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.conditional_bimodality_diagnostics import (
    compare_p_late_with_r_predictions,
    compute_fixed_predictor_conditional_densities,
    compute_ppc_by_predictor_quantile,
    compute_r_model_residuals,
    run_latency_regime_conditional_bimodality_diagnostics,
    simulate_counterfactual_predictor_distributions,
)
from cas.hazard_behavior.plot_latency_regime import MODEL_C, MODEL_R1, MODEL_R2


def test_fixed_predictor_conditional_density_outputs_per_ms_area(tmp_path: Path) -> None:
    event_data, stan_dir = _write_bimodality_inputs(tmp_path)
    conditional = compute_fixed_predictor_conditional_densities(
        event_data=event_data,
        component_parameters=pd.read_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv"),
        gating_coefficients=pd.read_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv"),
        regression_coefficients=pd.read_csv(stan_dir / "behaviour_latency_regime_regression_coefficients.csv"),
        shifted_lognormal_diagnostics=None,
        best_student_t_r=MODEL_R1,
        best_lognormal_r=None,
        fit_metrics=json.loads((stan_dir / "behaviour_latency_regime_fit_metrics.json").read_text(encoding="utf-8")),
        nu=4.0,
        conditions=[{"condition_name": "mean_predictors", "z_information_rate_lag_best": 0.0, "z_prop_expected_cumulative_info_lag_best": 0.0}],
    )
    subset = conditional.loc[
        (conditional["model_name"] == MODEL_R1) & (conditional["condition_name"] == "mean_predictors")
    ].copy()
    area = np.trapezoid(subset["density_per_ms"].to_numpy(dtype=float), subset["latency_ms"].to_numpy(dtype=float))
    assert area == pytest.approx(1.0, abs=0.03)


def test_ppc_by_quantile_preserves_counts_and_negative_latencies(tmp_path: Path) -> None:
    event_data, stan_dir = _write_bimodality_inputs(tmp_path)
    density = compute_ppc_by_predictor_quantile(
        event_data=event_data,
        ppc_table=pd.read_csv(stan_dir / "behaviour_latency_regime_posterior_predictive.csv"),
        best_student_t_r=MODEL_R1,
        best_lognormal_r=None,
    )
    observed = density.loc[
        density["predictor_name"] == "z_information_rate_lag_best"
    ].copy()
    summary = observed.loc[observed["model_name"] == "observed", ["quantile_bin", "n_events"]].drop_duplicates()
    assert int(summary["n_events"].sum()) == len(event_data)
    assert float(observed["latency_ms"].min()) < 0.0


def test_r_model_residual_computation_matches_simple_mock_data() -> None:
    event_data = pd.DataFrame(
        {
            "row_index": [0, 1],
            "episode_id": ["e0", "e1"],
            "latency_from_partner_offset": [0.20, 0.50],
            "z_information_rate_lag_best": [0.0, 1.0],
            "z_prop_expected_cumulative_info_lag_best": [0.0, 0.0],
            "run": ["1", "1"],
            "time_within_run": [0.0, 0.1],
        }
    )
    predictions = pd.DataFrame(
        {
            "row_index": [0, 1],
            "model_name": [MODEL_R2, MODEL_R2],
            "predicted_mu_s": [0.10, 0.30],
            "predicted_sigma_s": [0.05, 0.10],
            "predicted_mu_log": [np.nan, np.nan],
            "predicted_sigma_log": [np.nan, np.nan],
            "shift_seconds": [np.nan, np.nan],
            "predicted_mean_s": [0.10, 0.30],
        }
    )
    residuals, summary = compute_r_model_residuals(event_data=event_data, regression_predictions=predictions)
    assert residuals["residual_s"].tolist() == pytest.approx([0.10, 0.20])
    assert residuals["standardised_residual"].tolist() == pytest.approx([2.0, 2.0])
    assert not summary.empty


def test_p_late_vs_r_prediction_merge_preserves_row_alignment(tmp_path: Path) -> None:
    event_data, stan_dir = _write_bimodality_inputs(tmp_path)
    regression_predictions = pd.DataFrame(
        {
            "row_index": event_data["row_index"],
            "model_name": [MODEL_R2] * len(event_data),
            "predicted_mu_s": np.linspace(0.05, 0.40, len(event_data)),
            "predicted_sigma_s": [0.08] * len(event_data),
            "predicted_mu_log": [np.nan] * len(event_data),
            "predicted_sigma_log": [np.nan] * len(event_data),
            "shift_seconds": [np.nan] * len(event_data),
            "predicted_mean_s": np.linspace(0.05, 0.40, len(event_data)),
        }
    )
    merged, _ = compare_p_late_with_r_predictions(
        event_data=event_data,
        event_probabilities=pd.read_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv"),
        regression_predictions=regression_predictions,
        r_model_name=MODEL_R2,
    )
    assert merged["row_index"].tolist() == event_data["row_index"].tolist()
    assert merged["r_mu_s"].tolist() == pytest.approx(regression_predictions["predicted_mu_s"].tolist())


def test_missing_optional_models_do_not_crash_diagnostics(tmp_path: Path) -> None:
    event_data, stan_dir = _write_bimodality_inputs(tmp_path, include_lognormal=False)
    event_csv = tmp_path / "behaviour_latency_regime_data.csv"
    event_data.drop(columns=["row_index"]).to_csv(event_csv, index=False)
    result = run_latency_regime_conditional_bimodality_diagnostics(
        event_data_csv=event_csv,
        stan_results_dir=stan_dir,
        figures_dir=tmp_path / "figures",
        diagnostics_dir=tmp_path / "diagnostics",
    )
    assert result.report_path.exists()
    assert (tmp_path / "diagnostics" / "latency_regime_p_late_vs_r_predictions.csv").exists()
    assert (tmp_path / "figures" / "behaviour_latency_regime_latency_vs_information_rate_coloured_by_p_late.png").exists()
    assert (tmp_path / "figures" / "behaviour_latency_regime_latency_vs_expected_cum_info_coloured_by_p_late.png").exists()
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "No shifted-lognormal regression model was available" in report_text


def test_counterfactual_fixed_predictor_simulation_replaces_predictors(tmp_path: Path) -> None:
    event_data, stan_dir = _write_bimodality_inputs(tmp_path)
    regression_predictions = pd.DataFrame(
        {
            "row_index": event_data["row_index"],
            "model_name": [MODEL_R2] * len(event_data),
            "predicted_mu_s": event_data["z_information_rate_lag_best"] * 0.20,
            "predicted_sigma_s": [0.08] * len(event_data),
            "predicted_mu_log": [np.nan] * len(event_data),
            "predicted_sigma_log": [np.nan] * len(event_data),
            "shift_seconds": [np.nan] * len(event_data),
            "predicted_mean_s": event_data["z_information_rate_lag_best"] * 0.20,
        }
    )
    density = simulate_counterfactual_predictor_distributions(
        event_data=event_data,
        component_parameters=pd.read_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv"),
        gating_coefficients=pd.read_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv"),
        event_probabilities=pd.read_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv"),
        regression_predictions=regression_predictions,
        regression_coefficients=pd.read_csv(stan_dir / "behaviour_latency_regime_regression_coefficients.csv"),
        shifted_lognormal_diagnostics=None,
        best_student_t_r=MODEL_R2,
        best_lognormal_r=None,
        fit_metrics=json.loads((stan_dir / "behaviour_latency_regime_fit_metrics.json").read_text(encoding="utf-8")),
        nu=4.0,
    )
    observed = density.loc[
        (density["model_name"] == MODEL_R2) & (density["scenario"] == "observed_predictors")
    ].copy()
    fixed = density.loc[
        (density["model_name"] == MODEL_R2) & (density["scenario"] == "fixed_predictors")
    ].copy()
    assert not np.allclose(observed["density_draw_mean"].to_numpy(dtype=float), fixed["density_draw_mean"].to_numpy(dtype=float))


def _write_bimodality_inputs(tmp_path: Path, *, include_lognormal: bool = True) -> tuple[pd.DataFrame, Path]:
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    event_data = pd.DataFrame(
        {
            "dyad_id": ["d1"] * 8,
            "run": ["1", "1", "1", "1", "2", "2", "2", "2"],
            "speaker": ["A"] * 8,
            "participant_speaker_id": ["d1_A"] * 8,
            "participant_speaker": ["A"] * 8,
            "episode_id": [f"ep-{i}" for i in range(8)],
            "event": [1] * 8,
            "latency_from_partner_offset": [-0.08, -0.01, 0.04, 0.10, 0.18, 0.26, 0.42, 0.95],
            "z_information_rate_lag_best": [-1.5, -1.0, -0.5, 0.0, 0.3, 0.8, 1.1, 1.5],
            "z_prop_expected_cumulative_info_lag_best": [-1.2, -0.9, -0.3, 0.0, 0.4, 0.9, 1.2, 1.4],
            "z_time_within_run": [-1.0, -0.7, -0.3, 0.0, 0.1, 0.3, 0.7, 1.0],
            "z_time_within_run_squared": [1.0, 0.49, 0.09, 0.0, 0.01, 0.09, 0.49, 1.0],
            "time_within_run": np.linspace(0.0, 20.0, 8),
        }
    ).reset_index(drop=True)
    event_data["row_index"] = np.arange(len(event_data), dtype=int)

    pd.DataFrame(
        {
            "model_name": [MODEL_C, MODEL_C, MODEL_C, MODEL_C],
            "component": ["early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma"],
            "mean": [0.02, 0.30, 0.06, 0.12],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)

    pd.DataFrame(
        {
            "model_name": [MODEL_C] * 5,
            "term": ["alpha", "beta_rate", "beta_expected", "gamma_time", "run_effect_run_2"],
            "mean": [0.0, 1.0, 0.8, 0.0, 0.3],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)

    probs = event_data[
        [
            "dyad_id",
            "run",
            "speaker",
            "participant_speaker_id",
            "participant_speaker",
            "episode_id",
            "latency_from_partner_offset",
            "z_information_rate_lag_best",
            "z_prop_expected_cumulative_info_lag_best",
            "z_time_within_run",
            "z_time_within_run_squared",
            "time_within_run",
        ]
    ].copy()
    probs["model_name"] = MODEL_C
    probs["p_late_mean"] = np.linspace(0.1, 0.95, len(probs))
    probs["p_late_q2_5"] = np.clip(probs["p_late_mean"] - 0.05, 0.0, 1.0)
    probs["p_late_q50"] = probs["p_late_mean"]
    probs["p_late_q97_5"] = np.clip(probs["p_late_mean"] + 0.05, 0.0, 1.0)
    probs.to_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv", index=False)

    regression_rows = [
        {"model_name": MODEL_R1, "coefficient_group": "location", "term": "alpha_mu", "mean": 0.10},
        {"model_name": MODEL_R1, "coefficient_group": "location", "term": "beta_mu_rate", "mean": 0.08},
        {"model_name": MODEL_R1, "coefficient_group": "location", "term": "beta_mu_expected", "mean": 0.03},
        {"model_name": MODEL_R2, "coefficient_group": "location", "term": "alpha_mu", "mean": 0.08},
        {"model_name": MODEL_R2, "coefficient_group": "location", "term": "beta_mu_rate", "mean": 0.10},
        {"model_name": MODEL_R2, "coefficient_group": "location", "term": "beta_mu_expected", "mean": 0.04},
        {"model_name": MODEL_R2, "coefficient_group": "scale", "term": "alpha_sigma", "mean": np.log(0.08)},
        {"model_name": MODEL_R2, "coefficient_group": "scale", "term": "beta_sigma_rate", "mean": 0.10},
    ]
    if include_lognormal:
        regression_rows.extend(
            [
                {"model_name": "model_r3_shifted_lognormal_location_regression", "coefficient_group": "location", "term": "alpha_log", "mean": -1.8},
                {"model_name": "model_r3_shifted_lognormal_location_regression", "coefficient_group": "location", "term": "beta_log_rate", "mean": 0.2},
            ]
        )
        pd.DataFrame(
            {
                "model_name": ["model_r3_shifted_lognormal_location_regression"],
                "shift_seconds": [-0.20],
            }
        ).to_csv(stan_dir / "behaviour_latency_regime_shifted_lognormal_diagnostics.csv", index=False)
    pd.DataFrame(regression_rows).to_csv(stan_dir / "behaviour_latency_regime_regression_coefficients.csv", index=False)

    loo_rows = [
        {"model_name": MODEL_C, "delta_looic_from_best": 0.0, "elpd_loo": -100.0},
        {"model_name": MODEL_R1, "delta_looic_from_best": 5.0, "elpd_loo": -102.5},
        {"model_name": MODEL_R2, "delta_looic_from_best": 2.0, "elpd_loo": -101.0},
    ]
    if include_lognormal:
        loo_rows.append({"model_name": "model_r3_shifted_lognormal_location_regression", "delta_looic_from_best": 4.0, "elpd_loo": -102.0})
    pd.DataFrame(loo_rows).to_csv(stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False)

    ppc_rows: list[dict[str, object]] = []
    for model_name, offset in ((MODEL_C, 0.0), (MODEL_R1, 0.02), (MODEL_R2, 0.01)):
        for draw_id in range(3):
            for value in event_data["latency_from_partner_offset"].to_list():
                ppc_rows.append(
                    {
                        "model_name": model_name,
                        "draw_id": draw_id,
                        "statistic": "y_rep",
                        "value": np.nan,
                        "y_rep_value": float(value + offset + 0.005 * draw_id),
                    }
                )
    pd.DataFrame(ppc_rows).to_csv(stan_dir / "behaviour_latency_regime_posterior_predictive.csv", index=False)

    fit_metrics = {
        "nu": 4.0,
        "controls": {
            "use_run_controls": True,
            "use_time_controls": True,
            "run_reference_level": "1",
        },
    }
    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text(json.dumps(fit_metrics), encoding="utf-8")
    return event_data, stan_dir
