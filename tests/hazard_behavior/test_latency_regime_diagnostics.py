from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.diagnose_latency_regime import (
    compute_density_check_table,
    compute_event_averaged_mixture_density,
    diagnose_behaviour_latency_regime,
    infer_latency_unit,
    summarize_p_late_table,
)


def test_infer_latency_unit_seconds_and_milliseconds() -> None:
    assert infer_latency_unit(np.array([0.12, 0.18, 0.25, 0.31])) == "seconds"
    assert infer_latency_unit(np.array([120.0, 180.0, 250.0, 310.0])) == "milliseconds"


def test_event_averaged_density_differs_from_fixed_half_when_mean_weight_differs() -> None:
    grid = np.array([-0.1, 0.0, 0.2, 1.0], dtype=float)
    event_averaged = compute_event_averaged_mixture_density(
        latency_grid_s=grid,
        p_late_values=np.array([0.1, 0.2, 0.3], dtype=float),
        mu_early=0.0,
        sigma_early=0.08,
        mu_late=1.0,
        sigma_late=0.1,
        nu=4.0,
    )
    fixed_half = compute_event_averaged_mixture_density(
        latency_grid_s=grid,
        p_late_values=np.array([0.5], dtype=float),
        mu_early=0.0,
        sigma_early=0.08,
        mu_late=1.0,
        sigma_late=0.1,
        nu=4.0,
    )
    assert not np.allclose(event_averaged, fixed_half)


def test_p_late_summary_contains_expected_overall_metrics() -> None:
    summary = summarize_p_late_table(
        pd.DataFrame(
            {
                "latency_from_partner_offset": [-0.05, 0.10, 0.25, 1.10],
                "p_late_mean": [0.01, 0.20, 0.70, 0.99],
            }
        )
    )
    overall = summary.loc[summary["summary_type"] == "overall"].set_index("metric")
    assert float(overall.loc["min", "value"]) == 0.01
    assert float(overall.loc["max", "value"]) == 0.99
    assert float(overall.loc["mean", "value"]) > 0.4


def test_diagnostics_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    event_csv, stan_dir = _write_fake_diagnostic_inputs(tmp_path, include_ppc=True)
    output_dir = tmp_path / "diagnostics"

    result = diagnose_behaviour_latency_regime(
        event_data_csv=event_csv,
        stan_results_dir=stan_dir,
        output_dir=output_dir,
    )

    assert result.report_path.exists()
    expected_paths = {
        "latency_regime_latency_summary.csv",
        "latency_regime_component_summary.csv",
        "latency_regime_gating_summary.csv",
        "latency_regime_p_late_summary.csv",
        "latency_regime_density_check.csv",
        "latency_regime_observed_histogram_raw.png",
        "latency_regime_observed_histogram_zoom.png",
        "latency_regime_component_overlay_check_full.png",
        "latency_regime_component_overlay_check_zoom.png",
        "latency_regime_posterior_predictive_check_full.png",
        "latency_regime_posterior_predictive_check_full_with_skew_unimodal.png",
        "latency_regime_posterior_predictive_check_zoom.png",
        "latency_regime_p_late_distribution.png",
        "latency_regime_diagnostic_report.md",
    }
    assert expected_paths <= {path.name for path in output_dir.iterdir()}


def test_diagnostics_missing_ppc_records_report_and_continues(tmp_path: Path) -> None:
    event_csv, stan_dir = _write_fake_diagnostic_inputs(tmp_path, include_ppc=False)
    output_dir = tmp_path / "diagnostics"

    result = diagnose_behaviour_latency_regime(
        event_data_csv=event_csv,
        stan_results_dir=stan_dir,
        output_dir=output_dir,
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "Posterior predictive file available: `False`." in report_text
    assert (output_dir / "latency_regime_posterior_predictive_check_full.png").exists()
    assert (output_dir / "latency_regime_posterior_predictive_check_zoom.png").exists()


def test_density_check_table_uses_available_inputs(tmp_path: Path) -> None:
    event_csv, stan_dir = _write_fake_diagnostic_inputs(tmp_path, include_ppc=True)
    event_data = pd.read_csv(event_csv)
    density = compute_density_check_table(
        latency=event_data["latency_from_partner_offset"].to_numpy(dtype=float),
        component_parameters=pd.read_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv"),
        model_c_summary=pd.read_csv(stan_dir / "behaviour_latency_model_c_mixture_of_experts_summary.csv"),
        gating_coefficients=pd.read_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv"),
        event_probabilities=pd.read_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv"),
        fit_metrics=json.loads((stan_dir / "behaviour_latency_regime_fit_metrics.json").read_text(encoding="utf-8")),
    )
    assert {
        "latency_s",
        "latency_ms",
        "early_component_density",
        "late_component_density",
        "mixture_density_event_averaged",
        "mixture_density_alpha_only",
        "mixture_density_50_50",
    } <= set(density.columns)


def _write_fake_diagnostic_inputs(tmp_path: Path, *, include_ppc: bool) -> tuple[Path, Path]:
    event_csv = tmp_path / "behaviour_latency_regime_data.csv"
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-003"],
            "run": ["1", "1", "2", "1"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["dyad-001_A", "dyad-001_B", "dyad-002_A", "dyad-003_B"],
            "participant_speaker": ["A", "B", "A", "B"],
            "episode_id": ["ep-1", "ep-2", "ep-3", "ep-4"],
            "event": [1, 1, 1, 1],
            "latency_from_partner_offset": [-0.08, 0.18, 0.24, 1.05],
            "z_information_rate_lag_best": [-0.8, -0.2, 0.4, 0.8],
            "z_prop_expected_cumulative_info_lag_best": [-1.0, -0.2, 0.5, 1.0],
        }
    ).to_csv(event_csv, index=False)
    pd.DataFrame(
        {
            "model_name": [
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
            ],
            "component": ["early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma"],
            "mean": [0.12, 1.00, 0.08, 0.20],
            "sd": [0.01, 0.02, 0.01, 0.02],
            "q2_5": [0.10, 0.96, 0.06, 0.16],
            "q50": [0.12, 1.00, 0.08, 0.20],
            "q97_5": [0.14, 1.04, 0.10, 0.24],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)
    pd.DataFrame(
        {
            "term": ["alpha", "beta_rate", "beta_expected"],
            "mean": [-2.0, -1.0, 2.5],
            "sd": [0.2, 0.2, 0.2],
            "q2_5": [-2.4, -1.4, 2.1],
            "q50": [-2.0, -1.0, 2.5],
            "q97_5": [-1.6, -0.6, 2.9],
            "prob_gt_zero": [0.0, 0.0, 1.0],
            "prob_lt_zero": [1.0, 1.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-003"],
            "run": ["1", "1", "2", "1"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["dyad-001_A", "dyad-001_B", "dyad-002_A", "dyad-003_B"],
            "participant_speaker": ["A", "B", "A", "B"],
            "episode_id": ["ep-1", "ep-2", "ep-3", "ep-4"],
            "latency_from_partner_offset": [-0.08, 0.18, 0.24, 1.05],
            "z_information_rate_lag_best": [-0.8, -0.2, 0.4, 0.8],
            "z_prop_expected_cumulative_info_lag_best": [-1.0, -0.2, 0.5, 1.0],
            "p_late_mean": [0.01, 0.08, 0.20, 0.98],
            "p_late_q2_5": [0.00, 0.02, 0.12, 0.90],
            "p_late_q50": [0.01, 0.07, 0.20, 0.99],
            "p_late_q97_5": [0.03, 0.15, 0.30, 1.00],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv", index=False)
    pd.DataFrame(
        {
            "model_name": ["model_a_one_student_t", "model_b_two_student_t_mixture", "model_c_mixture_of_experts"],
            "elpd_loo": [-100.0, -90.0, -70.0],
            "se_elpd_loo": [3.0, 3.0, 4.0],
            "p_loo": [2.0, 3.0, 4.0],
            "looic": [200.0, 180.0, 140.0],
            "se_looic": [6.0, 6.0, 8.0],
            "delta_elpd_from_best": [30.0, 20.0, 0.0],
            "delta_looic_from_best": [60.0, 40.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["mu[1]", "mu[2]", "sigma[1]", "sigma[2]", "alpha", "beta_rate", "beta_expected"],
            "mean": [0.12, 1.00, 0.08, 0.20, -2.0, -1.0, 2.5],
            "median": [0.12, 1.00, 0.08, 0.20, -2.0, -1.0, 2.5],
            "sd": [0.01, 0.02, 0.01, 0.02, 0.2, 0.2, 0.2],
            "mad": [0.01, 0.02, 0.01, 0.02, 0.2, 0.2, 0.2],
            "q5": [0.10, 0.96, 0.06, 0.16, -2.4, -1.4, 2.1],
            "q95": [0.14, 1.04, 0.10, 0.24, -1.6, -0.6, 2.9],
        }
    ).to_csv(stan_dir / "behaviour_latency_model_c_mixture_of_experts_summary.csv", index=False)
    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text(
        json.dumps({"nu": 4, "loo_available": True, "workflow_warnings": []}),
        encoding="utf-8",
    )
    if include_ppc:
        ppc_rows: list[dict[str, object]] = []
        for draw_id in range(5):
            for value in (-0.10, 0.10, 0.20, 1.00):
                ppc_rows.append(
                    {
                        "model_name": "model_c_mixture_of_experts",
                        "draw_id": draw_id,
                        "statistic": "y_rep",
                        "value": np.nan,
                        "y_rep_value": value + 0.01 * draw_id,
                    }
                )
        pd.DataFrame(ppc_rows).to_csv(stan_dir / "behaviour_latency_regime_posterior_predictive.csv", index=False)
    return event_csv, stan_dir
