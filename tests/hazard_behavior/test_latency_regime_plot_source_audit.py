from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.plot_latency_regime import (
    MODEL_A,
    MODEL_B,
    MODEL_C,
    MODEL_R2,
    _build_plot_source_audit_summary,
    compute_model_b_density_diagnostic,
    compute_r2_density_diagnostic,
    count_local_maxima,
    plot_behaviour_latency_regime_results,
    select_exact_model_rows,
)


def test_exact_model_filter_returns_only_requested_rows() -> None:
    table = pd.DataFrame(
        {
            "model_name": [MODEL_B, MODEL_B, MODEL_R2, MODEL_R2],
            "term": ["mu", "sigma", "alpha_mu", "alpha_sigma"],
            "mean": [0.1, 0.2, 0.3, -1.0],
        }
    )

    model_b = select_exact_model_rows(table, model_name=MODEL_B, source_table="fake.csv")
    model_r2 = select_exact_model_rows(table, model_name=MODEL_R2, source_table="fake.csv")

    assert set(model_b["model_name"]) == {MODEL_B}
    assert set(model_r2["model_name"]) == {MODEL_R2}


def test_r2_conditional_density_is_unimodal() -> None:
    event_data = pd.DataFrame(
        {
            "latency_from_partner_offset": np.linspace(-0.1, 1.2, 12),
            "z_information_rate_lag_best": np.linspace(-1.0, 1.0, 12),
            "z_prop_expected_cumulative_info_lag_best": np.linspace(-1.0, 1.0, 12),
        }
    )
    coeffs = pd.DataFrame(
        {
            "model_name": [MODEL_R2] * 6,
            "coefficient_group": ["location", "location", "location", "scale", "scale", "scale"],
            "term": ["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            "mean": [0.18, 0.0, 0.0, np.log(0.10), 0.0, 0.0],
        }
    )
    grid_ms = np.linspace(-300.0, 1400.0, 2500)

    diagnostic = compute_r2_density_diagnostic(
        event_data=event_data,
        regression_coefficients=coeffs,
        model_name=MODEL_R2,
        grid_ms=grid_ms,
        nu=6.0,
    )

    assert diagnostic is not None
    maxima = count_local_maxima(grid_ms, diagnostic["r2_conditional_density_per_ms"].to_numpy(dtype=float))
    assert maxima == 1


def test_r2_marginal_density_can_be_bimodal() -> None:
    event_data = pd.DataFrame(
        {
            "latency_from_partner_offset": np.linspace(-0.1, 1.2, 8),
            "z_information_rate_lag_best": [-3.0, -2.8, -3.2, -2.9, 3.1, 2.8, 3.0, 3.2],
            "z_prop_expected_cumulative_info_lag_best": [0.0] * 8,
        }
    )
    coeffs = pd.DataFrame(
        {
            "model_name": [MODEL_R2] * 6,
            "coefficient_group": ["location", "location", "location", "scale", "scale", "scale"],
            "term": ["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            "mean": [0.45, 0.18, 0.0, np.log(0.045), 0.0, 0.0],
        }
    )
    grid_ms = np.linspace(-400.0, 1600.0, 4000)

    diagnostic = compute_r2_density_diagnostic(
        event_data=event_data,
        regression_coefficients=coeffs,
        model_name=MODEL_R2,
        grid_ms=grid_ms,
        nu=20.0,
    )

    assert diagnostic is not None
    maxima = count_local_maxima(grid_ms, diagnostic["r2_marginal_density_per_ms"].to_numpy(dtype=float), prominence_threshold=1e-8)
    assert maxima >= 2


def test_model_b_density_uses_model_b_rows_only() -> None:
    component_parameters = pd.DataFrame(
        {
            "model_name": [MODEL_B, MODEL_B, MODEL_B, MODEL_B, MODEL_C, MODEL_C, MODEL_C, MODEL_C],
            "component": ["early", "late", "early", "late", "early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma", "mu", "mu", "sigma", "sigma"],
            "mean": [0.00, 0.90, 0.05, 0.06, 10.0, 12.0, 1.0, 1.5],
        }
    )
    grid_ms = np.linspace(-400.0, 1600.0, 2000)

    diagnostic = compute_model_b_density_diagnostic(
        component_parameters=component_parameters,
        grid_ms=grid_ms,
        nu=8.0,
    )

    assert diagnostic is not None
    peak_latency_ms = float(diagnostic.loc[diagnostic["model_b_mixture_density_per_ms"].idxmax(), "latency_ms"])
    assert peak_latency_ms < 1000.0


def test_plot_source_audit_outputs_are_written(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_audit_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    outputs = plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    audit_dir = tmp_path / "diagnostics" / "plot_source_audit"
    assert (audit_dir / "latency_regime_plot_source_audit.csv").exists()
    assert (audit_dir / "latency_regime_plot_source_audit.json").exists()
    assert (audit_dir / "latency_regime_plot_source_audit_report.md").exists()
    assert (audit_dir / "latency_regime_r2_density_diagnostic.csv").exists()
    assert (audit_dir / "latency_regime_model_b_density_diagnostic.csv").exists()
    assert (audit_dir / "latency_regime_model_c_density_diagnostic.csv").exists()
    assert (audit_dir / "latency_regime_r2_conditional_vs_marginal_density.png").exists()
    assert (audit_dir / "latency_regime_model_b_component_check.png").exists()
    assert (audit_dir / "latency_regime_model_c_component_check.png").exists()
    assert (audit_dir / "latency_regime_model_overlay_audited.png").exists()
    assert outputs["plot_source_audit_csv"] == audit_dir / "latency_regime_plot_source_audit.csv"

    audit_table = pd.read_csv(audit_dir / "latency_regime_plot_source_audit.csv")
    assert {
        "figure_name",
        "curve_label",
        "intended_model_name",
        "actual_model_name_used",
        "source_file",
        "source_table",
        "n_source_rows_used",
        "parameter_names_used",
        "density_type",
        "predictor_values_used",
        "x_unit",
        "y_density_unit",
        "conversion_applied",
        "notes",
    } <= set(audit_table.columns)


def test_density_scaling_is_retained_in_r2_diagnostic() -> None:
    event_data = pd.DataFrame(
        {
            "latency_from_partner_offset": np.linspace(-0.1, 1.1, 10),
            "z_information_rate_lag_best": np.linspace(-0.6, 0.6, 10),
            "z_prop_expected_cumulative_info_lag_best": np.linspace(-0.5, 0.5, 10),
        }
    )
    coeffs = pd.DataFrame(
        {
            "model_name": [MODEL_R2] * 6,
            "coefficient_group": ["location", "location", "location", "scale", "scale", "scale"],
            "term": ["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            "mean": [0.20, 0.02, -0.01, np.log(0.12), 0.0, 0.0],
        }
    )
    grid_ms = np.linspace(-1000.0, 2000.0, 5000)

    diagnostic = compute_r2_density_diagnostic(
        event_data=event_data,
        regression_coefficients=coeffs,
        model_name=MODEL_R2,
        grid_ms=grid_ms,
        nu=8.0,
    )

    assert diagnostic is not None
    conditional_area = np.trapezoid(diagnostic["r2_conditional_density_per_ms"], diagnostic["latency_ms"])
    marginal_area = np.trapezoid(diagnostic["r2_marginal_density_per_ms"], diagnostic["latency_ms"])
    assert conditional_area == pytest.approx(1.0, abs=0.02)
    assert marginal_area == pytest.approx(1.0, abs=0.02)


def test_label_source_mismatch_is_flagged_in_summary() -> None:
    audit_df = pd.DataFrame(
        {
            "figure_name": ["demo.png"],
            "curve_label": ["R2 conditional"],
            "intended_model_name": [MODEL_R2],
            "actual_model_name_used": [MODEL_B],
        }
    )
    summary = _build_plot_source_audit_summary(
        stan_results_dir=Path("/tmp/demo"),
        audit_df=audit_df,
        r2_diag=None,
        model_b_diag=None,
        model_c_diag=None,
        maxima_payload={"r2_conditional": None, "r2_marginal": None, "model_b_mixture": None, "model_c_event_averaged": None},
        mismatches=audit_df.to_dict(orient="records"),
        best_student_t_r=MODEL_R2,
    )

    assert summary["answers"]["C_any_labels_swapped"] is True


def _write_fake_audit_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    event_csv = tmp_path / "behaviour_latency_regime_data.csv"

    event_data = pd.DataFrame(
        {
            "dyad_id": ["d1"] * 10,
            "run": ["1"] * 10,
            "speaker": ["A"] * 10,
            "participant_speaker_id": ["d1_A"] * 10,
            "participant_speaker": ["A"] * 10,
            "episode_id": [f"ep-{idx}" for idx in range(10)],
            "event": [1] * 10,
            "latency_from_partner_offset": [-0.08, -0.03, 0.02, 0.08, 0.16, 0.22, 0.30, 0.42, 0.92, 1.08],
            "z_information_rate_lag_best": [-2.0, -1.8, -1.5, -0.5, -0.2, 0.2, 0.5, 1.6, 1.8, 2.0],
            "z_prop_expected_cumulative_info_lag_best": [0.0, 0.1, -0.1, 0.0, -0.2, 0.2, 0.0, -0.1, 0.1, 0.0],
        }
    )
    event_data.to_csv(event_csv, index=False)

    pd.DataFrame(
        {
            "model_name": [MODEL_B, MODEL_B, MODEL_B, MODEL_B, MODEL_B, MODEL_C, MODEL_C, MODEL_C, MODEL_C],
            "component": ["early", "late", "early", "late", "late", "early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma", "pi_late", "mu", "mu", "sigma", "sigma"],
            "mean": [0.00, 0.95, 0.06, 0.08, 0.35, 0.02, 0.98, 0.07, 0.10],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)

    pd.DataFrame(
        {
            "model_name": [MODEL_C, MODEL_C, MODEL_C],
            "term": ["alpha", "beta_rate", "beta_expected"],
            "mean": [-0.2, 0.9, 0.0],
            "q2_5": [-0.4, 0.6, -0.2],
            "q50": [-0.2, 0.9, 0.0],
            "q97_5": [0.0, 1.2, 0.2],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)

    probs = event_data.loc[
        :,
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
        ],
    ].copy()
    probs["model_name"] = MODEL_C
    probs["p_late_mean"] = 1.0 / (1.0 + np.exp(-(0.2 + 1.2 * probs["z_information_rate_lag_best"])))
    probs["p_late_q2_5"] = np.clip(probs["p_late_mean"] - 0.08, 0.0, 1.0)
    probs["p_late_q50"] = probs["p_late_mean"]
    probs["p_late_q97_5"] = np.clip(probs["p_late_mean"] + 0.08, 0.0, 1.0)
    probs.to_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv", index=False)

    pd.DataFrame(
        {
            "model_name": [MODEL_R2] * 6,
            "coefficient_group": ["location", "location", "location", "scale", "scale", "scale"],
            "term": ["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            "mean": [0.45, 0.17, 0.0, np.log(0.05), 0.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_regression_coefficients.csv", index=False)

    ppc_rows: list[dict[str, object]] = []
    for draw_id in range(5):
        for model_name, base_values in [
            (MODEL_A, [-0.05, 0.05, 0.16, 0.28]),
            (MODEL_B, [-0.02, 0.02, 0.95, 1.00]),
            (MODEL_C, [0.00, 0.08, 0.25, 0.95]),
            (MODEL_R2, [0.00, 0.05, 0.85, 0.95]),
        ]:
            for value in base_values:
                ppc_rows.append(
                    {
                        "model_name": model_name,
                        "draw_id": draw_id,
                        "statistic": "y_rep",
                        "value": np.nan,
                        "y_rep_value": value + 0.01 * draw_id,
                    }
                )
    pd.DataFrame(ppc_rows).to_csv(stan_dir / "behaviour_latency_regime_posterior_predictive.csv", index=False)

    pd.DataFrame(
        {
            "model_name": [MODEL_A, MODEL_B, MODEL_C, MODEL_R2],
            "delta_looic_from_best": [12.0, 5.0, 1.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False)

    pd.DataFrame({"variable": ["xi", "omega", "alpha_skew"], "mean": [0.20, 0.13, 1.1]}).to_csv(
        stan_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv",
        index=False,
    )
    pd.DataFrame({"variable": ["mu", "sigma"], "mean": [0.25, 0.22]}).to_csv(
        stan_dir / "behaviour_latency_model_a_one_student_t_summary.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "model_name": [MODEL_R2] * 6,
            "predictor_name": ["z_information_rate_lag_best"] * 6,
            "predictor_value": np.linspace(-2.5, 2.5, 6),
            "latency_q10": np.linspace(0.05, 0.20, 6),
            "latency_q50": np.linspace(0.10, 0.55, 6),
            "latency_q90": np.linspace(0.20, 0.95, 6),
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_regression_predictions.csv", index=False)

    pd.DataFrame({"model_name": [], "shift_seconds": []}).to_csv(
        stan_dir / "behaviour_latency_regime_shifted_lognormal_diagnostics.csv",
        index=False,
    )
    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text(json.dumps({"nu": 8.0}), encoding="utf-8")
    return stan_dir, event_csv
