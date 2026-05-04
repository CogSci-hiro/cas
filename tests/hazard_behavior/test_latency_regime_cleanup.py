from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.hazard_behavior.plot_latency_regime import (
    MODEL_LABELS,
    plot_behaviour_latency_regime_results,
    plot_gating_coefficients,
    plot_latency_regime_loo_comparison,
    plot_probability_by_predictor,
)


def test_active_stan_files_and_retired_constrained_location() -> None:
    assert Path("scripts/stan/behaviour_latency_one_student_t.stan").exists()
    assert Path("scripts/stan/behaviour_latency_skew_unimodal.stan").exists()
    assert Path("scripts/stan/behaviour_latency_two_student_t_mixture.stan").exists()
    assert Path("scripts/stan/behaviour_latency_mixture_of_experts.stan").exists()
    assert Path("scripts/stan/behaviour_latency_regression_student_t_location.stan").exists()
    assert Path("scripts/stan/behaviour_latency_regression_student_t_location_scale.stan").exists()
    assert Path("scripts/stan/behaviour_latency_regression_shifted_lognormal_location.stan").exists()
    assert Path("scripts/stan/behaviour_latency_regression_shifted_lognormal_location_scale.stan").exists()
    assert not Path("scripts/stan/behaviour_latency_constrained_mixture_of_experts.stan").exists()
    assert Path("legacy/behavior/scripts/stan/behaviour_latency_constrained_mixture_of_experts.stan").exists()
    active_r = Path("scripts/r/fit_behaviour_latency_regime_stan.R").read_text(encoding="utf-8")
    assert "model_r1_student_t_location_regression" in active_r
    assert "model_r2_student_t_location_scale_regression" in active_r
    assert "model_r3_shifted_lognormal_location_regression" in active_r
    assert "model_r4_shifted_lognormal_location_scale_regression" in active_r
    assert "model_d_constrained_mixture_of_experts" not in active_r
    assert "--fit-constrained-mixture" not in active_r


def test_loo_plot_filters_retired_model(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}
    original_bar = plt.Axes.bar

    def capture_bar(self, x, *args, **kwargs):
        captured["x"] = [str(v) for v in x]
        return original_bar(self, x, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "bar", capture_bar)

    loo = pd.DataFrame(
        {
            "model_name": [
                "model_a_one_student_t",
                "model_s_skew_unimodal",
                "model_b_two_student_t_mixture",
                "model_c_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
            ],
            "delta_looic_from_best": [7.0, 3.0, 2.0, 0.0, 1.0],
        }
    )
    plot_latency_regime_loo_comparison(loo_table=loo, output_path=tmp_path / "loo.png")
    labels = captured.get("x", [])
    assert "constrained mixture of experts" not in labels
    assert len(labels) == 4


def test_component_plot_and_diagnostics_use_model_c_only(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_inputs_with_c_and_d(tmp_path)
    out_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=out_dir,
    )

    density = pd.read_csv(tmp_path / "diagnostics" / "latency_regime_density_scaling_check.csv")
    mixture_row = density.loc[density["density_name"] == "mixture_density_event_averaged_per_ms"]
    assert not mixture_row.empty
    assert float(mixture_row["area_on_displayed_grid"].iloc[0]) > 0.5


def test_gating_coefficient_plot_uses_model_c_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, np.ndarray] = {}
    original_errorbar = plt.Axes.errorbar

    def capture_errorbar(self, *args, **kwargs):
        captured["x"] = np.asarray(kwargs.get("x"), dtype=float)
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "errorbar", capture_errorbar)

    table = pd.DataFrame(
        {
            "model_name": [
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
            ],
            "term": ["beta_rate", "beta_expected", "beta_rate", "beta_expected"],
            "q2_5": [0.1, -0.3, 9.0, 9.0],
            "q50": [0.4, -0.1, 10.0, 10.0],
            "q97_5": [0.7, 0.2, 11.0, 11.0],
        }
    )
    plot_gating_coefficients(gating_coefficients=table, output_path=tmp_path / "gating.png")
    assert np.all(captured["x"] < 1.0)


def test_event_probability_plot_uses_model_c_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, np.ndarray] = {}
    original_scatter = plt.Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        captured["x"] = np.asarray(x, dtype=float)
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", capture_scatter)

    table = pd.DataFrame(
        {
            "model_name": ["model_c_mixture_of_experts", "model_d_constrained_mixture_of_experts"],
            "z_information_rate_lag_best": [0.25, 5.0],
            "p_late_mean": [0.4, 0.9],
            "p_late_q2_5": [0.2, 0.8],
            "p_late_q97_5": [0.6, 1.0],
        }
    )
    plot_probability_by_predictor(
        event_probabilities=table,
        predictor_column="z_information_rate_lag_best",
        output_path=tmp_path / "prob.png",
        x_label="Information rate (z)",
        model_name="model_c_mixture_of_experts",
        title="test",
    )
    assert np.allclose(captured["x"], np.array([0.25]))


def test_no_constrained_outputs_generated(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_inputs_with_c_and_d(tmp_path)
    out_dir = tmp_path / "figures"
    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=out_dir,
    )
    generated = {path.name for path in out_dir.iterdir()}
    assert not any("constrained" in name for name in generated)


def test_legacy_inventory_mentions_retired_constrained_model() -> None:
    text = Path("legacy/behavior/CLEANUP_INVENTORY.md").read_text(encoding="utf-8").lower()
    assert "constrained latency-regime" in text


def test_hazard_pipeline_isolation_imports_without_latency_regime_artifacts() -> None:
    from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline  # noqa: F401

    assert "model_d_constrained_mixture_of_experts" not in MODEL_LABELS


def _write_fake_inputs_with_c_and_d(tmp_path: Path) -> tuple[Path, Path]:
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    event_csv = tmp_path / "events.csv"

    pd.DataFrame(
        {
            "latency_from_partner_offset": [-0.08, -0.01, 0.05, 0.20, 0.32, 0.40],
            "z_information_rate_lag_best": [-1.0, -0.5, -0.1, 0.3, 0.8, 1.1],
            "z_prop_expected_cumulative_info_lag_best": [-0.9, -0.4, -0.2, 0.2, 0.7, 1.0],
        }
    ).to_csv(event_csv, index=False)

    pd.DataFrame(
        {
            "model_name": [
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
            ],
            "component": ["early", "late", "early", "late", "early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma", "mu", "mu", "sigma", "sigma"],
            "mean": [0.04, 0.26, 0.07, 0.10, 0.90, 1.20, 0.04, 0.06],
            "sd": [0.01] * 8,
            "q2_5": [0.02, 0.22, 0.05, 0.07, 0.85, 1.15, 0.03, 0.05],
            "q50": [0.04, 0.26, 0.07, 0.10, 0.90, 1.20, 0.04, 0.06],
            "q97_5": [0.06, 0.30, 0.09, 0.13, 0.95, 1.25, 0.05, 0.07],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)

    pd.DataFrame(
        {
            "model_name": ["model_c_mixture_of_experts", "model_d_constrained_mixture_of_experts"],
            "term": ["alpha", "alpha"],
            "mean": [0.1, 3.0],
            "sd": [0.1, 0.1],
            "q2_5": [0.0, 2.8],
            "q50": [0.1, 3.0],
            "q97_5": [0.2, 3.2],
            "prob_gt_zero": [0.9, 1.0],
            "prob_lt_zero": [0.1, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)

    pd.DataFrame(
        {
            "model_name": [
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_d_constrained_mixture_of_experts",
            ],
            "latency_from_partner_offset": [0.02, 0.21, 1.5],
            "z_information_rate_lag_best": [-0.2, 0.4, 9.0],
            "z_prop_expected_cumulative_info_lag_best": [-0.1, 0.6, 9.0],
            "p_late_mean": [0.2, 0.7, 1.0],
            "p_late_q2_5": [0.1, 0.5, 1.0],
            "p_late_q50": [0.2, 0.7, 1.0],
            "p_late_q97_5": [0.3, 0.9, 1.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv", index=False)

    pd.DataFrame({"model_name": ["model_a_one_student_t"], "delta_looic_from_best": [0.0]}).to_csv(
        stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False
    )
    pd.DataFrame({"draw_id": [1, 1, 1], "y_rep_value": [-0.02, 0.06, 0.30]}).to_csv(
        stan_dir / "behaviour_latency_regime_posterior_predictive.csv", index=False
    )
    pd.DataFrame({"variable": ["xi", "omega", "alpha_skew"], "mean": [0.18, 0.12, 1.2]}).to_csv(
        stan_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv", index=False
    )
    pd.DataFrame({"variable": ["mu", "sigma"], "mean": [0.14, 0.18]}).to_csv(
        stan_dir / "behaviour_latency_model_a_one_student_t_summary.csv", index=False
    )
    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text('{"nu": 4}', encoding="utf-8")
    return stan_dir, event_csv
