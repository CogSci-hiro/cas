from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from cas.hazard_behavior.plot_latency_regime import (
    _evaluate_density_per_ms,
    check_density_area,
    mixture_density_per_ms,
    plot_behaviour_latency_regime_results,
    student_t_density_per_ms,
)


def test_student_t_density_unit_conversion_and_area() -> None:
    grid_s = np.linspace(-3.0, 3.0, 5000)
    grid_ms = grid_s * 1000.0
    mu_s = 0.2
    sigma_s = 0.35
    nu = 9.0

    density_s = t.pdf(grid_s, df=nu, loc=mu_s, scale=sigma_s)
    density_ms = student_t_density_per_ms(grid_ms, mu_s=mu_s, sigma_s=sigma_s, nu=nu)

    assert np.allclose(density_ms, density_s / 1000.0, rtol=1e-9, atol=1e-12)
    area = np.trapezoid(density_ms, grid_ms)
    assert area == pytest.approx(1.0, abs=0.01)


def test_density_helper_converts_ms_grid_to_seconds_before_evaluation() -> None:
    seen_grid: dict[str, np.ndarray] = {}

    def capture(grid_s: np.ndarray) -> np.ndarray:
        seen_grid["grid_s"] = np.asarray(grid_s, dtype=float)
        return np.ones_like(grid_s)

    grid_ms = np.array([-200.0, 0.0, 125.0, 800.0], dtype=float)
    density_ms = _evaluate_density_per_ms(grid_ms=grid_ms, density_in_seconds_fn=capture)

    assert np.allclose(seen_grid["grid_s"], grid_ms / 1000.0)
    assert np.allclose(density_ms, np.full_like(grid_ms, 0.001, dtype=float))


def test_mixture_density_conversion_matches_event_averaged_formula() -> None:
    grid_s = np.linspace(-0.4, 1.4, 1500)
    early_density_s = norm.pdf(grid_s, loc=0.05, scale=0.08)
    late_density_s = norm.pdf(grid_s, loc=0.85, scale=0.16)
    p_late = np.array([0.10, 0.35, 0.60, 0.90], dtype=float)

    expected = np.mean(
        (1.0 - p_late[:, None]) * early_density_s[None, :] + p_late[:, None] * late_density_s[None, :],
        axis=0,
    ) / 1000.0
    actual = mixture_density_per_ms(
        early_density_s=early_density_s,
        late_density_s=late_density_s,
        p_late_values=p_late,
    )

    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_check_density_area_flags_1000x_scaling_error() -> None:
    grid_s = np.linspace(-3.0, 3.0, 3000)
    grid_ms = grid_s * 1000.0
    good_density_ms = norm.pdf(grid_s, loc=0.0, scale=1.0) / 1000.0
    bad_density_ms = norm.pdf(grid_s, loc=0.0, scale=1.0)

    good = check_density_area(grid_ms, good_density_ms, "good")
    bad = check_density_area(grid_ms, bad_density_ms, "bad")

    assert bool(good["ok"])
    assert float(good["area"]) == pytest.approx(1.0, abs=0.02)
    assert not bool(bad["ok"])
    assert float(bad["area"]) > 100.0


def test_plot_smoke_writes_density_scaling_diagnostics(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    outputs = plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    assert (output_dir / "behaviour_latency_regime_components.png").exists()
    diagnostics_dir = tmp_path / "diagnostics"
    csv_path = diagnostics_dir / "latency_regime_density_scaling_check.csv"
    json_path = diagnostics_dir / "latency_regime_density_scaling_check.json"
    report_path = diagnostics_dir / "latency_regime_density_scaling_report.md"

    assert csv_path.exists()
    assert json_path.exists()
    assert report_path.exists()
    assert outputs["density_scaling_csv"] == csv_path
    assert outputs["density_scaling_json"] == json_path
    assert outputs["density_scaling_report"] == report_path

    density_table = pd.read_csv(csv_path)
    assert {
        "model_name",
        "density_name",
        "grid_min_ms",
        "grid_max_ms",
        "area_on_displayed_grid",
    } <= set(density_table.columns)
    assert "mixture_density_event_averaged_per_ms" in set(density_table["density_name"])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["conversion_applied"] is True
    for row in payload["area_checks"]:
        value = row.get("area_on_displayed_grid")
        if value is None:
            continue
        assert 0.0 < float(value) < 2.0



def _write_fake_latency_plot_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    event_csv = tmp_path / "behaviour_latency_regime_data.csv"

    pd.DataFrame(
        {
            "dyad_id": ["d1"] * 8,
            "run": ["1"] * 8,
            "speaker": ["A"] * 8,
            "participant_speaker_id": ["d1_A"] * 8,
            "participant_speaker": ["A"] * 8,
            "episode_id": [f"e{i}" for i in range(8)],
            "event": [1] * 8,
            "latency_from_partner_offset": [-0.08, -0.02, 0.03, 0.07, 0.15, 0.24, 0.35, 0.95],
            "z_information_rate_lag_best": np.linspace(-1.0, 1.0, 8),
            "z_prop_expected_cumulative_info_lag_best": np.linspace(-0.8, 1.2, 8),
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
            "mean": [0.00, 0.22, 0.08, 0.12],
            "sd": [0.02] * 4,
            "q2_5": [-0.03, 0.18, 0.05, 0.08],
            "q50": [0.00, 0.22, 0.08, 0.12],
            "q97_5": [0.03, 0.26, 0.11, 0.16],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)

    probs = pd.read_csv(event_csv).loc[
        :,
        [
            "dyad_id",
            "run",
            "speaker",
            "participant_speaker",
            "episode_id",
            "latency_from_partner_offset",
            "z_information_rate_lag_best",
            "z_prop_expected_cumulative_info_lag_best",
        ],
    ]
    probs_c = probs.copy()
    probs_c["model_name"] = "model_c_mixture_of_experts"
    probs_c["p_late_mean"] = np.linspace(0.1, 0.9, len(probs_c))
    probs_c["p_late_q2_5"] = np.clip(probs_c["p_late_mean"] - 0.08, 0.0, 1.0)
    probs_c["p_late_q50"] = probs_c["p_late_mean"]
    probs_c["p_late_q97_5"] = np.clip(probs_c["p_late_mean"] + 0.08, 0.0, 1.0)

    pd.concat([probs_c], ignore_index=True).to_csv(
        stan_dir / "behaviour_latency_regime_event_probabilities.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "model_name": ["model_c_mixture_of_experts"],
            "term": ["alpha"],
            "mean": [0.2],
            "sd": [0.1],
            "q2_5": [0.0],
            "q50": [0.2],
            "q97_5": [0.4],
            "prob_gt_zero": [0.95],
            "prob_lt_zero": [0.05],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)

    pd.DataFrame(
        {
            "variable": ["xi", "omega", "alpha_skew"],
            "mean": [0.18, 0.13, 1.1],
        }
    ).to_csv(stan_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv", index=False)

    pd.DataFrame(
        {
            "variable": ["mu", "sigma"],
            "mean": [0.16, 0.20],
        }
    ).to_csv(stan_dir / "behaviour_latency_model_a_one_student_t_summary.csv", index=False)

    ppc_rows: list[dict[str, object]] = []
    for draw_id in range(4):
        for value in (-0.06, -0.01, 0.06, 0.20, 0.30):
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

    pd.DataFrame(
        {
            "model_name": ["model_a_one_student_t", "model_c_mixture_of_experts"],
            "delta_looic_from_best": [6.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False)

    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text(json.dumps({"nu": 4.0}), encoding="utf-8")
    return stan_dir, event_csv
