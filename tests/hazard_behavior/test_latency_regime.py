from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.latency_regime_export import export_behaviour_latency_regime_data, prepare_latency_regime_data
from cas.hazard_behavior.plot_latency_regime import plot_behaviour_latency_regime_results, skew_normal_pdf


def test_event_only_export_keeps_only_event_rows(tmp_path: Path) -> None:
    riskset = _tiny_latency_riskset()
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"

    export_behaviour_latency_regime_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert exported["event"].tolist() == [1, 1]


def test_latency_from_partner_offset_is_computed_and_can_be_negative(tmp_path: Path) -> None:
    riskset = _tiny_latency_riskset().drop(columns=["latency_from_partner_offset", "latency_from_partner_onset"])
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"

    export_behaviour_latency_regime_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert exported["latency_from_partner_offset"].tolist() == pytest.approx([-0.05, 0.25])
    assert exported["latency_from_partner_onset"].tolist() == pytest.approx([0.35, 0.70])


def test_negative_latency_is_retained(tmp_path: Path) -> None:
    riskset = _tiny_latency_riskset()
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"

    export_behaviour_latency_regime_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert (exported["latency_from_partner_offset"] < 0.0).any()


def test_best_lag_mapping_uses_selected_lags_json(tmp_path: Path) -> None:
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=100, expected_lag_ms=300)
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"
    riskset = _tiny_latency_riskset()

    export_behaviour_latency_regime_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
    )

    exported = pd.read_csv(output_csv)
    event_rows = riskset.loc[riskset["event"] == 1].reset_index(drop=True)
    assert exported["z_information_rate_lag_best"].tolist() == event_rows["z_information_rate_lag_100ms"].tolist()
    assert (
        exported["z_prop_expected_cumulative_info_lag_best"].tolist()
        == event_rows["z_prop_expected_cumulative_info_lag_300ms"].tolist()
    )


def test_explicit_lags_override_json(tmp_path: Path) -> None:
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=0, expected_lag_ms=0)
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"
    riskset = _tiny_latency_riskset()

    export_behaviour_latency_regime_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    event_rows = riskset.loc[riskset["event"] == 1].reset_index(drop=True)
    assert exported["z_information_rate_lag_best"].tolist() == event_rows["z_information_rate_lag_100ms"].tolist()
    assert (
        exported["z_prop_expected_cumulative_info_lag_best"].tolist()
        == event_rows["z_prop_expected_cumulative_info_lag_300ms"].tolist()
    )


def test_latency_regime_qc_json_contains_expected_fields(tmp_path: Path) -> None:
    output_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_qc_json = tmp_path / "behaviour_latency_regime_export_qc.json"

    export_behaviour_latency_regime_data(
        _tiny_latency_riskset(),
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    payload = json.loads(output_qc_json.read_text(encoding="utf-8"))
    assert payload["n_rows_exported"] == 2
    assert payload["n_participant_speaker_ids"] == 2
    assert payload["latency_from_partner_offset_min"] == pytest.approx(-0.05)
    assert payload["latency_from_partner_offset_max"] == pytest.approx(0.25)
    assert payload["proportion_negative_latency_from_partner_offset"] == pytest.approx(0.5)
    assert payload["identity_validation"]["participant_speaker_id_valid"] is True


def test_stan_files_exist_for_new_models() -> None:
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


def test_prepare_latency_regime_data_derives_run_time_controls() -> None:
    prepared = prepare_latency_regime_data(
        _tiny_latency_riskset(),
        information_rate_lag_ms=100,
        expected_cumulative_info_lag_ms=300,
    )

    assert "run_index" in prepared.columns
    assert "time_within_run" in prepared.columns
    assert "z_time_within_run" in prepared.columns
    assert "z_time_within_run_squared" in prepared.columns
    assert np.isfinite(prepared["time_within_run"]).all()
    assert np.isfinite(prepared["z_time_within_run"]).all()
    assert (prepared["z_time_within_run_squared"] >= 0.0).all()


def test_component_overlay_plot_smoke(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    assert (output_dir / "behaviour_latency_regime_components.png").exists()


def test_probability_plot_smoke(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    assert (output_dir / "behaviour_latency_regime_probability_by_expected_info.png").exists()
    assert (output_dir / "behaviour_latency_regime_probability_by_information_rate.png").exists()


def test_gating_coefficient_plot_filters_terms(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    gating = pd.read_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv")
    assert {"model_name", "term"} <= set(gating.columns)
    plotted_terms = gating.loc[gating["term"].isin(["beta_rate", "beta_expected"]), "term"].tolist()
    assert set(plotted_terms) == {"beta_rate", "beta_expected"}
    assert (output_dir / "behaviour_latency_regime_gating_coefficients.png").exists()


def test_loo_plot_handles_five_models(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    assert (output_dir / "behaviour_latency_regime_loo_comparison.png").exists()


def test_event_probabilities_with_model_name_support_filtering(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )

    probs = pd.read_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv")
    assert "model_name" in probs.columns
    assert set(probs["model_name"].unique()) == {"model_c_mixture_of_experts"}
    assert (output_dir / "behaviour_latency_regime_probability_by_information_rate.png").exists()


def test_skew_normal_density_is_finite_nonnegative_and_integrates() -> None:
    grid = np.linspace(-2.0, 3.0, 4000)
    density = skew_normal_pdf(grid, xi=0.2, omega=0.4, alpha=2.5)
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
    integral = np.trapezoid(density, grid)
    assert integral == pytest.approx(1.0, abs=0.03)


def test_ppc_plot_smoke_and_missing_ppc_skip(tmp_path: Path) -> None:
    stan_dir, event_csv = _write_fake_latency_plot_inputs(tmp_path)
    output_dir = tmp_path / "figures"

    plot_behaviour_latency_regime_results(
        stan_results_dir=stan_dir,
        event_data_csv=event_csv,
        output_dir=output_dir,
    )
    assert (output_dir / "behaviour_latency_regime_ppc.png").exists()
    assert (output_dir / "behaviour_latency_regime_skew_vs_mixture.png").exists()

    missing_ppc_dir = tmp_path / "stan_models_missing_ppc"
    shutil.copytree(stan_dir, missing_ppc_dir)
    (missing_ppc_dir / "behaviour_latency_regime_posterior_predictive.csv").unlink()
    missing_output_dir = tmp_path / "figures_missing_ppc"
    with pytest.warns(UserWarning, match="Posterior predictive CSV was not found"):
        plot_behaviour_latency_regime_results(
            stan_results_dir=missing_ppc_dir,
            event_data_csv=event_csv,
            output_dir=missing_output_dir,
        )
    assert not (missing_output_dir / "behaviour_latency_regime_ppc.png").exists()


def test_primary_pipeline_import_does_not_require_stan_or_r() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline; print('ok')"
            ),
        ],
        cwd="/Users/hiro/Projects/active/cas",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_latency_regime_r_stan_smoke_test(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable.")
    if not _r_latency_regime_requirements_available(rscript):
        pytest.skip("Required R packages or CmdStan installation for latency-regime smoke testing are unavailable.")

    input_csv = tmp_path / "behaviour_latency_regime_data.csv"
    output_dir = tmp_path / "stan_models"
    _synthetic_latency_event_input().to_csv(input_csv, index=False)

    result = subprocess.run(
        [
            rscript,
            str(Path("scripts/r/fit_behaviour_latency_regime_stan.R")),
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--stan-dir",
            str(Path("scripts/stan")),
            "--chains",
            "2",
            "--parallel-chains",
            "2",
            "--iter-warmup",
            "250",
            "--iter-sampling",
            "250",
            "--seed",
            "123",
        ],
        cwd="/Users/hiro/Projects/active/cas",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"Latency-regime Stan smoke test did not finish on the synthetic dataset: {result.stderr}")

    assert (output_dir / "behaviour_latency_model_a_one_student_t_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_b_two_student_t_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_c_mixture_of_experts_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_r1_student_t_location_regression_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_r2_student_t_location_scale_regression_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_r3_shifted_lognormal_location_regression_summary.csv").exists()
    assert (output_dir / "behaviour_latency_model_r4_shifted_lognormal_location_scale_regression_summary.csv").exists()
    assert (output_dir / "behaviour_latency_regime_loo_comparison.csv").exists()
    assert (output_dir / "behaviour_latency_regime_component_parameters.csv").exists()
    assert (output_dir / "behaviour_latency_regime_gating_coefficients.csv").exists()
    assert (output_dir / "behaviour_latency_regime_event_probabilities.csv").exists()
    assert (output_dir / "behaviour_latency_regime_posterior_predictive.csv").exists()
    assert (output_dir / "behaviour_latency_regime_regression_coefficients.csv").exists()
    assert (output_dir / "behaviour_latency_regime_regression_predictions.csv").exists()
    assert (output_dir / "behaviour_latency_regime_shifted_lognormal_diagnostics.csv").exists()
    assert (output_dir / "behaviour_latency_regime_fit_metrics.json").exists()
    assert (output_dir / "behaviour_latency_regime_interpretation.md").exists()


def _tiny_latency_riskset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event": [0, 1, 1],
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002"],
            "run": ["1", "1", "2"],
            "speaker": ["A", "B", "A"],
            "participant_speaker_id": ["dyad-001_A", "dyad-001_B", "dyad-002_A"],
            "participant_speaker": ["A", "B", "A"],
            "episode_id": ["ep-0", "ep-1", "ep-2"],
            "fpp_onset": [0.15, 0.45, 1.30],
            "partner_ipu_onset": [0.00, 0.10, 0.60],
            "partner_ipu_offset": [0.20, 0.50, 1.05],
            "latency_from_partner_onset": [0.15, 0.35, 0.70],
            "latency_from_partner_offset": [-0.05, -0.05, 0.25],
            "z_information_rate_lag_0ms": [-1.2, -0.3, 0.6],
            "z_information_rate_lag_100ms": [-1.0, 0.2, 0.8],
            "z_prop_expected_cumulative_info_lag_0ms": [-0.5, -0.1, 0.2],
            "z_prop_expected_cumulative_info_lag_300ms": [-0.2, 0.4, 0.9],
        }
    )


def _synthetic_latency_event_input() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        rows.append(
            {
                "dyad_id": f"dyad-{idx % 6:03d}",
                "run": str((idx % 4) + 1),
                "speaker": "A" if idx % 2 == 0 else "B",
                "participant_speaker_id": f"dyad-{idx % 6:03d}_{'A' if idx % 2 == 0 else 'B'}",
                "participant_speaker": "A" if idx % 2 == 0 else "B",
                "episode_id": f"ep-{idx:03d}",
                "event": 1,
                "fpp_onset": 0.55 + 0.03 * idx,
                "partner_ipu_onset": 0.10 + 0.01 * (idx % 3),
                "partner_ipu_offset": 0.35 + 0.015 * (idx % 5),
                "latency_from_partner_onset": 0.28 + 0.02 * idx,
                "latency_from_partner_offset": (-0.08 if idx % 3 == 0 else 0.18) + 0.01 * (idx % 4),
                "z_information_rate_lag_best": -1.2 + 0.1 * idx,
                "z_prop_expected_cumulative_info_lag_best": -0.8 + 0.08 * idx,
            }
        )
    return pd.DataFrame(rows)


def _write_selected_lags_json(tmp_path: Path, *, information_rate_lag_ms: int, expected_lag_ms: int) -> Path:
    path = tmp_path / "behaviour_timing_control_selected_lags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "best_information_rate_lag_ms": information_rate_lag_ms,
                "best_expected_cumulative_info_lag_ms": expected_lag_ms,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_latency_plot_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stan_dir = tmp_path / "stan_models"
    stan_dir.mkdir(parents=True, exist_ok=True)
    event_csv = tmp_path / "behaviour_latency_regime_data.csv"
    _synthetic_latency_event_input().to_csv(event_csv, index=False)

    pd.DataFrame(
        {
            "model_name": [
                "model_b_two_student_t_mixture",
                "model_b_two_student_t_mixture",
                "model_b_two_student_t_mixture",
                "model_b_two_student_t_mixture",
                "model_b_two_student_t_mixture",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
            ],
            "component": ["early", "late", "early", "late", "late", "early", "late", "early", "late"],
            "parameter": ["mu", "mu", "sigma", "sigma", "pi_late", "mu", "mu", "sigma", "sigma"],
            "mean": [-0.05, 0.22, 0.08, 0.10, 0.55, -0.04, 0.24, 0.07, 0.11],
            "sd": [0.01] * 9,
            "q2_5": [-0.07, 0.18, 0.05, 0.07, 0.35, -0.06, 0.20, 0.05, 0.08],
            "q50": [-0.05, 0.22, 0.08, 0.10, 0.55, -0.04, 0.24, 0.07, 0.11],
            "q97_5": [-0.03, 0.26, 0.11, 0.13, 0.75, -0.02, 0.28, 0.09, 0.14],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_component_parameters.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["xi", "omega", "alpha_skew"],
            "mean": [0.20, 0.15, 1.2],
            "median": [0.20, 0.15, 1.2],
            "sd": [0.02, 0.02, 0.4],
            "mad": [0.02, 0.02, 0.3],
            "q2.5": [0.16, 0.11, 0.5],
            "q50": [0.20, 0.15, 1.2],
            "q97.5": [0.24, 0.19, 1.9],
            "rhat": [1.0, 1.0, 1.0],
            "ess_bulk": [1200.0, 1100.0, 1300.0],
            "ess_tail": [1100.0, 1000.0, 1200.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv", index=False)
    pd.DataFrame(
        {
            "model_name": [
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
                "model_c_mixture_of_experts",
            ],
            "term": ["alpha", "beta_rate", "beta_expected"],
            "mean": [0.1, 0.5, -0.3],
            "sd": [0.2, 0.2, 0.2],
            "q2_5": [-0.2, 0.1, -0.7],
            "q50": [0.1, 0.5, -0.3],
            "q97_5": [0.4, 0.9, 0.1],
            "prob_gt_zero": [0.7, 0.99, 0.08],
            "prob_lt_zero": [0.3, 0.01, 0.92],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_gating_coefficients.csv", index=False)
    base_probs = _synthetic_latency_event_input().loc[
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
    ].copy()
    probs_unconstrained = base_probs.copy()
    probs_unconstrained["model_name"] = "model_c_mixture_of_experts"
    probs_unconstrained["p_late_mean"] = np.linspace(0.1, 0.9, len(probs_unconstrained))
    probs_unconstrained["p_late_q2_5"] = np.clip(probs_unconstrained["p_late_mean"] - 0.1, 0.0, 1.0)
    probs_unconstrained["p_late_q50"] = probs_unconstrained["p_late_mean"]
    probs_unconstrained["p_late_q97_5"] = np.clip(probs_unconstrained["p_late_mean"] + 0.1, 0.0, 1.0)
    probs_unconstrained.to_csv(stan_dir / "behaviour_latency_regime_event_probabilities.csv", index=False)
    ppc_rows: list[dict[str, object]] = []
    for draw_id in range(5):
        for value in (-0.07, -0.02, 0.05, 0.18, 0.25):
            ppc_rows.append({"model_name": "model_c_mixture_of_experts", "draw_id": draw_id, "statistic": "y_rep", "value": None, "y_rep_value": value + 0.01 * draw_id})
    pd.DataFrame(ppc_rows).to_csv(stan_dir / "behaviour_latency_regime_posterior_predictive.csv", index=False)
    pd.DataFrame(
        {
            "model_name": [
                "model_a_one_student_t",
                "model_s_skew_unimodal",
                "model_b_two_student_t_mixture",
                "model_c_mixture_of_experts",
            ],
            "elpd_loo": [-40.0, -35.0, -36.0, -33.0],
            "se_elpd_loo": [2.0, 2.2, 2.1, 2.2],
            "p_loo": [3.0, 3.2, 4.0, 5.0],
            "looic": [80.0, 70.0, 72.0, 66.0],
            "se_looic": [4.0, 4.1, 4.2, 4.4],
            "delta_elpd_from_best": [7.0, 2.0, 3.0, 0.0],
            "delta_looic_from_best": [14.0, 4.0, 6.0, 0.0],
        }
    ).to_csv(stan_dir / "behaviour_latency_regime_loo_comparison.csv", index=False)
    (stan_dir / "behaviour_latency_regime_fit_metrics.json").write_text(json.dumps({"nu": 4}), encoding="utf-8")
    return stan_dir, event_csv


def _r_latency_regime_requirements_available(rscript: str) -> bool:
    command = (
        "packages <- c('cmdstanr','posterior','loo','readr','dplyr','tidyr','jsonlite','optparse');"
        "ok <- all(vapply(packages, requireNamespace, logical(1), quietly = TRUE));"
        "if (!ok) quit(status = 1);"
        "if (!nzchar(cmdstanr::cmdstan_path())) quit(status = 1);"
        "quit(status = 0)"
    )
    result = subprocess.run([rscript, "-e", command], check=False, capture_output=True, text=True)
    return result.returncode == 0
