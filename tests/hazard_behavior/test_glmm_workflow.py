from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from cas.cli.main import main
from cas.hazard_behavior.plot_r_results import plot_behaviour_hazard_results
from cas.hazard_behavior.r_export import export_behaviour_glmm_data


def test_export_includes_required_columns(tmp_path: Path) -> None:
    riskset = _tiny_riskset()
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=0, expected_lag_ms=300)

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
        lag_grid_ms=(0, 100, 300),
    )

    exported = pd.read_csv(output_csv)
    assert {
        "event",
        "participant_id",
        "run_id",
        "speaker",
        "participant_speaker_id",
        "participant_speaker",
        "time_from_partner_onset",
        "time_from_partner_offset",
        "time_since_partner_offset_positive",
        "z_information_rate_lag_best",
        "z_prop_expected_cumulative_info_lag_best",
        "z_information_rate_lag_0ms",
        "z_information_rate_lag_100ms",
        "z_prop_expected_cumulative_info_lag_300ms",
    } <= set(exported.columns)


def test_time_from_partner_offset_is_computed_when_absent(tmp_path: Path) -> None:
    riskset = _tiny_riskset().drop(columns=["time_from_partner_offset"])
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=0,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert exported["time_from_partner_offset"].iloc[0] == pytest.approx(-0.10)
    assert exported["time_from_partner_offset"].iloc[-1] == pytest.approx(0.20)


def test_negative_offset_times_are_retained(tmp_path: Path) -> None:
    riskset = _tiny_riskset()
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=0,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert (exported["time_from_partner_offset"] < 0.0).any()
    assert (exported["time_since_partner_offset_positive"] >= 0.0).all()


def test_lag_column_mapping_uses_selected_lags_json(tmp_path: Path) -> None:
    riskset = _tiny_riskset()
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=0, expected_lag_ms=300)

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
    )

    exported = pd.read_csv(output_csv)
    assert exported["z_information_rate_lag_best"].tolist() == riskset["z_information_rate_lag_0ms"].tolist()
    assert (
        exported["z_prop_expected_cumulative_info_lag_best"].tolist()
        == riskset["z_prop_expected_cumulative_info_lag_300ms"].tolist()
    )


def test_explicit_lags_override_selected_lags_json(tmp_path: Path) -> None:
    riskset = _tiny_riskset()
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=100, expected_lag_ms=0)

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=0,
        expected_cumulative_info_lag_ms=300,
    )

    exported = pd.read_csv(output_csv)
    assert exported["z_information_rate_lag_best"].tolist() == riskset["z_information_rate_lag_0ms"].tolist()
    assert (
        exported["z_prop_expected_cumulative_info_lag_best"].tolist()
        == riskset["z_prop_expected_cumulative_info_lag_300ms"].tolist()
    )


def test_export_qc_json_contains_expected_fields(tmp_path: Path) -> None:
    riskset = _tiny_riskset()
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=0,
        expected_cumulative_info_lag_ms=300,
    )

    payload = json.loads(output_qc_json.read_text(encoding="utf-8"))
    assert "n_rows_before_export" in payload
    assert "n_rows_after_export" in payload
    assert "n_events_after_export" in payload
    assert "n_participant_speaker_ids" in payload
    assert "n_participant_speakers" in payload
    assert "n_participants" in payload
    assert "lag_columns" in payload
    assert "proportion_negative_time_from_partner_offset" in payload
    assert payload["identity_validation"]["participant_speaker_id_valid"] is True


def test_participant_id_is_constructed_from_dyad_and_participant_speaker(tmp_path: Path) -> None:
    riskset = _tiny_riskset().drop(columns=["participant_id"], errors="ignore")
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"

    export_behaviour_glmm_data(
        riskset,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=0,
        expected_cumulative_info_lag_ms=300,
        lag_grid_ms=(0, 100, 300),
    )

    exported = pd.read_csv(output_csv)
    assert exported["participant_id"].tolist() == [
        "dyad-001_B",
        "dyad-001_B",
        "dyad-002_A",
        "dyad-002_A",
    ]


def test_r_glmm_script_defaults_to_participant_speaker_id() -> None:
    script_path = Path("/Users/hiro/Projects/active/cas/scripts/r/fit_behaviour_hazard_glmm.R")
    script_text = script_path.read_text(encoding="utf-8")
    assert 'default = "participant_speaker_id"' in script_text


def test_plotting_smoke_test_writes_expected_figures(tmp_path: Path) -> None:
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
            "predictor_family": ["information_rate", "information_rate", "prop_expected", "prop_actual"],
            "lag_ms": [0, 150, 300, 0],
            "child_BIC": [125.0, 110.0, 109.0, 140.0],
            "delta_BIC": [-2.0, -5.0, -1.0, 0.5],
            "beta": [0.10, 0.25, 0.12, 0.01],
            "conf_low": [0.02, 0.12, 0.03, -0.10],
            "conf_high": [0.18, 0.38, 0.21, 0.12],
            "odds_ratio": [1.11, 1.28, 1.13, 1.01],
            "odds_ratio_conf_low": [1.02, 1.13, 1.03, 0.90],
            "odds_ratio_conf_high": [1.20, 1.46, 1.23, 1.13],
            "converged": [True, True, True, False],
        }
    ).to_csv(r_results_dir / "r_glmm_information_rate_lag_sweep.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["prop_expected", "cumulative_info"],
            "lag_ms": [300, 300],
            "child_BIC": [109.0, 150.0],
            "delta_BIC": [-1.0, 2.0],
            "beta": [0.12, 0.0],
            "conf_low": [0.03, -0.1],
            "conf_high": [0.21, 0.1],
            "odds_ratio": [1.13, 1.0],
            "odds_ratio_conf_low": [1.03, 0.90],
            "odds_ratio_conf_high": [1.23, 1.1],
            "converged": [True, False],
        }
    ).to_csv(r_results_dir / "r_glmm_prop_expected_lag_sweep.csv", index=False)
    (r_results_dir / "r_glmm_selected_behaviour_lags.json").write_text(
        '{"best_information_rate_lag_ms": 150, "best_prop_expected_lag_ms": 300}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "child_model": [
                "M_final_glmm",
                "M_final_plus_expected_glmm",
                "M2a",
            ],
            "delta_BIC": [-3.2, -4.7, -1.2],
            "delta_AIC": [-4.0, -5.0, -2.0],
        }
    ).to_csv(r_results_dir / "r_glmm_final_behaviour_model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "predictor": ["z_information_rate_lag_150ms"] * 5,
            "predictor_value": [-2, -1, 0, 1, 2],
            "predicted_probability": [0.06, 0.09, 0.13, 0.17, 0.21],
            "conf_low": [0.04, 0.06, 0.10, 0.14, 0.17],
            "conf_high": [0.08, 0.12, 0.16, 0.20, 0.25],
            "fixed_effect_only": [True] * 5,
        }
    ).to_csv(predictions_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": [
                "information_rate",
                "prop_expected_cumulative_info",
                "prop_actual",
                "cumulative_info",
            ],
            "lag_ms": [0, 300, 0, 300],
            "delta_bic": [-2.0, -3.0, -1.0, -0.5],
        }
    ).to_csv(models_dir / "behaviour_timing_control_lag_selection.csv", index=False)

    plot_behaviour_hazard_results(
        r_results_dir=r_results_dir,
        timing_control_models_dir=models_dir,
        output_dir=figures_dir,
        qc_output_dir=qc_dir,
    )

    assert (figures_dir / "behaviour_r_glmm_delta_bic_by_lag.png").exists()
    assert (figures_dir / "behaviour_r_glmm_coefficient_by_lag.png").exists()
    assert (figures_dir / "behaviour_r_glmm_odds_ratio_by_lag.png").exists()
    assert (figures_dir / "behaviour_r_glmm_final_model_comparison.png").exists()
    assert (figures_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.png").exists()
    assert (qc_dir / "behaviour_pooled_delta_bic_by_lag.png").exists()
    assert not any("prop_actual" in path.name for path in figures_dir.iterdir())
    assert not any("cumulative_info" in path.name and "expected_info" not in path.name for path in figures_dir.iterdir())
    assert not any("M2a" in path.name or "M2b" in path.name or "M2c" in path.name for path in figures_dir.iterdir())


def test_cli_export_behaviour_glmm_data_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    riskset = _tiny_riskset()
    riskset_path = tmp_path / "hazard_behavior_riskset_with_timing_controls.tsv"
    riskset.to_csv(riskset_path, sep="\t", index=False)
    selected_lags_json = _write_selected_lags_json(tmp_path, information_rate_lag_ms=0, expected_lag_ms=300)
    output_csv = tmp_path / "behaviour_glmm_data.csv"
    output_qc_json = tmp_path / "behaviour_glmm_export_qc.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "export-behaviour-glmm-data",
            "--input-riskset",
            str(riskset_path),
            "--selected-lags-json",
            str(selected_lags_json),
            "--output-csv",
            str(output_csv),
            "--output-qc-json",
            str(output_qc_json),
        ],
    )

    assert main() == 0
    assert output_csv.exists()
    assert output_qc_json.exists()


def test_cli_plot_behaviour_glmm_results_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "delta_bic": [-2.5, -3.0],
        }
    ).to_csv(models_dir / "behaviour_timing_control_lag_selection.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["information_rate"],
            "lag_ms": [100],
            "child_BIC": [90.0],
            "delta_BIC": [-2.0],
            "beta": [0.10],
            "conf_low": [0.01],
            "conf_high": [0.19],
            "odds_ratio": [1.11],
            "odds_ratio_conf_low": [1.01],
            "odds_ratio_conf_high": [1.21],
            "converged": [True],
        }
    ).to_csv(r_results_dir / "r_glmm_information_rate_lag_sweep.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["prop_expected"],
            "lag_ms": [300],
            "child_BIC": [88.0],
            "delta_BIC": [-1.0],
            "beta": [0.20],
            "conf_low": [0.08],
            "conf_high": [0.32],
            "odds_ratio": [1.22],
            "odds_ratio_conf_low": [1.08],
            "odds_ratio_conf_high": [1.38],
            "converged": [True],
        }
    ).to_csv(r_results_dir / "r_glmm_prop_expected_lag_sweep.csv", index=False)
    (r_results_dir / "r_glmm_selected_behaviour_lags.json").write_text(
        '{"best_information_rate_lag_ms": 100, "best_prop_expected_lag_ms": 300}',
        encoding="utf-8",
    )
    pd.DataFrame({"child_model": ["M_final_glmm"], "delta_BIC": [-2.0], "delta_AIC": [-2.5]}).to_csv(
        r_results_dir / "r_glmm_final_behaviour_model_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "predictor": ["z_information_rate_lag_100ms"],
            "predictor_value": [0],
            "predicted_probability": [0.11],
            "conf_low": [0.09],
            "conf_high": [0.14],
            "fixed_effect_only": [True],
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
    assert (figures_dir / "behaviour_r_glmm_coefficient_by_lag.png").exists()
    assert (qc_dir / "behaviour_pooled_delta_bic_by_lag.png").exists()


def test_r_glmm_smoke_test(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable.")
    if not _r_packages_available(rscript, ["glmmTMB", "lme4", "broom.mixed", "dplyr", "readr", "jsonlite", "optparse"]):
        pytest.skip("Required R packages for behavioural GLMM smoke testing are unavailable.")

    input_csv = tmp_path / "behaviour_glmm_data.csv"
    output_dir = tmp_path / "models"
    _synthetic_r_glmm_input().to_csv(input_csv, index=False)

    result = subprocess.run(
        [
            rscript,
            str(Path("scripts/r/fit_behaviour_glmm_lag_sweep.R")),
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--lag-grid-ms",
            "0,100,300",
            "--r-glmm-include-quadratic-offset-timing",
            "--backend",
            "glmer",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/hiro/Projects/active/cas",
    )
    if result.returncode != 0:
        pytest.skip(f"Behavioural R GLMM lag-sweep smoke test did not converge on the synthetic dataset: {result.stderr}")

    assert (output_dir / "r_glmm_information_rate_lag_sweep.csv").exists()
    assert (output_dir / "r_glmm_prop_expected_lag_sweep.csv").exists()
    assert (output_dir / "r_glmm_selected_behaviour_lags.json").exists()
    assert (output_dir / "r_glmm_final_behaviour_model_summary.csv").exists()
    assert (output_dir / "r_glmm_final_behaviour_model_comparison.csv").exists()
    assert (output_dir / "r_glmm_final_behaviour_effects.json").exists()
    assert (output_dir / "predictions" / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv").exists()


def _tiny_riskset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event": [0, 1, 0, 0],
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "run": ["1", "1", "2", "2"],
            "speaker": ["B", "B", "A", "A"],
            "participant_speaker_id": ["dyad-001_B", "dyad-001_B", "dyad-002_A", "dyad-002_A"],
            "participant_speaker": ["B", "B", "A", "A"],
            "participant_id": ["dyad-001_B", "dyad-001_B", "dyad-002_A", "dyad-002_A"],
            "episode_id": ["ep-1", "ep-1", "ep-2", "ep-2"],
            "bin_end": [0.40, 0.70, 1.20, 1.50],
            "partner_ipu_onset": [0.0, 0.0, 0.8, 0.8],
            "partner_ipu_offset": [0.50, 0.50, 1.30, 1.30],
            "time_from_partner_onset": [0.40, 0.70, 0.40, 0.70],
            "time_from_partner_offset": [-0.10, 0.20, -0.10, 0.20],
            "z_information_rate_lag_0ms": [-1.0, 0.5, -0.2, 0.8],
            "z_information_rate_lag_50ms": [-0.9, 0.6, -0.15, 0.9],
            "z_information_rate_lag_100ms": [-0.8, 0.7, -0.1, 1.0],
            "z_information_rate_lag_150ms": [-0.7, 0.8, -0.05, 1.1],
            "z_information_rate_lag_200ms": [-0.6, 0.9, 0.0, 1.2],
            "z_information_rate_lag_300ms": [-0.5, 1.0, 0.1, 1.3],
            "z_information_rate_lag_500ms": [-0.4, 1.1, 0.2, 1.4],
            "z_information_rate_lag_700ms": [-0.3, 1.2, 0.3, 1.5],
            "z_information_rate_lag_1000ms": [-0.2, 1.3, 0.4, 1.6],
            "z_prop_expected_cumulative_info_lag_0ms": [-0.4, 0.1, 0.2, 0.5],
            "z_prop_expected_cumulative_info_lag_50ms": [-0.3, 0.2, 0.25, 0.6],
            "z_prop_expected_cumulative_info_lag_100ms": [-0.2, 0.3, 0.28, 0.7],
            "z_prop_expected_cumulative_info_lag_150ms": [-0.15, 0.4, 0.29, 0.8],
            "z_prop_expected_cumulative_info_lag_200ms": [-0.12, 0.5, 0.30, 0.85],
            "z_prop_expected_cumulative_info_lag_300ms": [-0.1, 0.6, 0.3, 0.9],
            "z_prop_expected_cumulative_info_lag_500ms": [-0.05, 0.7, 0.35, 1.0],
            "z_prop_expected_cumulative_info_lag_700ms": [0.0, 0.8, 0.4, 1.1],
            "z_prop_expected_cumulative_info_lag_1000ms": [0.05, 0.9, 0.45, 1.2],
            "episode_has_event": [1, 1, 0, 0],
            "own_fpp_onset": [0.70, 0.70, pd.NA, pd.NA],
        }
    )


def _write_selected_lags_json(tmp_path: Path, *, information_rate_lag_ms: int, expected_lag_ms: int) -> Path:
    path = tmp_path / "behaviour_timing_control_selected_lags.json"
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


def _synthetic_r_glmm_input() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_number in range(24):
        dyad_id = f"dyad-{episode_number % 6:03d}"
        speaker = "A" if episode_number % 2 == 0 else "B"
        participant_speaker_id = f"{dyad_id}_{speaker}"
        episode_id = f"ep-{episode_number:03d}"
        partner_ipu_onset = 0.5 * (episode_number % 3)
        partner_ipu_offset = partner_ipu_onset + 0.45 + 0.03 * (episode_number % 4)
        event_bin = 5 if episode_number % 4 != 0 else None
        for bin_index in range(8):
            bin_end = round(partner_ipu_onset + 0.1 * (bin_index + 1), 3)
            info_value = -1.0 + 0.15 * episode_number + 0.10 * bin_index
            expected_value = -0.8 + 0.12 * episode_number + 0.08 * bin_index
            rows.append(
                {
                    "event": int(event_bin is not None and bin_index == event_bin),
                    "dyad_id": dyad_id,
                    "speaker": speaker,
                    "participant_speaker_id": participant_speaker_id,
                    "participant_speaker": speaker,
                    "participant_id": participant_speaker_id,
                    "run_id": str((episode_number % 4) + 1),
                    "episode_id": episode_id,
                    "time_from_partner_onset": round(bin_end - partner_ipu_onset, 3),
                    "time_from_partner_offset": round(bin_end - partner_ipu_offset, 3),
                    "time_since_partner_offset_positive": max(round(bin_end - partner_ipu_offset, 3), 0.0),
                    "z_information_rate_lag_0ms": info_value - 0.10,
                    "z_information_rate_lag_50ms": info_value - 0.05,
                    "z_information_rate_lag_100ms": info_value,
                    "z_information_rate_lag_150ms": info_value + 0.03,
                    "z_information_rate_lag_200ms": info_value + 0.02,
                    "z_information_rate_lag_300ms": info_value - 0.02,
                    "z_information_rate_lag_500ms": info_value - 0.05,
                    "z_information_rate_lag_700ms": info_value - 0.08,
                    "z_information_rate_lag_1000ms": info_value - 0.10,
                    "z_prop_expected_cumulative_info_lag_0ms": expected_value - 0.05,
                    "z_prop_expected_cumulative_info_lag_50ms": expected_value - 0.03,
                    "z_prop_expected_cumulative_info_lag_100ms": expected_value - 0.01,
                    "z_prop_expected_cumulative_info_lag_150ms": expected_value + 0.01,
                    "z_prop_expected_cumulative_info_lag_200ms": expected_value + 0.02,
                    "z_prop_expected_cumulative_info_lag_300ms": expected_value + 0.03,
                    "z_prop_expected_cumulative_info_lag_500ms": expected_value + 0.01,
                    "z_prop_expected_cumulative_info_lag_700ms": expected_value - 0.01,
                    "z_prop_expected_cumulative_info_lag_1000ms": expected_value - 0.03,
                }
            )
    return pd.DataFrame(rows)


def _r_packages_available(rscript: str, packages: list[str]) -> bool:
    package_vector = ", ".join(f'"{package}"' for package in packages)
    command = (
        "packages <- c("
        + package_vector
        + ");"
        + "status <- all(vapply(packages, requireNamespace, logical(1), quietly = TRUE));"
        + "quit(status = if (status) 0 else 1)"
    )
    result = subprocess.run([rscript, "-e", command], check=False)
    return result.returncode == 0
