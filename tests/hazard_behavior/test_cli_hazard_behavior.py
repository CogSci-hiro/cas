from __future__ import annotations

import json
import sys
from pathlib import Path

from cas.cli.main import main


def test_cli_creates_key_outputs(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.csv"
    surprisal_path = tmp_path / "toy_desc-lmSurprisal_features.tsv"
    out_dir = tmp_path / "results"

    events_path.write_text(
        "\n".join(
            [
                "dyad_id,run,speaker,fpp_onset,fpp_offset,fpp_label",
                "dyad-001,1,B,1.20,1.50,FPP_RFC_TAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    surprisal_path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\tonset\tduration\tword\tsurprisal\talignment_status",
                "dyad-001\t1\tA\t0.00\t0.10\toui\t1.0\tok",
                "dyad-001\t1\tA\t0.30\t0.10\talors\t2.0\tok",
                "dyad-001\t1\tA\t0.70\t0.10\trouge\t3.0\tok",
                "dyad-001\t1\tB\t1.20\t0.10\tok\t1.5\tok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "hazard-behavior-fpp",
            "--events",
            str(events_path),
            "--surprisal",
            str(surprisal_path),
            "--out-dir",
            str(out_dir),
            "--event-positive-only",
            "--overwrite",
        ],
    )

    assert main() == 0
    assert (out_dir / "riskset" / "hazard_behavior_riskset.tsv").exists()
    assert (out_dir / "riskset" / "hazard_behavior_episode_summary.tsv").exists()
    assert (out_dir / "riskset" / "partner_ipu_table.tsv").exists()
    assert (out_dir / "riskset" / "partner_ipu_episode_summary.tsv").exists()
    assert (out_dir / "riskset" / "partner_ipu_anchor_qc.json").exists()
    assert (out_dir / "riskset" / "riskset_event_qc.json").exists()
    assert (out_dir / "riskset" / "event_rows_debug.tsv").exists()
    assert (out_dir / "riskset" / "episode_validation_qc.json").exists()
    assert (out_dir / "riskset" / "excluded_episodes.tsv").exists()
    assert (out_dir / "models" / "model_comparison_behaviour.csv").exists()
    assert (out_dir / "models" / "model_fit_metrics.json").exists()
    assert (out_dir / "models" / "behaviour_primary_model_summary.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_model_comparison.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_effects.json").exists()
    assert (out_dir / "models" / "behaviour_primary_fit_metrics.json").exists()
    assert (out_dir / "models" / "behaviour_primary_stat_tests.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_stat_tests.json").exists()
    assert (out_dir / "models" / "behaviour_primary_publication_table.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_interpretation.txt").exists()
    assert (out_dir / "riskset" / "prop_actual_saturation_qc.json").exists()
    assert (out_dir / "riskset" / "event_rate_by_prop_actual_saturation.csv").exists()
    assert (out_dir / "riskset" / "observed_event_rate_by_time_bin.csv").exists()
    assert (out_dir / "riskset" / "observed_event_rate_nonzero_bins.csv").exists()
    assert (out_dir / "riskset" / "observed_event_rate_plot_qc.json").exists()
    assert (out_dir / "figures" / "prop_actual_by_time_from_partner_onset.png").exists()
    assert (out_dir / "figures" / "prop_actual_saturation_by_time.png").exists()
    assert (out_dir / "figures" / "event_rate_by_prop_actual_saturation.png").exists()
    assert (out_dir / "figures" / "observed_event_rate_by_time_bin.png").exists()
    assert (out_dir / "figures" / "event_time_from_partner_onset_distribution.png").exists()
    assert (out_dir / "figures" / "episode_duration_distribution.png").exists()
    assert (out_dir / "figures" / "fpp_latency_from_partner_offset_distribution.png").exists()
    assert (out_dir / "figures" / "fpp_latency_from_partner_offset_before_exclusion.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_coefficients.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_model_comparison.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_predicted_hazard_prop_expected.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_predicted_hazard_information_rate.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_observed_event_rate.png").exists()
    assert (out_dir / "figures" / "behaviour_primary_lag_sensitivity.png").exists()

    payload = json.loads((out_dir / "models" / "model_fit_metrics.json").read_text(encoding="utf-8"))
    assert "models" in payload


def test_cli_creates_neural_lowlevel_outputs(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "events.csv"
    surprisal_path = tmp_path / "toy_desc-lmSurprisal_features.tsv"
    neural_path = tmp_path / "toy_desc-lowlevelNeural_features.tsv"
    out_dir = tmp_path / "results_neural"

    events_path.write_text(
        "\n".join(
            [
                "dyad_id,run,speaker,fpp_onset,fpp_offset,fpp_label",
                "dyad-001,1,B,1.20,1.50,FPP_RFC_TAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    surprisal_path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\tonset\tduration\tword\tsurprisal\talignment_status",
                "dyad-001\t1\tA\t0.00\t0.10\toui\t1.0\tok",
                "dyad-001\t1\tA\t0.30\t0.10\talors\t2.0\tok",
                "dyad-001\t1\tA\t0.70\t0.10\trouge\t3.0\tok",
                "dyad-001\t1\tB\t1.20\t0.10\tok\t1.5\tok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    neural_path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\ttime\tamp_cz\talpha_parietal\tbeta_frontal",
                "dyad-001\t1\tB\t0.50\t0.1\t0.2\t0.3",
                "dyad-001\t1\tB\t0.70\t0.2\t0.3\t0.4",
                "dyad-001\t1\tB\t0.85\t0.3\t0.4\t0.5",
                "dyad-001\t1\tB\t0.95\t0.4\t0.5\t0.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "hazard-behavior-fpp",
            "--events",
            str(events_path),
            "--surprisal",
            str(surprisal_path),
            "--neural-features",
            str(neural_path),
            "--fit-neural-lowlevel-models",
            "--out-dir",
            str(out_dir),
            "--event-positive-only",
            "--overwrite",
        ],
    )

    assert main() == 0
    assert (out_dir / "models" / "neural_lowlevel_model_summary.csv").exists()
    assert (out_dir / "models" / "neural_lowlevel_model_comparison.csv").exists()
    assert (out_dir / "models" / "neural_lowlevel_effects.json").exists()
    assert (out_dir / "figures" / "neural_lowlevel_pca_variance.png").exists()
    assert (out_dir / "figures" / "neural_lowlevel_model_comparison.png").exists()
