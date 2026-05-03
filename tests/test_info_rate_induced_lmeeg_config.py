from __future__ import annotations

from pathlib import Path

import yaml
import pandas as pd

from cas.info_rate_induced_lmeeg.pipeline import (
    _trial_metadata_from_epochs,
    load_info_rate_induced_lmeeg_config,
)


def _write_paths_yaml(project_root: Path, derivatives_root: Path) -> None:
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {"derivatives_root": str(derivatives_root)}
    (config_dir / "paths.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_analysis_yaml(config_path: Path, induced_source_epochs_dir: str) -> None:
    payload = {
        "analysis_name": "info_rate_induced_lmeeg",
        "input": {
            "induced_source_epochs_dir": induced_source_epochs_dir,
            "behaviour_riskset_path": "reports/hazard_behavior_final/fpp_vs_spp/combined_riskset.parquet",
        },
        "output": {"out_dir": "results/info_rate_induced_lmeeg"},
        "windows": {
            "anchor_window_s": {"start": -1.5, "end": 0.0},
            "neural_bin_width_s": 0.050,
            "info_bin_width_s": 0.050,
            "min_causal_lag_s": 0.050,
            "max_causal_lag_s": 1.000,
        },
        "morlet": {
            "bands": {
                "alpha": {"start_hz": 8.0, "end_hz": 12.0, "step_hz": 1.0},
                "beta": {"start_hz": 13.0, "end_hz": 30.0, "step_hz": 1.0},
            },
            "n_cycles": {"mode": "frequency_divisor", "divisor": 2.0},
            "baseline": {"enabled": False},
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_derivatives_prefixed_induced_dir_is_normalized_against_derivatives_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    derivatives_root = tmp_path / "derivatives_root"
    (derivatives_root / "induced_source_epochs").mkdir(parents=True, exist_ok=True)
    (derivatives_root / "reports" / "hazard_behavior_final" / "fpp_vs_spp").mkdir(parents=True, exist_ok=True)

    _write_paths_yaml(project_root, derivatives_root)
    config_path = project_root / "config" / "info_rate_induced_lmeeg.yaml"
    _write_analysis_yaml(config_path, "derivatives/induced_source_epochs")

    config = load_info_rate_induced_lmeeg_config(config_path)
    assert config.induced_source_epochs_dir == (derivatives_root / "induced_source_epochs").resolve()


def test_plain_induced_dir_is_resolved_against_derivatives_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    derivatives_root = tmp_path / "derivatives_root"
    (derivatives_root / "induced_source_epochs").mkdir(parents=True, exist_ok=True)
    (derivatives_root / "reports" / "hazard_behavior_final" / "fpp_vs_spp").mkdir(parents=True, exist_ok=True)

    _write_paths_yaml(project_root, derivatives_root)
    config_path = project_root / "config" / "info_rate_induced_lmeeg.yaml"
    _write_analysis_yaml(config_path, "induced_source_epochs")

    config = load_info_rate_induced_lmeeg_config(config_path)
    assert config.induced_source_epochs_dir == (derivatives_root / "induced_source_epochs").resolve()


def test_trial_metadata_allows_missing_planned_total_info_column(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    derivatives_root = tmp_path / "derivatives_root"
    (derivatives_root / "induced_source_epochs").mkdir(parents=True, exist_ok=True)
    (derivatives_root / "reports" / "hazard_behavior_final" / "fpp_vs_spp").mkdir(parents=True, exist_ok=True)
    _write_paths_yaml(project_root, derivatives_root)
    config_path = project_root / "config" / "info_rate_induced_lmeeg.yaml"
    _write_analysis_yaml(config_path, "induced_source_epochs")
    config = load_info_rate_induced_lmeeg_config(config_path)

    metadata = pd.DataFrame(
        {
            "subject_id": ["sub-001"],
            "run": [1],
            "anchor_type": ["fpp"],
            "event_onset_conversation_s": [2.0],
            "fpp_onset": [2.0],
            "fpp_offset": [2.5],
            "spp_onset": [1.2],
            "spp_offset": [1.9],
            "fpp_duration": [0.5],
            "spp_duration": [0.7],
            "time_within_run": [2.0],
        }
    )

    trial_meta = _trial_metadata_from_epochs(
        metadata,
        record_subject="001",
        record_run="1",
        anchor_onset_decimals=config.anchor_onset_round_decimals,
        config=config,
    )
    assert "planned_response_total_information" in trial_meta.columns
    assert trial_meta["planned_response_total_information"].isna().all()
