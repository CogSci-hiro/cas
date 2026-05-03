from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cas.source_dics.config import load_source_dics_config
from cas.source_dics.dics import SourcePowerResult
from cas.source_dics.events import prepare_epoch_metadata, split_anchor_metadata
from cas.source_dics.export import LONG_TABLE_REQUIRED_COLUMNS, export_long_table
from cas.source_dics.io import EpochRecord


def _write_config(tmp_path: Path, *, overrides: dict | None = None) -> Path:
    base = {
        "paths": {
            "events_csv": str((tmp_path / "data" / "events" / "events.csv").relative_to(tmp_path)),
            "epochs_dir": str((tmp_path / "data" / "epochs").relative_to(tmp_path)),
            "source_dir": str((tmp_path / "data" / "source_dics").relative_to(tmp_path)),
            "derivatives_dir": str((tmp_path / "results" / "source_dics").relative_to(tmp_path)),
            "lmeeeg_dir": str((tmp_path / "results" / "source_dics" / "lmeeeg").relative_to(tmp_path)),
        },
        "events": {
            "anchor_types": ["FPP", "SPP"],
            "subject_column": "subject",
            "dyad_column": "dyad",
            "run_column": "run",
            "anchor_type_column": "anchor_type",
            "onset_column": "onset",
            "duration_column": "duration",
            "latency_column": "latency",
            "label_column": "label",
        },
        "epoching": {
            "tmin": -2.0,
            "tmax": 0.5,
            "baseline": None,
            "reject_by_annotation": True,
        },
        "dics": {
            "method": "dics",
            "common_filter": True,
            "filter_tmin": -1.5,
            "filter_tmax": -0.1,
            "analysis_tmin": -1.5,
            "analysis_tmax": -0.1,
            "bands": {
                "alpha": {"fmin": 8.0, "fmax": 12.0},
                "beta": {"fmin": 13.0, "fmax": 30.0},
            },
            "csd_method": "multitaper",
            "mt_bandwidth": None,
            "regularization": 0.05,
            "pick_ori": "max-power",
            "weight_norm": "unit-noise-gain",
            "reduce_rank": True,
            "real_filter": True,
            "n_jobs": 1,
        },
        "source_space": {
            "kind": "surface",
            "mode": "fsaverage_surface",
            "spacing": "oct6",
            "subjects_dir": str((tmp_path / "data" / "freesurfer").relative_to(tmp_path)),
            "subject": "fsaverage",
            "trans": "fsaverage",
            "bem": None,
            "forward_template": str((tmp_path / "data" / "forward-template.fif").relative_to(tmp_path)),
            "parcellation": "aparc",
            "aggregate_to_labels": False,
            "aggregation": "mean",
        },
        "output": {
            "save_filters": True,
            "save_trial_power": True,
            "save_long_table": True,
            "save_qc": True,
            "overwrite": False,
            "long_table_chunk_rows": 20,
        },
        "lmeeeg": {
            "enabled": True,
            "contrast_of_interest": "anchor_typeFPP",
            "formula": "power ~ anchor_type + duration + latency + run + time_within_run + (1 | subject)",
            "dependent_variable": "power",
            "test_predictors": ["anchor_type"],
            "predictors": [
                "anchor_type",
                "duration",
                "latency",
                "run",
                "time_within_run",
            ],
        },
        "logging": {"verbose": True, "progress": True},
    }
    if overrides:
        for key, value in overrides.items():
            base[key] = value

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "events").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "epochs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "freesurfer").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "events" / "events.csv").write_text("pair_id\npair-001\n", encoding="utf-8")
    (tmp_path / "data" / "forward-template.fif").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "config" / "source_dics.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return config_path


def test_source_dics_config_loads_and_creates_output_dirs(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    config = load_source_dics_config(config_path)

    assert config.paths.derivatives_dir.exists()
    assert config.paths.qc_dir.exists()
    assert config.paths.long_table_dir.exists()
    assert config.dics.bands["alpha"].fmin == 8.0
    assert config.lmeeeg.contrast_of_interest == "anchor_typeFPP"
    assert config.lmeeeg.test_predictors == ("anchor_type",)


def test_source_dics_invalid_time_windows_fail(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        overrides={
            "dics": {
                "method": "dics",
                "common_filter": True,
                "filter_tmin": -1.5,
                "filter_tmax": 0.1,
                "analysis_tmin": -1.5,
                "analysis_tmax": -0.1,
                "bands": {"alpha": {"fmin": 8.0, "fmax": 12.0}},
                "csd_method": "multitaper",
                "mt_bandwidth": None,
                "regularization": 0.05,
                "pick_ori": "max-power",
                "weight_norm": "unit-noise-gain",
                "reduce_rank": True,
                "real_filter": True,
                "n_jobs": 1,
            }
        },
    )

    with pytest.raises(ValueError, match="filter_tmax"):
        load_source_dics_config(config_path)


def test_source_dics_invalid_source_space_kind_fails(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        overrides={
            "source_space": {
                "kind": "fsaverage_vol_or_surface",
                "mode": "fsaverage_surface",
                "spacing": "oct6",
                "subjects_dir": str((tmp_path / "data" / "freesurfer").relative_to(tmp_path)),
                "subject": "fsaverage",
                "trans": "fsaverage",
                "bem": None,
                "forward_template": str((tmp_path / "data" / "forward-template.fif").relative_to(tmp_path)),
                "parcellation": "aparc",
                "aggregate_to_labels": False,
                "aggregation": "mean",
            }
        },
    )

    with pytest.raises(ValueError, match="source_space.kind"):
        load_source_dics_config(config_path)


def test_prepare_epoch_metadata_filters_to_fpp_and_spp(tmp_path: Path) -> None:
    record = EpochRecord(
        subject_id="sub-001",
        run_id="1",
        task="conversation",
        epochs_path=tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
    )
    metadata = pd.DataFrame(
        {
            "event_family": ["fpp", "spp", "fpp"],
            "pair_id": ["pair-001", "pair-002", "pair-003"],
            "subject_id": ["sub-001", "sub-001", "sub-001"],
            "recording_id": ["dyad-001", "dyad-001", "dyad-001"],
            "run": ["1", "1", "1"],
            "event_onset_conversation_s": [1.0, 2.0, 3.0],
            "information_rate_lag_200ms_z": [0.1, 0.2, 0.3],
        }
    )
    events_table = pd.DataFrame(
        {
            "pair_id": ["pair-001", "pair-002", "pair-003"],
            "fpp_label": ["A", "B", "C"],
            "spp_label": ["X", "Y", "Z"],
            "fpp_duration": [0.5, 0.6, 0.7],
            "spp_duration": [0.3, 0.4, 0.5],
            "latency": [0.2, 0.25, 0.3],
            "fpp_onset": [1.0, 2.0, 3.0],
            "spp_onset": [1.2, 2.3, 3.4],
        }
    )
    config = load_source_dics_config(_write_config(tmp_path))

    prepared = prepare_epoch_metadata(
        metadata,
        events_table=events_table,
        events_config=config.events,
        record=record,
    )
    selections = split_anchor_metadata(prepared)

    assert set(prepared["anchor_type"]) == {"FPP", "SPP"}
    assert set(selections) == {"FPP", "SPP"}
    assert selections["FPP"].epoch_indices.tolist() == [0, 2]
    assert prepared["run"].dtype.kind in {"i", "u", "f"}
    assert selections["SPP"].epoch_indices.tolist() == [1]
    assert {"subject", "dyad", "run", "event_id", "time_within_run"}.issubset(prepared.columns)


def test_source_dics_long_table_export_has_expected_schema(tmp_path: Path) -> None:
    config = load_source_dics_config(_write_config(tmp_path))
    record = EpochRecord(
        subject_id="sub-001",
        run_id="1",
        task="conversation",
        epochs_path=tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
    )
    metadata = pd.DataFrame(
        {
            "subject": ["sub-001", "sub-001"],
            "dyad": ["dyad-001", "dyad-001"],
            "run": ["1", "1"],
            "event_id": ["pair-001", "pair-002"],
            "anchor_type": ["FPP", "FPP"],
            "label": ["A", "B"],
            "duration": [0.5, 0.7],
            "latency": [0.2, 0.3],
            "time_within_run": [1.0, 2.0],
            "information_rate_lag_200ms_z": [0.1, 0.2],
            "prop_expected_cumulative_info_lag_200ms_z": [0.3, 0.4],
            "upcoming_utterance_information_content": [1.5, 1.8],
            "n_tokens": [3, 4],
        }
    )
    result = SourcePowerResult(
        anchor_type="FPP",
        band_name="alpha",
        source_ids=["lh:1", "rh:2"],
        times=np.asarray([-1.5, -1.0], dtype=float),
        power=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        ),
        stcs=[],
    )

    written_paths = export_long_table(result, metadata, record=record, config=config)

    assert written_paths
    if written_paths[0].suffix == ".parquet":
        table = pd.read_parquet(written_paths[0])
    else:
        table = pd.read_csv(written_paths[0])
    assert list(table.columns) == list(LONG_TABLE_REQUIRED_COLUMNS)
    assert len(table) == 8


def test_source_dics_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cas.cli.main", "source-dics-fpp-spp", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "source-dics-fpp-spp" in result.stdout
    assert "--config" in result.stdout
    assert "--subjects" in result.stdout


def test_source_dics_target_name_exists() -> None:
    targets_path = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "targets.smk"
    target_text = targets_path.read_text(encoding="utf-8")

    assert "rule source_dics_fpp_spp_alpha_beta_all:" in target_text
    assert "rule source_dics_all:" in target_text
