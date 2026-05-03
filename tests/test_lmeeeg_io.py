from __future__ import annotations

import json
from pathlib import Path

import yaml

from cas.stats import lmeeeg_pipeline


def test_lmeeeg_model_modality_defaults_to_evoked():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {"models": {"demo": {"formula": "~ run"}}},
    }

    epochs_path, metadata_path = lmeeeg_pipeline._resolve_pooled_source_paths(
        runtime_config,
        model_name="demo",
        epochs_path="/tmp/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
    )

    assert epochs_path.as_posix().endswith("/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif")
    assert metadata_path is None


def test_lmeeeg_input_paths_use_induced_epochs_when_requested():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {"models": {"demo": {"formula": "~ run", "modality": "induced"}}},
    }

    epochs_path, metadata_path = lmeeeg_pipeline._resolve_pooled_source_paths(
        runtime_config,
        model_name="demo",
        epochs_path="/tmp/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
        band_name="theta",
    )

    assert epochs_path.as_posix().endswith("/induced_epochs/theta/sub-001/epochs-time_s.fif")
    assert metadata_path is not None
    assert metadata_path.as_posix().endswith("/induced_epochs/theta/sub-001/metadata-time_s.csv")


def test_lmeeeg_induced_models_default_to_all_configured_bands():
    config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                    "modality": "induced",
                }
            }
        },
        "induced_epochs": {
            "bands": ["theta", "alpha", "beta"],
        },
    }

    assert lmeeeg_pipeline._resolve_induced_band_names(config) == ["theta", "alpha", "beta"]


def test_lmeeeg_input_paths_use_requested_induced_band():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {"models": {"demo": {"formula": "~ run", "modality": "induced"}}},
    }

    epochs_path, metadata_path = lmeeeg_pipeline._resolve_pooled_source_paths(
        runtime_config,
        model_name="demo",
        epochs_path="/tmp/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
        band_name="alpha",
    )

    assert epochs_path.as_posix().endswith("/induced_epochs/alpha/sub-001/epochs-time_s.fif")
    assert metadata_path is not None
    assert metadata_path.as_posix().endswith("/induced_epochs/alpha/sub-001/metadata-time_s.csv")


def test_lmeeeg_input_paths_support_configured_induced_epochs_subdir():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "input": {"induced_epochs_subdir": "induced_epochs_custom"},
            "models": {"demo": {"formula": "~ run", "modality": "induced"}},
        },
    }

    epochs_path, metadata_path = lmeeeg_pipeline._resolve_pooled_source_paths(
        runtime_config,
        model_name="demo",
        epochs_path="/tmp/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
        band_name="beta",
    )

    assert epochs_path.as_posix().endswith("/induced_epochs_custom/beta/sub-001/epochs-time_s.fif")
    assert metadata_path is not None
    assert metadata_path.as_posix().endswith("/induced_epochs_custom/beta/sub-001/metadata-time_s.csv")


def test_lmeeeg_output_paths_are_namespaced_by_modality():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {"models": {"demo": {"formula": "~ run", "modality": "induced"}}},
    }

    output_dir = lmeeeg_pipeline._model_output_dir(runtime_config, model_name="demo", band_name="alpha")

    assert output_dir.as_posix().endswith("/lmeeeg/induced/alpha/demo")


def test_lmeeeg_analysis_expands_induced_models_across_all_bands(
    tmp_path: Path,
    monkeypatch,
):
    config = {
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                    "modality": "induced",
                }
            }
        },
        "induced_epochs": {
            "bands": ["theta", "alpha", "beta"],
        },
    }
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    observed_band_names: list[str | None] = []

    def fake_run_pooled_model(*, runtime_config, model_name, epochs_paths, band_name):
        observed_band_names.append(band_name)
        return {"model_name": model_name, "band_name": band_name, "status": "ok"}

    monkeypatch.setattr(lmeeeg_pipeline, "_run_pooled_model", fake_run_pooled_model)

    summary = lmeeeg_pipeline.run_pooled_lmeeeg_analysis(
        epochs_paths=[],
        config_path=config_path,
        output_dir=tmp_path / "lmeeeg" / "summary",
    )

    assert observed_band_names == ["theta", "alpha", "beta"]
    summary_path = tmp_path / "lmeeeg" / "summary" / "lmeeeg_analysis_summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
