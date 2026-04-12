from __future__ import annotations

from argparse import Namespace

from rate.lmeeeg.io import (
    build_lmeeeg_model_output_paths,
    resolve_lmeeeg_input_paths,
    resolve_lmeeeg_model_band_names,
    resolve_lmeeeg_model_modality,
)
from rate.cli.commands import lmeeeg_subject


def test_lmeeeg_model_modality_defaults_to_evoked():
    config = {"lmeeeg": {"models": {"demo": {"formula": "~ run"}}}}

    assert resolve_lmeeeg_model_modality(config, model_name="demo") == "evoked"


def test_lmeeeg_input_paths_use_induced_epochs_when_requested():
    config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                    "modality": "induced",
                }
            }
        },
        "induced_epochs": {
            "bands": ["theta"],
        },
    }
    row = {"subject_id": "sub-001", "task": "conversation", "run": 1, "dyad_id": "dyad-001", "eeg_path": "unused"}

    input_paths = resolve_lmeeeg_input_paths(config, row, config_section="lmeeeg", model_name="demo")

    assert input_paths.epochs_output_path.as_posix().endswith("/induced_epochs/theta/sub-001/epochs-time_s.fif")
    assert input_paths.metadata_output_path.as_posix().endswith("/induced_epochs/theta/sub-001/metadata-time_s.csv")


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

    assert resolve_lmeeeg_model_band_names(config, model_name="demo") == ["theta", "alpha", "beta"]


def test_lmeeeg_input_paths_use_requested_induced_band():
    config = {
        "paths": {"out_dir": "/tmp/out"},
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
    row = {"subject_id": "sub-001", "task": "conversation", "run": 1, "dyad_id": "dyad-001", "eeg_path": "unused"}

    input_paths = resolve_lmeeeg_input_paths(config, row, config_section="lmeeeg", model_name="demo", band_name="alpha")

    assert input_paths.epochs_output_path.as_posix().endswith("/induced_epochs/alpha/sub-001/epochs-time_s.fif")
    assert input_paths.metadata_output_path.as_posix().endswith("/induced_epochs/alpha/sub-001/metadata-time_s.csv")


def test_lmeeeg_output_paths_are_namespaced_by_modality():
    config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                    "modality": "induced",
                }
            }
        },
    }

    output_paths = build_lmeeeg_model_output_paths(config, model_name="demo", band_name="alpha")

    assert output_paths.summary_output_path.as_posix().endswith("/lmeeeg/induced/alpha/demo/summary.json")


def test_lmeeeg_subject_run_expands_induced_models_across_all_bands(monkeypatch):
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
    observed_band_names: list[str | None] = []

    monkeypatch.setattr(
        lmeeeg_subject,
        "select_lmeeeg_rows",
        lambda *args, **kwargs: [{"subject_id": "sub-001", "task": "conversation", "run": 1}],
    )

    def fake_fit_one_model(*args, **kwargs):
        observed_band_names.append(kwargs.get("band_name"))
        return {
            "model_name": kwargs["model_name"],
            "band_name": kwargs.get("band_name"),
            "summary_output": f"{kwargs['model_name']}.json",
            "n_trials_used": 1,
            "n_input_rows": 1,
        }

    monkeypatch.setattr(lmeeeg_subject, "_fit_one_model", fake_fit_one_model)

    exit_code = lmeeeg_subject.run(
        Namespace(
            dataset_index=None,
            subject_id=None,
            task=None,
            run=None,
            config_section="lmeeeg",
            target=None,
            model_name=None,
            epochs_path=None,
            metadata_path=None,
            run_all=False,
            output=None,
        ),
        config=config,
    )

    assert exit_code == 0
    assert observed_band_names == ["theta", "alpha", "beta"]
