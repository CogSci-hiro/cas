from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cas.stats.lmeeeg_pipeline import (
    _augment_lmeeeg_metadata,
    _row_from_epochs_path,
    _resolve_pooled_source_paths,
    load_lmeeeg_config,
    run_pooled_lmeeeg_analysis,
    select_epochs_from_config,
)


class FakeEpochs:
    def __init__(self, metadata: pd.DataFrame, selection: np.ndarray | None = None):
        self.metadata = metadata.reset_index(drop=True)
        self.selection = np.arange(len(self.metadata), dtype=int) if selection is None else np.asarray(selection, dtype=int)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item):
        if isinstance(item, str):
            mask = self.metadata["event_type"].astype(str) == item
        else:
            values = np.asarray(item)
            if values.dtype == bool:
                mask = values
            else:
                mask = np.zeros(len(self.metadata), dtype=bool)
                mask[values.astype(int)] = True
        return FakeEpochs(self.metadata.loc[mask].reset_index(drop=True), self.selection[mask])


def test_load_lmeeeg_config_supports_section_wrapper(tmp_path):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  selection:
    event_type: "self_fpp_onset"
  models:
    demo:
      formula: "~ proximity"
      test_predictors: ["proximity"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_lmeeeg_config(config_path)
    assert loaded["selection"]["event_type"] == "self_fpp_onset"
    assert "demo" in loaded["models"]


def test_select_epochs_from_config_applies_event_type_and_metadata_query():
    metadata = pd.DataFrame(
        {
            "event_type": ["self_fpp_onset", "self_fpp_onset", "other_fpp_onset"],
            "latency": [0.1, 0.4, 0.9],
            "subject_id": ["sub-001", "sub-001", "sub-002"],
        }
    )
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {
            "event_type": "self_fpp_onset",
            "metadata_query": "latency >= 0.3",
        },
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    selected_epochs, selected_metadata = select_epochs_from_config(epochs, metadata, config)

    assert len(selected_epochs) == 1
    assert selected_epochs.selection.tolist() == [1]
    assert selected_metadata["latency"].tolist() == [0.4]


def test_select_epochs_from_config_raises_on_empty_selection():
    metadata = pd.DataFrame({"event_type": ["self_fpp_onset"], "latency": [0.1]})
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {"event_type": "other_fpp_onset"},
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    with pytest.raises(ValueError, match="zero rows"):
        select_epochs_from_config(epochs, metadata, config)


def test_augment_lmeeeg_metadata_derives_duration_from_event_family():
    metadata = pd.DataFrame(
        {
            "event_family": ["spp", "fpp"],
            "spp_duration": [0.25, 0.40],
            "fpp_duration": [1.2, 1.5],
        }
    )

    augmented = _augment_lmeeeg_metadata(metadata)

    assert augmented["duration_s"].tolist() == [0.25, 1.5]


def test_augment_lmeeeg_metadata_splits_label_classes():
    metadata = pd.DataFrame(
        {
            "fpp_label": ["FPP_RFC_DECL", "FPP_RFC_INT"],
            "spp_label": ["SPP_CONF_SIMP", "SPP_DISC_CORR"],
        }
    )

    augmented = _augment_lmeeeg_metadata(metadata)

    assert augmented["fpp_class_1"].tolist() == ["RFC", "RFC"]
    assert augmented["fpp_class_2"].tolist() == ["DECL", "INT"]
    assert augmented["spp_class_1"].tolist() == ["CONF", "DISC"]
    assert augmented["spp_class_2"].tolist() == ["SIMP", "CORR"]


def test_select_epochs_from_config_uses_metadata_event_type_fallback():
    metadata = pd.DataFrame(
        {
            "event_type": ["other_spp_onset", "self_spp_onset"],
            "latency": [0.2, 0.4],
        }
    )
    epochs = FakeEpochs(metadata)
    config = {
        "selection": {"event_type": "other_spp_onset"},
        "models": {"demo": {"formula": "~ latency", "test_predictors": ["latency"]}},
    }

    selected_epochs, selected_metadata = select_epochs_from_config(epochs, metadata, config)

    assert len(selected_epochs) == 1
    assert selected_metadata["event_type"].tolist() == ["other_spp_onset"]


def test_row_from_epochs_path_parses_subject_task_and_run():
    row = _row_from_epochs_path("/tmp/sub-001_task-conversation_run-2_desc-tasklocked_epo.fif")

    assert row == {
        "subject_id": "sub-001",
        "task": "conversation",
        "run": "2",
    }


def test_run_pooled_lmeeeg_analysis_uses_model_specific_input_modality(tmp_path, monkeypatch):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  models:
    demo:
      modality: "induced"
      formula: "~ run"
      test_predictors: ["run"]
induced_epochs:
  bands: ["theta"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class DummyEpochs:
        def __init__(self) -> None:
            self.ch_names = ["Cz"]
            self.times = np.asarray([0.1], dtype=float)

        def __len__(self) -> int:
            return 1

        def get_data(self, copy: bool = True) -> np.ndarray:
            return np.ones((1, 1, 1), dtype=float)

    loaded_paths: list[tuple[Path, Path | None]] = []

    def fake_load_epochs_with_metadata(epochs_path, *, metadata_csv=None):
        loaded_paths.append((Path(epochs_path), None if metadata_csv is None else Path(metadata_csv)))
        metadata = pd.DataFrame({"run": [1], "event_type": ["self_fpp_onset"]})
        return DummyEpochs(), metadata

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline.load_epochs_with_metadata", fake_load_epochs_with_metadata)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline.build_lmeeeg_trial_data_from_arrays",
        lambda **kwargs: SimpleNamespace(
            eeg_data=kwargs["eeg_data"],
            channel_names=kwargs["channel_names"],
            times=kwargs["times"],
        ),
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._fit_one_model",
        lambda runtime_config, trial_data, *, model_name, band_name=None: {"summary_output": f"{model_name}.json"},
    )
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._run_model_inference",
        lambda runtime_config, trial_data, *, model_name, band_name=None: [],
    )

    summary = run_pooled_lmeeeg_analysis(
        epochs_paths=[tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif"],
        config_path=config_path,
        output_dir=tmp_path / "out" / "lmeeeg",
    )

    assert loaded_paths
    assert loaded_paths[0][0].as_posix().endswith("/out/induced_epochs/theta/sub-001/epochs-time_s.fif")
    assert loaded_paths[0][1].as_posix().endswith("/out/induced_epochs/theta/sub-001/metadata-time_s.csv")
    assert summary["n_files_input"] == 1


def test_run_pooled_lmeeeg_analysis_expands_induced_model_over_multiple_bands(tmp_path, monkeypatch):
    config_path = tmp_path / "lmeeeg.yaml"
    config_path.write_text(
        """
lmeeeg:
  models:
    demo:
      modality: "induced"
      formula: "~ run"
      test_predictors: ["run"]
induced_epochs:
  bands: ["theta", "alpha", "beta"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class DummyEpochs:
        def __init__(self) -> None:
            self.ch_names = ["Cz"]
            self.times = np.asarray([0.1], dtype=float)

        def __len__(self) -> int:
            return 1

        def get_data(self, copy: bool = True) -> np.ndarray:
            return np.ones((1, 1, 1), dtype=float)

    loaded_paths: list[tuple[Path, Path | None]] = []
    fit_band_names: list[str | None] = []

    def fake_load_epochs_with_metadata(epochs_path, *, metadata_csv=None):
        loaded_paths.append((Path(epochs_path), None if metadata_csv is None else Path(metadata_csv)))
        metadata = pd.DataFrame({"run": [1], "event_type": ["self_fpp_onset"]})
        return DummyEpochs(), metadata

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline.load_epochs_with_metadata", fake_load_epochs_with_metadata)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline.build_lmeeeg_trial_data_from_arrays",
        lambda **kwargs: SimpleNamespace(
            eeg_data=kwargs["eeg_data"],
            channel_names=kwargs["channel_names"],
            times=kwargs["times"],
        ),
    )

    def fake_fit_one_model(runtime_config, trial_data, *, model_name, band_name=None):
        fit_band_names.append(band_name)
        return {"summary_output": f"{model_name}.json"}

    monkeypatch.setattr("cas.stats.lmeeeg_pipeline._fit_one_model", fake_fit_one_model)
    monkeypatch.setattr(
        "cas.stats.lmeeeg_pipeline._run_model_inference",
        lambda runtime_config, trial_data, *, model_name, band_name=None: [],
    )

    summary = run_pooled_lmeeeg_analysis(
        epochs_paths=[tmp_path / "sub-001_task-conversation_run-1_desc-tasklocked_epo.fif"],
        config_path=config_path,
        output_dir=tmp_path / "out" / "lmeeeg",
    )

    assert [path[0].parent.parent.name for path in loaded_paths] == ["theta", "alpha", "beta"]
    assert fit_band_names == ["theta", "alpha", "beta"]
    assert len(summary["models"]) == 3


def test_resolve_pooled_source_paths_keeps_evoked_input_path():
    runtime_config = {
        "paths": {"out_dir": "/tmp/out"},
        "lmeeeg": {
            "models": {
                "demo": {
                    "formula": "~ run",
                }
            }
        },
    }

    epochs_path, metadata_path = _resolve_pooled_source_paths(
        runtime_config=runtime_config,
        model_name="demo",
        epochs_path="/tmp/out/epochs/sub-001/eeg/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif",
    )

    assert epochs_path.as_posix().endswith("/tmp/out/epochs/sub-001/eeg/sub-001_task-conversation_run-1_desc-tasklocked_epo.fif")
    assert metadata_path is None
