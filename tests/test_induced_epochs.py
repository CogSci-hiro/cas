from __future__ import annotations

import numpy as np
import pandas as pd

from rate.induced_epochs.io import build_induced_epoch_band_paths
from rate.induced_epochs.transform import (
    build_induced_epochs,
    resolve_induced_band_names,
)


def _make_source_epochs():
    import mne

    sfreq = 128.0
    n_epochs = 3
    n_channels = 2
    n_times = 512
    times = np.arange(n_times, dtype=float) / sfreq
    oscillation = np.sin(2.0 * np.pi * 6.0 * times)
    data = np.tile(oscillation, (n_epochs, n_channels, 1))
    data[1] *= 0.5
    data[2] *= 1.5

    info = mne.create_info(["Cz", "Pz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    events = np.array(
        [
            [100, 0, 1],
            [300, 0, 1],
            [500, 0, 2],
        ],
        dtype=int,
    )
    metadata = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001", "sub-001"],
            "event_type": ["a", "a", "b"],
        }
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-1.0,
        event_id={"a": 1, "b": 2},
        metadata=metadata,
        verbose="ERROR",
    )


def test_induced_epochs_default_band_is_theta():
    assert resolve_induced_band_names({}) == ["theta"]


def test_build_induced_epochs_preserves_epoch_structure():
    source_epochs = _make_source_epochs()
    induced_epochs = build_induced_epochs(source_epochs, band_name="theta", config={})

    assert len(induced_epochs) == len(source_epochs)
    assert induced_epochs.ch_names == source_epochs.ch_names
    assert np.array_equal(induced_epochs.events, source_epochs.events)
    assert induced_epochs.event_id == source_epochs.event_id
    assert np.allclose(induced_epochs.times, source_epochs.times)
    assert induced_epochs.metadata is not None
    pd.testing.assert_frame_equal(
        induced_epochs.metadata.reset_index(drop=True),
        source_epochs.metadata.reset_index(drop=True),
    )
    assert induced_epochs.get_data(copy=True).shape == source_epochs.get_data(copy=True).shape


def test_induced_epoch_paths_keep_evoked_epoch_filenames():
    config = {"paths": {"out_dir": "/tmp/out"}}
    row = {"subject_id": "sub-001", "task": "talk", "run": 1, "dyad_id": "dyad-001", "eeg_path": "unused"}

    output_paths = build_induced_epoch_band_paths(config, row, band_name="theta")

    assert output_paths.epochs_output_path.as_posix().endswith("/induced_epochs/theta/sub-001/epochs-time_s.fif")
    assert output_paths.metadata_output_path.as_posix().endswith("/induced_epochs/theta/sub-001/metadata-time_s.csv")
