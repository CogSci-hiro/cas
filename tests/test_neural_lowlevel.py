from __future__ import annotations

import mne
import numpy as np

from cas.neural.lowlevel import build_lowlevel_neural_feature_table


def test_build_lowlevel_neural_feature_table_exports_expected_columns() -> None:
    sfreq = 64.0
    times = np.arange(0.0, 5.0, 1.0 / sfreq, dtype=float)
    signal_a = np.sin(2 * np.pi * 10.0 * times)
    signal_b = np.sin(2 * np.pi * 20.0 * times)
    raw = mne.io.RawArray(
        np.vstack([signal_a, signal_b]),
        mne.create_info(["Cz", "Pz"], sfreq=sfreq, ch_types=["eeg", "eeg"]),
        verbose="ERROR",
    )

    table = build_lowlevel_neural_feature_table(
        raw,
        dyad_id="dyad-001",
        run="1",
        speaker="A",
    )

    assert list(table.columns[:4]) == ["dyad_id", "run", "speaker", "time"]
    assert {"amp_Cz", "amp_Pz", "alpha_Cz", "alpha_Pz", "beta_Cz", "beta_Pz"} <= set(table.columns)
    assert len(table) == len(times)
    assert table["dyad_id"].nunique() == 1
    assert table["speaker"].iloc[0] == "A"
