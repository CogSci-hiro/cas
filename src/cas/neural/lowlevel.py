"""Continuous low-level neural feature tables for hazard modelling."""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

DEFAULT_BANDS_HZ: dict[str, tuple[float, float]] = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def export_lowlevel_neural_feature_table(
    *,
    raw_path: str | Path,
    output_path: str | Path,
    dyad_id: str,
    run: str | int,
    speaker: str,
) -> Path:
    """Read one preprocessed EEG recording and export a low-level feature TSV."""

    _configure_mne_runtime()
    print(
        f"[lowlevel-neural] reading raw={raw_path} dyad_id={dyad_id} run={run} speaker={speaker}",
        flush=True,
    )
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
    table = build_lowlevel_neural_feature_table(
        raw,
        dyad_id=dyad_id,
        run=run,
        speaker=speaker,
    )
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(resolved_output_path, sep="\t", index=False)
    print(
        f"[lowlevel-neural] wrote rows={len(table)} columns={len(table.columns)} output={resolved_output_path}",
        flush=True,
    )
    return resolved_output_path


def build_lowlevel_neural_feature_table(
    raw: mne.io.BaseRaw,
    *,
    dyad_id: str,
    run: str | int,
    speaker: str,
) -> pd.DataFrame:
    """Build amplitude, alpha-envelope, and beta-envelope features over time."""

    working = raw.copy().pick("eeg")
    if len(working.ch_names) == 0:
        raise ValueError("Low-level neural feature export requires at least one EEG channel.")

    data = working.get_data()
    times = np.asarray(working.times, dtype=float)
    channel_names = [_sanitize_feature_name(channel_name) for channel_name in working.ch_names]

    output = pd.DataFrame(
        {
            "dyad_id": str(dyad_id),
            "run": str(run),
            "speaker": str(speaker),
            "time": times,
        }
    )
    _append_channelwise_features(output, prefix="amp_", values=data, channel_names=channel_names)

    sfreq = float(working.info["sfreq"])
    for band_name, (low_hz, high_hz) in DEFAULT_BANDS_HZ.items():
        print(
            f"[lowlevel-neural] computing {band_name} envelope for dyad_id={dyad_id} run={run} speaker={speaker}",
            flush=True,
        )
        band_data = mne.filter.filter_data(
            data.copy(),
            sfreq=sfreq,
            l_freq=low_hz,
            h_freq=high_hz,
            verbose="ERROR",
        )
        envelope = np.abs(hilbert(band_data, axis=-1))
        _append_channelwise_features(output, prefix=f"{band_name}_", values=envelope, channel_names=channel_names)
    return output


def _append_channelwise_features(
    table: pd.DataFrame,
    *,
    prefix: str,
    values: np.ndarray,
    channel_names: list[str],
) -> None:
    sample_by_channel = np.asarray(values, dtype=float).T
    for column_index, channel_name in enumerate(channel_names):
        table[f"{prefix}{channel_name}"] = sample_by_channel[:, column_index]


def _sanitize_feature_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "channel"


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
