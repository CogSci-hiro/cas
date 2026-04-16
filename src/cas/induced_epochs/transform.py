from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np
from scipy.signal import hilbert


DEFAULT_BANDS_HZ: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def resolve_induced_band_names(config: Mapping[str, Any]) -> list[str]:
    induced_cfg = dict(config.get("induced_epochs") or {})
    bands = induced_cfg.get("bands")
    if bands is None:
        return ["theta"]
    if not isinstance(bands, list) or not bands:
        raise ValueError("`induced_epochs.bands` must be a non-empty list.")
    return [str(band) for band in bands]


def resolve_induced_band_limits_hz(band_name: str, config: Mapping[str, Any]) -> tuple[float, float]:
    induced_cfg = dict(config.get("induced_epochs") or {})
    custom_limits = induced_cfg.get("band_limits_hz") or {}
    if band_name in custom_limits:
        low_hz, high_hz = custom_limits[band_name]
        return float(low_hz), float(high_hz)
    if band_name not in DEFAULT_BANDS_HZ:
        raise ValueError(f"Unknown induced band '{band_name}'.")
    return DEFAULT_BANDS_HZ[band_name]


def build_induced_epochs(source_epochs, *, band_name: str, config: Mapping[str, Any]):
    """Compute band-limited Hilbert envelope epochs while preserving structure."""
    _configure_mne_runtime()
    import mne

    low_hz, high_hz = resolve_induced_band_limits_hz(band_name, config)
    data = source_epochs.get_data(copy=True)
    sfreq = float(source_epochs.info["sfreq"])
    filtered = mne.filter.filter_data(
        data,
        sfreq=sfreq,
        l_freq=low_hz,
        h_freq=high_hz,
        verbose="ERROR",
    )
    envelope = np.abs(hilbert(filtered, axis=-1))

    induced = mne.EpochsArray(
        envelope,
        source_epochs.info.copy(),
        events=source_epochs.events.copy(),
        tmin=float(source_epochs.times[0]),
        event_id=dict(source_epochs.event_id),
        metadata=None if source_epochs.metadata is None else source_epochs.metadata.copy().reset_index(drop=True),
        verbose="ERROR",
    )
    return induced
