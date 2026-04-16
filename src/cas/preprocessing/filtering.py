"""Filtering helpers for EEG preprocessing."""

from __future__ import annotations

from typing import Any, Mapping

import mne


def bandpass_filter_raw(
    raw: mne.io.BaseRaw,
    low_cut_hz: float | None,
    high_cut_hz: float | None,
) -> mne.io.BaseRaw:
    """Apply broad filtering to a raw EEG recording."""

    if low_cut_hz is None and high_cut_hz is None:
        return raw

    if low_cut_hz is not None and low_cut_hz < 0:
        raise ValueError("High-pass cutoff must be non-negative.")
    if high_cut_hz is not None and high_cut_hz <= 0:
        raise ValueError("Low-pass cutoff must be positive.")
    if low_cut_hz is not None and high_cut_hz is not None and low_cut_hz >= high_cut_hz:
        raise ValueError("High-pass cutoff must be lower than low-pass cutoff.")

    raw.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, picks="eeg")
    return raw


def apply_bandpass_filter(raw: mne.io.BaseRaw, config: Mapping[str, Any]) -> mne.io.BaseRaw:
    """Apply config-driven band-pass filtering."""

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    filter_cfg = dict(preprocessing_cfg.get("filter") or {})

    low_cut_hz = filter_cfg.get("l_freq_hz", preprocessing_cfg.get("highpass_hz"))
    high_cut_hz = filter_cfg.get("h_freq_hz", preprocessing_cfg.get("lowpass_hz"))
    return bandpass_filter_raw(
        raw,
        low_cut_hz=float(low_cut_hz) if low_cut_hz is not None else None,
        high_cut_hz=float(high_cut_hz) if high_cut_hz is not None else None,
    )
