"""Downsampling utilities for EEG preprocessing."""

from __future__ import annotations

from typing import Any, Mapping

import mne

SAMPLING_RATE_TOLERANCE_HZ = 1e-9


def _resolve_sampling_rate_hz(
    sampling_rate_hz: float | Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None = None,
) -> float | None:
    if isinstance(sampling_rate_hz, Mapping):
        config = sampling_rate_hz
        sampling_rate_hz = None

    if config is not None:
        preprocessing_cfg = dict(config.get("preprocessing") or {})
        downsample_cfg = dict(preprocessing_cfg.get("downsample") or {})
        configured = downsample_cfg.get("sfreq_hz", preprocessing_cfg.get("downsample_hz"))
        if configured is not None:
            return float(configured)

    if sampling_rate_hz is None:
        return None
    return float(sampling_rate_hz)


def downsample_raw(
    raw: mne.io.BaseRaw,
    sampling_rate_hz: float | Mapping[str, Any] | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> mne.io.BaseRaw:
    """Resample a raw EEG recording to a target sampling rate."""

    resolved_sampling_rate_hz = _resolve_sampling_rate_hz(sampling_rate_hz, config=config)
    if resolved_sampling_rate_hz is None:
        return raw

    if resolved_sampling_rate_hz <= 0:
        raise ValueError("Target sampling rate must be positive.")

    current_sampling_rate_hz = float(raw.info["sfreq"])
    if abs(current_sampling_rate_hz - resolved_sampling_rate_hz) <= SAMPLING_RATE_TOLERANCE_HZ:
        return raw

    if resolved_sampling_rate_hz > current_sampling_rate_hz:
        raise ValueError(
            "Target sampling rate cannot exceed the current sampling rate. "
            f"Current: {current_sampling_rate_hz} Hz, requested: {resolved_sampling_rate_hz} Hz."
        )

    raw.resample(sfreq=resolved_sampling_rate_hz)
    return raw
