from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np


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
    """Compute single-trial Morlet induced-power epochs while preserving structure.

    The returned epochs contain band-averaged induced power (mean across sampled
    frequencies within the requested band), not evoked power from averaged ERPs.
    """
    _configure_mne_runtime()
    import mne

    low_hz, high_hz = resolve_induced_band_limits_hz(band_name, config)
    induced_cfg = dict(config.get("induced_epochs") or {})
    morlet_cfg = dict(induced_cfg.get("morlet") or {})
    freq_step_hz = float(morlet_cfg.get("freq_step_hz", 1.0))
    if freq_step_hz <= 0.0:
        raise ValueError("induced_epochs.morlet.freq_step_hz must be positive.")
    n_cycles_cfg = morlet_cfg.get("n_cycles", {"mode": "frequency_divisor", "divisor": 2.0})
    use_fft = bool(morlet_cfg.get("use_fft", True))
    decim = int(morlet_cfg.get("decim", 1))

    frequencies = np.arange(low_hz, high_hz + (0.5 * freq_step_hz), freq_step_hz, dtype=float)
    if frequencies.size == 0:
        frequencies = np.asarray([0.5 * (low_hz + high_hz)], dtype=float)

    if isinstance(n_cycles_cfg, dict):
        mode = str(n_cycles_cfg.get("mode", "frequency_divisor"))
        if mode != "frequency_divisor":
            raise ValueError("induced_epochs.morlet.n_cycles.mode must be 'frequency_divisor'.")
        divisor = float(n_cycles_cfg.get("divisor", 2.0))
        if divisor <= 0.0:
            raise ValueError("induced_epochs.morlet.n_cycles.divisor must be positive.")
        n_cycles = frequencies / divisor
    elif isinstance(n_cycles_cfg, (int, float)):
        n_cycles = np.full(frequencies.shape, float(n_cycles_cfg), dtype=float)
    else:
        n_cycles = np.asarray(n_cycles_cfg, dtype=float)
        if n_cycles.shape != frequencies.shape:
            raise ValueError(
                "induced_epochs.morlet.n_cycles must be scalar or match the frequency grid length."
            )

    data = source_epochs.get_data(copy=True)
    sfreq = float(source_epochs.info["sfreq"])
    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=frequencies,
        n_cycles=n_cycles,
        output="power",
        use_fft=use_fft,
        decim=decim,
        zero_mean=True,
        verbose="ERROR",
    )
    # Average over sampled frequencies to keep a band-level trial x channel x time array.
    band_power = power.mean(axis=2)

    induced = mne.EpochsArray(
        band_power.astype(np.float32, copy=False),
        source_epochs.info.copy(),
        events=source_epochs.events.copy(),
        tmin=float(source_epochs.times[0]),
        event_id=dict(source_epochs.event_id),
        metadata=None if source_epochs.metadata is None else source_epochs.metadata.copy().reset_index(drop=True),
        verbose="ERROR",
    )
    return induced
