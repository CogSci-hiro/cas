from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def downsample_power_time(
    power_array: np.ndarray,
    times: np.ndarray,
    target_sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average power over time bins while preserving non-time dimensions."""
    values = np.asarray(power_array, dtype=float)
    time_values = np.asarray(times, dtype=float)

    if values.ndim < 1:
        raise ValueError("power_array must have at least one dimension.")
    if time_values.ndim != 1:
        raise ValueError("times must be a 1D array.")
    if values.shape[-1] != time_values.shape[0]:
        raise ValueError("The last axis of power_array must match the time axis length.")
    if time_values.size < 2:
        raise ValueError("At least two time points are required for downsampling.")
    if target_sfreq <= 0.0:
        raise ValueError("target_sfreq must be positive.")

    diffs = np.diff(time_values)
    if not np.all(diffs > 0):
        raise ValueError("times must be strictly increasing.")

    current_sfreq = 1.0 / float(np.median(diffs))
    if target_sfreq >= current_sfreq:
        return values.copy(), time_values.copy()

    bin_width_s = 1.0 / float(target_sfreq)
    start_s = float(time_values[0] - (diffs[0] / 2.0))
    stop_s = float(time_values[-1] + (diffs[-1] / 2.0))

    edges = np.arange(start_s, stop_s + bin_width_s, bin_width_s, dtype=float)
    if edges[-1] < stop_s:
        edges = np.append(edges, stop_s)

    bin_ids = np.digitize(time_values, edges[1:-1], right=False)
    kept_bins = np.unique(bin_ids)

    downsampled = []
    downsampled_times = []
    for bin_id in kept_bins:
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        downsampled.append(values[..., mask].mean(axis=-1))
        downsampled_times.append(float((edges[bin_id] + edges[bin_id + 1]) / 2.0))

    if not downsampled:
        raise ValueError("Downsampling produced zero bins.")

    return (
        np.stack(downsampled, axis=-1).astype(values.dtype, copy=False),
        np.asarray(downsampled_times, dtype=float),
    )


def resolve_downsampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the induced-power downsampling configuration section."""
    lmeeeg_cfg = dict(config.get("lmeeeg") or config)
    return dict(lmeeeg_cfg.get("post_power_downsampling") or {})
