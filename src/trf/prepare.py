"""Lightweight TRF run preparation helpers."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def _as_2d_array(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"{label} must be 1D or 2D, got shape {values.shape}.")
    return values


def _resample_run(array: np.ndarray, *, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(array, dtype=float)
    source_hz = float(source_sfreq)
    target_hz = float(target_sfreq)
    scale = 1000
    up = int(round(target_hz * scale))
    down = int(round(source_hz * scale))
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor
    return np.asarray(resample_poly(np.asarray(array, dtype=float), up, down, axis=0), dtype=float)


def prepare_trf_runs(
    *,
    eeg_runs: list[np.ndarray],
    predictor_runs: list[np.ndarray],
    eeg_sfreq: float,
    predictor_sfreq: float,
    target_sfreq: float,
    tmin_s: float,
    tmax_s: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Validate and optionally resample TRF inputs run by run."""

    del tmin_s, tmax_s

    if len(eeg_runs) != len(predictor_runs):
        raise ValueError(
            f"Expected the same number of EEG and predictor runs, got {len(eeg_runs)} and {len(predictor_runs)}."
        )

    prepared_X: list[np.ndarray] = []
    prepared_Y: list[np.ndarray] = []
    for run_index, (predictor_run, eeg_run) in enumerate(zip(predictor_runs, eeg_runs), start=1):
        X = _as_2d_array(predictor_run, label=f"predictor run {run_index}")
        Y = _as_2d_array(eeg_run, label=f"EEG run {run_index}")

        X = _resample_run(X, source_sfreq=float(predictor_sfreq), target_sfreq=float(target_sfreq))
        Y = _resample_run(Y, source_sfreq=float(eeg_sfreq), target_sfreq=float(target_sfreq))

        n_samples = min(X.shape[0], Y.shape[0])
        if n_samples <= 1:
            raise ValueError(f"Run {run_index} is too short after alignment: {n_samples} samples.")

        prepared_X.append(np.asarray(X[:n_samples], dtype=float))
        prepared_Y.append(np.asarray(Y[:n_samples], dtype=float))

    return prepared_X, prepared_Y
