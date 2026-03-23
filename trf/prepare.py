"""Utilities to prepare run-wise data for TRF models."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def _as_2d(a: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"`{name}` must be 1D or 2D per run.")
    if x.shape[0] < 2:
        raise ValueError(f"`{name}` run must have at least 2 samples.")
    if not np.isfinite(x).all():
        raise ValueError(f"`{name}` contains NaN or infinite values.")
    return x


def _resample_run(x: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    if src_sfreq <= 0 or dst_sfreq <= 0:
        raise ValueError("Sampling rates must be > 0.")
    if np.isclose(src_sfreq, dst_sfreq):
        return x.copy()
    ratio = Fraction(dst_sfreq / src_sfreq).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def _lag_matrix(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    n_samples, n_features = x.shape
    out = np.zeros((n_samples, n_features * len(lags)), dtype=float)
    for i, lag in enumerate(lags):
        c0, c1 = i * n_features, (i + 1) * n_features
        if lag == 0:
            out[:, c0:c1] = x
        elif lag > 0:
            out[lag:, c0:c1] = x[:-lag, :]
        else:
            out[:lag, c0:c1] = x[-lag:, :]
    return out


def prepare_trf_runs(
    eeg_runs: list[np.ndarray],
    predictor_runs: list[np.ndarray],
    eeg_sfreq: float,
    predictor_sfreq: float,
    target_sfreq: float,
    tmin_s: float,
    tmax_s: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resample run-wise EEG/predictors and build lagged predictor matrices."""
    if len(eeg_runs) != len(predictor_runs):
        raise ValueError("`eeg_runs` and `predictor_runs` must have same number of runs.")
    if len(eeg_runs) == 0:
        raise ValueError("At least one run is required.")
    if tmin_s > tmax_s:
        raise ValueError("`tmin_s` must be <= `tmax_s`.")

    lag_start = int(np.round(tmin_s * target_sfreq))
    lag_stop = int(np.round(tmax_s * target_sfreq))
    lags = np.arange(lag_start, lag_stop + 1, dtype=int)
    if lags.size == 0:
        raise ValueError("No lags were generated from `[tmin_s, tmax_s]`.")

    X_runs: list[np.ndarray] = []
    Y_runs: list[np.ndarray] = []
    for run_idx, (eeg_run, pred_run) in enumerate(zip(eeg_runs, predictor_runs)):
        eeg = _as_2d(eeg_run, name=f"eeg_runs[{run_idx}]")
        pred = _as_2d(pred_run, name=f"predictor_runs[{run_idx}]")

        eeg_r = _resample_run(eeg, eeg_sfreq, target_sfreq)
        pred_r = _resample_run(pred, predictor_sfreq, target_sfreq)

        if eeg_r.shape[0] != pred_r.shape[0]:
            raise ValueError(
                f"Run {run_idx} length mismatch after resampling: "
                f"EEG={eeg_r.shape[0]} vs predictor={pred_r.shape[0]}."
            )
        if not np.isfinite(eeg_r).all() or not np.isfinite(pred_r).all():
            raise ValueError(f"Run {run_idx} contains NaN or infinite after resampling.")

        X_runs.append(_lag_matrix(pred_r, lags))
        Y_runs.append(eeg_r)

    return X_runs, Y_runs

