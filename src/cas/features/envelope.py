"""Speech envelope extraction utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def extract_hilbert_envelope(
    signal: np.ndarray,
    sampling_rate_hz: float,
    lowpass_hz: float = 10.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Extract a speech envelope using a Hilbert-based pipeline.

    Pipeline:
    1. Hilbert transform
    2. Magnitude (absolute value)
    3. Low-pass filter
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("`signal` must be a 1D array.")

    if sampling_rate_hz <= 0:
        raise ValueError("`sampling_rate_hz` must be > 0.")

    nyquist_hz = sampling_rate_hz / 2.0
    if not 0 < lowpass_hz < nyquist_hz:
        raise ValueError(
            f"`lowpass_hz` must be in (0, {nyquist_hz:.3f}) for the given sampling rate."
        )

    if filter_order < 1:
        raise ValueError("`filter_order` must be >= 1.")

    analytic_signal = hilbert(x)
    envelope = np.abs(analytic_signal)

    b, a = butter(filter_order, lowpass_hz / nyquist_hz, btype="low")
    filtered_envelope = filtfilt(b, a, envelope)
    return np.asarray(filtered_envelope)

