"""Posterior-entropy helpers for the hazard analysis."""

from __future__ import annotations

import numpy as np


def compute_posterior_entropy(
    posterior_probabilities: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Compute Shannon entropy from posterior state probabilities.

    Parameters
    ----------
    posterior_probabilities
        Array with shape ``(n_samples, n_states)`` containing posterior state
        probabilities.
    epsilon
        Small positive value used to stabilize ``log`` at 0.

    Returns
    -------
    numpy.ndarray
        Entropy vector with shape ``(n_samples,)``.
    """

    posterior = np.asarray(posterior_probabilities, dtype=float)
    if posterior.ndim != 2:
        raise ValueError("`posterior_probabilities` must be 2D.")
    if posterior.shape[1] == 0:
        raise ValueError("`posterior_probabilities` must contain at least one state.")
    if not np.isfinite(posterior).all():
        raise ValueError("`posterior_probabilities` contains NaN or infinite values.")
    if np.any(posterior < 0.0):
        raise ValueError("`posterior_probabilities` cannot contain negative values.")

    stabilized = np.clip(posterior, epsilon, 1.0)
    entropy = -np.sum(stabilized * np.log(stabilized), axis=1)
    if not np.isfinite(entropy).all():
        raise ValueError("Entropy contains NaN or infinite values after processing.")
    return entropy


def normalize_entropy(entropy: np.ndarray, n_states: int) -> np.ndarray:
    """Normalize entropy by ``log(K)``.

    Parameters
    ----------
    entropy
        Entropy vector.
    n_states
        Number of HMM states ``K``.

    Returns
    -------
    numpy.ndarray
        Normalized entropy vector in ``[0, 1]`` when the posterior is valid.
    """

    if n_states <= 1:
        raise ValueError("`n_states` must exceed 1 for log(K) normalization.")
    values = np.asarray(entropy, dtype=float)
    return values / float(np.log(n_states))
