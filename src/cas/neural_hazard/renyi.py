from __future__ import annotations

import re

import numpy as np
import pandas as pd


def discover_state_probability_columns(df: pd.DataFrame) -> list[str]:
    """Discover HMM state-probability columns.

    Parameters
    ----------
    df
        Input table containing neural features.

    Returns
    -------
    list of str
        Matched state-probability column names.
    """

    patterns = [
        re.compile(r"^state_probability_\d+$"),
        re.compile(r"^state_prob_\d+$"),
        re.compile(r"^gamma_\d+$"),
    ]
    cols = [c for c in df.columns if any(p.match(str(c)) for p in patterns)]
    return sorted(cols)


def compute_renyi_entropy(
    state_probabilities: np.ndarray,
    alpha: float,
    epsilon: float = 1e-12,
    alpha_one_tolerance: float = 1e-6,
) -> np.ndarray:
    """Compute per-row Rényi entropy from state probabilities.

    Parameters
    ----------
    state_probabilities : np.ndarray
        2D array of shape (n_samples, n_states).
    alpha : float
        Rényi alpha parameter.
    epsilon : float, optional
        Minimum probability used before log/power operations.
    alpha_one_tolerance : float, optional
        If ``abs(alpha - 1) < alpha_one_tolerance``, uses Shannon entropy.

    Returns
    -------
    np.ndarray
        Entropy vector of shape (n_samples,).
    """

    probs = np.asarray(state_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("state_probabilities must be a 2D array (n_samples, n_states).")
    if probs.shape[1] < 1:
        raise ValueError("state_probabilities must have at least one state column.")
    if not np.isfinite(probs).all():
        raise ValueError("state_probabilities contains non-finite values.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")

    row_sums_before = probs.sum(axis=1)
    if np.any(~np.isfinite(row_sums_before)) or np.any(row_sums_before <= 0.0):
        raise ValueError("Each probability row must have a finite positive sum before normalization.")

    clipped = np.clip(probs, epsilon, None)
    clipped_sum = clipped.sum(axis=1, keepdims=True)
    if np.any(clipped_sum <= 0.0) or np.any(~np.isfinite(clipped_sum)):
        raise ValueError("Probability row sums became invalid after clipping.")
    p = clipped / clipped_sum

    if abs(float(alpha) - 1.0) < float(alpha_one_tolerance):
        return -np.sum(p * np.log(p), axis=1)

    a = float(alpha)
    renyi_inner = np.sum(np.power(p, a), axis=1)
    return (1.0 / (1.0 - a)) * np.log(renyi_inner)
