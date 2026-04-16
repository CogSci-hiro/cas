"""Precomputed ICA helpers for EEG preprocessing."""

from __future__ import annotations

from pathlib import Path

import mne


def load_ica(ica_path: str | Path) -> mne.preprocessing.ICA:
    """Load a precomputed ICA solution from disk.

    Parameters
    ----------
    ica_path
        Path to a saved MNE ICA file.

    Returns
    -------
    mne.preprocessing.ICA
        Loaded ICA object.

    Raises
    ------
    FileNotFoundError
        If the ICA file does not exist.

    Usage example
    -------------
    >>> ica = load_ica("sub-001_ica.fif")
    """
    resolved_ica_path = Path(ica_path)
    if not resolved_ica_path.exists():
        raise FileNotFoundError(f"ICA file not found: {resolved_ica_path}")

    return mne.preprocessing.read_ica(resolved_ica_path)


def apply_precomputed_ica(
    raw: mne.io.BaseRaw,
    ica_path: str | Path,
) -> mne.io.BaseRaw:
    """Apply a precomputed ICA solution to a raw EEG recording.

    Parameters
    ----------
    raw
        Raw EEG recording.
    ica_path
        Path to a saved MNE ICA file.

    Returns
    -------
    mne.io.BaseRaw
        The same raw object after ICA application.

    Usage example
    -------------
    >>> raw = apply_precomputed_ica(raw, "sub-001_ica.fif")
    """
    ica = load_ica(ica_path)
    ica.apply(raw)
    return raw
