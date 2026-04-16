"""Rereferencing helpers for EEG preprocessing."""

from __future__ import annotations

import mne


def apply_average_reference(
    raw: mne.io.BaseRaw,
    projection: bool = False,
) -> mne.io.BaseRaw:
    """Apply an average EEG reference to a raw recording.

    Parameters
    ----------
    raw
        Raw EEG recording.
    projection
        If ``True``, add an average-reference projection instead of applying
        the rereference directly.

    Returns
    -------
    mne.io.BaseRaw
        The same raw object after average rereferencing.

    Usage example
    -------------
    >>> raw = apply_average_reference(raw)
    """
    raw.set_eeg_reference(
        ref_channels="average",
        projection=projection,
        ch_type="eeg",
    )
    return raw
