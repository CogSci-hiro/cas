"""Legacy CAS EEG preprocessing pipeline.

This module preserves an older CAS implementation for reference and rollback.
Current preprocessing entrypoints live in ``cas.preprocessing.pipeline``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path

import mne
import numpy as np

from cas.preprocessing.bad_channels import (
    read_bad_channels_from_bids_tsv,
    set_bad_channels,
)
from cas.preprocessing.channels import split_channels
from cas.preprocessing.downsample import downsample_raw
from cas.preprocessing.events import extract_events
from cas.preprocessing.filtering import bandpass_filter_raw
from cas.preprocessing.ica import apply_precomputed_ica
from cas.preprocessing.rereference import apply_average_reference

LOGGER = logging.getLogger(__name__)
_MISSING_DIGITIZATION_ERROR = 'Cannot fit headshape without digitization, info["dig"] is None'
_NONFINITE_INTERPOLATION_ERROR = "array must not contain infs or NaNs"
_DEFAULT_INTERPOLATION_ORIGIN_M = (0.0, 0.0, 0.04)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """All in-memory outputs of preprocessing for one run."""

    eeg_raw: mne.io.BaseRaw
    emg_data: np.ndarray
    emg_channel_names: list[str]
    dropped_channel_names: list[str]
    events_rows: list[dict[str, str | int | float]]
    summary: dict[str, object]


def _interpolate_bad_channels(raw: mne.io.BaseRaw) -> None:
    """Interpolate bad EEG channels, degrading gracefully when geometry is incomplete."""
    try:
        raw.interpolate_bads()
    except RuntimeError as exc:
        if _MISSING_DIGITIZATION_ERROR not in str(exc):
            raise

        LOGGER.warning(
            "Missing digitization metadata for bad-channel interpolation; "
            "retrying with fixed origin %s m.",
            _DEFAULT_INTERPOLATION_ORIGIN_M,
        )
        try:
            raw.interpolate_bads(origin=_DEFAULT_INTERPOLATION_ORIGIN_M)
        except ValueError as value_exc:
            if _NONFINITE_INTERPOLATION_ERROR not in str(value_exc):
                raise
            LOGGER.warning(
                "Skipping bad-channel interpolation because channel geometry "
                "still produced non-finite values after fixed-origin fallback."
            )
    except ValueError as exc:
        if _NONFINITE_INTERPOLATION_ERROR not in str(exc):
            raise
        LOGGER.warning(
            "Skipping bad-channel interpolation because channel geometry "
            "contains non-finite values."
        )


def preprocess_run(
    raw: mne.io.BaseRaw,
    *,
    channels_tsv_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
    ica_path: str | Path | None = None,
    target_sampling_rate_hz: float | None = None,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    montage_name: str | None = "standard_1020",
    annotation_pairing_margin_s: float = 1.0,
    eeg_channel_names: list[str] | None = None,
    eeg_channel_patterns: list[str] | None = None,
    emg_channel_names: list[str] | None = None,
    emg_channel_patterns: list[str] | None = None,
    interpolate_bad_channels: bool = True,
    apply_rereference: bool = True,
    keep_emg: bool = True,
) -> PreprocessingResult:
    """Run the original hard-coded preprocessing pipeline for one recording."""
    raw_copy = raw.copy()
    sampling_rate_before_hz = float(raw_copy.info["sfreq"])

    extracted_events = extract_events(
        raw_copy,
        annotation_path=annotation_path,
        annotation_pairing_margin_s=annotation_pairing_margin_s,
    )

    channel_split = split_channels(
        raw_copy,
        eeg_channel_names=eeg_channel_names,
        eeg_channel_patterns=eeg_channel_patterns,
        emg_channel_names=emg_channel_names,
        emg_channel_patterns=emg_channel_patterns,
    )

    eeg_raw = raw_copy.copy().pick(channel_split.eeg_channel_names)
    if keep_emg and channel_split.emg_channel_names:
        emg_raw = raw_copy.copy().pick(channel_split.emg_channel_names)
        emg_data = emg_raw.get_data()
        emg_sampling_rate_hz = float(emg_raw.info["sfreq"])
    else:
        emg_data = np.empty((0, 0), dtype=float)
        emg_sampling_rate_hz = None

    if montage_name:
        LOGGER.info("Applying montage %s to EEG channels.", montage_name)
        eeg_raw.set_montage(montage_name, on_missing="ignore")

    loaded_bad_channel_names: list[str] = []
    if channels_tsv_path is not None:
        channels_path = Path(channels_tsv_path)
        LOGGER.info("Loading bad channels from %s.", channels_path)
        loaded_bad_channel_names = read_bad_channels_from_bids_tsv(channels_path)
        LOGGER.info("Setting %d bad channels on EEG raw.", len(loaded_bad_channel_names))
        set_bad_channels(eeg_raw, loaded_bad_channel_names)

    if target_sampling_rate_hz is not None:
        LOGGER.info("Downsampling EEG to %.3f Hz.", target_sampling_rate_hz)
        downsample_raw(eeg_raw, sampling_rate_hz=target_sampling_rate_hz)

    if low_cut_hz is not None or high_cut_hz is not None:
        LOGGER.info(
            "Applying EEG filter with low_cut_hz=%s and high_cut_hz=%s.",
            low_cut_hz,
            high_cut_hz,
        )
        bandpass_filter_raw(
            eeg_raw,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
        )

    if ica_path is not None:
        resolved_ica_path = Path(ica_path)
        if resolved_ica_path.exists():
            LOGGER.info("Applying precomputed ICA from %s.", resolved_ica_path)
            apply_precomputed_ica(eeg_raw, resolved_ica_path)
        else:
            LOGGER.info("ICA file not found at %s; skipping ICA.", resolved_ica_path)

    bad_channel_names_before_interpolation = list(eeg_raw.info.get("bads", []))
    if interpolate_bad_channels and bad_channel_names_before_interpolation:
        LOGGER.info(
            "Interpolating %d bad EEG channels.",
            len(bad_channel_names_before_interpolation),
        )
        _interpolate_bad_channels(eeg_raw)
    elif interpolate_bad_channels:
        LOGGER.info("Skipping bad channel interpolation because no bad channels are set.")

    if apply_rereference:
        LOGGER.info("Applying average EEG reference.")
        apply_average_reference(eeg_raw)

    event_source_counts = Counter(str(row.get("source", "")) for row in extracted_events.rows)
    summary = {
        "sampling_rate_before_hz": sampling_rate_before_hz,
        "sampling_rate_after_hz": float(eeg_raw.info["sfreq"]),
        "event_source": extracted_events.source,
        "event_count": len(extracted_events.rows),
        "event_source_counts": dict(sorted(event_source_counts.items())),
        "loaded_bad_channels": loaded_bad_channel_names,
        "bad_channels_before_interpolation": bad_channel_names_before_interpolation,
        "bad_channels_after_preprocessing": list(eeg_raw.info.get("bads", [])),
        "kept_eeg_channels": channel_split.eeg_channel_names,
        "kept_emg_channels": channel_split.emg_channel_names if keep_emg else [],
        "dropped_channels": channel_split.dropped_channel_names,
        "montage": montage_name,
        "emg_sampling_rate_hz": emg_sampling_rate_hz,
    }

    return PreprocessingResult(
        eeg_raw=eeg_raw,
        emg_data=emg_data,
        emg_channel_names=channel_split.emg_channel_names if keep_emg else [],
        dropped_channel_names=channel_split.dropped_channel_names,
        events_rows=extracted_events.rows,
        summary=summary,
    )


def preprocess_raw(
    raw: mne.io.BaseRaw,
    *,
    channels_tsv_path: str | Path | None = None,
    annotation_path: str | Path | None = None,
    ica_path: str | Path | None = None,
    target_sampling_rate_hz: float | None = None,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    montage_name: str | None = "standard_1020",
    annotation_pairing_margin_s: float = 1.0,
    eeg_channel_names: list[str] | None = None,
    eeg_channel_patterns: list[str] | None = None,
    emg_channel_names: list[str] | None = None,
    emg_channel_patterns: list[str] | None = None,
    interpolate_bad_channels: bool = True,
    apply_rereference: bool = True,
) -> mne.io.BaseRaw:
    """Backwards-compatible wrapper returning only the preprocessed EEG raw."""

    return preprocess_run(
        raw,
        channels_tsv_path=channels_tsv_path,
        annotation_path=annotation_path,
        ica_path=ica_path,
        target_sampling_rate_hz=target_sampling_rate_hz,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        montage_name=montage_name,
        annotation_pairing_margin_s=annotation_pairing_margin_s,
        eeg_channel_names=eeg_channel_names,
        eeg_channel_patterns=eeg_channel_patterns,
        emg_channel_names=emg_channel_names,
        emg_channel_patterns=emg_channel_patterns,
        interpolate_bad_channels=interpolate_bad_channels,
        apply_rereference=apply_rereference,
        keep_emg=False,
    ).eeg_raw
