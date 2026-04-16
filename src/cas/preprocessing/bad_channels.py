"""Bad-channel utilities for EEG preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_BIDS_CHANNEL_COLUMNS = ("name", "status")
BAD_STATUS_VALUE = "bad"
MISSING_DIGITIZATION_ERROR = 'Cannot fit headshape without digitization, info["dig"] is None'
NONFINITE_INTERPOLATION_ERROR = "array must not contain infs or NaNs"
DEFAULT_INTERPOLATION_ORIGIN_M = (0.0, 0.0, 0.04)


def read_bad_channels_from_bids_tsv(tsv_path: str | Path) -> list[str]:
    """Read bad EEG channel names from a BIDS ``channels.tsv`` file."""

    channels_tsv_path = Path(tsv_path)
    if not channels_tsv_path.exists():
        raise FileNotFoundError(f"BIDS channels TSV not found: {channels_tsv_path}")

    channel_table = pd.read_csv(channels_tsv_path, sep="\t")
    missing_columns = [
        column_name
        for column_name in REQUIRED_BIDS_CHANNEL_COLUMNS
        if column_name not in channel_table.columns
    ]
    if missing_columns:
        missing_column_names = ", ".join(missing_columns)
        raise ValueError(
            f"BIDS channels TSV is missing required columns: {missing_column_names}"
        )

    status_series = channel_table["status"].fillna("").astype(str).str.strip().str.lower()
    name_series = channel_table["name"].fillna("").astype(str).str.strip()

    bad_channel_names: list[str] = []
    for channel_name, channel_status in zip(name_series, status_series, strict=False):
        if channel_status != BAD_STATUS_VALUE or not channel_name:
            continue
        if channel_name not in bad_channel_names:
            bad_channel_names.append(channel_name)

    return bad_channel_names


def set_bad_channels(
    raw: mne.io.BaseRaw,
    bad_channel_names: list[str],
) -> mne.io.BaseRaw:
    """Set bad channels on an MNE raw object."""

    existing_bad_channel_names = list(raw.info.get("bads", []))
    available_channel_names = set(raw.ch_names)

    merged_bad_channel_names = list(existing_bad_channel_names)
    missing_channel_names: list[str] = []

    for channel_name in bad_channel_names:
        if channel_name not in available_channel_names:
            missing_channel_names.append(channel_name)
            continue
        if channel_name not in merged_bad_channel_names:
            merged_bad_channel_names.append(channel_name)

    if missing_channel_names:
        missing_channel_names_str = ", ".join(sorted(set(missing_channel_names)))
        LOGGER.warning("Ignoring bad channels not present in raw: %s", missing_channel_names_str)

    raw.info["bads"] = merged_bad_channel_names
    return raw


def load_bad_channels_from_bids_tsv(
    channels_tsv_path: str | Path,
    *,
    raw: mne.io.BaseRaw,
) -> list[str]:
    """Load BIDS bad-channel annotations and apply them to ``raw``."""

    bad_channel_names = read_bad_channels_from_bids_tsv(channels_tsv_path)
    set_bad_channels(raw, bad_channel_names)
    return list(raw.info.get("bads", []))


def interpolate_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Interpolate marked bad EEG channels with conservative fallbacks."""

    if not raw.info.get("bads"):
        LOGGER.info("No bad channels are marked on raw.info['bads']; skipping interpolation.")
        return raw

    invalid_position_names: list[str] = []
    bad_name_set = set(raw.info["bads"])
    channel_types = raw.get_channel_types()
    for channel_name, channel_type, channel_info in zip(
        raw.ch_names,
        channel_types,
        raw.info["chs"],
        strict=False,
    ):
        if channel_type != "eeg":
            continue
        position = np.asarray(channel_info["loc"][:3], dtype=float)
        if not np.isfinite(position).all():
            invalid_position_names.append(channel_name)

    invalid_position_names = list(dict.fromkeys(invalid_position_names))
    invalid_bad_position_names = [name for name in invalid_position_names if name in bad_name_set]
    if invalid_bad_position_names:
        LOGGER.warning(
            "Skipping interpolation for bad EEG channels with non-finite sensor positions: %s",
            ", ".join(invalid_bad_position_names),
        )
        raw.info["bads"] = [name for name in raw.info["bads"] if name not in invalid_bad_position_names]
        if not raw.info["bads"]:
            return raw

    try:
        raw.interpolate_bads(reset_bads=True, exclude=invalid_position_names, verbose="ERROR")
    except RuntimeError as exc:
        if MISSING_DIGITIZATION_ERROR not in str(exc):
            raise
        LOGGER.warning(
            "Missing digitization metadata for bad-channel interpolation; retrying with fixed origin %s m.",
            DEFAULT_INTERPOLATION_ORIGIN_M,
        )
        try:
            raw.interpolate_bads(
                reset_bads=True,
                exclude=invalid_position_names,
                origin=DEFAULT_INTERPOLATION_ORIGIN_M,
                verbose="ERROR",
            )
        except ValueError as value_exc:
            if NONFINITE_INTERPOLATION_ERROR not in str(value_exc):
                raise
            LOGGER.warning(
                "Skipping bad-channel interpolation because channel geometry still produced non-finite values."
            )
    except ValueError as exc:
        if NONFINITE_INTERPOLATION_ERROR not in str(exc):
            raise
        LOGGER.warning(
            "Skipping bad-channel interpolation because channel geometry contains non-finite values."
        )

    return raw
