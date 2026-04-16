"""Channel-selection helpers for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import mne
import pandas as pd


@dataclass(frozen=True, slots=True)
class ChannelSplit:
    """Description of separated channel groups."""

    eeg_names: list[str]
    emg_names: list[str]
    dropped_names: list[str]

    @property
    def eeg_channel_names(self) -> list[str]:
        return self.eeg_names

    @property
    def emg_channel_names(self) -> list[str]:
        return self.emg_names

    @property
    def dropped_channel_names(self) -> list[str]:
        return self.dropped_names


REFERENCE_NAME_PATTERN = re.compile(r"^(?:[lr]ear|[lr]pa|[am][12]|mastoids?)$", flags=re.IGNORECASE)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _match_any(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def _normalize_bids_channel_type(value: str) -> str:
    token = str(value).strip().lower()
    type_map = {
        "eeg": "eeg",
        "emg": "emg",
        "ecg": "ecg",
        "eog": "eog",
        "stim": "stim",
        "misc": "misc",
        "bio": "bio",
        "ref": "misc",
        "reference": "misc",
    }
    return type_map.get(token, "misc")


def _coerce_reference_like_channel_type(name: str, normalized_type: str) -> str:
    if normalized_type == "eeg" and REFERENCE_NAME_PATTERN.match(str(name).strip()):
        return "misc"
    return normalized_type


def apply_bids_channel_types(raw: mne.io.BaseRaw, channels_tsv_path: Path | str) -> dict[str, str]:
    """Apply BIDS channel types from ``channels.tsv`` onto an MNE raw object."""

    path = Path(channels_tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"BIDS channels.tsv file does not exist: {path}")

    table = pd.read_csv(path, sep="\t")
    if table.empty or "type" not in table.columns:
        return {}

    name_column = "name" if "name" in table.columns else "channel_name" if "channel_name" in table.columns else None
    if name_column is None:
        raise ValueError(f"{path} is missing a channel-name column ('name' or 'channel_name').")

    raw_lookup = {name: name for name in raw.ch_names}
    raw_lookup_lower = {name.lower(): name for name in raw.ch_names}
    channel_type_map: dict[str, str] = {}

    for row in table.itertuples(index=False):
        channel_name = str(getattr(row, name_column)).strip()
        if not channel_name:
            continue
        raw_name = raw_lookup.get(channel_name) or raw_lookup_lower.get(channel_name.lower())
        if raw_name is None:
            continue
        normalized_type = _normalize_bids_channel_type(getattr(row, "type"))
        channel_type_map[raw_name] = _coerce_reference_like_channel_type(raw_name, normalized_type)

    if channel_type_map:
        raw.set_channel_types(channel_type_map, on_unit_change="ignore")
    return channel_type_map


def split_eeg_and_emg_channels(raw: mne.io.BaseRaw, config: Mapping[str, Any]) -> ChannelSplit:
    """Split a raw object into EEG and EMG channel groups."""

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    channel_cfg = dict(preprocessing_cfg.get("channels") or {})

    emg_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("emg_names")) if name.strip()}
    emg_patterns = _compile_patterns(_as_string_list(channel_cfg.get("emg_patterns")))
    eeg_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("eeg_names")) if name.strip()}
    eeg_patterns = _compile_patterns(_as_string_list(channel_cfg.get("eeg_name_patterns")))
    drop_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("drop_names")) if name.strip()}

    channel_type_map: dict[str, str] = {}
    emg_names: list[str] = []
    for channel_name in raw.ch_names:
        if channel_name in emg_exact_names or _match_any(channel_name, emg_patterns):
            channel_type_map[channel_name] = "emg"
            emg_names.append(channel_name)
    if channel_type_map:
        raw.set_channel_types(channel_type_map, on_unit_change="ignore")

    eeg_names = [name for name in raw.copy().pick("eeg").ch_names]
    if not eeg_names:
        for channel_name in raw.ch_names:
            if channel_name in emg_names or channel_name in drop_exact_names:
                continue
            if channel_name in eeg_exact_names or _match_any(channel_name, eeg_patterns):
                eeg_names.append(channel_name)

    if not eeg_names:
        raise ValueError("No EEG channels were identified after channel splitting.")

    keep_names = set(eeg_names) | set(emg_names)
    dropped_names = [name for name in raw.ch_names if name not in keep_names]
    return ChannelSplit(
        eeg_names=sorted(eeg_names, key=raw.ch_names.index),
        emg_names=sorted(set(emg_names), key=raw.ch_names.index),
        dropped_names=dropped_names,
    )


def apply_montage(raw: mne.io.BaseRaw, config: Mapping[str, Any]) -> None:
    """Attach a configured montage to the EEG raw object."""

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    montage_cfg = preprocessing_cfg.get("montage", preprocessing_cfg.get("montage_name"))

    if isinstance(montage_cfg, Mapping):
        montage_name = str(montage_cfg.get("name", "standard_1020"))
        on_missing = str(montage_cfg.get("on_missing", "ignore"))
        match_case = bool(montage_cfg.get("match_case", False))
    else:
        montage_name = str(montage_cfg or "standard_1020")
        on_missing = "ignore"
        match_case = False

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing=on_missing, match_case=match_case)


def split_channels(
    raw: mne.io.BaseRaw,
    *,
    eeg_channel_names: list[str] | None = None,
    eeg_channel_patterns: list[str] | None = None,
    emg_channel_names: list[str] | None = None,
    emg_channel_patterns: list[str] | None = None,
) -> ChannelSplit:
    """Split channels into EEG, EMG, and dropped groups."""

    config: dict[str, Any] = {
        "preprocessing": {
            "channels": {
                "eeg_names": list(eeg_channel_names or []),
                "eeg_name_patterns": list(eeg_channel_patterns or []),
                "emg_names": list(emg_channel_names or []),
                "emg_patterns": list(emg_channel_patterns or []),
            }
        }
    }
    return split_eeg_and_emg_channels(raw, config)
