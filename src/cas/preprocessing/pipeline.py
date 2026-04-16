"""CAS EEG preprocessing pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np

from cas.preprocessing.bad_channels import (
    interpolate_bad_channels as interpolate_bad_eeg_channels,
    load_bad_channels_from_bids_tsv,
)
from cas.preprocessing.channels import (
    apply_bids_channel_types,
    apply_montage,
    split_eeg_and_emg_channels,
)
from cas.preprocessing.downsample import downsample_raw
from cas.preprocessing.events import _extract_annotation_events, extract_events_table
from cas.preprocessing.filtering import apply_bandpass_filter
from cas.preprocessing.ica import apply_precomputed_ica
from cas.preprocessing.rereference import apply_average_reference

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """All in-memory outputs of preprocessing for one run."""

    eeg_raw: mne.io.BaseRaw
    emg_data: np.ndarray
    emg_channel_names: list[str]
    dropped_channel_names: list[str]
    events_rows: list[dict[str, str | int | float]]
    summary: dict[str, object]


def _save_intermediate_raw(
    raw: mne.io.BaseRaw,
    *,
    intermediates_dir: Path | None,
    step_index: int,
    step_name: str,
) -> str | None:
    """Persist an intermediate EEG snapshot when requested."""

    if intermediates_dir is None:
        return None

    intermediates_dir.mkdir(parents=True, exist_ok=True)
    output_path = intermediates_dir / f"{step_index:02d}_{step_name}_raw.fif"
    raw.save(output_path, overwrite=True, verbose="ERROR")
    return str(output_path)


def _build_preprocessing_config(
    *,
    target_sampling_rate_hz: float | None,
    low_cut_hz: float | None,
    high_cut_hz: float | None,
    montage_name: str | None,
    eeg_channel_names: list[str] | None,
    eeg_channel_patterns: list[str] | None,
    emg_channel_names: list[str] | None,
    emg_channel_patterns: list[str] | None,
) -> dict[str, object]:
    preprocessing_config: dict[str, object] = {
        "channels": {
            "eeg_names": list(eeg_channel_names or []),
            "eeg_name_patterns": list(eeg_channel_patterns or []),
            "emg_names": list(emg_channel_names or []),
            "emg_patterns": list(emg_channel_patterns or []),
        },
        "events": {},
    }

    if montage_name:
        preprocessing_config["montage"] = montage_name
    if target_sampling_rate_hz is not None:
        preprocessing_config["downsample_hz"] = float(target_sampling_rate_hz)
    if low_cut_hz is not None or high_cut_hz is not None:
        preprocessing_config["filter"] = {
            "l_freq_hz": float(low_cut_hz) if low_cut_hz is not None else None,
            "h_freq_hz": float(high_cut_hz) if high_cut_hz is not None else None,
        }

    return {"preprocessing": preprocessing_config}


def _event_rows_from_event_table(
    raw: mne.io.BaseRaw,
    *,
    config: dict[str, object],
    annotation_path: str | Path | None,
    annotation_pairing_margin_s: float,
) -> tuple[list[dict[str, str | int | float]], str]:
    annotation_rows = _extract_annotation_events(
        raw,
        annotation_path=annotation_path,
        annotation_pairing_margin_s=annotation_pairing_margin_s,
    )
    if annotation_rows:
        return annotation_rows, "annotations"

    events_table, event_id_map = extract_events_table(raw, config)
    label_lookup = {int(value): str(key) for key, value in event_id_map.items()}
    rows: list[dict[str, str | int | float]] = []
    for event_index, row in enumerate(events_table.itertuples(index=False), start=1):
        sample = int(row.sample)
        event_id = int(row.event_id)
        rows.append(
            {
                "event_index": event_index,
                "source": "mne",
                "sample": sample,
                "onset_s": float(row.time_s),
                "duration_s": 0.0,
                "event_id": event_id,
                "label": label_lookup.get(event_id, f"event_{event_id}"),
                "pair_id": "",
                "part": "",
                "response": "",
                "speaker_fpp": "",
                "speaker_spp": "",
                "fpp_label": "",
                "spp_label": "",
                "fpp_onset": "",
                "fpp_offset": "",
                "spp_onset": "",
                "spp_offset": "",
                "latency": "",
            }
        )
    return rows, "mne"


def _apply_required_precomputed_ica(raw: mne.io.BaseRaw, ica_path: str | Path | None) -> None:
    """Apply precomputed ICA, failing when it is requested but unavailable."""

    if ica_path is None:
        return

    resolved_ica_path = Path(ica_path)
    if not resolved_ica_path.exists():
        raise FileNotFoundError(f"ICA file not found: {resolved_ica_path}")

    try:
        LOGGER.info("Applying precomputed ICA from %s.", resolved_ica_path)
        apply_precomputed_ica(raw, resolved_ica_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to apply precomputed ICA from {resolved_ica_path}: {exc}"
        ) from exc


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
    intermediates_dir: str | Path | None = None,
) -> PreprocessingResult:
    """Run the CAS preprocessing pipeline for one recording."""

    raw_copy = raw.copy()
    resolved_intermediates_dir = Path(intermediates_dir) if intermediates_dir is not None else None
    sampling_rate_before_hz = float(raw_copy.info["sfreq"])
    intermediate_files: list[dict[str, str]] = []
    preprocessing_config = _build_preprocessing_config(
        target_sampling_rate_hz=target_sampling_rate_hz,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        montage_name=montage_name,
        eeg_channel_names=eeg_channel_names,
        eeg_channel_patterns=eeg_channel_patterns,
        emg_channel_names=emg_channel_names,
        emg_channel_patterns=emg_channel_patterns,
    )

    events_rows, event_source = _event_rows_from_event_table(
        raw_copy,
        config=preprocessing_config,
        annotation_path=annotation_path,
        annotation_pairing_margin_s=annotation_pairing_margin_s,
    )
    if channels_tsv_path is not None:
        apply_bids_channel_types(raw_copy, channels_tsv_path)

    channel_split = split_eeg_and_emg_channels(raw_copy, preprocessing_config)
    eeg_raw = raw_copy.copy().pick(channel_split.eeg_names)

    if keep_emg and channel_split.emg_names:
        emg_raw = raw_copy.copy().pick(channel_split.emg_names)
        emg_data = emg_raw.get_data()
        emg_sampling_rate_hz = float(emg_raw.info["sfreq"])
    else:
        emg_data = np.empty((0, 0), dtype=float)
        emg_sampling_rate_hz = None

    if montage_name:
        LOGGER.info("Applying montage %s to EEG channels.", montage_name)
        apply_montage(eeg_raw, preprocessing_config)

    loaded_bad_channel_names: list[str] = []
    if channels_tsv_path is not None:
        LOGGER.info("Loading bad channels from %s.", channels_tsv_path)
        loaded_bad_channel_names = load_bad_channels_from_bids_tsv(channels_tsv_path, raw=eeg_raw)

    if target_sampling_rate_hz is not None:
        LOGGER.info("Downsampling EEG to %.3f Hz.", target_sampling_rate_hz)
        downsample_raw(eeg_raw, preprocessing_config)
        downsample_path = _save_intermediate_raw(
            eeg_raw,
            intermediates_dir=resolved_intermediates_dir,
            step_index=1,
            step_name="downsample",
        )
        if downsample_path is not None:
            intermediate_files.append({"step": "downsample", "path": downsample_path})

    if low_cut_hz is not None or high_cut_hz is not None:
        LOGGER.info(
            "Filtering EEG with low_cut_hz=%s high_cut_hz=%s.",
            low_cut_hz,
            high_cut_hz,
        )
        apply_bandpass_filter(eeg_raw, preprocessing_config)
        filter_path = _save_intermediate_raw(
            eeg_raw,
            intermediates_dir=resolved_intermediates_dir,
            step_index=2,
            step_name="filter",
        )
        if filter_path is not None:
            intermediate_files.append({"step": "filter", "path": filter_path})

    _apply_required_precomputed_ica(eeg_raw, ica_path)
    ica_snapshot_path = _save_intermediate_raw(
        eeg_raw,
        intermediates_dir=resolved_intermediates_dir,
        step_index=3,
        step_name="apply_ica",
    )
    if ica_snapshot_path is not None:
        intermediate_files.append({"step": "apply_ica", "path": ica_snapshot_path})

    bad_channel_names_before_interpolation = list(eeg_raw.info.get("bads", []))
    if interpolate_bad_channels and bad_channel_names_before_interpolation:
        LOGGER.info(
            "Interpolating %d bad EEG channels.",
            len(bad_channel_names_before_interpolation),
        )
        interpolate_bad_eeg_channels(eeg_raw)
        interpolate_path = _save_intermediate_raw(
            eeg_raw,
            intermediates_dir=resolved_intermediates_dir,
            step_index=4,
            step_name="interpolate_bad_channels",
        )
        if interpolate_path is not None:
            intermediate_files.append({"step": "interpolate_bad_channels", "path": interpolate_path})
    elif interpolate_bad_channels:
        LOGGER.info("Skipping bad channel interpolation because no bad channels are set.")

    if apply_rereference:
        LOGGER.info("Applying average EEG reference.")
        apply_average_reference(eeg_raw)
        rereference_path = _save_intermediate_raw(
            eeg_raw,
            intermediates_dir=resolved_intermediates_dir,
            step_index=5,
            step_name="rereference",
        )
        if rereference_path is not None:
            intermediate_files.append({"step": "rereference", "path": rereference_path})

    event_source_counts = Counter(str(row.get("source", "")) for row in events_rows)
    summary: dict[str, Any] = {
        "sampling_rate_before_hz": sampling_rate_before_hz,
        "sampling_rate_after_hz": float(eeg_raw.info["sfreq"]),
        "event_source": event_source,
        "event_count": len(events_rows),
        "event_source_counts": dict(sorted(event_source_counts.items())),
        "loaded_bad_channels": loaded_bad_channel_names,
        "bad_channels_before_interpolation": bad_channel_names_before_interpolation,
        "bad_channels_after_preprocessing": list(eeg_raw.info.get("bads", [])),
        "kept_eeg_channels": channel_split.eeg_names,
        "kept_emg_channels": channel_split.emg_names if keep_emg else [],
        "dropped_channels": channel_split.dropped_names,
        "montage": montage_name,
        "emg_sampling_rate_hz": emg_sampling_rate_hz,
        "intermediate_files": intermediate_files,
    }

    return PreprocessingResult(
        eeg_raw=eeg_raw,
        emg_data=emg_data,
        emg_channel_names=channel_split.emg_names if keep_emg else [],
        dropped_channel_names=channel_split.dropped_names,
        events_rows=events_rows,
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
    intermediates_dir: str | Path | None = None,
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
        intermediates_dir=intermediates_dir,
    ).eeg_raw
