"""EEG preprocessing utilities for MNE raw objects."""

from cas.preprocessing.bad_channels import (
    interpolate_bad_channels,
    load_bad_channels_from_bids_tsv,
    read_bad_channels_from_bids_tsv,
    set_bad_channels,
)
from cas.preprocessing.config import (
    PreprocessingOutputLayout,
    PreprocessingRunPaths,
    build_preprocessing_run_paths,
    load_preprocessing_output_layout,
    resolve_output_dir,
    resolve_paths_map,
    resolve_preprocessing_output_layout,
)
from cas.preprocessing.channels import (
    ChannelSplit,
    apply_bids_channel_types,
    apply_montage,
    split_channels,
    split_eeg_and_emg_channels,
)
from cas.preprocessing.downsample import downsample_raw
from cas.preprocessing.events import ExtractedEvents, extract_events, extract_events_table, write_events_tsv
from cas.preprocessing.filtering import apply_bandpass_filter, bandpass_filter_raw
from cas.preprocessing.ica import apply_precomputed_ica, load_ica
from cas.preprocessing.pipeline import PreprocessingResult, preprocess_raw, preprocess_run
from cas.preprocessing.qc import write_preprocessing_aggregates
from cas.preprocessing.rereference import apply_average_reference

__all__ = [
    "ChannelSplit",
    "ExtractedEvents",
    "PreprocessingOutputLayout",
    "PreprocessingResult",
    "PreprocessingRunPaths",
    "apply_average_reference",
    "apply_bandpass_filter",
    "apply_bids_channel_types",
    "apply_montage",
    "apply_precomputed_ica",
    "bandpass_filter_raw",
    "build_preprocessing_run_paths",
    "downsample_raw",
    "extract_events",
    "extract_events_table",
    "interpolate_bad_channels",
    "load_ica",
    "load_bad_channels_from_bids_tsv",
    "load_preprocessing_output_layout",
    "preprocess_raw",
    "preprocess_run",
    "read_bad_channels_from_bids_tsv",
    "resolve_output_dir",
    "resolve_paths_map",
    "resolve_preprocessing_output_layout",
    "set_bad_channels",
    "split_channels",
    "split_eeg_and_emg_channels",
    "write_preprocessing_aggregates",
    "write_events_tsv",
]
