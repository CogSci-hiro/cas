"""Metadata normalization for the source-level DICS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cas.source_dics.config import EventsConfig
from cas.source_dics.io import EpochRecord

OPTIONAL_EVENT_COLUMNS: tuple[str, ...] = (
    "time_within_run",
    "information_rate_lag_200ms_z",
    "prop_expected_cumulative_info_lag_200ms_z",
    "upcoming_utterance_information_content",
    "n_tokens",
)


def normalize_anchor_type(value: Any) -> str:
    text = str(value).strip().upper()
    if text not in {"FPP", "SPP"}:
        raise ValueError(f"Unexpected anchor_type {value!r}; expected FPP or SPP.")
    return text


def _infer_anchor_type(metadata: pd.DataFrame, *, events_config: EventsConfig) -> pd.Series:
    if events_config.anchor_type_column in metadata.columns:
        return metadata[events_config.anchor_type_column].map(normalize_anchor_type)
    if "anchor_type" in metadata.columns:
        return metadata["anchor_type"].map(normalize_anchor_type)
    if "event_family" in metadata.columns:
        return metadata["event_family"].map(normalize_anchor_type)
    if "event_type" in metadata.columns:
        inferred = metadata["event_type"].astype(str).str.extract(r"(fpp|spp)", expand=False)
        if inferred.isna().any():
            raise ValueError("Could not infer anchor_type from `event_type` for all rows.")
        return inferred.str.upper()
    raise ValueError(
        "Could not infer anchor_type from epochs metadata. "
        "Provide an `anchor_type`/`event_family` column or a parseable `event_type` column."
    )


def _fill_anchor_specific_columns(table: pd.DataFrame) -> pd.DataFrame:
    working = table.copy()
    anchor_type = working["anchor_type"].astype(str)
    if "label" not in working.columns:
        working["label"] = pd.Series([None] * len(working), dtype="object")
    if "fpp_label" in working.columns:
        working.loc[anchor_type == "FPP", "label"] = working.loc[anchor_type == "FPP", "fpp_label"]
    if "spp_label" in working.columns:
        working.loc[anchor_type == "SPP", "label"] = working.loc[anchor_type == "SPP", "spp_label"]

    if "duration" not in working.columns:
        working["duration"] = np.nan
    if "fpp_duration" in working.columns:
        working.loc[anchor_type == "FPP", "duration"] = pd.to_numeric(
            working.loc[anchor_type == "FPP", "fpp_duration"],
            errors="coerce",
        )
    if "spp_duration" in working.columns:
        working.loc[anchor_type == "SPP", "duration"] = pd.to_numeric(
            working.loc[anchor_type == "SPP", "spp_duration"],
            errors="coerce",
        )

    if "onset" not in working.columns:
        working["onset"] = np.nan
    if "fpp_onset" in working.columns:
        working.loc[anchor_type == "FPP", "onset"] = pd.to_numeric(
            working.loc[anchor_type == "FPP", "fpp_onset"],
            errors="coerce",
        )
    if "spp_onset" in working.columns:
        working.loc[anchor_type == "SPP", "onset"] = pd.to_numeric(
            working.loc[anchor_type == "SPP", "spp_onset"],
            errors="coerce",
        )
    return working


def _merge_events_table(metadata: pd.DataFrame, events_table: pd.DataFrame) -> pd.DataFrame:
    if "pair_id" not in metadata.columns or "pair_id" not in events_table.columns:
        return metadata
    event_subset = events_table.drop_duplicates(subset=["pair_id"]).copy()
    merged = metadata.merge(event_subset, how="left", on="pair_id", suffixes=("", "_events"))
    for column_name in event_subset.columns:
        merged_column = f"{column_name}_events"
        if merged_column not in merged.columns:
            continue
        if column_name not in merged.columns:
            merged[column_name] = merged[merged_column]
        else:
            merged[column_name] = merged[column_name].where(
                merged[column_name].notna(),
                merged[merged_column],
            )
        merged = merged.drop(columns=[merged_column])
    return merged


def prepare_epoch_metadata(
    metadata: pd.DataFrame | None,
    *,
    events_table: pd.DataFrame,
    events_config: EventsConfig,
    record: EpochRecord,
) -> pd.DataFrame:
    """Normalize epochs metadata into a source-power modeling table.

    Usage example
    -------------
    >>> import pandas as pd
    >>> record = EpochRecord(subject_id="sub-001", run_id="1", task="conversation", epochs_path=Path("/tmp/demo-epo.fif"))  # doctest: +SKIP
    >>> prepare_epoch_metadata(pd.DataFrame({"event_family": ["fpp"]}), events_table=pd.DataFrame(), events_config=cfg.events, record=record)  # doctest: +SKIP
    """

    working = (
        pd.DataFrame(index=np.arange(0 if metadata is None else len(metadata)))
        if metadata is None
        else metadata.copy()
    )
    working["_epoch_index"] = np.arange(len(working), dtype=int)
    working = _merge_events_table(working, events_table=events_table)
    working["anchor_type"] = _infer_anchor_type(working, events_config=events_config)
    working = working.loc[
        working["anchor_type"].isin(events_config.anchor_types)
    ].reset_index(drop=True)
    working = _fill_anchor_specific_columns(working)

    if events_config.subject_column not in working.columns:
        if "subject_id" in working.columns:
            working[events_config.subject_column] = working["subject_id"]
        else:
            working[events_config.subject_column] = record.subject_id
    if events_config.dyad_column not in working.columns:
        for candidate in ("dyad_id", "recording_id"):
            if candidate in working.columns:
                working[events_config.dyad_column] = working[candidate]
                break
        else:
            working[events_config.dyad_column] = np.nan
    if events_config.run_column not in working.columns:
        if "run" in working.columns:
            working[events_config.run_column] = working["run"]
        else:
            working[events_config.run_column] = record.run_id

    if events_config.label_column not in working.columns:
        working[events_config.label_column] = working.get("label", np.nan)
    if events_config.onset_column not in working.columns:
        if "event_onset_conversation_s" in working.columns:
            working[events_config.onset_column] = pd.to_numeric(
                working["event_onset_conversation_s"],
                errors="coerce",
            )
        else:
            working[events_config.onset_column] = pd.to_numeric(working.get("onset"), errors="coerce")
    if events_config.duration_column not in working.columns:
        working[events_config.duration_column] = pd.to_numeric(working.get("duration"), errors="coerce")
    if events_config.latency_column not in working.columns:
        working[events_config.latency_column] = pd.to_numeric(working.get("latency"), errors="coerce")

    if "time_within_run" not in working.columns:
        if "event_onset_conversation_s" in working.columns:
            working["time_within_run"] = pd.to_numeric(
                working["event_onset_conversation_s"],
                errors="coerce",
            )
        else:
            working["time_within_run"] = pd.to_numeric(
                working[events_config.onset_column],
                errors="coerce",
            )

    if "event_id" not in working.columns:
        if "source_event_id" in working.columns:
            working["event_id"] = working["source_event_id"].astype(str)
        elif "pair_id" in working.columns:
            working["event_id"] = working["pair_id"].astype(str)
        else:
            working["event_id"] = [
                f"{record.subject_id}_run-{record.run_id}_{anchor}_{index:04d}"
                for index, anchor in enumerate(working["anchor_type"].astype(str), start=1)
            ]

    if "subject" not in working.columns:
        working["subject"] = working[events_config.subject_column].astype(str)
    if "dyad" not in working.columns:
        working["dyad"] = working[events_config.dyad_column]
    if "run" not in working.columns:
        working["run"] = working[events_config.run_column]
    working["run"] = pd.to_numeric(working["run"], errors="coerce")

    return working.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class AnchorEpochSelection:
    anchor_type: str
    epoch_indices: np.ndarray
    metadata: pd.DataFrame


def split_anchor_metadata(metadata: pd.DataFrame) -> dict[str, AnchorEpochSelection]:
    selections: dict[str, AnchorEpochSelection] = {}
    for anchor_type, subset in metadata.groupby("anchor_type", sort=False, observed=True):
        selections[str(anchor_type)] = AnchorEpochSelection(
            anchor_type=str(anchor_type),
            epoch_indices=subset["_epoch_index"].to_numpy(dtype=int),
            metadata=subset.reset_index(drop=True),
        )
    return selections
