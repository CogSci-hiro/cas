"""Export helpers for source-level DICS power results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cas.source_dics.config import SourceDicsConfig
from cas.source_dics.dics import SourcePowerResult
from cas.source_dics.io import build_record_stem, write_table
from cas.source_dics.io import EpochRecord


LONG_TABLE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject",
    "dyad",
    "run",
    "event_id",
    "anchor_type",
    "label",
    "band",
    "source_id",
    "time",
    "power",
    "duration",
    "latency",
    "time_within_run",
    "information_rate_lag_200ms_z",
    "prop_expected_cumulative_info_lag_200ms_z",
    "upcoming_utterance_information_content",
    "n_tokens",
)


@dataclass(frozen=True, slots=True)
class ExportedArtifacts:
    trial_power_path: Path | None
    metadata_path: Path | None
    long_table_paths: tuple[Path, ...]


def export_trial_power(
    result: SourcePowerResult,
    metadata: pd.DataFrame,
    *,
    record: EpochRecord,
    config: SourceDicsConfig,
) -> Path:
    stem = build_record_stem(record, anchor_type=result.anchor_type, band_name=result.band_name)
    output_path = config.paths.trial_power_dir / f"{stem}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        power=result.power.astype(np.float32),
        times=result.times.astype(np.float32),
        source_ids=np.asarray(result.source_ids, dtype=object),
        event_ids=metadata["event_id"].astype(str).to_numpy(dtype=object),
    )
    return output_path


def export_metadata(metadata: pd.DataFrame, *, record: EpochRecord, result: SourcePowerResult, config: SourceDicsConfig) -> Path:
    stem = build_record_stem(record, anchor_type=result.anchor_type, band_name=result.band_name)
    output_path = config.paths.metadata_dir / f"{stem}.csv"
    write_table(metadata, output_path)
    return output_path


def _flatten_power_chunk(
    power: np.ndarray,
    metadata: pd.DataFrame,
    *,
    source_ids: list[str],
    times: np.ndarray,
    band_name: str,
    start_event: int,
    stop_event: int,
) -> pd.DataFrame:
    event_metadata = metadata.iloc[start_event:stop_event].reset_index(drop=True)
    event_power = power[start_event:stop_event]
    n_events, n_sources, n_times = event_power.shape
    if n_events == 0:
        return pd.DataFrame(columns=list(LONG_TABLE_REQUIRED_COLUMNS))

    source_values = np.asarray(source_ids, dtype=object)
    time_values = np.asarray(times, dtype=float)
    repeated_metadata = event_metadata.loc[event_metadata.index.repeat(n_sources * n_times)].reset_index(drop=True)
    repeated_metadata["band"] = band_name
    repeated_metadata["source_id"] = np.tile(np.repeat(source_values, n_times), n_events)
    repeated_metadata["time"] = np.tile(time_values, n_events * n_sources)
    repeated_metadata["power"] = event_power.reshape(-1).astype(np.float32)
    for column_name in LONG_TABLE_REQUIRED_COLUMNS:
        if column_name not in repeated_metadata.columns:
            repeated_metadata[column_name] = np.nan
    return repeated_metadata.loc[:, list(LONG_TABLE_REQUIRED_COLUMNS)]


def export_long_table(
    result: SourcePowerResult,
    metadata: pd.DataFrame,
    *,
    record: EpochRecord,
    config: SourceDicsConfig,
) -> tuple[Path, ...]:
    stem = build_record_stem(record, anchor_type=result.anchor_type, band_name=result.band_name)
    band_dir = config.paths.long_table_dir / f"band-{result.band_name}"
    band_dir.mkdir(parents=True, exist_ok=True)

    rows_per_event = max(1, len(result.source_ids) * len(result.times))
    events_per_chunk = max(1, config.output.long_table_chunk_rows // rows_per_event)
    written_paths: list[Path] = []

    for part_index, start_event in enumerate(range(0, len(metadata), events_per_chunk)):
        stop_event = min(len(metadata), start_event + events_per_chunk)
        table = _flatten_power_chunk(
            result.power,
            metadata,
            source_ids=result.source_ids,
            times=result.times,
            band_name=result.band_name,
            start_event=start_event,
            stop_event=stop_event,
        )
        output_path = band_dir / f"{stem}_part-{part_index:03d}.parquet"
        try:
            write_table(table, output_path)
        except Exception:
            fallback_path = output_path.with_suffix(".csv.gz")
            write_table(table, fallback_path)
            written_paths.append(fallback_path)
        else:
            written_paths.append(output_path)
    return tuple(written_paths)


def summarize_mean_power(result: SourcePowerResult, metadata: pd.DataFrame) -> pd.DataFrame:
    mean_power = result.power.mean(axis=(1, 2))
    summary = metadata.loc[:, ["anchor_type"]].copy()
    summary["band"] = result.band_name
    summary["mean_power"] = mean_power
    if "label" in metadata.columns:
        summary["label"] = metadata["label"]
    return summary
