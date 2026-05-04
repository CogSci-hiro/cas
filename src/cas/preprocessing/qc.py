"""Aggregate preprocessing metadata and QC outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def write_preprocessing_aggregates(
    *,
    summary_paths: list[str | Path],
    preprocessing_summary_tsv: str | Path,
    bad_channels_tsv: str | Path,
    rejected_segments_tsv: str | Path,
    qc_summary_json: str | Path,
) -> dict[str, Path]:
    summary_rows: list[dict[str, Any]] = []
    bad_channel_rows: list[dict[str, Any]] = []
    rejected_segment_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        subject_id = str(payload.get("subject_id", ""))
        task = str(payload.get("task", ""))
        run = str(payload.get("run", ""))
        dyad_id = str(payload.get("dyad_id", ""))

        summary_rows.append(
            {
                "subject_id": subject_id,
                "task": task,
                "run": run,
                "dyad_id": dyad_id,
                "input_path": str(payload.get("input_path", "")),
                "eeg_output_path": str(payload.get("eeg_output_path", "")),
                "emg_output_path": str(payload.get("emg_output_path", "")),
                "events_output_path": str(payload.get("events_output_path", "")),
                "sampling_rate_before_hz": payload.get("sampling_rate_before_hz"),
                "sampling_rate_after_hz": payload.get("sampling_rate_after_hz"),
                "event_source": str(payload.get("event_source", "")),
                "event_count": int(payload.get("event_count", 0)),
                "n_kept_eeg_channels": len(_string_list(payload.get("kept_eeg_channels"))),
                "n_kept_emg_channels": len(_string_list(payload.get("kept_emg_channels"))),
                "n_dropped_channels": len(_string_list(payload.get("dropped_channels"))),
                "n_loaded_bad_channels": len(_string_list(payload.get("loaded_bad_channels"))),
                "n_bad_channels_before_interpolation": len(
                    _string_list(payload.get("bad_channels_before_interpolation"))
                ),
                "n_bad_channels_after_preprocessing": len(
                    _string_list(payload.get("bad_channels_after_preprocessing"))
                ),
                "save_intermediates": bool(payload.get("save_intermediates", False)),
                "n_intermediate_files": len(payload.get("intermediate_files") or []),
            }
        )

        for stage_name, key in (
            ("loaded", "loaded_bad_channels"),
            ("before_interpolation", "bad_channels_before_interpolation"),
            ("after_preprocessing", "bad_channels_after_preprocessing"),
        ):
            for channel_name in _string_list(payload.get(key)):
                bad_channel_rows.append(
                    {
                        "subject_id": subject_id,
                        "task": task,
                        "run": run,
                        "dyad_id": dyad_id,
                        "stage": stage_name,
                        "channel_name": channel_name,
                    }
                )

    summary_table_path = Path(preprocessing_summary_tsv)
    bad_channels_path = Path(bad_channels_tsv)
    rejected_segments_path = Path(rejected_segments_tsv)
    qc_summary_path = Path(qc_summary_json)
    for path in (summary_table_path, bad_channels_path, rejected_segments_path, qc_summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        summary_rows,
        columns=[
            "subject_id",
            "task",
            "run",
            "dyad_id",
            "input_path",
            "eeg_output_path",
            "emg_output_path",
            "events_output_path",
            "sampling_rate_before_hz",
            "sampling_rate_after_hz",
            "event_source",
            "event_count",
            "n_kept_eeg_channels",
            "n_kept_emg_channels",
            "n_dropped_channels",
            "n_loaded_bad_channels",
            "n_bad_channels_before_interpolation",
            "n_bad_channels_after_preprocessing",
            "save_intermediates",
            "n_intermediate_files",
        ],
    ).sort_values(by=["subject_id", "task", "run"], kind="stable").to_csv(
        summary_table_path, sep="\t", index=False
    )
    pd.DataFrame(
        bad_channel_rows,
        columns=["subject_id", "task", "run", "dyad_id", "stage", "channel_name"],
    ).sort_values(by=["subject_id", "task", "run", "stage", "channel_name"], kind="stable").to_csv(
        bad_channels_path, sep="\t", index=False
    )
    pd.DataFrame(
        rejected_segment_rows,
        columns=["subject_id", "task", "run", "dyad_id", "segment_start_s", "segment_end_s", "reason"],
    ).to_csv(rejected_segments_path, sep="\t", index=False)

    qc_summary = {
        "n_runs": len(summary_rows),
        "n_annotation_runs": sum(1 for row in summary_rows if row["event_source"] == "annotations"),
        "n_mne_runs": sum(1 for row in summary_rows if row["event_source"] == "mne"),
        "total_events": int(sum(int(row["event_count"]) for row in summary_rows)),
        "runs_with_bad_channels": sum(
            1 for row in summary_rows if int(row["n_bad_channels_before_interpolation"]) > 0
        ),
        "runs_with_saved_intermediates": sum(1 for row in summary_rows if bool(row["save_intermediates"])),
        "rejected_segment_count": len(rejected_segment_rows),
        "summary_table": str(summary_table_path),
        "bad_channels_table": str(bad_channels_path),
        "rejected_segments_table": str(rejected_segments_path),
    }
    qc_summary_path.write_text(json.dumps(qc_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "preprocessing_summary_tsv": summary_table_path,
        "bad_channels_tsv": bad_channels_path,
        "rejected_segments_tsv": rejected_segments_path,
        "qc_summary_json": qc_summary_path,
    }
