"""QC output helpers for the source DICS pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cas.source_dics.io import write_json, write_table


def write_config_snapshot(config_payload: dict[str, Any], *, qc_dir: Path) -> Path:
    return write_json(config_payload, qc_dir / "config_resolved.json")


def write_run_summary(summary: dict[str, Any], *, qc_dir: Path) -> Path:
    return write_json(summary, qc_dir / "run_summary.json")


def write_filter_windows(*, qc_dir: Path, filter_tmin: float, filter_tmax: float, analysis_tmin: float, analysis_tmax: float) -> Path:
    payload = {
        "filter_window_s": {"tmin": filter_tmin, "tmax": filter_tmax},
        "analysis_window_s": {"tmin": analysis_tmin, "tmax": analysis_tmax},
        "guardrail": "Post-onset samples may exist only as TF margin and are excluded from filter estimation and final statistics.",
    }
    return write_json(payload, qc_dir / "source_dics_filter_windows.json")


def write_missing_inputs(rows: list[dict[str, Any]], *, qc_dir: Path) -> Path:
    table = pd.DataFrame(rows)
    return write_table(table, qc_dir / "missing_inputs.csv")


def write_event_counts(table: pd.DataFrame, *, qc_dir: Path) -> Path:
    counts = (
        table.groupby(["subject", "anchor_type"], observed=True)
        .size()
        .rename("n_events")
        .reset_index()
        .sort_values(["subject", "anchor_type"], kind="mergesort")
    )
    return write_table(counts, qc_dir / "event_counts_by_subject_anchor_type.csv")


def write_band_subject_counts(rows: list[dict[str, Any]], *, qc_dir: Path) -> Path:
    table = pd.DataFrame(rows)
    return write_table(table, qc_dir / "event_counts_by_band_subject.csv")


def write_mean_power(rows: list[pd.DataFrame], *, qc_dir: Path) -> Path | None:
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    grouped = (
        combined.groupby(["band", "anchor_type"], observed=True)["mean_power"]
        .mean()
        .reset_index()
        .sort_values(["band", "anchor_type"], kind="mergesort")
    )
    return write_table(grouped, qc_dir / "mean_power_by_band_anchor_type.csv")
