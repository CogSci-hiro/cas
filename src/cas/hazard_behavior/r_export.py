"""Export behavioural hazard risk sets for downstream R GLMM fitting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id
from cas.hazard_behavior.io import write_json, write_table

REQUIRED_EXPORT_COLUMNS = (
    "event",
    "dyad_id",
    "participant_id",
    "run_id",
    "speaker",
    "participant_speaker_id",
    "participant_speaker",
    "episode_id",
    "bin_end",
    "partner_ipu_onset",
    "partner_ipu_offset",
    "time_from_partner_onset",
    "time_from_partner_offset",
    "time_since_partner_offset_positive",
    "z_information_rate_lag_best",
    "z_prop_expected_cumulative_info_lag_best",
)
OPTIONAL_EXPORT_COLUMNS = (
    "is_censored_episode",
    "fpp_latency_from_partner_onset",
    "fpp_latency_from_partner_offset",
)


@dataclass(frozen=True, slots=True)
class BehaviourGlmmExportResult:
    """Paths and metadata for an exported behavioural GLMM dataset."""

    output_csv: Path
    output_qc_json: Path
    information_rate_lag_ms: int
    expected_cumulative_info_lag_ms: int
    n_rows_after_export: int


def export_behaviour_glmm_data(
    riskset_table: pd.DataFrame,
    *,
    output_csv: Path,
    output_qc_json: Path,
    selected_lags_json: Path | None = None,
    information_rate_lag_ms: int | None = None,
    expected_cumulative_info_lag_ms: int | None = None,
    lag_grid_ms: tuple[int, ...] | None = None,
) -> BehaviourGlmmExportResult:
    """Create a compact behavioural GLMM CSV and QC JSON for R."""

    working = riskset_table.copy()
    n_rows_before_export = int(len(working))
    n_events_before_export = int(pd.to_numeric(working.get("event"), errors="coerce").fillna(0).sum())
    resolved_lags = resolve_behaviour_glmm_lags(
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
    )
    information_rate_lag_ms = resolved_lags["information_rate_lag_ms"]
    expected_cumulative_info_lag_ms = resolved_lags["expected_cumulative_info_lag_ms"]

    working = _prepare_export_columns(
        working,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
        lag_grid_ms=lag_grid_ms,
    )

    dropped_missing_counts_by_column: dict[str, int] = {}
    for column_name in REQUIRED_EXPORT_COLUMNS:
        dropped_missing_counts_by_column[column_name] = int(working[column_name].isna().sum())
    required_mask = working.loc[:, REQUIRED_EXPORT_COLUMNS].notna().all(axis=1)
    lag_columns = _available_lag_columns(working, lag_grid_ms=lag_grid_ms)
    exported = working.loc[
        required_mask,
        list(REQUIRED_EXPORT_COLUMNS) + lag_columns + _available_optional_columns(working),
    ].copy()

    exported["event"] = pd.to_numeric(exported["event"], errors="raise").astype(int)
    exported["run_id"] = exported["run_id"].astype(str)
    exported["speaker"] = exported["speaker"].astype(str)
    exported["participant_speaker_id"] = exported["participant_speaker_id"].astype(str)
    exported["participant_speaker"] = exported["participant_speaker"].astype(str)
    exported["participant_id"] = exported["participant_id"].astype(str)
    exported["dyad_id"] = exported["dyad_id"].astype(str)
    exported["episode_id"] = exported["episode_id"].astype(str)

    write_table(exported, output_csv, sep=",")

    offset_values = pd.to_numeric(exported["time_from_partner_offset"], errors="coerce")
    identity_validation = validate_participant_speaker_id(
        exported,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
    )
    qc_payload = {
        "n_rows_before_export": n_rows_before_export,
        "n_rows_after_export": int(len(exported)),
        "n_events_before_export": n_events_before_export,
        "n_events_after_export": int(exported["event"].sum()),
        "n_dyads": int(exported["dyad_id"].nunique()),
        "n_participants": int(exported["participant_id"].nunique()),
        "n_participant_speaker_ids": int(exported["participant_speaker_id"].nunique()),
        "n_participant_speakers": int(exported["participant_speaker"].nunique()),
        "n_episodes": int(exported["episode_id"].nunique()),
        "information_rate_lag_ms": int(information_rate_lag_ms),
        "expected_cumulative_info_lag_ms": int(expected_cumulative_info_lag_ms),
        "lag_grid_ms": [int(lag_ms) for lag_ms in (lag_grid_ms or ())],
        "required_columns": list(REQUIRED_EXPORT_COLUMNS),
        "lag_columns": lag_columns,
        "dropped_missing_counts_by_column": dropped_missing_counts_by_column,
        "time_from_partner_offset_min": float(offset_values.min()),
        "time_from_partner_offset_max": float(offset_values.max()),
        "proportion_negative_time_from_partner_offset": float((offset_values < 0.0).mean()),
        "event_levels_present": sorted(int(value) for value in exported["event"].dropna().unique()),
        "identity_validation": identity_validation,
    }
    write_json(qc_payload, output_qc_json)
    return BehaviourGlmmExportResult(
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=int(information_rate_lag_ms),
        expected_cumulative_info_lag_ms=int(expected_cumulative_info_lag_ms),
        n_rows_after_export=int(len(exported)),
    )


def export_behaviour_glmm_data_from_path(
    *,
    input_riskset: Path,
    output_csv: Path,
    output_qc_json: Path,
    selected_lags_json: Path | None = None,
    information_rate_lag_ms: int | None = None,
    expected_cumulative_info_lag_ms: int | None = None,
    lag_grid_ms: tuple[int, ...] | None = None,
) -> BehaviourGlmmExportResult:
    """Load a risk-set table from disk and export it for R GLMM fitting."""

    riskset_table = pd.read_csv(input_riskset, sep=None, engine="python")
    return export_behaviour_glmm_data(
        riskset_table,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
        lag_grid_ms=lag_grid_ms,
    )


def resolve_behaviour_glmm_lags(
    *,
    selected_lags_json: Path | None,
    information_rate_lag_ms: int | None,
    expected_cumulative_info_lag_ms: int | None,
) -> dict[str, int]:
    """Resolve final behavioural GLMM lags from JSON and optional CLI overrides."""

    lag_payload: dict[str, Any] = {}
    if selected_lags_json is not None:
        lag_payload = json.loads(selected_lags_json.read_text(encoding="utf-8"))
    resolved_information_rate = (
        int(information_rate_lag_ms)
        if information_rate_lag_ms is not None
        else _coerce_required_lag(lag_payload, "best_information_rate_lag_ms")
    )
    resolved_expected = (
        int(expected_cumulative_info_lag_ms)
        if expected_cumulative_info_lag_ms is not None
        else _coerce_required_lag(lag_payload, "best_expected_cumulative_info_lag_ms")
    )
    return {
        "information_rate_lag_ms": resolved_information_rate,
        "expected_cumulative_info_lag_ms": resolved_expected,
    }


def _prepare_export_columns(
    riskset_table: pd.DataFrame,
    *,
    information_rate_lag_ms: int,
    expected_cumulative_info_lag_ms: int,
    lag_grid_ms: tuple[int, ...] | None,
) -> pd.DataFrame:
    working = riskset_table.copy()
    if "dyad_id" not in working.columns:
        if "participant_speaker" in working.columns:
            working["dyad_id"] = working["participant_speaker"].astype(str)
        else:
            raise ValueError("Cannot construct export without either `dyad_id` or `participant_speaker`.")
    if "speaker" not in working.columns and "participant_speaker" in working.columns:
        legacy_participant = working["participant_speaker"].astype(str)
        inferred_speaker = legacy_participant.str.rsplit("_", n=1).str[-1]
        canonical_from_legacy = working["dyad_id"].astype(str) + "_" + inferred_speaker
        if legacy_participant.equals(canonical_from_legacy):
            working["speaker"] = inferred_speaker
            if "participant_speaker_id" not in working.columns:
                working["participant_speaker_id"] = legacy_participant
        else:
            working["speaker"] = legacy_participant
    if "speaker" not in working.columns:
        raise ValueError("Cannot construct export without `speaker` or a role-labeled `participant_speaker`.")
    working = ensure_participant_speaker_id(
        working,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
        overwrite="participant_speaker_id" not in working.columns,
    )
    if "participant_speaker" not in working.columns:
        working["participant_speaker"] = working["speaker"].astype(str)
    if "participant_id" not in working.columns:
        working["participant_id"] = working["participant_speaker_id"].astype(str)
    if "run_id" not in working.columns:
        if "run" in working.columns:
            working["run_id"] = working["run"]
        else:
            working["run_id"] = "run-unknown"
    if "time_from_partner_offset" not in working.columns:
        missing_dependencies = [
            column_name
            for column_name in ("bin_end", "partner_ipu_offset")
            if column_name not in working.columns
        ]
        if missing_dependencies:
            raise ValueError(
                "Cannot derive `time_from_partner_offset`; missing column(s): "
                + ", ".join(missing_dependencies)
            )
        working["time_from_partner_offset"] = (
            pd.to_numeric(working["bin_end"], errors="coerce")
            - pd.to_numeric(working["partner_ipu_offset"], errors="coerce")
        )
    working["time_since_partner_offset_positive"] = pd.to_numeric(
        working["time_from_partner_offset"],
        errors="coerce",
    ).clip(lower=0.0)
    if "is_censored_episode" not in working.columns:
        episode_has_event = pd.to_numeric(working.get("episode_has_event"), errors="coerce")
        if episode_has_event.notna().any():
            working["is_censored_episode"] = (episode_has_event.fillna(0).astype(int) == 0).astype(int)
    if "fpp_latency_from_partner_onset" not in working.columns and "own_fpp_onset" in working.columns:
        working["fpp_latency_from_partner_onset"] = (
            pd.to_numeric(working["own_fpp_onset"], errors="coerce")
            - pd.to_numeric(working["partner_ipu_onset"], errors="coerce")
        )
    if "fpp_latency_from_partner_offset" not in working.columns and "own_fpp_onset" in working.columns:
        working["fpp_latency_from_partner_offset"] = (
            pd.to_numeric(working["own_fpp_onset"], errors="coerce")
            - pd.to_numeric(working["partner_ipu_offset"], errors="coerce")
        )

    info_column = f"z_information_rate_lag_{int(information_rate_lag_ms)}ms"
    expected_column = f"z_prop_expected_cumulative_info_lag_{int(expected_cumulative_info_lag_ms)}ms"
    missing_predictors = [
        column_name
        for column_name in (info_column, expected_column)
        if column_name not in working.columns
    ]
    if missing_predictors:
        raise ValueError(
            "Required behavioural GLMM lagged predictor column(s) were not found: "
            + ", ".join(missing_predictors)
        )
    working["z_information_rate_lag_best"] = pd.to_numeric(working[info_column], errors="coerce")
    working["z_prop_expected_cumulative_info_lag_best"] = pd.to_numeric(
        working[expected_column],
        errors="coerce",
    )
    _validate_required_lag_columns(working, lag_grid_ms=lag_grid_ms)
    for column_name in (
        "event",
        "bin_end",
        "partner_ipu_onset",
        "partner_ipu_offset",
        "time_from_partner_onset",
        "time_from_partner_offset",
        "time_since_partner_offset_positive",
    ):
        if column_name in working.columns:
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    return working


def _available_optional_columns(table: pd.DataFrame) -> list[str]:
    return [column_name for column_name in OPTIONAL_EXPORT_COLUMNS if column_name in table.columns]


def _coerce_required_lag(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(
            "Behaviour GLMM export requires timing-controlled selected lags or explicit CLI lag values. "
            f"Missing `{key}`."
    )
    return int(value)


def _available_lag_columns(table: pd.DataFrame, *, lag_grid_ms: tuple[int, ...] | None) -> list[str]:
    configured_grid = tuple(sorted({int(lag_ms) for lag_ms in (lag_grid_ms or ())}))
    ordered_columns: list[str] = []
    for predictor_family in ("information_rate", "prop_expected_cumulative_info"):
        for lag_ms in configured_grid:
            column_name = f"z_{predictor_family}_lag_{int(lag_ms)}ms"
            if column_name in table.columns:
                ordered_columns.append(column_name)
    return ordered_columns


def _validate_required_lag_columns(table: pd.DataFrame, *, lag_grid_ms: tuple[int, ...] | None) -> None:
    if not lag_grid_ms:
        return
    missing_columns = []
    for predictor_family in ("information_rate", "prop_expected_cumulative_info"):
        for lag_ms in lag_grid_ms:
            column_name = f"z_{predictor_family}_lag_{int(lag_ms)}ms"
            if column_name not in table.columns:
                missing_columns.append(column_name)
    if missing_columns:
        raise ValueError(
            "Required behavioural GLMM lag-sweep predictor column(s) were not found: "
            + ", ".join(missing_columns)
        )
