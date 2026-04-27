"""Event-only export helpers for the exploratory behavioural latency-regime analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id
from cas.hazard_behavior.io import write_json, write_table
from cas.hazard_behavior.r_export import resolve_behaviour_glmm_lags

REQUIRED_EVENT_EXPORT_COLUMNS = (
    "dyad_id",
    "run",
    "speaker",
    "participant_speaker_id",
    "participant_speaker",
    "episode_id",
    "event",
    "fpp_onset",
    "partner_ipu_onset",
    "partner_ipu_offset",
    "latency_from_partner_onset",
    "latency_from_partner_offset",
    "z_information_rate_lag_best",
    "z_prop_expected_cumulative_info_lag_best",
)

OPTIONAL_EVENT_EXPORT_COLUMNS = (
    "time_within_run",
    "z_time_within_run",
    "z_time_within_run_squared",
    "run_index",
)

TIME_WITHIN_RUN_SOURCE_COLUMNS = (
    "time_within_run",
    "fpp_onset",
    "partner_ipu_offset",
    "partner_ipu_onset",
)


@dataclass(frozen=True, slots=True)
class BehaviourLatencyRegimeExportResult:
    """Paths and metadata for an event-only latency-regime export."""

    output_csv: Path
    output_qc_json: Path
    information_rate_lag_ms: int
    expected_cumulative_info_lag_ms: int
    n_rows_exported: int


def export_behaviour_latency_regime_data(
    riskset_table: pd.DataFrame,
    *,
    output_csv: Path,
    output_qc_json: Path,
    selected_lags_json: Path | None = None,
    information_rate_lag_ms: int | None = None,
    expected_cumulative_info_lag_ms: int | None = None,
    verbose: bool = False,
) -> BehaviourLatencyRegimeExportResult:
    """Create a compact event-only CSV for the exploratory latency-regime Stan analysis."""

    resolved_lags = resolve_behaviour_glmm_lags(
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
    )
    working = prepare_latency_regime_data(
        riskset_table,
        information_rate_lag_ms=resolved_lags["information_rate_lag_ms"],
        expected_cumulative_info_lag_ms=resolved_lags["expected_cumulative_info_lag_ms"],
    )
    n_rows_input = int(len(working))
    event_values = pd.to_numeric(working.get("event"), errors="coerce")
    n_events_input = int(event_values.fillna(0).sum())
    if verbose:
        print(
            "Preparing event-only latency-regime export "
            f"(n_rows_input={n_rows_input}, n_events_input={n_events_input}, "
            f"information_rate_lag_ms={resolved_lags['information_rate_lag_ms']}, "
            f"expected_cumulative_info_lag_ms={resolved_lags['expected_cumulative_info_lag_ms']})."
        )
    working = working.loc[pd.to_numeric(working["event"], errors="coerce") == 1].copy()
    if verbose:
        print(f"Filtered to observed FPP events (n_event_rows={len(working)}).")

    dropped_missing_counts_by_column = {
        column_name: int(working[column_name].isna().sum())
        for column_name in REQUIRED_EVENT_EXPORT_COLUMNS
    }
    required_mask = working.loc[:, REQUIRED_EVENT_EXPORT_COLUMNS].notna().all(axis=1)
    export_columns = list(REQUIRED_EVENT_EXPORT_COLUMNS) + [
        column_name for column_name in OPTIONAL_EVENT_EXPORT_COLUMNS if column_name in working.columns
    ]
    exported = working.loc[required_mask, export_columns].copy()

    exported["event"] = pd.to_numeric(exported["event"], errors="raise").astype(int)
    for column_name in ("dyad_id", "run", "speaker", "participant_speaker_id", "participant_speaker", "episode_id"):
        exported[column_name] = exported[column_name].astype(str)

    write_table(exported, output_csv, sep=",")
    if verbose:
        print(f"Wrote latency-regime event CSV to {output_csv} (n_rows_exported={len(exported)}).")

    latency = pd.to_numeric(exported["latency_from_partner_offset"], errors="coerce")
    identity_validation = validate_participant_speaker_id(
        exported,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
    )
    qc_payload = {
        "n_rows_input": n_rows_input,
        "n_events_input": n_events_input,
        "n_rows_exported": int(len(exported)),
        "n_dyads": int(exported["dyad_id"].nunique()),
        "n_participant_speaker_ids": int(exported["participant_speaker_id"].nunique()),
        "n_participant_speakers": int(exported["participant_speaker"].nunique()),
        "n_episodes": int(exported["episode_id"].nunique()),
        "information_rate_lag_ms": int(resolved_lags["information_rate_lag_ms"]),
        "expected_cumulative_info_lag_ms": int(resolved_lags["expected_cumulative_info_lag_ms"]),
        "latency_from_partner_offset_min": float(latency.min()),
        "latency_from_partner_offset_max": float(latency.max()),
        "latency_from_partner_offset_mean": float(latency.mean()),
        "latency_from_partner_offset_median": float(latency.median()),
        "proportion_negative_latency_from_partner_offset": float((latency < 0.0).mean()),
        "required_columns": list(REQUIRED_EVENT_EXPORT_COLUMNS),
        "optional_columns": [column_name for column_name in OPTIONAL_EVENT_EXPORT_COLUMNS if column_name in exported.columns],
        "dropped_missing_counts_by_column": dropped_missing_counts_by_column,
        "controls": {
            "run_available": bool(exported["run"].notna().any()),
            "time_within_run_available": "time_within_run" in exported.columns
            and bool(pd.to_numeric(exported["time_within_run"], errors="coerce").notna().any()),
            "time_within_run_source": _resolve_time_within_run_source(working),
        },
        "identity_validation": identity_validation,
    }
    write_json(qc_payload, output_qc_json)
    if verbose:
        print(f"Wrote latency-regime export QC JSON to {output_qc_json}.")
    return BehaviourLatencyRegimeExportResult(
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        information_rate_lag_ms=int(resolved_lags["information_rate_lag_ms"]),
        expected_cumulative_info_lag_ms=int(resolved_lags["expected_cumulative_info_lag_ms"]),
        n_rows_exported=int(len(exported)),
    )


def export_behaviour_latency_regime_data_from_path(
    *,
    input_riskset: Path,
    output_csv: Path,
    output_qc_json: Path,
    selected_lags_json: Path | None = None,
    information_rate_lag_ms: int | None = None,
    expected_cumulative_info_lag_ms: int | None = None,
    verbose: bool = False,
) -> BehaviourLatencyRegimeExportResult:
    """Load a risk-set table from disk and export an event-only latency dataset."""

    riskset_table = pd.read_csv(input_riskset, sep=None, engine="python")
    return export_behaviour_latency_regime_data(
        riskset_table,
        output_csv=output_csv,
        output_qc_json=output_qc_json,
        selected_lags_json=selected_lags_json,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
        verbose=verbose,
    )


def _prepare_latency_regime_columns(
    riskset_table: pd.DataFrame,
    *,
    information_rate_lag_ms: int,
    expected_cumulative_info_lag_ms: int,
) -> pd.DataFrame:
    working = riskset_table.copy()
    if "dyad_id" not in working.columns:
        raise ValueError("Latency-regime export requires `dyad_id`.")
    if "run" not in working.columns:
        if "run_id" in working.columns:
            working["run"] = working["run_id"]
        else:
            working["run"] = "run-unknown"
    if "speaker" not in working.columns:
        if "participant_speaker" not in working.columns:
            raise ValueError("Latency-regime export requires `speaker` or `participant_speaker`.")
        legacy_participant = working["participant_speaker"].astype(str)
        inferred_speaker = legacy_participant.str.rsplit("_", n=1).str[-1]
        canonical_from_legacy = working["dyad_id"].astype(str) + "_" + inferred_speaker
        if legacy_participant.equals(canonical_from_legacy):
            working["speaker"] = inferred_speaker
            if "participant_speaker_id" not in working.columns:
                working["participant_speaker_id"] = legacy_participant
        else:
            working["speaker"] = legacy_participant
    working = ensure_participant_speaker_id(
        working,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
        overwrite="participant_speaker_id" not in working.columns,
    )
    if "participant_speaker" not in working.columns:
        working["participant_speaker"] = working["speaker"].astype(str)
    if "fpp_onset" not in working.columns:
        if "own_fpp_onset" not in working.columns:
            raise ValueError("Latency-regime export requires `fpp_onset` or `own_fpp_onset`.")
        working["fpp_onset"] = working["own_fpp_onset"]
    if "latency_from_partner_onset" not in working.columns:
        working["latency_from_partner_onset"] = (
            pd.to_numeric(working["fpp_onset"], errors="coerce")
            - pd.to_numeric(working["partner_ipu_onset"], errors="coerce")
        )
    if "latency_from_partner_offset" not in working.columns:
        working["latency_from_partner_offset"] = (
            pd.to_numeric(working["fpp_onset"], errors="coerce")
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
            "Required latency-regime lagged predictor column(s) were not found: "
            + ", ".join(missing_predictors)
        )
    working["z_information_rate_lag_best"] = pd.to_numeric(working[info_column], errors="coerce")
    working["z_prop_expected_cumulative_info_lag_best"] = pd.to_numeric(working[expected_column], errors="coerce")

    numeric_columns = (
        "event",
        "fpp_onset",
        "partner_ipu_onset",
        "partner_ipu_offset",
        "latency_from_partner_onset",
        "latency_from_partner_offset",
        "z_information_rate_lag_best",
        "z_prop_expected_cumulative_info_lag_best",
    )
    for column_name in numeric_columns:
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    return working


def prepare_latency_regime_data(
    riskset_table: pd.DataFrame,
    *,
    information_rate_lag_ms: int,
    expected_cumulative_info_lag_ms: int,
) -> pd.DataFrame:
    """Prepare event-level latency-regime data with optional timing controls.

    Parameters
    ----------
    riskset_table
        Behavioural hazard risk-set table containing event rows and lagged predictors.
    information_rate_lag_ms
        Selected lag in milliseconds for the information-rate predictor.
    expected_cumulative_info_lag_ms
        Selected lag in milliseconds for the expected cumulative information predictor.

    Returns
    -------
    pandas.DataFrame
        Event-level table with latency variables in seconds, lag-best predictors,
        and optional run/time-within-run control columns when available or derivable.
    """

    working = _prepare_latency_regime_columns(
        riskset_table,
        information_rate_lag_ms=information_rate_lag_ms,
        expected_cumulative_info_lag_ms=expected_cumulative_info_lag_ms,
    )
    working = _add_latency_regime_controls(working)
    return working


def _add_latency_regime_controls(riskset_table: pd.DataFrame) -> pd.DataFrame:
    """Add optional run and time-within-run controls for latency-regime models."""

    working = riskset_table.copy()
    run_labels = working["run"].astype(str).fillna("run-unknown")
    working["run_index"] = pd.Categorical(run_labels).codes + 1

    time_within_run = _derive_time_within_run_seconds(working)
    if time_within_run is not None:
        working["time_within_run"] = time_within_run
        z_time = _zscore_series(pd.to_numeric(working["time_within_run"], errors="coerce"))
        working["z_time_within_run"] = z_time
        working["z_time_within_run_squared"] = np.square(z_time)
    return working


def _derive_time_within_run_seconds(riskset_table: pd.DataFrame) -> pd.Series | None:
    """Return a per-run continuous timing covariate in seconds when available."""

    working = riskset_table.copy()
    if "time_within_run" in working.columns:
        values = pd.to_numeric(working["time_within_run"], errors="coerce")
        if values.notna().any():
            return values

    for candidate in ("fpp_onset", "partner_ipu_offset", "partner_ipu_onset"):
        if candidate not in working.columns:
            continue
        values = pd.to_numeric(working[candidate], errors="coerce")
        if not values.notna().any():
            continue
        group_columns = [column_name for column_name in ("dyad_id", "run") if column_name in working.columns]
        if not group_columns:
            return values - float(values.min())
        group_min = values.groupby([working[column_name].astype(str) for column_name in group_columns]).transform("min")
        return values - group_min
    return None


def _zscore_series(values: pd.Series) -> pd.Series:
    """Z-score a numeric series while preserving missing values."""

    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    mean_value = float(finite.mean())
    std_value = float(finite.std(ddof=0))
    if std_value <= 0.0 or not np.isfinite(std_value):
        return pd.Series(0.0, index=values.index, dtype=float).where(numeric.notna(), np.nan)
    return (numeric - mean_value) / std_value


def _resolve_time_within_run_source(riskset_table: pd.DataFrame) -> str:
    """Describe how the time-within-run control was obtained."""

    for column_name in TIME_WITHIN_RUN_SOURCE_COLUMNS:
        if column_name not in riskset_table.columns:
            continue
        values = pd.to_numeric(riskset_table[column_name], errors="coerce")
        if not values.notna().any():
            continue
        if column_name == "time_within_run":
            return "observed:time_within_run"
        return f"derived:{column_name}"
    return "unavailable"
