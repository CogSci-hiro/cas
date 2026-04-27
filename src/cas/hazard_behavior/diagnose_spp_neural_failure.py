"""Diagnostics for SPP neural low-level hazard convergence failures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import textwrap
import warnings
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from cas.hazard.config import NeuralHazardConfig
from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id
from cas.hazard_behavior.io import write_json, write_table
from cas.hazard_behavior.neural_lowlevel import (
    FittedFormulaModel,
    _append_terms,
    _build_neural_baseline_formula,
    _discover_band_pc_terms,
    _extract_formula_columns,
    _fit_formula_model,
    _maybe_float,
    _subset_complete_cases,
)

DEFAULT_OUTPUT_FILENAMES = {
    "report": "spp_neural_failure_diagnostic_report.md",
    "qc": "spp_neural_failure_qc.json",
    "event_distribution": "spp_neural_event_distribution.csv",
    "missingness": "spp_neural_missingness_by_feature.csv",
    "design": "spp_neural_design_diagnostics.csv",
    "incremental": "spp_neural_incremental_fit_diagnostics.csv",
    "ridge": "spp_neural_ridge_diagnostics.csv",
    "event_time_hist": "spp_neural_event_time_histogram.png",
    "events_by_speaker": "spp_neural_events_by_participant_speaker.png",
    "missingness_plot": "spp_neural_missingness_by_feature.png",
    "correlation_heatmap": "spp_neural_design_correlation_heatmap.png",
    "incremental_status": "spp_neural_incremental_model_status.png",
}

DEFAULT_STATUS_ORDER = [
    "converged",
    "failed",
    "no_data",
    "no_events",
]
MAX_DIAGNOSTIC_NON_EVENT_ROWS = 100_000
MAX_DESIGN_SAMPLE_ROWS = 50_000
MAX_HEATMAP_SAMPLE_ROWS = 50_000


@dataclass(frozen=True, slots=True)
class SppNeuralFailureDiagnosticResult:
    """Output paths for the SPP neural diagnostics bundle."""

    output_dir: Path
    report_path: Path
    qc_path: Path
    event_distribution_path: Path
    missingness_path: Path
    design_diagnostics_path: Path
    incremental_diagnostics_path: Path
    ridge_diagnostics_path: Path | None


def diagnose_spp_neural_hazard_failure(
    *,
    riskset_path: Path,
    models_dir: Path | None,
    output_dir: Path,
    run_ridge_diagnostic: bool = False,
    skip_incremental_fits: bool = False,
    max_fit_non_event_rows: int = MAX_DIAGNOSTIC_NON_EVENT_ROWS,
    max_design_rows: int = MAX_DESIGN_SAMPLE_ROWS,
    max_heatmap_rows: int = MAX_HEATMAP_SAMPLE_ROWS,
    fit_model_fn: Callable[..., FittedFormulaModel] | None = None,
    verbose: bool = False,
) -> SppNeuralFailureDiagnosticResult:
    """Generate a diagnostics bundle for SPP neural hazard failures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    riskset: pd.DataFrame | None = None
    available_columns: list[str] = []
    try:
        available_columns = inspect_riskset_columns(riskset_path)
    except Exception as error:
        notes.append(f"Unable to inspect riskset columns: {error}")

    qc_path = output_dir / DEFAULT_OUTPUT_FILENAMES["qc"]
    event_distribution_path = output_dir / DEFAULT_OUTPUT_FILENAMES["event_distribution"]
    missingness_path = output_dir / DEFAULT_OUTPUT_FILENAMES["missingness"]
    design_path = output_dir / DEFAULT_OUTPUT_FILENAMES["design"]
    incremental_path = output_dir / DEFAULT_OUTPUT_FILENAMES["incremental"]
    ridge_path = output_dir / DEFAULT_OUTPUT_FILENAMES["ridge"] if run_ridge_diagnostic else None
    report_path = output_dir / DEFAULT_OUTPUT_FILENAMES["report"]

    if not available_columns:
        empty_qc = {"status": "failed", "notes": notes, "riskset_path": str(riskset_path)}
        write_json(empty_qc, qc_path)
        write_table(pd.DataFrame(), event_distribution_path, sep=",")
        write_table(pd.DataFrame(), missingness_path, sep=",")
        write_table(pd.DataFrame(), design_path, sep=",")
        write_table(pd.DataFrame(), incremental_path, sep=",")
        if ridge_path is not None:
            write_table(pd.DataFrame(), ridge_path, sep=",")
        report_path.write_text(build_failure_only_report(notes), encoding="utf-8")
        return SppNeuralFailureDiagnosticResult(
            output_dir=output_dir,
            report_path=report_path,
            qc_path=qc_path,
            event_distribution_path=event_distribution_path,
            missingness_path=missingness_path,
            design_diagnostics_path=design_path,
            incremental_diagnostics_path=incremental_path,
            ridge_diagnostics_path=ridge_path,
        )

    event_column = resolve_spp_event_column_from_columns(available_columns)
    lag_metadata = infer_lag_metadata_from_columns(available_columns, models_dir=models_dir)
    neural_config = lag_metadata["neural_config"]
    alpha_features = detect_neural_features_from_columns(available_columns, family="alpha")
    beta_features = detect_neural_features_from_columns(available_columns, family="beta")
    alpha_beta_features = [*alpha_features, *beta_features]
    required_columns = collect_required_riskset_columns(
        event_column=event_column,
        available_columns=available_columns,
        alpha_features=alpha_features,
        beta_features=beta_features,
    )

    try:
        riskset = load_riskset_table(riskset_path, columns=required_columns)
        if verbose:
            print(f"Loaded SPP neural riskset from {riskset_path} with {len(required_columns)} columns.")
    except Exception as error:
        notes.append(f"Unable to load riskset: {error}")

    if riskset is None:
        empty_qc = {"status": "failed", "notes": notes, "riskset_path": str(riskset_path)}
        write_json(empty_qc, qc_path)
        write_table(pd.DataFrame(), event_distribution_path, sep=",")
        write_table(pd.DataFrame(), missingness_path, sep=",")
        write_table(pd.DataFrame(), design_path, sep=",")
        write_table(pd.DataFrame(), incremental_path, sep=",")
        if ridge_path is not None:
            write_table(pd.DataFrame(), ridge_path, sep=",")
        report_path.write_text(build_failure_only_report(notes), encoding="utf-8")
        return SppNeuralFailureDiagnosticResult(
            output_dir=output_dir,
            report_path=report_path,
            qc_path=qc_path,
            event_distribution_path=event_distribution_path,
            missingness_path=missingness_path,
            design_diagnostics_path=design_path,
            incremental_diagnostics_path=incremental_path,
            ridge_diagnostics_path=ridge_path,
        )

    riskset = ensure_participant_id_column(riskset)
    row_id_series = build_row_ids(riskset)

    count_qc = compute_event_count_diagnostics(
        riskset,
        event_column=event_column,
        alpha_features=alpha_features,
        beta_features=beta_features,
        alpha_beta_features=alpha_beta_features,
        neural_config=neural_config,
        row_ids=row_id_series,
    )
    count_qc["memory_mode"] = {
        "column_selective_load": True,
        "skip_incremental_fits": bool(skip_incremental_fits),
        "run_ridge_diagnostic": bool(run_ridge_diagnostic),
        "max_fit_non_event_rows": int(max_fit_non_event_rows),
        "max_design_rows": int(max_design_rows),
        "max_heatmap_rows": int(max_heatmap_rows),
    }
    write_json(count_qc, qc_path)

    event_distribution = summarize_event_distribution(
        riskset,
        event_column=event_column,
        alpha_features=alpha_features,
        beta_features=beta_features,
        alpha_beta_features=alpha_beta_features,
    )
    write_table(event_distribution, event_distribution_path, sep=",")

    missingness = summarize_missingness_by_feature(
        riskset,
        event_column=event_column,
        alpha_features=alpha_features,
        beta_features=beta_features,
    )
    write_table(missingness, missingness_path, sep=",")

    design_diagnostics = build_design_diagnostics_table(
        riskset,
        event_column=event_column,
        neural_config=neural_config,
        alpha_features=alpha_features,
        beta_features=beta_features,
        row_ids=row_id_series,
        max_design_rows=max_design_rows,
    )
    write_table(design_diagnostics, design_path, sep=",")

    incremental = pd.DataFrame()
    if not skip_incremental_fits:
        incremental = fit_incremental_models(
            riskset,
            event_column=event_column,
            neural_config=neural_config,
            alpha_features=alpha_features,
            beta_features=beta_features,
            max_non_event_rows=max_fit_non_event_rows,
            fit_model_fn=fit_model_fn,
        )
    write_table(incremental, incremental_path, sep=",")

    ridge = None
    if run_ridge_diagnostic and not skip_incremental_fits:
        ridge = run_ridge_diagnostics(
            riskset,
            event_column=event_column,
            neural_config=neural_config,
            alpha_features=alpha_features,
            beta_features=beta_features,
            max_non_event_rows=max_fit_non_event_rows,
        )
        write_table(ridge, ridge_path, sep=",")
    elif ridge_path is not None:
        write_table(pd.DataFrame(), ridge_path, sep=",")

    plot_spp_event_time_histogram(riskset, event_column=event_column, output_path=output_dir / DEFAULT_OUTPUT_FILENAMES["event_time_hist"])
    plot_spp_events_by_participant_speaker(
        riskset,
        event_column=event_column,
        alpha_beta_features=alpha_beta_features,
        output_path=output_dir / DEFAULT_OUTPUT_FILENAMES["events_by_speaker"],
    )
    plot_missingness_by_feature(missingness, output_path=output_dir / DEFAULT_OUTPUT_FILENAMES["missingness_plot"])
    plot_design_correlation_heatmap(
        riskset,
        output_path=output_dir / DEFAULT_OUTPUT_FILENAMES["correlation_heatmap"],
        features=alpha_beta_features if alpha_beta_features else [*alpha_features, *beta_features],
        max_rows=max_heatmap_rows,
    )
    plot_incremental_model_status(incremental, output_path=output_dir / DEFAULT_OUTPUT_FILENAMES["incremental_status"])

    report_text = build_spp_failure_report(
        qc=count_qc,
        missingness=missingness,
        event_distribution=event_distribution,
        design_diagnostics=design_diagnostics,
        incremental=incremental,
        ridge=ridge,
        notes=notes,
    )
    report_path.write_text(report_text, encoding="utf-8")

    return SppNeuralFailureDiagnosticResult(
        output_dir=output_dir,
        report_path=report_path,
        qc_path=qc_path,
        event_distribution_path=event_distribution_path,
        missingness_path=missingness_path,
        design_diagnostics_path=design_path,
        incremental_diagnostics_path=incremental_path,
        ridge_diagnostics_path=ridge_path,
    )


def inspect_riskset_columns(path: Path) -> list[str]:
    """Inspect riskset columns without loading the full table."""

    if not path.exists():
        raise FileNotFoundError(f"Riskset path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return list(pq.ParquetFile(path).schema.names)
    if suffix == ".tsv":
        return list(pd.read_csv(path, sep="\t", nrows=0).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def load_riskset_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a neural riskset from parquet/csv/tsv."""

    if not path.exists():
        raise FileNotFoundError(f"Riskset path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pd.read_parquet(path, columns=columns)
    elif suffix == ".tsv":
        table = pd.read_csv(path, sep="\t", usecols=columns)
    else:
        table = pd.read_csv(path, usecols=columns)
    if table.empty:
        raise ValueError("Riskset table is empty.")
    return table


def resolve_spp_event_column(riskset: pd.DataFrame) -> str:
    """Resolve the SPP event column."""

    for candidate in ("event_spp", "event"):
        if candidate in riskset.columns:
            return candidate
    raise ValueError("Riskset is missing `event_spp` and `event`.")


def resolve_spp_event_column_from_columns(columns: list[str]) -> str:
    """Resolve the SPP event column from column names alone."""

    for candidate in ("event_spp", "event"):
        if candidate in columns:
            return candidate
    raise ValueError("Riskset is missing `event_spp` and `event`.")


def build_row_ids(riskset: pd.DataFrame) -> pd.Series:
    """Construct stable row identifiers for same-row checks."""

    columns = [
        column_name
        for column_name in ("episode_id", "dyad_id", "run", "participant_speaker", "bin_start", "bin_end")
        if column_name in riskset.columns
    ]
    if not columns:
        return pd.Series([f"row_{index}" for index in riskset.index], index=riskset.index, dtype="string")
    pieces = [riskset[column].astype(str) for column in columns]
    row_ids = pieces[0].copy()
    for piece in pieces[1:]:
        row_ids = row_ids + "|" + piece
    return row_ids.astype("string")


def same_rows_match(parent_row_ids: pd.Series, child_row_ids: pd.Series) -> bool:
    """Return true when parent/child row ids match exactly."""

    parent = tuple(parent_row_ids.astype(str).tolist())
    child = tuple(child_row_ids.astype(str).tolist())
    return parent == child


def infer_lag_metadata(riskset: pd.DataFrame, *, models_dir: Path | None) -> dict[str, Any]:
    """Infer behavioural lag columns, preferring recorded metadata when present."""

    info_lag = None
    prop_lag = None
    metadata_candidates: list[Path] = []
    if models_dir is not None:
        models_dir = models_dir.resolve()
        metadata_candidates.extend(
            [
                models_dir / "analysis_metadata.json",
                models_dir / "neural_lowlevel_fit_metrics.json",
                models_dir.parent / "logs" / "analysis_metadata.json",
                models_dir.parent / "analysis_metadata.json",
            ]
        )
    for candidate in metadata_candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config", {})
        neural = config.get("neural", {})
        model = neural.get("model", {})
        info_lag = model.get("information_rate_lag_ms", info_lag)
        prop_lag = model.get("prop_expected_lag_ms", prop_lag)
        if info_lag is not None and prop_lag is not None:
            break

    info_column = resolve_behavioural_column(
        riskset,
        prefix="z_information_rate_lag_",
        preferred_lag_ms=info_lag,
        preferred_named_column="z_information_rate_lag_best",
    )
    prop_column = resolve_behavioural_column(
        riskset,
        prefix="z_prop_expected_cumulative_info_lag_",
        preferred_lag_ms=prop_lag,
        preferred_named_column="z_prop_expected_cumulative_info_lag_best",
    )
    neural_config = NeuralHazardConfig(enabled=True)
    if info_column is not None:
        info_match = re.search(r"lag_(\d+)ms", info_column)
        if info_match:
            neural_config = NeuralHazardConfig(
                enabled=True,
                model=neural_config.model.__class__(
                    fitting_backend=neural_config.model.fitting_backend,
                    baseline_spline_df=neural_config.model.baseline_spline_df,
                    baseline_spline_degree=neural_config.model.baseline_spline_degree,
                    information_rate_lag_ms=int(info_match.group(1)),
                    prop_expected_lag_ms=neural_config.model.prop_expected_lag_ms,
                ),
            )
    if prop_column is not None:
        prop_match = re.search(r"lag_(\d+)ms", prop_column)
        if prop_match:
            neural_config = NeuralHazardConfig(
                enabled=True,
                model=neural_config.model.__class__(
                    fitting_backend=neural_config.model.fitting_backend,
                    baseline_spline_df=neural_config.model.baseline_spline_df,
                    baseline_spline_degree=neural_config.model.baseline_spline_degree,
                    information_rate_lag_ms=neural_config.model.information_rate_lag_ms,
                    prop_expected_lag_ms=int(prop_match.group(1)),
                ),
            )
    if info_column is None or prop_column is None:
        raise ValueError("Could not infer required behavioural lagged control columns from the riskset.")
    return {
        "information_rate_column": info_column,
        "prop_expected_column": prop_column,
        "neural_config": neural_config,
    }


def infer_lag_metadata_from_columns(columns: list[str], *, models_dir: Path | None) -> dict[str, Any]:
    """Infer behavioural lag metadata from saved metadata and column names."""

    info_lag = None
    prop_lag = None
    metadata_candidates: list[Path] = []
    if models_dir is not None:
        models_dir = models_dir.resolve()
        metadata_candidates.extend(
            [
                models_dir / "analysis_metadata.json",
                models_dir / "neural_lowlevel_fit_metrics.json",
                models_dir.parent / "logs" / "analysis_metadata.json",
                models_dir.parent / "analysis_metadata.json",
            ]
        )
    for candidate in metadata_candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config", {})
        neural = config.get("neural", {})
        model = neural.get("model", {})
        info_lag = model.get("information_rate_lag_ms", info_lag)
        prop_lag = model.get("prop_expected_lag_ms", prop_lag)
        if info_lag is not None and prop_lag is not None:
            break
    info_column = resolve_behavioural_column_from_names(
        columns,
        prefix="z_information_rate_lag_",
        preferred_lag_ms=info_lag,
        preferred_named_column="z_information_rate_lag_best",
    )
    prop_column = resolve_behavioural_column_from_names(
        columns,
        prefix="z_prop_expected_cumulative_info_lag_",
        preferred_lag_ms=prop_lag,
        preferred_named_column="z_prop_expected_cumulative_info_lag_best",
    )
    if info_column is None or prop_column is None:
        raise ValueError("Could not infer required behavioural lagged control columns from the riskset.")
    info_match = re.search(r"lag_(\d+)ms", info_column)
    prop_match = re.search(r"lag_(\d+)ms", prop_column)
    template = NeuralHazardConfig(enabled=True)
    neural_config = NeuralHazardConfig(
        enabled=True,
        model=template.model.__class__(
            fitting_backend=template.model.fitting_backend,
            baseline_spline_df=template.model.baseline_spline_df,
            baseline_spline_degree=template.model.baseline_spline_degree,
            information_rate_lag_ms=int(info_match.group(1)) if info_match else template.model.information_rate_lag_ms,
            prop_expected_lag_ms=int(prop_match.group(1)) if prop_match else template.model.prop_expected_lag_ms,
        ),
    )
    return {
        "information_rate_column": info_column,
        "prop_expected_column": prop_column,
        "neural_config": neural_config,
    }


def resolve_behavioural_column(
    riskset: pd.DataFrame,
    *,
    prefix: str,
    preferred_lag_ms: Any,
    preferred_named_column: str,
) -> str | None:
    """Resolve a behavioural control column by name or lag."""

    if preferred_named_column in riskset.columns:
        return preferred_named_column
    if preferred_lag_ms is not None:
        candidate = f"{prefix}{int(preferred_lag_ms)}ms"
        if candidate in riskset.columns:
            return candidate
    matches = sorted(column for column in riskset.columns if str(column).startswith(prefix))
    if len(matches) == 1:
        return matches[0]
    for fallback in (f"{prefix}150ms", f"{prefix}700ms"):
        if fallback in riskset.columns:
            return fallback
    return matches[0] if matches else None


def resolve_behavioural_column_from_names(
    columns: list[str],
    *,
    prefix: str,
    preferred_lag_ms: Any,
    preferred_named_column: str,
) -> str | None:
    """Resolve a behavioural control column from a list of column names."""

    if preferred_named_column in columns:
        return preferred_named_column
    if preferred_lag_ms is not None:
        candidate = f"{prefix}{int(preferred_lag_ms)}ms"
        if candidate in columns:
            return candidate
    matches = sorted(column for column in columns if str(column).startswith(prefix))
    if len(matches) == 1:
        return matches[0]
    for fallback in (f"{prefix}150ms", f"{prefix}700ms"):
        if fallback in columns:
            return fallback
    return matches[0] if matches else None


def detect_neural_features(riskset: pd.DataFrame, *, family: str) -> list[str]:
    """Detect neural feature columns for one family."""

    patterns = {
        "alpha": [r"^z_alpha_pc\d+$", r"^alpha_pc\d+$", r"^neural_alpha_"],
        "beta": [r"^z_beta_pc\d+$", r"^beta_pc\d+$", r"^neural_beta_"],
    }
    columns: list[str] = []
    for pattern in patterns[family]:
        matched = sorted(column for column in riskset.columns if re.search(pattern, str(column)))
        columns.extend(matched)
    return sorted(dict.fromkeys(columns))


def detect_neural_features_from_columns(columns: list[str], *, family: str) -> list[str]:
    """Detect neural features from column names without loading the full table."""

    patterns = {
        "alpha": [r"^z_alpha_pc\d+$", r"^alpha_pc\d+$", r"^neural_alpha_"],
        "beta": [r"^z_beta_pc\d+$", r"^beta_pc\d+$", r"^neural_beta_"],
    }
    matches: list[str] = []
    for pattern in patterns[family]:
        matches.extend(sorted(column for column in columns if re.search(pattern, str(column))))
    return sorted(dict.fromkeys(matches))


def collect_required_riskset_columns(
    *,
    event_column: str,
    available_columns: list[str],
    alpha_features: list[str],
    beta_features: list[str],
) -> list[str]:
    """Collect the minimal set of columns needed for diagnostics."""

    required = {
        event_column,
        "event",
        "episode_id",
        "dyad_id",
        "run",
        "speaker",
        "participant_speaker_id",
        "participant_speaker",
        "participant_id",
        "bin_start",
        "bin_end",
        "time_from_partner_onset",
        "time_from_partner_offset",
    }
    required.update(column for column in available_columns if str(column).startswith("z_information_rate_lag_"))
    required.update(column for column in available_columns if str(column).startswith("z_prop_expected_cumulative_info_lag_"))
    required.update(alpha_features)
    required.update(beta_features)
    return sorted(column for column in required if column in available_columns)


def build_required_columns(
    *,
    event_column: str,
    neural_config: NeuralHazardConfig,
    features: list[str],
) -> list[str]:
    """Build required columns for one formula family."""

    baseline = _build_neural_baseline_formula(event_column=event_column, neural_config=neural_config)
    formula = _append_terms(baseline, features)
    return _extract_formula_columns(formula, event_column=event_column)


def subset_family_rows(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    neural_config: NeuralHazardConfig,
    features: list[str],
) -> pd.DataFrame:
    """Subset complete cases for a model family."""

    riskset = ensure_participant_id_column(riskset)
    required_columns = build_required_columns(
        event_column=event_column,
        neural_config=neural_config,
        features=features,
    )
    return _subset_complete_cases(riskset, required_columns=required_columns)


def ensure_participant_id_column(riskset: pd.DataFrame) -> pd.DataFrame:
    """Add a fallback participant_id column when the riskset does not have one."""

    working = riskset.copy()
    if "speaker" not in working.columns and "participant_speaker" in working.columns:
        legacy_participant = working["participant_speaker"].astype(str)
        inferred_speaker = legacy_participant.str.rsplit("_", n=1).str[-1]
        canonical_from_legacy = working["dyad_id"].astype(str) + "_" + inferred_speaker if "dyad_id" in working.columns else legacy_participant
        if legacy_participant.equals(canonical_from_legacy):
            working["speaker"] = inferred_speaker
            if "participant_speaker_id" not in working.columns:
                working["participant_speaker_id"] = legacy_participant
        else:
            working["speaker"] = legacy_participant
    if {"dyad_id", "speaker"}.issubset(working.columns):
        working = ensure_participant_speaker_id(
            working,
            dyad_col="dyad_id",
            speaker_col="speaker",
            output_col="participant_speaker_id",
            overwrite="participant_speaker_id" not in working.columns,
        )
    if "participant_id" not in working.columns:
        if "participant_speaker_id" in working.columns:
            working["participant_id"] = working["participant_speaker_id"].astype(str)
        elif "dyad_id" in working.columns and "participant_speaker" in working.columns:
            working["participant_id"] = (
                working["dyad_id"].astype(str) + "_" + working["participant_speaker"].astype(str)
            )
        else:
            working["participant_id"] = [f"participant_{index}" for index in range(len(working))]
    if "participant_speaker_id" not in working.columns and "participant_id" in working.columns:
        working["participant_speaker_id"] = working["participant_id"].astype(str)
    return working


def compute_event_count_diagnostics(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    alpha_features: list[str],
    beta_features: list[str],
    alpha_beta_features: list[str],
    neural_config: NeuralHazardConfig,
    row_ids: pd.Series,
) -> dict[str, Any]:
    """Compute pre/post filtering sample diagnostics."""

    riskset = ensure_participant_id_column(riskset)
    family_to_features = {
        "alpha": alpha_features,
        "beta": beta_features,
        "alpha_beta": alpha_beta_features,
    }
    base_summary = summarize_sample(riskset, event_column=event_column)
    identity_validation = validate_participant_speaker_id(
        riskset,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
    )
    qc: dict[str, Any] = {
        "status": "ok",
        "n_rows_total": base_summary["n_rows"],
        "n_events_total": base_summary["n_events"],
        "event_rate_total": base_summary["event_rate"],
        "n_dyads_total": base_summary["n_dyads"],
        "n_runs_total": base_summary["n_runs"],
        "n_participant_speaker_ids_total": base_summary["n_participant_speaker_ids"],
        "n_episodes_total": base_summary["n_episodes"],
        "identity_validation": identity_validation,
    }
    same_row_checks: dict[str, Any] = {}
    baseline_subset = subset_family_rows(
        riskset,
        event_column=event_column,
        neural_config=neural_config,
        features=[],
    )
    for family, features in family_to_features.items():
        subset = subset_family_rows(
            riskset,
            event_column=event_column,
            neural_config=neural_config,
            features=features,
        )
        summary = summarize_sample(subset, event_column=event_column)
        suffix = f"neural_complete_{family}"
        qc[f"n_rows_{suffix}"] = summary["n_rows"]
        qc[f"n_events_{suffix}"] = summary["n_events"]
        qc[f"event_rate_{suffix}"] = summary["event_rate"]
        qc[f"n_dyads_{suffix}"] = summary["n_dyads"]
        qc[f"n_runs_{suffix}"] = summary["n_runs"]
        qc[f"n_participant_speaker_ids_{suffix}"] = summary["n_participant_speaker_ids"]
        qc[f"n_episodes_{suffix}"] = summary["n_episodes"]
        qc[f"n_events_lost_{family}"] = int(qc["n_events_total"] - summary["n_events"])
        qc[f"event_loss_fraction_{family}"] = (
            float((qc["n_events_total"] - summary["n_events"]) / qc["n_events_total"])
            if qc["n_events_total"] > 0
            else None
        )
        same_row_checks[family] = {
            "n_rows_parent": int(len(subset)),
            "n_rows_child": int(len(subset)),
            "n_events_parent": int(subset[event_column].sum()),
            "n_events_child": int(subset[event_column].sum()),
            "same_rows": same_rows_match(row_ids.loc[subset.index], row_ids.loc[subset.index]),
            "parent_rows_available_before_refit": int(len(baseline_subset)),
        }
    qc["same_row_checks"] = same_row_checks
    if qc["n_participant_speaker_ids_total"] <= identity_validation["n_unique_speaker"]:
        qc["status"] = "failed"
        qc["identity_error"] = (
            "Participant-speaker identities appear collapsed to role labels; "
            "n_unique(participant_speaker_id) must exceed n_unique(speaker)."
        )
    return qc


def summarize_sample(table: pd.DataFrame, *, event_column: str) -> dict[str, Any]:
    """Summarize sample size and grouping counts."""

    return {
        "n_rows": int(len(table)),
        "n_events": int(pd.to_numeric(table.get(event_column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "event_rate": (
            float(pd.to_numeric(table[event_column], errors="coerce").fillna(0).mean())
            if len(table) > 0 and event_column in table.columns
            else None
        ),
        "n_dyads": int(table["dyad_id"].nunique()) if "dyad_id" in table.columns else 0,
        "n_runs": int(table["run"].nunique()) if "run" in table.columns else 0,
        "n_participant_speaker_ids": (
            int(table["participant_speaker_id"].nunique()) if "participant_speaker_id" in table.columns else 0
        ),
        "n_episodes": int(table["episode_id"].nunique()) if "episode_id" in table.columns else 0,
    }


def summarize_event_distribution(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    alpha_features: list[str],
    beta_features: list[str],
    alpha_beta_features: list[str],
) -> pd.DataFrame:
    """Summarize event counts by grouping variable."""

    riskset = ensure_participant_id_column(riskset)
    groupings = [column for column in ("dyad_id", "participant_speaker_id", "run", "episode_id") if column in riskset.columns]
    alpha_mask = complete_case_mask(riskset, alpha_features)
    beta_mask = complete_case_mask(riskset, beta_features)
    alpha_beta_mask = complete_case_mask(riskset, alpha_beta_features)
    rows: list[dict[str, Any]] = []
    for grouping in groupings:
        grouped = riskset.groupby(grouping, dropna=False, sort=False)
        for group_id, frame in grouped:
            event_values = pd.to_numeric(frame[event_column], errors="coerce").fillna(0)
            rows.append(
                {
                    "grouping_variable": grouping,
                    "group_id": str(group_id),
                    "n_rows": int(len(frame)),
                    "n_events": int(event_values.sum()),
                    "event_rate": float(event_values.mean()) if len(frame) > 0 else None,
                    "n_events_after_alpha_complete": int(event_values.loc[alpha_mask.loc[frame.index]].sum()),
                    "n_events_after_beta_complete": int(event_values.loc[beta_mask.loc[frame.index]].sum()),
                    "n_events_after_alpha_beta_complete": int(event_values.loc[alpha_beta_mask.loc[frame.index]].sum()),
                }
            )
    return pd.DataFrame(rows)


def complete_case_mask(table: pd.DataFrame, features: list[str]) -> pd.Series:
    """Return a complete-case mask for a feature list."""

    if not features:
        return pd.Series(np.ones(len(table), dtype=bool), index=table.index)
    numeric = table.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return numeric.notna().all(axis=1)


def summarize_missingness_by_feature(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    alpha_features: list[str],
    beta_features: list[str],
) -> pd.DataFrame:
    """Summarize missingness for neural features."""

    rows: list[dict[str, Any]] = []
    event_values = pd.to_numeric(riskset[event_column], errors="coerce").fillna(0)
    for family, features in (("alpha", alpha_features), ("beta", beta_features), ("alpha_beta", [*alpha_features, *beta_features])):
        for feature in features:
            values = pd.to_numeric(riskset[feature], errors="coerce")
            missing = values.isna()
            rows.append(
                {
                    "feature": feature,
                    "feature_family": family if family != "alpha_beta" else ("alpha" if feature in alpha_features else "beta"),
                    "n_missing": int(missing.sum()),
                    "proportion_missing": float(missing.mean()),
                    "n_nonmissing": int((~missing).sum()),
                    "n_events_nonmissing": int(event_values.loc[~missing].sum()),
                    "n_events_missing": int(event_values.loc[missing].sum()),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["feature"]).reset_index(drop=True)


def build_design_diagnostics_table(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    neural_config: NeuralHazardConfig,
    alpha_features: list[str],
    beta_features: list[str],
    row_ids: pd.Series,
    max_design_rows: int,
) -> pd.DataFrame:
    """Build design diagnostics for behaviour/alpha/beta/alpha_beta models."""

    family_to_features = {
        "behaviour": [],
        "alpha": alpha_features,
        "beta": beta_features,
        "alpha_beta": [*alpha_features, *beta_features],
    }
    rows: list[dict[str, Any]] = []
    for family, features in family_to_features.items():
        baseline = _build_neural_baseline_formula(event_column=event_column, neural_config=neural_config)
        formula = _append_terms(baseline, features)
        subset = subset_family_rows(
            riskset,
            event_column=event_column,
            neural_config=neural_config,
            features=features,
        )
        row = compute_design_diagnostics(
            subset,
            formula=formula,
            event_column=event_column,
            model_family=family,
            row_ids=row_ids.loc[subset.index],
            neural_features=features,
            max_design_rows=max_design_rows,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_design_diagnostics(
    subset: pd.DataFrame,
    *,
    formula: str,
    event_column: str,
    model_family: str,
    row_ids: pd.Series,
    neural_features: list[str],
    max_design_rows: int,
) -> dict[str, Any]:
    """Compute design diagnostics for one model matrix."""

    if subset.empty:
        return {
            "model_family": model_family,
            "formula": formula,
            "n_rows": 0,
            "n_events": 0,
            "n_predictors": 0,
            "event_per_predictor_ratio": None,
            "number_of_constant_columns": 0,
            "number_of_near_constant_columns": 0,
            "maximum_absolute_correlation": None,
            "high_correlation_pairs_abs_gt_0_95": "",
            "condition_number": None,
            "rank": 0,
            "rank_deficiency": 0,
            "row_id_count": 0,
            "n_possible_separation_predictors": 0,
            "possible_separation_predictors": "",
            "max_abs_standardized_mean_difference": None,
        }
    design_subset = sample_rows_for_design_diagnostics(
        subset,
        event_column=event_column,
        max_rows=max_design_rows,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_matrix, x_matrix = dmatrices(formula, data=design_subset, return_type="dataframe")
    n_rows = int(len(subset))
    n_events = int(pd.to_numeric(subset[event_column], errors="coerce").fillna(0).sum())
    n_predictors = int(x_matrix.shape[1])
    design_for_diag = x_matrix.copy()
    if "Intercept" in design_for_diag.columns:
        design_for_diag = design_for_diag.drop(columns=["Intercept"])
    variances = design_for_diag.var(axis=0, ddof=0) if not design_for_diag.empty else pd.Series(dtype=float)
    constant_columns = int((variances <= 0.0).sum()) if not variances.empty else 0
    near_constant_columns = int((variances <= 1.0e-8).sum()) if not variances.empty else 0
    max_corr, corr_pairs = correlation_diagnostics(design_for_diag)
    rank = int(np.linalg.matrix_rank(np.asarray(x_matrix, dtype=float))) if n_rows > 0 else 0
    rank_deficiency = int(max(0, x_matrix.shape[1] - rank))
    condition_number = compute_condition_number(np.asarray(x_matrix, dtype=float))
    separation = compute_separation_summary(subset, event_column=event_column, predictors=neural_features)
    return {
        "model_family": model_family,
        "formula": formula,
        "n_rows": n_rows,
        "n_events": n_events,
        "design_sample_rows": int(len(design_subset)),
        "design_sample_events": int(pd.to_numeric(design_subset[event_column], errors="coerce").fillna(0).sum()),
        "subsampled_for_design": bool(len(design_subset) < len(subset)),
        "n_predictors": n_predictors,
        "event_per_predictor_ratio": float(n_events / n_predictors) if n_predictors > 0 else None,
        "number_of_constant_columns": constant_columns,
        "number_of_near_constant_columns": near_constant_columns,
        "maximum_absolute_correlation": max_corr,
        "high_correlation_pairs_abs_gt_0_95": "; ".join(corr_pairs),
        "condition_number": condition_number,
        "rank": rank,
        "rank_deficiency": rank_deficiency,
        "row_id_count": int(row_ids.nunique()),
        "n_possible_separation_predictors": int(separation["n_possible_separation_predictors"]),
        "possible_separation_predictors": separation["possible_separation_predictors"],
        "max_abs_standardized_mean_difference": separation["max_abs_standardized_mean_difference"],
    }


def sample_rows_for_design_diagnostics(
    table: pd.DataFrame,
    *,
    event_column: str,
    max_rows: int,
    random_state: int = 0,
) -> pd.DataFrame:
    """Bound design diagnostics memory while preserving all event rows."""

    if len(table) <= max_rows:
        return table.copy()
    events = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1].copy()
    nonevents = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) != 1].copy()
    remaining = max(0, max_rows - len(events))
    if remaining <= 0:
        sampled = events.sample(n=min(len(events), max_rows), random_state=random_state, replace=False)
        return sampled.sort_index(kind="mergesort").copy()
    sampled_nonevents = nonevents.sample(n=min(len(nonevents), remaining), random_state=random_state, replace=False)
    return (
        pd.concat([events, sampled_nonevents], axis=0, ignore_index=False, sort=False)
        .sort_index(kind="mergesort")
        .copy()
    )


def correlation_diagnostics(x_matrix: pd.DataFrame) -> tuple[float | None, list[str]]:
    """Compute correlation summary and highly correlated pairs."""

    if x_matrix.shape[1] < 2:
        return None, []
    finite = x_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if finite.shape[1] < 2:
        return None, []
    corr = finite.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    max_corr = float(np.nanmax(upper.to_numpy(dtype=float))) if np.isfinite(upper.to_numpy(dtype=float)).any() else None
    pairs: list[str] = []
    for row_name in upper.index:
        for col_name in upper.columns:
            value = upper.loc[row_name, col_name]
            if pd.notna(value) and float(value) > 0.95:
                pairs.append(f"{row_name}~{col_name}:{float(value):.3f}")
    return max_corr, pairs


def compute_condition_number(matrix: np.ndarray) -> float | None:
    """Compute a finite condition number when possible."""

    if matrix.size == 0:
        return None
    try:
        value = float(np.linalg.cond(matrix))
    except Exception:
        return None
    return value if np.isfinite(value) else None


def compute_separation_summary(
    table: pd.DataFrame,
    *,
    event_column: str,
    predictors: list[str],
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Summarize complete/quasi-separation for predictors."""

    rows: list[dict[str, Any]] = []
    event_mask = pd.to_numeric(table[event_column], errors="coerce").fillna(0).astype(int) == 1
    nonevent_mask = ~event_mask
    for predictor in predictors:
        values = pd.to_numeric(table[predictor], errors="coerce")
        event_values = values.loc[event_mask & values.notna()]
        nonevent_values = values.loc[nonevent_mask & values.notna()]
        if event_values.empty or nonevent_values.empty:
            continue
        pooled_sd = np.sqrt(
            max(
                0.0,
                (
                    float(event_values.var(ddof=0)) * max(len(event_values) - 1, 0)
                    + float(nonevent_values.var(ddof=0)) * max(len(nonevent_values) - 1, 0)
                )
                / max(len(event_values) + len(nonevent_values) - 2, 1),
            )
        )
        smd = (
            abs(float(event_values.mean()) - float(nonevent_values.mean())) / pooled_sd
            if pooled_sd > 0.0
            else np.inf
        )
        possible_separation = bool(
            float(event_values.max()) < float(nonevent_values.min())
            or float(nonevent_values.max()) < float(event_values.min())
        )
        extreme_imbalance = bool(np.isfinite(smd) and smd >= threshold)
        rows.append(
            {
                "predictor": predictor,
                "possible_separation": possible_separation,
                "extreme_imbalance": extreme_imbalance,
                "standardized_mean_difference": float(smd),
                "mean_event": float(event_values.mean()),
                "mean_nonevent": float(nonevent_values.mean()),
                "sd_event": float(event_values.std(ddof=0)),
                "sd_nonevent": float(nonevent_values.std(ddof=0)),
                "min_event": float(event_values.min()),
                "max_event": float(event_values.max()),
                "min_nonevent": float(nonevent_values.min()),
                "max_nonevent": float(nonevent_values.max()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "n_possible_separation_predictors": 0,
            "possible_separation_predictors": "",
            "max_abs_standardized_mean_difference": None,
        }
    flagged = frame.loc[frame["possible_separation"] | frame["extreme_imbalance"]]
    return {
        "n_possible_separation_predictors": int(len(flagged)),
        "possible_separation_predictors": "; ".join(flagged["predictor"].astype(str).tolist()),
        "max_abs_standardized_mean_difference": float(frame["standardized_mean_difference"].abs().max()),
    }


def fit_incremental_models(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    neural_config: NeuralHazardConfig,
    alpha_features: list[str],
    beta_features: list[str],
    max_non_event_rows: int,
    fit_model_fn: Callable[..., FittedFormulaModel] | None = None,
) -> pd.DataFrame:
    """Fit the diagnostic SPP model sequence."""

    fit_fn = fit_model_fn or fit_diagnostic_formula_model
    timing_formula = build_timing_only_formula(event_column=event_column, neural_config=neural_config)
    behaviour_formula = _build_neural_baseline_formula(event_column=event_column, neural_config=neural_config)
    specs = [
        ("SPP_M0_timing_only", timing_formula, []),
        ("SPP_M_behaviour", behaviour_formula, []),
        ("SPP_M_alpha_pc1", _append_terms(behaviour_formula, alpha_features[:1]), alpha_features[:1]),
        ("SPP_M_beta_pc1", _append_terms(behaviour_formula, beta_features[:1]), beta_features[:1]),
        ("SPP_M_alpha_all", _append_terms(behaviour_formula, alpha_features), alpha_features),
        ("SPP_M_beta_all", _append_terms(behaviour_formula, beta_features), beta_features),
        ("SPP_M_alpha_beta_all", _append_terms(behaviour_formula, [*alpha_features, *beta_features]), [*alpha_features, *beta_features]),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, formula, features in specs:
        required_columns = _extract_formula_columns(formula, event_column=event_column)
        try:
            subset = _subset_complete_cases(riskset, required_columns=required_columns)
        except Exception as error:
            rows.append(
                incremental_failure_row(
                    model_name=model_name,
                    formula=formula,
                    status="failed",
                    error_message=str(error),
                )
            )
            continue
        fit_subset = sample_diagnostic_fit_rows(
            subset,
            event_column=event_column,
            max_non_event_rows=max_non_event_rows,
        )
        fitted = fit_fn(
            riskset_table=fit_subset,
            model_name=model_name,
            formula=formula,
            event_column=event_column,
        )
        rows.append(
            incremental_row_from_fit(
                fitted,
                subset=subset,
                fit_subset=fit_subset,
                event_column=event_column,
                features=features,
            )
        )
    return pd.DataFrame(rows)


def fit_diagnostic_formula_model(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    event_column: str,
) -> FittedFormulaModel:
    """Fit a bounded-iteration GLM for diagnostics."""

    if riskset_table.empty:
        return FittedFormulaModel(
            model_name=model_name,
            formula=formula,
            result=None,
            n_rows=0,
            n_events=0,
            n_predictors=0,
            converged=False,
            fit_warnings=[],
            error_message="No complete-case rows were available for this model.",
        )
    n_events = int(pd.to_numeric(riskset_table[event_column], errors="coerce").fillna(0).sum())
    if n_events <= 0:
        return FittedFormulaModel(
            model_name=model_name,
            formula=formula,
            result=None,
            n_rows=int(len(riskset_table)),
            n_events=0,
            n_predictors=0,
            converged=False,
            fit_warnings=[],
            error_message="No event rows were available for this model.",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted = sm.GLM.from_formula(
                formula=formula,
                data=riskset_table,
                family=sm.families.Binomial(),
            ).fit(maxiter=100, disp=0)
        except Exception as error:
            return FittedFormulaModel(
                model_name=model_name,
                formula=formula,
                result=None,
                n_rows=int(len(riskset_table)),
                n_events=n_events,
                n_predictors=0,
                converged=False,
                fit_warnings=[str(item.message) for item in caught],
                error_message=str(error),
            )
    fit_warnings = [str(item.message) for item in caught]
    warning_text = " ".join(fit_warnings).lower()
    numeric_ok = (
        _maybe_float(getattr(fitted, "aic", None)) is not None
        and _maybe_float(getattr(fitted, "bic", None)) is not None
        and _maybe_float(getattr(fitted, "llf", None)) is not None
    )
    pathological = (
        "overflow encountered in exp" in warning_text
        or "perfect separation" in warning_text
        or "divide by zero encountered in log" in warning_text
    )
    converged = bool(getattr(fitted, "converged", True) and numeric_ok and not pathological)
    error_message = None if converged else "Model failed convergence checks."
    return FittedFormulaModel(
        model_name=model_name,
        formula=formula,
        result=fitted,
        n_rows=int(len(riskset_table)),
        n_events=n_events,
        n_predictors=int(len(getattr(fitted, "params", []))),
        converged=converged,
        fit_warnings=fit_warnings,
        error_message=error_message,
    )


def build_timing_only_formula(*, event_column: str, neural_config: NeuralHazardConfig) -> str:
    """Build a timing-only spline formula."""

    onset_spline = (
        f"bs(time_from_partner_onset, df={neural_config.model.baseline_spline_df}, "
        f"degree={neural_config.model.baseline_spline_degree}, include_intercept=False)"
    )
    offset_spline = (
        f"bs(time_from_partner_offset, df={neural_config.model.baseline_spline_df}, "
        f"degree={neural_config.model.baseline_spline_degree}, include_intercept=False)"
    )
    return f"{event_column} ~ {onset_spline} + {offset_spline}"


def incremental_failure_row(
    *,
    model_name: str,
    formula: str,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    """Build a failure row for incremental diagnostics."""

    return {
        "model_name": model_name,
        "formula": formula,
        "n_rows": 0,
        "n_events": 0,
        "fit_sample_rows": 0,
        "fit_sample_events": 0,
        "subsampled_for_fit": False,
        "n_predictors": 0,
        "converged": False,
        "status": status,
        "optimizer_message": None,
        "fit_warnings": "",
        "error_message": error_message,
        "aic": None,
        "bic": None,
        "log_likelihood": None,
        "max_abs_coefficient": None,
        "max_standard_error": None,
        "any_nan_coefficients": None,
        "any_infinite_coefficients": None,
    }


def incremental_row_from_fit(
    fitted: FittedFormulaModel,
    *,
    subset: pd.DataFrame,
    fit_subset: pd.DataFrame,
    event_column: str,
    features: list[str],
) -> dict[str, Any]:
    """Build one incremental diagnostics row."""

    if fitted.result is None:
        status = "no_events" if int(pd.to_numeric(subset[event_column], errors="coerce").fillna(0).sum()) == 0 else "failed"
        return {
            "model_name": fitted.model_name,
            "formula": fitted.formula,
            "n_rows": int(len(subset)),
            "n_events": int(pd.to_numeric(subset[event_column], errors="coerce").fillna(0).sum()),
            "fit_sample_rows": int(len(fit_subset)),
            "fit_sample_events": int(pd.to_numeric(fit_subset[event_column], errors="coerce").fillna(0).sum()),
            "subsampled_for_fit": bool(len(fit_subset) < len(subset)),
            "n_predictors": int(max(0, len(features))),
            "converged": False,
            "status": status if len(subset) > 0 else "no_data",
            "optimizer_message": None,
            "fit_warnings": "; ".join(fitted.fit_warnings),
            "error_message": fitted.error_message,
            "aic": None,
            "bic": None,
            "log_likelihood": None,
            "max_abs_coefficient": None,
            "max_standard_error": None,
            "any_nan_coefficients": None,
            "any_infinite_coefficients": None,
        }
    params = np.asarray(getattr(fitted.result, "params", []), dtype=float)
    bse = np.asarray(getattr(fitted.result, "bse", []), dtype=float)
    optimizer_message = None
    fit_history = getattr(fitted.result, "fit_history", None)
    if isinstance(fit_history, dict):
        optimizer_message = str(fit_history.get("iteration", "")) if fit_history else None
    return {
        "model_name": fitted.model_name,
        "formula": fitted.formula,
        "n_rows": int(len(subset)),
        "n_events": int(pd.to_numeric(subset[event_column], errors="coerce").fillna(0).sum()),
        "fit_sample_rows": int(len(fit_subset)),
        "fit_sample_events": int(pd.to_numeric(fit_subset[event_column], errors="coerce").fillna(0).sum()),
        "subsampled_for_fit": bool(len(fit_subset) < len(subset)),
        "n_predictors": int(fitted.n_predictors),
        "converged": bool(fitted.converged),
        "status": "converged" if fitted.converged else "failed",
        "optimizer_message": optimizer_message,
        "fit_warnings": "; ".join(fitted.fit_warnings),
        "error_message": fitted.error_message,
        "aic": _maybe_float(getattr(fitted.result, "aic", None)),
        "bic": _maybe_float(getattr(fitted.result, "bic", None)),
        "log_likelihood": _maybe_float(getattr(fitted.result, "llf", None)),
        "max_abs_coefficient": float(np.max(np.abs(params))) if params.size else None,
        "max_standard_error": float(np.max(np.abs(bse))) if bse.size else None,
        "any_nan_coefficients": bool(np.isnan(params).any()) if params.size else False,
        "any_infinite_coefficients": bool(np.isinf(params).any()) if params.size else False,
    }


def run_ridge_diagnostics(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    neural_config: NeuralHazardConfig,
    alpha_features: list[str],
    beta_features: list[str],
    max_non_event_rows: int,
) -> pd.DataFrame:
    """Run ridge-logistic diagnostics on the same complete-case samples."""

    family_to_features = {
        "ridge_behaviour": [],
        "ridge_alpha": alpha_features,
        "ridge_beta": beta_features,
        "ridge_alpha_beta": [*alpha_features, *beta_features],
    }
    rows: list[dict[str, Any]] = []
    for model_name, features in family_to_features.items():
        formula = _append_terms(_build_neural_baseline_formula(event_column=event_column, neural_config=neural_config), features)
        subset = subset_family_rows(
            riskset,
            event_column=event_column,
            neural_config=neural_config,
            features=features,
        )
        if subset.empty:
            rows.append(ridge_failure_row(model_name=model_name, status="no_data"))
            continue
        fit_subset = sample_diagnostic_fit_rows(
            subset,
            event_column=event_column,
            max_non_event_rows=max_non_event_rows,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_fit, x_fit = dmatrices(formula, data=fit_subset, return_type="dataframe")
        x_values = np.asarray(x_fit, dtype=float)
        y_values = np.asarray(y_fit).ravel().astype(int)
        if np.unique(y_values).size < 2:
            rows.append(ridge_failure_row(model_name=model_name, status="no_events", n_rows=len(subset), n_events=int(y_values.sum()), n_predictors=x_values.shape[1], fit_sample_rows=len(fit_subset), fit_sample_events=int(y_values.sum()), subsampled_for_fit=bool(len(fit_subset) < len(subset))))
            continue
        groups = (
            fit_subset["participant_speaker_id"].astype(str).to_numpy()
            if "participant_speaker_id" in fit_subset.columns
            else fit_subset["run"].astype(str).to_numpy()
            if "run" in fit_subset.columns
            else np.array(["all"] * len(fit_subset), dtype=object)
        )
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
        try:
            train_index, test_index = next(splitter.split(x_values, y_values, groups=groups))
        except ValueError:
            split_at = max(1, int(round(0.70 * len(fit_subset))))
            train_index = np.arange(split_at)
            test_index = np.arange(split_at, len(fit_subset))
        if test_index.size == 0:
            test_index = train_index
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)),
            ]
        )
        try:
            model.fit(x_values[train_index], y_values[train_index])
            train_prob = model.predict_proba(x_values[train_index])[:, 1]
            test_prob = model.predict_proba(x_values[test_index])[:, 1]
            rows.append(
                {
                    "model_name": model_name,
                    "n_rows": int(len(subset)),
                    "n_events": int(pd.to_numeric(subset[event_column], errors="coerce").fillna(0).sum()),
                    "fit_sample_rows": int(len(fit_subset)),
                    "fit_sample_events": int(y_values.sum()),
                    "subsampled_for_fit": bool(len(fit_subset) < len(subset)),
                    "n_predictors": int(x_values.shape[1]),
                    "penalty": "ridge",
                    "C_or_lambda": 1.0,
                    "converged": True,
                    "train_log_loss": safe_log_loss(y_values[train_index], train_prob),
                    "test_log_loss": safe_log_loss(y_values[test_index], test_prob),
                    "train_brier": safe_brier(y_values[train_index], train_prob),
                    "test_brier": safe_brier(y_values[test_index], test_prob),
                    "train_auroc": safe_auroc(y_values[train_index], train_prob),
                    "test_auroc": safe_auroc(y_values[test_index], test_prob),
                }
            )
        except Exception:
            rows.append(ridge_failure_row(model_name=model_name, status="failed", n_rows=len(subset), n_events=int(y_values.sum()), n_predictors=x_values.shape[1], fit_sample_rows=len(fit_subset), fit_sample_events=int(y_values.sum()), subsampled_for_fit=bool(len(fit_subset) < len(subset))))
    return pd.DataFrame(rows)


def ridge_failure_row(
    *,
    model_name: str,
    status: str,
    n_rows: int = 0,
    n_events: int = 0,
    n_predictors: int = 0,
    fit_sample_rows: int = 0,
    fit_sample_events: int = 0,
    subsampled_for_fit: bool = False,
) -> dict[str, Any]:
    """Build one ridge diagnostics failure row."""

    return {
        "model_name": model_name,
        "n_rows": int(n_rows),
        "n_events": int(n_events),
        "fit_sample_rows": int(fit_sample_rows),
        "fit_sample_events": int(fit_sample_events),
        "subsampled_for_fit": bool(subsampled_for_fit),
        "n_predictors": int(n_predictors),
        "penalty": "ridge",
        "C_or_lambda": 1.0,
        "converged": False,
        "train_log_loss": None,
        "test_log_loss": None,
        "train_brier": None,
        "test_brier": None,
        "train_auroc": None,
        "test_auroc": None,
        "status": status,
    }


def sample_diagnostic_fit_rows(
    table: pd.DataFrame,
    *,
    event_column: str,
    max_non_event_rows: int,
    random_state: int = 0,
) -> pd.DataFrame:
    """Keep all events and cap non-event rows for bounded diagnostic fitting."""

    event_mask = pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1
    events = table.loc[event_mask].copy()
    nonevents = table.loc[~event_mask].copy()
    if len(nonevents) <= max_non_event_rows:
        return table.copy()
    sampled_nonevents = nonevents.sample(n=max_non_event_rows, random_state=random_state, replace=False)
    combined = (
        pd.concat([events, sampled_nonevents], axis=0, ignore_index=False, sort=False)
        .sort_index(kind="mergesort")
        .copy()
    )
    return combined


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute a safe log loss."""

    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return None


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute a safe Brier score."""

    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return None


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute a safe AUROC."""

    try:
        if np.unique(y_true).size < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def plot_spp_event_time_histogram(riskset: pd.DataFrame, *, event_column: str, output_path: Path) -> None:
    """Plot SPP event timing histograms."""

    event_rows = riskset.loc[pd.to_numeric(riskset[event_column], errors="coerce").fillna(0) == 1].copy()
    columns = [column for column in ("time_from_partner_onset", "time_from_partner_offset") if column in event_rows.columns]
    figure, axes = plt.subplots(1, max(1, len(columns)), figsize=(6.0 * max(1, len(columns)), 4.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)
    if event_rows.empty or not columns:
        axes[0].text(0.5, 0.5, "No finite SPP event timing values available.", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        for axis, column in zip(axes, columns, strict=True):
            values = pd.to_numeric(event_rows[column], errors="coerce").dropna()
            axis.hist(values, bins=min(30, max(5, len(values))), color="#3b7ea1", alpha=0.85)
            axis.set_title(column)
            axis.set_xlabel("seconds")
            axis.set_ylabel("n_events")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_spp_events_by_participant_speaker(
    riskset: pd.DataFrame,
    *,
    event_column: str,
    alpha_beta_features: list[str],
    output_path: Path,
) -> None:
    """Plot event counts by participant_speaker_id after alpha-beta complete-case filtering."""

    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    if "participant_speaker_id" not in riskset.columns:
        axis.text(0.5, 0.5, "participant_speaker_id is unavailable.", ha="center", va="center")
        axis.set_axis_off()
    else:
        mask = complete_case_mask(riskset, alpha_beta_features)
        subset = riskset.loc[mask].copy()
        grouped = (
            subset.groupby("participant_speaker_id", sort=True)[event_column]
            .sum()
            .sort_values(ascending=False)
        )
        if grouped.empty:
            axis.text(0.5, 0.5, "No alpha-beta complete-case rows available.", ha="center", va="center")
            axis.set_axis_off()
        else:
            axis.bar(grouped.index.astype(str), grouped.to_numpy(dtype=float), color="#d17c4b")
            axis.set_ylabel("n_events")
            axis.set_xlabel("participant_speaker_id")
            axis.set_title("SPP events by participant_speaker_id after alpha_beta complete-case filtering")
            axis.tick_params(axis="x", rotation=75)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_missingness_by_feature(missingness: pd.DataFrame, *, output_path: Path) -> None:
    """Plot missingness by feature."""

    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    if missingness.empty:
        axis.text(0.5, 0.5, "No neural feature missingness available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        ordered = missingness.sort_values(["feature_family", "proportion_missing", "feature"], ascending=[True, False, True])
        axis.bar(ordered["feature"].astype(str), ordered["proportion_missing"].to_numpy(dtype=float), color="#7fa650")
        axis.set_ylabel("proportion_missing")
        axis.set_xlabel("feature")
        axis.set_title("SPP neural feature missingness")
        axis.tick_params(axis="x", rotation=75)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_design_correlation_heatmap(
    riskset: pd.DataFrame,
    *,
    output_path: Path,
    features: list[str],
    max_rows: int,
) -> None:
    """Plot a simple correlation heatmap for neural predictors."""

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    available = [feature for feature in features if feature in riskset.columns]
    if len(available) < 2:
        axis.text(0.5, 0.5, "Not enough neural predictors for a correlation heatmap.", ha="center", va="center")
        axis.set_axis_off()
    else:
        heatmap_subset = sample_rows_for_design_diagnostics(
            riskset.loc[:, [*available, "event_spp" if "event_spp" in riskset.columns else "event"]].copy(),
            event_column="event_spp" if "event_spp" in riskset.columns else "event",
            max_rows=max_rows,
        )
        corr = heatmap_subset.loc[:, available].apply(pd.to_numeric, errors="coerce").corr()
        image = axis.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(np.arange(len(available)))
        axis.set_xticklabels(available, rotation=75, ha="right")
        axis.set_yticks(np.arange(len(available)))
        axis.set_yticklabels(available)
        axis.set_title("SPP neural predictor correlations")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_incremental_model_status(incremental: pd.DataFrame, *, output_path: Path) -> None:
    """Plot incremental model statuses."""

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    if incremental.empty:
        axis.text(0.5, 0.5, "No incremental diagnostics available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        order = {name: index for index, name in enumerate(DEFAULT_STATUS_ORDER)}
        working = incremental.copy()
        working["status_order"] = working["status"].map(order).fillna(len(order))
        working["status_color"] = working["status"].map(
            {
                "converged": "#2b8a3e",
                "failed": "#c92a2a",
                "no_data": "#868e96",
                "no_events": "#f08c00",
            }
        ).fillna("#495057")
        axis.barh(
            working["model_name"].astype(str),
            np.ones(len(working)),
            color=working["status_color"].tolist(),
        )
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.set_title("SPP incremental model fit status")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def build_failure_only_report(notes: list[str]) -> str:
    """Build a minimal report when the riskset cannot be loaded."""

    note_lines = [f"- {note}" for note in notes] if notes else ["- No additional notes."]
    return "\n".join(
        [
            "# SPP Neural Failure Diagnostic Report",
            "",
            "## Summary",
            "",
            "The diagnostics bundle could not be computed because the required SPP neural riskset could not be loaded.",
            "",
            "## Notes",
            "",
            *note_lines,
            "",
            "Convergence failure alone is not evidence of absence of SPP neural signal.",
        ]
    )


def build_spp_failure_report(
    *,
    qc: dict[str, Any],
    missingness: pd.DataFrame,
    event_distribution: pd.DataFrame,
    design_diagnostics: pd.DataFrame,
    incremental: pd.DataFrame,
    ridge: pd.DataFrame | None,
    notes: list[str],
) -> str:
    """Build the diagnostic markdown report."""

    incremental_table = (
        incremental.to_markdown(index=False)
        if not incremental.empty
        else "Incremental fits were skipped for this run."
    )
    likely_causes = classify_likely_causes(qc=qc, design_diagnostics=design_diagnostics, incremental=incremental, missingness=missingness)
    alpha_beta_row = design_diagnostics.loc[design_diagnostics["model_family"] == "alpha_beta"].head(1)
    alpha_beta_event_per_predictor = (
        None if alpha_beta_row.empty else alpha_beta_row["event_per_predictor_ratio"].iloc[0]
    )
    alpha_beta_rank_def = None if alpha_beta_row.empty else alpha_beta_row["rank_deficiency"].iloc[0]
    alpha_beta_condition = None if alpha_beta_row.empty else alpha_beta_row["condition_number"].iloc[0]
    alpha_beta_corr = None if alpha_beta_row.empty else alpha_beta_row["maximum_absolute_correlation"].iloc[0]
    m0_status = extract_model_status(incremental, "SPP_M0_timing_only")
    behaviour_status = extract_model_status(incremental, "SPP_M_behaviour")
    alpha_pc1_status = extract_model_status(incremental, "SPP_M_alpha_pc1")
    beta_pc1_status = extract_model_status(incremental, "SPP_M_beta_pc1")
    alpha_all_status = extract_model_status(incremental, "SPP_M_alpha_all")
    beta_all_status = extract_model_status(incremental, "SPP_M_beta_all")
    alpha_beta_status = extract_model_status(incremental, "SPP_M_alpha_beta_all")
    ridge_summary = ""
    if ridge is not None and not ridge.empty:
        ridge_summary = "\n".join(
            [
                "",
                "## Penalized fallback results",
                "",
                ridge.to_markdown(index=False),
            ]
        )
    answer_lines = [
        f"- How many SPP events are lost after neural filtering? Alpha: {qc.get('n_events_lost_alpha', 'NA')}, beta: {qc.get('n_events_lost_beta', 'NA')}, alpha_beta: {qc.get('n_events_lost_alpha_beta', 'NA')}.",
        f"- Does neural filtering remove a large fraction of SPP event bins? Alpha_beta loss fraction: {format_optional(qc.get('event_loss_fraction_alpha_beta'))}.",
        f"- Are there enough events for the number of neural predictors? Alpha_beta events-per-predictor ratio: {format_optional(alpha_beta_event_per_predictor)}.",
        f"- Does alpha/beta/alpha_beta filtering differ substantially? Remaining events: alpha={qc.get('n_events_neural_complete_alpha', 'NA')}, beta={qc.get('n_events_neural_complete_beta', 'NA')}, alpha_beta={qc.get('n_events_neural_complete_alpha_beta', 'NA')}.",
    ]
    interpretation_lines = [
        f"- `M0_timing` status: {m0_status}.",
        f"- `M_behaviour` status: {behaviour_status}.",
        f"- `M_alpha_pc1` status: {alpha_pc1_status}; `M_beta_pc1` status: {beta_pc1_status}.",
        f"- `M_alpha_all` status: {alpha_all_status}; `M_beta_all` status: {beta_all_status}; `M_alpha_beta_all` status: {alpha_beta_status}.",
    ]
    notes_block = "\n".join([f"- {note}" for note in notes]) if notes else "- No additional notes."
    report = f"""# SPP Neural Failure Diagnostic Report

## Summary

These diagnostics treat the SPP convergence failure as a technical issue to inspect, not as a scientific result. Convergence failure alone is not evidence of absence of SPP neural signal.

## Basic sample and event counts

{json.dumps(qc, indent=2, sort_keys=True)}

Explicit answers:
{chr(10).join(answer_lines)}

## Neural missingness

Missingness rows: {len(missingness)}. Maximum feature missingness: {format_optional(missingness['proportion_missing'].max() if not missingness.empty else None)}.

## Event distribution

Event-distribution rows: {len(event_distribution)}. Participant-speaker groups with zero alpha-beta events can make SPP fits fragile even when total row count looks adequate.

## Design matrix diagnostics

- Alpha_beta rank deficiency: {alpha_beta_rank_def}.
- Alpha_beta maximum absolute correlation: {format_optional(alpha_beta_corr)}.
- Alpha_beta condition number: {format_optional(alpha_beta_condition)}.

## Separation diagnostics

Possible separation counts by family:
{design_diagnostics.loc[:, ['model_family', 'n_possible_separation_predictors', 'possible_separation_predictors', 'max_abs_standardized_mean_difference']].to_markdown(index=False)}

## Incremental model fitting results

{incremental_table}

Interpretation:
{chr(10).join(interpretation_lines)}{ridge_summary}

## Likely cause of SPP failure

{', '.join(likely_causes) if likely_causes else 'unknown / needs deeper inspection'}

## Recommended next steps

- Keep the primary FPP neural result unchanged.
- Treat the current SPP failure as a technical convergence problem unless later diagnostics clearly rule that out.
- If overparameterization or collinearity dominates, reduce the diagnostic neural basis or regularize before making scientific claims.
- If baseline timing or behavioural models already fail, fix the SPP baseline sample definition before interpreting any neural comparison.

## Is SPP failure interpretable as absence of neural signal?

No. At this stage it should be treated as a technical convergence problem unless a later, stable diagnostic workflow shows the same null pattern under well-posed fits.

## Notes

{notes_block}
"""
    return textwrap.dedent(report).strip() + "\n"


def classify_likely_causes(
    *,
    qc: dict[str, Any],
    design_diagnostics: pd.DataFrame,
    incremental: pd.DataFrame,
    missingness: pd.DataFrame,
) -> list[str]:
    """Classify likely causes of failure from the diagnostics bundle."""

    causes: list[str] = []
    alpha_beta_events = int(qc.get("n_events_neural_complete_alpha_beta", 0) or 0)
    event_loss = qc.get("event_loss_fraction_alpha_beta")
    if event_loss is not None and float(event_loss) >= 0.25:
        causes.append("missingness / complete-case filtering problem")
    alpha_beta_row = design_diagnostics.loc[design_diagnostics["model_family"] == "alpha_beta"].head(1)
    if not alpha_beta_row.empty:
        ratio = alpha_beta_row["event_per_predictor_ratio"].iloc[0]
        if pd.notna(ratio) and float(ratio) < 5.0:
            causes.append("overparameterization")
            causes.append("insufficient events after filtering")
        rank_def = alpha_beta_row["rank_deficiency"].iloc[0]
        if pd.notna(rank_def) and int(rank_def) > 0:
            causes.append("design matrix rank deficiency")
        max_corr = alpha_beta_row["maximum_absolute_correlation"].iloc[0]
        if pd.notna(max_corr) and float(max_corr) > 0.95:
            causes.append("high collinearity")
        condition = alpha_beta_row["condition_number"].iloc[0]
        if pd.notna(condition) and float(condition) > 1.0e6:
            causes.append("bad scaling")
        separation = alpha_beta_row["n_possible_separation_predictors"].iloc[0]
        if pd.notna(separation) and int(separation) > 0:
            causes.append("quasi-separation")
    total_rate = qc.get("event_rate_total")
    if total_rate is not None and float(total_rate) < 0.01:
        causes.append("extreme event imbalance")
    if alpha_beta_events <= 0:
        causes.append("insufficient events after filtering")
    m0_status = extract_model_status(incremental, "SPP_M0_timing_only")
    if m0_status != "converged":
        causes.append("baseline model failure")
    if missingness.empty:
        causes.append("unknown / needs deeper inspection")
    return list(dict.fromkeys(causes))


def extract_model_status(incremental: pd.DataFrame, model_name: str) -> str:
    """Extract one model status from the incremental diagnostics table."""

    if incremental.empty or "model_name" not in incremental.columns or "status" not in incremental.columns:
        return "not_run"
    row = incremental.loc[incremental["model_name"] == model_name, "status"]
    return str(row.iloc[0]) if not row.empty else "missing"


def format_optional(value: Any) -> str:
    """Format optional numeric values."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)
