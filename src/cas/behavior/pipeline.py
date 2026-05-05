"""Behavioral hazard pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.behavior._r_backend import BACKEND_ID as R_BACKEND_ID, BACKEND_NOTE as R_BACKEND_NOTE, run_r_lag_selection, run_r_model_bundle
from cas.behavior.comparisons import MODEL_COMPARISONS
from cas.behavior.config import BehaviorHazardConfig, load_behavior_hazard_config
from cas.behavior.diagnostics import bins_by_subject, collinearity_summary, event_rate_summary
from cas.behavior.formulas import formula_metadata, render_fixed_formula, render_formula
from cas.behavior.models import RIDGE_BACKEND_ID, STANDARD_BACKEND_ID, odds_ratio_rows
from cas.behavior.predictors import standardize_predictors
from cas.behavior.risksets import build_anchor_riskset
from cas.behavior.summaries import ensure_behavior_directories
from cas.viz.behavior.lag_selection import plot_lag_selection
from cas.viz.behavior.lag_sensitivity import plot_lag_sensitivity
from cas.viz.behavior.primary_effects import plot_primary_effects
from cas.viz.behavior.qc import plot_qc_bars
from cas.viz.behavior.timing_heatmaps import plot_timing_information_interaction


LOGGER = logging.getLogger(__name__)
OVERLAP_FILTER_DEFINITION = "event_latency_from_partner_offset_s < 0"
OVERLAP_FILTER_SOURCE_COLUMNS = (
    "event_latency_from_partner_offset_s",
    "latency_from_partner_offset_s",
)


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", force=True)
    logging.getLogger("cas.behavior").setLevel(level)


def _log(verbose: bool, message: str, *args: object) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_parquet(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _analysis_metadata(config: BehaviorHazardConfig, *, overlap_filter_column: str | None = None) -> dict[str, object]:
    return {
        "only_overlap": bool(config.only_overlap),
        "overlap_filter_column": str(overlap_filter_column or ("event_latency_from_partner_offset_s" if config.only_overlap else "")),
        "overlap_filter_definition": OVERLAP_FILTER_DEFINITION if config.only_overlap else "",
    }


def _append_analysis_metadata(table: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    out = table.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def _event_count(table: pd.DataFrame) -> int:
    if "event" in table.columns:
        return int(pd.to_numeric(table["event"], errors="coerce").fillna(0).sum())
    if "event_bin" in table.columns:
        return int(pd.to_numeric(table["event_bin"], errors="coerce").fillna(0).sum())
    if "episode_id" in table.columns:
        return int(pd.Series(table["episode_id"]).dropna().nunique())
    return 0


def _event_rate(table: pd.DataFrame) -> float:
    if len(table) == 0:
        return float("nan")
    return float(pd.to_numeric(table.get("event", 0), errors="coerce").fillna(0).mean())


def _derive_overlap_latency_column(table: pd.DataFrame) -> tuple[pd.Series, str]:
    searched: list[str] = []
    for column in OVERLAP_FILTER_SOURCE_COLUMNS:
        searched.append(column)
        if column in table.columns:
            return pd.to_numeric(table[column], errors="coerce"), column
    if {"own_fpp_onset", "partner_ipu_offset"} <= set(table.columns):
        return (
            pd.to_numeric(table["own_fpp_onset"], errors="coerce")
            - pd.to_numeric(table["partner_ipu_offset"], errors="coerce"),
            "own_fpp_onset - partner_ipu_offset",
        )
    if {"participant_response_onset_s", "partner_offset_s"} <= set(table.columns):
        return (
            pd.to_numeric(table["participant_response_onset_s"], errors="coerce")
            - pd.to_numeric(table["partner_offset_s"], errors="coerce"),
            "participant_response_onset_s - partner_offset_s",
        )
    searched.extend(
        [
            "own_fpp_onset - partner_ipu_offset",
            "participant_response_onset_s - partner_offset_s",
        ]
    )
    raise ValueError(
        "Unable to apply behavior.only_overlap=true because no overlap-latency column was found. "
        f"Searched: {', '.join(searched)}"
    )


def _overlap_counts_row(
    *,
    anchor_type: str,
    n_rows_before: int,
    n_rows_after: int,
    n_events_before: int,
    n_events_after: int,
    event_rate_before: float,
    event_rate_after: float,
) -> dict[str, object]:
    return {
        "anchor_type": str(anchor_type),
        "n_rows_before": int(n_rows_before),
        "n_rows_after": int(n_rows_after),
        "n_events_before": int(n_events_before),
        "n_events_after": int(n_events_after),
        "event_rate_before": float(event_rate_before) if np.isfinite(event_rate_before) else np.nan,
        "event_rate_after": float(event_rate_after) if np.isfinite(event_rate_after) else np.nan,
    }


def _apply_overlap_filter(
    table: pd.DataFrame,
    *,
    config: BehaviorHazardConfig,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    latency, overlap_filter_column = _derive_overlap_latency_column(table)
    metadata = _analysis_metadata(config, overlap_filter_column=overlap_filter_column)
    base_table = _append_analysis_metadata(table, metadata)
    if not config.only_overlap:
        counts = {
            **metadata,
            "n_rows_before_overlap_filter": int(len(base_table)),
            "n_rows_after_overlap_filter": int(len(base_table)),
            "n_events_before_overlap_filter": _event_count(base_table),
            "n_events_after_overlap_filter": _event_count(base_table),
            "event_rate_before_overlap_filter": _event_rate(base_table),
            "event_rate_after_overlap_filter": _event_rate(base_table),
        }
        by_anchor = pd.DataFrame(
            [
                _overlap_counts_row(
                    anchor_type=str(anchor_type),
                    n_rows_before=len(anchor_rows),
                    n_rows_after=len(anchor_rows),
                    n_events_before=_event_count(anchor_rows),
                    n_events_after=_event_count(anchor_rows),
                    event_rate_before=_event_rate(anchor_rows),
                    event_rate_after=_event_rate(anchor_rows),
                )
                for anchor_type, anchor_rows in base_table.groupby("anchor_type", sort=False)
            ]
        )
        by_anchor = _append_analysis_metadata(by_anchor, metadata)
        return base_table, counts, by_anchor

    if "episode_id" not in table.columns:
        raise ValueError(
            "Unable to apply behavior.only_overlap=true because the riskset is missing `episode_id`, "
            "so full event-level riskset fragments cannot be preserved."
        )

    working = table.copy()
    working["_overlap_latency"] = latency
    episode_summary = (
        working.groupby("episode_id", sort=False)
        .agg(
            anchor_type=("anchor_type", "first"),
            overlap_latency=("_overlap_latency", "first"),
            latency_non_null=("_overlap_latency", lambda values: int(pd.Series(values).notna().sum())),
        )
        .reset_index()
    )
    keep_episode_ids = episode_summary.loc[
        pd.to_numeric(episode_summary["overlap_latency"], errors="coerce") < 0.0,
        "episode_id",
    ]
    filtered = working.loc[working["episode_id"].isin(set(keep_episode_ids))].drop(columns="_overlap_latency").copy()
    filtered = _append_analysis_metadata(filtered, metadata)

    counts = {
        **metadata,
        "n_rows_before_overlap_filter": int(len(base_table)),
        "n_rows_after_overlap_filter": int(len(filtered)),
        "n_events_before_overlap_filter": _event_count(base_table),
        "n_events_after_overlap_filter": _event_count(filtered),
        "event_rate_before_overlap_filter": _event_rate(base_table),
        "event_rate_after_overlap_filter": _event_rate(filtered),
    }
    if counts["n_rows_after_overlap_filter"] > counts["n_rows_before_overlap_filter"]:
        raise AssertionError("Overlap-only filtering increased the row count unexpectedly.")
    if counts["n_events_after_overlap_filter"] > counts["n_events_before_overlap_filter"]:
        raise AssertionError("Overlap-only filtering increased the event count unexpectedly.")
    if filtered.empty or counts["n_events_after_overlap_filter"] <= 0:
        raise ValueError(
            "behavior.only_overlap=true produced an empty overlap-only subset or zero remaining events. "
            f"Column used: {overlap_filter_column}"
        )

    by_anchor_rows: list[dict[str, object]] = []
    anchor_levels = list(dict.fromkeys([str(value) for value in base_table["anchor_type"].dropna().tolist()]))
    for anchor_type in anchor_levels:
        before_rows = base_table.loc[base_table["anchor_type"].astype(str) == anchor_type].copy()
        after_rows = filtered.loc[filtered["anchor_type"].astype(str) == anchor_type].copy()
        by_anchor_rows.append(
            _overlap_counts_row(
                anchor_type=anchor_type,
                n_rows_before=len(before_rows),
                n_rows_after=len(after_rows),
                n_events_before=_event_count(before_rows),
                n_events_after=_event_count(after_rows),
                event_rate_before=_event_rate(before_rows),
                event_rate_after=_event_rate(after_rows),
            )
        )
    by_anchor = _append_analysis_metadata(pd.DataFrame(by_anchor_rows), metadata)
    return filtered, counts, by_anchor


def _empty_csv(path: Path, columns: list[str], *, note: str) -> Path:
    frame = pd.DataFrame(columns=columns)
    frame.loc[0, columns[0]] = note
    return _write_csv(frame, path)


def _write_placeholder_figure(path: Path, *, title: str, message: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _riskset_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "fpp": paths["risksets"] / "fpp.parquet",
        "spp": paths["risksets"] / "spp_control.parquet",
        "pooled": paths["risksets"] / "pooled_fpp_spp.parquet",
    }


def _predictor_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "fpp": paths["predictors"] / "fpp_with_lags.parquet",
        "spp": paths["predictors"] / "spp_control_with_lags.parquet",
        "pooled": paths["predictors"] / "pooled_with_lags.parquet",
        "scaling": paths["predictors"] / "standardization_summary.csv",
    }


def _overlap_report_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "summary": paths["diagnostics"] / "overlap_filter_summary.json",
        "by_anchor": paths["tables"] / "overlap_filter_by_anchor.csv",
    }


def _load_predictor_tables(paths: dict[str, Path], config_path: str | Path, *, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    return (
        pd.read_parquet(predictor_paths["fpp"]),
        pd.read_parquet(predictor_paths["spp"]),
        pd.read_parquet(predictor_paths["pooled"]),
    )


def _load_selected_lag_payload(paths: dict[str, Path], config_path: str | Path, *, verbose: bool = False) -> dict[str, object]:
    lag_path = paths["lag_selection"] / "selected_lag.json"
    if not lag_path.exists():
        select_lag(config_path, verbose=verbose)
    return json.loads(lag_path.read_text(encoding="utf-8"))


def _backend_metadata(config: BehaviorHazardConfig) -> dict[str, object]:
    if config.model_backend == "glmm":
        return {
            "model_backend": "glmm",
            "backend_runtime": R_BACKEND_ID,
            "covariance_type": "mixed_model",
            "cluster_variable": None,
            "backend_notes": [R_BACKEND_NOTE],
            "random_effects": ["(1 | dyad_id)", "(1 | subject)"],
        }
    return {
        "model_backend": "glm",
        "backend_runtime": STANDARD_BACKEND_ID,
        "covariance_type": "model_based",
        "cluster_variable": None,
        "backend_notes": ["Behavioral hazard GLM compatibility mode uses fixed-effect binomial glm with no random effects."],
        "random_effects": [],
    }


def _build_model_specs(config: BehaviorHazardConfig, *, selected_lag_ms: int) -> list[dict[str, object]]:
    backend = config.model_backend
    specs: list[dict[str, object]] = []
    rows = [
        ("fpp", "FPP", "M_0"),
        ("fpp", "FPP", "M_1"),
        ("fpp", "FPP", "M_2"),
        ("fpp", "FPP", "M_3"),
        ("fpp", "FPP", "M_4"),
        ("spp", "SPP", "M_0"),
        ("spp", "SPP", "M_1"),
        ("spp", "SPP", "M_2"),
        ("spp", "SPP", "M_3"),
        ("spp", "SPP", "M_4"),
        ("pooled", "pooled_fpp_spp", "M_pooled_main"),
        ("pooled", "pooled_fpp_spp", "M_pooled_anchor_interaction"),
    ]
    for anchor_subset, dataset, model_id in rows:
        formula_fixed = render_fixed_formula(model_id, lag_ms=selected_lag_ms)
        specs.append(
            {
                "anchor_subset": anchor_subset,
                "dataset": dataset,
                "model_id": model_id,
                "formula_fixed": formula_fixed,
                "formula_full": render_formula(model_id, lag_ms=selected_lag_ms, backend=backend),
            }
        )
    return specs


def _build_comparison_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for anchor_subset, pairs in MODEL_COMPARISONS.items():
        for parent_model_id, child_model_id in pairs:
            specs.append(
                {
                    "comparison_id": f"{anchor_subset}__{parent_model_id}__vs__{child_model_id}",
                    "anchor_subset": anchor_subset,
                    "parent_model_id": parent_model_id,
                    "child_model_id": child_model_id,
                }
            )
    return specs


def build_risksets(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    _log(verbose, "[behavior hazard] Building FPP risk set")
    fpp = build_anchor_riskset(config, anchor="FPP", verbose=verbose)
    _log(verbose, "[behavior hazard] Building SPP matched-control risk set")
    spp = build_anchor_riskset(config, anchor="SPP", verbose=verbose)
    pooled = pd.concat([fpp, spp], ignore_index=True, sort=False)
    outputs = _riskset_paths(paths)
    _write_parquet(fpp, outputs["fpp"])
    _write_parquet(spp, outputs["spp"])
    _write_parquet(pooled, outputs["pooled"])
    return outputs


def add_predictors(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    riskset_paths = _riskset_paths(paths)
    if not all(path.exists() for path in riskset_paths.values()):
        build_risksets(config_path, verbose=verbose)
    fpp = pd.read_parquet(riskset_paths["fpp"])
    spp = pd.read_parquet(riskset_paths["spp"])
    pooled = pd.read_parquet(riskset_paths["pooled"])
    pooled_filtered, overlap_counts, overlap_by_anchor = _apply_overlap_filter(pooled, config=config)
    fpp_filtered = pooled_filtered.loc[pooled_filtered["anchor_type"].astype(str) == "FPP"].copy()
    spp_filtered = pooled_filtered.loc[pooled_filtered["anchor_type"].astype(str) == "SPP"].copy()
    if config.only_overlap and _event_count(fpp_filtered) <= 0:
        raise ValueError("behavior.only_overlap=true left no FPP events for lag selection.")
    if config.only_overlap and fpp_filtered["episode_id"].nunique() < 2:
        LOGGER.warning("Overlap-only subset retained fewer than two FPP episodes; lag selection may be unstable.")
    fpp_pred, spp_pred, pooled_pred, scaling = standardize_predictors(
        fpp_filtered,
        spp_filtered,
        pooled_filtered,
        config=config,
        verbose=verbose,
    )
    metadata = _analysis_metadata(config, overlap_filter_column=str(overlap_counts["overlap_filter_column"]))
    fpp_pred = _append_analysis_metadata(fpp_pred, metadata)
    spp_pred = _append_analysis_metadata(spp_pred, metadata)
    pooled_pred = _append_analysis_metadata(pooled_pred, metadata)
    scaling = _append_analysis_metadata(scaling, metadata)
    outputs = _predictor_paths(paths)
    _write_parquet(fpp_pred, outputs["fpp"])
    _write_parquet(spp_pred, outputs["spp"])
    _write_parquet(pooled_pred, outputs["pooled"])
    _write_csv(scaling, outputs["scaling"])
    overlap_outputs = _overlap_report_paths(paths)
    _write_json(overlap_outputs["summary"], overlap_counts)
    _write_csv(overlap_by_anchor, overlap_outputs["by_anchor"])
    return outputs


def select_lag(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    fpp = pd.read_parquet(predictor_paths["fpp"])
    score_path = paths["lag_selection"] / "candidate_lag_scores.csv"
    selected_path = paths["lag_selection"] / "selected_lag.json"
    run_r_lag_selection(
        fpp_table=fpp,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        score_path=score_path,
        selected_path=selected_path,
        lag_sensitivity_path=paths["diagnostics"] / "lag_sensitivity.csv",
        verbose=verbose,
    )
    metadata = _analysis_metadata(config, overlap_filter_column=str(fpp.get("overlap_filter_column", pd.Series([""])).iloc[0]))
    lag_scores = _append_analysis_metadata(pd.read_csv(score_path), metadata)
    _write_csv(lag_scores, score_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected.update(metadata)
    selected["n_rows"] = int(len(fpp))
    selected["n_events"] = _event_count(fpp)
    selected["event_rate"] = _event_rate(fpp)
    _write_json(selected_path, selected)
    _log(verbose, "[behavior hazard] Selected shared lag: %d ms", int(selected["selected_lag_ms"]))
    return {"scores": score_path, "selected": selected_path}


def _write_model_jsons_from_r_outputs(model_metrics: pd.DataFrame, paths: dict[str, Path]) -> None:
    output_dir_by_anchor = {
        "fpp": paths["models_fpp"],
        "spp": paths["models_spp"],
        "pooled": paths["models_pooled"],
    }
    for row in model_metrics.to_dict("records"):
        output_dir = output_dir_by_anchor[str(row["anchor_subset"])]
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            key: row.get(key)
            for key in [
                "anchor_subset",
                "model_id",
                "model_backend",
                "lag_selection_criterion",
                "selected_lag_ms",
                "formula_fixed",
                "formula_full",
                "random_effects",
                "covariance_type",
                "n",
                "k",
                "logLik",
                "AIC",
                "BIC",
                "converged",
                "warnings",
                "only_overlap",
                "overlap_filter_column",
                "overlap_filter_definition",
            ]
        }
        _write_json(output_dir / f"{payload['model_id']}.json", payload)


def fit_models(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    selected_lag_ms = int(selected_payload["selected_lag_ms"])
    fpp, spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_model_coefficients.csv"
    run_r_model_bundle(
        fpp_table=fpp,
        spp_table=spp,
        pooled_table=pooled,
        selected_lag_ms=selected_lag_ms,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        model_specs=_build_model_specs(config, selected_lag_ms=selected_lag_ms),
        comparison_specs=[],
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=paths["logs"] / "r_model_comparisons.csv",
        convergence_path=paths["logs"] / "r_convergence_warnings.csv",
        lag_sensitivity_path=None,
        verbose=verbose,
    )
    metadata = _analysis_metadata(config, overlap_filter_column=str(fpp.get("overlap_filter_column", pd.Series([""])).iloc[0]))
    model_metrics = _append_analysis_metadata(pd.read_csv(model_metrics_path), metadata)
    coefficients = _append_analysis_metadata(pd.read_csv(coefficient_path), metadata)
    _write_csv(model_metrics, model_metrics_path)
    _write_csv(coefficients, coefficient_path)
    convergence_path = paths["logs"] / "r_convergence_warnings.csv"
    if convergence_path.exists():
        _write_csv(_append_analysis_metadata(pd.read_csv(convergence_path), metadata), convergence_path)
    comparison_path = paths["logs"] / "r_model_comparisons.csv"
    if comparison_path.exists():
        _write_csv(_append_analysis_metadata(pd.read_csv(comparison_path), metadata), comparison_path)
    _write_model_jsons_from_r_outputs(model_metrics, paths)
    return {"models_root": paths["models"]}


def build_tables(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    selected_lag_ms = int(selected_payload["selected_lag_ms"])
    fpp, spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_table_coefficients.csv"
    comparison_path = paths["logs"] / "r_table_comparisons.csv"
    convergence_path = paths["diagnostics"] / "convergence_warnings.csv"
    run_r_model_bundle(
        fpp_table=fpp,
        spp_table=spp,
        pooled_table=pooled,
        selected_lag_ms=selected_lag_ms,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        model_specs=_build_model_specs(config, selected_lag_ms=selected_lag_ms),
        comparison_specs=_build_comparison_specs(),
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=comparison_path,
        convergence_path=convergence_path,
        lag_sensitivity_path=None,
        figure_prediction_path=paths["tables"] / "figure_predictions.csv",
        timing_heatmap_path=paths["tables"] / "timing_heatmap_predictions.csv",
        three_way_path=paths["tables"] / "three_way_heatmap_predictions.csv",
        verbose=verbose,
    )
    model_metrics = pd.read_csv(model_metrics_path)
    coefficients = pd.read_csv(coefficient_path)
    comparisons = pd.read_csv(comparison_path)
    metadata = _analysis_metadata(config, overlap_filter_column=str(pooled.get("overlap_filter_column", pd.Series([""])).iloc[0]))
    model_metrics = _append_analysis_metadata(model_metrics, metadata)
    coefficients = _append_analysis_metadata(coefficients, metadata)
    comparisons = _append_analysis_metadata(comparisons, metadata)
    _write_csv(model_metrics, paths["tables"] / "model_summary.csv")
    _write_csv(coefficients, paths["tables"] / "coefficient_summary.csv")
    _write_csv(comparisons, paths["tables"] / "model_comparisons.csv")
    _write_csv(_append_analysis_metadata(odds_ratio_rows(coefficients), metadata), paths["tables"] / "odds_ratios.csv")
    _write_csv(_append_analysis_metadata(event_rate_summary(pooled), metadata), paths["tables"] / "event_rate_summary.csv")
    collinearity_columns = [
        "z_time_from_partner_onset_s",
        "z_time_from_partner_offset_s",
        "z_time_from_partner_offset_s_squared",
        f"z_information_rate_lag_{selected_lag_ms}",
        f"z_prop_expected_cum_info_lag_{selected_lag_ms}",
    ]
    _write_csv(_append_analysis_metadata(bins_by_subject(pooled), metadata), paths["tables"] / "bins_by_subject.csv")
    _write_csv(_append_analysis_metadata(collinearity_summary(pooled, collinearity_columns), metadata), paths["tables"] / "collinearity_summary.csv")
    overlap_reports = _overlap_report_paths(paths)
    if overlap_reports["by_anchor"].exists():
        overlap_by_anchor = pd.read_csv(overlap_reports["by_anchor"])
    else:
        overlap_by_anchor = pd.DataFrame(columns=["anchor_type", "n_rows_before", "n_rows_after", "n_events_before", "n_events_after", "event_rate_before", "event_rate_after"])
    _write_csv(_append_analysis_metadata(overlap_by_anchor, metadata), overlap_reports["by_anchor"])
    lag_sensitivity_path = paths["diagnostics"] / "lag_sensitivity.csv"
    if not lag_sensitivity_path.exists():
        _empty_csv(lag_sensitivity_path, ["candidate_lag_ms", "term", "predictor", "estimate", "ci_low", "ci_high", "backend"], note="Lag sensitivity was not available.")
    else:
        _write_csv(_append_analysis_metadata(pd.read_csv(lag_sensitivity_path), metadata), lag_sensitivity_path)
    _empty_csv(paths["diagnostics"] / "loo_subject_summary.csv", ["left_out_subject", "term", "estimate", "ci_low", "ci_high", "notes"], note="Disabled by config or not implemented.")
    summarize_outputs(config_path, verbose=verbose)
    return {"tables": paths["tables"] / "model_comparisons.csv"}


def summarize_outputs(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    backend_meta = _backend_metadata(config)
    summary_path = paths["logs"] / "summary_manifest.json"
    payload = {
        "hazard_root": str(paths["hazard_root"]),
        "tables": str(paths["tables"]),
        "diagnostics": str(paths["diagnostics"]),
        "figures_main": str(paths["figures_main"]),
        "figures_supp": str(paths["figures_supp"]),
        "figures_qc": str(paths["figures_qc"]),
        "selected_lag_ms": int(selected_payload["selected_lag_ms"]),
        "selector_model_id": str(selected_payload["selector_model_id"]),
        "anchor_subset": str(selected_payload["anchor_subset"]),
        "model_backend": backend_meta["model_backend"],
        "lag_selection_criterion": config.lag_selection_criterion,
        "backend": backend_meta["backend_runtime"],
        "covariance_type": backend_meta["covariance_type"],
        "cluster_variable": backend_meta["cluster_variable"],
        "backend_notes": list(backend_meta["backend_notes"]),
        "possible_backends": ["glm", "glmm"],
        "available_runtime_backends": [R_BACKEND_ID, STANDARD_BACKEND_ID, RIDGE_BACKEND_ID],
    }
    payload.update(_analysis_metadata(config))
    overlap_summary_path = _overlap_report_paths(paths)["summary"]
    if overlap_summary_path.exists():
        payload["overlap_filter_summary"] = json.loads(overlap_summary_path.read_text(encoding="utf-8"))
    _write_json(summary_path, payload)
    return {"summary": summary_path}


def render_figures(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "figure_predictions.csv").exists():
        build_tables(config_path, verbose=verbose)
    lag_scores = pd.read_csv(paths["lag_selection"] / "candidate_lag_scores.csv")
    selected = json.loads((paths["lag_selection"] / "selected_lag.json").read_text(encoding="utf-8"))
    selected_lag_ms = int(selected["selected_lag_ms"])
    primary_predictions = pd.read_csv(paths["tables"] / "figure_predictions.csv")
    timing_predictions = pd.read_csv(paths["tables"] / "timing_heatmap_predictions.csv")
    lag_sensitivity = pd.read_csv(paths["diagnostics"] / "lag_sensitivity.csv")
    metadata = _analysis_metadata(config, overlap_filter_column=str(selected.get("overlap_filter_column", "")))
    title_suffix = " [overlap-only]" if config.only_overlap else ""
    primary_predictions = _append_analysis_metadata(primary_predictions, metadata)
    timing_predictions = _append_analysis_metadata(timing_predictions, metadata)
    _write_csv(primary_predictions, paths["tables"] / "figure_predictions.csv")
    _write_csv(timing_predictions, paths["tables"] / "timing_heatmap_predictions.csv")
    if (paths["tables"] / "three_way_heatmap_predictions.csv").exists():
        three_way = pd.read_csv(paths["tables"] / "three_way_heatmap_predictions.csv")
        _write_csv(_append_analysis_metadata(three_way, metadata), paths["tables"] / "three_way_heatmap_predictions.csv")
    if not lag_sensitivity.empty:
        lag_sensitivity = _append_analysis_metadata(lag_sensitivity, metadata)
        _write_csv(lag_sensitivity, paths["diagnostics"] / "lag_sensitivity.csv")
    outputs: dict[str, Path] = {}
    outputs["lag_selection"] = plot_lag_selection(lag_scores, paths["figures_main"] / "fig01_lag_selection.png", title_suffix=title_suffix)
    outputs["primary_effects"] = plot_primary_effects(primary_predictions, paths["figures_main"] / "fig02_primary_information_effects.png", title_suffix=title_suffix)
    outputs["timing"] = plot_timing_information_interaction(
        timing_predictions.loc[timing_predictions["panel"].astype(str) == "A"].copy(),
        timing_predictions.loc[timing_predictions["panel"].astype(str) == "B"].copy(),
        paths["figures_main"] / "fig03_timing_information_interaction.png",
        title_suffix=title_suffix,
    )
    if not lag_sensitivity.empty:
        plot_lag_sensitivity(lag_sensitivity, selected_lag_ms=selected_lag_ms, output_path=paths["figures_supp"] / "figS01_lag_sensitivity.png")
    else:
        _write_placeholder_figure(paths["figures_supp"] / "figS01_lag_sensitivity.png", title="Lag sensitivity", message="Lag sensitivity table was unavailable.")
    _write_placeholder_figure(paths["figures_supp"] / "figS02_extra_timing_maps.png", title="Supplementary timing maps", message="Main Figure 3 now uses ridge plots from M_4.")
    _write_placeholder_figure(paths["figures_supp"] / "figS03_three_way_interaction.png", title="Three-way interaction", message="The active behavioral target uses per-anchor M_0–M_4 models plus the pooled omnibus anchor-interaction model, and does not fit an exploratory three-way interaction model.")
    return outputs


def render_qc(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "event_rate_summary.csv").exists():
        build_tables(config_path, verbose=verbose)
    event_rates = pd.read_csv(paths["tables"] / "event_rate_summary.csv")
    bins = pd.read_csv(paths["tables"] / "bins_by_subject.csv")
    collinearity = pd.read_csv(paths["tables"] / "collinearity_summary.csv")
    convergence = pd.read_csv(paths["diagnostics"] / "convergence_warnings.csv")
    outputs = {
        "event_rates": plot_qc_bars(event_rates, x="anchor_type", y="event_rate", title="Event rates", output_path=paths["figures_qc"] / "qc_event_rates.png"),
        "bins_by_subject": plot_qc_bars(bins, x="subject", y="n_bins", title="Bins by subject", output_path=paths["figures_qc"] / "qc_bins_by_subject.png"),
        "collinearity": plot_qc_bars(collinearity, x="term", y="vif", title="Collinearity", output_path=paths["figures_qc"] / "qc_collinearity.png"),
        "model_convergence": plot_qc_bars(
            convergence.assign(count=1).groupby("model_id", as_index=False)["count"].sum() if not convergence.empty else pd.DataFrame({"model_id": [], "count": []}),
            x="model_id",
            y="count",
            title="Model convergence warnings",
            output_path=paths["figures_qc"] / "qc_model_convergence.png",
        ),
    }
    return outputs


def run_behavior_hazard_stage(stage: str, *, config_path: str | Path, verbose: bool = False) -> dict[str, Path]:
    _configure_logging(verbose=verbose)
    _log(verbose, "[behavior hazard] Starting stage: %s", stage)
    if stage == "build-risksets":
        return build_risksets(config_path, verbose=verbose)
    if stage == "add-predictors":
        return add_predictors(config_path, verbose=verbose)
    if stage == "select-lag":
        return select_lag(config_path, verbose=verbose)
    if stage == "fit-models":
        return fit_models(config_path, verbose=verbose)
    if stage == "tables":
        return build_tables(config_path, verbose=verbose)
    if stage == "summarize":
        return summarize_outputs(config_path, verbose=verbose)
    if stage == "figures":
        return render_figures(config_path, verbose=verbose)
    if stage == "qc":
        return render_qc(config_path, verbose=verbose)
    if stage == "all":
        build_risksets(config_path, verbose=verbose)
        add_predictors(config_path, verbose=verbose)
        select_lag(config_path, verbose=verbose)
        fit_models(config_path, verbose=verbose)
        build_tables(config_path, verbose=verbose)
        render_figures(config_path, verbose=verbose)
        render_qc(config_path, verbose=verbose)
        return summarize_outputs(config_path, verbose=verbose)
    raise ValueError(f"Unknown behavior hazard stage: {stage}")
