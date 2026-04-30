"""Neural low-level hazard comparison with strict same-row parent/child fitting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
from typing import Any, Callable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from cas.hazard.config import HazardAnalysisConfig, NeuralHazardConfig, config_to_metadata_dict
from cas.hazard.io import (
    load_lowlevel_neural_tables,
    load_normalized_events_table,
    load_surprisal_table,
    prepare_neural_output_directories,
    write_neural_hazard_table,
)
from cas.hazard.pipeline import (  # Reuse existing feature engineering without modifying legacy package code.
    _add_behavioural_controls_to_neural_riskset,
    _add_neural_features_and_pcs,
)
from cas.hazard.riskset import build_neural_partner_ipu_risksets
from cas.hazard_behavior.progress import progress_iterable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NeuralLowlevelRunResult:
    """High-level output handle for neural low-level runs."""

    output_dir: Path
    comparison_path: Path
    fit_metrics_path: Path
    coefficients_path: Path


@dataclass(frozen=True, slots=True)
class FittedFormulaModel:
    """One fitted formula model on a fixed subset."""

    model_name: str
    formula: str
    result: Any | None
    n_rows: int
    n_events: int
    n_predictors: int
    converged: bool
    fit_warnings: list[str]
    error_message: str | None
    row_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def success(self) -> bool:
        return self.result is not None and self.error_message is None


@dataclass(frozen=True, slots=True)
class NeuralLagWindow:
    """One guarded causal neural lag window."""

    lag_start_ms: int
    lag_end_ms: int

    @property
    def lag_label(self) -> str:
        return f"{int(self.lag_start_ms)}_{int(self.lag_end_ms)}ms"

    @property
    def window_start_lag_s(self) -> float:
        return float(self.lag_end_ms) / 1000.0

    @property
    def window_end_lag_s(self) -> float:
        return float(self.lag_start_ms) / 1000.0


def run_neural_lowlevel_hazard_analysis(config: HazardAnalysisConfig) -> NeuralLowlevelRunResult:
    """Run same-row neural low-level hazard analysis and write corrected outputs."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    neural_config = config.neural
    neural_config.validate()
    out_dir = (neural_config.out_dir or config.output.output_dir).resolve()
    output_dirs = prepare_neural_output_directories(out_dir, overwrite=config.misc.overwrite)

    events_table, event_warnings = load_normalized_events_table(neural_config.events_path or config.input.events_table_path)
    surprisal_table, surprisal_warnings = load_surprisal_table(
        neural_config.input.surprisal_paths if neural_config.input else tuple()
    )
    lowlevel_table = load_lowlevel_neural_tables(
        neural_config.input.lowlevel_neural_paths if neural_config.input else tuple()
    )
    riskset_build_result = build_neural_partner_ipu_risksets(
        events_table=events_table,
        surprisal_table=surprisal_table,
        neural_config=neural_config,
    )

    if neural_config.select_neural_lags:
        return _run_neural_lowlevel_lag_selection_analysis(
            config=config,
            output_dirs=output_dirs,
            surprisal_table=surprisal_table,
            lowlevel_table=lowlevel_table,
            riskset_build_result=riskset_build_result,
            warnings_payload=[*event_warnings, *surprisal_warnings, *riskset_build_result.warnings],
        )

    comparison_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    hazard_paths: dict[str, Path] = {}
    counts_by_event_and_family: dict[str, dict[str, dict[str, int]]] = {}
    spp_failure_messages: list[str] = []

    for event_type in ("fpp", "spp"):
        if event_type not in riskset_build_result.risksets_by_event:
            continue
        LOGGER.info("Preparing neural low-level riskset for event_type=%s", event_type)
        try:
            riskset_table = riskset_build_result.risksets_by_event[event_type]
            episodes_table = riskset_build_result.episode_summaries_by_event[event_type]
            enriched = _add_behavioural_controls_to_neural_riskset(
                riskset_table,
                episodes_table=episodes_table,
                surprisal_table=surprisal_table,
                config=config,
            )
            enriched_with_neural, _neural_qc = _add_neural_features_and_pcs(
                enriched,
                lowlevel_table=lowlevel_table,
                config=config,
            )
            comparison_table, coefficients_table, family_counts = fit_event_neural_comparisons(
                riskset_table=enriched_with_neural,
                event_type=event_type,
                neural_config=neural_config,
            )
        except Exception as error:
            if event_type != "spp":
                raise
            LOGGER.warning("SPP processing failed; recording failed_convergence rows: %s", error)
            comparison_table = pd.DataFrame(
                [
                    _failed_comparison_row(
                        event_type="spp",
                        neural_family=family,
                        parent_model=f"M_behaviour_SPP_{family}_sample",
                        child_model=(
                            "M_alpha_beta_SPP"
                            if family == "alpha_beta"
                            else f"M_{family}_SPP"
                        ),
                        n_rows_candidate=0,
                        n_events_candidate=0,
                        error_message=str(error),
                        fit_warnings=[],
                    )
                    for family in ("alpha", "beta", "alpha_beta")
                ]
            )
            coefficients_table = pd.DataFrame()
            family_counts = {
                "alpha": {"n_rows": 0, "n_events": 0},
                "beta": {"n_rows": 0, "n_events": 0},
                "alpha_beta": {"n_rows": 0, "n_events": 0},
            }
            enriched_with_neural = pd.DataFrame()
        counts_by_event_and_family[event_type] = family_counts
        if event_type == "spp":
            failed_rows = comparison_table.loc[
                comparison_table["status"].astype(str) == "failed_convergence",
                "error_message",
            ].dropna()
            spp_failure_messages.extend(str(value) for value in failed_rows.tolist())
        comparison_frames.append(comparison_table)
        coefficient_frames.append(coefficients_table)
        if not enriched_with_neural.empty:
            hazard_paths[event_type] = write_neural_hazard_table(
                enriched_with_neural,
                output_dir=output_dirs["riskset"],
                event_type=event_type,
            )

    comparison_output = (
        pd.concat(comparison_frames, ignore_index=True, sort=False)
        if comparison_frames
        else pd.DataFrame()
    )
    coefficients_output = (
        pd.concat(coefficient_frames, ignore_index=True, sort=False)
        if coefficient_frames
        else pd.DataFrame()
    )
    comparison_path = output_dirs["models"] / "neural_lowlevel_model_comparison.csv"
    comparison_output.to_csv(comparison_path, index=False)
    # Backward-compatible legacy output name expected by workflow.
    comparison_output.to_csv(output_dirs["models"] / "neural_model_comparison.csv", index=False)

    coefficients_path = output_dirs["models"] / "neural_lowlevel_model_summary.csv"
    coefficients_output.to_csv(coefficients_path, index=False)
    # Backward-compatible legacy output name expected by workflow.
    coefficients_output.to_csv(output_dirs["models"] / "neural_coefficients.csv", index=False)

    fit_metrics_payload = {
        "corrected_same_row_comparisons": True,
        "fpp_comparisons_valid": bool(
            not comparison_output.empty
            and comparison_output.loc[
                comparison_output["event_type"].astype(str) == "fpp",
                "comparison_valid",
            ].astype(bool).all()
        ),
        "spp_converged": bool(
            not comparison_output.empty
            and (
                comparison_output.loc[
                    comparison_output["event_type"].astype(str) == "spp",
                    "status",
                ].astype(str)
                != "failed_convergence"
            ).all()
        ),
        "spp_failure_reason": "; ".join(spp_failure_messages) if spp_failure_messages else None,
        "n_rows_by_event_type_and_family": {
            event_type: {family: stats_dict.get("n_rows", 0) for family, stats_dict in family_dict.items()}
            for event_type, family_dict in counts_by_event_and_family.items()
        },
        "n_events_by_event_type_and_family": {
            event_type: {family: stats_dict.get("n_events", 0) for family, stats_dict in family_dict.items()}
            for event_type, family_dict in counts_by_event_and_family.items()
        },
        "warnings": [*event_warnings, *surprisal_warnings, *riskset_build_result.warnings],
    }
    fit_metrics_path = output_dirs["models"] / "neural_lowlevel_fit_metrics.json"
    fit_metrics_path.write_text(json.dumps(fit_metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Backward-compatible legacy output name expected by workflow.
    (output_dirs["models"] / "neural_fit_metrics.json").write_text(
        json.dumps(fit_metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plotted_valid = plot_neural_lowlevel_model_comparison(comparison_output, output_dirs["figures"])
    plot_neural_lowlevel_coefficients(coefficients_output, output_dirs["figures"])
    plot_neural_lowlevel_power_qc(hazard_paths, output_dirs["figures"])
    # Backward-compatible legacy figure names expected by workflow.
    _plot_legacy_delta_aliases(plotted_valid, output_dirs["figures"])

    (output_dirs["logs"] / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "config": config_to_metadata_dict(config),
                "notes": [
                    "Neural low-level comparisons use strict same-row parent/child fitting.",
                    "SPP failures are reported as failed_convergence and excluded from interpretation.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dirs["models"] / "model_summary.txt").write_text(
        "Neural low-level hazard analysis completed. See neural_lowlevel_model_comparison.csv.\n",
        encoding="utf-8",
    )
    return NeuralLowlevelRunResult(
        output_dir=output_dirs["root"],
        comparison_path=comparison_path,
        fit_metrics_path=fit_metrics_path,
        coefficients_path=coefficients_path,
    )


def _run_neural_lowlevel_lag_selection_analysis(
    *,
    config: HazardAnalysisConfig,
    output_dirs: dict[str, Path],
    surprisal_table: pd.DataFrame,
    lowlevel_table: pd.DataFrame,
    riskset_build_result: Any,
    warnings_payload: list[str],
) -> NeuralLowlevelRunResult:
    neural_config = config.neural
    lag_windows = _iter_neural_lag_windows(neural_config)
    LOGGER.info(
        "Starting neural lag selection with %d lag windows, selection criterion=%s, null permutations=%d, skip_spp_on_failure=%s",
        int(len(lag_windows)),
        neural_config.neural_lag_selection_criterion,
        int(neural_config.neural_null_permutations),
        bool(neural_config.skip_spp_on_failure),
    )
    comparison_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    null_frames: list[pd.DataFrame] = []
    hazard_paths: dict[str, Path] = {}
    qc_by_event: dict[str, Any] = {}
    spp_failure_messages: list[str] = []

    for event_type in ("fpp", "spp"):
        if event_type not in riskset_build_result.risksets_by_event:
            continue
        if event_type == "spp" and neural_config.skip_spp_on_failure:
            message = (
                "Skipped SPP neural lag selection because skip_spp_on_failure=true "
                "and SPP lag sweeps are currently unstable on this dataset."
            )
            LOGGER.warning(message)
            comparison_frames.append(pd.DataFrame(_failed_event_lag_rows(event_type=event_type, lag_windows=lag_windows, error_message=message)))
            qc_by_event[event_type] = {"status": "failed_convergence", "error_message": message, "skipped": True}
            spp_failure_messages.append(message)
            continue
        LOGGER.info("Preparing neural lag sweep for event_type=%s", event_type)
        try:
            riskset_table = riskset_build_result.risksets_by_event[event_type]
            episodes_table = riskset_build_result.episode_summaries_by_event[event_type]
            enriched = _add_behavioural_controls_to_neural_riskset(
                riskset_table,
                episodes_table=episodes_table,
                surprisal_table=surprisal_table,
                config=config,
            )
            event_comparison_frames: list[pd.DataFrame] = []
            event_coefficient_frames: list[pd.DataFrame] = []
            event_null_frames: list[pd.DataFrame] = []
            lag_qc_rows: list[dict[str, object]] = []
            default_hazard_table: pd.DataFrame | None = None
            for lag_window in progress_iterable(
                lag_windows,
                total=len(lag_windows),
                description=f"neural lags ({event_type.upper()})",
                enabled=True,
            ):
                LOGGER.info(
                    "Running event_type=%s neural lag=%s (%d-%d ms)",
                    event_type,
                    lag_window.lag_label,
                    int(lag_window.lag_start_ms),
                    int(lag_window.lag_end_ms),
                )
                lag_config = _config_with_neural_lag(config, lag_window)
                enriched_with_neural, neural_qc = _add_neural_features_and_pcs(
                    enriched,
                    lowlevel_table=lowlevel_table,
                    config=lag_config,
                )
                if default_hazard_table is None or lag_window.lag_label == "100_500ms":
                    default_hazard_table = enriched_with_neural
                comparison_table, coefficients_table, family_counts = fit_event_neural_comparisons(
                    riskset_table=enriched_with_neural,
                    event_type=event_type,
                    neural_config=lag_config.neural,
                    lag_window=lag_window,
                )
                event_comparison_frames.append(comparison_table)
                event_coefficient_frames.append(coefficients_table)
                lag_qc_rows.append(
                    {
                        "lag_label": lag_window.lag_label,
                        "lag_start_ms": lag_window.lag_start_ms,
                        "lag_end_ms": lag_window.lag_end_ms,
                        "family_counts": family_counts,
                        "neural_qc": neural_qc,
                    }
                )
                if event_type == "fpp" and int(neural_config.neural_null_permutations) > 0:
                    LOGGER.info(
                        "Running circular-shift null for event_type=%s lag=%s with %d permutations",
                        event_type,
                        lag_window.lag_label,
                        int(neural_config.neural_null_permutations),
                    )
                    event_null_frames.append(
                        _compute_lag_null_summaries(
                            riskset_table=enriched_with_neural,
                            event_type=event_type,
                            neural_config=lag_config.neural,
                            lag_window=lag_window,
                            n_permutations=int(neural_config.neural_null_permutations),
                        )
                    )
            if default_hazard_table is not None:
                hazard_paths[event_type] = write_neural_hazard_table(
                    default_hazard_table,
                    output_dir=output_dirs["riskset"],
                    event_type=event_type,
                )
            qc_by_event[event_type] = {"lag_qc": lag_qc_rows}
            comparison_frames.extend(event_comparison_frames)
            coefficient_frames.extend(event_coefficient_frames)
            null_frames.extend(event_null_frames)
            LOGGER.info(
                "Completed neural lag sweep for event_type=%s: comparisons=%d null_summaries=%d",
                event_type,
                int(sum(len(frame) for frame in event_comparison_frames)),
                int(sum(len(frame) for frame in event_null_frames)),
            )
        except Exception as error:
            if event_type != "spp" or not neural_config.skip_spp_on_failure:
                raise
            LOGGER.warning("SPP lag sweep failed; recording failed_convergence rows: %s", error)
            failed_rows = _failed_event_lag_rows(event_type=event_type, lag_windows=lag_windows, error_message=str(error))
            comparison_frames.append(pd.DataFrame(failed_rows))
            qc_by_event[event_type] = {"status": "failed_convergence", "error_message": str(error)}
            spp_failure_messages.append(str(error))

    comparison_output = pd.concat(comparison_frames, ignore_index=True, sort=False) if comparison_frames else pd.DataFrame()
    coefficients_output = pd.concat(coefficient_frames, ignore_index=True, sort=False) if coefficient_frames else pd.DataFrame()
    null_output = pd.concat(null_frames, ignore_index=True, sort=False) if null_frames else _empty_null_summary_table()
    selected_output = select_best_neural_lags(comparison_output)

    comparison_path = output_dirs["models"] / "neural_lowlevel_model_comparison.csv"
    lag_selection_path = output_dirs["models"] / "neural_lowlevel_lag_selection.csv"
    selected_comparison_path = output_dirs["models"] / "neural_lowlevel_selected_model_comparison.csv"
    null_summary_path = output_dirs["models"] / "neural_lowlevel_lag_null_summary.csv"
    coefficients_path = output_dirs["models"] / "neural_lowlevel_model_summary.csv"
    selected_lags_path = output_dirs["models"] / "neural_lowlevel_selected_lags.json"

    comparison_output.to_csv(comparison_path, index=False)
    comparison_output.to_csv(lag_selection_path, index=False)
    comparison_output.to_csv(output_dirs["models"] / "neural_model_comparison.csv", index=False)
    selected_output.to_csv(selected_comparison_path, index=False)
    null_output.to_csv(null_summary_path, index=False)
    coefficients_output.to_csv(coefficients_path, index=False)
    coefficients_output.to_csv(output_dirs["models"] / "neural_coefficients.csv", index=False)

    selected_lags_payload = build_selected_lags_payload(
        comparison_output=comparison_output,
        selected_output=selected_output,
        null_output=null_output,
        neural_config=neural_config,
        spp_failure_messages=spp_failure_messages,
    )
    selected_lags_path.write_text(json.dumps(selected_lags_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fit_metrics_payload = {
        "corrected_same_row_comparisons": True,
        "same_row_comparisons_enforced": True,
        "selection_criterion": "delta_bic",
        "delta_bic_convention": "child_bic - parent_bic; negative favours neural child",
        "neural_lag_grid_ms": [
            {"lag_start_ms": lag_window.lag_start_ms, "lag_end_ms": lag_window.lag_end_ms, "lag_label": lag_window.lag_label}
            for lag_window in lag_windows
        ],
        "selected_lags": selected_lags_payload.get("best_lag_by_event_type_and_family"),
        "n_valid_models": _count_models_by_event_and_family(comparison_output, valid=True),
        "n_failed_models": _count_models_by_event_and_family(comparison_output, valid=False),
        "spp_failure_status": "; ".join(spp_failure_messages) if spp_failure_messages else None,
        "null_permutations": int(neural_config.neural_null_permutations),
        "null_grouping_unit": "participant_speaker_id x run",
        "warnings": warnings_payload,
        "event_qc": qc_by_event,
    }
    fit_metrics_path = output_dirs["models"] / "neural_lowlevel_fit_metrics.json"
    fit_metrics_path.write_text(json.dumps(fit_metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dirs["models"] / "neural_fit_metrics.json").write_text(
        json.dumps(fit_metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plotted_valid = plot_neural_lowlevel_model_comparison(selected_output, output_dirs["figures"])
    plot_neural_lowlevel_coefficients(coefficients_output, output_dirs["figures"])
    plot_neural_lowlevel_power_qc(hazard_paths, output_dirs["figures"])
    _plot_legacy_delta_aliases(plotted_valid, output_dirs["figures"])
    plot_neural_lowlevel_delta_bic_by_lag(comparison_output, selected_output, output_dirs["figures"])
    plot_neural_lowlevel_lag_null_comparison(null_output, selected_output, output_dirs["figures"])
    plot_neural_lowlevel_selected_model_comparison(selected_output, output_dirs["figures"])

    (output_dirs["logs"] / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "config": config_to_metadata_dict(config),
                "notes": [
                    "Neural low-level lag selection uses guarded causal windows and strict same-row parent/child fitting.",
                    "SPP failures are reported as failed_convergence and excluded from interpretation.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dirs["models"] / "model_summary.txt").write_text(
        "Neural low-level lag selection completed. See neural_lowlevel_lag_selection.csv.\n",
        encoding="utf-8",
    )
    return NeuralLowlevelRunResult(
        output_dir=output_dirs["root"],
        comparison_path=comparison_path,
        fit_metrics_path=fit_metrics_path,
        coefficients_path=coefficients_path,
    )


def compute_neural_lag_bounds(bin_end: float, *, lag_start_ms: int, lag_end_ms: int) -> tuple[float, float]:
    window_start = float(bin_end) - (float(lag_end_ms) / 1000.0)
    window_end = float(bin_end) - (float(lag_start_ms) / 1000.0)
    return window_start, window_end


def select_neural_lag_mask(
    times: np.ndarray | pd.Series,
    *,
    bin_end: float,
    lag_start_ms: int,
    lag_end_ms: int,
) -> np.ndarray:
    window_start, window_end = compute_neural_lag_bounds(
        float(bin_end),
        lag_start_ms=int(lag_start_ms),
        lag_end_ms=int(lag_end_ms),
    )
    time_values = np.asarray(times, dtype=float)
    return (time_values >= window_start) & (time_values <= window_end)


def _iter_neural_lag_windows(neural_config: NeuralHazardConfig) -> list[NeuralLagWindow]:
    return [
        NeuralLagWindow(lag_start_ms=int(lag_start_ms), lag_end_ms=int(lag_end_ms))
        for lag_start_ms, lag_end_ms in neural_config.neural_lag_grid_ms
    ]


def _config_with_neural_lag(config: HazardAnalysisConfig, lag_window: NeuralLagWindow) -> HazardAnalysisConfig:
    return replace(
        config,
        neural=replace(
            config.neural,
            window=replace(
                config.neural.window,
                start_lag_s=lag_window.window_start_lag_s,
                end_lag_s=lag_window.window_end_lag_s,
            ),
        ),
    )


def _failed_event_lag_rows(*, event_type: str, lag_windows: list[NeuralLagWindow], error_message: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    suffix = event_type.upper()
    for lag_window in lag_windows:
        for family in ("alpha", "beta", "alpha_beta"):
            rows.append(
                _failed_comparison_row(
                    event_type=event_type,
                    neural_family=family,
                    lag_window=lag_window,
                    parent_model=f"M_behaviour_{suffix}_{family}_sample_lag_{lag_window.lag_label}",
                    child_model=(
                        f"M_alpha_beta_{suffix}_lag_{lag_window.lag_label}"
                        if family == "alpha_beta"
                        else f"M_{family}_{suffix}_lag_{lag_window.lag_label}"
                    ),
                    n_rows_candidate=0,
                    n_events_candidate=0,
                    error_message=error_message,
                    fit_warnings=[],
                )
            )
    return rows


def select_best_neural_lags(comparison_output: pd.DataFrame) -> pd.DataFrame:
    if comparison_output.empty:
        return comparison_output.copy()
    working = comparison_output.copy()
    working["delta_bic"] = pd.to_numeric(working.get("delta_bic"), errors="coerce")
    working = working.loc[working["comparison_valid"].astype(bool) & np.isfinite(working["delta_bic"])].copy()
    if working.empty:
        return comparison_output.head(0).copy()
    selected_rows: list[pd.Series] = []
    for _, group in working.groupby(["event_type", "neural_family"], sort=True):
        selected_rows.append(group.sort_values(["delta_bic", "delta_aic"], kind="mergesort").iloc[0])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_selected_lags_payload(
    *,
    comparison_output: pd.DataFrame,
    selected_output: pd.DataFrame,
    null_output: pd.DataFrame,
    neural_config: NeuralHazardConfig,
    spp_failure_messages: list[str],
) -> dict[str, object]:
    best: dict[str, dict[str, object]] = {}
    for event_type in ("fpp", "spp"):
        event_rows = selected_output.loc[selected_output["event_type"].astype(str) == event_type].copy()
        event_payload: dict[str, object] = {}
        for family in ("alpha", "beta", "alpha_beta"):
            family_rows = event_rows.loc[event_rows["neural_family"].astype(str) == family]
            if family_rows.empty:
                event_payload[family] = {"status": "failed_convergence" if event_type == "spp" and spp_failure_messages else "unavailable"}
                continue
            row = family_rows.iloc[0]
            event_payload[family] = {
                "lag_start_ms": int(row["lag_start_ms"]),
                "lag_end_ms": int(row["lag_end_ms"]),
                "lag_label": str(row["lag_label"]),
                "delta_bic": _maybe_float(row["delta_bic"]),
                "delta_aic": _maybe_float(row["delta_aic"]),
            }
            if not null_output.empty:
                match = null_output.loc[
                    (null_output["event_type"].astype(str) == event_type)
                    & (null_output["neural_family"].astype(str) == family)
                    & (null_output["lag_label"].astype(str) == str(row["lag_label"]))
                ]
                if not match.empty:
                    event_payload[family]["empirical_p_delta_bic"] = _maybe_float(match.iloc[0]["empirical_p_delta_bic"])
        best[event_type] = event_payload
    return {
        "selection_criterion": "delta_bic",
        "selection_criterion_config": neural_config.neural_lag_selection_criterion,
        "delta_bic_convention": "child_bic - parent_bic; negative favours neural child",
        "best_lag_by_event_type_and_family": best,
        "n_valid_lag_models_by_event_type_and_family": _count_models_by_event_and_family(comparison_output, valid=True),
        "n_failed_lag_models_by_event_type_and_family": _count_models_by_event_and_family(comparison_output, valid=False),
    }


def _count_models_by_event_and_family(comparison_output: pd.DataFrame, *, valid: bool) -> dict[str, dict[str, int]]:
    if comparison_output.empty:
        return {}
    counts: dict[str, dict[str, int]] = {}
    for (event_type, neural_family), group in comparison_output.groupby(["event_type", "neural_family"], sort=True):
        mask = group["comparison_valid"].astype(bool) if valid else ~group["comparison_valid"].astype(bool)
        counts.setdefault(str(event_type), {})[str(neural_family)] = int(mask.sum())
    return counts


def _compute_lag_null_summaries(
    *,
    riskset_table: pd.DataFrame,
    event_type: str,
    neural_config: NeuralHazardConfig,
    lag_window: NeuralLagWindow,
    n_permutations: int,
) -> pd.DataFrame:
    event_column = f"event_{str(event_type).lower()}"
    baseline_formula = _build_neural_baseline_formula(event_column=event_column, neural_config=neural_config)
    family_to_terms = {
        "alpha": _discover_band_pc_terms(riskset_table, band_name="alpha"),
        "beta": _discover_band_pc_terms(riskset_table, band_name="beta"),
    }
    family_to_terms["alpha_beta"] = [*family_to_terms["alpha"], *family_to_terms["beta"]]
    rows: list[dict[str, object]] = []
    for family, family_terms in family_to_terms.items():
        LOGGER.info(
            "Preparing lag-specific null summary for event_type=%s neural_family=%s lag=%s",
            event_type,
            family,
            lag_window.lag_label,
        )
        child_formula = _append_terms(baseline_formula, family_terms)
        required_columns = _extract_formula_columns(child_formula, event_column=event_column)
        subset = _subset_complete_cases(riskset_table, required_columns=required_columns)
        rows.append(
            _compute_family_lag_null_summary(
                subset=subset,
                event_type=event_type,
                event_column=event_column,
                neural_family=family,
                family_terms=family_terms,
                baseline_formula=baseline_formula,
                child_formula=child_formula,
                lag_window=lag_window,
                n_permutations=n_permutations,
            )
        )
    return pd.DataFrame(rows)


def _compute_family_lag_null_summary(
    *,
    subset: pd.DataFrame,
    event_type: str,
    event_column: str,
    neural_family: str,
    family_terms: list[str],
    baseline_formula: str,
    child_formula: str,
    lag_window: NeuralLagWindow,
    n_permutations: int,
) -> dict[str, object]:
    suffix = str(event_type).upper()
    parent_name = f"M_behaviour_{suffix}_{neural_family}_sample_lag_{lag_window.lag_label}"
    child_name = (
        f"M_alpha_beta_{suffix}_lag_{lag_window.lag_label}"
        if neural_family == "alpha_beta"
        else f"M_{neural_family}_{suffix}_lag_{lag_window.lag_label}"
    )
    parent_fit = _fit_formula_model(
        riskset_table=subset,
        model_name=parent_name,
        formula=baseline_formula,
        event_column=event_column,
    )
    child_fit = _fit_formula_model(
        riskset_table=subset,
        model_name=child_name,
        formula=child_formula,
        event_column=event_column,
    )
    real_row = _build_comparison_row(
        event_type=str(event_type).lower(),
        neural_family=neural_family,
        parent_fit=parent_fit,
        child_fit=child_fit,
        n_rows_candidate=int(len(subset)),
        n_events_candidate=int(pd.to_numeric(subset.get(event_column), errors="coerce").fillna(0).sum()),
        lag_window=lag_window,
    )
    if not bool(real_row["comparison_valid"]):
        LOGGER.warning(
            "Skipping null permutations for event_type=%s neural_family=%s lag=%s because the real comparison is invalid.",
            event_type,
            neural_family,
            lag_window.lag_label,
        )
        return {
            **{key: real_row.get(key) for key in ("event_type", "neural_family", "lag_start_ms", "lag_end_ms", "lag_label")},
            "real_delta_bic": real_row.get("delta_bic"),
            "null_median_delta_bic": None,
            "null_q025_delta_bic": None,
            "null_q975_delta_bic": None,
            "empirical_p_delta_bic": None,
            "n_permutations_successful": 0,
            "n_permutations_failed": int(n_permutations),
            "status": str(real_row["status"]),
        }
    rng = np.random.default_rng(int(lag_window.lag_start_ms * 1000 + lag_window.lag_end_ms))
    null_values: list[float] = []
    n_failed = 0
    for _ in progress_iterable(
        range(int(n_permutations)),
        total=int(n_permutations),
        description=f"null {event_type.upper()} {neural_family} {lag_window.lag_label}",
        enabled=True,
    ):
        shifted = subset.copy()
        shifted.loc[:, family_terms] = _circular_shift_feature_block(
            shifted.loc[:, family_terms],
            grouping_table=shifted,
            rng=rng,
        )
        shifted_child = _fit_formula_model(
            riskset_table=shifted,
            model_name=child_name,
            formula=child_formula,
            event_column=event_column,
        )
        shifted_row = _build_comparison_row(
            event_type=str(event_type).lower(),
            neural_family=neural_family,
            parent_fit=parent_fit,
            child_fit=shifted_child,
            n_rows_candidate=int(len(shifted)),
            n_events_candidate=int(pd.to_numeric(shifted.get(event_column), errors="coerce").fillna(0).sum()),
            lag_window=lag_window,
        )
        if bool(shifted_row["comparison_valid"]) and shifted_row["delta_bic"] is not None:
            null_values.append(float(shifted_row["delta_bic"]))
        else:
            n_failed += 1
    null_array = np.asarray(null_values, dtype=float)
    LOGGER.info(
        "Finished null summary for event_type=%s neural_family=%s lag=%s: successful=%d failed=%d",
        event_type,
        neural_family,
        lag_window.lag_label,
        int(len(null_values)),
        int(n_failed),
    )
    return {
        "event_type": str(event_type).lower(),
        "neural_family": neural_family,
        "lag_start_ms": int(lag_window.lag_start_ms),
        "lag_end_ms": int(lag_window.lag_end_ms),
        "lag_label": lag_window.lag_label,
        "real_delta_bic": real_row.get("delta_bic"),
        "null_median_delta_bic": _nanquantile(null_array, 0.50),
        "null_q025_delta_bic": _nanquantile(null_array, 0.025),
        "null_q975_delta_bic": _nanquantile(null_array, 0.975),
        "empirical_p_delta_bic": compute_neural_empirical_p_value(real_row.get("delta_bic"), null_array),
        "n_permutations_successful": int(len(null_values)),
        "n_permutations_failed": int(n_failed),
        "status": "ok" if len(null_values) > 0 else "failed",
    }


def _circular_shift_feature_block(
    feature_frame: pd.DataFrame,
    *,
    grouping_table: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    shifted = feature_frame.copy()
    group_columns = [name for name in ("participant_speaker_id", "run") if name in grouping_table.columns]
    if not group_columns:
        group_columns = ["run"] if "run" in grouping_table.columns else []
    grouped_rows = (
        grouping_table.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [(None, grouping_table)]
    )
    for _, rows in grouped_rows:
        row_index = rows.index
        if len(row_index) <= 1:
            continue
        values = shifted.loc[row_index].to_numpy(dtype=float, copy=True)
        offset = int(rng.integers(1, len(row_index)))
        shifted.loc[row_index] = np.roll(values, shift=offset, axis=0)
    return shifted


def compute_neural_empirical_p_value(real_delta_bic: object, null_delta_bic: np.ndarray) -> float | None:
    real_value = _maybe_float(real_delta_bic)
    if real_value is None:
        return None
    finite = np.asarray(null_delta_bic, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    count = int(np.count_nonzero(finite <= real_value))
    return float((1 + count) / (1 + finite.size))


def _nanquantile(values: np.ndarray, quantile: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.nanquantile(finite, quantile))


def _empty_null_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "neural_family",
            "lag_start_ms",
            "lag_end_ms",
            "lag_label",
            "real_delta_bic",
            "null_median_delta_bic",
            "null_q025_delta_bic",
            "null_q975_delta_bic",
            "empirical_p_delta_bic",
            "n_permutations_successful",
            "n_permutations_failed",
            "status",
        ]
    )


def fit_event_neural_comparisons(
    *,
    riskset_table: pd.DataFrame,
    event_type: str,
    neural_config: NeuralHazardConfig,
    lag_window: NeuralLagWindow | None = None,
    fit_model_fn: Callable[..., FittedFormulaModel] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """Fit same-row parent/child neural comparisons for one event type."""

    event_text = str(event_type).lower()
    if event_text not in {"fpp", "spp"}:
        raise ValueError("`event_type` must be one of fpp or spp.")
    event_column = f"event_{event_text}"
    if event_column not in riskset_table.columns:
        raise ValueError(f"Risk set is missing required event column `{event_column}`.")

    fit_fn = fit_model_fn or _fit_formula_model
    baseline_formula = _build_neural_baseline_formula(event_column=event_column, neural_config=neural_config)
    alpha_terms = _discover_band_pc_terms(riskset_table, band_name="alpha")
    beta_terms = _discover_band_pc_terms(riskset_table, band_name="beta")
    suffix = event_text.upper()
    family_to_terms = {
        "alpha": alpha_terms,
        "beta": beta_terms,
        "alpha_beta": [*alpha_terms, *beta_terms],
    }

    comparison_rows: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    counts_by_family: dict[str, dict[str, int]] = {}
    for family, family_terms in family_to_terms.items():
        model_stem = f"M_{family}_{suffix}" if family != "alpha_beta" else f"M_alpha_beta_{suffix}"
        parent_stem = f"M_behaviour_{suffix}_{family}_sample"
        child_model_name = (
            f"{model_stem}_lag_{lag_window.lag_label}" if lag_window is not None else model_stem
        )
        parent_model_name = (
            f"{parent_stem}_lag_{lag_window.lag_label}" if lag_window is not None else parent_stem
        )
        child_formula = _append_terms(baseline_formula, family_terms)
        required_columns = _extract_formula_columns(child_formula, event_column=event_column)
        subset = _subset_complete_cases(riskset_table, required_columns=required_columns)
        n_rows_candidate = int(len(subset))
        n_events_candidate = int(subset[event_column].sum()) if event_column in subset.columns else 0
        counts_by_family[family] = {"n_rows": n_rows_candidate, "n_events": n_events_candidate}
        LOGGER.info(
            "Fitting event=%s family=%s on neural-complete rows: n_rows=%d n_events=%d",
            event_text,
            family,
            n_rows_candidate,
            n_events_candidate,
        )

        try:
            parent_fit = fit_fn(
                riskset_table=subset,
                model_name=parent_model_name,
                formula=baseline_formula,
                event_column=event_column,
            )
            child_fit = fit_fn(
                riskset_table=subset,
                model_name=child_model_name,
                formula=child_formula,
                event_column=event_column,
            )
        except Exception as error:  # pragma: no cover - defensive guard for injected fit fns.
            row = _failed_comparison_row(
                event_type=event_text,
                neural_family=family,
                parent_model=parent_model_name,
                child_model=child_model_name,
                n_rows_candidate=n_rows_candidate,
                n_events_candidate=n_events_candidate,
                error_message=str(error),
                fit_warnings=[],
                lag_window=lag_window,
            )
            comparison_rows.append(row)
            continue

        row = _build_comparison_row(
            event_type=event_text,
            neural_family=family,
            parent_fit=parent_fit,
            child_fit=child_fit,
            n_rows_candidate=n_rows_candidate,
            n_events_candidate=n_events_candidate,
            lag_window=lag_window,
        )
        comparison_rows.append(row)
        if child_fit.success:
            coefficient_frames.append(
                _extract_coefficients(
                    event_type=event_text,
                    neural_family=family,
                    fitted=child_fit,
                )
            )

    comparison_output = pd.DataFrame(comparison_rows)
    coefficients_output = (
        pd.concat(coefficient_frames, ignore_index=True, sort=False)
        if coefficient_frames
        else pd.DataFrame(
            columns=[
                "event_type",
                "neural_family",
                "model_name",
                "term",
                "estimate",
                "std_error",
                "z_value",
                "p_value",
                "conf_low",
                "conf_high",
                "odds_ratio",
            ]
        )
    )
    return comparison_output, coefficients_output, counts_by_family


def _build_comparison_row(
    *,
    event_type: str,
    neural_family: str,
    parent_fit: FittedFormulaModel,
    child_fit: FittedFormulaModel,
    n_rows_candidate: int,
    n_events_candidate: int,
    lag_window: NeuralLagWindow | None = None,
) -> dict[str, object]:
    same_rows = bool(
        parent_fit.n_rows == child_fit.n_rows
        and parent_fit.n_events == child_fit.n_events
        and tuple(parent_fit.row_ids) == tuple(child_fit.row_ids)
    )
    parent_aic = _maybe_float(getattr(parent_fit.result, "aic", None)) if parent_fit.success else None
    child_aic = _maybe_float(getattr(child_fit.result, "aic", None)) if child_fit.success else None
    parent_bic = _maybe_float(getattr(parent_fit.result, "bic", None)) if parent_fit.success else None
    child_bic = _maybe_float(getattr(child_fit.result, "bic", None)) if child_fit.success else None
    parent_ll = _maybe_float(getattr(parent_fit.result, "llf", None)) if parent_fit.success else None
    child_ll = _maybe_float(getattr(child_fit.result, "llf", None)) if child_fit.success else None
    df_difference = (
        int(child_fit.n_predictors - parent_fit.n_predictors)
        if parent_fit.success and child_fit.success
        else None
    )
    converged = bool(parent_fit.converged and child_fit.converged)
    comparison_valid = bool(
        same_rows
        and parent_fit.success
        and child_fit.success
        and converged
        and parent_aic is not None
        and child_aic is not None
        and parent_bic is not None
        and child_bic is not None
        and parent_ll is not None
        and child_ll is not None
    )
    lrt_statistic = None
    p_value = None
    if comparison_valid and df_difference is not None and df_difference > 0:
        lrt_statistic = float(2.0 * (child_ll - parent_ll))
        p_value = float(stats.chi2.sf(lrt_statistic, df=df_difference))

    status = "ok" if comparison_valid else ("invalid_rows" if not same_rows else "failed_convergence")
    fit_warnings = [*parent_fit.fit_warnings, *child_fit.fit_warnings]
    error_message = child_fit.error_message or parent_fit.error_message
    row = {
        "event_type": event_type,
        "neural_family": neural_family,
        "lag_start_ms": None if lag_window is None else int(lag_window.lag_start_ms),
        "lag_end_ms": None if lag_window is None else int(lag_window.lag_end_ms),
        "lag_label": None if lag_window is None else lag_window.lag_label,
        "parent_model": parent_fit.model_name,
        "child_model": child_fit.model_name,
        "comparison": f"{child_fit.model_name} vs {parent_fit.model_name}",
        "status": status,
        "comparison_valid": comparison_valid,
        "same_rows": same_rows,
        "parent_aic": parent_aic,
        "child_aic": child_aic,
        "delta_aic": (child_aic - parent_aic) if comparison_valid else None,
        "parent_bic": parent_bic,
        "child_bic": child_bic,
        "delta_bic": (child_bic - parent_bic) if comparison_valid else None,
        "parent_log_likelihood": parent_ll,
        "child_log_likelihood": child_ll,
        "lrt_statistic": lrt_statistic if comparison_valid else None,
        "df_difference": df_difference,
        "p_value": p_value if comparison_valid else None,
        "n_rows_parent": int(parent_fit.n_rows),
        "n_rows_child": int(child_fit.n_rows),
        "n_events_parent": int(parent_fit.n_events),
        "n_events_child": int(child_fit.n_events),
        "n_predictors_parent": int(parent_fit.n_predictors),
        "n_predictors_child": int(child_fit.n_predictors),
        "n_rows_parent_candidate": int(n_rows_candidate),
        "n_rows_child_candidate": int(n_rows_candidate),
        "n_events_parent_candidate": int(n_events_candidate),
        "n_events_child_candidate": int(n_events_candidate),
        "fit_warnings": "; ".join(str(value) for value in fit_warnings if value),
        "error_message": error_message,
    }
    return row


def _failed_comparison_row(
    *,
    event_type: str,
    neural_family: str,
    parent_model: str,
    child_model: str,
    n_rows_candidate: int,
    n_events_candidate: int,
    error_message: str,
    fit_warnings: list[str],
    lag_window: NeuralLagWindow | None = None,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "neural_family": neural_family,
        "lag_start_ms": None if lag_window is None else int(lag_window.lag_start_ms),
        "lag_end_ms": None if lag_window is None else int(lag_window.lag_end_ms),
        "lag_label": None if lag_window is None else lag_window.lag_label,
        "parent_model": parent_model,
        "child_model": child_model,
        "comparison": f"{child_model} vs {parent_model}",
        "status": "failed_convergence",
        "comparison_valid": False,
        "same_rows": False,
        "parent_aic": None,
        "child_aic": None,
        "delta_aic": None,
        "parent_bic": None,
        "child_bic": None,
        "delta_bic": None,
        "parent_log_likelihood": None,
        "child_log_likelihood": None,
        "lrt_statistic": None,
        "df_difference": None,
        "p_value": None,
        "n_rows_parent": None,
        "n_rows_child": None,
        "n_events_parent": None,
        "n_events_child": None,
        "n_predictors_parent": None,
        "n_predictors_child": None,
        "n_rows_parent_candidate": int(n_rows_candidate),
        "n_rows_child_candidate": int(n_rows_candidate),
        "n_events_parent_candidate": int(n_events_candidate),
        "n_events_child_candidate": int(n_events_candidate),
        "fit_warnings": "; ".join(str(value) for value in fit_warnings if value),
        "error_message": error_message,
    }


def _fit_formula_model(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    event_column: str,
) -> FittedFormulaModel:
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
            row_ids=tuple(),
        )
    if int(riskset_table[event_column].sum()) <= 0:
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
            row_ids=_build_row_ids(riskset_table),
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted = sm.GLM.from_formula(
                formula=formula,
                data=riskset_table,
                family=sm.families.Binomial(),
            ).fit()
        except Exception as error:
            return FittedFormulaModel(
                model_name=model_name,
                formula=formula,
                result=None,
                n_rows=int(len(riskset_table)),
                n_events=int(riskset_table[event_column].sum()),
                n_predictors=0,
                converged=False,
                fit_warnings=[str(item.message) for item in caught],
                error_message=str(error),
                row_ids=_build_row_ids(riskset_table),
            )
    fit_warnings = [str(item.message) for item in caught]
    converged = bool(getattr(fitted, "converged", True))
    numeric_ok = (
        _maybe_float(getattr(fitted, "aic", None)) is not None
        and _maybe_float(getattr(fitted, "bic", None)) is not None
        and _maybe_float(getattr(fitted, "llf", None)) is not None
    )
    warning_text = " ".join(fit_warnings).lower()
    pathological = ("overflow encountered in exp" in warning_text) or ("perfect separation" in warning_text)
    if not numeric_ok:
        fit_warnings.append("Model returned non-finite AIC/BIC/LLF.")
    converged = bool(converged and numeric_ok and not pathological)
    error_message = None if converged else "Model failed convergence checks."
    return FittedFormulaModel(
        model_name=model_name,
        formula=formula,
        result=fitted,
        n_rows=int(len(riskset_table)),
        n_events=int(riskset_table[event_column].sum()),
        n_predictors=int(len(getattr(fitted, "params", []))),
        converged=converged,
        fit_warnings=fit_warnings,
        error_message=error_message,
        row_ids=_build_row_ids(riskset_table),
    )


def _subset_complete_cases(table: pd.DataFrame, *, required_columns: list[str]) -> pd.DataFrame:
    missing = [column_name for column_name in required_columns if column_name not in table.columns]
    if missing:
        raise ValueError("Required columns are missing from riskset: " + ", ".join(sorted(missing)))
    working = table.copy()
    for column_name in required_columns:
        if column_name in {"episode_id", "dyad_id", "participant_id"}:
            continue
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    mask = np.ones(len(working), dtype=bool)
    for column_name in required_columns:
        if column_name in {"episode_id", "dyad_id", "participant_id"}:
            mask &= working[column_name].notna().to_numpy()
        else:
            mask &= np.isfinite(working[column_name].to_numpy(dtype=float))
    subset = working.loc[mask].copy()
    if "event_fpp" in subset.columns:
        subset["event_fpp"] = subset["event_fpp"].astype(int)
    if "event_spp" in subset.columns:
        subset["event_spp"] = subset["event_spp"].astype(int)
    return subset


def _build_row_ids(table: pd.DataFrame) -> tuple[str, ...]:
    columns = [name for name in ("episode_id", "dyad_id", "run", "participant_speaker_id", "participant_id", "bin_end") if name in table.columns]
    if not columns:
        return tuple(str(index) for index in table.index)
    parts = [table.index.to_series().astype(str)]
    for column_name in columns:
        parts.append(table[column_name].astype(str))
    row_ids = pd.concat(parts, axis=1).agg("|".join, axis=1)
    return tuple(row_ids.tolist())


def _extract_coefficients(*, event_type: str, neural_family: str, fitted: FittedFormulaModel) -> pd.DataFrame:
    if fitted.result is None:
        return pd.DataFrame()
    conf = fitted.result.conf_int()
    frame = pd.DataFrame(
        {
            "event_type": event_type,
            "neural_family": neural_family,
            "model_name": fitted.model_name,
            "term": fitted.result.params.index.astype(str),
            "estimate": np.asarray(fitted.result.params, dtype=float),
            "std_error": np.asarray(fitted.result.bse, dtype=float),
            "z_value": np.asarray(fitted.result.tvalues, dtype=float),
            "p_value": np.asarray(fitted.result.pvalues, dtype=float),
            "conf_low": np.asarray(conf.iloc[:, 0], dtype=float),
            "conf_high": np.asarray(conf.iloc[:, 1], dtype=float),
        }
    )
    frame["odds_ratio"] = np.exp(frame["estimate"])
    return frame


def _build_neural_baseline_formula(*, event_column: str, neural_config: NeuralHazardConfig) -> str:
    information_rate = f"z_information_rate_lag_{int(neural_config.model.information_rate_lag_ms)}ms"
    prop_expected = f"z_prop_expected_cumulative_info_lag_{int(neural_config.model.prop_expected_lag_ms)}ms"
    return (
        f"{event_column} ~ time_from_partner_onset + time_from_partner_offset"
        f" + I(time_from_partner_offset ** 2) + {information_rate} + {prop_expected}"
    )


def _discover_band_pc_terms(riskset_table: pd.DataFrame, *, band_name: str) -> list[str]:
    prefix = f"z_{band_name}_pc"
    terms = sorted(column_name for column_name in riskset_table.columns if str(column_name).startswith(prefix))
    if not terms:
        raise ValueError(f"No PCA predictors were found for `{band_name}`.")
    return terms


def _append_terms(formula: str, terms: list[str]) -> str:
    if not terms:
        return formula
    lhs, rhs = formula.split("~", 1)
    return f"{lhs.strip()} ~ {rhs.strip()} + " + " + ".join(terms)


def _extract_formula_columns(formula: str, *, event_column: str) -> list[str]:
    import re

    columns = {
        event_column,
        "time_from_partner_onset",
        "time_from_partner_offset",
        "dyad_id",
        "participant_id",
        "episode_id",
    }
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula)):
        if token.startswith("z_"):
            columns.add(token)
    return sorted(columns)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def plot_neural_lowlevel_delta_bic_by_lag(
    comparison_table: pd.DataFrame,
    selected_table: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    working = comparison_table.loc[
        comparison_table["comparison_valid"].astype(bool)
        & (comparison_table["event_type"].astype(str) == "fpp")
    ].copy()
    working["delta_bic"] = pd.to_numeric(working["delta_bic"], errors="coerce")
    working["lag_start_ms"] = pd.to_numeric(working["lag_start_ms"], errors="coerce")
    working["lag_end_ms"] = pd.to_numeric(working["lag_end_ms"], errors="coerce")
    working = working.loc[
        np.isfinite(working["delta_bic"])
        & np.isfinite(working["lag_start_ms"])
        & np.isfinite(working["lag_end_ms"])
    ].copy()
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    if working.empty:
        axis.text(0.5, 0.5, "No valid FPP lag comparisons available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        colors = {"alpha": "#3b7ea1", "beta": "#7fa650", "alpha_beta": "#d17c4b"}
        working["lag_midpoint_ms"] = (working["lag_start_ms"] + working["lag_end_ms"]) / 2.0
        for family in ("alpha", "beta", "alpha_beta"):
            family_rows = working.loc[working["neural_family"].astype(str) == family].sort_values("lag_midpoint_ms")
            if family_rows.empty:
                continue
            axis.plot(
                family_rows["lag_midpoint_ms"],
                family_rows["delta_bic"],
                marker="o",
                color=colors[family],
                label=family,
            )
        selected = selected_table.loc[
            selected_table["comparison_valid"].astype(bool)
            & (selected_table["event_type"].astype(str) == "fpp")
        ].copy()
        if not selected.empty:
            selected["lag_midpoint_ms"] = (
                pd.to_numeric(selected["lag_start_ms"], errors="coerce")
                + pd.to_numeric(selected["lag_end_ms"], errors="coerce")
            ) / 2.0
            selected["delta_bic"] = pd.to_numeric(selected["delta_bic"], errors="coerce")
            for _, row in selected.iterrows():
                axis.scatter(
                    [row["lag_midpoint_ms"]],
                    [row["delta_bic"]],
                    s=95,
                    facecolors="none",
                    edgecolors=colors.get(str(row["neural_family"]), "#222222"),
                    linewidths=2.0,
                )
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_xlabel("neural lag midpoint (ms)")
        axis.set_ylabel("delta_bic")
        axis.set_title("Neural lag selection: delta_bic = child - parent; lower is better")
        axis.legend(title="neural_family")
    output_path = figures_dir / "neural_lowlevel_delta_bic_by_lag.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def plot_neural_lowlevel_lag_null_comparison(
    null_summary: pd.DataFrame,
    selected_table: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_table.loc[
        selected_table["comparison_valid"].astype(bool)
        & (selected_table["event_type"].astype(str) == "fpp")
    ].copy()
    merged = pd.DataFrame()
    if not selected.empty and not null_summary.empty:
        merged = selected.merge(
            null_summary,
            on=["event_type", "neural_family", "lag_start_ms", "lag_end_ms", "lag_label"],
            how="left",
        )
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if merged.empty:
        axis.text(0.5, 0.5, "No valid selected-lag null summaries available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        order = ["alpha", "beta", "alpha_beta"]
        working = merged.copy()
        working["x"] = [order.index(value) if value in order else idx for idx, value in enumerate(working["neural_family"].astype(str))]
        working["real_delta_bic"] = pd.to_numeric(working["real_delta_bic"], errors="coerce")
        working["null_median_delta_bic"] = pd.to_numeric(working["null_median_delta_bic"], errors="coerce")
        working["null_q025_delta_bic"] = pd.to_numeric(working["null_q025_delta_bic"], errors="coerce")
        working["null_q975_delta_bic"] = pd.to_numeric(working["null_q975_delta_bic"], errors="coerce")
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        for _, row in working.iterrows():
            axis.vlines(row["x"], row["null_q025_delta_bic"], row["null_q975_delta_bic"], color="#91a6c6", linewidth=4.0)
            axis.scatter([row["x"]], [row["null_median_delta_bic"]], color="#43658b", s=45, zorder=3)
            axis.scatter([row["x"]], [row["real_delta_bic"]], color="#bc4b51", s=60, zorder=4)
        axis.set_xticks(working["x"])
        axis.set_xticklabels(working["neural_family"].astype(str))
        axis.set_xlabel("neural_family")
        axis.set_ylabel("delta_bic")
        axis.set_title("Selected neural lag null comparison")
    output_path = figures_dir / "neural_lowlevel_lag_null_comparison.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def plot_neural_lowlevel_selected_model_comparison(selected_table: pd.DataFrame, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotted = selected_table.loc[
        selected_table["comparison_valid"].astype(bool)
        & (selected_table["event_type"].astype(str) == "fpp")
    ].copy()
    plotted["delta_bic"] = pd.to_numeric(plotted["delta_bic"], errors="coerce")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    if plotted.empty:
        axis.text(0.5, 0.5, "No valid selected-lag FPP comparisons available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        order = ["alpha", "beta", "alpha_beta"]
        heights = [
            float(plotted.loc[plotted["neural_family"].astype(str) == family, "delta_bic"].iloc[0])
            if not plotted.loc[plotted["neural_family"].astype(str) == family].empty
            else np.nan
            for family in order
        ]
        axis.bar(order, heights, color=["#3b7ea1", "#7fa650", "#d17c4b"])
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_xlabel("neural_family")
        axis.set_ylabel("delta_bic")
        axis.set_title("Selected neural model comparison")
    output_path = figures_dir / "neural_lowlevel_selected_model_comparison.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def plot_neural_lowlevel_model_comparison(comparison_table: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    """Plot delta-BIC for valid same-row comparisons only."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    valid = comparison_table.loc[comparison_table["comparison_valid"].astype(bool)].copy()
    valid["delta_bic"] = pd.to_numeric(valid["delta_bic"], errors="coerce")
    valid = valid.loc[np.isfinite(valid["delta_bic"])].copy()
    plotted = valid.loc[valid["event_type"].astype(str) == "fpp"].copy()

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if plotted.empty:
        axis.text(0.5, 0.5, "No valid same-row comparisons available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        order = ["alpha", "beta", "alpha_beta"]
        bars = []
        heights = []
        for family in order:
            subset = plotted.loc[plotted["neural_family"].astype(str) == family, "delta_bic"]
            if subset.empty:
                continue
            bars.append(family)
            heights.append(float(subset.iloc[0]))
        axis.bar(bars, heights, color=["#3b7ea1", "#7fa650", "#d17c4b"][: len(bars)])
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_ylabel("delta_bic (child - parent)")
        axis.set_xlabel("neural_family")
        axis.set_title("Neural Low-Level Model Comparison (FPP, valid same-row only)")
    figure.tight_layout()
    figure.savefig(figures_dir / "neural_lowlevel_model_comparison.png", dpi=300)
    plt.close(figure)
    return plotted


def _plot_legacy_delta_aliases(valid_fpp: pd.DataFrame, figures_dir: Path) -> None:
    """Keep legacy delta figure filenames while plotting only valid rows."""

    for metric in ("delta_bic", "delta_aic"):
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        subset = valid_fpp.copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.loc[np.isfinite(subset[metric])].copy()
        if subset.empty:
            axis.text(0.5, 0.5, "No valid same-row comparisons available.", ha="center", va="center")
            axis.set_axis_off()
        else:
            order = ["alpha", "beta", "alpha_beta"]
            bars = []
            heights = []
            for family in order:
                rows = subset.loc[subset["neural_family"].astype(str) == family, metric]
                if rows.empty:
                    continue
                bars.append(family)
                heights.append(float(rows.iloc[0]))
            axis.bar(bars, heights, color=["#3b7ea1", "#7fa650", "#d17c4b"][: len(bars)])
            axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
            axis.set_ylabel(f"{metric} (child - parent)")
            axis.set_xlabel("neural_family")
            axis.set_title(f"FPP {metric} (valid same-row comparisons)")
        figure.tight_layout()
        figure.savefig(figures_dir / f"neural_{metric}_fpp_vs_spp.png", dpi=300)
        plt.close(figure)


def plot_neural_lowlevel_coefficients(coefficients_table: pd.DataFrame, figures_dir: Path) -> None:
    """Plot neural coefficient estimates for converged models."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    subset = coefficients_table.loc[
        coefficients_table["term"].astype(str).str.startswith("z_alpha_pc")
        | coefficients_table["term"].astype(str).str.startswith("z_beta_pc")
    ].copy()
    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    if subset.empty:
        axis.text(0.5, 0.5, "No converged neural coefficient estimates available.", ha="center", va="center")
        axis.set_axis_off()
    else:
        x_labels = [f"{event}-{term}" for event, term in zip(subset["event_type"], subset["term"], strict=True)]
        axis.bar(np.arange(len(subset)), pd.to_numeric(subset["estimate"], errors="coerce"), color="#3b7ea1")
        axis.set_xticks(np.arange(len(subset)))
        axis.set_xticklabels(x_labels, rotation=65, ha="right")
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_ylabel("Coefficient estimate")
        axis.set_title("Neural PC coefficients (converged models only)")
    figure.tight_layout()
    figure.savefig(figures_dir / "neural_coefficients_fpp_vs_spp.png", dpi=300)
    plt.close(figure)


def plot_neural_lowlevel_power_qc(hazard_paths: dict[str, Path], figures_dir: Path) -> None:
    """Plot basic neural power QC scatter for available risksets."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for path in hazard_paths.values():
        if path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True, sort=False)
    available_columns = [column_name for column_name in ("z_alpha_pc1", "z_beta_pc1") if column_name in combined.columns]
    figure, axes = plt.subplots(2, max(1, len(available_columns)), figsize=(5.5 * max(1, len(available_columns)), 7.0))
    if len(available_columns) == 0:
        axes[0].text(0.5, 0.5, "No PC columns found.", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    else:
        if len(available_columns) == 1:
            axes = np.array([[axes[0]], [axes[1]]], dtype=object)
        for column_index, predictor in enumerate(available_columns):
            for row_index, x_column in enumerate(("time_from_partner_onset", "time_from_partner_offset")):
                axis = axes[row_index, column_index]
                x_values = pd.to_numeric(combined[x_column], errors="coerce")
                y_values = pd.to_numeric(combined[predictor], errors="coerce")
                mask = np.isfinite(x_values) & np.isfinite(y_values)
                axis.scatter(x_values[mask], y_values[mask], s=5, alpha=0.15, color="#3b7ea1")
                axis.set_xlabel(x_column)
                axis.set_ylabel(predictor)
                axis.set_title(f"{predictor} vs {x_column}")
    figure.tight_layout()
    figure.savefig(figures_dir / "neural_power_by_partner_time.png", dpi=300)
    plt.close(figure)
