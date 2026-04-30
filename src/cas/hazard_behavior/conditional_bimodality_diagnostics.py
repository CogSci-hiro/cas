"""Conditional-vs-marginal bimodality diagnostics for latency-regime analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr, t

from cas.hazard_behavior.plot_latency_regime import (
    MODEL_C,
    MODEL_LABELS,
    MODEL_R1,
    MODEL_R2,
    MODEL_R3,
    MODEL_R4,
    _choose_best_model,
    density_per_second_to_per_millisecond,
    milliseconds_to_seconds,
    seconds_to_milliseconds,
    student_t_density_per_ms,
)

LATENCY_MS_PER_SECOND = 1000.0
DEFAULT_FIXED_PREDICTOR_VALUES = (-1.0, 0.0, 1.0)
MIN_EVENTS_PER_QUANTILE_BIN = 20
DEFAULT_NU = 4.0
PLACEHOLDER_DENSITY_COLUMNS = (
    "density_per_ms",
    "density_draw_mean",
    "density_draw_lower",
    "density_draw_upper",
)


@dataclass(frozen=True, slots=True)
class ConditionalBimodalityDiagnosticsResult:
    """Paths for the conditional-vs-marginal diagnostics bundle."""

    figures_dir: Path
    diagnostics_dir: Path
    report_path: Path


def run_latency_regime_conditional_bimodality_diagnostics(
    *,
    event_data_csv: Path,
    stan_results_dir: Path,
    figures_dir: Path,
    diagnostics_dir: Path,
    verbose: bool = False,
) -> ConditionalBimodalityDiagnosticsResult:
    """Generate conditional-vs-marginal bimodality diagnostics.

    Parameters
    ----------
    event_data_csv
        Event-only latency-regime input CSV.
    stan_results_dir
        Directory containing the event-only latency-regime model outputs.
    figures_dir
        Directory where diagnostic figures should be written.
    diagnostics_dir
        Directory where diagnostic tables and the report should be written.
    verbose
        Whether to print progress messages.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Reading event-only latency-regime data from {event_data_csv}.")
    event_data = pd.read_csv(event_data_csv).reset_index(drop=True)
    event_data["row_index"] = np.arange(len(event_data), dtype=int)
    event_data = _coerce_event_columns(event_data)

    inputs = _load_latency_regime_outputs(stan_results_dir=stan_results_dir)
    fit_metrics = inputs["fit_metrics"]
    nu = float((fit_metrics or {}).get("nu", DEFAULT_NU))

    best_models = choose_best_r_models(inputs["loo_table"])
    warnings_list: list[str] = []
    if best_models["best_student_t_r"] is None:
        warnings_list.append("No Student-t regression model was available in the LOO table.")
    if best_models["best_lognormal_r"] is None:
        warnings_list.append("No shifted-lognormal regression model was available in the LOO table.")

    fixed_conditions = _default_fixed_conditions()
    conditional_density = compute_fixed_predictor_conditional_densities(
        event_data=event_data,
        component_parameters=inputs["component_parameters"],
        gating_coefficients=inputs["gating_coefficients"],
        regression_coefficients=inputs["regression_coefficients"],
        shifted_lognormal_diagnostics=inputs["shifted_lognormal_diagnostics"],
        best_student_t_r=best_models["best_student_t_r"],
        best_lognormal_r=best_models["best_lognormal_r"],
        fit_metrics=fit_metrics,
        nu=nu,
        conditions=fixed_conditions,
    )
    conditional_density_path = diagnostics_dir / "latency_regime_conditional_density_fixed_predictors.csv"
    conditional_density.to_csv(conditional_density_path, index=False)
    _plot_conditional_density_fixed_predictors(
        conditional_density=conditional_density,
        output_path=figures_dir / "behaviour_latency_regime_conditional_density_fixed_predictors.png",
    )

    ppc_by_quantile = compute_ppc_by_predictor_quantile(
        event_data=event_data,
        ppc_table=inputs["ppc_table"],
        best_student_t_r=best_models["best_student_t_r"],
        best_lognormal_r=best_models["best_lognormal_r"],
    )
    ppc_by_quantile_path = diagnostics_dir / "latency_regime_ppc_by_predictor_quantile.csv"
    ppc_by_quantile.to_csv(ppc_by_quantile_path, index=False)
    _plot_ppc_by_predictor_quantile(
        density_table=ppc_by_quantile,
        predictor_name="z_prop_expected_cumulative_info_lag_best",
        output_path=figures_dir / "behaviour_latency_regime_ppc_by_expected_info_quantile.png",
        title="Posterior predictive by expected-info quantile",
        x_label="Latency from partner offset (ms)",
    )
    _plot_ppc_by_predictor_quantile(
        density_table=ppc_by_quantile,
        predictor_name="z_information_rate_lag_best",
        output_path=figures_dir / "behaviour_latency_regime_ppc_by_information_rate_quantile.png",
        title="Posterior predictive by information-rate quantile",
        x_label="Latency from partner offset (ms)",
    )

    regression_predictions = _compute_event_level_regression_predictions(
        event_data=event_data,
        regression_coefficients=inputs["regression_coefficients"],
        shifted_lognormal_diagnostics=inputs["shifted_lognormal_diagnostics"],
        nu=nu,
    )
    residual_table, residual_summary = compute_r_model_residuals(
        event_data=event_data,
        regression_predictions=regression_predictions,
    )
    residual_table_path = diagnostics_dir / "latency_regime_r_model_residuals.csv"
    residual_summary_path = diagnostics_dir / "latency_regime_r_model_residual_bimodality_summary.csv"
    residual_table.to_csv(residual_table_path, index=False)
    residual_summary.to_csv(residual_summary_path, index=False)
    _plot_r_model_residual_density(
        residual_table=residual_table,
        output_path=figures_dir / "behaviour_latency_regime_r_model_residual_density.png",
    )

    pointwise_elpd = compute_pointwise_elpd_differences(
        event_data=event_data,
        component_parameters=inputs["component_parameters"],
        event_probabilities=inputs["event_probabilities"],
        regression_predictions=regression_predictions,
        best_student_t_r=best_models["best_student_t_r"],
        best_lognormal_r=best_models["best_lognormal_r"],
        nu=nu,
    )
    pointwise_elpd_path = diagnostics_dir / "latency_regime_pointwise_elpd_differences.csv"
    pointwise_elpd.to_csv(pointwise_elpd_path, index=False)
    _plot_pointwise_difference(
        pointwise_table=pointwise_elpd,
        x_column="observed_latency_ms",
        output_path=figures_dir / "behaviour_latency_regime_pointwise_elpd_c_minus_r_by_latency.png",
        x_label="Observed latency (ms)",
    )
    _plot_pointwise_difference(
        pointwise_table=pointwise_elpd,
        x_column="z_prop_expected_cumulative_info_lag_best",
        output_path=figures_dir / "behaviour_latency_regime_pointwise_elpd_c_minus_r_by_expected_info.png",
        x_label="Expected cumulative information (z)",
    )
    _plot_pointwise_difference(
        pointwise_table=pointwise_elpd,
        x_column="z_information_rate_lag_best",
        output_path=figures_dir / "behaviour_latency_regime_pointwise_elpd_c_minus_r_by_information_rate.png",
        x_label="Information rate (z)",
    )
    _plot_pointwise_difference(
        pointwise_table=pointwise_elpd,
        x_column="p_late_model_c",
        output_path=figures_dir / "behaviour_latency_regime_pointwise_elpd_c_minus_r_by_p_late.png",
        x_label="Model C P(late)",
    )

    p_late_vs_r, p_late_corr = compare_p_late_with_r_predictions(
        event_data=event_data,
        event_probabilities=inputs["event_probabilities"],
        regression_predictions=regression_predictions,
        r_model_name=best_models["best_student_t_r"] or best_models["best_any_r"],
    )
    p_late_vs_r_path = diagnostics_dir / "latency_regime_p_late_vs_r_predictions.csv"
    p_late_corr_path = diagnostics_dir / "latency_regime_p_late_r_prediction_correlations.csv"
    p_late_vs_r.to_csv(p_late_vs_r_path, index=False)
    p_late_corr.to_csv(p_late_corr_path, index=False)
    _plot_p_late_vs_r_mu(
        merged=p_late_vs_r,
        output_path=figures_dir / "behaviour_latency_regime_p_late_vs_r_mu.png",
    )
    _plot_latency_vs_r_mu_coloured_by_p_late(
        merged=p_late_vs_r,
        output_path=figures_dir / "behaviour_latency_regime_latency_vs_r_mu_coloured_by_p_late.png",
    )
    _plot_latency_vs_predictor_coloured_by_p_late(
        merged=p_late_vs_r,
        predictor_column="z_information_rate_lag_best",
        predictor_label="Information rate (z)",
        output_path=figures_dir / "behaviour_latency_regime_latency_vs_information_rate_coloured_by_p_late.png",
    )
    _plot_latency_vs_predictor_coloured_by_p_late(
        merged=p_late_vs_r,
        predictor_column="z_prop_expected_cumulative_info_lag_best",
        predictor_label="Expected cumulative information (z)",
        output_path=figures_dir / "behaviour_latency_regime_latency_vs_expected_cum_info_coloured_by_p_late.png",
    )

    counterfactual_density = simulate_counterfactual_predictor_distributions(
        event_data=event_data,
        component_parameters=inputs["component_parameters"],
        gating_coefficients=inputs["gating_coefficients"],
        event_probabilities=inputs["event_probabilities"],
        regression_predictions=regression_predictions,
        regression_coefficients=inputs["regression_coefficients"],
        shifted_lognormal_diagnostics=inputs["shifted_lognormal_diagnostics"],
        best_student_t_r=best_models["best_student_t_r"],
        best_lognormal_r=best_models["best_lognormal_r"],
        fit_metrics=fit_metrics,
        nu=nu,
    )
    counterfactual_density_path = diagnostics_dir / "latency_regime_counterfactual_predictor_distribution.csv"
    counterfactual_density.to_csv(counterfactual_density_path, index=False)
    _plot_counterfactual_predictor_distribution(
        density_table=counterfactual_density,
        output_path=figures_dir / "behaviour_latency_regime_counterfactual_predictor_distribution.png",
    )

    report_path = diagnostics_dir / "latency_regime_conditional_vs_marginal_bimodality_report.md"
    report_path.write_text(
        write_conditional_vs_marginal_report(
            event_data=event_data,
            best_models=best_models,
            conditional_density=conditional_density,
            ppc_by_quantile=ppc_by_quantile,
            residual_summary=residual_summary,
            pointwise_differences=pointwise_elpd,
            p_late_correlations=p_late_corr,
            counterfactual_density=counterfactual_density,
            fit_metrics=fit_metrics,
            warnings_list=warnings_list,
        ),
        encoding="utf-8",
    )

    return ConditionalBimodalityDiagnosticsResult(
        figures_dir=figures_dir,
        diagnostics_dir=diagnostics_dir,
        report_path=report_path,
    )


def _load_latency_regime_outputs(stan_results_dir: Path) -> dict[str, object]:
    return {
        "component_parameters": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_component_parameters.csv"),
        "gating_coefficients": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_gating_coefficients.csv"),
        "event_probabilities": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_event_probabilities.csv"),
        "loo_table": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_loo_comparison.csv"),
        "ppc_table": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv"),
        "regression_coefficients": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_regression_coefficients.csv"),
        "regression_predictions": _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_regression_predictions.csv"),
        "shifted_lognormal_diagnostics": _read_csv_if_exists(
            stan_results_dir / "behaviour_latency_regime_shifted_lognormal_diagnostics.csv"
        ),
        "fit_metrics": _read_json_if_exists(stan_results_dir / "behaviour_latency_regime_fit_metrics.json"),
    }


def choose_best_r_models(loo_table: pd.DataFrame | None) -> dict[str, str | None]:
    """Choose the best available regression competitors from the LOO table."""

    best_student_t_r = _choose_best_model(loo_table, candidates=(MODEL_R1, MODEL_R2))
    best_lognormal_r = _choose_best_model(loo_table, candidates=(MODEL_R3, MODEL_R4))
    best_any_r = _choose_best_model(loo_table, candidates=(MODEL_R1, MODEL_R2, MODEL_R3, MODEL_R4))
    return {
        "best_student_t_r": best_student_t_r,
        "best_lognormal_r": best_lognormal_r,
        "best_any_r": best_any_r,
    }


def compute_fixed_predictor_conditional_densities(
    *,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    regression_coefficients: pd.DataFrame | None,
    shifted_lognormal_diagnostics: pd.DataFrame | None,
    best_student_t_r: str | None,
    best_lognormal_r: str | None,
    fit_metrics: dict[str, object] | None,
    nu: float,
    conditions: list[dict[str, object]],
) -> pd.DataFrame:
    """Compute model-implied densities at fixed predictor settings."""

    grid_ms = _build_latency_grid_ms(event_data["latency_from_partner_offset"].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    run_reference = _resolve_run_reference(fit_metrics)
    for condition in conditions:
        rate = float(condition["z_information_rate_lag_best"])
        expected = float(condition["z_prop_expected_cumulative_info_lag_best"])
        z_time = float(condition.get("z_time_within_run", 0.0))
        z_time2 = float(condition.get("z_time_within_run_squared", 0.0))
        features = {
            "z_information_rate_lag_best": rate,
            "z_prop_expected_cumulative_info_lag_best": expected,
            "z_time_within_run": z_time,
            "z_time_within_run_squared": z_time2,
            "run": run_reference,
        }
        c_density = _conditional_density_model_c(
            grid_ms=grid_ms,
            component_parameters=component_parameters,
            gating_coefficients=gating_coefficients,
            features=features,
            nu=nu,
        )
        rows.extend(
            _density_rows_for_condition(
                model_name=MODEL_C,
                condition_name=str(condition["condition_name"]),
                grid_ms=grid_ms,
                density_ms=c_density,
                features=features,
                run_reference=run_reference,
            )
        )
        if best_student_t_r is not None:
            r_density = _conditional_density_regression_model(
                model_name=best_student_t_r,
                grid_ms=grid_ms,
                regression_coefficients=regression_coefficients,
                shifted_lognormal_diagnostics=shifted_lognormal_diagnostics,
                features=features,
                nu=nu,
            )
            rows.extend(
                _density_rows_for_condition(
                    model_name=best_student_t_r,
                    condition_name=str(condition["condition_name"]),
                    grid_ms=grid_ms,
                    density_ms=r_density,
                    features=features,
                    run_reference=run_reference,
                )
            )
        if best_lognormal_r is not None:
            r_density = _conditional_density_regression_model(
                model_name=best_lognormal_r,
                grid_ms=grid_ms,
                regression_coefficients=regression_coefficients,
                shifted_lognormal_diagnostics=shifted_lognormal_diagnostics,
                features=features,
                nu=nu,
            )
            rows.extend(
                _density_rows_for_condition(
                    model_name=best_lognormal_r,
                    condition_name=str(condition["condition_name"]),
                    grid_ms=grid_ms,
                    density_ms=r_density,
                    features=features,
                    run_reference=run_reference,
                )
            )
    return pd.DataFrame(rows)


def compute_ppc_by_predictor_quantile(
    *,
    event_data: pd.DataFrame,
    ppc_table: pd.DataFrame | None,
    best_student_t_r: str | None,
    best_lognormal_r: str | None,
) -> pd.DataFrame:
    """Compute posterior predictive density summaries within predictor quantile bins."""

    rows: list[dict[str, object]] = []
    model_names = [MODEL_C]
    if best_student_t_r is not None:
        model_names.append(best_student_t_r)
    if best_lognormal_r is not None:
        model_names.append(best_lognormal_r)

    for predictor_name in ("z_prop_expected_cumulative_info_lag_best", "z_information_rate_lag_best"):
        quantile_table = _assign_predictor_quantile_bins(
            event_data=event_data,
            predictor_name=predictor_name,
        )
        if quantile_table.empty:
            continue
        for quantile_bin, subset in quantile_table.groupby("quantile_bin", sort=False):
            observed_values_ms = seconds_to_milliseconds(subset["latency_from_partner_offset"].to_numpy(dtype=float))
            grid_ms = _build_latency_grid_ms(subset["latency_from_partner_offset"].to_numpy(dtype=float))
            observed_density_ms = _safe_kde_density_per_ms(observed_values_ms, grid_ms)
            rows.extend(
                _density_rows_with_quantile_meta(
                    predictor_name=predictor_name,
                    quantile_subset=subset,
                    quantile_bin=str(quantile_bin),
                    model_name="observed",
                    grid_ms=grid_ms,
                    density_ms=observed_density_ms,
                )
            )
            for model_name in model_names:
                model_draw_densities = _ppc_draw_densities_for_bin(
                    ppc_table=ppc_table,
                    model_name=model_name,
                    row_indices=subset["row_index"].to_numpy(dtype=int),
                    grid_ms=grid_ms,
                )
                if model_draw_densities.size == 0:
                    density_ms = np.full_like(grid_ms, np.nan, dtype=float)
                    lower = density_ms.copy()
                    upper = density_ms.copy()
                else:
                    density_ms = np.nanmean(model_draw_densities, axis=0)
                    lower = np.nanquantile(model_draw_densities, 0.025, axis=0)
                    upper = np.nanquantile(model_draw_densities, 0.975, axis=0)
                rows.extend(
                    _density_rows_with_quantile_meta(
                        predictor_name=predictor_name,
                        quantile_subset=subset,
                        quantile_bin=str(quantile_bin),
                        model_name=model_name,
                        grid_ms=grid_ms,
                        density_ms=density_ms,
                        lower=lower,
                        upper=upper,
                    )
                )
    return pd.DataFrame(rows)


def compute_r_model_residuals(
    *,
    event_data: pd.DataFrame,
    regression_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute event-level regression residuals and bimodality summaries."""

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    working = regression_predictions.copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["model_name", "residual_type", "n_events", "n_detected_peaks", "peak_locations_ms", "notes"])
    merged = event_data.merge(working, on="row_index", how="inner", suffixes=("", "_pred"))
    for model_name, subset in merged.groupby("model_name", sort=False):
        observed_s = subset["latency_from_partner_offset"].to_numpy(dtype=float)
        mu_s = subset["predicted_mu_s"].to_numpy(dtype=float)
        sigma_s = subset["predicted_sigma_s"].to_numpy(dtype=float)
        residual_s = observed_s - mu_s
        sigma_safe = np.where(np.isfinite(sigma_s) & (sigma_s > 0.0), sigma_s, np.nan)
        standardised = residual_s / sigma_safe
        for idx, record in subset.iterrows():
            current_sigma_s = float(record["predicted_sigma_s"]) if pd.notna(record["predicted_sigma_s"]) else math.nan
            rows.append(
                {
                    "row_index": int(record["row_index"]),
                    "episode_id": record.get("episode_id"),
                    "model_name": model_name,
                    "observed_latency_s": float(record["latency_from_partner_offset"]),
                    "observed_latency_ms": float(record["latency_from_partner_offset"]) * LATENCY_MS_PER_SECOND,
                    "predicted_mu_s": float(record["predicted_mu_s"]),
                    "predicted_mu_ms": float(record["predicted_mu_s"]) * LATENCY_MS_PER_SECOND,
                    "predicted_sigma_s": current_sigma_s,
                    "residual_s": float(record["latency_from_partner_offset"] - record["predicted_mu_s"]),
                    "residual_ms": float(record["latency_from_partner_offset"] - record["predicted_mu_s"]) * LATENCY_MS_PER_SECOND,
                    "standardised_residual": (
                        float((record["latency_from_partner_offset"] - record["predicted_mu_s"]) / current_sigma_s)
                        if np.isfinite(current_sigma_s) and current_sigma_s > 0.0
                        else math.nan
                    ),
                    "z_information_rate_lag_best": float(record["z_information_rate_lag_best"]),
                    "z_prop_expected_cumulative_info_lag_best": float(record["z_prop_expected_cumulative_info_lag_best"]),
                    "run": record.get("run"),
                    "time_within_run": record.get("time_within_run", math.nan),
                }
            )
        summary_rows.append(_residual_peak_summary(model_name=model_name, residual_type="residual_ms", values_ms=residual_s * LATENCY_MS_PER_SECOND))
        if np.isfinite(standardised).any():
            summary_rows.append(
                _residual_peak_summary(
                    model_name=model_name,
                    residual_type="standardised_residual",
                    values_ms=standardised[np.isfinite(standardised)],
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def compute_pointwise_elpd_differences(
    *,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    regression_predictions: pd.DataFrame,
    best_student_t_r: str | None,
    best_lognormal_r: str | None,
    nu: float,
) -> pd.DataFrame:
    """Compute pointwise Model C minus R-model predictive fit differences.

    Notes
    -----
    This function uses pointwise log-likelihood under posterior-mean parameter
    summaries. If exact pointwise LOO contributions are unavailable, the output
    remains informative but should be described as pointwise log-likelihood
    differences rather than exact ELPD differences.
    """

    c_inputs = _model_c_event_inputs(
        event_data=event_data,
        component_parameters=component_parameters,
        event_probabilities=event_probabilities,
    )
    rows: list[dict[str, object]] = []
    comparison_models = [model_name for model_name in (best_student_t_r, best_lognormal_r) if model_name is not None]
    if c_inputs.empty or regression_predictions.empty or not comparison_models:
        return pd.DataFrame(
            columns=[
                "row_index",
                "episode_id",
                "comparison",
                "difference_metric",
                "model_left",
                "model_right",
                "elpd_difference",
                "observed_latency_s",
                "observed_latency_ms",
                "z_information_rate_lag_best",
                "z_prop_expected_cumulative_info_lag_best",
                "p_late_model_c",
                "r_model_predicted_mu_s",
                "r_model_predicted_mu_ms",
                "r_model_predicted_sigma_s",
                "run",
                "time_within_run",
            ]
        )
    regression_lookup = regression_predictions.loc[regression_predictions["model_name"].isin(comparison_models)].copy()
    merged = c_inputs.merge(regression_lookup, on="row_index", how="inner")
    for _, row in merged.iterrows():
        c_log_lik = _model_c_log_likelihood(
            y_s=float(row["latency_from_partner_offset"]),
            p_late=float(row["p_late_model_c"]),
            early_mu_s=float(row["early_mu_s"]),
            late_mu_s=float(row["late_mu_s"]),
            early_sigma_s=float(row["early_sigma_s"]),
            late_sigma_s=float(row["late_sigma_s"]),
            nu=nu,
        )
        r_log_lik = _regression_row_log_likelihood(
            y_s=float(row["latency_from_partner_offset"]),
            model_name=str(row["model_name"]),
            predicted_mu_s=float(row["predicted_mu_s"]),
            predicted_sigma_s=float(row["predicted_sigma_s"]) if pd.notna(row["predicted_sigma_s"]) else math.nan,
            predicted_mu_log=float(row["predicted_mu_log"]) if pd.notna(row["predicted_mu_log"]) else math.nan,
            predicted_sigma_log=float(row["predicted_sigma_log"]) if pd.notna(row["predicted_sigma_log"]) else math.nan,
            shift_seconds=float(row["shift_seconds"]) if pd.notna(row["shift_seconds"]) else math.nan,
            nu=nu,
        )
        rows.append(
            {
                "row_index": int(row["row_index"]),
                "episode_id": row.get("episode_id"),
                "comparison": f"{MODEL_C}_minus_{row['model_name']}",
                "difference_metric": "pointwise_log_lik_mean_difference",
                "model_left": MODEL_C,
                "model_right": row["model_name"],
                "elpd_difference": c_log_lik - r_log_lik,
                "observed_latency_s": float(row["latency_from_partner_offset"]),
                "observed_latency_ms": float(row["latency_from_partner_offset"]) * LATENCY_MS_PER_SECOND,
                "z_information_rate_lag_best": float(row["z_information_rate_lag_best"]),
                "z_prop_expected_cumulative_info_lag_best": float(row["z_prop_expected_cumulative_info_lag_best"]),
                "p_late_model_c": float(row["p_late_model_c"]),
                "r_model_predicted_mu_s": float(row["predicted_mu_s"]),
                "r_model_predicted_mu_ms": float(row["predicted_mu_s"]) * LATENCY_MS_PER_SECOND,
                "r_model_predicted_sigma_s": float(row["predicted_sigma_s"]) if pd.notna(row["predicted_sigma_s"]) else math.nan,
                "run": row.get("run"),
                "time_within_run": row.get("time_within_run", math.nan),
            }
        )
    return pd.DataFrame(rows)


def compare_p_late_with_r_predictions(
    *,
    event_data: pd.DataFrame,
    event_probabilities: pd.DataFrame | None,
    regression_predictions: pd.DataFrame,
    r_model_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge Model C P(late) with event-level regression predictions and correlations."""

    if event_probabilities is None or event_probabilities.empty or r_model_name is None:
        return pd.DataFrame(), pd.DataFrame(columns=["r_model_name", "metric", "correlation"])
    probs = event_probabilities.copy()
    probs["row_index"] = np.arange(len(probs), dtype=int)
    probs = probs.loc[probs["model_name"] == MODEL_C].copy()
    if probs.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["r_model_name", "metric", "correlation"])
    preds = regression_predictions.loc[regression_predictions["model_name"] == r_model_name].copy()
    merged = event_data.merge(probs[["row_index", "p_late_mean"]], on="row_index", how="left")
    merged = merged.merge(
        preds[["row_index", "model_name", "predicted_mu_s", "predicted_sigma_s"]],
        on="row_index",
        how="left",
    )
    merged = merged.rename(columns={"p_late_mean": "p_late_model_c", "model_name": "r_model_name"})
    merged["observed_latency_ms"] = merged["latency_from_partner_offset"] * LATENCY_MS_PER_SECOND
    merged["r_mu_s"] = merged["predicted_mu_s"]
    merged["r_mu_ms"] = merged["predicted_mu_s"] * LATENCY_MS_PER_SECOND
    merged["r_sigma_s"] = merged["predicted_sigma_s"]
    merged["time_within_run"] = pd.to_numeric(merged.get("time_within_run"), errors="coerce")
    merged = merged[
        [
            "row_index",
            "episode_id",
            "observed_latency_ms",
            "latency_from_partner_offset",
            "p_late_model_c",
            "r_model_name",
            "r_mu_s",
            "r_mu_ms",
            "r_sigma_s",
            "z_information_rate_lag_best",
            "z_prop_expected_cumulative_info_lag_best",
            "run",
            "time_within_run",
        ]
    ].rename(columns={"latency_from_partner_offset": "observed_latency_s"})
    corr_rows: list[dict[str, object]] = []
    valid_mu = merged[["p_late_model_c", "r_mu_s"]].dropna()
    if len(valid_mu) >= 2:
        corr_rows.append(
            {
                "r_model_name": r_model_name,
                "metric": "pearson_p_late_vs_r_mu",
                "correlation": float(pearsonr(valid_mu["p_late_model_c"], valid_mu["r_mu_s"]).statistic),
            }
        )
        corr_rows.append(
            {
                "r_model_name": r_model_name,
                "metric": "spearman_p_late_vs_r_mu",
                "correlation": float(spearmanr(valid_mu["p_late_model_c"], valid_mu["r_mu_s"]).statistic),
            }
        )
    valid_sigma = merged[["p_late_model_c", "r_sigma_s"]].dropna()
    if len(valid_sigma) >= 2 and valid_sigma["r_sigma_s"].nunique() > 1 and valid_sigma["p_late_model_c"].nunique() > 1:
        corr_rows.append(
            {
                "r_model_name": r_model_name,
                "metric": "pearson_p_late_vs_r_sigma",
                "correlation": float(pearsonr(valid_sigma["p_late_model_c"], valid_sigma["r_sigma_s"]).statistic),
            }
        )
        corr_rows.append(
            {
                "r_model_name": r_model_name,
                "metric": "spearman_p_late_vs_r_sigma",
                "correlation": float(spearmanr(valid_sigma["p_late_model_c"], valid_sigma["r_sigma_s"]).statistic),
            }
        )
    return merged, pd.DataFrame(corr_rows)


def simulate_counterfactual_predictor_distributions(
    *,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    regression_predictions: pd.DataFrame,
    regression_coefficients: pd.DataFrame | None,
    shifted_lognormal_diagnostics: pd.DataFrame | None,
    best_student_t_r: str | None,
    best_lognormal_r: str | None,
    fit_metrics: dict[str, object] | None,
    nu: float,
) -> pd.DataFrame:
    """Simulate pooled densities under observed versus fixed predictors."""

    grid_ms = _build_latency_grid_ms(event_data["latency_from_partner_offset"].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    models = [model_name for model_name in (MODEL_C, best_student_t_r, best_lognormal_r) if model_name is not None]
    if not models:
        return pd.DataFrame(columns=["model_name", "scenario", "latency_ms", "density_per_ms", "density_draw_mean", "density_draw_lower", "density_draw_upper", "notes"])
    fixed_features = {
        "z_information_rate_lag_best": 0.0,
        "z_prop_expected_cumulative_info_lag_best": 0.0,
        "z_time_within_run": 0.0,
        "z_time_within_run_squared": 0.0,
        "run": _resolve_run_reference(fit_metrics),
    }
    for model_name in models:
        if model_name == MODEL_C:
            observed_density = _event_averaged_model_c_density(
                event_data=event_data,
                component_parameters=component_parameters,
                gating_coefficients=gating_coefficients,
                event_probabilities=event_probabilities,
                grid_ms=grid_ms,
                use_event_probabilities=True,
                nu=nu,
            )
            fixed_density = _conditional_density_model_c(
                grid_ms=grid_ms,
                component_parameters=component_parameters,
                gating_coefficients=gating_coefficients,
                features=fixed_features,
                nu=nu,
            )
        else:
            observed_density = _event_averaged_regression_density(
                event_data=event_data,
                regression_predictions=regression_predictions,
                model_name=model_name,
                grid_ms=grid_ms,
                nu=nu,
            )
            fixed_density = _conditional_density_regression_model(
                model_name=model_name,
                grid_ms=grid_ms,
                regression_coefficients=regression_coefficients,
                shifted_lognormal_diagnostics=shifted_lognormal_diagnostics,
                features=fixed_features,
                nu=nu,
            )
        rows.extend(_scenario_density_rows(model_name=model_name, scenario="observed_predictors", grid_ms=grid_ms, density_ms=observed_density))
        rows.extend(_scenario_density_rows(model_name=model_name, scenario="fixed_predictors", grid_ms=grid_ms, density_ms=fixed_density))
    return pd.DataFrame(rows)


def write_conditional_vs_marginal_report(
    *,
    event_data: pd.DataFrame,
    best_models: dict[str, str | None],
    conditional_density: pd.DataFrame,
    ppc_by_quantile: pd.DataFrame,
    residual_summary: pd.DataFrame,
    pointwise_differences: pd.DataFrame,
    p_late_correlations: pd.DataFrame,
    counterfactual_density: pd.DataFrame,
    fit_metrics: dict[str, object] | None,
    warnings_list: list[str],
) -> str:
    """Build the conditional-vs-marginal diagnostic report."""

    report_lines = [
        "# Latency Regime Conditional vs Marginal Bimodality",
        "",
        "A posterior-predictive density pooled over the observed event mix can be bimodal even when the model is conditionally unimodal. Therefore, marginal bimodality should not be interpreted as evidence for latent components unless it persists at fixed predictor values or within comparable predictor strata.",
        "",
        "The key comparison is whether Model C explains conditional structure beyond the single-regime regression models.",
        "",
        "## What each diagnostic tests",
        "- Fixed-predictor conditional density: whether multimodality persists when predictors are held constant.",
        "- PPC by predictor quantile: whether within-stratum distributions remain bimodal or instead shift smoothly across strata.",
        "- R-model residual density: whether bimodality remains after regressing mean or scale on the continuous predictors.",
        "- Pointwise C minus R fit difference: where Model C improves over single-regime alternatives.",
        "- P(late) vs R-model predictions: whether Model C is mainly tracking the same early-to-late axis as a continuous regression.",
        "- Counterfactual predictor pooling: whether pooled bimodality disappears once predictors are fixed.",
        "",
        "## Available models",
        f"- Best Student-t R competitor: `{best_models['best_student_t_r']}`.",
        f"- Best shifted-lognormal R competitor: `{best_models['best_lognormal_r']}`.",
        f"- Controls metadata: `{(fit_metrics or {}).get('controls', {})}`.",
    ]
    if warnings_list:
        report_lines.extend(["", "## Warnings"])
        report_lines.extend([f"- {warning}" for warning in warnings_list])

    conditional_summary = _summarize_conditional_bimodality(conditional_density)
    report_lines.extend(["", "## Fixed-predictor conditional density"])
    report_lines.extend([f"- {line}" for line in conditional_summary])

    ppc_summary = _summarize_ppc_quantiles(ppc_by_quantile)
    report_lines.extend(["", "## PPC by predictor quantile"])
    report_lines.extend([f"- {line}" for line in ppc_summary])

    residual_lines = _summarize_residual_bimodality(residual_summary)
    report_lines.extend(["", "## R-model residual densities"])
    report_lines.extend([f"- {line}" for line in residual_lines])

    pointwise_lines = _summarize_pointwise_differences(pointwise_differences)
    report_lines.extend(["", "## Pointwise C minus R fit differences"])
    report_lines.extend([f"- {line}" for line in pointwise_lines])

    corr_lines = _summarize_p_late_correlations(p_late_correlations)
    report_lines.extend(["", "## P(late) versus R-model predictions"])
    report_lines.extend([f"- {line}" for line in corr_lines])

    counterfactual_lines = _summarize_counterfactual(counterfactual_density)
    report_lines.extend(["", "## Counterfactual predictor pooling"])
    report_lines.extend([f"- {line}" for line in counterfactual_lines])

    report_lines.extend(
        [
            "",
            "## Cautious interpretation",
            "- If each predictor stratum is broadly unimodal while the pooled density appears bimodal, that supports continuous predictor-dependent latency modulation.",
            "- If Model C remains bimodal at fixed predictor values while the R models remain conditionally unimodal, that is stronger evidence for latent timing component structure.",
            "- If Model C improves mainly for very late events, it is safer to describe this as a main latency component versus longer-latency/right-tail component.",
            "- If both Model C and the R models recover similar marginal early/late structure, the robust finding is posterior-predictive early/late structure rather than proof of discrete psychological regimes.",
            "",
            f"Observed events included: `{len(event_data)}`.",
        ]
    )
    return "\n".join(report_lines) + "\n"


def _coerce_event_columns(event_data: pd.DataFrame) -> pd.DataFrame:
    working = event_data.copy()
    numeric_columns = [
        "latency_from_partner_offset",
        "z_information_rate_lag_best",
        "z_prop_expected_cumulative_info_lag_best",
        "z_time_within_run",
        "z_time_within_run_squared",
        "time_within_run",
    ]
    for column_name in numeric_columns:
        if column_name in working.columns:
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
        else:
            working[column_name] = 0.0
    if "run" in working.columns:
        working["run"] = working["run"].astype(str)
    else:
        working["run"] = "run-unknown"
    return working


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_fixed_conditions() -> list[dict[str, object]]:
    return [
        {"condition_name": "low_expected_info", "z_information_rate_lag_best": 0.0, "z_prop_expected_cumulative_info_lag_best": -1.0},
        {"condition_name": "mean_predictors", "z_information_rate_lag_best": 0.0, "z_prop_expected_cumulative_info_lag_best": 0.0},
        {"condition_name": "high_expected_info", "z_information_rate_lag_best": 0.0, "z_prop_expected_cumulative_info_lag_best": 1.0},
        {"condition_name": "high_information_rate", "z_information_rate_lag_best": 1.0, "z_prop_expected_cumulative_info_lag_best": 0.0},
        {"condition_name": "low_information_rate", "z_information_rate_lag_best": -1.0, "z_prop_expected_cumulative_info_lag_best": 0.0},
    ]


def _build_latency_grid_ms(latency_s: np.ndarray) -> np.ndarray:
    finite = np.asarray(latency_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(-500.0, 1500.0, 500)
    lower_ms = min(-500.0, float(np.quantile(finite, 0.01) * LATENCY_MS_PER_SECOND) - 150.0)
    upper_ms = max(1500.0, float(np.quantile(finite, 0.99) * LATENCY_MS_PER_SECOND) + 150.0)
    return np.linspace(lower_ms, upper_ms, 600)


def _resolve_run_reference(fit_metrics: dict[str, object] | None) -> str:
    controls = dict((fit_metrics or {}).get("controls") or {})
    value = controls.get("run_reference_level")
    return str(value) if value is not None and str(value) else "reference_run"


def _extract_component_mean(component_parameters: pd.DataFrame | None, *, component: str, parameter: str) -> float:
    if component_parameters is None or component_parameters.empty:
        return math.nan
    subset = component_parameters.loc[
        (component_parameters["model_name"] == MODEL_C)
        & (component_parameters["component"] == component)
        & (component_parameters["parameter"] == parameter),
        "mean",
    ]
    if subset.empty:
        return math.nan
    return float(pd.to_numeric(subset, errors="coerce").iloc[0])


def _conditional_density_model_c(
    *,
    grid_ms: np.ndarray,
    component_parameters: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    features: dict[str, object],
    nu: float,
) -> np.ndarray:
    early_mu = _extract_component_mean(component_parameters, component="early", parameter="mu")
    late_mu = _extract_component_mean(component_parameters, component="late", parameter="mu")
    early_sigma = _extract_component_mean(component_parameters, component="early", parameter="sigma")
    late_sigma = _extract_component_mean(component_parameters, component="late", parameter="sigma")
    if not all(np.isfinite([early_mu, late_mu, early_sigma, late_sigma])):
        return np.full_like(grid_ms, np.nan, dtype=float)
    p_late = _evaluate_named_linear_terms(
        coefficient_table=gating_coefficients,
        model_name=MODEL_C,
        features=features,
    )
    if np.isfinite(p_late):
        p_late = 1.0 / (1.0 + np.exp(-p_late))
    else:
        p_late = 0.5
    early_density_s = t.pdf(milliseconds_to_seconds(grid_ms), df=nu, loc=early_mu, scale=early_sigma)
    late_density_s = t.pdf(milliseconds_to_seconds(grid_ms), df=nu, loc=late_mu, scale=late_sigma)
    return density_per_second_to_per_millisecond((1.0 - p_late) * early_density_s + p_late * late_density_s)


def _conditional_density_regression_model(
    *,
    model_name: str,
    grid_ms: np.ndarray,
    regression_coefficients: pd.DataFrame | None,
    shifted_lognormal_diagnostics: pd.DataFrame | None,
    features: dict[str, object],
    nu: float,
) -> np.ndarray:
    if model_name in {MODEL_R1, MODEL_R2}:
        mu_s = _evaluate_named_linear_terms(
            coefficient_table=regression_coefficients,
            model_name=model_name,
            features=features,
            coefficient_group="location",
        )
        sigma_eta = _evaluate_named_linear_terms(
            coefficient_table=regression_coefficients,
            model_name=model_name,
            features=features,
            coefficient_group="scale",
        )
        sigma_s = float(np.exp(sigma_eta)) if np.isfinite(sigma_eta) else _fallback_constant_sigma(regression_coefficients, model_name)
        return student_t_density_per_ms(grid_ms, mu_s=mu_s, sigma_s=sigma_s, nu=nu)
    mu_log = _evaluate_named_linear_terms(
        coefficient_table=regression_coefficients,
        model_name=model_name,
        features=features,
        coefficient_group="location",
    )
    sigma_log_eta = _evaluate_named_linear_terms(
        coefficient_table=regression_coefficients,
        model_name=model_name,
        features=features,
        coefficient_group="scale",
    )
    sigma_log = float(np.exp(sigma_log_eta)) if np.isfinite(sigma_log_eta) else _fallback_constant_sigma(regression_coefficients, model_name)
    shift_seconds = _resolve_shift_seconds(shifted_lognormal_diagnostics, model_name=model_name)
    return _shifted_lognormal_density_per_ms(grid_ms=grid_ms, shift_seconds=shift_seconds, mu_log=mu_log, sigma_log=sigma_log)


def _density_rows_for_condition(
    *,
    model_name: str,
    condition_name: str,
    grid_ms: np.ndarray,
    density_ms: np.ndarray,
    features: dict[str, object],
    run_reference: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for latency_ms, density_value in zip(grid_ms, density_ms, strict=False):
        rows.append(
            {
                "model_name": model_name,
                "condition_name": condition_name,
                "z_information_rate_lag_best": float(features["z_information_rate_lag_best"]),
                "z_prop_expected_cumulative_info_lag_best": float(features["z_prop_expected_cumulative_info_lag_best"]),
                "z_time_within_run": float(features.get("z_time_within_run", 0.0)),
                "run_reference": run_reference,
                "latency_ms": float(latency_ms),
                "density_per_ms": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_mean": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_lower": math.nan,
                "density_draw_upper": math.nan,
            }
        )
    return rows


def _evaluate_named_linear_terms(
    *,
    coefficient_table: pd.DataFrame | None,
    model_name: str,
    features: dict[str, object],
    coefficient_group: str | None = None,
) -> float:
    if coefficient_table is None or coefficient_table.empty:
        return math.nan
    working = coefficient_table.loc[coefficient_table["model_name"] == model_name].copy()
    if coefficient_group is not None and "coefficient_group" in working.columns:
        working = working.loc[working["coefficient_group"] == coefficient_group].copy()
    if working.empty:
        return math.nan
    total = 0.0
    seen_intercept = False
    for _, row in working.iterrows():
        term = str(row["term"])
        coefficient = float(pd.to_numeric(row["mean"], errors="coerce"))
        if not np.isfinite(coefficient):
            continue
        if term.startswith("alpha"):
            total += coefficient
            seen_intercept = True
            continue
        if term.endswith("_rate") or term == "beta_rate":
            total += coefficient * float(features.get("z_information_rate_lag_best", 0.0))
            continue
        if term.endswith("_expected") or term == "beta_expected":
            total += coefficient * float(features.get("z_prop_expected_cumulative_info_lag_best", 0.0))
            continue
        if term.endswith("_time") or term == "gamma_time":
            total += coefficient * float(features.get("z_time_within_run", 0.0))
            continue
        if term.endswith("_time2") or term == "gamma_time2":
            total += coefficient * float(features.get("z_time_within_run_squared", 0.0))
            continue
        if "run_effect_" in term:
            run_value = str(features.get("run", ""))
            expected_suffix = f"run_{run_value}"
            total += coefficient * float(term.endswith(expected_suffix))
    return total if seen_intercept or len(working) > 0 else math.nan


def _fallback_constant_sigma(regression_coefficients: pd.DataFrame | None, model_name: str) -> float:
    if regression_coefficients is None or regression_coefficients.empty:
        return 0.1
    working = regression_coefficients.loc[regression_coefficients["model_name"] == model_name].copy()
    if working.empty:
        return 0.1
    for term_name in ("sigma", "sigma_log", "alpha_sigma", "alpha_sigma_log"):
        subset = working.loc[working["term"] == term_name, "mean"]
        if subset.empty:
            continue
        value = float(pd.to_numeric(subset, errors="coerce").iloc[0])
        if term_name.startswith("alpha_"):
            return float(np.exp(value))
        return value
    return 0.1


def _resolve_shift_seconds(shifted_lognormal_diagnostics: pd.DataFrame | None, *, model_name: str) -> float:
    if shifted_lognormal_diagnostics is None or shifted_lognormal_diagnostics.empty:
        return 0.0
    subset = shifted_lognormal_diagnostics.loc[shifted_lognormal_diagnostics["model_name"] == model_name, "shift_seconds"]
    if subset.empty:
        return 0.0
    return float(pd.to_numeric(subset, errors="coerce").iloc[0])


def _shifted_lognormal_density_per_ms(
    *,
    grid_ms: np.ndarray,
    shift_seconds: float,
    mu_log: float,
    sigma_log: float,
) -> np.ndarray:
    grid_s = milliseconds_to_seconds(grid_ms)
    shifted = grid_s - float(shift_seconds)
    density_s = np.zeros_like(shifted, dtype=float)
    positive = shifted > 0.0
    if np.any(positive) and np.isfinite(sigma_log) and sigma_log > 0.0:
        density_s[positive] = (
            1.0
            / (shifted[positive] * sigma_log * np.sqrt(2.0 * np.pi))
            * np.exp(-((np.log(shifted[positive]) - mu_log) ** 2) / (2.0 * sigma_log**2))
        )
    return density_per_second_to_per_millisecond(density_s)


def _assign_predictor_quantile_bins(*, event_data: pd.DataFrame, predictor_name: str) -> pd.DataFrame:
    working = event_data[["row_index", "latency_from_partner_offset", predictor_name]].copy()
    working[predictor_name] = pd.to_numeric(working[predictor_name], errors="coerce")
    working = working.dropna(subset=[predictor_name, "latency_from_partner_offset"]).copy()
    if working.empty:
        return pd.DataFrame()
    n_bins = 4
    codes = pd.qcut(working[predictor_name], q=n_bins, labels=False, duplicates="drop")
    if codes.nunique() < n_bins or int(codes.value_counts().min()) < MIN_EVENTS_PER_QUANTILE_BIN:
        n_bins = 3
        codes = pd.qcut(working[predictor_name], q=n_bins, labels=False, duplicates="drop")
    working["quantile_index"] = pd.to_numeric(codes, errors="coerce")
    working = working.dropna(subset=["quantile_index"]).copy()
    working["quantile_index"] = working["quantile_index"].astype(int)
    if working.empty:
        return pd.DataFrame()
    quantiles = pd.qcut(working[predictor_name], q=working["quantile_index"].nunique(), duplicates="drop")
    intervals = pd.Series(quantiles, index=working.index)
    working["quantile_bin"] = [f"Q{index + 1}" for index in working["quantile_index"]]
    working["quantile_low"] = [float(interval.left) for interval in intervals]
    working["quantile_high"] = [float(interval.right) for interval in intervals]
    return working


def _ppc_draw_densities_for_bin(
    *,
    ppc_table: pd.DataFrame | None,
    model_name: str,
    row_indices: np.ndarray,
    grid_ms: np.ndarray,
) -> np.ndarray:
    if ppc_table is None or ppc_table.empty:
        return np.empty((0, len(grid_ms)), dtype=float)
    required = {"model_name", "draw_id", "statistic", "y_rep_value"}
    if not required <= set(ppc_table.columns):
        return np.empty((0, len(grid_ms)), dtype=float)
    working = ppc_table.loc[
        (ppc_table["model_name"] == model_name)
        & (ppc_table["statistic"] == "y_rep")
    ].copy()
    if working.empty:
        return np.empty((0, len(grid_ms)), dtype=float)
    working["row_index"] = working.groupby("draw_id").cumcount()
    working["y_rep_value"] = pd.to_numeric(working["y_rep_value"], errors="coerce")
    working = working.loc[working["row_index"].isin(row_indices)].dropna(subset=["y_rep_value"]).copy()
    densities: list[np.ndarray] = []
    for _, subset in working.groupby("draw_id", sort=False):
        samples_ms = seconds_to_milliseconds(subset["y_rep_value"].to_numpy(dtype=float))
        densities.append(_safe_kde_density_per_ms(samples_ms, grid_ms))
    return np.asarray(densities, dtype=float)


def _safe_kde_density_per_ms(samples_ms: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(samples_ms, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full_like(grid_ms, np.nan, dtype=float)
    if values.size == 1 or np.allclose(values, values[0]):
        sigma = 25.0
        density = np.exp(-0.5 * ((grid_ms - values.mean()) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        area = np.trapezoid(density, grid_ms)
        return density / area if area > 0 else density
    kde = gaussian_kde(values)
    return kde(grid_ms)


def _density_rows_with_quantile_meta(
    *,
    predictor_name: str,
    quantile_subset: pd.DataFrame,
    quantile_bin: str,
    model_name: str,
    grid_ms: np.ndarray,
    density_ms: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    quantile_low = float(quantile_subset["quantile_low"].iloc[0])
    quantile_high = float(quantile_subset["quantile_high"].iloc[0])
    n_events = int(len(quantile_subset))
    if lower is None:
        lower = np.full_like(density_ms, np.nan, dtype=float)
    if upper is None:
        upper = np.full_like(density_ms, np.nan, dtype=float)
    for latency_ms, density_value, lower_value, upper_value in zip(grid_ms, density_ms, lower, upper, strict=False):
        rows.append(
            {
                "predictor_name": predictor_name,
                "quantile_bin": quantile_bin,
                "quantile_low": quantile_low,
                "quantile_high": quantile_high,
                "n_events": n_events,
                "model_name": model_name,
                "latency_ms": float(latency_ms),
                "density_per_ms": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_mean": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_lower": float(lower_value) if np.isfinite(lower_value) else math.nan,
                "density_draw_upper": float(upper_value) if np.isfinite(upper_value) else math.nan,
            }
        )
    return rows


def _compute_event_level_regression_predictions(
    *,
    event_data: pd.DataFrame,
    regression_coefficients: pd.DataFrame | None,
    shifted_lognormal_diagnostics: pd.DataFrame | None,
    nu: float,
) -> pd.DataFrame:
    if regression_coefficients is None or regression_coefficients.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                "model_name",
                "predicted_mu_s",
                "predicted_sigma_s",
                "predicted_mu_log",
                "predicted_sigma_log",
                "shift_seconds",
                "predicted_mean_s",
            ]
        )
    rows: list[dict[str, object]] = []
    for model_name, _ in regression_coefficients.groupby("model_name", sort=False):
        for _, event_row in event_data.iterrows():
            features = {
                "z_information_rate_lag_best": float(event_row["z_information_rate_lag_best"]),
                "z_prop_expected_cumulative_info_lag_best": float(event_row["z_prop_expected_cumulative_info_lag_best"]),
                "z_time_within_run": float(event_row.get("z_time_within_run", 0.0)),
                "z_time_within_run_squared": float(event_row.get("z_time_within_run_squared", 0.0)),
                "run": str(event_row.get("run", "")),
            }
            if model_name in {MODEL_R1, MODEL_R2}:
                mu_s = _evaluate_named_linear_terms(
                    coefficient_table=regression_coefficients,
                    model_name=model_name,
                    features=features,
                    coefficient_group="location",
                )
                sigma_eta = _evaluate_named_linear_terms(
                    coefficient_table=regression_coefficients,
                    model_name=model_name,
                    features=features,
                    coefficient_group="scale",
                )
                sigma_s = float(np.exp(sigma_eta)) if np.isfinite(sigma_eta) else _fallback_constant_sigma(regression_coefficients, model_name)
                rows.append(
                    {
                        "row_index": int(event_row["row_index"]),
                        "model_name": model_name,
                        "predicted_mu_s": mu_s,
                        "predicted_sigma_s": sigma_s,
                        "predicted_mu_log": math.nan,
                        "predicted_sigma_log": math.nan,
                        "shift_seconds": math.nan,
                        "predicted_mean_s": mu_s,
                    }
                )
            elif model_name in {MODEL_R3, MODEL_R4}:
                mu_log = _evaluate_named_linear_terms(
                    coefficient_table=regression_coefficients,
                    model_name=model_name,
                    features=features,
                    coefficient_group="location",
                )
                sigma_log_eta = _evaluate_named_linear_terms(
                    coefficient_table=regression_coefficients,
                    model_name=model_name,
                    features=features,
                    coefficient_group="scale",
                )
                sigma_log = float(np.exp(sigma_log_eta)) if np.isfinite(sigma_log_eta) else _fallback_constant_sigma(regression_coefficients, model_name)
                shift_seconds = _resolve_shift_seconds(shifted_lognormal_diagnostics, model_name=model_name)
                predicted_mean = shift_seconds + float(np.exp(mu_log + 0.5 * sigma_log**2))
                rows.append(
                    {
                        "row_index": int(event_row["row_index"]),
                        "model_name": model_name,
                        "predicted_mu_s": predicted_mean,
                        "predicted_sigma_s": math.nan,
                        "predicted_mu_log": mu_log,
                        "predicted_sigma_log": sigma_log,
                        "shift_seconds": shift_seconds,
                        "predicted_mean_s": predicted_mean,
                    }
                )
    return pd.DataFrame(rows)


def _residual_peak_summary(*, model_name: str, residual_type: str, values_ms: np.ndarray) -> dict[str, object]:
    values = np.asarray(values_ms, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "model_name": model_name,
            "residual_type": residual_type,
            "n_events": 0,
            "n_detected_peaks": 0,
            "peak_locations_ms": "",
            "notes": "No finite residuals were available.",
        }
    grid = np.linspace(float(np.quantile(values, 0.01)) - 50.0, float(np.quantile(values, 0.99)) + 50.0, 500)
    density = _safe_kde_density_per_ms(values, grid)
    peaks = _detect_density_peaks(grid, density)
    peak_locations = ", ".join(f"{location:.1f}" for location in peaks)
    return {
        "model_name": model_name,
        "residual_type": residual_type,
        "n_events": int(values.size),
        "n_detected_peaks": int(len(peaks)),
        "peak_locations_ms": peak_locations,
        "notes": "KDE peak count used as a lightweight residual bimodality summary.",
    }


def _detect_density_peaks(grid: np.ndarray, density: np.ndarray) -> list[float]:
    x = np.asarray(grid, dtype=float)
    y = np.asarray(density, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size < 3:
        return []
    peak_indices: list[int] = []
    baseline = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
    threshold = baseline * 0.05
    for idx in range(1, len(y) - 1):
        if y[idx] > y[idx - 1] and y[idx] > y[idx + 1] and y[idx] >= threshold:
            peak_indices.append(idx)
    return [float(x[idx]) for idx in peak_indices]


def _model_c_event_inputs(
    *,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
) -> pd.DataFrame:
    if event_probabilities is None or event_probabilities.empty:
        return pd.DataFrame()
    probs = event_probabilities.copy()
    probs["row_index"] = np.arange(len(probs), dtype=int)
    probs = probs.loc[probs["model_name"] == MODEL_C].copy()
    if probs.empty:
        return pd.DataFrame()
    probs = probs.rename(columns={"p_late_mean": "p_late_model_c"})
    early_mu = _extract_component_mean(component_parameters, component="early", parameter="mu")
    late_mu = _extract_component_mean(component_parameters, component="late", parameter="mu")
    early_sigma = _extract_component_mean(component_parameters, component="early", parameter="sigma")
    late_sigma = _extract_component_mean(component_parameters, component="late", parameter="sigma")
    merged = event_data.merge(probs[["row_index", "p_late_model_c"]], on="row_index", how="left")
    merged["early_mu_s"] = early_mu
    merged["late_mu_s"] = late_mu
    merged["early_sigma_s"] = early_sigma
    merged["late_sigma_s"] = late_sigma
    return merged


def _model_c_log_likelihood(
    *,
    y_s: float,
    p_late: float,
    early_mu_s: float,
    late_mu_s: float,
    early_sigma_s: float,
    late_sigma_s: float,
    nu: float,
) -> float:
    early = float(t.pdf(y_s, df=nu, loc=early_mu_s, scale=early_sigma_s))
    late = float(t.pdf(y_s, df=nu, loc=late_mu_s, scale=late_sigma_s))
    mixture_density = (1.0 - p_late) * early + p_late * late
    return float(np.log(max(mixture_density, 1e-12)))


def _regression_row_log_likelihood(
    *,
    y_s: float,
    model_name: str,
    predicted_mu_s: float,
    predicted_sigma_s: float,
    predicted_mu_log: float,
    predicted_sigma_log: float,
    shift_seconds: float,
    nu: float,
) -> float:
    if model_name in {MODEL_R1, MODEL_R2}:
        density = float(t.pdf(y_s, df=nu, loc=predicted_mu_s, scale=predicted_sigma_s))
        return float(np.log(max(density, 1e-12)))
    shifted = y_s - shift_seconds
    if shifted <= 0.0 or not np.isfinite(predicted_sigma_log) or predicted_sigma_log <= 0.0:
        return float(np.log(1e-12))
    density = (
        1.0
        / (shifted * predicted_sigma_log * np.sqrt(2.0 * np.pi))
        * np.exp(-((np.log(shifted) - predicted_mu_log) ** 2) / (2.0 * predicted_sigma_log**2))
    )
    return float(np.log(max(float(density), 1e-12)))


def _event_averaged_model_c_density(
    *,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    grid_ms: np.ndarray,
    use_event_probabilities: bool,
    nu: float,
) -> np.ndarray:
    early_mu = _extract_component_mean(component_parameters, component="early", parameter="mu")
    late_mu = _extract_component_mean(component_parameters, component="late", parameter="mu")
    early_sigma = _extract_component_mean(component_parameters, component="early", parameter="sigma")
    late_sigma = _extract_component_mean(component_parameters, component="late", parameter="sigma")
    if not all(np.isfinite([early_mu, late_mu, early_sigma, late_sigma])):
        return np.full_like(grid_ms, np.nan, dtype=float)
    grid_s = milliseconds_to_seconds(grid_ms)
    early_density_s = t.pdf(grid_s, df=nu, loc=early_mu, scale=early_sigma)
    late_density_s = t.pdf(grid_s, df=nu, loc=late_mu, scale=late_sigma)
    if use_event_probabilities and event_probabilities is not None and not event_probabilities.empty:
        probs = pd.to_numeric(
            event_probabilities.loc[event_probabilities["model_name"] == MODEL_C, "p_late_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
    else:
        probs = np.asarray(
            [
                1.0
                / (1.0 + np.exp(-_evaluate_named_linear_terms(coefficient_table=gating_coefficients, model_name=MODEL_C, features={
                    "z_information_rate_lag_best": float(row["z_information_rate_lag_best"]),
                    "z_prop_expected_cumulative_info_lag_best": float(row["z_prop_expected_cumulative_info_lag_best"]),
                    "z_time_within_run": float(row.get("z_time_within_run", 0.0)),
                    "z_time_within_run_squared": float(row.get("z_time_within_run_squared", 0.0)),
                    "run": str(row.get("run", "")),
                })))
                for _, row in event_data.iterrows()
            ],
            dtype=float,
        )
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        probs = np.array([0.5], dtype=float)
    density_s = np.mean((1.0 - probs[:, None]) * early_density_s[None, :] + probs[:, None] * late_density_s[None, :], axis=0)
    return density_per_second_to_per_millisecond(density_s)


def _event_averaged_regression_density(
    *,
    event_data: pd.DataFrame,
    regression_predictions: pd.DataFrame,
    model_name: str,
    grid_ms: np.ndarray,
    nu: float,
) -> np.ndarray:
    subset = regression_predictions.loc[regression_predictions["model_name"] == model_name].copy()
    if subset.empty:
        return np.full_like(grid_ms, np.nan, dtype=float)
    densities = []
    for _, row in subset.iterrows():
        if model_name in {MODEL_R1, MODEL_R2}:
            densities.append(
                student_t_density_per_ms(
                    grid_ms,
                    mu_s=float(row["predicted_mu_s"]),
                    sigma_s=float(row["predicted_sigma_s"]),
                    nu=nu,
                )
            )
        else:
            densities.append(
                _shifted_lognormal_density_per_ms(
                    grid_ms=grid_ms,
                    shift_seconds=float(row["shift_seconds"]),
                    mu_log=float(row["predicted_mu_log"]),
                    sigma_log=float(row["predicted_sigma_log"]),
                )
            )
    return np.nanmean(np.asarray(densities, dtype=float), axis=0)


def _scenario_density_rows(*, model_name: str, scenario: str, grid_ms: np.ndarray, density_ms: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for latency_ms, density_value in zip(grid_ms, density_ms, strict=False):
        rows.append(
            {
                "model_name": model_name,
                "scenario": scenario,
                "latency_ms": float(latency_ms),
                "density_per_ms": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_mean": float(density_value) if np.isfinite(density_value) else math.nan,
                "density_draw_lower": math.nan,
                "density_draw_upper": math.nan,
                "notes": "Posterior mean density only." if np.isfinite(density_value) else "Density unavailable.",
            }
        )
    return rows


def _plot_conditional_density_fixed_predictors(*, conditional_density: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), constrained_layout=True)
    axes_flat = axes.flatten()
    if conditional_density.empty:
        for axis in axes_flat:
            _draw_placeholder(axis, "Conditional density unavailable")
        _save_figure(figure, output_path)
        return
    conditions = list(dict.fromkeys(conditional_density["condition_name"].astype(str)))
    for axis, condition_name in zip(axes_flat, conditions, strict=False):
        subset = conditional_density.loc[conditional_density["condition_name"] == condition_name].copy()
        for model_name, model_rows in subset.groupby("model_name", sort=False):
            axis.plot(
                model_rows["latency_ms"],
                model_rows["density_draw_mean"],
                linewidth=2.0,
                label=MODEL_LABELS.get(model_name, model_name),
            )
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_title(condition_name.replace("_", " "))
        axis.set_xlabel("Latency from partner offset (ms)")
        axis.set_ylabel("Density (per ms)")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(conditions) :]:
        axis.axis("off")
    _save_figure(figure, output_path)


def _plot_ppc_by_predictor_quantile(
    *,
    density_table: pd.DataFrame,
    predictor_name: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    axes_flat = axes.flatten()
    subset = density_table.loc[density_table["predictor_name"] == predictor_name].copy()
    if subset.empty:
        for axis in axes_flat:
            _draw_placeholder(axis, "No PPC rows available")
        _save_figure(figure, output_path)
        return
    quantiles = list(dict.fromkeys(subset["quantile_bin"].astype(str)))
    for axis, quantile_bin in zip(axes_flat, quantiles, strict=False):
        bin_rows = subset.loc[subset["quantile_bin"] == quantile_bin].copy()
        for model_name, model_rows in bin_rows.groupby("model_name", sort=False):
            linewidth = 2.2 if model_name == "observed" else 1.8
            axis.plot(model_rows["latency_ms"], model_rows["density_draw_mean"], linewidth=linewidth, label=MODEL_LABELS.get(model_name, model_name))
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_title(f"{quantile_bin} ({int(bin_rows['n_events'].iloc[0])} events)")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Density (per ms)")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(quantiles) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=14)
    _save_figure(figure, output_path)


def _plot_r_model_residual_density(*, residual_table: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    if residual_table.empty:
        for axis in axes:
            _draw_placeholder(axis, "No residual rows available")
        _save_figure(figure, output_path)
        return
    residual_specs = [
        ("residual_ms", "Residual (ms)", "residual_ms"),
        ("standardised_residual", "Standardised residual", "standardised_residual"),
    ]
    for axis, (_, x_label, value_column) in zip(axes, residual_specs, strict=False):
        for model_name, subset in residual_table.groupby("model_name", sort=False):
            values = pd.to_numeric(subset[value_column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            grid = np.linspace(float(np.quantile(values, 0.01)) - 50.0, float(np.quantile(values, 0.99)) + 50.0, 500)
            density = _safe_kde_density_per_ms(values, grid)
            axis.plot(grid, density, linewidth=2.0, label=MODEL_LABELS.get(model_name, model_name))
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Density")
        axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, output_path)


def _plot_pointwise_difference(*, pointwise_table: pd.DataFrame, x_column: str, output_path: Path, x_label: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if pointwise_table.empty or x_column not in pointwise_table.columns:
        _draw_placeholder(axis, "Pointwise difference unavailable")
        _save_figure(figure, output_path)
        return
    for comparison, subset in pointwise_table.groupby("comparison", sort=False):
        x = pd.to_numeric(subset[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(subset["elpd_difference"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            continue
        axis.scatter(x, y, alpha=0.35, s=18, label=comparison)
        smooth = _binned_smoother(x, y)
        if not smooth.empty:
            axis.plot(smooth["x"], smooth["y"], linewidth=2.0)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("C minus R pointwise log-lik difference")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, output_path)


def _plot_p_late_vs_r_mu(*, merged: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    if merged.empty:
        _draw_placeholder(axis, "P(late) vs R mu unavailable")
        _save_figure(figure, output_path)
        return
    axis.scatter(merged["r_mu_ms"], merged["p_late_model_c"], alpha=0.45, s=24, color="#1d3557")
    smooth = _binned_smoother(merged["r_mu_ms"].to_numpy(dtype=float), merged["p_late_model_c"].to_numpy(dtype=float))
    if not smooth.empty:
        axis.plot(smooth["x"], smooth["y"], linewidth=2.0, color="#e76f51")
    axis.set_xlabel("R-model predicted mean latency (ms)")
    axis.set_ylabel("Model C P(late)")
    _save_figure(figure, output_path)


def _plot_latency_vs_r_mu_coloured_by_p_late(*, merged: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.5))
    if merged.empty:
        _draw_placeholder(axis, "Latency vs R mu unavailable")
        _save_figure(figure, output_path)
        return
    scatter = axis.scatter(
        merged["r_mu_ms"],
        merged["observed_latency_ms"],
        c=merged["p_late_model_c"],
        cmap="viridis",
        alpha=0.7,
        s=26,
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Model C P(late)")
    axis.set_xlabel("R-model predicted mean latency (ms)")
    axis.set_ylabel("Observed latency (ms)")
    _save_figure(figure, output_path)


def _plot_latency_vs_predictor_coloured_by_p_late(
    *,
    merged: pd.DataFrame,
    predictor_column: str,
    predictor_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.5))
    if merged.empty or predictor_column not in merged.columns:
        _draw_placeholder(axis, "Latency vs predictor unavailable")
        _save_figure(figure, output_path)
        return
    working = merged[[predictor_column, "observed_latency_ms", "p_late_model_c"]].copy()
    for column_name in working.columns:
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    working = working.dropna().copy()
    if working.empty:
        _draw_placeholder(axis, "Latency vs predictor unavailable")
        _save_figure(figure, output_path)
        return
    scatter = axis.scatter(
        working["observed_latency_ms"],
        working[predictor_column],
        c=working["p_late_model_c"],
        cmap="inferno",
        alpha=0.78,
        s=28,
        edgecolors="none",
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Model C P(late)")
    axis.set_xlabel("Response latency from partner offset (ms)")
    axis.set_ylabel(predictor_label)
    _save_figure(figure, output_path)


def _plot_counterfactual_predictor_distribution(*, density_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    if density_table.empty:
        _draw_placeholder(axis, "Counterfactual density unavailable")
        _save_figure(figure, output_path)
        return
    for (model_name, scenario), subset in density_table.groupby(["model_name", "scenario"], sort=False):
        linestyle = "-" if scenario == "observed_predictors" else "--"
        axis.plot(
            subset["latency_ms"],
            subset["density_draw_mean"],
            linewidth=2.0,
            linestyle=linestyle,
            label=f"{MODEL_LABELS.get(model_name, model_name)}: {scenario.replace('_', ' ')}",
        )
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Density (per ms)")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, output_path)


def _draw_placeholder(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _binned_smoother(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if x.size < 3:
        return pd.DataFrame(columns=["x", "y"])
    bins = np.linspace(float(x.min()), float(x.max()), num=min(n_bins, x.size) + 1)
    bucket = np.digitize(x, bins[1:-1], right=False)
    rows = []
    for bucket_id in np.unique(bucket):
        mask = bucket == bucket_id
        if mask.sum() == 0:
            continue
        rows.append({"x": float(np.mean(x[mask])), "y": float(np.mean(y[mask]))})
    return pd.DataFrame(rows).sort_values("x")


def _summarize_conditional_bimodality(conditional_density: pd.DataFrame) -> list[str]:
    if conditional_density.empty:
        return ["Conditional densities were unavailable."]
    lines: list[str] = []
    for (model_name, condition_name), subset in conditional_density.groupby(["model_name", "condition_name"], sort=False):
        peaks = _detect_density_peaks(
            subset["latency_ms"].to_numpy(dtype=float),
            subset["density_draw_mean"].to_numpy(dtype=float),
        )
        lines.append(f"{MODEL_LABELS.get(model_name, model_name)} at `{condition_name}` had `{len(peaks)}` detected KDE peak(s).")
    return lines


def _summarize_ppc_quantiles(ppc_by_quantile: pd.DataFrame) -> list[str]:
    if ppc_by_quantile.empty:
        return ["Predictor-stratified PPC rows were unavailable."]
    lines: list[str] = []
    for (predictor_name, quantile_bin, model_name), subset in ppc_by_quantile.groupby(
        ["predictor_name", "quantile_bin", "model_name"],
        sort=False,
    ):
        peaks = _detect_density_peaks(subset["latency_ms"].to_numpy(dtype=float), subset["density_draw_mean"].to_numpy(dtype=float))
        lines.append(f"{predictor_name} {quantile_bin} for `{model_name}` had `{len(peaks)}` detected peak(s).")
    return lines[:12]


def _summarize_residual_bimodality(residual_summary: pd.DataFrame) -> list[str]:
    if residual_summary.empty:
        return ["Residual summaries were unavailable."]
    return [
        f"{row['model_name']} {row['residual_type']} peak count = {row['n_detected_peaks']} at [{row['peak_locations_ms']}]."
        for _, row in residual_summary.iterrows()
    ]


def _summarize_pointwise_differences(pointwise_differences: pd.DataFrame) -> list[str]:
    if pointwise_differences.empty:
        return ["Pointwise C minus R comparisons were unavailable."]
    lines: list[str] = []
    for comparison, subset in pointwise_differences.groupby("comparison", sort=False):
        mean_diff = float(pd.to_numeric(subset["elpd_difference"], errors="coerce").mean())
        late_threshold = float(np.quantile(subset["observed_latency_ms"], 0.75))
        late_mean = float(subset.loc[subset["observed_latency_ms"] >= late_threshold, "elpd_difference"].mean())
        early_mean = float(subset.loc[subset["observed_latency_ms"] < late_threshold, "elpd_difference"].mean())
        if late_mean > early_mean:
            emphasis = "improvements were larger in the late tail."
        else:
            emphasis = "improvements were not concentrated only in the late tail."
        lines.append(f"{comparison}: mean difference = {mean_diff:.3f}; {emphasis}")
    return lines


def _summarize_p_late_correlations(p_late_correlations: pd.DataFrame) -> list[str]:
    if p_late_correlations.empty:
        return ["P(late) versus regression prediction correlations were unavailable."]
    lines = []
    for _, row in p_late_correlations.iterrows():
        lines.append(f"{row['metric']} = {float(row['correlation']):.3f} for `{row['r_model_name']}`.")
    return lines


def _summarize_counterfactual(counterfactual_density: pd.DataFrame) -> list[str]:
    if counterfactual_density.empty:
        return ["Counterfactual predictor-pooling densities were unavailable."]
    lines: list[str] = []
    for (model_name, scenario), subset in counterfactual_density.groupby(["model_name", "scenario"], sort=False):
        peaks = _detect_density_peaks(subset["latency_ms"].to_numpy(dtype=float), subset["density_draw_mean"].to_numpy(dtype=float))
        lines.append(f"{MODEL_LABELS.get(model_name, model_name)} under `{scenario}` had `{len(peaks)}` detected peak(s).")
    return lines
