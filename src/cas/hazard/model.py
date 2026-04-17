"""Model fitting for pooled discrete-time partner-onset hazard analysis."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_FULL_TERMS = ("intercept", "tau_seconds", "entropy_z")
DEFAULT_BASELINE_TERMS = ("intercept", "tau_seconds")
QUADRATIC_TIME_TERM = "tau_seconds_sq"


@dataclass(frozen=True, slots=True)
class HazardModelResult:
    """Outputs from hazard-model fitting."""

    coefficients: pd.DataFrame
    summary_text: str
    fit_metrics: dict[str, Any]
    prediction_table: pd.DataFrame
    backend_used: str
    should_skip_prediction_plot: bool
    prediction_skip_reason: str | None


def fit_pooled_discrete_time_hazard_model(
    hazard_table: pd.DataFrame,
    *,
    fitting_backend: str,
    include_quadratic_time: bool,
    prefer_random_intercept_subject: bool,
) -> HazardModelResult:
    """Fit the pooled partner-onset hazard model and a time-only baseline."""

    full_terms = list(DEFAULT_FULL_TERMS)
    baseline_terms = list(DEFAULT_BASELINE_TERMS)
    if include_quadratic_time:
        full_terms.append(QUADRATIC_TIME_TERM)
        baseline_terms.append(QUADRATIC_TIME_TERM)

    subject_ids = hazard_table["subject_id"].astype(str).to_numpy()
    response = hazard_table["event"].to_numpy(dtype=float)
    backend_used = _resolve_backend(
        fitting_backend=fitting_backend,
        prefer_random_intercept_subject=prefer_random_intercept_subject,
    )
    full_result, full_fit_warnings, fallback_reason = _fit_statsmodels_backend(
        hazard_table=hazard_table,
        design_terms=full_terms,
        subject_ids=subject_ids,
        response=response,
        backend_used=backend_used,
    )
    baseline_result, baseline_fit_warnings, _ = _fit_statsmodels_backend(
        hazard_table=hazard_table,
        design_terms=baseline_terms,
        subject_ids=subject_ids,
        response=response,
        backend_used=backend_used,
    )

    coefficients = _build_coefficients_table(
        fitted_result=full_result,
        parameter_names=full_terms,
    )
    fit_metrics = _build_fit_metrics(
        full_result=full_result,
        baseline_result=baseline_result,
        hazard_table=hazard_table,
        backend_used=backend_used,
        fallback_reason=fallback_reason,
        full_fit_warnings=full_fit_warnings,
        baseline_fit_warnings=baseline_fit_warnings,
    )
    prediction_table = build_prediction_table(
        hazard_table=hazard_table,
        fitted_result=full_result,
        include_quadratic_time=include_quadratic_time,
    )
    should_skip_prediction_plot, prediction_skip_reason = _assess_prediction_plot_pathology(
        fit_metrics=fit_metrics,
        prediction_table=prediction_table,
    )
    summary_text = _build_summary_text(
        full_result=full_result,
        baseline_result=baseline_result,
        backend_used=backend_used,
        fallback_reason=fallback_reason,
        fit_metrics=fit_metrics,
        prediction_skip_reason=prediction_skip_reason,
    )

    return HazardModelResult(
        coefficients=coefficients,
        summary_text=summary_text,
        fit_metrics=fit_metrics,
        prediction_table=prediction_table,
        backend_used=backend_used,
        should_skip_prediction_plot=should_skip_prediction_plot,
        prediction_skip_reason=prediction_skip_reason,
    )


def build_prediction_table(
    *,
    hazard_table: pd.DataFrame,
    fitted_result: Any,
    include_quadratic_time: bool,
) -> pd.DataFrame:
    """Build model-based predicted hazard curves for plotting."""

    tau_values = np.sort(hazard_table["tau_seconds"].unique())
    entropy_mean = float(hazard_table["entropy_z"].mean())
    entropy_std = float(hazard_table["entropy_z"].std(ddof=0))
    scenarios = {
        "low_entropy": entropy_mean - entropy_std,
        "mean_entropy": entropy_mean,
        "high_entropy": entropy_mean + entropy_std,
    }

    rows: list[dict[str, float | str]] = []
    for scenario_name, entropy_value in scenarios.items():
        prediction_frame = pd.DataFrame(
            {
                "intercept": np.ones(tau_values.shape[0], dtype=float),
                "tau_seconds": tau_values,
                "entropy_z": np.full(tau_values.shape[0], entropy_value, dtype=float),
            }
        )
        if include_quadratic_time:
            prediction_frame["tau_seconds_sq"] = tau_values**2
        linear_predictor = np.dot(
            prediction_frame.to_numpy(dtype=float),
            np.asarray(fitted_result.params, dtype=float),
        )
        predicted_hazard = 1.0 / (1.0 + np.exp(-linear_predictor))
        for tau_seconds, hazard_value in zip(tau_values, predicted_hazard, strict=True):
            rows.append(
                {
                    "scenario": scenario_name,
                    "tau_seconds": float(tau_seconds),
                    "entropy_z": float(entropy_value),
                    "predicted_hazard": float(hazard_value),
                }
            )
    return pd.DataFrame(rows)


def _fit_statsmodels_backend(
    *,
    hazard_table: pd.DataFrame,
    design_terms: list[str],
    subject_ids: np.ndarray,
    response: np.ndarray,
    backend_used: str,
) -> tuple[Any, list[str], str]:
    """Fit one statsmodels backend instance for a given formula."""

    design_matrix = _build_design_matrix(hazard_table=hazard_table, design_terms=design_terms)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if backend_used == "gee":
            fitted_result = sm.GEE(
                endog=response,
                exog=design_matrix,
                groups=subject_ids,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
            fallback_reason = (
                "Used GEE with subject clustering as the first-pass fallback because a stable "
                "binomial random-intercept GLMM backend is not currently wired into this Python codebase."
            )
        elif backend_used == "glm_cluster":
            fitted_result = sm.GLM(
                endog=response,
                exog=design_matrix,
                family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": subject_ids})
            fallback_reason = (
                "Used pooled binomial GLM with cluster-robust standard errors by subject as the documented fallback."
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_used}")
    fit_warnings = list(dict.fromkeys(str(warning.message) for warning in caught_warnings))
    return fitted_result, fit_warnings, fallback_reason


def _resolve_backend(*, fitting_backend: str, prefer_random_intercept_subject: bool) -> str:
    """Resolve the actual model backend to use."""

    if fitting_backend == "auto":
        del prefer_random_intercept_subject
        return "gee"
    return fitting_backend


def _build_design_matrix(*, hazard_table: pd.DataFrame, design_terms: list[str]) -> pd.DataFrame:
    """Build the fixed-effects design matrix."""

    design_matrix = pd.DataFrame(index=hazard_table.index)
    design_matrix["intercept"] = 1.0
    design_matrix["tau_seconds"] = hazard_table["tau_seconds"].to_numpy(dtype=float)
    design_matrix["entropy_z"] = hazard_table["entropy_z"].to_numpy(dtype=float)
    if QUADRATIC_TIME_TERM in design_terms:
        design_matrix[QUADRATIC_TIME_TERM] = hazard_table["tau_seconds_sq"].to_numpy(dtype=float)
    return design_matrix.loc[:, design_terms]


def _build_coefficients_table(*, fitted_result: Any, parameter_names: list[str]) -> pd.DataFrame:
    """Build a tidy coefficient table."""

    confidence_intervals = fitted_result.conf_int()
    z_values = np.asarray(fitted_result.params) / np.asarray(fitted_result.bse)
    coefficient_table = pd.DataFrame(
        {
            "term": parameter_names,
            "estimate": np.asarray(fitted_result.params, dtype=float),
            "std_error": np.asarray(fitted_result.bse, dtype=float),
            "z_value": z_values.astype(float),
            "p_value": np.asarray(fitted_result.pvalues, dtype=float),
            "conf_low": np.asarray(confidence_intervals.iloc[:, 0], dtype=float),
            "conf_high": np.asarray(confidence_intervals.iloc[:, 1], dtype=float),
        }
    )
    coefficient_table["null_hypothesis"] = "coefficient = 0"
    return coefficient_table


def _build_fit_metrics(
    *,
    full_result: Any,
    baseline_result: Any,
    hazard_table: pd.DataFrame,
    backend_used: str,
    fallback_reason: str,
    full_fit_warnings: list[str],
    baseline_fit_warnings: list[str],
) -> dict[str, Any]:
    """Build a JSON-friendly fit-metrics summary."""

    full_log_likelihood = _maybe_float(getattr(full_result, "llf", None))
    baseline_log_likelihood = _maybe_float(getattr(baseline_result, "llf", None))
    full_aic = _maybe_float(getattr(full_result, "aic", None))
    baseline_aic = _maybe_float(getattr(baseline_result, "aic", None))
    full_bic = _maybe_float(getattr(full_result, "bic", None))
    baseline_bic = _maybe_float(getattr(baseline_result, "bic", None))
    full_llnull = _maybe_float(getattr(full_result, "llnull", None))
    pseudo_r2 = None
    if full_log_likelihood is not None and full_llnull not in (None, 0.0):
        pseudo_r2 = float(1.0 - (full_log_likelihood / full_llnull))

    entropy_parameter_index = list(full_result.params.index).index("entropy_z")
    entropy_confidence_interval = full_result.conf_int().iloc[entropy_parameter_index]
    n_positive_episodes = int(hazard_table.groupby("event_id")["event"].max().sum())
    n_episodes = int(hazard_table["event_id"].nunique())
    n_censored_episodes = int(hazard_table.groupby("event_id")["censored_episode"].first().sum())

    return {
        "hazard_clock": "time since partner onset",
        "backend_used": backend_used,
        "used_fallback": True,
        "fallback_reason": fallback_reason,
        "n_subjects": int(hazard_table["subject_id"].nunique()),
        "n_episodes": n_episodes,
        "n_positive_episodes": n_positive_episodes,
        "n_censored_episodes": n_censored_episodes,
        "n_person_period_rows": int(len(hazard_table)),
        "n_positive_rows": int(hazard_table["event"].sum()),
        "full_model": {
            "log_likelihood": full_log_likelihood,
            "aic": full_aic,
            "bic": full_bic,
            "pseudo_r2_mcfadden": pseudo_r2,
            "converged": bool(getattr(full_result, "converged", True)),
            "fit_warnings": full_fit_warnings,
        },
        "baseline_time_only_model": {
            "log_likelihood": baseline_log_likelihood,
            "aic": baseline_aic,
            "bic": baseline_bic,
            "converged": bool(getattr(baseline_result, "converged", True)),
            "fit_warnings": baseline_fit_warnings,
        },
        "model_comparison": {
            "delta_aic_full_minus_baseline": (
                None if full_aic is None or baseline_aic is None else float(full_aic - baseline_aic)
            ),
            "delta_bic_full_minus_baseline": (
                None if full_bic is None or baseline_bic is None else float(full_bic - baseline_bic)
            ),
            "delta_log_likelihood_full_minus_baseline": (
                None
                if full_log_likelihood is None or baseline_log_likelihood is None
                else float(full_log_likelihood - baseline_log_likelihood)
            ),
        },
        "entropy_effect": {
            "null_hypothesis": "beta_entropy = 0",
            "estimate": float(full_result.params["entropy_z"]),
            "std_error": float(full_result.bse["entropy_z"]),
            "z_value": float(full_result.params["entropy_z"] / full_result.bse["entropy_z"]),
            "p_value": float(full_result.pvalues["entropy_z"]),
            "conf_low": float(entropy_confidence_interval.iloc[0]),
            "conf_high": float(entropy_confidence_interval.iloc[1]),
        },
    }


def _build_summary_text(
    *,
    full_result: Any,
    baseline_result: Any,
    backend_used: str,
    fallback_reason: str,
    fit_metrics: dict[str, Any],
    prediction_skip_reason: str | None,
) -> str:
    """Build a concise model summary text report."""

    lines = [
        "Pooled discrete-time hazard model anchored at partner onset",
        "",
        "Hazard clock: time since partner onset",
        "Null hypothesis for the main predictor: beta_entropy = 0",
        f"Backend used: {backend_used}",
        f"Fallback status: documented first-pass fallback ({fallback_reason})",
        f"Subjects: {fit_metrics['n_subjects']}",
        f"Episodes: {fit_metrics['n_episodes']}",
        f"Positive episodes: {fit_metrics['n_positive_episodes']}",
        f"Censored episodes: {fit_metrics['n_censored_episodes']}",
        f"Person-period rows: {fit_metrics['n_person_period_rows']}",
        "",
        "Entropy effect:",
        (
            "estimate={estimate:.6f}, std_error={std_error:.6f}, z={z_value:.6f}, "
            "p={p_value:.6g}, 95% CI=[{conf_low:.6f}, {conf_high:.6f}]"
        ).format(**fit_metrics["entropy_effect"]),
        "",
        "Model comparison against the time-only baseline:",
        str(fit_metrics["model_comparison"]),
    ]
    if prediction_skip_reason is not None:
        lines.extend(["", f"Predicted hazard plot note: {prediction_skip_reason}"])
    if fit_metrics["full_model"]["fit_warnings"]:
        lines.extend(["", "Captured full-model fit warnings:"])
        lines.extend(f"- {warning_text}" for warning_text in fit_metrics["full_model"]["fit_warnings"])
    lines.extend(
        [
            "",
            "Full model summary:",
            str(full_result.summary()),
            "",
            "Baseline time-only model summary:",
            str(baseline_result.summary()),
        ]
    )
    return "\n".join(lines)


def _assess_prediction_plot_pathology(
    *,
    fit_metrics: dict[str, Any],
    prediction_table: pd.DataFrame,
) -> tuple[bool, str | None]:
    """Decide whether the predicted-hazard plot should be skipped."""

    combined_warnings = fit_metrics["full_model"]["fit_warnings"] + fit_metrics["baseline_time_only_model"]["fit_warnings"]
    if any("Perfect separation" in warning_text for warning_text in combined_warnings):
        return True, "Skipped predicted hazard plot because statsmodels reported perfect separation."
    predicted_values = prediction_table["predicted_hazard"].to_numpy(dtype=float)
    if np.all((predicted_values <= 1.0e-6) | (predicted_values >= 1.0 - 1.0e-6)):
        return True, "Skipped predicted hazard plot because model predictions were numerically saturated at 0/1."
    return False, None


def _maybe_float(value: Any) -> float | None:
    """Convert a numeric fit metric to float when finite."""

    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value
