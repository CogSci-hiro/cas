"""Model fitting for the behavioral hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cas.behavior.formulas import render_formula, strip_random_effects

STANDARD_BACKEND_ID = "statsmodels_glm_binomial_no_random_effects"
STANDARD_BACKEND_NOTE = (
    "Requested formulas may include random intercept syntax such as `(1 | subject)`, "
    "but the active behavior wrapper strips random effects and fits a fixed-effects binomial GLM."
)
RIDGE_BACKEND_ID = "statsmodels_glm_binomial_ridge_no_random_effects"
RIDGE_BACKEND_NOTE = (
    "The unpenalized fixed-effects GLM was numerically unstable for this dataset/model, "
    "so the behavior wrapper refit a ridge-regularized binomial GLM with random effects still stripped."
)
REGULARIZED_ALPHA = 5e-4
LINEAR_PREDICTOR_CLIP = 25.0
PROBABILITY_CLIP = 1e-9
UNSTABLE_PARAM_THRESHOLD = 25.0
BOUNDARY_SHARE_THRESHOLD = 0.98
UNSTABLE_WARNING_SNIPPETS = (
    "overflow encountered in exp",
    "divide by zero encountered in log",
    "invalid value encountered in multiply",
)


@dataclass(frozen=True, slots=True)
class FittedBehaviorModel:
    model_id: str
    requested_formula: str
    fitted_formula: str
    result: object
    n_rows: int
    n_events: int
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    backend: str
    random_effects_requested: bool
    random_effects_applied: bool
    regularization_alpha: float | None
    notes: tuple[str, ...]
    warnings: tuple[str, ...]


def _captured_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for item in caught:
        message = str(item.message)
        if message:
            messages.append(message)
    return messages


def _safe_predict_probabilities(result: object, frame: pd.DataFrame) -> np.ndarray:
    try:
        probabilities = np.asarray(result.predict(frame), dtype=float)
    except Exception:
        probabilities = np.asarray([], dtype=float)
    probabilities = np.ravel(probabilities)
    if probabilities.size:
        finite = np.isfinite(probabilities)
        probabilities = probabilities.astype(float, copy=False)
        probabilities[~finite] = np.nan
        probabilities = np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return probabilities


def _manual_information_criteria(result: object, table: pd.DataFrame) -> tuple[float, float, float]:
    event = pd.to_numeric(table["event"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probabilities = _safe_predict_probabilities(result, table)
    if probabilities.size != event.size or probabilities.size == 0:
        return float("nan"), float("nan"), float("nan")
    log_likelihood = float(np.sum(event * np.log(probabilities) + (1.0 - event) * np.log1p(-probabilities)))
    n_parameters = int(len(getattr(result, "params", [])))
    n_rows = max(int(event.size), 1)
    aic = float(2.0 * n_parameters - 2.0 * log_likelihood)
    bic = float(np.log(n_rows) * n_parameters - 2.0 * log_likelihood)
    return log_likelihood, aic, bic


def _boundary_share(result: object, table: pd.DataFrame) -> float:
    probabilities = _safe_predict_probabilities(result, table)
    if probabilities.size == 0:
        return 1.0
    return float(np.mean((probabilities <= 1e-6) | (probabilities >= 1.0 - 1e-6)))


def _fit_is_unstable(result: object, fit_warnings: list[str], table: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    params = pd.to_numeric(getattr(result, "params", pd.Series(dtype=float)), errors="coerce")
    max_abs_param = float(params.abs().max()) if len(params) else float("nan")
    if not np.isfinite(max_abs_param):
        reasons.append("Model returned non-finite parameter estimates.")
    elif max_abs_param > UNSTABLE_PARAM_THRESHOLD:
        reasons.append(
            f"Model returned extreme coefficient magnitudes (max abs estimate {max_abs_param:.2f} > {UNSTABLE_PARAM_THRESHOLD:.2f})."
        )
    if any(snippet in warning.lower() for warning in fit_warnings for snippet in UNSTABLE_WARNING_SNIPPETS):
        reasons.append("Model emitted numerical overflow/log-likelihood warnings during fitting.")
    boundary_share = _boundary_share(result, table)
    if boundary_share >= BOUNDARY_SHARE_THRESHOLD:
        reasons.append(
            f"Model predictions were saturated at probability boundaries for {boundary_share:.1%} of rows."
        )
    return (len(reasons) > 0), reasons


def _fit_glm(model: sm.GLM) -> tuple[object, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit()
    return result, _captured_warning_messages(caught)


def _fit_regularized_glm(model: sm.GLM) -> tuple[object, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_regularized(alpha=REGULARIZED_ALPHA, L1_wt=0.0, maxiter=300)
    return result, _captured_warning_messages(caught)


def fit_formula_model(
    table: pd.DataFrame,
    *,
    model_id: str,
    formula: str,
    requested_formula: str | None = None,
) -> FittedBehaviorModel:
    original_formula = str(requested_formula or formula)
    fitted_formula = str(formula)
    model = sm.GLM.from_formula(fitted_formula, data=table, family=sm.families.Binomial())
    result, fit_warnings = _fit_glm(model)
    backend = STANDARD_BACKEND_ID
    regularization_alpha: float | None = None
    notes: list[str] = []
    unstable, instability_reasons = _fit_is_unstable(result, fit_warnings, table)
    if unstable:
        regularized_result, regularized_warnings = _fit_regularized_glm(model)
        regularized_log_likelihood, regularized_aic, regularized_bic = _manual_information_criteria(regularized_result, table)
        if np.isfinite(regularized_log_likelihood):
            result = regularized_result
            fit_warnings.extend(instability_reasons)
            fit_warnings.extend(regularized_warnings)
            backend = RIDGE_BACKEND_ID
            regularization_alpha = REGULARIZED_ALPHA
            notes.append(RIDGE_BACKEND_NOTE)
            log_likelihood = regularized_log_likelihood
            aic = regularized_aic
            bic = regularized_bic
        else:
            fit_warnings.extend(instability_reasons)
            log_likelihood, aic, bic = _manual_information_criteria(result, table)
    else:
        log_likelihood, aic, bic = _manual_information_criteria(result, table)
    if not np.isfinite(log_likelihood):
        fit_warnings.append("Non-finite log-likelihood returned by the active behavior model backend.")
    if not np.isfinite(aic):
        fit_warnings.append("Non-finite AIC returned by the active behavior model backend.")
    if not np.isfinite(bic):
        fit_warnings.append("Non-finite BIC returned by the active behavior model backend.")
    random_effects_requested = original_formula != fitted_formula or "(1 |" in original_formula
    if random_effects_requested:
        notes.insert(0, STANDARD_BACKEND_NOTE)
    return FittedBehaviorModel(
        model_id=model_id,
        requested_formula=original_formula,
        fitted_formula=fitted_formula,
        result=result,
        n_rows=int(len(table)),
        n_events=int(pd.to_numeric(table["event"], errors="coerce").fillna(0).sum()),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        converged=bool(getattr(result, "converged", True)),
        backend=backend,
        random_effects_requested=random_effects_requested,
        random_effects_applied=False,
        regularization_alpha=regularization_alpha,
        notes=tuple(notes),
        warnings=tuple(fit_warnings),
    )


def fit_registered_model(table: pd.DataFrame, *, model_id: str, lag_ms: int) -> FittedBehaviorModel:
    requested_formula = render_formula(model_id, lag_ms=lag_ms)
    return fit_formula_model(
        table,
        model_id=model_id,
        formula=strip_random_effects(requested_formula),
        requested_formula=requested_formula,
    )


def model_metadata(fitted: FittedBehaviorModel, *, selected_lag_ms: int | None = None) -> dict[str, object]:
    return {
        "model_id": fitted.model_id,
        "requested_formula": fitted.requested_formula,
        "fitted_formula": fitted.fitted_formula,
        "n_rows": fitted.n_rows,
        "n_events": fitted.n_events,
        "log_likelihood": fitted.log_likelihood,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "converged": fitted.converged,
        "backend": fitted.backend,
        "random_effects_requested": fitted.random_effects_requested,
        "random_effects_applied": fitted.random_effects_applied,
        "regularization_alpha": fitted.regularization_alpha,
        "selected_lag_ms": selected_lag_ms,
        "notes": list(fitted.notes),
        "warnings": list(fitted.warnings),
    }


def coefficient_rows(
    fitted: FittedBehaviorModel,
    *,
    dataset: str,
    selected_lag_ms: int,
) -> list[dict[str, object]]:
    conf = fitted.result.conf_int() if hasattr(fitted.result, "conf_int") else None
    bse = getattr(fitted.result, "bse", None)
    rows: list[dict[str, object]] = []
    for term in fitted.result.params.index:
        estimate = float(fitted.result.params[term])
        if bse is not None and term in bse.index:
            standard_error = float(bse[term])
        else:
            standard_error = np.nan
        z_value = estimate / standard_error if np.isfinite(standard_error) and standard_error != 0.0 else np.nan
        p_value = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        ci_low = float(conf.loc[term, 0]) if conf is not None and term in conf.index else np.nan
        ci_high = float(conf.loc[term, 1]) if conf is not None and term in conf.index else np.nan
        rows.append(
            {
                "dataset": dataset,
                "model_id": fitted.model_id,
                "term": str(term),
                "estimate": estimate,
                "std_error": standard_error,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "backend": fitted.backend,
                "selected_lag_ms": int(selected_lag_ms),
                "notes": " | ".join(fitted.notes),
                "z_value": z_value,
            }
        )
    return rows


def odds_ratio_rows(coefficients: pd.DataFrame) -> pd.DataFrame:
    out = coefficients.copy()
    out["odds_ratio"] = np.exp(pd.to_numeric(out["estimate"], errors="coerce"))
    out["ci_low"] = np.exp(pd.to_numeric(out["ci_low"], errors="coerce"))
    out["ci_high"] = np.exp(pd.to_numeric(out["ci_high"], errors="coerce"))
    required_columns = {
        "model_id": "",
        "term": "",
        "odds_ratio": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "selected_lag_ms": np.nan,
        "model_backend": "",
        "backend": "",
        "covariance_type": "",
        "cluster_variable": "",
        "notes": "",
    }
    for column, default in required_columns.items():
        if column not in out.columns:
            out[column] = default
    return out.loc[:, list(required_columns)]


def comparison_row(
    parent: FittedBehaviorModel,
    child: FittedBehaviorModel,
    *,
    dataset: str,
    selected_lag_ms: int,
) -> dict[str, object]:
    df_difference = float(getattr(child.result, "df_model", 0.0)) - float(getattr(parent.result, "df_model", 0.0))
    lr_stat = 2.0 * (child.log_likelihood - parent.log_likelihood)
    p_value = float(stats.chi2.sf(lr_stat, df_difference)) if df_difference > 0.0 else np.nan
    return {
        "dataset": dataset,
        "comparison_id": f"{parent.model_id}__vs__{child.model_id}",
        "model_reduced": parent.model_id,
        "model_full": child.model_id,
        "log_likelihood_reduced": parent.log_likelihood,
        "log_likelihood_full": child.log_likelihood,
        "delta_log_likelihood": child.log_likelihood - parent.log_likelihood,
        "lr_statistic": lr_stat,
        "statistic": lr_stat,
        "p_value": p_value,
        "backend": child.backend,
        "selected_lag_ms": int(selected_lag_ms),
        "notes": " | ".join(dict.fromkeys([*parent.notes, *child.notes])),
        "df_difference": df_difference,
        "delta_aic": child.aic - parent.aic,
        "delta_bic": child.bic - parent.bic,
    }


def prediction_summary_frame(fitted: FittedBehaviorModel, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    probabilities = _safe_predict_probabilities(fitted.result, frame)
    out["predicted_hazard"] = probabilities if probabilities.size else np.nan
    if hasattr(fitted.result, "get_prediction") and fitted.backend == STANDARD_BACKEND_ID:
        summary = fitted.result.get_prediction(frame).summary_frame()
        lower_column = "mean_ci_lower" if "mean_ci_lower" in summary.columns else None
        upper_column = "mean_ci_upper" if "mean_ci_upper" in summary.columns else None
        out["ci_low"] = pd.to_numeric(summary[lower_column], errors="coerce") if lower_column else np.nan
        out["ci_high"] = pd.to_numeric(summary[upper_column], errors="coerce") if upper_column else np.nan
    else:
        out["ci_low"] = np.nan
        out["ci_high"] = np.nan
    out["predicted_hazard"] = pd.to_numeric(out["predicted_hazard"], errors="coerce").clip(PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    out["ci_low"] = pd.to_numeric(out["ci_low"], errors="coerce")
    out["ci_high"] = pd.to_numeric(out["ci_high"], errors="coerce")
    return out
