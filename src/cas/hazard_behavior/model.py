"""Model fitting for behavioural discrete-time hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import statsmodels.api as sm

from cas.hazard_behavior.config import BehaviourHazardConfig

MODEL_PREDICTOR_MAP = {
    "M0": [],
    "M1": ["z_information_rate"],
    "M2a": ["z_information_rate", "z_prop_actual_cumulative_info"],
    "M2b": ["z_information_rate", "z_cumulative_info"],
    "M2c": ["z_information_rate", "z_prop_expected_cumulative_info"],
}


@dataclass(frozen=True, slots=True)
class FittedBehaviourModel:
    """Container for a fitted statsmodels GLM."""

    model_name: str
    formula: str
    result: Any
    summary_table: pd.DataFrame
    fit_metrics: dict[str, Any]
    robust_covariance_used: bool
    warnings: list[str]


def build_model_formulas(config: BehaviourHazardConfig) -> dict[str, str]:
    """Build the formula strings for the behavioural hazard models."""

    spline = (
        f"bs(time_from_partner_onset, df={config.baseline_spline_df}, "
        f"degree={config.baseline_spline_degree}, include_intercept=False)"
    )
    formulas: dict[str, str] = {}
    for model_name, predictors in MODEL_PREDICTOR_MAP.items():
        rhs = [spline] + predictors
        formulas[model_name] = "event ~ " + " + ".join(rhs)
    return formulas


def fit_binomial_glm(
    riskset_table: pd.DataFrame,
    *,
    model_name: str,
    config: BehaviourHazardConfig,
) -> FittedBehaviourModel:
    """Fit one behavioural binomial GLM with optional cluster-robust covariance."""

    formulas = build_model_formulas(config)
    formula = formulas[model_name]
    cluster_column = _resolve_cluster_column(riskset_table, config.cluster_column)
    warnings_list: list[str] = []
    robust_covariance_used = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm = sm.GLM.from_formula(formula, data=riskset_table, family=sm.families.Binomial())
        try:
            fitted = glm.fit(cov_type="cluster", cov_kwds={"groups": riskset_table[cluster_column]})
        except Exception as error:  # pragma: no cover - fallback path
            fitted = glm.fit()
            robust_covariance_used = False
            warnings_list.append(
                f"Cluster-robust covariance failed for {model_name}; fell back to non-robust covariance: {error}"
            )
        warnings_list.extend(str(item.message) for item in caught)

    summary_table = extract_model_summary(fitted_result=fitted, model_name=model_name)
    fit_metrics = compute_fit_metrics(
        fitted_result=fitted,
        riskset_table=riskset_table,
        model_name=model_name,
        formula=formula,
        cluster_column=cluster_column,
        robust_covariance_used=robust_covariance_used,
        warnings_list=warnings_list,
    )
    return FittedBehaviourModel(
        model_name=model_name,
        formula=formula,
        result=fitted,
        summary_table=summary_table,
        fit_metrics=fit_metrics,
        robust_covariance_used=robust_covariance_used,
        warnings=warnings_list,
    )


def extract_model_summary(*, fitted_result: Any, model_name: str) -> pd.DataFrame:
    """Extract a tidy model summary table."""

    conf = fitted_result.conf_int()
    summary = pd.DataFrame(
        {
            "model_name": model_name,
            "term": fitted_result.params.index.astype(str),
            "estimate": np.asarray(fitted_result.params, dtype=float),
            "standard_error": np.asarray(fitted_result.bse, dtype=float),
            "z_value": np.asarray(fitted_result.tvalues, dtype=float),
            "p_value": np.asarray(fitted_result.pvalues, dtype=float),
            "conf_low": np.asarray(conf.iloc[:, 0], dtype=float),
            "conf_high": np.asarray(conf.iloc[:, 1], dtype=float),
        }
    )
    summary["odds_ratio"] = np.exp(summary["estimate"])
    summary["odds_ratio_conf_low"] = np.exp(summary["conf_low"])
    summary["odds_ratio_conf_high"] = np.exp(summary["conf_high"])
    return summary


def compute_fit_metrics(
    *,
    fitted_result: Any,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    cluster_column: str,
    robust_covariance_used: bool,
    warnings_list: list[str],
) -> dict[str, Any]:
    """Compute model fit and in-sample predictive metrics."""

    predicted = np.asarray(fitted_result.predict(riskset_table), dtype=float)
    observed = riskset_table["event"].to_numpy(dtype=int)
    predicted_metrics = np.clip(predicted, 1.0e-8, 1.0 - 1.0e-8)
    log_loss_value: float | None = None
    brier_score_value: float | None = None
    auroc = None
    if np.isfinite(predicted_metrics).all():
        log_loss_value = float(log_loss(observed, predicted_metrics, labels=[0, 1]))
        brier_score_value = float(brier_score_loss(observed, predicted_metrics))
        if len(np.unique(observed)) > 1:
            auroc = float(roc_auc_score(observed, predicted_metrics))
    else:
        warnings_list.append(
            f"Predicted probabilities for {model_name} contained NaN or infinite values; in-sample predictive metrics were set to null."
        )
    return {
        "model_name": model_name,
        "n_rows": int(len(riskset_table)),
        "n_events": int(riskset_table["event"].sum()),
        "n_episodes": int(riskset_table["episode_id"].nunique()),
        "log_likelihood": _maybe_float(getattr(fitted_result, "llf", None)),
        "aic": _maybe_float(getattr(fitted_result, "aic", None)),
        "bic": _maybe_float(getattr(fitted_result, "bic", None)),
        "pseudo_r2": _compute_pseudo_r2(fitted_result),
        "converged": bool(getattr(fitted_result, "converged", True)),
        "warnings": warnings_list,
        "formula": formula,
        "cluster_variable": cluster_column,
        "robust_covariance_used": robust_covariance_used,
        "log_loss_in_sample": log_loss_value,
        "brier_score_in_sample": brier_score_value,
        "auroc_in_sample": auroc,
    }


def compare_nested_models(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    """Compare nested behavioural models using likelihood-ratio tests."""

    comparisons = [("M1", "M0"), ("M2a", "M1"), ("M2b", "M1"), ("M2c", "M1")]
    rows: list[dict[str, object]] = []
    for full_name, reduced_name in comparisons:
        full = fitted_models[full_name].result
        reduced = fitted_models[reduced_name].result
        lr_stat = 2.0 * (float(full.llf) - float(reduced.llf))
        df_diff = int(len(full.params) - len(reduced.params))
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
        rows.append(
            {
                "comparison": f"{full_name} vs {reduced_name}",
                "full_model": full_name,
                "reduced_model": reduced_name,
                "likelihood_ratio_statistic": lr_stat,
                "df_difference": df_diff,
                "p_value": p_value,
                "delta_aic": _safe_difference(
                    _maybe_float(getattr(full, "aic", None)),
                    _maybe_float(getattr(reduced, "aic", None)),
                ),
                "delta_bic": _safe_difference(_maybe_float(getattr(full, "bic", None)), _maybe_float(getattr(reduced, "bic", None))),
                "log_loss_in_sample": fitted_models[full_name].fit_metrics["log_loss_in_sample"],
                "brier_score_in_sample": fitted_models[full_name].fit_metrics["brier_score_in_sample"],
                "auroc_in_sample": fitted_models[full_name].fit_metrics["auroc_in_sample"],
            }
        )
    return pd.DataFrame(rows)


def generate_prediction_grids(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    config: BehaviourHazardConfig,
) -> dict[str, pd.DataFrame]:
    """Generate prediction grids for required model plots."""

    grids: dict[str, pd.DataFrame] = {}
    time_values = np.linspace(
        float(riskset_table["time_from_partner_onset"].min()),
        float(riskset_table["time_from_partner_onset"].max()),
        num=100,
    )
    grids["predicted_hazard_by_time"] = _predict_for_grid(
        fitted_model=fitted_models["M0"],
        predictor_name="time_from_partner_onset",
        predictor_values=time_values,
    )
    predictor_grid_map = {
        "predicted_hazard_by_information_rate": ("M1", "z_information_rate"),
        "predicted_hazard_by_prop_actual_cumulative_info": ("M2a", "z_prop_actual_cumulative_info"),
        "predicted_hazard_by_cumulative_info": ("M2b", "z_cumulative_info"),
        "predicted_hazard_by_prop_expected_cumulative_info": ("M2c", "z_prop_expected_cumulative_info"),
    }
    for grid_name, (model_name, predictor_name) in predictor_grid_map.items():
        grids[grid_name] = _predict_for_grid(
            fitted_model=fitted_models[model_name],
            predictor_name=predictor_name,
            predictor_values=np.linspace(-2.5, 2.5, num=100),
        )
    return grids


def _predict_for_grid(
    *,
    fitted_model: FittedBehaviourModel,
    predictor_name: str,
    predictor_values: np.ndarray,
) -> pd.DataFrame:
    base_row = {
        "time_from_partner_onset": 0.5,
        "z_information_rate": 0.0,
        "z_prop_actual_cumulative_info": 0.0,
        "z_cumulative_info": 0.0,
        "z_prop_expected_cumulative_info": 0.0,
    }
    if predictor_name == "time_from_partner_onset":
        rows = [{**base_row, "time_from_partner_onset": float(value)} for value in predictor_values]
    else:
        rows = [{**base_row, predictor_name: float(value)} for value in predictor_values]
    prediction_frame = pd.DataFrame(rows)
    predicted = np.asarray(fitted_model.result.predict(prediction_frame), dtype=float)
    output = prediction_frame.copy()
    output["predicted_hazard"] = predicted
    output["predictor_name"] = predictor_name
    return output


def _resolve_cluster_column(riskset_table: pd.DataFrame, preferred_column: str | None) -> str:
    candidates = [preferred_column, "subject_id", "dyad_id", "episode_id"]
    for candidate in candidates:
        if candidate and candidate in riskset_table.columns:
            return candidate
    return "episode_id"


def _compute_pseudo_r2(fitted_result: Any) -> float | None:
    llf = _maybe_float(getattr(fitted_result, "llf", None))
    llnull = _maybe_float(getattr(fitted_result, "llnull", None))
    if llf is None or llnull in (None, 0.0):
        return None
    return float(1.0 - (llf / llnull))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)
