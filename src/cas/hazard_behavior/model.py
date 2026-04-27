"""Active behavioural timing-control hazard models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as glm

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.features import zscore_predictors

glm.SET_USE_BIC_LLF(True)

FINAL_MODEL_SEQUENCE = ("M0_timing", "M1_rate", "M2_expected")


@dataclass(frozen=True, slots=True)
class FittedBehaviourModel:
    """Container for a fitted timing-control model."""

    model_name: str
    formula: str
    result: Any
    summary_table: pd.DataFrame
    fit_metrics: dict[str, Any]
    robust_covariance_used: bool
    warnings: list[str]


def primary_unscaled_column_name(feature_name: str, lag_ms: int) -> str:
    return f"{feature_name}_lag_{int(lag_ms)}ms"


def primary_z_column_name(feature_name: str, lag_ms: int) -> str:
    return f"z_{primary_unscaled_column_name(feature_name, lag_ms)}"


def build_timing_control_baseline_terms(config: BehaviourHazardConfig) -> str:
    onset_spline = (
        f"bs(time_from_partner_onset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    offset_spline = (
        f"bs(time_from_partner_offset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    return f"{onset_spline} + {offset_spline}"


def build_timing_control_model_formulas(config: BehaviourHazardConfig) -> dict[str, str]:
    baseline_terms = build_timing_control_baseline_terms(config)
    information_rate_term = primary_z_column_name("information_rate", config.primary_information_rate_lag_ms)
    prop_expected_term = primary_z_column_name(
        "prop_expected_cumulative_info",
        config.primary_prop_expected_lag_ms,
    )
    return {
        "M0_timing": f"event ~ {baseline_terms}",
        "M1_rate": f"event ~ {baseline_terms} + {information_rate_term}",
        "M2_expected": f"event ~ {baseline_terms} + {information_rate_term} + {prop_expected_term}",
    }


def ensure_primary_predictors_available(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    required_unscaled = [
        primary_unscaled_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_unscaled_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
    ]
    missing = [column_name for column_name in required_unscaled if column_name not in riskset_table.columns]
    if missing:
        raise ValueError(
            "Timing-control behavioural models require lagged predictor columns that are missing: "
            + ", ".join(missing)
        )

    table = riskset_table.copy()
    required_z = [f"z_{column_name}" for column_name in required_unscaled]
    missing_z = [column_name for column_name in required_z if column_name not in table.columns]
    scaling: dict[str, dict[str, float]] = {}
    if missing_z:
        zscore_result = zscore_predictors(table, predictors=required_unscaled)
        table = zscore_result.table
        scaling = {name: zscore_result.scaling[name] for name in required_unscaled if name in zscore_result.scaling}
    return table, scaling


def validate_timing_control_model_table(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    if config.episode_anchor != "partner_ipu":
        raise ValueError("Timing-control behavioural models require partner-IPU anchored risk sets.")
    if riskset_table.empty:
        raise ValueError("Timing-control behavioural model risk-set table is empty.")

    required_columns = [
        "event",
        "episode_id",
        "time_from_partner_onset",
        "time_from_partner_offset",
    ]
    missing = [column_name for column_name in required_columns if column_name not in riskset_table.columns]
    if missing:
        raise ValueError("Timing-control behavioural model table is missing required columns: " + ", ".join(missing))

    working = riskset_table.copy()
    working["event"] = pd.to_numeric(working["event"], errors="coerce")
    if working["event"].isna().any() or not working["event"].isin([0, 1]).all():
        raise ValueError("Timing-control behavioural model requires the `event` column to contain integer 0/1 values.")
    for column_name in ("time_from_partner_onset", "time_from_partner_offset"):
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
        if not np.isfinite(working[column_name]).all():
            raise ValueError(f"Timing-control behavioural model requires finite values in `{column_name}`.")
    if int(working["event"].sum()) <= 0:
        raise ValueError("Timing-control behavioural model requires at least one event row.")
    if int((working["event"] == 0).sum()) <= 0:
        raise ValueError("Timing-control behavioural model requires at least one non-event row.")
    return working


def fit_timing_control_behaviour_models(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> dict[str, FittedBehaviourModel]:
    validated = validate_timing_control_model_table(riskset_table, config=config)
    validated, _scaling = ensure_primary_predictors_available(validated, config=config)
    validated = _prepare_timing_control_model_table(validated, config=config)
    formulas = build_timing_control_model_formulas(config)
    fitted_models: dict[str, FittedBehaviourModel] = {}
    for model_name in FINAL_MODEL_SEQUENCE:
        fitted_models[model_name] = _fit_formula_model(
            riskset_table=validated,
            model_name=model_name,
            formula=formulas[model_name],
            config=config,
        )
    return fitted_models


def select_timing_control_best_lags(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validated = validate_timing_control_model_table(riskset_table, config=config)
    lag_grid = tuple(int(lag_ms) for lag_ms in config.lag_grid_ms)
    lagged_predictors = [
        primary_unscaled_column_name("information_rate", lag_ms) for lag_ms in lag_grid
    ] + [
        primary_unscaled_column_name("prop_expected_cumulative_info", lag_ms) for lag_ms in lag_grid
    ]
    validated = _ensure_z_predictors_available(validated, predictors=lagged_predictors)
    baseline_terms = build_timing_control_baseline_terms(config)
    baseline_columns = ["event", "episode_id", "time_from_partner_onset", "time_from_partner_offset"]

    selection_rows: list[dict[str, Any]] = []

    rate_terms = [primary_z_column_name("information_rate", lag_ms) for lag_ms in lag_grid]
    rate_table = _shared_complete_case_subset(validated, required_columns=baseline_columns + rate_terms)
    rate_parent_model = _fit_formula_model(
        riskset_table=rate_table,
        model_name="M0_timing",
        formula=f"event ~ {baseline_terms}",
        config=config,
    )
    for lag_ms in lag_grid:
        term = primary_z_column_name("information_rate", lag_ms)
        child_name = f"M1_rate_lag_{int(lag_ms)}ms"
        child_model = _fit_formula_model(
            riskset_table=rate_table,
            model_name=child_name,
            formula=f"event ~ {baseline_terms} + {term}",
            config=config,
        )
        comparison = compare_nested_models(
            {"M0_timing": rate_parent_model, child_name: child_model},
            comparisons=[(child_name, "M0_timing")],
        )
        selection_rows.extend(
            _selection_rows(comparison, predictor_family="information_rate", lag_ms=lag_ms)
        )

    rate_selection = pd.DataFrame(selection_rows)
    best_information_rate = select_best_timing_control_lag(rate_selection, predictor_family="information_rate")
    best_information_rate_lag_ms = int(best_information_rate["lag_ms"])
    best_information_rate_term = primary_z_column_name("information_rate", best_information_rate_lag_ms)

    expected_terms = [primary_z_column_name("prop_expected_cumulative_info", lag_ms) for lag_ms in lag_grid]
    expected_table = _shared_complete_case_subset(
        validated,
        required_columns=baseline_columns + [best_information_rate_term] + expected_terms,
    )
    expected_parent_model = _fit_formula_model(
        riskset_table=expected_table,
        model_name="M1_rate",
        formula=f"event ~ {baseline_terms} + {best_information_rate_term}",
        config=config,
    )
    for lag_ms in lag_grid:
        term = primary_z_column_name("prop_expected_cumulative_info", lag_ms)
        child_name = f"M2_expected_lag_{int(lag_ms)}ms"
        child_model = _fit_formula_model(
            riskset_table=expected_table,
            model_name=child_name,
            formula=f"event ~ {baseline_terms} + {best_information_rate_term} + {term}",
            config=config,
        )
        comparison = compare_nested_models(
            {"M1_rate": expected_parent_model, child_name: child_model},
            comparisons=[(child_name, "M1_rate")],
        )
        selection_rows.extend(
            _selection_rows(comparison, predictor_family="prop_expected_cumulative_info", lag_ms=lag_ms)
        )

    selection_table = pd.DataFrame(selection_rows)
    best_expected = select_best_timing_control_lag(
        selection_table,
        predictor_family="prop_expected_cumulative_info",
    )
    selected_lags = {
        "best_information_rate_lag_ms": best_information_rate_lag_ms,
        "best_information_rate_delta_aic": float(best_information_rate["delta_aic"]),
        "best_expected_cumulative_info_lag_ms": int(best_expected["lag_ms"]),
        "best_expected_cumulative_info_delta_aic": float(best_expected["delta_aic"]),
        "lag_selection_parent_for_information_rate": "M0_timing",
        "lag_selection_parent_for_expected_cumulative_info": "M1_rate",
        "delta_aic_convention": "child_aic - parent_aic; negative favours child",
        "delta_bic_convention": "child_bic - parent_bic; negative favours child",
    }
    return selection_table, selected_lags


def select_best_timing_control_lag(
    selection_table: pd.DataFrame,
    *,
    predictor_family: str,
) -> pd.Series:
    family_rows = selection_table.loc[
        selection_table["predictor_family"].astype(str) == str(predictor_family)
    ].copy()
    if family_rows.empty:
        raise ValueError(f"No timing-control lag rows were found for predictor family `{predictor_family}`.")
    family_rows["delta_aic"] = pd.to_numeric(family_rows["delta_aic"], errors="coerce")
    family_rows["lag_ms"] = pd.to_numeric(family_rows["lag_ms"], errors="coerce")
    family_rows = family_rows.loc[np.isfinite(family_rows["delta_aic"]) & np.isfinite(family_rows["lag_ms"])].copy()
    if family_rows.empty:
        raise ValueError(f"No finite timing-control lag rows were found for `{predictor_family}`.")
    return family_rows.sort_values(["delta_aic", "lag_ms"], ascending=[True, True]).reset_index(drop=True).iloc[0]


def compare_timing_control_models(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    comparison = compare_nested_models(
        fitted_models,
        comparisons=[
            ("M1_rate", "M0_timing"),
            ("M2_expected", "M1_rate"),
        ],
    )
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "parent_model",
                "child_model",
                "comparison",
                "parent_aic",
                "child_aic",
                "delta_aic",
                "parent_bic",
                "child_bic",
                "delta_bic",
                "parent_log_likelihood",
                "child_log_likelihood",
                "lrt_statistic",
                "df_difference",
                "p_value",
                "n_rows",
                "n_events",
                "log_loss_in_sample",
                "brier_score_in_sample",
                "auroc_in_sample",
            ]
        )
    return comparison.loc[
        :,
        [
            "parent_model",
            "child_model",
            "comparison",
            "parent_aic",
            "child_aic",
            "delta_aic",
            "parent_bic",
            "child_bic",
            "delta_bic",
            "parent_log_likelihood",
            "child_log_likelihood",
            "lrt_statistic",
            "df_difference",
            "p_value",
            "n_rows",
            "n_events",
            "log_loss_in_sample",
            "brier_score_in_sample",
            "auroc_in_sample",
        ],
    ]


def summarize_timing_control_fit_metrics(
    fitted_models: dict[str, FittedBehaviourModel],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in FINAL_MODEL_SEQUENCE:
        fitted_model = fitted_models[model_name]
        fit_metrics = fitted_model.fit_metrics
        rows.append(
            {
                "model_name": model_name,
                "formula": fitted_model.formula,
                "n_rows": fit_metrics.get("n_rows"),
                "n_events": fit_metrics.get("n_events"),
                "log_likelihood": fit_metrics.get("log_likelihood"),
                "aic": fit_metrics.get("aic"),
                "bic": fit_metrics.get("bic"),
                "converged": fit_metrics.get("converged"),
                "warnings": list(fitted_model.warnings),
                "cluster_column": fit_metrics.get("cluster_variable"),
                "robust_covariance_used": fit_metrics.get("robust_covariance_used"),
            }
        )
    return rows


def extract_model_summary(*, fitted_result: Any, model_name: str) -> pd.DataFrame:
    conf = fitted_result.conf_int()
    summary = pd.DataFrame(
        {
            "model_name": model_name,
            "term": fitted_result.params.index.astype(str),
            "estimate": np.asarray(fitted_result.params, dtype=float),
            "standard_error": np.asarray(fitted_result.bse, dtype=float),
            "statistic": np.asarray(fitted_result.tvalues, dtype=float),
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
            f"Predicted probabilities for {model_name} contained NaN or infinite values; predictive metrics were omitted."
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
    *,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for child_name, parent_name in comparisons:
        if child_name not in fitted_models or parent_name not in fitted_models:
            continue
        child = fitted_models[child_name].result
        parent = fitted_models[parent_name].result
        lr_stat = 2.0 * (float(child.llf) - float(parent.llf))
        df_diff = int(len(child.params) - len(parent.params))
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
        rows.append(
            {
                "lag_ms": _extract_lag_ms(child_name),
                "child_model": child_name,
                "parent_model": parent_name,
                "comparison": f"{child_name} vs {parent_name}",
                "child_aic": _maybe_float(getattr(child, "aic", None)),
                "parent_aic": _maybe_float(getattr(parent, "aic", None)),
                "child_bic": _maybe_float(getattr(child, "bic", None)),
                "parent_bic": _maybe_float(getattr(parent, "bic", None)),
                "child_log_likelihood": _maybe_float(getattr(child, "llf", None)),
                "parent_log_likelihood": _maybe_float(getattr(parent, "llf", None)),
                "lrt_statistic": lr_stat,
                "df_difference": df_diff,
                "p_value": p_value,
                "delta_aic": _safe_difference(
                    _maybe_float(getattr(child, "aic", None)),
                    _maybe_float(getattr(parent, "aic", None)),
                ),
                "delta_bic": _safe_difference(
                    _maybe_float(getattr(child, "bic", None)),
                    _maybe_float(getattr(parent, "bic", None)),
                ),
                "n_rows": int(fitted_models[child_name].fit_metrics["n_rows"]),
                "n_events": int(fitted_models[child_name].fit_metrics["n_events"]),
                "log_loss_in_sample": fitted_models[child_name].fit_metrics["log_loss_in_sample"],
                "brier_score_in_sample": fitted_models[child_name].fit_metrics["brier_score_in_sample"],
                "auroc_in_sample": fitted_models[child_name].fit_metrics["auroc_in_sample"],
            }
        )
    return pd.DataFrame(rows)


def _fit_formula_model(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    config: BehaviourHazardConfig,
) -> FittedBehaviourModel:
    cluster_column = _resolve_cluster_column(riskset_table, config.cluster_column)
    warnings_list: list[str] = []
    robust_covariance_used = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm_model = sm.GLM.from_formula(formula, data=riskset_table, family=sm.families.Binomial())
        try:
            fitted = glm_model.fit(cov_type="cluster", cov_kwds={"groups": riskset_table[cluster_column]})
        except Exception as error:  # pragma: no cover
            fitted = glm_model.fit()
            robust_covariance_used = False
            warnings_list.append(
                f"Cluster-robust covariance failed for {model_name}; fell back to non-robust covariance: {error}"
            )
        warnings_list.extend(str(item.message) for item in caught)
    return FittedBehaviourModel(
        model_name=model_name,
        formula=formula,
        result=fitted,
        summary_table=extract_model_summary(fitted_result=fitted, model_name=model_name),
        fit_metrics=compute_fit_metrics(
            fitted_result=fitted,
            riskset_table=riskset_table,
            model_name=model_name,
            formula=formula,
            cluster_column=cluster_column,
            robust_covariance_used=robust_covariance_used,
            warnings_list=warnings_list,
        ),
        robust_covariance_used=robust_covariance_used,
        warnings=warnings_list,
    )


def _prepare_timing_control_model_table(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    required_columns = [
        "event",
        "episode_id",
        "time_from_partner_onset",
        "time_from_partner_offset",
        primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
    ]
    return _shared_complete_case_subset(riskset_table, required_columns=required_columns)


def _ensure_z_predictors_available(
    riskset_table: pd.DataFrame,
    *,
    predictors: list[str],
) -> pd.DataFrame:
    missing_unscaled = [column_name for column_name in predictors if column_name not in riskset_table.columns]
    if missing_unscaled:
        raise ValueError("Required lagged predictor columns are missing: " + ", ".join(sorted(missing_unscaled)))
    missing_z = [f"z_{column_name}" for column_name in predictors if f"z_{column_name}" not in riskset_table.columns]
    if not missing_z:
        return riskset_table
    return zscore_predictors(riskset_table, predictors=predictors).table


def _shared_complete_case_subset(
    riskset_table: pd.DataFrame,
    *,
    required_columns: list[str],
) -> pd.DataFrame:
    working = riskset_table.copy()
    missing = [column_name for column_name in required_columns if column_name not in working.columns]
    if missing:
        raise ValueError("Required columns are missing from the model table: " + ", ".join(sorted(missing)))
    for column_name in required_columns:
        if column_name != "episode_id":
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    mask = np.ones(len(working), dtype=bool)
    for column_name in required_columns:
        if column_name == "episode_id":
            mask &= working[column_name].notna().to_numpy()
        else:
            mask &= np.isfinite(working[column_name].to_numpy(dtype=float))
    subset = working.loc[mask].copy()
    if subset.empty:
        raise ValueError("Shared complete-case subsetting removed all rows from the timing-control model table.")
    subset["event"] = subset["event"].astype(int)
    return subset


def _selection_rows(
    comparison: pd.DataFrame,
    *,
    predictor_family: str,
    lag_ms: int,
) -> list[dict[str, Any]]:
    if comparison.empty:
        return []
    working = comparison.copy()
    working["predictor_family"] = predictor_family
    working["lag_ms"] = int(lag_ms)
    return working.loc[
        :,
        [
            "predictor_family",
            "lag_ms",
            "parent_model",
            "child_model",
            "parent_aic",
            "child_aic",
            "delta_aic",
            "parent_bic",
            "child_bic",
            "delta_bic",
            "parent_log_likelihood",
            "child_log_likelihood",
            "lrt_statistic",
            "df_difference",
            "p_value",
            "n_rows",
            "n_events",
            "log_loss_in_sample",
            "brier_score_in_sample",
            "auroc_in_sample",
        ],
    ].to_dict(orient="records")


def _resolve_cluster_column(riskset_table: pd.DataFrame, preferred_column: str | None) -> str:
    candidates = [preferred_column, "participant_speaker_id", "participant_speaker", "subject_id", "dyad_id", "episode_id"]
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


def _extract_lag_ms(model_name: str) -> int | None:
    match = re.search(r"_lag_(\d+)ms$", model_name)
    if not match:
        return None
    return int(match.group(1))
