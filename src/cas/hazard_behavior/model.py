"""Model fitting for behavioural discrete-time hazard analysis."""

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

MODEL_PREDICTOR_MAP = {
    "M0": [],
    "M1": ["z_information_rate"],
    "M2a": ["z_information_rate", "z_prop_actual_cumulative_info"],
    "M2b": ["z_information_rate", "z_cumulative_info"],
    "M2c": ["z_information_rate", "z_prop_expected_cumulative_info"],
}
LAGGED_MODEL_FAMILY_PREDICTORS = {
    "information_rate": ("information_rate",),
    "prop_actual": ("information_rate", "prop_actual_cumulative_info"),
    "cumulative_info": ("information_rate", "cumulative_info"),
    "prop_expected": ("information_rate", "prop_expected_cumulative_info"),
}
LAGGED_MODEL_FAMILY_NAMES = {
    "information_rate": "M1_rate",
    "prop_actual": "M2a_prop_actual",
    "cumulative_info": "M2b_cumulative",
    "prop_expected": "M2c_prop_expected",
}
PRIMARY_MODEL_SEQUENCE = ("M0_time", "M1_rate", "M2_rate_prop_expected")
TIMING_CONTROL_MODEL_SEQUENCE = ("M0_timing", "M1_rate_best_timing", "M2_expected_best_timing")


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
    if config.fit_lagged_models:
        for model_name, predictors, _lag_ms, _family in build_lagged_model_specs(config):
            rhs = [spline] + list(predictors)
            formulas[model_name] = "event ~ " + " + ".join(rhs)
    return formulas


def build_lagged_model_specs(config: BehaviourHazardConfig) -> list[tuple[str, tuple[str, ...], int, str]]:
    """Return lagged model names and predictor lists."""

    specs: list[tuple[str, tuple[str, ...], int, str]] = []
    for lag_ms in config.lag_grid_ms:
        for family_name, base_predictors in LAGGED_MODEL_FAMILY_PREDICTORS.items():
            model_prefix = LAGGED_MODEL_FAMILY_NAMES[family_name]
            predictors = tuple(f"z_{predictor}_lag_{int(lag_ms)}ms" for predictor in base_predictors)
            specs.append((f"{model_prefix}_lag_{int(lag_ms)}", predictors, int(lag_ms), family_name))
    return specs


def build_primary_model_formulas(config: BehaviourHazardConfig) -> dict[str, str]:
    """Build formulas for the compact primary behavioural model sequence."""

    spline = (
        f"bs(time_from_partner_onset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    information_rate_term = primary_z_column_name("information_rate", config.primary_information_rate_lag_ms)
    prop_expected_term = primary_z_column_name(
        "prop_expected_cumulative_info",
        config.primary_prop_expected_lag_ms,
    )
    return {
        "M0_time": f"event ~ {spline}",
        "M1_rate": f"event ~ {spline} + {information_rate_term}",
        "M2_rate_prop_expected": f"event ~ {spline} + {information_rate_term} + {prop_expected_term}",
    }


def build_timing_control_model_formulas(config: BehaviourHazardConfig) -> dict[str, str]:
    """Build formulas for the onset-plus-offset timing-control models."""

    baseline_terms = build_timing_control_baseline_terms(config)
    information_rate_term = primary_z_column_name("information_rate", config.primary_information_rate_lag_ms)
    prop_expected_term = primary_z_column_name(
        "prop_expected_cumulative_info",
        config.primary_prop_expected_lag_ms,
    )
    return {
        "M0_timing": f"event ~ {baseline_terms}",
        "M1_rate_best_timing": f"event ~ {baseline_terms} + {information_rate_term}",
        "M2_expected_best_timing": f"event ~ {baseline_terms} + {information_rate_term} + {prop_expected_term}",
    }


def build_timing_control_baseline_terms(config: BehaviourHazardConfig) -> str:
    """Return the onset-plus-offset timing-control spline block."""

    onset_spline = (
        f"bs(time_from_partner_onset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    offset_spline = (
        f"bs(time_from_partner_offset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    return f"{onset_spline} + {offset_spline}"


def primary_unscaled_column_name(feature_name: str, lag_ms: int) -> str:
    """Return the unscaled primary predictor column name."""

    return f"{feature_name}_lag_{int(lag_ms)}ms"


def primary_z_column_name(feature_name: str, lag_ms: int) -> str:
    """Return the z-scored primary predictor column name."""

    return f"z_{primary_unscaled_column_name(feature_name, lag_ms)}"


def ensure_primary_predictors_available(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Ensure required primary lagged predictors and z-scored variants exist."""

    required_unscaled = [
        primary_unscaled_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_unscaled_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
    ]
    missing = [column_name for column_name in required_unscaled if column_name not in riskset_table.columns]
    if missing:
        raise ValueError(
            "Primary behavioural model requires lagged predictor columns that are missing: "
            + ", ".join(missing)
        )

    table = riskset_table.copy()
    scaling: dict[str, dict[str, float]] = {}
    required_z = [f"z_{column_name}" for column_name in required_unscaled]
    missing_z = [column_name for column_name in required_z if column_name not in table.columns]
    if missing_z:
        zscore_result = zscore_predictors(table, predictors=required_unscaled)
        table = zscore_result.table
        scaling = {name: zscore_result.scaling[name] for name in required_unscaled if name in zscore_result.scaling}
    return table, scaling


def validate_primary_model_table(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Validate the compact primary-model dataset and return a safe copy."""

    if config.episode_anchor != "partner_ipu":
        raise ValueError("Primary behavioural models require partner-IPU anchored risk sets (`episode_anchor='partner_ipu'`).")
    if riskset_table.empty:
        raise ValueError("Primary behavioural model risk-set table is empty.")
    if "anchor_source" in riskset_table.columns:
        anchor_values = set(riskset_table["anchor_source"].dropna().astype(str).unique())
        invalid_anchor_values = {
            value for value in anchor_values if value and "partner" not in value and "ipu" not in value
        }
        if invalid_anchor_values:
            raise ValueError(
                "Primary behavioural models found non-partner-IPU anchor rows: "
                + ", ".join(sorted(invalid_anchor_values))
            )

    if "event" not in riskset_table.columns:
        raise ValueError("Primary behavioural model risk-set table is missing required column: event")
    event_values = pd.to_numeric(riskset_table["event"], errors="coerce")
    if event_values.isna().any() or not event_values.isin([0, 1]).all():
        raise ValueError("Primary behavioural model requires the `event` column to contain integer 0/1 values only.")

    if "episode_id" not in riskset_table.columns:
        raise ValueError("Primary behavioural model risk-set table is missing required column: episode_id")
    if "episode_has_event" not in riskset_table.columns:
        raise ValueError("Primary behavioural model risk-set table is missing required column: episode_has_event")
    episode_flags = (
        pd.to_numeric(riskset_table["episode_has_event"], errors="coerce")
        .fillna(0)
        .astype(int)
        .groupby(riskset_table["episode_id"])
        .max()
    )
    event_counts = (
        pd.to_numeric(riskset_table["event"], errors="coerce")
        .fillna(0)
        .astype(int)
        .groupby(riskset_table["episode_id"])
        .sum()
    )
    positive_failures = event_counts.loc[(episode_flags == 1) & (event_counts != 1)]
    censored_failures = event_counts.loc[(episode_flags == 0) & (event_counts != 0)]
    if not positive_failures.empty:
        raise ValueError(
            "Primary behavioural validation failed: event-positive episodes must have exactly one event row. "
            f"Failed episode ids: {positive_failures.index.tolist()[:5]}"
        )
    if not censored_failures.empty:
        raise ValueError(
            "Primary behavioural validation failed: censored episodes must have zero event rows. "
            f"Failed episode ids: {censored_failures.index.tolist()[:5]}"
        )
    working, _scaling = ensure_primary_predictors_available(riskset_table, config=config)
    required_columns = [
        "time_from_partner_onset",
        primary_unscaled_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_unscaled_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
        primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
    ]
    for column_name in required_columns:
        if column_name not in working.columns:
            raise ValueError(f"Primary behavioural validation failed: missing required predictor column `{column_name}`.")
        numeric = pd.to_numeric(working[column_name], errors="coerce")
        if not np.isfinite(numeric).all():
            raise ValueError(
                "Primary behavioural validation failed: required predictor contains missing or non-finite values: "
                f"{column_name}"
            )
        if column_name.startswith("z_") or column_name.endswith("ms"):
            finite = numeric[np.isfinite(numeric)]
            if finite.empty or np.isclose(float(finite.min()), float(finite.max())):
                raise ValueError(
                    "Primary behavioural validation failed: required predictor is constant or undefined and "
                    f"cannot be interpreted: {column_name}"
                )

    if int(event_values.sum()) <= 0:
        raise ValueError("Primary behavioural model requires at least one event row.")
    if int((event_values == 0).sum()) <= 0:
        raise ValueError("Primary behavioural model requires at least one non-event row.")
    return working


def fit_primary_behaviour_models(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> dict[str, FittedBehaviourModel]:
    """Fit the compact primary behavioural model sequence."""

    validated = validate_primary_model_table(riskset_table, config=config)
    formulas = build_primary_model_formulas(config)
    cluster_column = _resolve_cluster_column(validated, config.cluster_column)
    fitted_models: dict[str, FittedBehaviourModel] = {}
    for model_name in PRIMARY_MODEL_SEQUENCE:
        warnings_list: list[str] = []
        formula = formulas[model_name]
        robust_covariance_used = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glm = sm.GLM.from_formula(formula, data=validated, family=sm.families.Binomial())
            try:
                fitted = glm.fit(cov_type="cluster", cov_kwds={"groups": validated[cluster_column]})
            except Exception as error:  # pragma: no cover - fallback path
                fitted = glm.fit()
                robust_covariance_used = False
                warnings_list.append(
                    f"Cluster-robust covariance failed for {model_name}; fell back to non-robust covariance: {error}"
                )
            warnings_list.extend(str(item.message) for item in caught)
        fitted_models[model_name] = FittedBehaviourModel(
            model_name=model_name,
            formula=formula,
            result=fitted,
            summary_table=extract_model_summary(fitted_result=fitted, model_name=model_name),
            fit_metrics=compute_fit_metrics(
                fitted_result=fitted,
                riskset_table=validated,
                model_name=model_name,
                formula=formula,
                cluster_column=cluster_column,
                robust_covariance_used=robust_covariance_used,
                warnings_list=warnings_list,
            ),
            robust_covariance_used=robust_covariance_used,
            warnings=warnings_list,
        )
    return fitted_models


def validate_timing_control_model_table(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Validate the timing-control model dataset and retain signed offset timings."""

    working = validate_primary_model_table(riskset_table, config=config)
    if "time_from_partner_offset" not in working.columns:
        raise ValueError("Timing-control behavioural model requires the `time_from_partner_offset` column.")
    offset_values = pd.to_numeric(working["time_from_partner_offset"], errors="coerce")
    if not np.isfinite(offset_values).all():
        raise ValueError(
            "Timing-control behavioural validation failed: `time_from_partner_offset` contains missing or non-finite values."
        )
    working = working.copy()
    working["time_from_partner_offset"] = offset_values.astype(float)
    return working


def fit_timing_control_behaviour_models(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> dict[str, FittedBehaviourModel]:
    """Fit the onset-plus-offset timing-control behavioural model sequence."""

    validated = validate_timing_control_model_table(riskset_table, config=config)
    validated = _prepare_timing_control_model_table(validated, config=config)
    formulas = build_timing_control_model_formulas(config)
    fitted_models: dict[str, FittedBehaviourModel] = {}
    for model_name in TIMING_CONTROL_MODEL_SEQUENCE:
        formula = formulas[model_name]
        fitted_models[model_name] = _fit_formula_model(
            riskset_table=validated,
            model_name=model_name,
            formula=formula,
            config=config,
        )
    return fitted_models


def select_timing_control_best_lags(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run timing-controlled lag selection against timing-controlled parent models."""

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

    rate_terms = [primary_z_column_name("information_rate", lag_ms) for lag_ms in lag_grid]
    rate_table = _shared_complete_case_subset(validated, required_columns=baseline_columns + rate_terms)
    rate_parent_name = "M0_timing"
    rate_parent_formula = f"event ~ {baseline_terms}"
    rate_parent_model = _fit_formula_model(
        riskset_table=rate_table,
        model_name=rate_parent_name,
        formula=rate_parent_formula,
        config=config,
    )
    selection_rows: list[dict[str, Any]] = []
    for lag_ms in lag_grid:
        term = primary_z_column_name("information_rate", lag_ms)
        child_name = f"M1_rate_lag_{int(lag_ms)}ms_timing"
        child_model = _fit_formula_model(
            riskset_table=rate_table,
            model_name=child_name,
            formula=f"{rate_parent_formula} + {term}",
            config=config,
        )
        comparison = compare_nested_models(
            {rate_parent_name: rate_parent_model, child_name: child_model},
            comparisons=[(child_name, rate_parent_name)],
        )
        selection_rows.extend(
            _timing_control_selection_rows(comparison, predictor_family="information_rate", lag_ms=lag_ms)
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
    expected_parent_name = "M1_rate_best_timing"
    expected_parent_formula = f"event ~ {baseline_terms} + {best_information_rate_term}"
    expected_parent_model = _fit_formula_model(
        riskset_table=expected_table,
        model_name=expected_parent_name,
        formula=expected_parent_formula,
        config=config,
    )
    for lag_ms in lag_grid:
        term = primary_z_column_name("prop_expected_cumulative_info", lag_ms)
        child_name = f"M2_expected_lag_{int(lag_ms)}ms_timing"
        child_model = _fit_formula_model(
            riskset_table=expected_table,
            model_name=child_name,
            formula=f"{expected_parent_formula} + {term}",
            config=config,
        )
        comparison = compare_nested_models(
            {expected_parent_name: expected_parent_model, child_name: child_model},
            comparisons=[(child_name, expected_parent_name)],
        )
        selection_rows.extend(
            _timing_control_selection_rows(
                comparison,
                predictor_family="prop_expected_cumulative_info",
                lag_ms=lag_ms,
            )
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
        "lag_selection_parent_for_information_rate": rate_parent_name,
        "lag_selection_parent_for_expected_cumulative_info": expected_parent_name,
        "delta_aic_convention": "child_aic - parent_aic; negative favours child",
    }
    return selection_table, selected_lags


def select_best_timing_control_lag(
    selection_table: pd.DataFrame,
    *,
    predictor_family: str,
) -> pd.Series:
    """Select the lag with the most negative delta AIC for a predictor family."""

    family_rows = selection_table.loc[
        selection_table["predictor_family"].astype(str) == str(predictor_family)
    ].copy()
    if family_rows.empty:
        raise ValueError(f"No timing-control lag rows were found for predictor family `{predictor_family}`.")
    family_rows["delta_aic"] = pd.to_numeric(family_rows["delta_aic"], errors="coerce")
    family_rows["lag_ms"] = pd.to_numeric(family_rows["lag_ms"], errors="coerce")
    family_rows = family_rows.loc[np.isfinite(family_rows["delta_aic"]) & np.isfinite(family_rows["lag_ms"])].copy()
    if family_rows.empty:
        raise ValueError(f"No finite timing-control delta AIC rows were found for `{predictor_family}`.")
    return family_rows.sort_values(["delta_aic", "lag_ms"], ascending=[True, True]).reset_index(drop=True).iloc[0]


def compare_primary_models(fitted_models: dict[str, FittedBehaviourModel]) -> pd.DataFrame:
    """Compare the compact primary behavioural models."""

    comparison = compare_nested_models(
        fitted_models,
        comparisons=[
            ("M1_rate", "M0_time"),
            ("M2_rate_prop_expected", "M1_rate"),
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
                "parent_log_likelihood",
                "child_log_likelihood",
                "lrt_statistic",
                "df_difference",
                "p_value",
                "n_rows",
                "n_events",
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
            "parent_log_likelihood",
            "child_log_likelihood",
            "lrt_statistic",
            "df_difference",
            "p_value",
            "n_rows",
            "n_events",
        ],
    ]


def compare_timing_control_models(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    """Compare the final timing-controlled behavioural models."""

    comparison = compare_nested_models(
        fitted_models,
        comparisons=[
            ("M1_rate_best_timing", "M0_timing"),
            ("M2_expected_best_timing", "M1_rate_best_timing"),
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


def summarize_primary_model_fit_metrics(
    fitted_models: dict[str, FittedBehaviourModel],
) -> list[dict[str, Any]]:
    """Collect compact primary-model fit metrics."""

    rows: list[dict[str, Any]] = []
    for model_name in PRIMARY_MODEL_SEQUENCE:
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
                "log_loss_in_sample": fit_metrics.get("log_loss_in_sample"),
                "brier_score_in_sample": fit_metrics.get("brier_score_in_sample"),
                "auroc_in_sample": fit_metrics.get("auroc_in_sample"),
                "converged": fit_metrics.get("converged"),
                "warnings": list(fitted_model.warnings),
                "cluster_column": fit_metrics.get("cluster_variable"),
                "robust_covariance_used": fit_metrics.get("robust_covariance_used"),
            }
        )
    return rows


def summarize_timing_control_fit_metrics(
    fitted_models: dict[str, FittedBehaviourModel],
) -> list[dict[str, Any]]:
    """Collect fit metrics for the timing-control models."""

    rows: list[dict[str, Any]] = []
    for model_name in TIMING_CONTROL_MODEL_SEQUENCE:
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


def build_primary_effects_payload(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    comparison_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> dict[str, Any]:
    """Build the compact primary-effects JSON payload."""

    m1_summary = fitted_models["M1_rate"].summary_table.set_index("term", drop=False)
    m2_summary = fitted_models["M2_rate_prop_expected"].summary_table.set_index("term", drop=False)
    information_rate_term = primary_z_column_name("information_rate", config.primary_information_rate_lag_ms)
    prop_expected_term = primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms)
    m1_vs_m0 = comparison_table.loc[
        comparison_table["comparison"] == "M1_rate vs M0_time"
    ].reset_index(drop=True)
    m2_vs_m1 = comparison_table.loc[
        comparison_table["comparison"] == "M2_rate_prop_expected vs M1_rate"
    ].reset_index(drop=True)
    delta_aic_m1_vs_m0 = float(m1_vs_m0.loc[0, "delta_aic"]) if not m1_vs_m0.empty else None
    delta_aic_m2_vs_m1 = float(m2_vs_m1.loc[0, "delta_aic"]) if not m2_vs_m1.empty else None
    beta_prop_expected = float(m2_summary.loc[prop_expected_term, "estimate"])
    prop_expected_positive = beta_prop_expected > 0.0
    m1_improves_m0 = bool(delta_aic_m1_vs_m0 is not None and delta_aic_m1_vs_m0 < 0.0)
    m2_improves_m1 = bool(delta_aic_m2_vs_m1 is not None and delta_aic_m2_vs_m1 < 0.0)
    episode_flags = (
        pd.to_numeric(riskset_table["episode_has_event"], errors="coerce")
        .fillna(0)
        .astype(int)
        .groupby(riskset_table["episode_id"])
        .max()
    )
    return {
        "information_rate_lag_ms": int(config.primary_information_rate_lag_ms),
        "prop_expected_lag_ms": int(config.primary_prop_expected_lag_ms),
        "beta_information_rate": float(m1_summary.loc[information_rate_term, "estimate"]),
        "p_information_rate": float(m1_summary.loc[information_rate_term, "p_value"]),
        "odds_ratio_information_rate": float(m1_summary.loc[information_rate_term, "odds_ratio"]),
        "beta_prop_expected": beta_prop_expected,
        "p_prop_expected": float(m2_summary.loc[prop_expected_term, "p_value"]),
        "odds_ratio_prop_expected": float(m2_summary.loc[prop_expected_term, "odds_ratio"]),
        "delta_aic_m1_vs_m0": delta_aic_m1_vs_m0,
        "delta_aic_m2_vs_m1": delta_aic_m2_vs_m1,
        "m1_improves_m0": m1_improves_m0,
        "m2_improves_m1": m2_improves_m1,
        "prop_expected_positive": prop_expected_positive,
        "main_prediction_supported": bool(m2_improves_m1 and prop_expected_positive),
        "n_rows": int(len(riskset_table)),
        "n_events": int(pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int).sum()),
        "n_episodes": int(riskset_table["episode_id"].nunique()),
        "n_event_positive_episodes": int((episode_flags == 1).sum()),
        "n_censored_episodes": int((episode_flags == 0).sum()),
    }


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
    *,
    comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Compare nested behavioural models using likelihood-ratio tests."""

    requested_comparisons = comparisons or _default_nested_model_comparisons(fitted_models)
    rows: list[dict[str, object]] = []
    for full_name, reduced_name in requested_comparisons:
        if full_name not in fitted_models or reduced_name not in fitted_models:
            continue
        full = fitted_models[full_name].result
        reduced = fitted_models[reduced_name].result
        lr_stat = 2.0 * (float(full.llf) - float(reduced.llf))
        df_diff = int(len(full.params) - len(reduced.params))
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
        lag_ms = _extract_lag_ms(full_name)
        rows.append(
            {
                "lag_ms": lag_ms,
                "child_model": full_name,
                "parent_model": reduced_name,
                "comparison": f"{full_name} vs {reduced_name}",
                "full_model": full_name,
                "reduced_model": reduced_name,
                "child_aic": _maybe_float(getattr(full, "aic", None)),
                "parent_aic": _maybe_float(getattr(reduced, "aic", None)),
                "child_bic": _maybe_float(getattr(full, "bic", None)),
                "parent_bic": _maybe_float(getattr(reduced, "bic", None)),
                "child_log_likelihood": _maybe_float(getattr(full, "llf", None)),
                "parent_log_likelihood": _maybe_float(getattr(reduced, "llf", None)),
                "lrt_statistic": lr_stat,
                "likelihood_ratio_statistic": lr_stat,
                "df_difference": df_diff,
                "p_value": p_value,
                "delta_aic": _safe_difference(
                    _maybe_float(getattr(full, "aic", None)),
                    _maybe_float(getattr(reduced, "aic", None)),
                ),
                "delta_bic": _safe_difference(
                    _maybe_float(getattr(full, "bic", None)),
                    _maybe_float(getattr(reduced, "bic", None)),
                ),
                "n_rows": int(fitted_models[full_name].fit_metrics["n_rows"]),
                "n_events": int(fitted_models[full_name].fit_metrics["n_events"]),
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


def summarize_lagged_model_coefficients(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    """Collect lagged coefficient summaries across model families."""

    frames: list[pd.DataFrame] = []
    for model_name, fitted_model in fitted_models.items():
        lag_ms = _extract_lag_ms(model_name)
        if lag_ms is None:
            continue
        summary = fitted_model.summary_table.copy()
        summary["lag_ms"] = int(lag_ms)
        frames.append(summary)
    if not frames:
        return pd.DataFrame(
            columns=[
                "lag_ms",
                "model_name",
                "term",
                "estimate",
                "standard_error",
                "conf_low",
                "conf_high",
                "p_value",
                "odds_ratio",
                "odds_ratio_conf_low",
                "odds_ratio_conf_high",
            ]
        )
    output = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "lag_ms",
        "model_name",
        "term",
        "estimate",
        "standard_error",
        "conf_low",
        "conf_high",
        "p_value",
        "odds_ratio",
        "odds_ratio_conf_low",
        "odds_ratio_conf_high",
    ]
    return output.loc[:, columns]


def summarize_lagged_model_fit(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    """Collect lagged model fit summaries."""

    rows: list[dict[str, object]] = []
    for model_name, fitted_model in fitted_models.items():
        lag_ms = _extract_lag_ms(model_name)
        if lag_ms is None:
            continue
        fit_metrics = fitted_model.fit_metrics
        rows.append(
            {
                "lag_ms": int(lag_ms),
                "model_name": model_name,
                "aic": fit_metrics.get("aic"),
                "bic": fit_metrics.get("bic"),
                "log_likelihood": fit_metrics.get("log_likelihood"),
                "n_rows": fit_metrics.get("n_rows"),
                "n_events": fit_metrics.get("n_events"),
                "converged": fit_metrics.get("converged"),
                "warnings": "; ".join(fitted_model.warnings),
            }
        )
    return pd.DataFrame(rows)


def summarize_best_lag_by_aic(
    fitted_models: dict[str, FittedBehaviourModel],
) -> pd.DataFrame:
    """Select descriptive best-lag rows by AIC within each lagged family."""

    family_rows: list[dict[str, object]] = []
    baseline_aic = fitted_models.get("M0").fit_metrics.get("aic") if "M0" in fitted_models else None
    for family_name, model_prefix in LAGGED_MODEL_FAMILY_NAMES.items():
        family_models = [
            (model_name, fitted_model)
            for model_name, fitted_model in fitted_models.items()
            if model_name.startswith(f"{model_prefix}_lag_")
        ]
        if not family_models:
            continue
        valid_models = [
            (model_name, fitted_model)
            for model_name, fitted_model in family_models
            if fitted_model.fit_metrics.get("aic") is not None
        ]
        if not valid_models:
            continue
        best_model_name, best_model = min(valid_models, key=lambda item: float(item[1].fit_metrics["aic"]))
        best_aic = float(best_model.fit_metrics["aic"])
        family_rows.append(
            {
                "model_family": family_name,
                "best_lag_ms": int(_extract_lag_ms(best_model_name) or 0),
                "best_aic": best_aic,
                "delta_aic_from_baseline": (
                    float(best_aic - baseline_aic) if baseline_aic is not None else None
                ),
                "best_model_name": best_model_name,
            }
        )
    return pd.DataFrame(family_rows)


def _fit_formula_model(
    *,
    riskset_table: pd.DataFrame,
    model_name: str,
    formula: str,
    config: BehaviourHazardConfig,
) -> FittedBehaviourModel:
    """Fit one behavioural GLM from an explicit formula."""

    cluster_column = _resolve_cluster_column(riskset_table, config.cluster_column)
    warnings_list: list[str] = []
    robust_covariance_used = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm_model = sm.GLM.from_formula(formula, data=riskset_table, family=sm.families.Binomial())
        try:
            fitted = glm_model.fit(cov_type="cluster", cov_kwds={"groups": riskset_table[cluster_column]})
        except Exception as error:  # pragma: no cover - fallback path
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
    """Use a shared complete-case subset for fair final timing-control comparisons."""

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
    """Create z-scored variants for any required lagged predictors that are still unscaled only."""

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
    """Return the shared complete-case subset across a required predictor set."""

    missing = [column_name for column_name in required_columns if column_name not in riskset_table.columns]
    if missing:
        raise ValueError("Required columns are missing from the model table: " + ", ".join(sorted(missing)))
    working = riskset_table.copy()
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


def _timing_control_selection_rows(
    comparison: pd.DataFrame,
    *,
    predictor_family: str,
    lag_ms: int,
) -> list[dict[str, Any]]:
    """Attach timing-control lag-selection metadata to comparison rows."""

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


def _default_nested_model_comparisons(
    fitted_models: dict[str, FittedBehaviourModel],
) -> list[tuple[str, str]]:
    comparisons: list[tuple[str, str]] = []
    for full_name, reduced_name in [("M1", "M0"), ("M2a", "M1"), ("M2b", "M1"), ("M2c", "M1")]:
        if full_name in fitted_models and reduced_name in fitted_models:
            comparisons.append((full_name, reduced_name))
    lag_map: dict[int, set[str]] = {}
    for model_name in fitted_models:
        lag_ms = _extract_lag_ms(model_name)
        if lag_ms is None:
            continue
        lag_map.setdefault(lag_ms, set()).add(model_name)
    for lag_ms, model_names in sorted(lag_map.items()):
        base_name = f"M1_rate_lag_{lag_ms}"
        if base_name in model_names and "M0" in fitted_models:
            comparisons.append((base_name, "M0"))
        for child_name in [
            f"M2a_prop_actual_lag_{lag_ms}",
            f"M2b_cumulative_lag_{lag_ms}",
            f"M2c_prop_expected_lag_{lag_ms}",
        ]:
            if child_name in model_names and base_name in fitted_models:
                comparisons.append((child_name, base_name))
    return comparisons


def _extract_lag_ms(model_name: str) -> int | None:
    match = re.search(r"_lag_(\d+)$", model_name)
    if not match:
        return None
    return int(match.group(1))
