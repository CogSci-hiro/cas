"""Low-level neural PCA and model fitting for behavioural hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.model import (
    FittedBehaviourModel,
    compute_fit_metrics,
    extract_model_summary,
    primary_z_column_name,
    _resolve_cluster_column,
)


@dataclass(frozen=True, slots=True)
class NeuralPCAResult:
    """PCA artifacts for low-level neural features."""

    transformed_table: pd.DataFrame
    pca_summary: pd.DataFrame
    loadings: pd.DataFrame
    feature_columns: tuple[str, ...]
    pc_columns: tuple[str, ...]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    cumulative_variance_selected: float


@dataclass(frozen=True, slots=True)
class NeuralLowLevelModelResult:
    """Outputs for the low-level neural hazard comparison."""

    comparison_table: pd.DataFrame
    summary_table: pd.DataFrame
    effects_payload: dict[str, Any]
    fit_metrics_payload: dict[str, Any]
    pca_result: NeuralPCAResult
    parent_model: FittedBehaviourModel
    child_model: FittedBehaviourModel
    model_table: pd.DataFrame
    retained_row_indices: tuple[int, ...]
    warnings: list[str]


def fit_lowlevel_neural_pca(
    riskset_table: pd.DataFrame,
    *,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
) -> NeuralPCAResult:
    """Fit PCA on neural-complete rows and return transformed PCs."""

    if not neural_feature_columns:
        raise ValueError("Neural PCA requires at least one neural feature column.")
    working = riskset_table.copy()
    feature_frame = working.loc[:, list(neural_feature_columns)].apply(pd.to_numeric, errors="coerce")
    complete_mask = feature_frame.notna().all(axis=1)
    complete = working.loc[complete_mask].copy()
    if complete.empty:
        raise ValueError("No neural-complete rows are available for PCA.")
    if int(pd.to_numeric(complete["event"], errors="coerce").fillna(0).sum()) <= 0:
        raise ValueError("Neural-complete model dataset must include at least one event row.")
    if int((pd.to_numeric(complete["event"], errors="coerce").fillna(0) == 0).sum()) <= 0:
        raise ValueError("Neural-complete model dataset must include at least one non-event row.")

    matrix = complete.loc[:, list(neural_feature_columns)].to_numpy(dtype=float)
    if config.neural_standardize_features:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
        scaler_mean = {
            column: float(mean_value)
            for column, mean_value in zip(neural_feature_columns, scaler.mean_, strict=True)
        }
        scaler_scale = {
            column: float(scale_value)
            for column, scale_value in zip(neural_feature_columns, scaler.scale_, strict=True)
        }
    else:
        scaler_mean = {column: 0.0 for column in neural_feature_columns}
        scaler_scale = {column: 1.0 for column in neural_feature_columns}

    pca = PCA()
    transformed = pca.fit_transform(matrix)
    selected_components = _select_neural_component_count(
        explained_variance_ratio=pca.explained_variance_ratio_,
        variance_threshold=config.neural_pca_variance_threshold,
        max_components=config.neural_pca_max_components,
        min_components=config.neural_pca_min_components,
        n_features=len(neural_feature_columns),
    )
    pc_columns = tuple(f"neural_pc{component_index + 1}" for component_index in range(selected_components))
    for column_index, column_name in enumerate(pc_columns):
        complete[column_name] = transformed[:, column_index]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1, dtype=int),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
            "selected_for_model": np.arange(1, len(cumulative) + 1) <= selected_components,
        }
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=list(neural_feature_columns),
        columns=[f"component_{index + 1}" for index in range(pca.components_.shape[0])],
    ).reset_index(names="feature")
    return NeuralPCAResult(
        transformed_table=complete,
        pca_summary=pca_summary,
        loadings=loadings,
        feature_columns=tuple(neural_feature_columns),
        pc_columns=pc_columns,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        cumulative_variance_selected=float(cumulative[selected_components - 1]),
    )


def add_neural_pcs(
    riskset_table: pd.DataFrame,
    *,
    pca_result: NeuralPCAResult,
) -> pd.DataFrame:
    """Merge selected neural PCs back onto the original row index."""

    pc_frame = pca_result.transformed_table.loc[:, list(pca_result.pc_columns)].copy()
    pc_frame["_riskset_row_index"] = pca_result.transformed_table.index.to_numpy(dtype=int)
    output = riskset_table.copy()
    output["_riskset_row_index"] = output.index.to_numpy(dtype=int)
    merged = output.merge(pc_frame, on="_riskset_row_index", how="left", sort=False)
    return merged


def fit_neural_lowlevel_models(
    riskset_table: pd.DataFrame,
    *,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
) -> NeuralLowLevelModelResult:
    """Fit M2-behaviour and M3-neural-lowlevel on identical neural-complete rows."""

    _validate_neural_model_inputs(riskset_table, neural_feature_columns=neural_feature_columns, config=config)
    pca_result = fit_lowlevel_neural_pca(riskset_table, neural_feature_columns=neural_feature_columns, config=config)
    model_table = pca_result.transformed_table.copy()
    retained_row_indices = tuple(int(index_value) for index_value in model_table.index.to_list())
    parent_formula = _build_neural_parent_formula(config)
    child_formula = parent_formula + " + " + " + ".join(pca_result.pc_columns)
    cluster_column = _resolve_cluster_column(model_table, config.neural_cluster_column or config.cluster_column)

    parent_model = _fit_formula_model(
        model_table,
        model_name="M2_behaviour_neural_sample",
        formula=parent_formula,
        cluster_column=cluster_column,
    )
    child_model = _fit_formula_model(
        model_table,
        model_name="M3_neural_lowlevel",
        formula=child_formula,
        cluster_column=cluster_column,
    )
    if parent_model.fit_metrics["n_rows"] != child_model.fit_metrics["n_rows"]:
        raise ValueError("M2 behavioural and M3 neural models must be fitted on the same number of rows.")
    if retained_row_indices != tuple(int(index_value) for index_value in model_table.index.to_list()):
        raise ValueError("Retained neural-complete rows changed unexpectedly during model fitting.")

    comparison_table = compare_neural_models(
        parent_model=parent_model,
        child_model=child_model,
        n_neural_pcs=len(pca_result.pc_columns),
    )
    summary_table = pd.concat(
        [parent_model.summary_table, child_model.summary_table],
        ignore_index=True,
        sort=False,
    )
    warnings_list = list(parent_model.warnings) + list(child_model.warnings)
    effects_payload = {
        "n_rows_full_riskset": int(len(riskset_table)),
        "n_rows_neural_complete": int(len(model_table)),
        "n_events_full_riskset": int(pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).sum()),
        "n_events_neural_complete": int(pd.to_numeric(model_table["event"], errors="coerce").fillna(0).sum()),
        "n_neural_features": int(len(neural_feature_columns)),
        "n_neural_pcs": int(len(pca_result.pc_columns)),
        "neural_pca_variance_threshold": float(config.neural_pca_variance_threshold),
        "neural_pca_cumulative_variance": float(pca_result.cumulative_variance_selected),
        "delta_aic_m3_vs_m2": float(comparison_table.loc[0, "delta_aic"]),
        "lrt_p_value_m3_vs_m2": float(comparison_table.loc[0, "p_value"]),
        "neural_model_improves_behaviour": bool(float(comparison_table.loc[0, "delta_aic"]) < 0.0),
        "neural_window_s": float(config.neural_window_s),
        "neural_guard_s": float(config.neural_guard_s),
        "cluster_column": config.neural_cluster_column or config.cluster_column,
        "robust_covariance_used": bool(child_model.robust_covariance_used),
    }
    fit_metrics_payload = {
        "models": {
            parent_model.model_name: parent_model.fit_metrics,
            child_model.model_name: child_model.fit_metrics,
        },
        "retained_row_indices": list(retained_row_indices),
        "n_neural_pcs": int(len(pca_result.pc_columns)),
        "n_neural_features": int(len(neural_feature_columns)),
    }
    return NeuralLowLevelModelResult(
        comparison_table=comparison_table,
        summary_table=summary_table,
        effects_payload=effects_payload,
        fit_metrics_payload=fit_metrics_payload,
        pca_result=pca_result,
        parent_model=parent_model,
        child_model=child_model,
        model_table=model_table,
        retained_row_indices=retained_row_indices,
        warnings=warnings_list,
    )


def compare_neural_models(
    *,
    parent_model: FittedBehaviourModel,
    child_model: FittedBehaviourModel,
    n_neural_pcs: int,
) -> pd.DataFrame:
    """Compare the nested neural model against the behavioural baseline."""

    parent_aic = float(parent_model.fit_metrics["aic"])
    child_aic = float(child_model.fit_metrics["aic"])
    parent_ll = float(parent_model.fit_metrics["log_likelihood"])
    child_ll = float(child_model.fit_metrics["log_likelihood"])
    df_difference = int(len(child_model.result.params) - len(parent_model.result.params))
    lrt_statistic = 2.0 * (child_ll - parent_ll)
    from scipy import stats  # local import to keep module-level dependency pattern light

    return pd.DataFrame(
        [
            {
                "parent_model": parent_model.model_name,
                "child_model": child_model.model_name,
                "test_name": "neural_lowlevel_addition",
                "parent_aic": parent_aic,
                "child_aic": child_aic,
                "delta_aic": child_aic - parent_aic,
                "parent_log_likelihood": parent_ll,
                "child_log_likelihood": child_ll,
                "lrt_statistic": lrt_statistic,
                "df_difference": df_difference,
                "p_value": float(stats.chi2.sf(lrt_statistic, df=df_difference)) if df_difference > 0 else np.nan,
                "n_rows": int(child_model.fit_metrics["n_rows"]),
                "n_events": int(child_model.fit_metrics["n_events"]),
                "n_neural_pcs": int(n_neural_pcs),
            }
        ]
    )


def _fit_formula_model(
    riskset_table: pd.DataFrame,
    *,
    model_name: str,
    formula: str,
    cluster_column: str,
) -> FittedBehaviourModel:
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


def _build_neural_parent_formula(config: BehaviourHazardConfig) -> str:
    spline = (
        f"bs(time_from_partner_onset, df={config.primary_model_baseline_spline_df}, "
        f"degree={config.primary_model_baseline_spline_degree}, include_intercept=False)"
    )
    info_rate = primary_z_column_name("information_rate", config.primary_information_rate_lag_ms)
    prop_expected = primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms)
    return f"event ~ {spline} + {info_rate} + {prop_expected}"


def _validate_neural_model_inputs(
    riskset_table: pd.DataFrame,
    *,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
) -> None:
    required_columns = {
        "event",
        "episode_id",
        "time_from_partner_onset",
        primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
        *neural_feature_columns,
    }
    missing = sorted(required_columns - set(riskset_table.columns))
    if missing:
        raise ValueError(
            "Neural low-level model is missing required risk-set columns: "
            + ", ".join(missing)
        )
    if not neural_feature_columns:
        raise ValueError("M3 neural model requires at least one neural feature column.")


def _select_neural_component_count(
    *,
    explained_variance_ratio: np.ndarray,
    variance_threshold: float,
    max_components: int,
    min_components: int,
    n_features: int,
) -> int:
    cumulative = np.cumsum(np.asarray(explained_variance_ratio, dtype=float))
    threshold_index = int(np.searchsorted(cumulative, variance_threshold, side="left")) + 1
    selected = max(min_components, threshold_index)
    selected = min(selected, max_components, n_features, len(explained_variance_ratio))
    if selected < 1:
        raise ValueError("Neural PCA must retain at least one component.")
    return int(selected)
