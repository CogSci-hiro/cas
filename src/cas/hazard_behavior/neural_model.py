"""Low-level neural PCA and model fitting for behavioural hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
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
from cas.hazard_behavior.neural_io import group_neural_feature_columns_by_family

LOGGER = logging.getLogger(__name__)
FAMILY_PC_PREFIX = {
    "amplitude": "amp",
    "alpha": "alpha",
    "beta": "beta",
    "combined": "neural",
}


@dataclass(frozen=True, slots=True)
class NeuralPCAResult:
    """PCA artifacts for one low-level neural family."""

    family: str
    pca_summary: pd.DataFrame
    loadings: pd.DataFrame
    feature_columns: tuple[str, ...]
    pc_columns: tuple[str, ...]
    cumulative_variance_selected: float


@dataclass(frozen=True, slots=True)
class NeuralLowLevelModelResult:
    """Outputs for the low-level neural hazard comparison."""

    comparison_table: pd.DataFrame
    family_comparison_table: pd.DataFrame
    summary_table: pd.DataFrame
    effects_payload: dict[str, Any]
    fit_metrics_payload: dict[str, Any]
    pca_results: dict[str, NeuralPCAResult]
    parent_model: FittedBehaviourModel
    child_model: FittedBehaviourModel
    family_models: dict[str, FittedBehaviourModel]
    model_table: pd.DataFrame
    retained_row_indices: tuple[int, ...]
    full_pc_columns: tuple[str, ...]
    available_families: tuple[str, ...]
    best_family_for_pc1: str | None
    warnings: list[str]


def fit_lowlevel_neural_pca(
    riskset_table: pd.DataFrame,
    *,
    family: str,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
) -> NeuralPCAResult:
    """Fit PCA for a single neural feature family on an already filtered table."""

    if not neural_feature_columns:
        raise ValueError(f"Neural PCA requires at least one feature column for family `{family}`.")
    LOGGER.info(
        "Fitting %s-family PCA on %d rows and %d features.",
        family,
        len(riskset_table),
        len(neural_feature_columns),
    )
    matrix = riskset_table.loc[:, list(neural_feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if config.neural_standardize_features:
        matrix = StandardScaler().fit_transform(matrix)
    pca = PCA()
    transformed = pca.fit_transform(matrix)
    selected_components = _select_neural_component_count(
        explained_variance_ratio=pca.explained_variance_ratio_,
        variance_threshold=config.neural_pca_variance_threshold,
        max_components=config.neural_pca_max_components,
        min_components=config.neural_pca_min_components,
        n_features=len(neural_feature_columns),
    )
    prefix = FAMILY_PC_PREFIX[family]
    pc_columns = tuple(f"{prefix}_pc{component_index + 1}" for component_index in range(selected_components))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame(
        {
            "family": family,
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
    loadings["family"] = family
    for column_index, column_name in enumerate(pc_columns):
        riskset_table[column_name] = transformed[:, column_index]
    return NeuralPCAResult(
        family=family,
        pca_summary=pca_summary,
        loadings=loadings,
        feature_columns=tuple(neural_feature_columns),
        pc_columns=pc_columns,
        cumulative_variance_selected=float(cumulative[selected_components - 1]),
    )


def fit_neural_lowlevel_models(
    riskset_table: pd.DataFrame,
    *,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
) -> NeuralLowLevelModelResult:
    """Fit M2, full M3, and family-specific M3 models on identical neural-complete rows."""

    _validate_neural_model_inputs(riskset_table, neural_feature_columns=neural_feature_columns, config=config)
    warnings_list: list[str] = []
    family_columns = _resolve_family_columns(
        neural_feature_columns=neural_feature_columns,
        config=config,
        warnings_list=warnings_list,
    )
    if not family_columns:
        raise ValueError("No neural feature families were available after grouping the selected neural columns.")

    required_complete_columns = [column for columns in family_columns.values() for column in columns]
    complete_mask = riskset_table.loc[:, required_complete_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    model_table = riskset_table.loc[complete_mask].copy()
    if model_table.empty:
        raise ValueError("No neural-complete rows are available for family-wise neural PCA.")
    if int(pd.to_numeric(model_table["event"], errors="coerce").fillna(0).sum()) <= 0:
        raise ValueError("Neural-complete model dataset must include at least one event row.")
    if int((pd.to_numeric(model_table["event"], errors="coerce").fillna(0) == 0).sum()) <= 0:
        raise ValueError("Neural-complete model dataset must include at least one non-event row.")

    pca_results: dict[str, NeuralPCAResult] = {}
    full_pc_columns: list[str] = []
    LOGGER.info(
        "Preparing low-level neural models with PCA mode=%s across families=%s.",
        config.neural_pca_mode,
        list(family_columns),
    )
    LOGGER.info(
        "Retained %d neural-complete rows out of %d total risk-set rows.",
        len(model_table),
        len(riskset_table),
    )
    for family_name, columns in family_columns.items():
        family_result = fit_lowlevel_neural_pca(
            model_table,
            family=family_name,
            neural_feature_columns=columns,
            config=config,
        )
        pca_results[family_name] = family_result
        full_pc_columns.extend(family_result.pc_columns)
    if not full_pc_columns:
        raise ValueError("M3 neural model requires at least one retained neural PC.")

    retained_row_indices = tuple(int(index_value) for index_value in model_table.index.to_list())
    parent_formula = _build_neural_parent_formula(config)
    cluster_column = _resolve_cluster_column(model_table, config.neural_cluster_column or config.cluster_column)
    parent_model = _fit_formula_model(
        model_table,
        model_name="M2_behaviour_neural_sample",
        formula=parent_formula,
        cluster_column=cluster_column,
    )
    family_models: dict[str, FittedBehaviourModel] = {}
    family_comparison_rows: list[pd.DataFrame] = []
    for family_name, family_result in pca_results.items():
        family_formula = parent_formula + " + " + " + ".join(family_result.pc_columns)
        LOGGER.info(
            "Fitting family-specific neural model %s with %d PCs.",
            f"M3_{FAMILY_PC_PREFIX[family_name]}",
            len(family_result.pc_columns),
        )
        family_model = _fit_formula_model(
            model_table,
            model_name=f"M3_{FAMILY_PC_PREFIX[family_name]}",
            formula=family_formula,
            cluster_column=cluster_column,
        )
        family_models[family_name] = family_model
        family_comparison_rows.append(
            compare_neural_models(
                parent_model=parent_model,
                child_model=family_model,
                n_neural_pcs=len(family_result.pc_columns),
                family=family_name,
                test_name=f"{family_name}_lowlevel_addition",
            )
        )

    LOGGER.info("Fitting full neural model M3_neural_lowlevel with %d total PCs.", len(full_pc_columns))
    child_model = _fit_formula_model(
        model_table,
        model_name="M3_neural_lowlevel",
        formula=parent_formula + " + " + " + ".join(full_pc_columns),
        cluster_column=cluster_column,
    )
    comparison_table = compare_neural_models(
        parent_model=parent_model,
        child_model=child_model,
        n_neural_pcs=len(full_pc_columns),
        family="all",
        test_name="neural_lowlevel_addition",
    )
    family_comparison_table = pd.concat(family_comparison_rows + [comparison_table], ignore_index=True, sort=False)
    summary_table = pd.concat(
        [parent_model.summary_table, child_model.summary_table, *[model.summary_table for model in family_models.values()]],
        ignore_index=True,
        sort=False,
    )
    warnings_list.extend(parent_model.warnings)
    warnings_list.extend(child_model.warnings)
    for model in family_models.values():
        warnings_list.extend(model.warnings)

    best_family_for_pc1 = _best_family_by_delta_aic(family_comparison_table)
    effects_payload = {
        "n_rows_full_riskset": int(len(riskset_table)),
        "n_rows_neural_complete": int(len(model_table)),
        "n_events_full_riskset": int(pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).sum()),
        "n_events_neural_complete": int(pd.to_numeric(model_table["event"], errors="coerce").fillna(0).sum()),
        "n_neural_features": int(len(neural_feature_columns)),
        "n_neural_pcs": int(len(full_pc_columns)),
        "neural_pca_mode": str(config.neural_pca_mode),
        "family_n_pcs": {
            family_name: int(len(result.pc_columns))
            for family_name, result in pca_results.items()
        },
        "family_cumulative_variance": {
            family_name: float(result.cumulative_variance_selected)
            for family_name, result in pca_results.items()
        },
        "neural_pca_variance_threshold": float(config.neural_pca_variance_threshold),
        "delta_aic_m3_vs_m2": float(comparison_table.loc[0, "delta_aic"]),
        "lrt_p_value_m3_vs_m2": float(comparison_table.loc[0, "p_value"]),
        "neural_model_improves_behaviour": bool(float(comparison_table.loc[0, "delta_aic"]) < 0.0),
        "neural_window_s": float(config.neural_window_s),
        "neural_guard_s": float(config.neural_guard_s),
        "cluster_column": config.neural_cluster_column or config.cluster_column,
        "robust_covariance_used": bool(child_model.robust_covariance_used),
        "available_families": list(pca_results),
        "best_family_for_pc1": best_family_for_pc1,
    }
    fit_metrics_payload = {
        "models": {
            parent_model.model_name: parent_model.fit_metrics,
            child_model.model_name: child_model.fit_metrics,
            **{model.model_name: model.fit_metrics for model in family_models.values()},
        },
        "retained_row_indices": list(retained_row_indices),
        "n_neural_pcs": int(len(full_pc_columns)),
        "n_neural_features": int(len(neural_feature_columns)),
        "neural_pca_mode": str(config.neural_pca_mode),
    }
    return NeuralLowLevelModelResult(
        comparison_table=comparison_table,
        family_comparison_table=family_comparison_table,
        summary_table=summary_table,
        effects_payload=effects_payload,
        fit_metrics_payload=fit_metrics_payload,
        pca_results=pca_results,
        parent_model=parent_model,
        child_model=child_model,
        family_models=family_models,
        model_table=model_table,
        retained_row_indices=retained_row_indices,
        full_pc_columns=tuple(full_pc_columns),
        available_families=tuple(pca_results),
        best_family_for_pc1=best_family_for_pc1,
        warnings=warnings_list,
    )


def compare_neural_models(
    *,
    parent_model: FittedBehaviourModel,
    child_model: FittedBehaviourModel,
    n_neural_pcs: int,
    family: str,
    test_name: str,
) -> pd.DataFrame:
    """Compare a nested neural model against the behavioural baseline."""

    parent_aic = float(parent_model.fit_metrics["aic"])
    child_aic = float(child_model.fit_metrics["aic"])
    parent_ll = float(parent_model.fit_metrics["log_likelihood"])
    child_ll = float(child_model.fit_metrics["log_likelihood"])
    df_difference = int(len(child_model.result.params) - len(parent_model.result.params))
    lrt_statistic = 2.0 * (child_ll - parent_ll)
    return pd.DataFrame(
        [
            {
                "family": family,
                "parent_model": parent_model.model_name,
                "child_model": child_model.model_name,
                "test_name": test_name,
                "n_pcs": int(n_neural_pcs),
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


def _resolve_family_columns(
    *,
    neural_feature_columns: tuple[str, ...],
    config: BehaviourHazardConfig,
    warnings_list: list[str],
) -> dict[str, tuple[str, ...]]:
    if config.neural_pca_mode == "combined":
        return {"combined": tuple(neural_feature_columns)}
    grouped = group_neural_feature_columns_by_family(neural_feature_columns)
    for family_name in ["amplitude", "alpha", "beta"]:
        if family_name not in grouped:
            warnings_list.append(f"No {family_name} neural features were available; continuing with available families.")
    return grouped


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


def _best_family_by_delta_aic(family_comparison_table: pd.DataFrame) -> str | None:
    families = family_comparison_table.loc[family_comparison_table["family"] != "all"].copy()
    if families.empty:
        return None
    families["delta_aic"] = pd.to_numeric(families["delta_aic"], errors="coerce")
    families = families.loc[np.isfinite(families["delta_aic"])]
    if families.empty:
        return None
    return str(families.sort_values("delta_aic", kind="mergesort").iloc[0]["family"])
