"""Final statistical reporting for the compact behavioural hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.io import write_json, write_table
from cas.hazard_behavior.model import (
    FittedBehaviourModel,
    PRIMARY_MODEL_SEQUENCE,
    primary_z_column_name,
)
from cas.hazard_behavior.plots import (
    plot_observed_event_rate_by_time_bin,
    plot_primary_coefficients,
    plot_primary_lag_sensitivity,
    plot_primary_leave_one_cluster,
    plot_primary_model_comparison,
    plot_primary_prediction_curve,
    summarize_observed_event_rate_by_time_bin,
)

PRIMARY_TEST_NAME = "M2_rate_prop_expected vs M1_rate"
SECONDARY_TEST_NAME = "M1_rate vs M0_time"


@dataclass(frozen=True, slots=True)
class PrimaryReportingResult:
    """Container for compact primary-reporting artifacts."""

    stat_tests_table: pd.DataFrame
    stat_tests_payload: dict[str, Any]
    publication_table: pd.DataFrame
    interpretation_text: str


def compute_primary_lrt_row(
    *,
    parent_model: str,
    child_model: str,
    parent_log_likelihood: float,
    child_log_likelihood: float,
    parent_df: int,
    child_df: int,
    parent_aic: float | None,
    child_aic: float | None,
    hypothesis: str,
    interpretation: str,
) -> dict[str, Any]:
    """Compute one primary-model likelihood-ratio test row."""

    lrt_statistic = float(2.0 * (child_log_likelihood - parent_log_likelihood))
    df_difference = int(child_df - parent_df)
    p_value = float(stats.chi2.sf(lrt_statistic, df_difference)) if df_difference > 0 else np.nan
    delta_aic = float(child_aic - parent_aic) if child_aic is not None and parent_aic is not None else None
    return {
        "test_type": "lrt",
        "test_name": f"{child_model} vs {parent_model}",
        "parent_model": parent_model,
        "child_model": child_model,
        "model_name": child_model,
        "term": None,
        "hypothesis": hypothesis,
        "parent_log_likelihood": float(parent_log_likelihood),
        "child_log_likelihood": float(child_log_likelihood),
        "lrt_statistic": lrt_statistic,
        "df_difference": df_difference,
        "p_value": p_value,
        "parent_aic": parent_aic,
        "child_aic": child_aic,
        "delta_aic": delta_aic,
        "estimate": None,
        "standard_error": None,
        "z_value": None,
        "conf_low": None,
        "conf_high": None,
        "odds_ratio": None,
        "odds_ratio_conf_low": None,
        "odds_ratio_conf_high": None,
        "interpretation": interpretation,
    }


def primary_supported(beta_prop_expected: float, delta_aic_m2_vs_m1: float) -> bool:
    """Return the project-default primary-support decision."""

    return bool(beta_prop_expected > 0.0 and delta_aic_m2_vs_m1 < 0.0)


def run_primary_stat_reporting(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    comparison_table: pd.DataFrame,
    lagged_model_comparison: pd.DataFrame,
    lagged_coefficients: pd.DataFrame,
    output_dirs: dict[str, Path],
    config: BehaviourHazardConfig,
) -> PrimaryReportingResult:
    """Run compact statistical reporting and publication-figure generation."""

    _validate_primary_reporting_inputs(
        riskset_table=riskset_table,
        fitted_models=fitted_models,
        comparison_table=comparison_table,
        config=config,
    )
    publication_table = build_primary_publication_table(fitted_models=fitted_models, config=config)
    stat_tests_table = build_primary_stat_tests_table(
        fitted_models=fitted_models,
        comparison_table=comparison_table,
        publication_table=publication_table,
        config=config,
    )
    stat_tests_payload = build_primary_stat_tests_payload(
        riskset_table=riskset_table,
        fitted_models=fitted_models,
        comparison_table=comparison_table,
        publication_table=publication_table,
        config=config,
    )
    interpretation_text = build_primary_interpretation_text(
        comparison_table=comparison_table,
        publication_table=publication_table,
        stat_tests_payload=stat_tests_payload,
    )

    write_table(stat_tests_table, output_dirs["models"] / "behaviour_primary_stat_tests.csv", sep=",")
    write_json(stat_tests_payload, output_dirs["models"] / "behaviour_primary_stat_tests.json")
    write_table(publication_table, output_dirs["models"] / "behaviour_primary_publication_table.csv", sep=",")
    (output_dirs["models"] / "behaviour_primary_interpretation.txt").write_text(
        interpretation_text,
        encoding="utf-8",
    )

    if config.make_primary_publication_figures:
        plot_primary_coefficients(
            publication_table,
            output_dirs["figures"] / "behaviour_primary_coefficients.png",
        )
        plot_primary_model_comparison(
            comparison_table,
            output_dirs["figures"] / "behaviour_primary_model_comparison.png",
        )
        prop_expected_predictions = build_primary_prediction_grid(
            riskset_table=riskset_table,
            fitted_model=fitted_models["M2_rate_prop_expected"],
            varying_term=primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
            held_constant_term=primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
            held_constant_value=0.0,
        )
        write_table(
            prop_expected_predictions,
            output_dirs["figures"] / "behaviour_primary_predicted_hazard_prop_expected.csv",
            sep=",",
        )
        plot_primary_prediction_curve(
            prop_expected_predictions,
            x_column=primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
            x_label="Expected-relative cumulative information, 300 ms (z)",
            title="Predicted behavioural hazard by delayed expected-relative information",
            output_path=output_dirs["figures"] / "behaviour_primary_predicted_hazard_prop_expected.png",
        )

        information_rate_predictions = build_primary_prediction_grid(
            riskset_table=riskset_table,
            fitted_model=fitted_models["M2_rate_prop_expected"],
            varying_term=primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
            held_constant_term=primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
            held_constant_value=0.0,
        )
        write_table(
            information_rate_predictions,
            output_dirs["figures"] / "behaviour_primary_predicted_hazard_information_rate.csv",
            sep=",",
        )
        plot_primary_prediction_curve(
            information_rate_predictions,
            x_column=primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
            x_label="Information rate, 0 ms (z)",
            title="Predicted behavioural hazard by local information rate",
            output_path=output_dirs["figures"] / "behaviour_primary_predicted_hazard_information_rate.png",
        )

        observed_event_rate, observed_event_rate_qc = summarize_observed_event_rate_by_time_bin(riskset_table)
        write_table(
            observed_event_rate,
            output_dirs["figures"] / "behaviour_primary_observed_event_rate.csv",
            sep=",",
        )
        plot_observed_event_rate_by_time_bin(
            observed_event_rate,
            observed_event_rate_qc,
            output_dirs["figures"] / "behaviour_primary_observed_event_rate.png",
        )

        lag_sensitivity_table = build_primary_lag_sensitivity_table(
            lagged_coefficients=lagged_coefficients,
            lagged_model_comparison=lagged_model_comparison,
        )
        write_table(
            lag_sensitivity_table,
            output_dirs["figures"] / "behaviour_primary_lag_sensitivity.csv",
            sep=",",
        )
        plot_primary_lag_sensitivity(
            lag_sensitivity_table,
            output_dirs["figures"] / "behaviour_primary_lag_sensitivity.png",
        )

    if config.run_primary_leave_one_cluster:
        leave_one_cluster = run_primary_leave_one_cluster_sensitivity(
            riskset_table=riskset_table,
            fitted_models=fitted_models,
            config=config,
        )
        write_table(
            leave_one_cluster,
            output_dirs["models"] / "leave_one_cluster_primary_effects.csv",
            sep=",",
        )
        plot_primary_leave_one_cluster(
            leave_one_cluster,
            full_beta=float(stat_tests_payload["primary_beta"]),
            output_path=output_dirs["figures"] / "behaviour_primary_leave_one_cluster.png",
        )

    return PrimaryReportingResult(
        stat_tests_table=stat_tests_table,
        stat_tests_payload=stat_tests_payload,
        publication_table=publication_table,
        interpretation_text=interpretation_text,
    )


def build_primary_stat_tests_table(
    *,
    fitted_models: dict[str, FittedBehaviourModel],
    comparison_table: pd.DataFrame,
    publication_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Build the combined final stat-tests table."""

    m0 = fitted_models["M0_time"]
    m1 = fitted_models["M1_rate"]
    m2 = fitted_models["M2_rate_prop_expected"]
    lrt_rows = [
        compute_primary_lrt_row(
            parent_model="M0_time",
            child_model="M1_rate",
            parent_log_likelihood=float(m0.result.llf),
            child_log_likelihood=float(m1.result.llf),
            parent_df=len(m0.result.params),
            child_df=len(m1.result.params),
            parent_aic=_to_float(m0.fit_metrics.get("aic")),
            child_aic=_to_float(m1.fit_metrics.get("aic")),
            hypothesis="Does local information rate improve the smooth time-only baseline?",
            interpretation="Tests whether local information rate improves the smooth time-only baseline.",
        ),
        compute_primary_lrt_row(
            parent_model="M1_rate",
            child_model="M2_rate_prop_expected",
            parent_log_likelihood=float(m1.result.llf),
            child_log_likelihood=float(m2.result.llf),
            parent_df=len(m1.result.params),
            child_df=len(m2.result.params),
            parent_aic=_to_float(m1.fit_metrics.get("aic")),
            child_aic=_to_float(m2.fit_metrics.get("aic")),
            hypothesis=(
                "Does delayed expected-relative accumulated partner information improve hazard "
                "prediction beyond time and local information rate?"
            ),
            interpretation=(
                "Primary behavioural test: tests whether expected-relative cumulative information "
                "lagged by 300 ms adds beyond time and local information rate."
            ),
        ),
    ]
    lrt_table = pd.DataFrame(lrt_rows)

    wald_table = publication_table.copy()
    wald_table["test_type"] = "wald"
    wald_table["test_name"] = (
        "wald_" + wald_table["model_name"].astype(str) + "_" + wald_table["term"].astype(str)
    )
    wald_table["parent_model"] = None
    wald_table["child_model"] = None
    wald_table["parent_log_likelihood"] = None
    wald_table["child_log_likelihood"] = None
    wald_table["lrt_statistic"] = None
    wald_table["df_difference"] = None
    wald_table["parent_aic"] = None
    wald_table["child_aic"] = None
    wald_table["delta_aic"] = None
    columns = [
        "test_type",
        "test_name",
        "parent_model",
        "child_model",
        "model_name",
        "term",
        "hypothesis",
        "parent_log_likelihood",
        "child_log_likelihood",
        "lrt_statistic",
        "df_difference",
        "p_value",
        "parent_aic",
        "child_aic",
        "delta_aic",
        "estimate",
        "standard_error",
        "z_value",
        "conf_low",
        "conf_high",
        "odds_ratio",
        "odds_ratio_conf_low",
        "odds_ratio_conf_high",
        "interpretation",
    ]
    combined_rows = lrt_table.loc[:, columns].to_dict(orient="records")
    combined_rows.extend(wald_table.loc[:, columns].to_dict(orient="records"))
    return pd.DataFrame(combined_rows, columns=columns)


def build_primary_publication_table(
    *,
    fitted_models: dict[str, FittedBehaviourModel],
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Extract the compact publication-ready coefficient table."""

    requested_rows = [
        (
            "M1_rate",
            primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
            "Does local information rate improve the smooth time-only baseline?",
            "Local information rate effect in the secondary behavioural model.",
        ),
        (
            "M2_rate_prop_expected",
            primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
            "Control effect of local information rate in the primary behavioural model.",
            "Local information rate effect after adding delayed expected-relative information.",
        ),
        (
            "M2_rate_prop_expected",
            primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
            "Primary behavioural coefficient: delayed expected-relative cumulative information.",
            "Expected-relative cumulative information lagged by 300 ms in the primary behavioural model.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, term_name, hypothesis, interpretation in requested_rows:
        summary = fitted_models[model_name].summary_table
        row = summary.loc[summary["term"].astype(str) == term_name]
        if row.empty:
            raise ValueError(f"Primary publication table could not find required term `{term_name}` in {model_name}.")
        selected = row.iloc[0].to_dict()
        rows.append(
            {
                "model_name": model_name,
                "term": term_name,
                "hypothesis": hypothesis,
                "estimate": float(selected["estimate"]),
                "standard_error": float(selected["standard_error"]),
                "z_value": float(selected["z_value"]),
                "p_value": float(selected["p_value"]),
                "conf_low": float(selected["conf_low"]),
                "conf_high": float(selected["conf_high"]),
                "odds_ratio": float(selected["odds_ratio"]),
                "odds_ratio_conf_low": float(selected["odds_ratio_conf_low"]),
                "odds_ratio_conf_high": float(selected["odds_ratio_conf_high"]),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_primary_stat_tests_payload(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    comparison_table: pd.DataFrame,
    publication_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> dict[str, Any]:
    """Build the summary JSON payload for the final primary stats layer."""

    primary_row = publication_table.loc[
        (publication_table["model_name"] == "M2_rate_prop_expected")
        & (
            publication_table["term"]
            == primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms)
        )
    ].iloc[0]
    secondary_row = publication_table.loc[
        (publication_table["model_name"] == "M1_rate")
        & (publication_table["term"] == primary_z_column_name("information_rate", config.primary_information_rate_lag_ms))
    ].iloc[0]
    m2_vs_m1 = comparison_table.loc[comparison_table["comparison"] == PRIMARY_TEST_NAME].iloc[0]
    m1_vs_m0 = comparison_table.loc[comparison_table["comparison"] == SECONDARY_TEST_NAME].iloc[0]
    event_numeric = pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int)
    episode_flags = (
        pd.to_numeric(riskset_table["episode_has_event"], errors="coerce")
        .fillna(0)
        .astype(int)
        .groupby(riskset_table["episode_id"])
        .max()
    )
    primary_beta = float(primary_row["estimate"])
    primary_delta_aic = float(m2_vs_m1["delta_aic"])
    m2_fit_metrics = fitted_models["M2_rate_prop_expected"].fit_metrics
    return {
        "n_rows": int(len(riskset_table)),
        "n_episodes": int(riskset_table["episode_id"].nunique()),
        "n_event_positive_episodes": int((episode_flags == 1).sum()),
        "n_censored_episodes": int((episode_flags == 0).sum()),
        "n_events": int(event_numeric.sum()),
        "event_rate_overall": float(event_numeric.mean()),
        "primary_test_name": PRIMARY_TEST_NAME,
        "primary_beta": primary_beta,
        "primary_standard_error": float(primary_row["standard_error"]),
        "primary_p_value": float(primary_row["p_value"]),
        "primary_odds_ratio": float(primary_row["odds_ratio"]),
        "primary_conf_low": float(primary_row["conf_low"]),
        "primary_conf_high": float(primary_row["conf_high"]),
        "primary_delta_aic": primary_delta_aic,
        "primary_lrt_p_value": float(m2_vs_m1["p_value"]),
        "primary_supported": primary_supported(primary_beta, primary_delta_aic),
        "primary_supported_strict": bool(
            primary_beta > 0.0 and primary_delta_aic < 0.0 and float(primary_row["p_value"]) < 0.05
        ),
        "secondary_beta_information_rate": float(secondary_row["estimate"]),
        "secondary_p_value_information_rate": float(secondary_row["p_value"]),
        "secondary_delta_aic_m1_vs_m0": float(m1_vs_m0["delta_aic"]),
        "secondary_lrt_p_value_m1_vs_m0": float(m1_vs_m0["p_value"]),
        "cluster_column": str(m2_fit_metrics.get("cluster_variable")),
        "robust_covariance_used": bool(m2_fit_metrics.get("robust_covariance_used", False)),
        "model_backend": str(config.default_model_family),
    }


def build_primary_interpretation_text(
    *,
    comparison_table: pd.DataFrame,
    publication_table: pd.DataFrame,
    stat_tests_payload: dict[str, Any],
) -> str:
    """Build the human-readable interpretation summary."""

    m1_vs_m0 = comparison_table.loc[comparison_table["comparison"] == SECONDARY_TEST_NAME].iloc[0]
    m2_vs_m1 = comparison_table.loc[comparison_table["comparison"] == PRIMARY_TEST_NAME].iloc[0]
    primary_row = publication_table.loc[
        publication_table["term"].astype(str).str.startswith("z_prop_expected_cumulative_info_lag_")
    ].iloc[0]
    rate_improves = "improved" if float(m1_vs_m0["delta_aic"]) < 0.0 else "did not improve"
    primary_sign = "positive" if float(primary_row["estimate"]) > 0.0 else "negative"
    primary_improves = "improved" if float(m2_vs_m1["delta_aic"]) < 0.0 else "did not improve"
    if bool(stat_tests_payload["primary_supported"]):
        final_interpretation = (
            "These results suggest that FPP initiation is not only shaped by generic time-from-partner-IPU onset "
            "and local information density, but also by the amount of partner information accumulated relative to "
            "expected IPU information approximately 300 ms earlier."
        )
    else:
        final_interpretation = (
            "These results do not provide clear evidence that delayed expected-relative accumulated information "
            "explains FPP timing beyond local information density and elapsed time."
        )
    return (
        "Behavioural FPP hazard analysis\n\n"
        "We fitted a partner-IPU anchored discrete-time hazard model predicting FPP onset in 50 ms bins. "
        "The baseline hazard was modelled as a smooth function of time since partner IPU onset.\n\n"
        f"Adding local information rate {rate_improves} the baseline model:\n"
        f"Delta AIC = {_fmt_num(m1_vs_m0['delta_aic'])}, "
        f"LRT chi^2({_fmt_int(m1_vs_m0['df_difference'])}) = {_fmt_num(m1_vs_m0['lrt_statistic'])}, "
        f"p = {_fmt_p(m1_vs_m0['p_value'])}.\n\n"
        "The primary behavioural model added expected-relative cumulative information lagged by 300 ms. "
        f"This term was {primary_sign}, beta = {_fmt_num(primary_row['estimate'])}, "
        f"OR = {_fmt_num(primary_row['odds_ratio'])}, "
        f"95% CI [{_fmt_num(primary_row['conf_low'])}, {_fmt_num(primary_row['conf_high'])}], "
        f"p = {_fmt_p(primary_row['p_value'])}. "
        f"The model {primary_improves} over the information-rate model:\n"
        f"Delta AIC = {_fmt_num(m2_vs_m1['delta_aic'])}, "
        f"LRT chi^2({_fmt_int(m2_vs_m1['df_difference'])}) = {_fmt_num(m2_vs_m1['lrt_statistic'])}, "
        f"p = {_fmt_p(m2_vs_m1['p_value'])}.\n\n"
        "Interpretation:\n"
        f"{final_interpretation}\n"
    )


def build_primary_prediction_grid(
    *,
    riskset_table: pd.DataFrame,
    fitted_model: FittedBehaviourModel,
    varying_term: str,
    held_constant_term: str,
    held_constant_value: float,
    n_points: int = 100,
) -> pd.DataFrame:
    """Build a non-extrapolating prediction grid for the primary M2 model."""

    if varying_term not in riskset_table.columns:
        raise ValueError(f"Primary prediction grid is missing required predictor `{varying_term}`.")
    finite_values = pd.to_numeric(riskset_table[varying_term], errors="coerce")
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.empty:
        raise ValueError(f"Primary prediction grid could not find finite values for `{varying_term}`.")
    lower = float(finite_values.quantile(0.05))
    upper = float(finite_values.quantile(0.95))
    representative_time = _select_representative_time(riskset_table)
    varying_values = np.linspace(lower, upper, n_points)
    frame = pd.DataFrame(
        {
            varying_term: varying_values,
            held_constant_term: float(held_constant_value),
            "time_from_partner_onset": float(representative_time),
        }
    )
    prediction = fitted_model.result.get_prediction(frame).summary_frame()
    frame["predicted_hazard"] = pd.to_numeric(prediction["mean"], errors="coerce").to_numpy(dtype=float)
    frame["conf_low"] = pd.to_numeric(prediction.get("mean_ci_lower"), errors="coerce").to_numpy(dtype=float)
    frame["conf_high"] = pd.to_numeric(prediction.get("mean_ci_upper"), errors="coerce").to_numpy(dtype=float)
    return frame


def build_primary_lag_sensitivity_table(
    *,
    lagged_coefficients: pd.DataFrame,
    lagged_model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact lag-sensitivity table for retained predictors."""

    coefficient_rows: list[pd.DataFrame] = []
    for family_name, term_prefix in [
        ("information_rate", "z_information_rate_lag_"),
        ("prop_expected", "z_prop_expected_cumulative_info_lag_"),
    ]:
        working = lagged_coefficients.loc[
            lagged_coefficients["term"].astype(str).str.startswith(term_prefix)
            & lagged_coefficients["model_name"].astype(str).str.contains(family_name)
        ].copy()
        if working.empty:
            continue
        working["family"] = family_name
        coefficient_rows.append(
            working.loc[:, ["lag_ms", "family", "estimate", "conf_low", "conf_high"]]
        )
    coefficient_table = (
        pd.concat(coefficient_rows, ignore_index=True, sort=False)
        if coefficient_rows
        else pd.DataFrame(columns=["lag_ms", "family", "estimate", "conf_low", "conf_high"])
    )
    delta_rows = lagged_model_comparison.copy()
    delta_rows["family"] = delta_rows["child_model"].map(_family_from_lagged_child_model)
    delta_rows = delta_rows.loc[delta_rows["family"].isin(["information_rate", "prop_expected"])]
    delta_rows = delta_rows.loc[:, ["lag_ms", "family", "delta_aic"]]
    return coefficient_table.merge(delta_rows, on=["lag_ms", "family"], how="outer").sort_values(
        ["family", "lag_ms"],
        kind="mergesort",
    ).reset_index(drop=True)


def run_primary_leave_one_cluster_sensitivity(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Run leave-one-cluster sensitivity for the primary M2 effect."""

    cluster_column = str(fitted_models["M2_rate_prop_expected"].fit_metrics.get("cluster_variable"))
    if not cluster_column or cluster_column not in riskset_table.columns:
        return pd.DataFrame(
            [
                {
                    "omitted_cluster": None,
                    "n_rows": int(len(riskset_table)),
                    "n_events": int(pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).sum()),
                    "beta_prop_expected": np.nan,
                    "se_prop_expected": np.nan,
                    "odds_ratio_prop_expected": np.nan,
                    "delta_aic_m2_vs_m1": np.nan,
                    "fit_status": "skipped",
                    "warning": f"Cluster column not available: {cluster_column}",
                }
            ]
        )
    from cas.hazard_behavior.model import compare_primary_models, fit_primary_behaviour_models

    rows: list[dict[str, Any]] = []
    prop_expected_term = primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms)
    for cluster_value in sorted(riskset_table[cluster_column].dropna().astype(str).unique()):
        subset = riskset_table.loc[riskset_table[cluster_column].astype(str) != cluster_value].copy()
        try:
            refit = fit_primary_behaviour_models(subset, config=config)
            comparison = compare_primary_models(refit)
            summary = refit["M2_rate_prop_expected"].summary_table.set_index("term", drop=False)
            delta_row = comparison.loc[comparison["comparison"] == PRIMARY_TEST_NAME].iloc[0]
            rows.append(
                {
                    "omitted_cluster": cluster_value,
                    "n_rows": int(len(subset)),
                    "n_events": int(pd.to_numeric(subset["event"], errors="coerce").fillna(0).sum()),
                    "beta_prop_expected": float(summary.loc[prop_expected_term, "estimate"]),
                    "se_prop_expected": float(summary.loc[prop_expected_term, "standard_error"]),
                    "odds_ratio_prop_expected": float(summary.loc[prop_expected_term, "odds_ratio"]),
                    "delta_aic_m2_vs_m1": float(delta_row["delta_aic"]),
                    "fit_status": "ok",
                    "warning": "",
                }
            )
        except Exception as error:  # pragma: no cover - optional sensitivity path
            rows.append(
                {
                    "omitted_cluster": cluster_value,
                    "n_rows": int(len(subset)),
                    "n_events": int(pd.to_numeric(subset["event"], errors="coerce").fillna(0).sum()),
                    "beta_prop_expected": np.nan,
                    "se_prop_expected": np.nan,
                    "odds_ratio_prop_expected": np.nan,
                    "delta_aic_m2_vs_m1": np.nan,
                    "fit_status": "failed",
                    "warning": str(error),
                }
            )
    return pd.DataFrame(rows)


def _validate_primary_reporting_inputs(
    *,
    riskset_table: pd.DataFrame,
    fitted_models: dict[str, FittedBehaviourModel],
    comparison_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> None:
    missing_models = [model_name for model_name in PRIMARY_MODEL_SEQUENCE if model_name not in fitted_models]
    if missing_models:
        raise ValueError("Primary reporting requires fitted models: " + ", ".join(missing_models))
    required_terms = [
        primary_z_column_name("information_rate", config.primary_information_rate_lag_ms),
        primary_z_column_name("prop_expected_cumulative_info", config.primary_prop_expected_lag_ms),
    ]
    m2_terms = set(fitted_models["M2_rate_prop_expected"].summary_table["term"].astype(str))
    for term_name in required_terms:
        if term_name not in m2_terms:
            raise ValueError(f"Primary reporting requires M2_rate_prop_expected term `{term_name}`.")
    comparison_names = set(comparison_table["comparison"].astype(str))
    for comparison_name in [SECONDARY_TEST_NAME, PRIMARY_TEST_NAME]:
        if comparison_name not in comparison_names:
            raise ValueError(f"Primary reporting requires comparison row `{comparison_name}`.")
    for _, row in comparison_table.iterrows():
        child_aic = _to_float(row.get("child_aic"))
        parent_aic = _to_float(row.get("parent_aic"))
        expected_delta = None if child_aic is None or parent_aic is None else child_aic - parent_aic
        if expected_delta is not None and not np.isclose(float(row["delta_aic"]), expected_delta):
            raise ValueError("Primary reporting validation failed: delta AIC must equal child AIC minus parent AIC.")
    event_values = pd.to_numeric(riskset_table["event"], errors="coerce")
    if int((event_values == 1).sum()) <= 0:
        raise ValueError("Primary reporting requires at least one event row.")
    if int((event_values == 0).sum()) <= 0:
        raise ValueError("Primary reporting requires both 0 and 1 event rows.")


def _family_from_lagged_child_model(model_name: Any) -> str | None:
    text = str(model_name)
    if text.startswith("M1_rate_lag_"):
        return "information_rate"
    if text.startswith("M2c_prop_expected_lag_"):
        return "prop_expected"
    return None


def _select_representative_time(riskset_table: pd.DataFrame) -> float:
    event_mask = pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int) == 1
    event_times = pd.to_numeric(
        riskset_table.loc[event_mask, "time_from_partner_onset"],
        errors="coerce",
    )
    if np.isfinite(event_times).any():
        return float(event_times[np.isfinite(event_times)].median())
    riskset_times = pd.to_numeric(riskset_table["time_from_partner_onset"], errors="coerce")
    finite_times = riskset_times[np.isfinite(riskset_times)]
    if finite_times.empty:
        raise ValueError("Primary prediction grid requires finite `time_from_partner_onset` values.")
    return float(finite_times.median())


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _fmt_num(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.3f}"


def _fmt_int(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return str(int(round(numeric)))


def _fmt_p(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    if numeric < 0.001:
        return "< 0.001"
    return f"{numeric:.3f}"
