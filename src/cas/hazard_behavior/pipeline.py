"""End-to-end behavioural hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_censored_episodes, build_event_positive_episodes
from cas.hazard_behavior.features import (
    add_information_features_to_riskset,
    add_lagged_information_features,
    compute_information_timing_summaries,
    zscore_predictors,
)
from cas.hazard_behavior.io import (
    ensure_output_directories,
    read_events_table,
    read_surprisal_tables,
    save_config_and_warnings,
    write_json,
    write_table,
)
from cas.hazard_behavior.neural_features import add_lowlevel_neural_features_to_riskset
from cas.hazard_behavior.neural_io import read_neural_feature_tables, select_neural_feature_columns
from cas.hazard_behavior.neural_model import NeuralLowLevelModelResult, fit_neural_lowlevel_models
from cas.hazard_behavior.neural_plots import (
    plot_neural_lowlevel_coefficients,
    plot_neural_lowlevel_feature_missingness,
    plot_neural_lowlevel_model_comparison,
    plot_neural_lowlevel_pca_variance,
)
from cas.hazard_behavior.model import (
    build_primary_effects_payload,
    build_lagged_model_specs,
    compare_primary_models,
    compare_timing_control_models,
    compare_nested_models,
    ensure_primary_predictors_available,
    fit_binomial_glm,
    fit_primary_behaviour_models,
    fit_timing_control_behaviour_models,
    generate_prediction_grids,
    select_timing_control_best_lags,
    summarize_primary_model_fit_metrics,
    summarize_timing_control_fit_metrics,
    summarize_best_lag_by_aic,
    summarize_lagged_model_coefficients,
    summarize_lagged_model_fit,
)
from cas.hazard_behavior.plots import (
    plot_behaviour_timing_control_delta_aic_by_lag,
    plot_primary_coefficients,
    plot_primary_model_comparison,
    safe_make_plots,
)
from cas.hazard_behavior.reporting import run_primary_stat_reporting
from cas.hazard_behavior.riskset import build_discrete_time_riskset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BehaviourHazardPipelineResult:
    """Paths to the main behavioural hazard outputs."""

    out_dir: Path
    riskset_path: Path | None
    episode_summary_path: Path
    feature_qc_path: Path
    episode_validation_qc_path: Path
    excluded_episodes_path: Path
    model_comparison_csv_path: Path | None
    model_fit_metrics_json_path: Path | None
    config_json_path: Path
    warnings_path: Path


def run_behaviour_hazard_pipeline(config: BehaviourHazardConfig) -> BehaviourHazardPipelineResult:
    """Run the behavioural first-pass FPP hazard pipeline.

    Usage example
    -------------
        result = run_behaviour_hazard_pipeline(config)
        print(result.model_comparison_csv_path)
    """

    _configure_logging()
    output_dirs = ensure_output_directories(config)
    warnings_list: list[str] = []

    LOGGER.info("Loading events table from %s", config.events_path)
    events_table, event_warnings = read_events_table(config.events_path)
    LOGGER.info("Loaded %d event rows.", len(events_table))
    LOGGER.info("Loading %d surprisal file(s).", len(config.surprisal_paths))
    surprisal_table, surprisal_warnings = read_surprisal_tables(
        config.surprisal_paths,
        unmatched_surprisal_strategy=config.unmatched_surprisal_strategy,
    )
    LOGGER.info("Loaded %d surprisal token rows after filtering.", len(surprisal_table))
    warnings_list.extend(event_warnings)
    warnings_list.extend(surprisal_warnings)

    LOGGER.info("Building event-positive episodes.")
    positive_result = build_event_positive_episodes(
        events_table=events_table,
        surprisal_table=surprisal_table,
        config=config,
    )
    LOGGER.info("Built %d anchored episodes.", len(positive_result.episodes))
    warnings_list.extend(positive_result.warnings)
    episode_validation_qc_path = write_json(
        positive_result.validation_qc,
        output_dirs["riskset"] / "episode_validation_qc.json",
    )
    excluded_episodes_path = write_table(
        positive_result.excluded_episodes,
        output_dirs["riskset"] / "excluded_episodes.tsv",
    )
    _validate_final_positive_episode_latencies(positive_result.episodes, config=config)
    partner_ipu_table_path = None
    partner_ipu_episode_summary_path = None
    partner_ipu_anchor_qc_path = None
    event_rows_debug_path = None
    episodes_table = positive_result.episodes
    if config.episode_anchor == "partner_ipu":
        partner_ipu_table_path = write_table(
            positive_result.partner_ipu_table,
            output_dirs["riskset"] / "partner_ipu_table.tsv",
        )
        partner_ipu_episode_summary_path = write_table(
            positive_result.episodes,
            output_dirs["riskset"] / "partner_ipu_episode_summary.tsv",
        )
        partner_ipu_anchor_qc_path = write_json(
            positive_result.validation_qc,
            output_dirs["riskset"] / "partner_ipu_anchor_qc.json",
        )
        event_rows_debug_path = write_table(
            positive_result.event_rows_debug,
            output_dirs["riskset"] / "event_rows_debug.tsv",
        )
    if config.include_censored:
        LOGGER.info("Building censored episodes.")
        censored_episodes = build_censored_episodes(
            events_table=events_table,
            surprisal_table=surprisal_table,
            positive_episodes=positive_result.episodes,
            config=config,
        )
        LOGGER.info("Built %d censored episodes.", len(censored_episodes))
        if not censored_episodes.empty:
            episodes_table = pd.concat([episodes_table, censored_episodes], ignore_index=True, sort=False)
        else:
            warnings_list.append(
                "No censored negative episodes were constructed; analysis is conditional on event timing among observed FPPs."
            )

    LOGGER.info("Building risk set.")
    riskset_result = build_discrete_time_riskset(episodes_table, config=config)
    warnings_list.extend(riskset_result.warnings)
    LOGGER.info("Adding information features to risk set.")
    riskset_with_features, expected_total_info_by_group = add_information_features_to_riskset(
        riskset_table=riskset_result.riskset_table,
        episodes_table=episodes_table,
        surprisal_table=surprisal_table,
        config=config,
    )
    LOGGER.info("Computing information timing summaries.")
    information_timing_summary = compute_information_timing_summaries(
        episodes_table=episodes_table,
        surprisal_table=surprisal_table,
        config=config,
    )
    LOGGER.info("Adding lagged information features.")
    lagged_feature_config = config
    if config.fit_primary_behaviour_models or config.fit_timing_control_models:
        required_lags = {
            int(config.primary_information_rate_lag_ms),
            int(config.primary_prop_expected_lag_ms),
        }
        augmented_lag_grid = tuple(sorted({int(lag_ms) for lag_ms in config.lag_grid_ms} | required_lags))
        if augmented_lag_grid != config.lag_grid_ms:
            lagged_feature_config = replace(config, lag_grid_ms=augmented_lag_grid)
    riskset_with_lagged_features, lagged_feature_qc = add_lagged_information_features(
        riskset_with_features,
        config=lagged_feature_config,
    )
    LOGGER.info("Scaling predictors.")
    zscore_result = zscore_predictors(riskset_with_lagged_features)
    riskset_table = zscore_result.table
    warnings_list.extend(zscore_result.warnings)
    if config.fit_primary_behaviour_models or config.fit_neural_lowlevel_models:
        riskset_table, primary_scaling = ensure_primary_predictors_available(riskset_table, config=config)
    else:
        primary_scaling = {}

    neural_fit_result: NeuralLowLevelModelResult | None = None
    neural_qc_payload: dict[str, object] | None = None
    if config.fit_neural_lowlevel_models:
        if not config.neural_features:
            raise ValueError(
                "Neural low-level model fitting was requested but no `neural_features` paths were provided."
            )
        LOGGER.info("Loading %d neural feature file(s).", len(config.neural_features))
        neural_table = read_neural_feature_tables(
            config.neural_features,
            time_column=config.neural_time_column,
            speaker_column=config.neural_speaker_column,
        )
        neural_feature_columns = select_neural_feature_columns(
            neural_table,
            feature_prefixes=config.neural_feature_prefixes,
            include_amplitude=config.neural_include_amplitude,
            include_alpha=config.neural_include_alpha,
            include_beta=config.neural_include_beta,
        )
        LOGGER.info("Adding low-level neural features to the risk set.")
        neural_augmentation = add_lowlevel_neural_features_to_riskset(
            riskset_table,
            neural_table,
            neural_feature_columns=neural_feature_columns,
            neural_window_s=config.neural_window_s,
            neural_guard_s=config.neural_guard_s,
        )
        riskset_table = neural_augmentation.riskset_table
        neural_qc_payload = neural_augmentation.qc
        LOGGER.info("Fitting low-level neural hazard models.")
        neural_fit_result = fit_neural_lowlevel_models(
            riskset_table,
            neural_feature_columns=neural_augmentation.neural_feature_columns,
            config=config,
        )
        warnings_list.extend(neural_fit_result.warnings)

    primary_fitted_models: dict[str, object] = {}
    primary_model_summary_path = None
    primary_model_comparison_path = None
    primary_effects_path = None
    primary_fit_metrics_path = None
    timing_control_lag_selection_path = None
    timing_control_selected_lags_path = None
    timing_control_summary_path = None
    timing_control_comparison_path = None
    timing_control_fit_metrics_path = None
    primary_stat_tests_path = None
    primary_stat_tests_json_path = None
    primary_publication_table_path = None
    primary_interpretation_path = None
    if config.fit_primary_behaviour_models:
        LOGGER.info("Fitting compact primary behavioural models.")
        primary_fitted_models = fit_primary_behaviour_models(riskset_table, config=config)
        primary_summary = pd.concat(
            [primary_fitted_models[model_name].summary_table for model_name in ["M0_time", "M1_rate", "M2_rate_prop_expected"]],
            ignore_index=True,
            sort=False,
        )
        primary_comparison = compare_primary_models(primary_fitted_models)
        primary_effects = build_primary_effects_payload(
            riskset_table=riskset_table,
            fitted_models=primary_fitted_models,
            comparison_table=primary_comparison,
            config=config,
        )
        primary_fit_metrics = {
            "models": summarize_primary_model_fit_metrics(primary_fitted_models),
            "scaling": primary_scaling,
        }
        primary_model_summary_path = write_table(
            primary_summary,
            output_dirs["models"] / "behaviour_primary_model_summary.csv",
            sep=",",
        )
        primary_model_comparison_path = write_table(
            primary_comparison,
            output_dirs["models"] / "behaviour_primary_model_comparison.csv",
            sep=",",
        )
        primary_effects_path = write_json(
            primary_effects,
            output_dirs["models"] / "behaviour_primary_effects.json",
        )
        primary_fit_metrics_path = write_json(
            primary_fit_metrics,
            output_dirs["models"] / "behaviour_primary_fit_metrics.json",
        )
        try:
            plot_primary_coefficients(
                primary_summary,
                output_dirs["figures"] / "behaviour_primary_coefficients.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            warnings_list.append(f"Failed to create figure behaviour_primary_coefficients.png: {error}")
        try:
            plot_primary_model_comparison(
                primary_comparison,
                output_dirs["figures"] / "behaviour_primary_model_comparison.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            warnings_list.append(f"Failed to create figure behaviour_primary_model_comparison.png: {error}")

    if config.fit_timing_control_models:
        LOGGER.info("Fitting behavioural timing-control models.")
        timing_control_selected_lags: dict[str, object] | None = None
        timing_control_lag_selection = pd.DataFrame()
        timing_control_config = config
        if config.select_lags_with_timing_controls:
            LOGGER.info("Selecting behavioural lags against timing-controlled parent models.")
            timing_control_lag_selection, timing_control_selected_lags = select_timing_control_best_lags(
                riskset_table,
                config=config,
            )
            timing_control_lag_selection_path = write_table(
                timing_control_lag_selection,
                output_dirs["models"] / "behaviour_timing_control_lag_selection.csv",
                sep=",",
            )
            timing_control_selected_lags_path = write_json(
                timing_control_selected_lags,
                output_dirs["models"] / "behaviour_timing_control_selected_lags.json",
            )
            timing_control_config = replace(
                config,
                primary_information_rate_lag_ms=int(timing_control_selected_lags["best_information_rate_lag_ms"]),
                primary_prop_expected_lag_ms=int(timing_control_selected_lags["best_expected_cumulative_info_lag_ms"]),
            )
        timing_control_fitted_models = fit_timing_control_behaviour_models(riskset_table, config=timing_control_config)
        timing_control_summary = pd.concat(
            [
                timing_control_fitted_models[model_name].summary_table
                for model_name in ["M0_timing", "M1_rate_best_timing", "M2_expected_best_timing"]
            ],
            ignore_index=True,
            sort=False,
        )
        timing_control_comparison = compare_timing_control_models(timing_control_fitted_models)
        timing_control_fit_metrics = {
            "models": summarize_timing_control_fit_metrics(timing_control_fitted_models),
            "selected_lags": timing_control_selected_lags
            or {
                "best_information_rate_lag_ms": int(timing_control_config.primary_information_rate_lag_ms),
                "best_expected_cumulative_info_lag_ms": int(timing_control_config.primary_prop_expected_lag_ms),
                "delta_aic_convention": "child_aic - parent_aic; negative favours child",
            },
        }
        if timing_control_selected_lags is None:
            timing_control_selected_lags = dict(timing_control_fit_metrics["selected_lags"])
            timing_control_selected_lags.setdefault("best_information_rate_delta_aic", None)
            timing_control_selected_lags.setdefault("best_expected_cumulative_info_delta_aic", None)
            timing_control_selected_lags.setdefault("lag_selection_parent_for_information_rate", "M0_timing")
            timing_control_selected_lags.setdefault(
                "lag_selection_parent_for_expected_cumulative_info",
                "M1_rate_best_timing",
            )
        if timing_control_selected_lags_path is None:
            timing_control_selected_lags_path = write_json(
                timing_control_selected_lags,
                output_dirs["models"] / "behaviour_timing_control_selected_lags.json",
            )
        timing_control_summary_path = write_table(
            timing_control_summary,
            output_dirs["models"] / "behaviour_timing_control_model_summary.csv",
            sep=",",
        )
        timing_control_comparison_path = write_table(
            timing_control_comparison,
            output_dirs["models"] / "behaviour_timing_control_model_comparison.csv",
            sep=",",
        )
        timing_control_fit_metrics_path = write_json(
            timing_control_fit_metrics,
            output_dirs["models"] / "behaviour_timing_control_fit_metrics.json",
        )
        try:
            plot_primary_coefficients(
                timing_control_summary,
                output_dirs["figures"] / "behaviour_timing_control_coefficients.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            warnings_list.append(f"Failed to create figure behaviour_timing_control_coefficients.png: {error}")
        try:
            plot_primary_model_comparison(
                timing_control_comparison,
                output_dirs["figures"] / "behaviour_timing_control_model_comparison.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            warnings_list.append(f"Failed to create figure behaviour_timing_control_model_comparison.png: {error}")
        if not timing_control_lag_selection.empty:
            try:
                plot_behaviour_timing_control_delta_aic_by_lag(
                    timing_control_lag_selection,
                    output_dirs["figures"] / "behaviour_timing_control_delta_aic_by_lag.png",
                )
            except Exception as error:  # pragma: no cover - plotting fallback
                warnings_list.append(f"Failed to create figure behaviour_timing_control_delta_aic_by_lag.png: {error}")

    fitted_models: dict[str, object] = {}
    lagged_fitted_models: dict[str, object] = {}
    all_fitted_models: dict[str, object] = {}
    model_comparison = pd.DataFrame()
    lagged_model_comparison = pd.DataFrame()
    lagged_coefficients = pd.DataFrame()
    baseline_summary_path = None
    information_rate_summary_path = None
    cumulative_summary_path = None
    model_comparison_csv_path = None
    model_comparison_json_path = None
    model_fit_metrics_json_path = None
    lagged_model_comparison_csv_path = None
    lagged_model_comparison_json_path = None
    coefficient_by_lag_path = None
    model_fit_by_lag_path = None
    best_lag_by_aic_path = None
    lagged_feature_scaling_path = None
    if config.run_behaviour_model_suite:
        LOGGER.info("Fitting behavioural hazard models.")
        for model_name in ["M0", "M1", "M2a", "M2b", "M2c"]:
            LOGGER.info("Fitting model %s.", model_name)
            fitted_models[model_name] = fit_binomial_glm(riskset_table, model_name=model_name, config=config)
            LOGGER.info(
                "Finished model %s: n_rows=%d, n_events=%d, aic=%s",
                model_name,
                fitted_models[model_name].fit_metrics["n_rows"],
                fitted_models[model_name].fit_metrics["n_events"],
                fitted_models[model_name].fit_metrics["aic"],
            )

        if config.fit_lagged_models:
            LOGGER.info("Fitting lagged behavioural hazard model families.")
            for model_name, _predictors, lag_ms, _family_name in build_lagged_model_specs(config):
                try:
                    lagged_fitted_models[model_name] = fit_binomial_glm(riskset_table, model_name=model_name, config=config)
                    LOGGER.info("Finished lagged model %s (lag=%d ms).", model_name, lag_ms)
                except Exception as error:  # pragma: no cover - resilient pipeline path
                    warning = f"Failed to fit lagged model {model_name}: {error}"
                    LOGGER.warning(warning)
                    warnings_list.append(warning)
        all_fitted_models = {**fitted_models, **lagged_fitted_models}

        LOGGER.info("Writing model summaries and comparison tables.")
        baseline_summary_path = write_table(
            fitted_models["M0"].summary_table,
            output_dirs["models"] / "hazard_baseline_gam_summary.csv",
            sep=",",
        )
        information_rate_summary_path = write_table(
            fitted_models["M1"].summary_table,
            output_dirs["models"] / "hazard_information_rate_summary.csv",
            sep=",",
        )
        cumulative_models_summary = pd.concat(
            [
                fitted_models["M2a"].summary_table,
                fitted_models["M2b"].summary_table,
                fitted_models["M2c"].summary_table,
            ],
            ignore_index=True,
            sort=False,
        )
        cumulative_summary_path = write_table(
            cumulative_models_summary,
            output_dirs["models"] / "hazard_cumulative_info_models.csv",
            sep=",",
        )

        model_comparison = compare_nested_models(fitted_models)
        model_comparison_csv_path = write_table(
            model_comparison,
            output_dirs["models"] / "model_comparison_behaviour.csv",
            sep=",",
        )
        model_comparison_json_path = write_json(
            {"rows": model_comparison.to_dict(orient="records")},
            output_dirs["models"] / "model_comparison_behaviour.json",
        )

        model_fit_metrics = {
            "models": {name: fitted_model.fit_metrics for name, fitted_model in all_fitted_models.items()},
            "scaling": zscore_result.scaling,
            "primary_scaling": primary_scaling,
            "expected_total_info_by_group": expected_total_info_by_group,
            "used_partner_anchor": positive_result.used_partner_anchor,
            "full_data_expected_information": True,
        }
        model_fit_metrics_json_path = write_json(
            model_fit_metrics,
            output_dirs["models"] / "model_fit_metrics.json",
        )

    feature_qc_payload = {
        "n_rows": int(len(riskset_table)),
        "n_episodes": int(riskset_table["episode_id"].nunique()),
        "n_events": int(riskset_table["event"].sum()),
        "prop_actual_min": float(riskset_table["prop_actual_cumulative_info"].min(skipna=True)),
        "prop_actual_max": float(riskset_table["prop_actual_cumulative_info"].max(skipna=True)),
        "prop_expected_min": float(riskset_table["prop_expected_cumulative_info"].min(skipna=True)),
        "prop_expected_max": float(riskset_table["prop_expected_cumulative_info"].max(skipna=True)),
    }
    feature_qc_path = write_json(
        feature_qc_payload,
        output_dirs["riskset"] / "hazard_behavior_feature_qc.json",
    )
    lagged_feature_qc_path = write_json(
        lagged_feature_qc,
        output_dirs["riskset"] / "lagged_feature_qc.json",
    )
    neural_feature_qc_path = None
    if neural_qc_payload is not None:
        neural_feature_qc_path = write_json(
            neural_qc_payload,
            output_dirs["riskset"] / "neural_feature_qc.json",
        )

    riskset_path = None
    if config.save_riskset:
        riskset_path = write_table(
            riskset_table,
            output_dirs["riskset"] / "hazard_behavior_riskset.tsv",
        )
        if config.fit_timing_control_models:
            write_table(
                riskset_table,
                output_dirs["riskset"] / "hazard_behavior_riskset_with_timing_controls.tsv",
            )
        if neural_qc_payload is not None:
            write_table(
                riskset_table,
                output_dirs["riskset"] / "hazard_behavior_riskset_with_neural.tsv",
            )
    lagged_feature_table_path = None
    if config.save_lagged_feature_table:
        lagged_feature_table_path = write_table(
            riskset_table,
            output_dirs["riskset"] / "hazard_behavior_riskset_with_lagged_features.tsv",
        )
    episode_summary = riskset_result.episode_summary.merge(
        information_timing_summary,
        on=["episode_id", "partner_ipu_id", "dyad_id", "run"],
        how="left",
    )
    episode_summary_path = write_table(
        episode_summary,
        output_dirs["riskset"] / "hazard_behavior_episode_summary.tsv",
    )
    information_timing_summary_path = write_table(
        information_timing_summary,
        output_dirs["riskset"] / "information_timing_summary.csv",
        sep=",",
    )
    if partner_ipu_episode_summary_path is not None:
        partner_ipu_episode_summary_path = write_table(
            episode_summary,
            output_dirs["riskset"] / "partner_ipu_episode_summary.tsv",
        )
    write_json(
        riskset_result.event_qc,
        output_dirs["riskset"] / "riskset_event_qc.json",
    )

    if config.run_behaviour_model_suite:
        lagged_model_comparison = compare_nested_models(all_fitted_models)
        lagged_model_comparison_csv_path = write_table(
            lagged_model_comparison,
            output_dirs["models"] / "model_comparison_lagged_behaviour.csv",
            sep=",",
        )
        lagged_model_comparison_json_path = write_json(
            {"rows": lagged_model_comparison.to_dict(orient="records")},
            output_dirs["models"] / "model_comparison_lagged_behaviour.json",
        )
        lagged_coefficients = summarize_lagged_model_coefficients(all_fitted_models)
        coefficient_by_lag_path = write_table(
            lagged_coefficients,
            output_dirs["models"] / "coefficient_by_lag.csv",
            sep=",",
        )
        lagged_fit_summary = summarize_lagged_model_fit(all_fitted_models)
        model_fit_by_lag_path = write_table(
            lagged_fit_summary,
            output_dirs["models"] / "model_fit_by_lag.csv",
            sep=",",
        )
        best_lag_by_aic = summarize_best_lag_by_aic(all_fitted_models)
        best_lag_by_aic_path = write_table(
            best_lag_by_aic,
            output_dirs["models"] / "best_lag_by_aic.csv",
            sep=",",
        )
        lagged_scaling = {
            predictor: stats
            for predictor, stats in zscore_result.scaling.items()
            if "_lag_" in predictor
        }
        lagged_feature_scaling_path = write_json(
            {
                "scaling": lagged_scaling,
                "warnings": zscore_result.warnings,
            },
            output_dirs["models"] / "lagged_feature_scaling.json",
        )

    if config.fit_primary_behaviour_models and config.fit_primary_stat_tests:
        LOGGER.info("Writing compact primary behavioural statistical reporting layer.")
        primary_reporting = run_primary_stat_reporting(
            riskset_table=riskset_table,
            fitted_models=primary_fitted_models,
            comparison_table=primary_comparison,
            lagged_model_comparison=lagged_model_comparison,
            lagged_coefficients=lagged_coefficients,
            output_dirs=output_dirs,
            config=config,
        )
        primary_stat_tests_path = output_dirs["models"] / "behaviour_primary_stat_tests.csv"
        primary_stat_tests_json_path = output_dirs["models"] / "behaviour_primary_stat_tests.json"
        primary_publication_table_path = output_dirs["models"] / "behaviour_primary_publication_table.csv"
        primary_interpretation_path = output_dirs["models"] / "behaviour_primary_interpretation.txt"
        del primary_reporting

    if config.run_behaviour_model_suite:
        LOGGER.info("Generating prediction grids and figures.")
        prediction_grids = generate_prediction_grids(
            riskset_table=riskset_table,
            fitted_models=fitted_models,
            config=config,
        )
        for grid_name, grid_table in prediction_grids.items():
            write_table(grid_table, output_dirs["models"] / f"{grid_name}.csv", sep=",")
        plot_outputs = safe_make_plots(
            riskset_table=riskset_table,
            episodes_table=episode_summary,
            prediction_grids=prediction_grids,
            figures_dir=output_dirs["figures"],
            warnings_list=warnings_list,
            candidate_episode_table=positive_result.candidate_episodes,
            lagged_model_comparison=lagged_model_comparison,
            lagged_coefficients=lagged_coefficients,
            information_timing_summary=information_timing_summary,
        )
        write_json(
            plot_outputs["prop_actual_saturation_qc"],
            output_dirs["riskset"] / "prop_actual_saturation_qc.json",
        )
        write_json(
            plot_outputs["observed_event_rate_plot_qc"],
            output_dirs["riskset"] / "observed_event_rate_plot_qc.json",
        )
        write_table(
            plot_outputs["observed_event_rate_by_time_bin"],
            output_dirs["riskset"] / "observed_event_rate_by_time_bin.csv",
            sep=",",
        )
        write_table(
            plot_outputs["observed_event_rate_nonzero_bins"],
            output_dirs["riskset"] / "observed_event_rate_nonzero_bins.csv",
            sep=",",
        )
        write_table(
            plot_outputs["event_rate_by_prop_actual_saturation"],
            output_dirs["riskset"] / "event_rate_by_prop_actual_saturation.csv",
            sep=",",
        )
    neural_warnings_path = None
    if neural_fit_result is not None:
        write_table(
            neural_fit_result.summary_table,
            output_dirs["models"] / "neural_lowlevel_model_summary.csv",
            sep=",",
        )
        write_table(
            neural_fit_result.comparison_table,
            output_dirs["models"] / "neural_lowlevel_model_comparison.csv",
            sep=",",
        )
        write_table(
            neural_fit_result.family_comparison_table,
            output_dirs["models"] / "neural_lowlevel_family_model_comparison.csv",
            sep=",",
        )
        write_json(
            neural_fit_result.fit_metrics_payload,
            output_dirs["models"] / "neural_lowlevel_fit_metrics.json",
        )
        write_json(
            neural_fit_result.effects_payload,
            output_dirs["models"] / "neural_lowlevel_effects.json",
        )
        for family_name in ["amplitude", "alpha", "beta"]:
            family_result = neural_fit_result.pca_results.get(family_name)
            if family_result is None:
                empty_summary = pd.DataFrame(
                    columns=[
                        "family",
                        "component",
                        "explained_variance_ratio",
                        "cumulative_explained_variance",
                        "selected_for_model",
                    ]
                )
                empty_loadings = pd.DataFrame(columns=["feature", "family"])
                write_table(
                    empty_summary,
                    output_dirs["models"] / f"neural_lowlevel_pca_summary_{family_name}.csv",
                    sep=",",
                )
                write_table(
                    empty_loadings,
                    output_dirs["models"] / f"neural_lowlevel_pca_loadings_{family_name}.csv",
                    sep=",",
                )
                continue
            write_table(
                family_result.pca_summary,
                output_dirs["models"] / f"neural_lowlevel_pca_summary_{family_name}.csv",
                sep=",",
            )
            write_table(
                family_result.loadings,
                output_dirs["models"] / f"neural_lowlevel_pca_loadings_{family_name}.csv",
                sep=",",
            )
        plot_neural_lowlevel_pca_variance(
            {family_name: result.pca_summary for family_name, result in neural_fit_result.pca_results.items()},
            variance_threshold=config.neural_pca_variance_threshold,
            output_path=output_dirs["figures"] / "neural_lowlevel_pca_variance.png",
        )
        plot_neural_lowlevel_model_comparison(
            neural_fit_result.family_comparison_table,
            output_dirs["figures"] / "neural_lowlevel_model_comparison.png",
        )
        plot_neural_lowlevel_coefficients(
            neural_fit_result.summary_table,
            output_dirs["figures"] / "neural_lowlevel_coefficients.png",
        )
        if neural_qc_payload is not None:
            plot_neural_lowlevel_feature_missingness(
                neural_qc_payload,
                output_dirs["figures"] / "neural_lowlevel_feature_missingness.png",
            )
        neural_pc1_grid = _build_neural_pc1_prediction_grid(neural_fit_result, config=config)
        write_table(
            neural_pc1_grid,
            output_dirs["figures"] / "neural_lowlevel_predicted_hazard_pc1.csv",
            sep=",",
        )
        _plot_neural_pc1_prediction_curve(
            neural_pc1_grid,
            output_dirs["figures"] / "neural_lowlevel_predicted_hazard_pc1.png",
        )
        neural_warnings_path = output_dirs["logs"] / "neural_lowlevel_warnings.txt"
        neural_warnings_path.write_text(
            "\n".join(neural_fit_result.warnings).rstrip() + ("\n" if neural_fit_result.warnings else ""),
            encoding="utf-8",
        )

    config_json_path, warnings_path = save_config_and_warnings(
        config=config,
        warnings=warnings_list,
        logs_dir=output_dirs["logs"],
    )
    LOGGER.info("Behavioural hazard pipeline finished. Outputs written to %s", output_dirs["root"])
    return BehaviourHazardPipelineResult(
        out_dir=output_dirs["root"],
        riskset_path=riskset_path,
        episode_summary_path=episode_summary_path,
        feature_qc_path=feature_qc_path,
        episode_validation_qc_path=episode_validation_qc_path,
        excluded_episodes_path=excluded_episodes_path,
        model_comparison_csv_path=model_comparison_csv_path,
        model_fit_metrics_json_path=model_fit_metrics_json_path,
        config_json_path=config_json_path,
        warnings_path=warnings_path,
    )


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _validate_final_positive_episode_latencies(
    episodes_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> None:
    if episodes_table.empty or not config.require_partner_offset_before_fpp or config.overlapping_episode_strategy == "keep":
        return
    latencies = pd.to_numeric(episodes_table.get("latency_from_partner_offset_s"), errors="coerce")
    if latencies is None:
        return
    invalid_mask = np.isfinite(latencies) & (latencies < -config.partner_offset_fpp_tolerance_s)
    if invalid_mask.any():
        raise ValueError(
            "Negative partner-offset-to-FPP latencies remain after episode validation. "
            "This indicates invalid partner IPU assignment."
        )


def _build_neural_pc1_prediction_grid(
    neural_fit_result: NeuralLowLevelModelResult,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    model_table = neural_fit_result.model_table
    family_name = neural_fit_result.best_family_for_pc1
    if family_name is None:
        raise ValueError("Cannot create PC1 prediction grid because no neural family PCs are available.")
    family_prefix = {"amplitude": "amp", "alpha": "alpha", "beta": "beta", "combined": "neural"}[family_name]
    pc1_column = f"{family_prefix}_pc1"
    if pc1_column not in model_table.columns:
        raise ValueError(f"Cannot create PC1 prediction grid because `{pc1_column}` is unavailable.")
    base_row = {
        "time_from_partner_onset": float(pd.to_numeric(model_table["time_from_partner_onset"], errors="coerce").median()),
        f"z_information_rate_lag_{int(config.primary_information_rate_lag_ms)}ms": 0.0,
        f"z_prop_expected_cumulative_info_lag_{int(config.primary_prop_expected_lag_ms)}ms": 0.0,
    }
    for pc_column in neural_fit_result.full_pc_columns:
        base_row[pc_column] = 0.0
    pc1_values = np.linspace(
        float(pd.to_numeric(model_table[pc1_column], errors="coerce").quantile(0.05)),
        float(pd.to_numeric(model_table[pc1_column], errors="coerce").quantile(0.95)),
        num=100,
    )
    prediction_grid = pd.DataFrame([{**base_row, pc1_column: float(value)} for value in pc1_values])
    prediction_grid["predicted_hazard"] = np.asarray(
        neural_fit_result.child_model.result.predict(prediction_grid),
        dtype=float,
    )
    prediction_grid["pc1_column"] = pc1_column
    prediction_grid["family"] = family_name
    return prediction_grid


def _plot_neural_pc1_prediction_curve(prediction_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    pc1_column = str(prediction_table["pc1_column"].iloc[0])
    family_name = str(prediction_table["family"].iloc[0])
    axis.plot(prediction_table[pc1_column], prediction_table["predicted_hazard"], color="#1f4e79", linewidth=2.2)
    axis.set_xlabel(pc1_column)
    axis.set_ylabel("Predicted hazard")
    axis.set_title(f"Predicted hazard as {family_name} PC1 varies")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
