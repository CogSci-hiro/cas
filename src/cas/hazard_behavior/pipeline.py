"""End-to-end behavioural hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_censored_episodes, build_event_positive_episodes
from cas.hazard_behavior.features import add_information_features_to_riskset, zscore_predictors
from cas.hazard_behavior.io import (
    ensure_output_directories,
    read_events_table,
    read_surprisal_tables,
    save_config_and_warnings,
    write_json,
    write_table,
)
from cas.hazard_behavior.model import compare_nested_models, fit_binomial_glm, generate_prediction_grids
from cas.hazard_behavior.plots import safe_make_plots
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
    model_comparison_csv_path: Path
    model_fit_metrics_json_path: Path
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
    LOGGER.info("Built %d event-positive episodes.", len(positive_result.episodes))
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
    LOGGER.info("Building censored episodes.")
    censored_episodes = build_censored_episodes(
        events_table=events_table,
        surprisal_table=surprisal_table,
        positive_episodes=positive_result.episodes,
        config=config,
    )
    LOGGER.info("Built %d censored episodes.", len(censored_episodes))

    episodes_table = positive_result.episodes
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
    LOGGER.info("Scaling predictors.")
    zscore_result = zscore_predictors(riskset_with_features)
    riskset_table = zscore_result.table

    LOGGER.info("Fitting behavioural hazard models.")
    fitted_models = {}
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
        "models": {name: fitted_model.fit_metrics for name, fitted_model in fitted_models.items()},
        "scaling": zscore_result.scaling,
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

    riskset_path = None
    if config.save_riskset:
        riskset_path = write_table(
            riskset_table,
            output_dirs["riskset"] / "hazard_behavior_riskset.tsv",
        )
    episode_summary_path = write_table(
        riskset_result.episode_summary,
        output_dirs["riskset"] / "hazard_behavior_episode_summary.tsv",
    )

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
        prediction_grids=prediction_grids,
        figures_dir=output_dirs["figures"],
        warnings_list=warnings_list,
        candidate_episode_table=positive_result.candidate_episodes,
    )
    write_json(
        plot_outputs["prop_actual_saturation_qc"],
        output_dirs["riskset"] / "prop_actual_saturation_qc.json",
    )
    write_table(
        plot_outputs["event_rate_by_prop_actual_saturation"],
        output_dirs["riskset"] / "event_rate_by_prop_actual_saturation.csv",
        sep=",",
    )

    config_json_path, warnings_path = save_config_and_warnings(
        config=config,
        warnings=warnings_list,
        logs_dir=output_dirs["logs"],
    )
    LOGGER.info("Behavioural hazard pipeline finished. Outputs written to %s", output_dirs["root"])

    del baseline_summary_path, information_rate_summary_path, cumulative_summary_path, model_comparison_json_path
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
