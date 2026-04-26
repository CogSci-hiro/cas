"""End-to-end active behavioural hazard preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path

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
from cas.hazard_behavior.model import (
    compare_timing_control_models,
    ensure_primary_predictors_available,
    fit_timing_control_behaviour_models,
    select_timing_control_best_lags,
    summarize_timing_control_fit_metrics,
)
from cas.hazard_behavior.plots import (
    plot_behaviour_delta_bic_by_lag,
    plot_information_rate_by_partner_time,
)
from cas.hazard_behavior.riskset import build_discrete_time_riskset

LOGGER = logging.getLogger(__name__)
SCREENING_NOTE_TEXT = (
    "Python pooled lag screening is for QC/screening only.\n"
    "Final behavioural lag inference uses the R GLMM lag sweep outputs.\n"
)


@dataclass(frozen=True, slots=True)
class BehaviourHazardPipelineResult:
    out_dir: Path
    riskset_path: Path | None
    timing_control_riskset_path: Path | None
    timing_control_lag_selection_path: Path | None
    timing_control_selected_lags_path: Path | None
    timing_control_summary_path: Path | None
    timing_control_comparison_path: Path | None
    timing_control_fit_metrics_path: Path | None
    config_json_path: Path
    warnings_path: Path


def run_behaviour_hazard_pipeline(config: BehaviourHazardConfig) -> BehaviourHazardPipelineResult:
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
    write_json(positive_result.validation_qc, output_dirs["riskset"] / "episode_validation_qc.json")
    write_table(positive_result.excluded_episodes, output_dirs["riskset"] / "excluded_episodes.tsv")

    episodes_table = positive_result.episodes
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
    required_lags = {
        int(config.primary_information_rate_lag_ms),
        int(config.primary_prop_expected_lag_ms),
    }
    augmented_lag_grid = tuple(
        sorted(
            {int(lag_ms) for lag_ms in config.lag_grid_ms}
            | {int(lag_ms) for lag_ms in config.r_glmm_lag_grid_ms}
            | required_lags
        )
    )
    lagged_feature_config = replace(config, lag_grid_ms=augmented_lag_grid)
    riskset_with_lagged_features, lagged_feature_qc = add_lagged_information_features(
        riskset_with_features,
        config=lagged_feature_config,
    )
    LOGGER.info("Scaling predictors.")
    zscore_result = zscore_predictors(riskset_with_lagged_features)
    riskset_table, primary_scaling = ensure_primary_predictors_available(zscore_result.table, config=config)
    warnings_list.extend(zscore_result.warnings)

    LOGGER.info("Selecting behavioural lags against timing-controlled parent models.")
    timing_control_lag_selection, timing_control_selected_lags = select_timing_control_best_lags(
        riskset_table,
        config=config,
    )
    timing_control_lag_selection_path = write_table(
        timing_control_lag_selection,
        output_dirs["models_lag_selection"] / "behaviour_timing_control_lag_selection.csv",
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

    LOGGER.info("Fitting behavioural timing-control models.")
    timing_control_fitted_models = fit_timing_control_behaviour_models(riskset_table, config=timing_control_config)
    timing_control_summary = pd.concat(
        [timing_control_fitted_models[model_name].summary_table for model_name in ["M0_timing", "M1_rate", "M2_expected"]],
        ignore_index=True,
        sort=False,
    )
    timing_control_comparison = compare_timing_control_models(timing_control_fitted_models)
    timing_control_fit_metrics = {
        "models": summarize_timing_control_fit_metrics(timing_control_fitted_models),
        "selected_lags": timing_control_selected_lags,
        "primary_scaling": primary_scaling,
        "expected_total_info_by_group": expected_total_info_by_group,
    }
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
    plot_behaviour_delta_bic_by_lag(
        timing_control_lag_selection,
        output_dirs["qc_plots_lag_selection"] / "behaviour_pooled_delta_bic_by_lag.png",
    )
    try:
        plot_information_rate_by_partner_time(riskset_table, output_dirs["figures"])
    except Exception as exc:  # pragma: no cover - defensive reporting guard
        warning_message = f"Information-rate timing diagnostic plot failed: {exc}"
        LOGGER.warning(warning_message)
        warnings_list.append(warning_message)
    screening_note_path = output_dirs["models"] / "behaviour_lag_screening_note.txt"
    screening_note_path.write_text(SCREENING_NOTE_TEXT, encoding="utf-8")

    write_json(
        {
            "n_rows": int(len(riskset_table)),
            "n_episodes": int(riskset_table["episode_id"].nunique()),
            "n_events": int(riskset_table["event"].sum()),
            "information_rate_min": float(pd.to_numeric(riskset_table["information_rate"], errors="coerce").min(skipna=True)),
            "information_rate_max": float(pd.to_numeric(riskset_table["information_rate"], errors="coerce").max(skipna=True)),
            "prop_expected_min": float(pd.to_numeric(riskset_table["prop_expected_cumulative_info"], errors="coerce").min(skipna=True)),
            "prop_expected_max": float(pd.to_numeric(riskset_table["prop_expected_cumulative_info"], errors="coerce").max(skipna=True)),
        },
        output_dirs["riskset"] / "hazard_behavior_feature_qc.json",
    )
    write_json(lagged_feature_qc, output_dirs["riskset"] / "lagged_feature_qc.json")
    write_json(riskset_result.event_qc, output_dirs["riskset"] / "riskset_event_qc.json")
    write_table(
        riskset_result.episode_summary.merge(
            information_timing_summary,
            on=["episode_id", "partner_ipu_id", "dyad_id", "run"],
            how="left",
        ),
        output_dirs["riskset"] / "hazard_behavior_episode_summary.tsv",
    )
    write_table(information_timing_summary, output_dirs["riskset"] / "information_timing_summary.csv", sep=",")

    riskset_path = None
    timing_control_riskset_path = None
    if config.save_riskset:
        riskset_path = write_table(riskset_table, output_dirs["riskset"] / "hazard_behavior_riskset.tsv")
        timing_control_riskset_path = write_table(
            riskset_table,
            output_dirs["riskset"] / "hazard_behavior_riskset_with_timing_controls.tsv",
        )

    config_json_path, warnings_path = save_config_and_warnings(
        config=timing_control_config,
        warnings=warnings_list,
        logs_dir=output_dirs["logs"],
    )
    LOGGER.info("Behavioural hazard pipeline finished. Outputs written to %s", output_dirs["root"])
    return BehaviourHazardPipelineResult(
        out_dir=output_dirs["root"],
        riskset_path=riskset_path,
        timing_control_riskset_path=timing_control_riskset_path,
        timing_control_lag_selection_path=timing_control_lag_selection_path,
        timing_control_selected_lags_path=timing_control_selected_lags_path,
        timing_control_summary_path=timing_control_summary_path,
        timing_control_comparison_path=timing_control_comparison_path,
        timing_control_fit_metrics_path=timing_control_fit_metrics_path,
        config_json_path=config_json_path,
        warnings_path=warnings_path,
    )


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
