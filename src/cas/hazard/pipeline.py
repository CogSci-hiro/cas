"""End-to-end first-pass partner-onset hazard analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cas.hazard.config import HazardAnalysisConfig, load_hazard_analysis_config
from cas.hazard.io import (
    build_analysis_metadata,
    load_dyad_table,
    load_events_table,
    load_pairing_issues_table,
    load_reused_tde_hmm_outputs,
    prepare_output_directory,
    write_coefficients_table,
    write_entropy_alignment_table,
    write_hazard_table,
    write_json_payload,
    write_model_summary,
    write_note_file,
)
from cas.hazard.model import fit_pooled_discrete_time_hazard_model
from cas.hazard.plots import (
    make_entropy_distribution_terminal_vs_nonterminal_artifact,
    make_model_vs_observed_hazard_artifact,
    make_observed_event_rate_by_entropy_quantile_artifact,
    make_observed_hazard_by_time_and_entropy_quantile_artifact,
    make_observed_hazard_by_time_bin_artifact,
    make_observed_hazard_by_time_bin_smoothed_artifact,
    plot_entropy_after_partner_onset,
    plot_entropy_histogram,
    plot_predicted_hazard_by_entropy,
    write_plot_interpretation_notes,
)
from cas.hazard.riskset import build_person_period_riskset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HazardAnalysisResult:
    """Top-level hazard-analysis outputs."""

    output_dir: Path
    hazard_table_path: Path
    entropy_alignment_table_path: Path
    coefficients_path: Path
    model_summary_path: Path
    fit_metrics_path: Path
    analysis_metadata_path: Path


def run_hazard_analysis_from_config(config_path: str | Path) -> HazardAnalysisResult:
    """Run the full first-pass hazard analysis from a config file."""

    config = load_hazard_analysis_config(config_path)
    return run_hazard_analysis(config)


def run_hazard_analysis(config: HazardAnalysisConfig) -> HazardAnalysisResult:
    """Run the full first-pass partner-onset hazard analysis."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Configuration
    output_dir = config.output.output_dir.resolve()
    prepare_output_directory(output_dir=output_dir, overwrite=config.misc.overwrite)

    # Load TDE-HMM outputs
    hmm_inputs = load_reused_tde_hmm_outputs(config)
    events_table = load_events_table(config.input.events_table_path)
    pairing_issues_table = load_pairing_issues_table(config.input.pairing_issues_table_path)
    dyad_table = load_dyad_table(config.input.dyads_csv_path)

    # Build risk-set table
    riskset_result = build_person_period_riskset(
        config=config,
        events_table=events_table,
        pairing_issues_table=pairing_issues_table,
        entropy_by_run=hmm_inputs.entropy_by_run,
        dyad_table=dyad_table,
    )
    LOGGER.info("Number of subjects in pooled analysis: %d", riskset_result.n_subjects)
    LOGGER.info("Number of episodes in pooled analysis: %d", riskset_result.n_episodes)
    LOGGER.info("Number of positive episodes: %d", riskset_result.n_positive_episodes)
    LOGGER.info("Number of censored episodes: %d", riskset_result.n_censored_episodes)
    LOGGER.info("Number of person-period rows: %d", len(riskset_result.hazard_table))

    # Fit hazard model
    model_result = fit_pooled_discrete_time_hazard_model(
        riskset_result.hazard_table,
        fitting_backend=config.model.fitting_backend,
        include_quadratic_time=config.model.include_quadratic_time,
        prefer_random_intercept_subject=config.model.prefer_random_intercept_subject,
    )
    LOGGER.info("Hazard-model backend used: %s", model_result.backend_used)

    hazard_table_path = write_hazard_table(riskset_result.hazard_table, output_dir)
    entropy_alignment_table_path = write_entropy_alignment_table(
        riskset_result.aligned_entropy_table,
        output_dir,
    )
    coefficients_path = write_coefficients_table(model_result.coefficients, output_dir)
    model_summary_path = write_model_summary(model_result.summary_text, output_dir)
    fit_metrics_path = write_json_payload(model_result.fit_metrics, output_dir / "fit_metrics.json")

    analysis_metadata = build_analysis_metadata(
        config=config,
        hmm_inputs=hmm_inputs,
        event_table_columns=list(events_table.columns),
        pairing_issues_columns=None if pairing_issues_table is None else list(pairing_issues_table.columns),
        warnings=riskset_result.warnings,
        subject_mapping_assumptions=riskset_result.subject_mapping_assumptions,
    )
    analysis_metadata_path = write_json_payload(
        analysis_metadata,
        output_dir / "analysis_metadata.json",
    )

    # QC outputs
    if config.qc.make_entropy_trajectory_plot:
        plot_entropy_after_partner_onset(riskset_result.aligned_entropy_table, output_dir=output_dir)
    if config.qc.make_entropy_histogram:
        plot_entropy_histogram(riskset_result.hazard_table, output_dir=output_dir)
    if config.qc.make_observed_hazard_by_time_bin_plot:
        make_observed_hazard_by_time_bin_artifact(
            riskset_result.hazard_table,
            output_dir=output_dir,
            plotting_config=config.plotting,
        )
    if config.qc.make_observed_hazard_by_time_bin_smoothed_plot:
        make_observed_hazard_by_time_bin_smoothed_artifact(
            riskset_result.hazard_table,
            output_dir=output_dir,
            plotting_config=config.plotting,
        )
    if config.qc.make_predicted_hazard_plot:
        if model_result.should_skip_prediction_plot:
            write_note_file(
                model_result.prediction_skip_reason or "Predicted hazard plot was skipped.",
                output_dir / "predicted_hazard_by_entropy.note.txt",
            )
        else:
            plot_predicted_hazard_by_entropy(model_result.prediction_table, output_dir=output_dir)
    if (
        config.qc.make_event_rate_by_entropy_quantile_plot
        or config.qc.make_observed_event_rate_by_entropy_quantile_plot
    ):
        make_observed_event_rate_by_entropy_quantile_artifact(
            riskset_result.hazard_table,
            output_dir=output_dir,
            plotting_config=config.plotting,
        )
    if config.qc.make_observed_hazard_by_time_and_entropy_quantile_plot:
        make_observed_hazard_by_time_and_entropy_quantile_artifact(
            riskset_result.hazard_table,
            output_dir=output_dir,
            plotting_config=config.plotting,
        )
    if config.qc.make_model_vs_observed_hazard_plot:
        make_model_vs_observed_hazard_artifact(
            riskset_result.hazard_table,
            model_result.prediction_table,
            output_dir=output_dir,
            plotting_config=config.plotting,
        )
    if config.qc.make_entropy_distribution_terminal_vs_nonterminal_plot:
        make_entropy_distribution_terminal_vs_nonterminal_artifact(
            riskset_result.hazard_table,
            output_dir=output_dir,
        )
    write_plot_interpretation_notes(output_dir)

    return HazardAnalysisResult(
        output_dir=output_dir,
        hazard_table_path=hazard_table_path,
        entropy_alignment_table_path=entropy_alignment_table_path,
        coefficients_path=coefficients_path,
        model_summary_path=model_summary_path,
        fit_metrics_path=fit_metrics_path,
        analysis_metadata_path=analysis_metadata_path,
    )
