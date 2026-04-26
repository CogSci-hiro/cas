"""End-to-end first-pass partner-onset hazard analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cas.hazard.config import HazardAnalysisConfig, config_to_metadata_dict, load_hazard_analysis_config
from cas.hazard.io import (
    build_analysis_metadata,
    load_lowlevel_neural_tables,
    load_normalized_events_table,
    load_surprisal_table,
    load_dyad_table,
    load_events_table,
    load_pairing_issues_table,
    load_reused_tde_hmm_outputs,
    prepare_neural_output_directories,
    prepare_output_directory,
    write_neural_coefficients,
    write_neural_hazard_table,
    write_neural_model_comparison,
    write_coefficients_table,
    write_entropy_alignment_table,
    write_hazard_table,
    write_json_payload,
    write_model_summary,
    write_note_file,
)
from cas.hazard.model import fit_pooled_discrete_time_hazard_model
from cas.hazard.model import fit_neural_hazard_model_family
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
from cas.hazard.riskset import build_neural_partner_ipu_risksets, build_person_period_riskset
from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.features import add_information_features_to_riskset, add_lagged_information_features, zscore_predictors

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
    neural_fpp_hazard_table_path: Path | None = None
    neural_spp_hazard_table_path: Path | None = None
    neural_model_comparison_path: Path | None = None
    neural_coefficients_path: Path | None = None


def run_hazard_analysis_from_config(config_path: str | Path) -> HazardAnalysisResult:
    """Run the full first-pass hazard analysis from a config file."""

    config = load_hazard_analysis_config(config_path)
    return run_hazard_analysis(config)


def run_hazard_analysis(config: HazardAnalysisConfig) -> HazardAnalysisResult:
    """Run the full first-pass partner-onset hazard analysis."""

    if config.mode == "neural_lowlevel" or config.neural.enabled:
        return _run_neural_lowlevel_hazard_analysis(config)

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


def _run_neural_lowlevel_hazard_analysis(config: HazardAnalysisConfig) -> HazardAnalysisResult:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    neural_config = config.neural
    neural_config.validate()
    out_dir = (neural_config.out_dir or config.output.output_dir).resolve()
    output_dirs = prepare_neural_output_directories(out_dir, overwrite=config.misc.overwrite)
    events_table, event_warnings = load_normalized_events_table(neural_config.events_path or config.input.events_table_path)
    surprisal_table, surprisal_warnings = load_surprisal_table(neural_config.input.surprisal_paths if neural_config.input else tuple())
    lowlevel_table = load_lowlevel_neural_tables(neural_config.input.lowlevel_neural_paths if neural_config.input else tuple())

    riskset_build_result = build_neural_partner_ipu_risksets(
        events_table=events_table,
        surprisal_table=surprisal_table,
        neural_config=neural_config,
    )
    model_comparison_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    fit_metadata: dict[str, Any] = {
        "warnings": [*event_warnings, *surprisal_warnings, *riskset_build_result.warnings],
        "event_qc": riskset_build_result.event_qc_by_event,
        "window": {
            "start_lag_s": neural_config.window.start_lag_s,
            "end_lag_s": neural_config.window.end_lag_s,
        },
    }
    hazard_paths: dict[str, Path] = {}

    for event_type, riskset_table in riskset_build_result.risksets_by_event.items():
        episodes_table = riskset_build_result.episode_summaries_by_event[event_type]
        enriched = _add_behavioural_controls_to_neural_riskset(
            riskset_table,
            episodes_table=episodes_table,
            surprisal_table=surprisal_table,
            config=config,
        )
        enriched_with_neural, neural_qc = _add_neural_features_and_pcs(
            enriched,
            lowlevel_table=lowlevel_table,
            config=config,
        )
        model_result = fit_neural_hazard_model_family(
            enriched_with_neural,
            event_type=event_type,
            neural_config=neural_config,
        )
        fit_metadata[f"{event_type}_neural_qc"] = neural_qc
        fit_metadata[f"{event_type}_model_warnings"] = model_result.warnings
        hazard_paths[event_type] = write_neural_hazard_table(
            enriched_with_neural,
            output_dir=output_dirs["riskset"],
            event_type=event_type,
        )
        model_comparison_frames.append(model_result.comparison_table)
        coefficient_frames.append(model_result.coefficients_table)

    model_comparison_table = pd.concat(model_comparison_frames, ignore_index=True, sort=False)
    coefficients_table = pd.concat(coefficient_frames, ignore_index=True, sort=False)
    neural_model_comparison_path = write_neural_model_comparison(model_comparison_table, output_dir=output_dirs["models"])
    neural_coefficients_path = write_neural_coefficients(coefficients_table, output_dir=output_dirs["models"])
    fit_metrics_path = write_json_payload(fit_metadata, output_dirs["models"] / "neural_fit_metrics.json")
    analysis_metadata_path = write_json_payload(
        {
            "config": config_to_metadata_dict(config),
            "notes": [
                "Primary neural model family uses participant-speaker alpha/beta power from guarded causal windows.",
                "Raw amplitude and HMM entropy are excluded from the primary neural model family.",
            ],
        },
        output_dirs["logs"] / "analysis_metadata.json",
    )
    _plot_neural_model_comparison(model_comparison_table, output_dirs["figures"])
    _plot_neural_coefficients(coefficients_table, output_dirs["figures"])
    _plot_neural_power_qc(
        {
            event_type: pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            for event_type, path in hazard_paths.items()
        },
        output_dirs["figures"],
    )

    placeholder_summary_path = output_dirs["models"] / "model_summary.txt"
    placeholder_summary_path.write_text(
        "Neural low-level hazard analysis completed. See neural_model_comparison.csv.\n",
        encoding="utf-8",
    )
    return HazardAnalysisResult(
        output_dir=output_dirs["root"],
        hazard_table_path=hazard_paths.get("fpp", next(iter(hazard_paths.values()))),
        entropy_alignment_table_path=hazard_paths.get("spp", next(iter(hazard_paths.values()))),
        coefficients_path=neural_coefficients_path,
        model_summary_path=placeholder_summary_path,
        fit_metrics_path=fit_metrics_path,
        analysis_metadata_path=analysis_metadata_path,
        neural_fpp_hazard_table_path=hazard_paths.get("fpp"),
        neural_spp_hazard_table_path=hazard_paths.get("spp"),
        neural_model_comparison_path=neural_model_comparison_path,
        neural_coefficients_path=neural_coefficients_path,
    )


def _add_behavioural_controls_to_neural_riskset(
    riskset_table: pd.DataFrame,
    *,
    episodes_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: HazardAnalysisConfig,
) -> pd.DataFrame:
    behaviour_config = BehaviourHazardConfig(
        events_path=config.neural.events_path or config.input.events_table_path,
        surprisal_paths=config.neural.input.surprisal_paths if config.neural.input else tuple(),
        out_dir=config.neural.out_dir or config.output.output_dir,
        bin_size_s=float(config.neural.bin_size_s),
        max_followup_s=float(config.neural.episode.max_followup_s),
        ipu_gap_threshold_s=float(config.neural.episode.ipu_gap_threshold_s),
        include_censored=bool(config.neural.episode.include_censored),
        lag_grid_ms=tuple(
            sorted(
                {
                    0,
                    int(config.neural.model.information_rate_lag_ms),
                    int(config.neural.model.prop_expected_lag_ms),
                }
            )
        ),
    )
    with_features, _expected_totals = add_information_features_to_riskset(
        riskset_table=riskset_table,
        episodes_table=episodes_table,
        surprisal_table=surprisal_table,
        config=behaviour_config,
    )
    with_lags, _lag_qc = add_lagged_information_features(with_features, config=behaviour_config)
    zscore_result = zscore_predictors(with_lags)
    table = zscore_result.table.copy()
    if "participant_id" not in table.columns:
        table["participant_id"] = table["dyad_id"].astype(str) + "_" + table["participant_speaker"].astype(str)
    required_columns = [
        f"z_information_rate_lag_{int(config.neural.model.information_rate_lag_ms)}ms",
        f"z_prop_expected_cumulative_info_lag_{int(config.neural.model.prop_expected_lag_ms)}ms",
    ]
    missing = [column_name for column_name in required_columns if column_name not in table.columns]
    if missing:
        raise ValueError(
            "Neural baseline behavioural controls are missing required lagged columns: "
            + ", ".join(missing)
        )
    return table


def _add_neural_features_and_pcs(
    riskset_table: pd.DataFrame,
    *,
    lowlevel_table: pd.DataFrame,
    config: HazardAnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = riskset_table.copy()
    alpha_columns = [column_name for column_name in lowlevel_table.columns if column_name.startswith("alpha_")]
    beta_columns = [column_name for column_name in lowlevel_table.columns if column_name.startswith("beta_")]
    if not alpha_columns or not beta_columns:
        raise ValueError("Low-level neural table must include alpha_* and beta_* columns.")

    alpha_output_columns = [f"mean_log_{column_name}" for column_name in alpha_columns]
    beta_output_columns = [f"mean_log_{column_name}" for column_name in beta_columns]
    for column_name in alpha_output_columns + beta_output_columns:
        table[column_name] = np.nan
    table["neural_window_start"] = pd.to_numeric(table["bin_end"], errors="coerce") - float(config.neural.window.start_lag_s)
    table["neural_window_end"] = pd.to_numeric(table["bin_end"], errors="coerce") - float(config.neural.window.end_lag_s)
    table["neural_speaker_used"] = table["participant_speaker"].astype(str)

    if not (pd.to_numeric(table["neural_window_end"], errors="coerce") <= pd.to_numeric(table["bin_start"], errors="coerce") + 1.0e-9).all():
        raise ValueError("Neural causal window end must not exceed hazard-bin start.")
    if not (table["neural_speaker_used"].astype(str) == table["participant_speaker"].astype(str)).all():
        raise ValueError("Neural features must use participant_speaker EEG, not partner EEG.")

    grouped_lowlevel = {
        (str(dyad_id), str(run), str(speaker)): frame.sort_values("time", kind="mergesort")
        for (dyad_id, run, speaker), frame in lowlevel_table.groupby(["dyad_id", "run", "speaker"], sort=False)
    }
    for group_key, group_rows in table.groupby(["dyad_id", "run", "participant_speaker"], sort=False):
        lowlevel = grouped_lowlevel.get((str(group_key[0]), str(group_key[1]), str(group_key[2])))
        if lowlevel is None or lowlevel.empty:
            continue
        times = pd.to_numeric(lowlevel["time"], errors="coerce").to_numpy(dtype=float)
        starts = pd.to_numeric(group_rows["neural_window_start"], errors="coerce").to_numpy(dtype=float)
        ends = pd.to_numeric(group_rows["neural_window_end"], errors="coerce").to_numpy(dtype=float)
        alpha_matrix = np.log(
            np.clip(lowlevel.loc[:, alpha_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), config.neural.window.epsilon, None)
        )
        beta_matrix = np.log(
            np.clip(lowlevel.loc[:, beta_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), config.neural.window.epsilon, None)
        )
        alpha_means = _windowed_means(times, alpha_matrix, starts=starts, ends=ends)
        beta_means = _windowed_means(times, beta_matrix, starts=starts, ends=ends)
        table.loc[group_rows.index, alpha_output_columns] = alpha_means
        table.loc[group_rows.index, beta_output_columns] = beta_means

    table, alpha_meta = _add_band_pcs(
        table,
        feature_columns=alpha_output_columns,
        band_name="alpha",
        config=config,
    )
    table, beta_meta = _add_band_pcs(
        table,
        feature_columns=beta_output_columns,
        band_name="beta",
        config=config,
    )
    qc = {
        "participant_speaker_eeg_used": True,
        "causal_guard_passed": True,
        "alpha_feature_columns": alpha_output_columns,
        "beta_feature_columns": beta_output_columns,
        "alpha_pca": alpha_meta,
        "beta_pca": beta_meta,
    }
    return table, qc


def _windowed_means(
    times: np.ndarray,
    matrix: np.ndarray,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    start_indices = np.searchsorted(times, starts, side="left")
    end_indices = np.searchsorted(times, ends, side="left")
    cumulative = np.vstack([np.zeros((1, matrix.shape[1]), dtype=float), np.cumsum(matrix, axis=0)])
    sums = cumulative[end_indices] - cumulative[start_indices]
    counts = (end_indices - start_indices).astype(float)
    means = np.full((starts.shape[0], matrix.shape[1]), np.nan, dtype=float)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid, None]
    return means


def _add_band_pcs(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    band_name: str,
    config: HazardAnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = table.copy()
    feature_frame = working.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(feature_frame).all(axis=1)
    valid = feature_frame.loc[valid_mask].to_numpy(dtype=float)
    if valid.size == 0:
        raise ValueError(f"No valid rows were available for {band_name} PCA.")
    scaler = StandardScaler()
    z_features = scaler.fit_transform(valid)
    if config.neural.pca.mode == "count":
        n_components = int(min(config.neural.pca.n_components, z_features.shape[0], z_features.shape[1]))
    else:
        n_components = float(config.neural.pca.variance_threshold)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(z_features)
    if scores.ndim == 1:
        scores = scores[:, None]
    for component_index in range(scores.shape[1]):
        score_values = scores[:, component_index]
        mean_value = float(np.mean(score_values))
        std_value = float(np.std(score_values, ddof=0))
        z_values = np.zeros_like(score_values) if std_value <= 0.0 else (score_values - mean_value) / std_value
        column_name = f"z_{band_name}_pc{component_index + 1}"
        working[column_name] = np.nan
        working.loc[valid_mask, column_name] = z_values
    return working, {
        "n_input_features": int(len(feature_columns)),
        "n_rows_valid": int(valid.shape[0]),
        "n_components_retained": int(scores.shape[1]),
        "explained_variance_ratio": [float(value) for value in np.asarray(pca.explained_variance_ratio_, dtype=float)],
    }


def _plot_neural_model_comparison(model_comparison: pd.DataFrame, figures_dir: Path) -> None:
    for metric in ("delta_bic", "delta_aic"):
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
        for axis, event_type in zip(axes, ["fpp", "spp"], strict=True):
            subset = model_comparison.loc[model_comparison["event_type"] == event_type].copy()
            subset["family"] = subset["child_model"].astype(str).str.replace(
                f"M_|_{event_type.upper()}",
                "",
                regex=True,
            ).str.lower()
            order = ["alpha", "beta", "alpha_beta"]
            heights = [
                float(pd.to_numeric(subset.loc[subset["family"] == family, metric], errors="coerce").iloc[0])
                if not subset.loc[subset["family"] == family].empty
                else np.nan
                for family in order
            ]
            axis.bar(order, heights, color=["#3b7ea1", "#7fa650", "#d17c4b"])
            axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
            axis.set_title(event_type.upper())
            axis.set_ylabel(f"{metric} (child - baseline)")
            axis.set_xlabel("Neural family")
        figure.tight_layout()
        figure.savefig(figures_dir / f"neural_{metric}_fpp_vs_spp.png", dpi=300)
        plt.close(figure)


def _plot_neural_coefficients(coefficients: pd.DataFrame, figures_dir: Path) -> None:
    subset = coefficients.loc[
        coefficients["term"].astype(str).str.startswith("z_alpha_pc")
        | coefficients["term"].astype(str).str.startswith("z_beta_pc")
    ].copy()
    if subset.empty:
        return
    subset["event_type"] = subset["event_type"].astype(str).str.upper()
    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    x_labels = [f"{event}-{term}" for event, term in zip(subset["event_type"], subset["term"], strict=True)]
    axis.bar(np.arange(len(subset)), pd.to_numeric(subset["estimate"], errors="coerce"), color="#3b7ea1")
    axis.set_xticks(np.arange(len(subset)))
    axis.set_xticklabels(x_labels, rotation=65, ha="right")
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Coefficient estimate")
    axis.set_title("Neural PC coefficients across FPP/SPP models")
    figure.tight_layout()
    figure.savefig(figures_dir / "neural_coefficients_fpp_vs_spp.png", dpi=300)
    plt.close(figure)


def _plot_neural_power_qc(risksets_by_event: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    combined = pd.concat(risksets_by_event.values(), ignore_index=True, sort=False)
    available_columns = [column_name for column_name in ("z_alpha_pc1", "z_beta_pc1") if column_name in combined.columns]
    if not available_columns:
        return
    figure, axes = plt.subplots(2, len(available_columns), figsize=(5.5 * len(available_columns), 7.5))
    if len(available_columns) == 1:
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)
    for column_index, predictor in enumerate(available_columns):
        for row_index, x_column in enumerate(["time_from_partner_onset", "time_from_partner_offset"]):
            axis = axes[row_index, column_index]
            x_values = pd.to_numeric(combined[x_column], errors="coerce")
            y_values = pd.to_numeric(combined[predictor], errors="coerce")
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            axis.scatter(x_values[mask], y_values[mask], s=5, alpha=0.15, color="#3b7ea1")
            axis.set_xlabel(x_column)
            axis.set_ylabel(predictor)
            axis.set_title(f"{predictor} vs {x_column}")
    figure.tight_layout()
    figure.savefig(figures_dir / "neural_power_by_partner_time.png", dpi=300)
    plt.close(figure)
