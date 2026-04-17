"""QC plotting helpers for the partner-onset hazard analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.hazard.config import PlottingConfig

LOGGER = logging.getLogger(__name__)
HISTOGRAM_BIN_COUNT = 40


@dataclass(frozen=True, slots=True)
class PlotArtifact:
    """Container for a plot and its companion summary table."""

    plot_path: Path
    csv_path: Path


def plot_entropy_after_partner_onset(
    aligned_entropy_table: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """Plot mean entropy aligned to partner onset."""

    summary = (
        aligned_entropy_table.groupby("tau_seconds", sort=True)["entropy"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(summary["tau_seconds"], summary["mean"], color="#1f4e79", linewidth=2.0)
    axis.fill_between(
        summary["tau_seconds"],
        summary["mean"] - summary["sem"].fillna(0.0),
        summary["mean"] + summary["sem"].fillna(0.0),
        alpha=0.25,
        color="#1f4e79",
    )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Entropy")
    axis.set_title("Entropy after partner onset")
    figure.tight_layout()

    output_path = output_dir / "entropy_after_partner_onset.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_entropy_histogram(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """Plot the hazard-table entropy distribution."""

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(hazard_table["entropy"].to_numpy(dtype=float), bins=HISTOGRAM_BIN_COUNT, color="#5b8c5a", alpha=0.85)
    axis.set_xlabel("Entropy")
    axis.set_ylabel("Count")
    axis.set_title("Entropy distribution in hazard table")
    figure.tight_layout()

    output_path = output_dir / "entropy_histogram.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_predicted_hazard_by_entropy(
    prediction_table: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """Plot model-predicted hazard over time since partner onset."""

    palette = {
        "low_entropy": "#2a6f97",
        "mean_entropy": "#6c757d",
        "high_entropy": "#c44536",
    }
    labels = {
        "low_entropy": "Entropy = -1 SD",
        "mean_entropy": "Entropy = mean",
        "high_entropy": "Entropy = +1 SD",
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    for scenario, frame in prediction_table.groupby("scenario", sort=False):
        axis.plot(
            frame["tau_seconds"],
            frame["predicted_hazard"],
            label=labels.get(str(scenario), str(scenario)),
            color=palette.get(str(scenario), "#333333"),
            linewidth=2.0,
        )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Predicted hazard")
    axis.set_title("Model-predicted hazard by entropy")
    axis.legend()
    figure.tight_layout()

    output_path = output_dir / "predicted_hazard_by_entropy.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def make_observed_hazard_by_time_bin_artifact(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
    plotting_config: PlottingConfig,
) -> PlotArtifact:
    """Write and plot the empirical hazard by time bin."""

    observed_hazard_table = compute_observed_hazard_by_time_bin(hazard_table)
    csv_path = output_dir / "observed_hazard_by_time_bin.csv"
    observed_hazard_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        observed_hazard_table["tau_seconds"],
        observed_hazard_table["observed_hazard"],
        color="#0f4c5c",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
    )
    sparse_mask = observed_hazard_table["n_at_risk"] < plotting_config.min_at_risk_per_bin
    if sparse_mask.any():
        axis.scatter(
            observed_hazard_table.loc[sparse_mask, "tau_seconds"],
            observed_hazard_table.loc[sparse_mask, "observed_hazard"],
            color="#c44536",
            s=14,
            label=f"n_at_risk < {plotting_config.min_at_risk_per_bin}",
        )
        LOGGER.info(
            "Observed hazard by time bin includes %d sparse bins below the at-risk threshold of %d",
            int(sparse_mask.sum()),
            plotting_config.min_at_risk_per_bin,
        )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Observed hazard")
    axis.set_title("Observed hazard by time since partner onset")
    if sparse_mask.any():
        axis.legend()
    figure.tight_layout()

    plot_path = output_dir / "observed_hazard_by_time_bin.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def make_observed_hazard_by_time_bin_smoothed_artifact(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
    plotting_config: PlottingConfig,
) -> PlotArtifact:
    """Write and plot a lightly smoothed empirical hazard curve."""

    observed_hazard_table = compute_observed_hazard_by_time_bin(hazard_table)
    smoothed_table = compute_smoothed_observed_hazard_by_time_bin(
        observed_hazard_table=observed_hazard_table,
        smoothing_window_bins=plotting_config.smoothing_window_bins,
    )
    csv_path = output_dir / "observed_hazard_by_time_bin_smoothed.csv"
    smoothed_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        smoothed_table["tau_seconds"],
        smoothed_table["observed_hazard"],
        color="#9ca3af",
        linewidth=1.2,
        alpha=0.75,
        label="Raw empirical hazard",
    )
    axis.plot(
        smoothed_table["tau_seconds"],
        smoothed_table["smoothed_observed_hazard"],
        color="#1f4e79",
        linewidth=2.2,
        label="Smoothed empirical hazard",
    )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Observed hazard")
    axis.set_title("Smoothed empirical hazard by time since partner onset")
    axis.legend()
    figure.tight_layout()

    plot_path = output_dir / "observed_hazard_by_time_bin_smoothed.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def make_observed_event_rate_by_entropy_quantile_artifact(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
    plotting_config: PlottingConfig,
) -> PlotArtifact:
    """Write and plot the descriptive event rate by entropy quantile."""

    summary_table = compute_observed_event_rate_by_entropy_quantile(
        hazard_table=hazard_table,
        entropy_quantile_count=plotting_config.entropy_quantile_count,
    )
    csv_path = output_dir / "observed_event_rate_by_entropy_quantile.csv"
    summary_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        summary_table["entropy_quantile_label"],
        summary_table["observed_event_rate"],
        color="#8c6d31",
        alpha=0.85,
    )
    axis.set_xlabel("Entropy quantile")
    axis.set_ylabel("Observed event rate")
    axis.set_title("Observed event rate by entropy quantile")
    axis.text(
        0.5,
        0.98,
        "Descriptive only; not adjusted for time since partner onset",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()

    plot_path = output_dir / "observed_event_rate_by_entropy_quantile.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def make_observed_hazard_by_time_and_entropy_quantile_artifact(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
    plotting_config: PlottingConfig,
) -> PlotArtifact:
    """Write and plot the empirical hazard by time and entropy group."""

    summary_table = compute_observed_hazard_by_time_and_entropy_group(
        hazard_table=hazard_table,
        entropy_group_count=plotting_config.entropy_group_count_for_time_plot,
    )
    csv_path = output_dir / "observed_hazard_by_time_and_entropy_quantile.csv"
    summary_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    palette = {"low entropy": "#2a6f97", "medium entropy": "#6c757d", "high entropy": "#c44536"}
    filtered_table = summary_table.copy()
    sparse_mask = filtered_table["n_at_risk"] < plotting_config.min_at_risk_per_bin
    if sparse_mask.any():
        LOGGER.info(
            "Suppressing %d sparse time-by-entropy cells from plotting because n_at_risk < %d",
            int(sparse_mask.sum()),
            plotting_config.min_at_risk_per_bin,
        )
    filtered_table.loc[sparse_mask, "observed_hazard"] = np.nan

    for entropy_group, frame in filtered_table.groupby("entropy_group", sort=False):
        axis.plot(
            frame["tau_seconds"],
            frame["observed_hazard"],
            linewidth=2.0,
            label=str(entropy_group),
            color=palette.get(str(entropy_group), "#333333"),
        )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Observed hazard")
    axis.set_title("Observed hazard by time since partner onset and entropy group")
    axis.legend(title="Entropy group")
    figure.tight_layout()

    plot_path = output_dir / "observed_hazard_by_time_and_entropy_quantile.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def make_model_vs_observed_hazard_artifact(
    hazard_table: pd.DataFrame,
    prediction_table: pd.DataFrame,
    *,
    output_dir: Path,
    plotting_config: PlottingConfig,
) -> PlotArtifact:
    """Write and plot model-predicted versus observed hazard by time."""

    observed_hazard_table = compute_observed_hazard_by_time_bin(hazard_table)
    mean_prediction = prediction_table.loc[
        prediction_table["scenario"].astype(str) == "mean_entropy",
        ["tau_seconds", "predicted_hazard"],
    ].rename(columns={"predicted_hazard": "predicted_hazard_entropy_mean"})
    model_vs_observed_table = observed_hazard_table.merge(
        mean_prediction,
        on="tau_seconds",
        how="left",
        validate="one_to_one",
    )
    for scenario_name in ("low_entropy", "high_entropy"):
        scenario_prediction = prediction_table.loc[
            prediction_table["scenario"].astype(str) == scenario_name,
            ["tau_seconds", "predicted_hazard"],
        ].rename(columns={"predicted_hazard": f"predicted_hazard_{scenario_name}"})
        model_vs_observed_table = model_vs_observed_table.merge(
            scenario_prediction,
            on="tau_seconds",
            how="left",
            validate="one_to_one",
        )

    csv_path = output_dir / "model_vs_observed_hazard_by_time.csv"
    model_vs_observed_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    sparse_mask = model_vs_observed_table["n_at_risk"] < plotting_config.min_at_risk_per_bin
    observed_for_plot = model_vs_observed_table["observed_hazard"].copy()
    observed_for_plot.loc[sparse_mask] = np.nan
    axis.plot(
        model_vs_observed_table["tau_seconds"],
        observed_for_plot,
        color="#0f4c5c",
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        label="Observed empirical hazard",
    )
    axis.plot(
        model_vs_observed_table["tau_seconds"],
        model_vs_observed_table["predicted_hazard_entropy_mean"],
        color="#c44536",
        linewidth=2.2,
        label="Model-predicted hazard (entropy = mean)",
    )
    axis.set_xlabel("Time since partner onset (s)")
    axis.set_ylabel("Hazard")
    axis.set_title("Model-predicted vs observed hazard")
    axis.legend()
    figure.tight_layout()

    plot_path = output_dir / "model_vs_observed_hazard_by_time.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def make_entropy_distribution_terminal_vs_nonterminal_artifact(
    hazard_table: pd.DataFrame,
    *,
    output_dir: Path,
) -> PlotArtifact:
    """Write and plot entropy distributions for event versus non-event rows."""

    summary_table = compute_entropy_distribution_terminal_vs_nonterminal_summary(hazard_table)
    csv_path = output_dir / "entropy_distribution_terminal_vs_nonterminal_summary.csv"
    summary_table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    for event_value, label, color in (
        (0, "event = 0", "#6c757d"),
        (1, "event = 1", "#c44536"),
    ):
        values = hazard_table.loc[hazard_table["event"] == event_value, "entropy_z"].to_numpy(dtype=float)
        axis.hist(values, bins=HISTOGRAM_BIN_COUNT, alpha=0.55, label=label, color=color, density=True)
    axis.set_xlabel("Entropy z-score")
    axis.set_ylabel("Density")
    axis.set_title("Entropy distribution for event vs non-event rows")
    axis.legend()
    figure.tight_layout()

    plot_path = output_dir / "entropy_distribution_terminal_vs_nonterminal.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return PlotArtifact(plot_path=plot_path, csv_path=csv_path)


def write_plot_interpretation_notes(output_dir: Path) -> Path:
    """Write a short note explaining model-based versus empirical plots."""

    note_text = (
        "predicted_hazard_by_entropy.png is model-based, so it is smooth because the fitted hazard model "
        "imposes a parametric form over time since partner onset.\n"
        "The observed_* plots are empirical summaries of the raw person-period hazard table and therefore "
        "can look jagged, sparse, or noisy.\n"
        "Differences between the model-based and observed curves are expected because the model is a "
        "simplified approximation to the raw data-generating pattern.\n"
    )
    output_path = output_dir / "plot_interpretation_notes.txt"
    output_path.write_text(note_text, encoding="utf-8")
    return output_path


def compute_observed_hazard_by_time_bin(hazard_table: pd.DataFrame) -> pd.DataFrame:
    """Compute the empirical hazard by time bin.

    Parameters
    ----------
    hazard_table
        Person-period hazard table.

    Returns
    -------
    pandas.DataFrame
        Table with one row per time bin containing at-risk counts, event
        counts, and the observed hazard.
    """

    summary_table = (
        hazard_table.groupby("tau_seconds", sort=True)["event"]
        .agg(n_at_risk="size", n_events="sum")
        .reset_index()
    )
    summary_table["observed_hazard"] = summary_table["n_events"] / summary_table["n_at_risk"]
    return summary_table


def compute_smoothed_observed_hazard_by_time_bin(
    *,
    observed_hazard_table: pd.DataFrame,
    smoothing_window_bins: int,
) -> pd.DataFrame:
    """Compute a lightly smoothed empirical hazard curve."""

    smoothed_table = observed_hazard_table.copy()
    smoothed_table["smoothed_observed_hazard"] = (
        smoothed_table["observed_hazard"]
        .rolling(window=smoothing_window_bins, center=True, min_periods=1)
        .mean()
    )
    return smoothed_table.loc[:, ["tau_seconds", "observed_hazard", "smoothed_observed_hazard"]]


def compute_observed_event_rate_by_entropy_quantile(
    *,
    hazard_table: pd.DataFrame,
    entropy_quantile_count: int,
) -> pd.DataFrame:
    """Compute descriptive event rates by entropy quantile."""

    working = hazard_table.copy()
    quantile_count = min(entropy_quantile_count, int(working["entropy_z"].nunique()))
    working["entropy_quantile_interval"] = pd.qcut(
        working["entropy_z"],
        q=quantile_count,
        duplicates="drop",
    )
    interval_categories = list(working["entropy_quantile_interval"].cat.categories)
    interval_to_index = {interval: index + 1 for index, interval in enumerate(interval_categories)}
    working["entropy_quantile"] = working["entropy_quantile_interval"].map(interval_to_index).astype(int)
    summary_table = (
        working.groupby(["entropy_quantile", "entropy_quantile_interval"], observed=True)
        .agg(
            n_rows=("event", "size"),
            n_events=("event", "sum"),
            entropy_min=("entropy_z", "min"),
            entropy_max=("entropy_z", "max"),
            entropy_mean=("entropy_z", "mean"),
        )
        .reset_index()
    )
    summary_table["entropy_quantile_label"] = summary_table["entropy_quantile"].map(
        lambda value: f"Q{value}"
    )
    summary_table["observed_event_rate"] = summary_table["n_events"] / summary_table["n_rows"]
    return summary_table[
        [
            "entropy_quantile",
            "entropy_quantile_label",
            "n_rows",
            "n_events",
            "observed_event_rate",
            "entropy_min",
            "entropy_max",
            "entropy_mean",
        ]
    ]


def compute_observed_hazard_by_time_and_entropy_group(
    *,
    hazard_table: pd.DataFrame,
    entropy_group_count: int,
) -> pd.DataFrame:
    """Compute empirical hazard by time and entropy group."""

    working = hazard_table.copy()
    group_count = min(entropy_group_count, int(working["entropy_z"].nunique()))
    working["entropy_group_interval"] = pd.qcut(
        working["entropy_z"],
        q=group_count,
        duplicates="drop",
    )
    interval_categories = list(working["entropy_group_interval"].cat.categories)
    if len(interval_categories) == 3:
        group_labels = ["low entropy", "medium entropy", "high entropy"]
    else:
        group_labels = [f"entropy group {index + 1}" for index in range(len(interval_categories))]
    interval_to_label = dict(zip(interval_categories, group_labels, strict=True))
    working["entropy_group"] = working["entropy_group_interval"].map(interval_to_label).astype(str)
    summary_table = (
        working.groupby(["tau_seconds", "entropy_group"], observed=True)["event"]
        .agg(n_at_risk="size", n_events="sum")
        .reset_index()
    )
    summary_table["observed_hazard"] = summary_table["n_events"] / summary_table["n_at_risk"]
    summary_table["entropy_group"] = summary_table["entropy_group"].astype(str)
    return summary_table[["tau_seconds", "entropy_group", "n_at_risk", "n_events", "observed_hazard"]]


def compute_entropy_distribution_terminal_vs_nonterminal_summary(
    hazard_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize entropy distributions for event versus non-event rows."""

    rows: list[dict[str, float | int | str]] = []
    for event_value, group_label in ((1, "event"), (0, "non_event")):
        values = hazard_table.loc[hazard_table["event"] == event_value, "entropy_z"].to_numpy(dtype=float)
        rows.append(
            {
                "group": group_label,
                "n_rows": int(values.size),
                "entropy_mean": float(np.mean(values)),
                "entropy_std": float(np.std(values, ddof=0)),
                "entropy_median": float(np.median(values)),
                "entropy_q25": float(np.quantile(values, 0.25)),
                "entropy_q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)
