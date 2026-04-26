"""Minimal plotting helpers for the active behavioural hazard pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_BIN_COUNT = 20
ACTIVE_PREDICTOR_FAMILIES = ("information_rate", "prop_expected_cumulative_info")
LOGGER = logging.getLogger(__name__)


def summarize_observed_event_rate_by_time_bin(
    riskset_table: pd.DataFrame,
    *,
    n_bins: int = TIME_BIN_COUNT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"time_from_partner_onset", "event"}
    missing = required - set(riskset_table.columns)
    if missing:
        raise ValueError("Observed event-rate summary requires columns: " + ", ".join(sorted(missing)))

    working = riskset_table.loc[:, ["time_from_partner_onset", "event"]].copy()
    working["time_from_partner_onset"] = pd.to_numeric(working["time_from_partner_onset"], errors="coerce")
    working["event"] = pd.to_numeric(working["event"], errors="coerce")
    initial_events = int(working["event"].fillna(0).astype(int).sum())
    mask = np.isfinite(working["time_from_partner_onset"]) & np.isfinite(working["event"])
    filtered = working.loc[mask].copy()
    filtered["event"] = filtered["event"].astype(int)

    qc = {
        "n_rows_input": int(len(riskset_table)),
        "n_events_input": initial_events,
        "n_rows_after_required_column_filter": int(len(filtered)),
        "n_events_after_required_column_filter": int(filtered["event"].sum()) if not filtered.empty else 0,
        "warning_flags": [],
    }
    if filtered.empty:
        qc["warning_flags"].append("empty_after_required_column_filter")
        empty = pd.DataFrame(columns=["time_bin", "time_midpoint", "n_rows", "n_events", "event_rate"])
        qc["sum_binned_events"] = 0
        return empty, qc

    min_time = float(filtered["time_from_partner_onset"].min())
    max_time = float(filtered["time_from_partner_onset"].max())
    if np.isclose(min_time, max_time):
        edges = np.array([min_time, max_time + 1.0e-6], dtype=float)
    else:
        edges = np.linspace(min_time, max_time, num=max(2, n_bins + 1))
        edges[-1] = max_time + 1.0e-9
    filtered["time_bin"] = pd.cut(
        filtered["time_from_partner_onset"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=False,
    )
    summary = (
        filtered.groupby("time_bin", observed=True)
        .agg(
            n_rows=("event", "size"),
            n_events=("event", "sum"),
            time_min=("time_from_partner_onset", "min"),
            time_max=("time_from_partner_onset", "max"),
        )
        .reset_index()
    )
    summary["time_midpoint"] = (summary["time_min"] + summary["time_max"]) / 2.0
    summary["event_rate"] = summary["n_events"] / summary["n_rows"]
    summary = summary.loc[:, ["time_bin", "time_midpoint", "n_rows", "n_events", "event_rate"]]
    qc["sum_binned_events"] = int(summary["n_events"].sum())
    if initial_events == 0:
        qc["warning_flags"].append("zero_event_rows_input")
    return summary, qc


def plot_observed_event_rate_by_time_bin(
    summary_table: pd.DataFrame,
    qc_payload: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    if summary_table.empty:
        _draw_placeholder(axis, "Observed event rate by time", "No finite rows were available.")
    else:
        axis.bar(summary_table["time_midpoint"], summary_table["event_rate"], width=0.08, color="#5b8e7d")
        axis.set_xlabel("Time from partner onset (s)")
        axis.set_ylabel("Observed event rate")
        axis.set_title("Observed event rate by time")
    if qc_payload.get("warning_flags"):
        axis.text(0.98, 0.98, "\n".join(qc_payload["warning_flags"]), ha="right", va="top", transform=axis.transAxes)
    _save_figure(figure, output_path)


def plot_behaviour_delta_bic_by_lag(selection_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    working = filter_active_lag_selection_rows(selection_table)
    if working.empty:
        _draw_placeholder(axis, "Pooled behavioural lag screening", "No lag-selection rows were available.")
        _save_figure(figure, output_path)
        return

    working["lag_ms"] = pd.to_numeric(working["lag_ms"], errors="coerce")
    working["delta_bic"] = pd.to_numeric(working["delta_bic"], errors="coerce")
    working = working.loc[
        np.isfinite(working["lag_ms"])
        & np.isfinite(working["delta_bic"])
    ].copy()
    if working.empty:
        _draw_placeholder(axis, "Pooled behavioural lag screening", "No active lag-selection rows were available.")
        _save_figure(figure, output_path)
        return

    color_map = {
        "information_rate": "#124559",
        "prop_expected_cumulative_info": "#5b8e7d",
    }
    label_map = {
        "information_rate": "Information rate",
        "prop_expected_cumulative_info": "Expected cumulative info",
    }
    for predictor_family in ACTIVE_PREDICTOR_FAMILIES:
        subset = working.loc[working["predictor_family"] == predictor_family].sort_values("lag_ms")
        if subset.empty:
            continue
        axis.plot(
            subset["lag_ms"],
            subset["delta_bic"],
            marker="o",
            linewidth=2.0,
            color=color_map[predictor_family],
            label=label_map[predictor_family],
        )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Lag (ms)")
    axis.set_ylabel("Delta BIC (child - parent)")
    axis.set_title("Pooled lag screening only - final inference uses R GLMM")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_fpp_latency_diagnostics(episode_summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    onset = pd.to_numeric(episode_summary.get("event_latency_from_partner_onset_s"), errors="coerce")
    offset = pd.to_numeric(episode_summary.get("event_latency_from_partner_offset_s"), errors="coerce")
    for axis, values, title in zip(
        axes,
        [onset, offset],
        ["Latency from onset", "Latency from offset"],
        strict=True,
    ):
        finite = values[np.isfinite(values)]
        if finite.empty:
            _draw_placeholder(axis, title, "No finite values were available.")
            continue
        axis.hist(finite, bins=20, color="#5b8e7d", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel("Seconds")
        axis.set_ylabel("Count")
    _save_figure(figure, output_path)


def filter_active_lag_selection_rows(selection_table: pd.DataFrame) -> pd.DataFrame:
    """Retain only lag-selection rows for the active behavioural predictor families."""

    return selection_table.loc[selection_table["predictor_family"].isin(ACTIVE_PREDICTOR_FAMILIES)].copy()


def compute_binned_median_iqr(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    n_bins: int,
    *,
    min_rows_per_bin: int = 5,
    trim_quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute binned median and IQR summaries for a time-aligned diagnostic plot."""

    required = {x_column, y_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Binned summary requires columns: " + ", ".join(sorted(missing)))

    working = data.loc[:, [x_column, y_column]].copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
    working = working.loc[np.isfinite(working[x_column]) & np.isfinite(working[y_column])].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "bin_start",
                "bin_end",
                "bin_center",
                "median_information_rate",
                "q25_information_rate",
                "q75_information_rate",
                "n_rows",
            ]
        )

    if trim_quantiles is not None:
        lower_q, upper_q = trim_quantiles
        lower_bound = float(working[x_column].quantile(lower_q))
        upper_bound = float(working[x_column].quantile(upper_q))
        working = working.loc[
            (working[x_column] >= lower_bound) & (working[x_column] <= upper_bound)
        ].copy()
        if working.empty:
            return pd.DataFrame(
                columns=[
                    "bin_start",
                    "bin_end",
                    "bin_center",
                    "median_information_rate",
                    "q25_information_rate",
                    "q75_information_rate",
                    "n_rows",
                ]
            )

    min_x = float(working[x_column].min())
    max_x = float(working[x_column].max())
    if np.isclose(min_x, max_x):
        max_x = min_x + 1.0e-6
    edges = np.linspace(min_x, max_x, num=max(2, int(n_bins) + 1))
    edges[-1] = max_x + 1.0e-9
    working["bin_index"] = pd.cut(
        working[x_column],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=False,
    )
    summary = (
        working.groupby("bin_index", observed=True)
        .agg(
            bin_start=(x_column, "min"),
            bin_end=(x_column, "max"),
            median_information_rate=(y_column, "median"),
            q25_information_rate=(y_column, lambda values: float(np.nanpercentile(values, 25))),
            q75_information_rate=(y_column, lambda values: float(np.nanpercentile(values, 75))),
            n_rows=(y_column, "size"),
        )
        .reset_index(drop=True)
    )
    summary = summary.loc[summary["n_rows"] >= int(min_rows_per_bin)].copy()
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "bin_start",
                "bin_end",
                "bin_center",
                "median_information_rate",
                "q25_information_rate",
                "q75_information_rate",
                "n_rows",
            ]
        )
    summary["bin_center"] = (summary["bin_start"] + summary["bin_end"]) / 2.0
    return summary.loc[
        :,
        [
            "bin_start",
            "bin_end",
            "bin_center",
            "median_information_rate",
            "q25_information_rate",
            "q75_information_rate",
            "n_rows",
        ],
    ].reset_index(drop=True)


def plot_information_rate_by_partner_time(
    riskset: pd.DataFrame,
    figures_dir: Path,
    information_rate_column: str = "information_rate_lag_150ms",
    n_bins: int = 40,
) -> Path | None:
    """Plot information-rate timing relative to partner onset and offset, and save the binned CSV."""

    required_time_columns = {"time_from_partner_onset", "time_from_partner_offset"}
    missing_time = required_time_columns - set(riskset.columns)
    if missing_time:
        message = "Information-rate timing plot requires columns: " + ", ".join(sorted(missing_time))
        warnings.warn(message, UserWarning, stacklevel=2)
        LOGGER.warning(message)
        return None

    y_column = information_rate_column
    y_label = "Information rate at t - 150 ms"
    if y_column not in riskset.columns:
        fallback_column = "z_information_rate_lag_150ms"
        if fallback_column not in riskset.columns:
            message = (
                "Information-rate timing plot requires `information_rate_lag_150ms` or "
                "`z_information_rate_lag_150ms`."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            LOGGER.warning(message)
            return None
        y_column = fallback_column
        y_label = "Information rate at t - 150 ms (z-scored)"
        message = "Falling back to z-scored information rate for timing diagnostic plot."
        warnings.warn(message, UserWarning, stacklevel=2)
        LOGGER.warning(message)

    onset_summary = compute_binned_median_iqr(
        riskset,
        "time_from_partner_onset",
        y_column,
        n_bins,
        min_rows_per_bin=5,
        trim_quantiles=None,
    )
    offset_summary = compute_binned_median_iqr(
        riskset,
        "time_from_partner_offset",
        y_column,
        n_bins,
        min_rows_per_bin=5,
        trim_quantiles=(0.01, 0.99),
    )
    if onset_summary.empty or offset_summary.empty:
        message = "Information-rate timing plot had too few finite rows to generate both panels."
        warnings.warn(message, UserWarning, stacklevel=2)
        LOGGER.warning(message)
        return None

    output_path = figures_dir / "information_rate_by_partner_time.png"
    csv_path = figures_dir / "information_rate_by_partner_time.csv"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    _plot_information_rate_summary_panel(
        axes[0],
        onset_summary,
        title="A. Aligned to partner IPU onset",
        x_label="Time from partner IPU onset (s)",
        y_label=y_label,
    )
    partner_duration = pd.to_numeric(riskset.get("partner_ipu_duration"), errors="coerce")
    if partner_duration is not None:
        finite_duration = partner_duration[np.isfinite(partner_duration)]
        if not finite_duration.empty:
            axes[0].axvline(float(np.median(finite_duration)), color="#666666", linestyle=":", linewidth=1.0)

    _plot_information_rate_summary_panel(
        axes[1],
        offset_summary,
        title="B. Aligned to partner IPU offset",
        x_label="Time from partner IPU offset (s)",
        y_label=y_label,
    )
    axes[1].axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    figure.suptitle("Information rate timing relative to partner IPU")
    _save_figure(figure, output_path)

    csv_table = pd.concat(
        [
            onset_summary.assign(alignment="partner_onset"),
            offset_summary.assign(alignment="partner_offset"),
        ],
        ignore_index=True,
        sort=False,
    )
    csv_table = csv_table.loc[
        :,
        [
            "alignment",
            "bin_start",
            "bin_end",
            "bin_center",
            "median_information_rate",
            "q25_information_rate",
            "q75_information_rate",
            "n_rows",
        ],
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_table.to_csv(csv_path, index=False)
    return output_path


def _plot_information_rate_summary_panel(
    axis: plt.Axes,
    summary_table: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    axis.plot(
        summary_table["bin_center"],
        summary_table["median_information_rate"],
        color="#124559",
        linewidth=2.0,
    )
    axis.fill_between(
        summary_table["bin_center"],
        summary_table["q25_information_rate"],
        summary_table["q75_information_rate"],
        color="#aec3b0",
        alpha=0.35,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


def _draw_placeholder(axis: plt.Axes, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
