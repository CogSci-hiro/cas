"""Plotting helpers for behavioural hazard analysis."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SATURATION_TOLERANCE = 1.0e-8
MIN_NON_SATURATED_ROWS_FOR_PLOT = 50
TIME_BIN_COUNT = 50
FIGURE_TIME_LIMIT_S = 3.0
OBSERVED_EVENT_RATE_REQUIRED_COLUMNS = ("time_from_partner_onset", "event")
OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_REQUIRED_FILTER = "dropped_all_event_rows_required_column_filter"
OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_BINNING = "dropped_all_event_rows_binning"
OBSERVED_EVENT_RATE_WARNING_ZERO_EVENT_ROWS_INPUT = "zero_event_rows_input"


def plot_prediction_curve(prediction_table: pd.DataFrame, *, x_column: str, title: str, output_path: Path) -> None:
    """Plot a predicted hazard curve."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(prediction_table[x_column], prediction_table["predicted_hazard"], linewidth=2.0)
    axis.set_xlabel(x_column)
    axis.set_ylabel("Predicted hazard")
    axis.set_title(title)
    axis.set_ylim(0.0, max(0.05, float(prediction_table["predicted_hazard"].max()) * 1.1))
    if x_column == "time_from_partner_onset":
        _apply_time_axis_limit(axis)
    _save_figure(figure, output_path)


def plot_primary_coefficients(summary_table: pd.DataFrame, output_path: Path) -> None:
    """Plot compact primary-model coefficient estimates with confidence intervals."""

    figure, axis = plt.subplots(figsize=(8, 4.8))
    term_series = summary_table["term"].astype(str)
    if "model_name" in summary_table.columns:
        model_mask = summary_table["model_name"].astype(str) == "M2_rate_prop_expected"
    else:
        model_mask = pd.Series(True, index=summary_table.index)
    working = summary_table.loc[
        model_mask
        & (
            term_series.str.startswith("z_information_rate_lag_")
            | term_series.str.startswith("z_prop_expected_cumulative_info_lag_")
        )
    ].copy()
    if working.empty:
        _draw_placeholder_panel(axis, title="Primary behavioural coefficients", message="No primary coefficient rows available.")
        _save_figure(figure, output_path)
        return

    working["estimate"] = pd.to_numeric(working["estimate"], errors="coerce")
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    if "odds_ratio" in working.columns:
        working["odds_ratio"] = pd.to_numeric(working["odds_ratio"], errors="coerce")
    working = working.drop_duplicates(subset=["term"], keep="last")
    working["label"] = working["term"].map(_format_primary_coefficient_label)
    working = working.loc[working["label"].notna()].reset_index(drop=True)
    if working.empty:
        _draw_placeholder_panel(axis, title="Primary behavioural coefficients", message="No matching primary coefficient rows available.")
        _save_figure(figure, output_path)
        return

    y_positions = np.arange(len(working))
    xerr = np.vstack(
        [
            working["estimate"].to_numpy(dtype=float) - working["conf_low"].to_numpy(dtype=float),
            working["conf_high"].to_numpy(dtype=float) - working["estimate"].to_numpy(dtype=float),
        ]
    )
    axis.errorbar(
        working["estimate"],
        y_positions,
        xerr=xerr,
        fmt="o",
        color="#1f4e79",
        ecolor="#7aa6d1",
        elinewidth=2.0,
        capsize=4,
    )
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(working["label"])
    axis.set_xlabel("Coefficient estimate, beta")
    axis.set_title("Primary behavioural hazard coefficients")
    if "odds_ratio" in working.columns:
        x_max = np.nanmax(np.vstack([working["estimate"], working["conf_high"]]))
        text_x = float(x_max) + max(0.02, abs(float(x_max)) * 0.05)
        for y_position, odds_ratio in zip(y_positions, working["odds_ratio"], strict=True):
            if np.isfinite(odds_ratio):
                axis.text(text_x, y_position, f"OR={odds_ratio:.2f}", va="center", fontsize=9)
    _save_figure(figure, output_path)


def plot_primary_model_comparison(comparison_table: pd.DataFrame, output_path: Path) -> None:
    """Plot delta AIC for the compact primary-model comparisons."""

    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    if comparison_table.empty:
        _draw_placeholder_panel(axis, title="Primary model comparison", message="No primary model comparisons available.")
        _save_figure(figure, output_path)
        return

    working = comparison_table.copy()
    working["delta_aic"] = pd.to_numeric(working["delta_aic"], errors="coerce")
    working = working.loc[np.isfinite(working["delta_aic"])].reset_index(drop=True)
    if working.empty:
        _draw_placeholder_panel(axis, title="Primary model comparison", message="No finite delta-AIC values available.")
        _save_figure(figure, output_path)
        return

    colors = ["#2a9d8f" if value < 0.0 else "#e76f51" for value in working["delta_aic"].to_numpy(dtype=float)]
    axis.bar(working["comparison"], working["delta_aic"], color=colors)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Delta AIC (child - parent)")
    axis.set_title("Primary behavioural model comparisons\nNegative Delta AIC favours the child model")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, output_path)


def plot_primary_prediction_curve(
    prediction_table: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot a publication-style primary prediction curve."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    working = prediction_table.copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working["predicted_hazard"] = pd.to_numeric(working["predicted_hazard"], errors="coerce")
    if "conf_low" not in working.columns:
        working["conf_low"] = np.nan
    if "conf_high" not in working.columns:
        working["conf_high"] = np.nan
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    working = working.loc[np.isfinite(working[x_column]) & np.isfinite(working["predicted_hazard"])].copy()
    if working.empty:
        _draw_placeholder_panel(axis, title=title, message="No finite prediction rows available.")
        _save_figure(figure, output_path)
        return
    axis.plot(working[x_column], working["predicted_hazard"], color="#1f4e79", linewidth=2.2)
    if np.isfinite(working["conf_low"]).any() and np.isfinite(working["conf_high"]).any():
        axis.fill_between(
            working[x_column],
            working["conf_low"],
            working["conf_high"],
            color="#9ec5e6",
            alpha=0.35,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Predicted hazard")
    axis.set_title(title)
    _save_figure(figure, output_path)


def plot_primary_lag_sensitivity(lag_sensitivity_table: pd.DataFrame, output_path: Path) -> None:
    """Plot the compact lag-sensitivity summary for retained predictors."""

    figure, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    if lag_sensitivity_table.empty:
        for axis in axes:
            _draw_placeholder_panel(axis, title="Primary lag sensitivity", message="No lag-sensitivity rows available.")
        _save_figure(figure, output_path)
        return
    color_map = {"information_rate": "#1f77b4", "prop_expected": "#ff7f0e"}
    label_map = {"information_rate": "Information rate", "prop_expected": "Prop. expected"}
    working = lag_sensitivity_table.copy()
    working["lag_ms"] = pd.to_numeric(working["lag_ms"], errors="coerce")
    working["estimate"] = pd.to_numeric(working["estimate"], errors="coerce")
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    working["delta_aic"] = pd.to_numeric(working["delta_aic"], errors="coerce")
    for family_name, family_rows in working.groupby("family", sort=False):
        family_rows = family_rows.sort_values("lag_ms")
        axes[0].plot(
            family_rows["lag_ms"],
            family_rows["estimate"],
            marker="o",
            linewidth=2.0,
            color=color_map.get(str(family_name), None),
            label=label_map.get(str(family_name), str(family_name)),
        )
        if np.isfinite(family_rows["conf_low"]).any() and np.isfinite(family_rows["conf_high"]).any():
            axes[0].fill_between(
                family_rows["lag_ms"],
                family_rows["conf_low"],
                family_rows["conf_high"],
                color=color_map.get(str(family_name), "#bbbbbb"),
                alpha=0.18,
            )
        axes[1].plot(
            family_rows["lag_ms"],
            family_rows["delta_aic"],
            marker="o",
            linewidth=2.0,
            color=color_map.get(str(family_name), None),
            label=label_map.get(str(family_name), str(family_name)),
        )
    for axis in axes:
        axis.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        axis.axvline(300.0, color="#999999", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Coefficient estimate")
    axes[0].set_title("Primary behavioural lag sensitivity")
    axes[1].set_ylabel("Delta AIC")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_title("Delta AIC = child AIC - parent AIC; negative favours child model")
    axes[0].legend(frameon=False)
    _save_figure(figure, output_path)


def plot_primary_leave_one_cluster(
    leave_one_cluster_table: pd.DataFrame,
    *,
    full_beta: float,
    output_path: Path,
) -> None:
    """Plot leave-one-cluster primary-beta sensitivity."""

    figure, axis = plt.subplots(figsize=(8, 4.8))
    working = leave_one_cluster_table.copy()
    working["beta_prop_expected"] = pd.to_numeric(working["beta_prop_expected"], errors="coerce")
    working = working.loc[np.isfinite(working["beta_prop_expected"])].reset_index(drop=True)
    if working.empty:
        _draw_placeholder_panel(axis, title="Leave-one-cluster sensitivity", message="No successful refits available.")
        _save_figure(figure, output_path)
        return
    axis.plot(working["omitted_cluster"].astype(str), working["beta_prop_expected"], marker="o", linestyle="none")
    axis.axhline(float(full_beta), color="#1f4e79", linestyle="--", linewidth=1.5)
    axis.set_ylabel("Beta for prop_expected")
    axis.set_title("Leave-one-cluster sensitivity")
    axis.tick_params(axis="x", rotation=45)
    _save_figure(figure, output_path)


def summarize_observed_event_rate_by_time_bin(
    riskset_table: pd.DataFrame,
    *,
    n_bins: int = TIME_BIN_COUNT,
    time_limit_s: float = FIGURE_TIME_LIMIT_S,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize observed event rate by time bin and emit auditable QC metadata."""

    event_column = _require_column(riskset_table, "event")
    time_column = _require_column(riskset_table, "time_from_partner_onset")
    event_numeric = _coerce_event_series(event_column)
    working = pd.DataFrame(
        {
            "time_from_partner_onset": pd.to_numeric(time_column, errors="coerce"),
            "event_numeric": event_numeric,
        },
        index=riskset_table.index,
    )
    if "bin_index" in riskset_table.columns:
        working["bin_index"] = pd.to_numeric(riskset_table["bin_index"], errors="coerce")

    qc: dict[str, Any] = {
        "n_rows_input": int(len(riskset_table)),
        "n_events_input": int(event_numeric.fillna(0).sum()),
        "event_dtype_input": str(event_column.dtype),
        "event_value_counts_input": _series_value_counts_json(event_column),
        "n_rows_after_required_column_filter": 0,
        "n_events_after_required_column_filter": 0,
        "required_columns_used": list(OBSERVED_EVENT_RATE_REQUIRED_COLUMNS),
        "n_rows_after_time_filter": 0,
        "n_events_after_time_filter": 0,
        "n_rows_after_binning": 0,
        "n_events_after_binning": 0,
        "n_time_bins": 0,
        "min_time_from_partner_onset": None,
        "max_time_from_partner_onset": None,
        "max_event_rate": None,
        "sum_binned_events": 0,
        "warning_flags": [],
    }

    required_mask = working["time_from_partner_onset"].notna() & working["event_numeric"].notna()
    required = working.loc[required_mask].copy()
    qc["n_rows_after_required_column_filter"] = int(len(required))
    qc["n_events_after_required_column_filter"] = int(required["event_numeric"].sum())

    filtered = required.loc[
        np.isfinite(required["time_from_partner_onset"])
        & (required["time_from_partner_onset"] >= 0.0)
        & (required["time_from_partner_onset"] <= time_limit_s)
    ].copy()
    qc["n_rows_after_time_filter"] = int(len(filtered))
    qc["n_events_after_time_filter"] = int(filtered["event_numeric"].sum())
    if not filtered.empty:
        qc["min_time_from_partner_onset"] = float(filtered["time_from_partner_onset"].min())
        qc["max_time_from_partner_onset"] = float(filtered["time_from_partner_onset"].max())

    warning_flags = _observed_event_rate_warning_flags(qc)
    qc["warning_flags"] = warning_flags

    if filtered.empty:
        return pd.DataFrame(columns=["time_bin_start", "time_bin_center", "n_rows", "n_events", "event_rate"]), qc

    if "bin_index" in filtered.columns and filtered["bin_index"].notna().any():
        binned = filtered.loc[filtered["bin_index"].notna()].copy()
        summary = (
            binned.groupby("bin_index", as_index=False)
            .agg(
                time_bin_start=("time_from_partner_onset", "min"),
                time_bin_center=("time_from_partner_onset", "mean"),
                n_rows=("event_numeric", "size"),
                n_events=("event_numeric", "sum"),
            )
            .sort_values("bin_index", kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        binned = filtered.copy()
        summary = _bin_observed_event_rate_without_bin_index(binned, n_bins=n_bins)

    if summary.empty:
        qc["n_rows_after_binning"] = 0
        qc["n_events_after_binning"] = 0
        qc["n_time_bins"] = 0
        qc["max_event_rate"] = None
        qc["sum_binned_events"] = 0
        if qc["n_events_after_required_column_filter"] > 0:
            qc["warning_flags"] = _with_unique_flag(
                qc["warning_flags"],
                OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_BINNING,
            )
        return summary, qc

    summary["n_rows"] = summary["n_rows"].astype(int)
    summary["n_events"] = summary["n_events"].astype(int)
    summary["event_rate"] = summary["n_events"] / summary["n_rows"]
    qc["n_rows_after_binning"] = int(summary["n_rows"].sum())
    qc["n_events_after_binning"] = int(summary["n_events"].sum())
    qc["n_time_bins"] = int(len(summary))
    qc["max_event_rate"] = float(summary["event_rate"].max()) if np.isfinite(summary["event_rate"]).any() else None
    qc["sum_binned_events"] = int(summary["n_events"].sum())
    if qc["n_events_after_required_column_filter"] > 0 and qc["n_events_after_binning"] == 0:
        qc["warning_flags"] = _with_unique_flag(
            qc["warning_flags"],
            OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_BINNING,
        )
    return summary, qc


def plot_observed_event_rate_by_time_bin(summary: pd.DataFrame, qc_summary: dict[str, Any], output_path: Path) -> None:
    """Plot observed event rate by time bin from a precomputed summary."""

    figure, axis = plt.subplots(figsize=(6, 4))
    if summary.empty:
        _draw_placeholder_panel(
            axis,
            title="Observed event rate by time bin",
            message="No rows available after filtering and binning.",
        )
    else:
        axis.plot(summary["time_bin_center"], summary["event_rate"], marker="o")
        max_event_rate = qc_summary.get("max_event_rate")
        title_suffix = ""
        if isinstance(max_event_rate, float) and np.isfinite(max_event_rate):
            if max_event_rate == 0.0:
                title_suffix = "\nobserved event rate is zero in all bins"
            else:
                title_suffix = f"\nmax event rate = {max_event_rate:.6g}"
            if 0.0 < max_event_rate < 0.01:
                axis.set_ylim(0.0, max_event_rate * 1.2)
    axis.set_xlabel("Time from partner onset (s)")
    axis.set_ylabel("Observed event rate")
    axis.set_title(f"Observed event rate by time bin{title_suffix}")
    _apply_time_axis_limit(axis)
    _save_figure(figure, output_path)


def plot_risk_set_qc(riskset_table: pd.DataFrame, output_path: Path) -> None:
    """Plot simple risk-set QC summaries."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    episode_lengths = riskset_table.groupby("episode_id")["bin_index"].max() + 1
    axes[0].hist(episode_lengths, bins=min(20, max(5, len(episode_lengths))))
    axes[0].set_title("Bins per episode")
    axes[0].set_xlabel("Bin count")
    axes[0].set_ylabel("Episodes")
    axes[1].hist(riskset_table["alignment_ok_fraction"].fillna(0.0), bins=10)
    axes[1].set_title("Alignment-ok fraction")
    axes[1].set_xlabel("Fraction")
    axes[1].set_ylabel("Rows")
    _save_figure(figure, output_path)


def plot_information_feature_distributions(
    riskset_table: pd.DataFrame,
    output_path: Path,
    *,
    tolerance: float = SATURATION_TOLERANCE,
) -> dict[str, int]:
    """Plot distributions of key information features.

    The ``prop_actual_cumulative_info`` panel excludes saturated values equal to
    1.0 within the provided tolerance, but the input table is not modified.
    """

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    features = [
        "information_rate",
        "cumulative_info",
        "prop_actual_cumulative_info",
        "prop_expected_cumulative_info",
    ]
    non_saturated_count = 0
    for axis, feature in zip(axes.flat, features, strict=True):
        values = pd.to_numeric(riskset_table[feature], errors="coerce")
        finite_values = values[np.isfinite(values)]
        if feature == "prop_actual_cumulative_info":
            non_saturated_values = finite_values.loc[~is_saturated_prop_actual(finite_values, tolerance=tolerance)]
            non_saturated_count = int(non_saturated_values.shape[0])
            if non_saturated_count >= MIN_NON_SATURATED_ROWS_FOR_PLOT:
                axis.hist(non_saturated_values.to_numpy(dtype=float), bins=20)
                axis.set_title("prop_actual_cumulative_info, excluding saturated values == 1")
                axis.set_xlabel("Value")
                axis.set_ylabel("Rows")
            else:
                _draw_placeholder_panel(
                    axis,
                    title="prop_actual_cumulative_info, excluding saturated values == 1",
                    message=(
                        "Too few non-saturated values to plot.\n"
                        f"Non-saturated rows: {non_saturated_count}"
                    ),
                )
        else:
            axis.hist(finite_values.to_numpy(dtype=float), bins=20)
            axis.set_title(feature)
    _save_figure(figure, output_path)
    return {"n_non_saturated_prop_actual_rows": non_saturated_count}


def compute_prop_actual_saturation_qc(
    riskset_table: pd.DataFrame,
    *,
    tolerance: float = SATURATION_TOLERANCE,
) -> dict[str, float | int | bool | None]:
    """Compute saturation QC metrics for ``prop_actual_cumulative_info``."""

    prop_actual = pd.to_numeric(riskset_table["prop_actual_cumulative_info"], errors="coerce")
    finite_mask = np.isfinite(prop_actual)
    saturated_mask = finite_mask & is_saturated_prop_actual(prop_actual, tolerance=tolerance)
    event_mask = pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int) == 1
    nonevent_mask = ~event_mask

    qc: dict[str, float | int | bool | None] = {
        "n_rows_total": int(len(riskset_table)),
        "n_rows_with_finite_prop_actual": int(finite_mask.sum()),
        "n_saturated_prop_actual_rows": int(saturated_mask.sum()),
        "proportion_saturated_prop_actual_rows": _safe_proportion(int(saturated_mask.sum()), int(finite_mask.sum())),
        "n_event_rows": int(event_mask.sum()),
        "n_event_rows_with_finite_prop_actual": int((event_mask & finite_mask).sum()),
        "n_saturated_event_rows": int((event_mask & saturated_mask).sum()),
        "proportion_saturated_event_rows": _safe_proportion(
            int((event_mask & saturated_mask).sum()),
            int((event_mask & finite_mask).sum()),
        ),
        "n_nonevent_rows": int(nonevent_mask.sum()),
        "n_nonevent_rows_with_finite_prop_actual": int((nonevent_mask & finite_mask).sum()),
        "n_saturated_nonevent_rows": int((nonevent_mask & saturated_mask).sum()),
        "proportion_saturated_nonevent_rows": _safe_proportion(
            int((nonevent_mask & saturated_mask).sum()),
            int((nonevent_mask & finite_mask).sum()),
        ),
        "tolerance_used": float(tolerance),
    }

    phase_qc = compute_partner_offset_phase_qc(riskset_table)
    qc.update(phase_qc)
    return qc


def compute_partner_offset_phase_qc(riskset_table: pd.DataFrame) -> dict[str, float | int | bool | None]:
    """Compute optional partner-offset diagnostics when the required columns exist."""

    required_columns = {"partner_ipu_offset", "own_fpp_onset", "event"}
    if not required_columns.issubset(riskset_table.columns):
        return {"phase_qc_available": False}

    partner_offset = pd.to_numeric(riskset_table["partner_ipu_offset"], errors="coerce")
    own_fpp_onset = pd.to_numeric(riskset_table["own_fpp_onset"], errors="coerce")
    event_mask = pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int) == 1
    finite_phase_mask = np.isfinite(partner_offset)
    if "bin_end" in riskset_table.columns:
        current_time = pd.to_numeric(riskset_table["bin_end"], errors="coerce")
        after_partner_offset_mask = finite_phase_mask & np.isfinite(current_time) & (current_time > partner_offset)
    elif "bin_start" in riskset_table.columns:
        current_time = pd.to_numeric(riskset_table["bin_start"], errors="coerce")
        after_partner_offset_mask = finite_phase_mask & np.isfinite(current_time) & (current_time > partner_offset)
    else:
        return {"phase_qc_available": False}

    latency_mask = event_mask & np.isfinite(partner_offset) & np.isfinite(own_fpp_onset)
    latencies = (own_fpp_onset - partner_offset).loc[latency_mask]
    return {
        "phase_qc_available": True,
        "n_rows_after_partner_ipu_offset": int(after_partner_offset_mask.sum()),
        "proportion_rows_after_partner_ipu_offset": _safe_proportion(
            int(after_partner_offset_mask.sum()),
            int(len(riskset_table)),
        ),
        "n_event_rows_after_partner_ipu_offset": int((after_partner_offset_mask & event_mask).sum()),
        "proportion_event_rows_after_partner_ipu_offset": _safe_proportion(
            int((after_partner_offset_mask & event_mask).sum()),
            int(event_mask.sum()),
        ),
        "median_event_latency_from_partner_ipu_offset_s": (
            float(latencies.median()) if not latencies.empty else None
        ),
        "mean_event_latency_from_partner_ipu_offset_s": (
            float(latencies.mean()) if not latencies.empty else None
        ),
    }


def plot_prop_actual_by_time_from_partner_onset(
    riskset_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot median and IQR of ``prop_actual_cumulative_info`` over time."""

    summary = summarize_prop_actual_by_time(riskset_table)
    figure, axis = plt.subplots(figsize=(7, 4))
    if summary.empty:
        _draw_placeholder_panel(axis, title="prop_actual by time from partner onset", message="No finite values available.")
    else:
        axis.plot(summary["time_bin_center"], summary["median"], linewidth=2.0, color="#1f77b4")
        axis.fill_between(
            summary["time_bin_center"],
            summary["q25"],
            summary["q75"],
            color="#1f77b4",
            alpha=0.2,
        )
        axis.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5)
        axis.set_xlabel("Time from partner onset (s)")
        axis.set_ylabel("prop_actual_cumulative_info")
        axis.set_title("prop_actual by time from partner onset")
        _apply_time_axis_limit(axis)
    _save_figure(figure, output_path)


def plot_prop_actual_saturation_by_time(
    riskset_table: pd.DataFrame,
    output_path: Path,
    *,
    tolerance: float = SATURATION_TOLERANCE,
) -> None:
    """Plot saturation proportion over time."""

    summary = summarize_prop_actual_saturation_by_time(riskset_table, tolerance=tolerance)
    figure, axis = plt.subplots(figsize=(7, 4))
    if summary.empty:
        _draw_placeholder_panel(axis, title="prop_actual saturation by time", message="No finite values available.")
    else:
        axis.plot(summary["time_bin_center"], summary["proportion_saturated"], linewidth=2.0, color="#d62728")
        axis.set_xlabel("Time from partner onset (s)")
        axis.set_ylabel("Proportion saturated")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("prop_actual saturation by time")
        _apply_time_axis_limit(axis)
    _save_figure(figure, output_path)


def compute_event_rate_by_prop_actual_saturation(
    riskset_table: pd.DataFrame,
    *,
    tolerance: float = SATURATION_TOLERANCE,
) -> pd.DataFrame:
    """Compute event rate by saturated versus non-saturated status."""

    prop_actual = pd.to_numeric(riskset_table["prop_actual_cumulative_info"], errors="coerce")
    finite_mask = np.isfinite(prop_actual)
    working = riskset_table.loc[finite_mask, ["event"]].copy()
    working["saturation_status"] = np.where(
        is_saturated_prop_actual(prop_actual.loc[finite_mask], tolerance=tolerance),
        "saturated",
        "non_saturated",
    )
    summary = (
        working.groupby("saturation_status", as_index=False)["event"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "n_rows", "sum": "n_events", "mean": "event_rate"})
    )
    summary["saturation_status"] = pd.Categorical(
        summary["saturation_status"],
        categories=["non_saturated", "saturated"],
        ordered=True,
    )
    return summary.sort_values("saturation_status").reset_index(drop=True)


def plot_event_rate_by_prop_actual_saturation(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot event rate by saturation status."""

    figure, axis = plt.subplots(figsize=(6, 4))
    if summary.empty:
        _draw_placeholder_panel(axis, title="Event rate by prop_actual saturation", message="No finite values available.")
    else:
        axis.bar(summary["saturation_status"].astype(str), summary["event_rate"], color=["#4c78a8", "#e45756"])
        axis.set_ylabel("Event rate")
        axis.set_title("Event rate by prop_actual saturation")
        axis.set_ylim(0.0, max(0.05, float(summary["event_rate"].max()) * 1.1))
    _save_figure(figure, output_path)


def plot_fpp_latency_from_partner_offset_distribution(
    riskset_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot event latency from partner IPU offset when columns are available."""

    latencies = compute_event_latencies_from_partner_offset(riskset_table)
    figure, axis = plt.subplots(figsize=(6, 4))
    if latencies is None:
        _draw_placeholder_panel(
            axis,
            title="FPP latency from partner offset",
            message="Required columns are missing for partner-offset diagnostics.",
        )
    elif latencies.empty:
        _draw_placeholder_panel(axis, title="FPP latency from partner offset", message="No finite event latencies available.")
    else:
        axis.hist(latencies.to_numpy(dtype=float), bins=20)
        axis.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
        axis.set_xlabel("Own FPP onset - partner IPU offset (s)")
        axis.set_ylabel("Event rows")
        axis.set_title("FPP latency from partner IPU offset\nNegative values indicate FPP onset during partner IPU / overlap.")
        _apply_time_axis_limit(axis, allow_negative=True)
    _save_figure(figure, output_path)


def plot_event_time_from_partner_onset_distribution(
    episodes_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot event latencies from partner IPU onset."""

    latencies = pd.to_numeric(episodes_table.get("event_latency_from_partner_onset_s"), errors="coerce")
    figure, axis = plt.subplots(figsize=(6, 4))
    if latencies is None or not np.isfinite(latencies).any():
        _draw_placeholder_panel(
            axis,
            title="Event time from partner onset",
            message="No finite event latencies available.",
        )
    else:
        axis.hist(latencies[np.isfinite(latencies)].to_numpy(dtype=float), bins=20)
        axis.set_xlabel("Own FPP onset - partner IPU onset (s)")
        axis.set_ylabel("Event-positive episodes")
        axis.set_title("Event time from partner onset")
        _apply_time_axis_limit(axis)
    _save_figure(figure, output_path)


def plot_episode_duration_distribution(
    episodes_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot episode duration distribution."""

    durations = pd.to_numeric(episodes_table.get("episode_duration_s"), errors="coerce")
    figure, axis = plt.subplots(figsize=(6, 4))
    if durations is None or not np.isfinite(durations).any():
        _draw_placeholder_panel(axis, title="Episode duration", message="No finite episode durations available.")
    else:
        axis.hist(durations[np.isfinite(durations)].to_numpy(dtype=float), bins=20)
        axis.set_xlabel("Episode duration (s)")
        axis.set_ylabel("Episodes")
        axis.set_title("Episode duration distribution")
        _apply_time_axis_limit(axis, allow_negative=False)
    _save_figure(figure, output_path)


def plot_fpp_latency_from_partner_offset_before_exclusion(
    candidate_episode_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot candidate episode latencies from partner IPU offset before exclusion."""

    latencies = pd.to_numeric(candidate_episode_table.get("latency_from_partner_offset_s"), errors="coerce")
    figure, axis = plt.subplots(figsize=(6, 4))
    if latencies is None or not np.isfinite(latencies).any():
        _draw_placeholder_panel(
            axis,
            title="FPP latency from partner offset before exclusion",
            message="No finite candidate episode latencies available.",
        )
    else:
        axis.hist(latencies[np.isfinite(latencies)].to_numpy(dtype=float), bins=20)
        axis.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
        axis.set_xlabel("Own FPP onset - partner IPU offset (s)")
        axis.set_ylabel("Candidate episodes")
        axis.set_title("FPP latency from partner IPU offset before exclusion")
        _apply_time_axis_limit(axis, allow_negative=True)
    _save_figure(figure, output_path)


def build_prop_actual_saturation_warnings(
    qc_summary: dict[str, float | int | bool | None],
    *,
    non_saturated_plot_count: int,
) -> list[str]:
    """Build human-readable warnings from the saturation QC summary."""

    warnings: list[str] = []
    overall = qc_summary.get("proportion_saturated_prop_actual_rows")
    if isinstance(overall, float) and overall > 0.8:
        warnings.append(
            "prop_actual_cumulative_info is saturated at 1.0 for "
            f"{overall * 100:.1f}% of finite risk-set rows. This likely means most risk-set bins occur "
            "after the partner IPU has finished."
        )
    event_rate = qc_summary.get("proportion_saturated_event_rows")
    if isinstance(event_rate, float) and event_rate > 0.8:
        warnings.append(
            "prop_actual_cumulative_info is saturated at 1.0 for "
            f"{event_rate * 100:.1f}% of finite event rows."
        )
    if non_saturated_plot_count < MIN_NON_SATURATED_ROWS_FOR_PLOT:
        warnings.append(
            "There are fewer than 50 non-saturated prop_actual_cumulative_info rows available for plotting "
            f"({non_saturated_plot_count} rows)."
        )
    if qc_summary.get("phase_qc_available") is False:
        warnings.append("Phase/partner-offset QC could not be computed because required columns are missing.")
    return warnings


def safe_make_plots(
    *,
    riskset_table: pd.DataFrame,
    episodes_table: pd.DataFrame | None,
    prediction_grids: dict[str, pd.DataFrame],
    figures_dir: Path,
    warnings_list: list[str],
    candidate_episode_table: pd.DataFrame | None = None,
    lagged_model_comparison: pd.DataFrame | None = None,
    lagged_coefficients: pd.DataFrame | None = None,
    information_timing_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Create all required plots without failing the full pipeline."""

    plot_specs = [
        ("predicted_hazard_by_time", "time_from_partner_onset", "Predicted hazard by time"),
        ("predicted_hazard_by_information_rate", "z_information_rate", "Predicted hazard by information rate"),
        (
            "predicted_hazard_by_prop_actual_cumulative_info",
            "z_prop_actual_cumulative_info",
            "Predicted hazard by proportion actual cumulative information",
        ),
        ("predicted_hazard_by_cumulative_info", "z_cumulative_info", "Predicted hazard by cumulative information"),
        (
            "predicted_hazard_by_prop_expected_cumulative_info",
            "z_prop_expected_cumulative_info",
            "Predicted hazard by proportion expected cumulative information",
        ),
    ]
    for plot_name, x_column, title in plot_specs:
        try:
            plot_prediction_curve(
                prediction_grids[plot_name],
                x_column=x_column,
                title=title,
                output_path=figures_dir / f"{plot_name}.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            _append_warning(warnings_list, f"Failed to create figure {plot_name}: {error}")

    prop_actual_saturation_qc = compute_prop_actual_saturation_qc(riskset_table)
    observed_event_rate_summary, observed_event_rate_qc = summarize_observed_event_rate_by_time_bin(riskset_table)
    for warning in _observed_event_rate_warning_messages(observed_event_rate_qc.get("warning_flags", [])):
        _append_warning(warnings_list, warning)
    distribution_metadata = {"n_non_saturated_prop_actual_rows": 0}
    try:
        plot_observed_event_rate_by_time_bin(
            observed_event_rate_summary,
            observed_event_rate_qc,
            figures_dir / "observed_event_rate_by_time_bin.png",
        )
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure observed_event_rate_by_time_bin.png: {error}")
    try:
        plot_risk_set_qc(riskset_table, figures_dir / "risk_set_qc.png")
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure risk_set_qc.png: {error}")

    try:
        distribution_metadata = plot_information_feature_distributions(
            riskset_table,
            figures_dir / "information_feature_distributions.png",
        )
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure information_feature_distributions.png: {error}")

    extra_plots = [
        (
            lambda table, path: plot_prop_actual_by_time_from_partner_onset(table, path),
            "prop_actual_by_time_from_partner_onset.png",
        ),
        (
            lambda table, path: plot_prop_actual_saturation_by_time(table, path),
            "prop_actual_saturation_by_time.png",
        ),
        (
            lambda table, path: plot_fpp_latency_from_partner_offset_distribution(table, path),
            "fpp_latency_from_partner_offset_distribution.png",
        ),
    ]
    for plot_function, filename in extra_plots:
        try:
            plot_function(riskset_table, figures_dir / filename)
        except Exception as error:  # pragma: no cover - plotting fallback
            _append_warning(warnings_list, f"Failed to create figure {filename}: {error}")

    if episodes_table is not None:
        episode_plots = [
            (plot_event_time_from_partner_onset_distribution, "event_time_from_partner_onset_distribution.png"),
            (plot_episode_duration_distribution, "episode_duration_distribution.png"),
        ]
        for plot_function, filename in episode_plots:
            try:
                plot_function(episodes_table, figures_dir / filename)
            except Exception as error:  # pragma: no cover - plotting fallback
                _append_warning(warnings_list, f"Failed to create figure {filename}: {error}")

    if candidate_episode_table is not None:
        try:
            plot_fpp_latency_from_partner_offset_before_exclusion(
                candidate_episode_table,
                figures_dir / "fpp_latency_from_partner_offset_before_exclusion.png",
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            _append_warning(
                warnings_list,
                f"Failed to create figure fpp_latency_from_partner_offset_before_exclusion.png: {error}",
            )

    lagged_plot_specs = [
        ("information_rate_coefficient_by_lag.png", "information_rate", "z_information_rate_lag_"),
        ("prop_actual_coefficient_by_lag.png", "prop_actual", "z_prop_actual_cumulative_info_lag_"),
        ("cumulative_info_coefficient_by_lag.png", "cumulative_info", "z_cumulative_info_lag_"),
        ("prop_expected_coefficient_by_lag.png", "prop_expected", "z_prop_expected_cumulative_info_lag_"),
    ]
    for filename, family_name, term_prefix in lagged_plot_specs:
        try:
            plot_coefficient_by_lag(
                lagged_coefficients if lagged_coefficients is not None else pd.DataFrame(),
                family_name=family_name,
                term_prefix=term_prefix,
                output_path=figures_dir / filename,
            )
        except Exception as error:  # pragma: no cover - plotting fallback
            _append_warning(warnings_list, f"Failed to create figure {filename}: {error}")

    try:
        plot_model_delta_aic_by_lag(
            lagged_model_comparison if lagged_model_comparison is not None else pd.DataFrame(),
            figures_dir / "model_delta_aic_by_lag.png",
        )
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure model_delta_aic_by_lag.png: {error}")

    try:
        plot_information_timing_summary(
            information_timing_summary if information_timing_summary is not None else pd.DataFrame(),
            figures_dir / "information_timing_summary.png",
        )
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure information_timing_summary.png: {error}")

    event_rate_summary = compute_event_rate_by_prop_actual_saturation(riskset_table)
    try:
        plot_event_rate_by_prop_actual_saturation(
            event_rate_summary,
            figures_dir / "event_rate_by_prop_actual_saturation.png",
        )
    except Exception as error:  # pragma: no cover - plotting fallback
        _append_warning(warnings_list, f"Failed to create figure event_rate_by_prop_actual_saturation.png: {error}")

    warnings_list.extend(
        build_prop_actual_saturation_warnings(
            prop_actual_saturation_qc,
            non_saturated_plot_count=int(distribution_metadata["n_non_saturated_prop_actual_rows"]),
        )
    )
    return {
        "observed_event_rate_by_time_bin": observed_event_rate_summary,
        "observed_event_rate_nonzero_bins": observed_event_rate_summary.loc[
            observed_event_rate_summary["n_events"] > 0
        ].reset_index(drop=True),
        "observed_event_rate_plot_qc": observed_event_rate_qc,
        "prop_actual_saturation_qc": prop_actual_saturation_qc,
        "event_rate_by_prop_actual_saturation": event_rate_summary,
        "distribution_metadata": distribution_metadata,
    }


def plot_coefficient_by_lag(
    coefficient_table: pd.DataFrame,
    *,
    family_name: str,
    term_prefix: str,
    output_path: Path,
) -> None:
    """Plot coefficient estimates by lag for one model family."""

    figure, axis = plt.subplots(figsize=(6.5, 4))
    if coefficient_table.empty:
        _draw_placeholder_panel(axis, title=f"{family_name} coefficient by lag", message="No lagged coefficient rows available.")
        _save_figure(figure, output_path)
        return

    working = coefficient_table.copy()
    working = working.loc[
        working["term"].astype(str).str.startswith(term_prefix)
        & working["model_name"].astype(str).str.contains(family_name)
    ].copy()
    if working.empty:
        _draw_placeholder_panel(axis, title=f"{family_name} coefficient by lag", message="No matching model-family rows available.")
        _save_figure(figure, output_path)
        return

    working["lag_ms"] = pd.to_numeric(working["lag_ms"], errors="coerce")
    working["estimate"] = pd.to_numeric(working["estimate"], errors="coerce")
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    working = working.loc[np.isfinite(working["lag_ms"]) & np.isfinite(working["estimate"])].sort_values("lag_ms")
    if working.empty:
        _draw_placeholder_panel(axis, title=f"{family_name} coefficient by lag", message="No finite lag/estimate rows available.")
    else:
        axis.plot(working["lag_ms"], working["estimate"], marker="o", linewidth=2.0)
        if np.isfinite(working["conf_low"]).any() and np.isfinite(working["conf_high"]).any():
            axis.fill_between(
                working["lag_ms"],
                working["conf_low"],
                working["conf_high"],
                alpha=0.2,
            )
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel("Coefficient estimate")
        axis.set_title(f"{family_name} coefficient by lag")
    _save_figure(figure, output_path)


def plot_model_delta_aic_by_lag(
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot delta AIC by lag and model family."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    if comparison_table.empty:
        _draw_placeholder_panel(axis, title="Model delta AIC by lag", message="No lagged model comparisons available.")
        _save_figure(figure, output_path)
        return

    working = comparison_table.copy()
    working["lag_ms"] = pd.to_numeric(working.get("lag_ms"), errors="coerce")
    working["delta_aic"] = pd.to_numeric(working.get("delta_aic"), errors="coerce")
    working = working.loc[np.isfinite(working["lag_ms"]) & np.isfinite(working["delta_aic"])].copy()
    working = working.loc[working["child_model"].astype(str).str.contains("_lag_")].copy()
    if working.empty:
        _draw_placeholder_panel(axis, title="Model delta AIC by lag", message="No finite lagged delta-AIC rows available.")
        _save_figure(figure, output_path)
        return

    working["model_family"] = working["child_model"].map(_infer_model_family_from_name)
    color_map = {
        "information_rate": "#1f77b4",
        "prop_actual": "#d62728",
        "cumulative_info": "#2ca02c",
        "prop_expected": "#ff7f0e",
    }
    for family_name, family_rows in working.groupby("model_family", sort=False):
        axis.plot(
            family_rows["lag_ms"],
            family_rows["delta_aic"],
            marker="o",
            linewidth=2.0,
            label=family_name,
            color=color_map.get(str(family_name), None),
        )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Lag (ms)")
    axis.set_ylabel("Delta AIC (child - parent)")
    axis.set_title("Model delta AIC by lag")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_behaviour_timing_control_delta_aic_by_lag(
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot timing-controlled delta AIC by lag for the two behavioural predictor families."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    if comparison_table.empty:
        _draw_placeholder_panel(
            axis,
            title="Timing-control delta AIC by lag",
            message="No timing-control lag-selection rows available.",
        )
        _save_figure(figure, output_path)
        return

    working = comparison_table.copy()
    working["lag_ms"] = pd.to_numeric(working.get("lag_ms"), errors="coerce")
    working["delta_aic"] = pd.to_numeric(working.get("delta_aic"), errors="coerce")
    working = working.loc[np.isfinite(working["lag_ms"]) & np.isfinite(working["delta_aic"])].copy()
    if working.empty:
        _draw_placeholder_panel(
            axis,
            title="Timing-control delta AIC by lag",
            message="No finite timing-control lag rows available.",
        )
        _save_figure(figure, output_path)
        return

    color_map = {
        "information_rate": "#1f77b4",
        "prop_expected_cumulative_info": "#ff7f0e",
    }
    label_map = {
        "information_rate": "information_rate",
        "prop_expected_cumulative_info": "prop_expected_cumulative_info",
    }
    for family_name, family_rows in working.groupby("predictor_family", sort=False):
        family_rows = family_rows.sort_values("lag_ms").reset_index(drop=True)
        axis.plot(
            family_rows["lag_ms"],
            family_rows["delta_aic"],
            marker="o",
            linewidth=2.0,
            label=label_map.get(str(family_name), str(family_name)),
            color=color_map.get(str(family_name)),
        )
        best_row = family_rows.sort_values(["delta_aic", "lag_ms"], ascending=[True, True]).iloc[0]
        axis.scatter(
            [best_row["lag_ms"]],
            [best_row["delta_aic"]],
            color=color_map.get(str(family_name)),
            s=50,
            zorder=3,
        )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Lag (ms)")
    axis.set_ylabel("Delta AIC (child - parent)")
    axis.set_title("Timing-control delta AIC by lag")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_information_timing_summary(
    information_timing_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot distributions of information timing summaries."""

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_list = list(axes.flat)
    columns = [
        ("info_centroid_s", "Information centroid"),
        ("info_t50_s", "info_t50_s"),
        ("info_t90_s", "info_t90_s"),
        ("info_prop_by_500ms", "info_prop_by_500ms"),
        ("info_prop_by_1000ms", "info_prop_by_1000ms"),
    ]
    if information_timing_summary.empty:
        for axis in axes_list:
            _draw_placeholder_panel(axis, title="Information timing summary", message="No episode-level summary rows available.")
        _save_figure(figure, output_path)
        return

    for axis, (column_name, title) in zip(axes_list, columns, strict=False):
        values = pd.to_numeric(information_timing_summary.get(column_name), errors="coerce")
        finite_values = values[np.isfinite(values)]
        if finite_values.empty:
            _draw_placeholder_panel(axis, title=title, message="No finite values available.")
        else:
            axis.hist(finite_values.to_numpy(dtype=float), bins=20)
            axis.set_title(title)
            axis.set_xlabel(column_name)
            axis.set_ylabel("Episodes")
    if len(axes_list) > len(columns):
        _draw_placeholder_panel(axes_list[-1], title="Information timing summary", message="Summary panels")
    _save_figure(figure, output_path)


def summarize_prop_actual_by_time(riskset_table: pd.DataFrame, *, n_bins: int = TIME_BIN_COUNT) -> pd.DataFrame:
    """Summarize prop_actual over time bins using median and IQR."""

    working = riskset_table.loc[:, ["time_from_partner_onset", "prop_actual_cumulative_info"]].copy()
    working["time_from_partner_onset"] = pd.to_numeric(working["time_from_partner_onset"], errors="coerce")
    working["prop_actual_cumulative_info"] = pd.to_numeric(working["prop_actual_cumulative_info"], errors="coerce")
    working = working.loc[
        np.isfinite(working["time_from_partner_onset"]) & np.isfinite(working["prop_actual_cumulative_info"])
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["time_bin_center", "median", "q25", "q75"])
    working["time_bin"] = pd.cut(working["time_from_partner_onset"], bins=min(n_bins, max(2, working.shape[0])))
    summary = (
        working.groupby("time_bin", observed=True)["prop_actual_cumulative_info"]
        .agg(median="median", q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75))
        .reset_index()
    )
    summary["time_bin_center"] = summary["time_bin"].map(lambda interval: float(interval.mid))
    return summary


def summarize_prop_actual_saturation_by_time(
    riskset_table: pd.DataFrame,
    *,
    tolerance: float = SATURATION_TOLERANCE,
    n_bins: int = TIME_BIN_COUNT,
) -> pd.DataFrame:
    """Summarize saturation proportion over time bins."""

    working = riskset_table.loc[:, ["time_from_partner_onset", "prop_actual_cumulative_info"]].copy()
    working["time_from_partner_onset"] = pd.to_numeric(working["time_from_partner_onset"], errors="coerce")
    working["prop_actual_cumulative_info"] = pd.to_numeric(working["prop_actual_cumulative_info"], errors="coerce")
    working = working.loc[
        np.isfinite(working["time_from_partner_onset"]) & np.isfinite(working["prop_actual_cumulative_info"])
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["time_bin_center", "proportion_saturated"])
    working["saturated"] = is_saturated_prop_actual(working["prop_actual_cumulative_info"], tolerance=tolerance).astype(float)
    working["time_bin"] = pd.cut(working["time_from_partner_onset"], bins=min(n_bins, max(2, working.shape[0])))
    summary = working.groupby("time_bin", observed=True)["saturated"].mean().reset_index(name="proportion_saturated")
    summary["time_bin_center"] = summary["time_bin"].map(lambda interval: float(interval.mid))
    return summary


def compute_event_latencies_from_partner_offset(riskset_table: pd.DataFrame) -> pd.Series | None:
    """Return event latencies from partner offset when required columns exist."""

    required_columns = {"partner_ipu_offset", "own_fpp_onset", "event"}
    if not required_columns.issubset(riskset_table.columns):
        return None
    event_mask = pd.to_numeric(riskset_table["event"], errors="coerce").fillna(0).astype(int) == 1
    partner_offset = pd.to_numeric(riskset_table["partner_ipu_offset"], errors="coerce")
    own_fpp_onset = pd.to_numeric(riskset_table["own_fpp_onset"], errors="coerce")
    latency_mask = event_mask & np.isfinite(partner_offset) & np.isfinite(own_fpp_onset)
    return (own_fpp_onset - partner_offset).loc[latency_mask]


def is_saturated_prop_actual(values: pd.Series, *, tolerance: float = SATURATION_TOLERANCE) -> pd.Series:
    """Return whether values are saturated at 1.0 within tolerance."""

    numeric_values = pd.to_numeric(values, errors="coerce")
    return np.isfinite(numeric_values) & (np.abs(numeric_values - 1.0) <= tolerance)


def _draw_placeholder_panel(axis: plt.Axes, *, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _safe_proportion(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _append_warning(warnings_list: list[str], warning: str) -> None:
    LOGGER.warning(warning)
    warnings_list.append(warning)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _format_primary_coefficient_label(term_name: str) -> str | None:
    if term_name.startswith("z_information_rate_lag_"):
        lag_text = term_name.removeprefix("z_information_rate_lag_").removesuffix("ms")
        return f"Information rate, {lag_text} ms"
    if term_name.startswith("z_prop_expected_cumulative_info_lag_"):
        lag_text = term_name.removeprefix("z_prop_expected_cumulative_info_lag_").removesuffix("ms")
        return f"Expected-relative cumulative information, {lag_text} ms"
    return None


def _infer_model_family_from_name(model_name: str) -> str:
    if model_name.startswith("M1_rate_"):
        return "information_rate"
    if model_name.startswith("M2a_prop_actual_"):
        return "prop_actual"
    if model_name.startswith("M2b_cumulative_"):
        return "cumulative_info"
    if model_name.startswith("M2c_prop_expected_"):
        return "prop_expected"
    return "unknown"


def _apply_time_axis_limit(axis: plt.Axes, *, allow_negative: bool = False) -> None:
    if allow_negative:
        axis.set_xlim(-FIGURE_TIME_LIMIT_S, FIGURE_TIME_LIMIT_S)
    else:
        axis.set_xlim(0.0, FIGURE_TIME_LIMIT_S)


def _require_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        raise KeyError(f"Observed event-rate plot requires column '{column}'.")
    return table[column]


def _coerce_event_series(event_column: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(event_column):
        return event_column.astype("Int64")

    if pd.api.types.is_numeric_dtype(event_column):
        numeric = pd.to_numeric(event_column, errors="coerce")
        finite_values = numeric.dropna()
        if not finite_values.isin([0, 1]).all():
            invalid_values = sorted({str(value) for value in finite_values.loc[~finite_values.isin([0, 1])].unique()})
            raise ValueError(
                "Observed event-rate plot expected numeric event values restricted to 0/1, "
                f"but found {invalid_values}."
            )
        return numeric.astype("Int64")

    if pd.api.types.is_string_dtype(event_column) or event_column.dtype == object:
        mapping = {"0": 0, "1": 1, "false": 0, "true": 1}
        normalized = event_column.map(lambda value: value.strip() if isinstance(value, str) else value)
        coerced = normalized.map(
            lambda value: (
                mapping[value.lower()]
                if isinstance(value, str) and value.lower() in mapping
                else np.nan
                if pd.isna(value)
                else value
            )
        )
        invalid_mask = coerced.map(lambda value: not pd.isna(value) and value not in {0, 1})
        if invalid_mask.any():
            invalid_values = sorted({str(value) for value in normalized.loc[invalid_mask].unique()})
            raise ValueError(
                "Observed event-rate plot expected string event values among "
                "['0', '1', 'False', 'True', 'false', 'true'], "
                f"but found {invalid_values}."
            )
        return pd.Series(coerced, index=event_column.index, dtype="Int64")

    raise ValueError(
        "Observed event-rate plot could not coerce the event column. "
        f"Unsupported dtype: {event_column.dtype}."
    )


def _bin_observed_event_rate_without_bin_index(working: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    if working.empty:
        return pd.DataFrame(columns=["time_bin_start", "time_bin_center", "n_rows", "n_events", "event_rate"])

    time_values = np.sort(working["time_from_partner_onset"].to_numpy(dtype=float))
    min_time = float(time_values.min())
    max_time = float(time_values.max())
    bin_width = _infer_time_bin_width(time_values, n_bins=n_bins)
    epsilon = max(1.0, abs(max_time)) * 1.0e-9
    bin_edges = np.arange(min_time, max_time + bin_width + epsilon, bin_width, dtype=float)
    if bin_edges.size < 2:
        bin_edges = np.array([min_time, max_time + bin_width + epsilon], dtype=float)
    working = working.copy()
    working["time_bin"] = pd.cut(
        working["time_from_partner_onset"],
        bins=bin_edges,
        include_lowest=True,
        right=False,
    )
    binned = working.loc[working["time_bin"].notna()].copy()
    summary = (
        binned.groupby("time_bin", observed=True, as_index=False)
        .agg(
            time_bin_start=("time_from_partner_onset", "min"),
            time_bin_center=("time_from_partner_onset", "mean"),
            n_rows=("event_numeric", "size"),
            n_events=("event_numeric", "sum"),
        )
        .sort_values("time_bin_start", kind="mergesort")
        .reset_index(drop=True)
    )
    return summary.drop(columns=["time_bin"], errors="ignore")


def _infer_time_bin_width(time_values: np.ndarray, *, n_bins: int) -> float:
    unique_times = np.unique(time_values)
    if unique_times.size >= 2:
        diffs = np.diff(unique_times)
        positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive_diffs.size:
            return float(positive_diffs.min())
    if unique_times.size >= 2:
        spread = float(unique_times.max() - unique_times.min())
        if spread > 0.0:
            return spread / max(1, min(n_bins, unique_times.size))
    return 1.0


def _series_value_counts_json(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    payload: dict[str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            json_key = "NaN"
        elif isinstance(key, (bool, np.bool_)):
            json_key = "True" if bool(key) else "False"
        else:
            json_key = str(key)
        payload[json_key] = int(value)
    return payload


def _observed_event_rate_warning_flags(qc: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(qc["n_events_input"]) == 0:
        flags.append(OBSERVED_EVENT_RATE_WARNING_ZERO_EVENT_ROWS_INPUT)
    if int(qc["n_events_input"]) > 0 and int(qc["n_events_after_required_column_filter"]) == 0:
        flags.append(OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_REQUIRED_FILTER)
    if int(qc["n_events_after_required_column_filter"]) > 0 and int(qc["n_events_after_time_filter"]) == 0:
        flags.append(OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_BINNING)
    return flags


def _observed_event_rate_warning_messages(flags: Iterable[str]) -> list[str]:
    messages: list[str] = []
    for flag in flags:
        if flag == OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_REQUIRED_FILTER:
            messages.append("Observed event-rate plot dropped all event rows during required-column filtering.")
        elif flag == OBSERVED_EVENT_RATE_WARNING_DROPPED_ALL_EVENTS_BINNING:
            messages.append("Observed event-rate plot dropped all event rows during time binning.")
        elif flag == OBSERVED_EVENT_RATE_WARNING_ZERO_EVENT_ROWS_INPUT:
            messages.append("Riskset contains zero event rows. Observed event-rate plot cannot show nonzero rates.")
    return messages


def _with_unique_flag(flags: Any, flag: str) -> list[str]:
    values = [str(value) for value in flags] if isinstance(flags, list) else []
    if flag not in values:
        values.append(flag)
    return values
