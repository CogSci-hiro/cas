"""Plotting helpers for behavioural hazard analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SATURATION_TOLERANCE = 1.0e-8
MIN_NON_SATURATED_ROWS_FOR_PLOT = 50
TIME_BIN_COUNT = 50


def plot_prediction_curve(prediction_table: pd.DataFrame, *, x_column: str, title: str, output_path: Path) -> None:
    """Plot a predicted hazard curve."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(prediction_table[x_column], prediction_table["predicted_hazard"], linewidth=2.0)
    axis.set_xlabel(x_column)
    axis.set_ylabel("Predicted hazard")
    axis.set_title(title)
    axis.set_ylim(0.0, max(0.05, float(prediction_table["predicted_hazard"].max()) * 1.1))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_observed_event_rate_by_time_bin(riskset_table: pd.DataFrame, output_path: Path) -> None:
    """Plot observed event rate by time bin."""

    summary = (
        riskset_table.groupby("bin_index", as_index=False)
        .agg(time_from_partner_onset=("time_from_partner_onset", "mean"), observed_event_rate=("event", "mean"))
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(summary["time_from_partner_onset"], summary["observed_event_rate"], marker="o")
    axis.set_xlabel("Time from partner onset (s)")
    axis.set_ylabel("Observed event rate")
    axis.set_title("Observed event rate by time bin")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
        axis.set_title("FPP latency from partner IPU offset")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    prediction_grids: dict[str, pd.DataFrame],
    figures_dir: Path,
    warnings_list: list[str],
    candidate_episode_table: pd.DataFrame | None = None,
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
    distribution_metadata = {"n_non_saturated_prop_actual_rows": 0}
    qc_specs = [
        (plot_observed_event_rate_by_time_bin, "observed_event_rate_by_time_bin.png"),
        (plot_risk_set_qc, "risk_set_qc.png"),
    ]
    for plot_function, filename in qc_specs:
        try:
            plot_function(riskset_table, figures_dir / filename)
        except Exception as error:  # pragma: no cover - plotting fallback
            _append_warning(warnings_list, f"Failed to create figure {filename}: {error}")

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
        "prop_actual_saturation_qc": prop_actual_saturation_qc,
        "event_rate_by_prop_actual_saturation": event_rate_summary,
        "distribution_metadata": distribution_metadata,
    }


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
