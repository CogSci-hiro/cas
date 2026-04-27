"""Plotting helpers for the FPP neural permutation null."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_permutation_null_outputs(
    *,
    real_comparison: pd.DataFrame,
    null_distribution: pd.DataFrame,
    summary: dict[str, object],
    figures_dir: Path,
) -> list[Path]:
    """Write the standard FPP neural permutation-null figures."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    successful = _successful_null_rows(null_distribution)
    paths = [
        _plot_delta_bic_histogram(successful, summary=summary, output_path=figures_dir / "fpp_neural_permutation_delta_bic_null.png"),
        _plot_delta_bic_ecdf(successful, summary=summary, output_path=figures_dir / "fpp_neural_permutation_delta_bic_ecdf.png"),
        _plot_real_vs_null_summary(successful, summary=summary, output_path=figures_dir / "fpp_neural_permutation_real_vs_null_summary.png"),
        _plot_shift_qc(successful, output_path=figures_dir / "fpp_neural_permutation_shift_qc.png"),
    ]
    _write_plot_metadata(
        real_comparison=real_comparison,
        null_distribution=successful,
        summary=summary,
        output_path=figures_dir / "fpp_neural_permutation_plot_metadata.json",
    )
    return paths


def plot_family_comparison(
    combined_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    """Plot the family-level real-vs-null comparison summary."""

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    working = combined_summary.copy()
    if working.empty:
        _draw_placeholder(axis, "FPP neural permutation family comparison", "No family summaries were available.")
        return _save_figure(figure, output_path)

    working["real_delta_bic"] = pd.to_numeric(working["real_delta_bic"], errors="coerce")
    working["null_q025_delta_bic"] = pd.to_numeric(working["null_q025_delta_bic"], errors="coerce")
    working["null_q975_delta_bic"] = pd.to_numeric(working["null_q975_delta_bic"], errors="coerce")
    working["null_median_delta_bic"] = pd.to_numeric(working["null_median_delta_bic"], errors="coerce")
    working["empirical_p_delta_bic"] = pd.to_numeric(working["empirical_p_delta_bic"], errors="coerce")
    order = ["alpha", "beta", "alpha_beta"]
    working["x"] = [order.index(value) if value in order else len(order) + idx for idx, value in enumerate(working["neural_family"].astype(str))]
    working = working.sort_values("x")

    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    for _, row in working.iterrows():
        x_value = float(row["x"])
        axis.vlines(x_value, row["null_q025_delta_bic"], row["null_q975_delta_bic"], color="#91a6c6", linewidth=4.0)
        axis.scatter([x_value], [row["null_median_delta_bic"]], color="#43658b", s=40, zorder=3)
        axis.scatter([x_value], [row["real_delta_bic"]], color="#bc4b51", s=55, zorder=4)
        p_value = row["empirical_p_delta_bic"]
        if np.isfinite(p_value):
            axis.text(x_value, row["real_delta_bic"], f" p={p_value:.3f}", ha="left", va="center", fontsize=9)
    axis.set_xticks(working["x"])
    axis.set_xticklabels(working["neural_family"].astype(str))
    axis.set_xlabel("neural_family")
    axis.set_ylabel("delta BIC (child - parent)")
    axis.set_title("FPP neural permutation null by family")
    return _save_figure(figure, output_path)


def _successful_null_rows(null_distribution: pd.DataFrame) -> pd.DataFrame:
    working = null_distribution.copy()
    if "status" in working.columns:
        working = working.loc[working["status"].astype(str) == "ok"].copy()
    for column_name in ("delta_bic", "delta_aic", "proportion_event_rows_changed", "proportion_event_episodes_changed"):
        if column_name in working.columns:
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    return working


def _plot_delta_bic_histogram(
    null_distribution: pd.DataFrame,
    *,
    summary: dict[str, object],
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    values = pd.to_numeric(null_distribution.get("delta_bic"), errors="coerce")
    values = values[np.isfinite(values)]
    real_delta_bic = _as_float(summary.get("real_delta_bic"))
    empirical_p = _as_float(summary.get("empirical_p_delta_bic"))
    if values.empty:
        _draw_placeholder(axis, "Permutation null: delta BIC", "No successful permutations were available.")
    else:
        axis.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))), color="#91a6c6", edgecolor="white")
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        if real_delta_bic is not None:
            axis.axvline(real_delta_bic, color="#bc4b51", linewidth=2.0)
        label = "more negative favours neural model"
        if empirical_p is not None:
            label = f"{label}\nempirical p={empirical_p:.3f}"
        axis.text(0.98, 0.98, label, ha="right", va="top", transform=axis.transAxes)
        axis.set_xlabel("delta BIC = child - parent")
        axis.set_ylabel("Count")
        axis.set_title("FPP neural permutation null distribution")
    return _save_figure(figure, output_path)


def _plot_delta_bic_ecdf(
    null_distribution: pd.DataFrame,
    *,
    summary: dict[str, object],
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    values = pd.to_numeric(null_distribution.get("delta_bic"), errors="coerce")
    values = np.sort(values[np.isfinite(values)])
    real_delta_bic = _as_float(summary.get("real_delta_bic"))
    empirical_p = _as_float(summary.get("empirical_p_delta_bic"))
    if values.size == 0:
        _draw_placeholder(axis, "Permutation null ECDF", "No successful permutations were available.")
    else:
        y_values = np.arange(1, values.size + 1, dtype=float) / float(values.size)
        axis.step(values, y_values, where="post", color="#43658b", linewidth=2.0)
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        if real_delta_bic is not None:
            axis.axvline(real_delta_bic, color="#bc4b51", linewidth=2.0)
        if empirical_p is not None:
            axis.text(0.98, 0.05, f"empirical p={empirical_p:.3f}", ha="right", va="bottom", transform=axis.transAxes)
        axis.set_xlabel("delta BIC = child - parent")
        axis.set_ylabel("ECDF")
        axis.set_title("FPP neural permutation null ECDF")
    return _save_figure(figure, output_path)


def _plot_real_vs_null_summary(
    null_distribution: pd.DataFrame,
    *,
    summary: dict[str, object],
    output_path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=False)
    metrics = [
        ("delta_bic", "Delta BIC"),
        ("delta_aic", "Delta AIC"),
    ]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = pd.to_numeric(null_distribution.get(metric), errors="coerce")
        values = values[np.isfinite(values)]
        real_value = _as_float(summary.get(f"real_{metric}"))
        if values.empty or real_value is None:
            _draw_placeholder(axis, title, "No successful permutations were available.")
            continue
        quantiles = np.nanpercentile(values, [2.5, 50.0, 97.5])
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.vlines(0.0, quantiles[0], quantiles[2], color="#91a6c6", linewidth=6.0)
        axis.scatter([0.0], [quantiles[1]], color="#43658b", s=45, zorder=3)
        axis.scatter([0.0], [real_value], color="#bc4b51", s=60, zorder=4)
        axis.set_xlim(-0.5, 0.5)
        axis.set_xticks([0.0])
        axis.set_xticklabels(["real vs null"])
        axis.set_ylabel(f"{title}\n(child - parent)")
        axis.set_title(title)
    figure.suptitle("Real neural improvement vs permutation null")
    return _save_figure(figure, output_path)


def _plot_shift_qc(null_distribution: pd.DataFrame, *, output_path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    changed = pd.to_numeric(null_distribution.get("proportion_event_rows_changed"), errors="coerce")
    changed = changed[np.isfinite(changed)]
    events_after = pd.to_numeric(null_distribution.get("n_events_after_shift"), errors="coerce")
    events_after = events_after[np.isfinite(events_after)]
    if changed.empty:
        _draw_placeholder(axes[0], "Shift QC", "No successful permutations were available.")
    else:
        axes[0].hist(changed, bins=min(30, max(8, int(np.sqrt(len(changed))))), color="#5b8e7d", edgecolor="white")
        axes[0].set_xlabel("Proportion of rows with changed event label")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Event-shift QC")
    if events_after.empty:
        _draw_placeholder(axes[1], "Events after shift", "No successful permutations were available.")
    else:
        axes[1].hist(events_after, bins=min(15, max(3, len(np.unique(events_after)))), color="#d8a47f", edgecolor="white")
        axes[1].set_xlabel("Events after shift")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Events preserved across permutations")
    return _save_figure(figure, output_path)


def _write_plot_metadata(
    *,
    real_comparison: pd.DataFrame,
    null_distribution: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    payload = {
        "real_rows": int(len(real_comparison)),
        "null_rows_successful": int(len(null_distribution)),
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _draw_placeholder(axis: plt.Axes, title: str, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    axis.set_title(title)


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def _as_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric
