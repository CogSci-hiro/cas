"""Timing x information-rate interaction plots for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


def _line_panel(axis, table: pd.DataFrame, *, title: str, norm: Normalize, cmap) -> None:
    rows = table.copy()
    rows["time_value_s"] = pd.to_numeric(rows["time_value_s"], errors="coerce")
    rows["predicted_hazard"] = pd.to_numeric(rows["predicted_hazard"], errors="coerce")
    rows["information_rate_original"] = pd.to_numeric(rows["information_rate_original"], errors="coerce")
    rows = rows.dropna(subset=["time_value_s", "predicted_hazard", "information_rate_original"])

    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Hazard probability")

    if rows.empty:
        axis.text(0.5, 0.5, "No prediction rows available", ha="center", va="center")
        axis.axis("off")
        return

    x_limits = (-2.0, 0.5)
    visible = rows.loc[rows["time_value_s"].between(*x_limits)]
    levels = list(rows["information_rate_original"].drop_duplicates())
    for level in levels:
        subset = rows.loc[rows["information_rate_original"] == level].sort_values("time_value_s", kind="mergesort")
        axis.plot(
            subset["time_value_s"],
            subset["predicted_hazard"],
            color=cmap(norm(float(level))),
            linewidth=1.6,
        )

    ylim_source = visible if not visible.empty else rows
    y_min = float(ylim_source["predicted_hazard"].min())
    y_max = float(ylim_source["predicted_hazard"].max())
    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = max((y_max - y_min) * 0.05, 1e-4)
        axis.set_ylim(y_min - pad, y_max + pad)
    axis.set_xlim(*x_limits)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_timing_information_interaction(onset: pd.DataFrame, offset: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rate_values = pd.concat(
        [
            pd.to_numeric(onset.get("information_rate_original"), errors="coerce"),
            pd.to_numeric(offset.get("information_rate_original"), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if rate_values.empty:
        rate_values = pd.Series([0.0, 1.0])
    rate_min = float(rate_values.min())
    rate_max = float(rate_values.max())
    if np.isclose(rate_min, rate_max):
        rate_max = rate_min + 1.0

    cmap = cm.get_cmap("inferno")
    norm = Normalize(vmin=rate_min, vmax=rate_max)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    _line_panel(axes[0], onset, title="Onset timing × information rate", norm=norm, cmap=cmap)
    _line_panel(axes[1], offset, title="Offset timing × information rate", norm=norm, cmap=cmap)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Information rate")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
