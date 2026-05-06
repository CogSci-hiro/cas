"""Timing-by-predictor interaction plots for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


def _line_panel(
    axis,
    table: pd.DataFrame,
    *,
    title: str,
    norm: Normalize,
    cmap,
    x_limits: tuple[float, float],
    predictor_column: str,
) -> None:
    rows = table.copy()
    rows["time_value_s"] = pd.to_numeric(rows["time_value_s"], errors="coerce")
    rows["predicted_hazard"] = pd.to_numeric(rows["predicted_hazard"], errors="coerce")
    rows[predictor_column] = pd.to_numeric(rows[predictor_column], errors="coerce")
    rows = rows.dropna(subset=["time_value_s", "predicted_hazard", predictor_column])

    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Hazard probability")

    if rows.empty:
        axis.text(0.5, 0.5, "No prediction rows available", ha="center", va="center")
        axis.axis("off")
        return

    visible = rows.loc[rows["time_value_s"].between(*x_limits)]
    levels = list(rows[predictor_column].drop_duplicates())
    for level in levels:
        subset = rows.loc[rows[predictor_column] == level].sort_values("time_value_s", kind="mergesort")
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


def plot_timing_interaction(
    onset: pd.DataFrame,
    offset: pd.DataFrame,
    output_path: Path,
    *,
    predictor_column: str,
    predictor_label: str,
    title_suffix: str = "",
    onset_x_limits: tuple[float, float] = (-1.0, 2.0),
    offset_x_limits: tuple[float, float] = (-1.0, 2.0),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor_values = pd.concat(
        [
            pd.to_numeric(onset.get(predictor_column), errors="coerce"),
            pd.to_numeric(offset.get(predictor_column), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if predictor_values.empty:
        predictor_values = pd.Series([0.0, 1.0])
    predictor_min = float(predictor_values.min())
    predictor_max = float(predictor_values.max())
    if np.isclose(predictor_min, predictor_max):
        predictor_max = predictor_min + 1.0

    cmap = cm.get_cmap("inferno")
    norm = Normalize(vmin=predictor_min, vmax=predictor_max)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    _line_panel(
        axes[0],
        onset,
        title=f"Onset timing × {predictor_label}{title_suffix}",
        norm=norm,
        cmap=cmap,
        x_limits=onset_x_limits,
        predictor_column=predictor_column,
    )
    _line_panel(
        axes[1],
        offset,
        title=f"Offset timing × {predictor_label}{title_suffix}",
        norm=norm,
        cmap=cmap,
        x_limits=offset_x_limits,
        predictor_column=predictor_column,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=predictor_label)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
