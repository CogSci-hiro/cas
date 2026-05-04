"""Timing heatmap figures for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _heatmap_axis(axis, table: pd.DataFrame, *, x_column: str, title: str) -> None:
    pivot = table.pivot_table(index="information_rate_original", columns=x_column, values="predicted_hazard", aggfunc="mean")
    image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks([0, pivot.shape[1] - 1], labels=[f"{float(pivot.columns.min()):.2f}", f"{float(pivot.columns.max()):.2f}"])
    axis.set_yticks([0, pivot.shape[0] - 1], labels=[f"{float(pivot.index.min()):.2f}", f"{float(pivot.index.max()):.2f}"])
    axis.set_title(title)
    axis.set_xlabel(x_column.replace("_", " "))
    axis.set_ylabel("Information rate")
    return image


def plot_timing_heatmaps(onset: pd.DataFrame, offset: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    image = _heatmap_axis(axes[0], onset, x_column="time_value_s", title="Onset-locked")
    _heatmap_axis(axes[1], offset, x_column="time_value_s", title="Offset-locked")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Predicted hazard")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
