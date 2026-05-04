"""Primary-effects figures for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_primary_effects(predictions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for axis, panel, title in zip(
        axes,
        ["A", "B", "C"],
        ["FPP by information rate", "FPP by expected cumulative info", "FPP vs SPP rate contrast"],
    ):
        subset = predictions.loc[predictions["panel"] == panel]
        for anchor_type, anchor_rows in subset.groupby("anchor_type", sort=False):
            anchor_rows = anchor_rows.sort_values("x_value_original", kind="mergesort")
            axis.plot(anchor_rows["x_value_original"], anchor_rows["predicted_hazard"], linewidth=2.0, label=str(anchor_type))
            if {"ci_low", "ci_high"} <= set(anchor_rows.columns):
                axis.fill_between(anchor_rows["x_value_original"], anchor_rows["ci_low"], anchor_rows["ci_high"], alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel("Back-transformed predictor")
    axes[0].set_ylabel("Predicted hazard")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
