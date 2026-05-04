"""Robustness figures for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_leave_one_subject_out(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not table.empty:
        for term, rows in table.groupby("term", sort=False):
            ax.plot(rows["left_out_subject"], rows["estimate"], marker="o", linewidth=1.8, label=str(term))
    ax.set_xlabel("Left-out subject")
    ax.set_ylabel("Coefficient estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
