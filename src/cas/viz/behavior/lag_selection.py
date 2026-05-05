"""Lag-selection figure for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lag_selection(scores: pd.DataFrame, output_path: Path, *, title_suffix: str = "") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    criterion = str(scores["lag_selection_criterion"].iloc[0]) if "lag_selection_criterion" in scores.columns and not scores.empty else "bic"
    y_column = "delta_BIC" if criterion == "bic" else "logLik"
    y_label = "Delta BIC" if criterion == "bic" else "Log-likelihood"
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(scores["lag_ms"], scores[y_column], marker="o", linewidth=2.0)
    selected = scores.loc[scores["selected"].astype(bool)]
    if not selected.empty:
        selected_lag = float(selected.iloc[0]["lag_ms"])
        selected_score = float(selected.iloc[0][y_column])
        plt.axvline(selected_lag, color="black", linestyle="--", linewidth=1.5)
        plt.annotate(
            f"Selected: {int(selected_lag)} ms",
            xy=(selected_lag, selected_score),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("Candidate lag (ms)")
    plt.ylabel(y_label)
    plt.title(f"Shared M_3 lag selection ({criterion}){title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
