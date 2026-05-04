"""Lag-selection figure for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lag_selection(scores: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(scores["candidate_lag_ms"], scores["delta_log_likelihood"], marker="o", linewidth=2.0)
    selected = scores.loc[scores["selected"].astype(bool)]
    if not selected.empty:
        selected_lag = float(selected.iloc[0]["candidate_lag_ms"])
        selected_score = float(selected.iloc[0]["delta_log_likelihood"])
        plt.axvline(selected_lag, color="black", linestyle="--", linewidth=1.5)
        plt.annotate(
            f"Selected: {int(selected_lag)} ms",
            xy=(selected_lag, selected_score),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("Candidate lag (ms)")
    plt.ylabel("Delta log likelihood")
    plt.title("Behavioral information lag selection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
