"""Lag-sensitivity figure for behavioral hazard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lag_sensitivity(table: pd.DataFrame, *, selected_lag_ms: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictors = list(table["predictor"].astype(str).drop_duplicates()) if not table.empty else ["information_rate", "prop_expected_cum_info"]
    fig, axes = plt.subplots(1, len(predictors), figsize=(6.5 * max(len(predictors), 1), 4.0), squeeze=False)
    for axis, predictor in zip(axes[0], predictors):
        rows = table.loc[table["predictor"].astype(str) == predictor].sort_values("candidate_lag_ms", kind="mergesort")
        axis.plot(rows["candidate_lag_ms"], rows["estimate"], marker="o", linewidth=2.0, label=predictor)
        axis.fill_between(rows["candidate_lag_ms"], rows["ci_low"], rows["ci_high"], alpha=0.2)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.axvline(float(selected_lag_ms), color="black", linestyle="--", linewidth=1.2)
        axis.set_title(str(predictor))
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel("Log-odds coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
