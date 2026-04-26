"""Matplotlib figures for low-level neural hazard outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_LABELS = {
    "amplitude": "Amplitude",
    "alpha": "Alpha",
    "beta": "Beta",
    "all": "All low-level PCs",
}
FAMILY_COLORS = {
    "amplitude": "#1d3557",
    "alpha": "#457b9d",
    "beta": "#2a9d8f",
    "all": "#e76f51",
}


def plot_neural_lowlevel_pca_variance(
    pca_summaries: dict[str, pd.DataFrame],
    *,
    variance_threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for family_name, summary in pca_summaries.items():
        axis.plot(
            summary["component"],
            summary["cumulative_explained_variance"],
            marker="o",
            linewidth=2.0,
            color=FAMILY_COLORS.get(family_name, None),
            label=FAMILY_LABELS.get(family_name, family_name),
        )
    axis.axhline(float(variance_threshold), color="#c1121f", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Low-level neural PCA variance by family")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_neural_lowlevel_model_comparison(comparison_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    working = comparison_table.copy()
    labels = [FAMILY_LABELS.get(str(value), str(value)) for value in working["family"]]
    colors = [FAMILY_COLORS.get(str(value), "#6c757d") for value in working["family"]]
    axis.bar(labels, working["delta_aic"].to_numpy(dtype=float), color=colors)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Delta AIC (child - parent)")
    axis.set_title("Neural family and full-model comparisons")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, output_path)


def plot_neural_lowlevel_coefficients(summary_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.6))
    working = summary_table.loc[
        (summary_table["model_name"].astype(str) == "M3_neural_lowlevel")
        & summary_table["term"].astype(str).str.startswith(("amp_pc", "alpha_pc", "beta_pc"))
    ].copy()
    if working.empty:
        axis.text(0.5, 0.5, "No neural family PC coefficients available.", ha="center", va="center")
        axis.set_axis_off()
        _save_figure(figure, output_path)
        return
    working["estimate"] = pd.to_numeric(working["estimate"], errors="coerce")
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    working["family"] = working["term"].map(_term_family)
    working = working.sort_values(["family", "term"], kind="mergesort").reset_index(drop=True)
    y_positions = np.arange(len(working))
    for index, row in working.iterrows():
        axis.errorbar(
            row["estimate"],
            y_positions[index],
            xerr=[[row["estimate"] - row["conf_low"]], [row["conf_high"] - row["estimate"]]],
            fmt="o",
            color=FAMILY_COLORS.get(str(row["family"]), "#1f4e79"),
            ecolor=FAMILY_COLORS.get(str(row["family"]), "#9ec5e6"),
            capsize=4,
        )
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(working["term"].astype(str))
    axis.set_xlabel("Coefficient estimate, beta")
    axis.set_title("Neural PC coefficients grouped by family")
    _save_figure(figure, output_path)


def plot_neural_lowlevel_feature_missingness(qc_payload: dict[str, object], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    rates = [
        float(qc_payload.get("n_rows_total", 0) and qc_payload.get("n_rows_missing_all_neural", 0) / qc_payload.get("n_rows_total", 1)),
        _family_missingness(qc_payload, "neural_amp_"),
        _family_missingness(qc_payload, "neural_alpha_"),
        _family_missingness(qc_payload, "neural_beta_"),
    ]
    labels = ["any-missing", "amplitude", "alpha", "beta"]
    colors = ["#6c757d", FAMILY_COLORS["amplitude"], FAMILY_COLORS["alpha"], FAMILY_COLORS["beta"]]
    axis.bar(labels, rates, color=colors)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Missingness rate")
    axis.set_title("Low-level neural feature missingness")
    _save_figure(figure, output_path)


def _family_missingness(qc_payload: dict[str, object], prefix: str) -> float:
    missingness_by_feature = qc_payload.get("missingness_by_feature", {})
    if not isinstance(missingness_by_feature, dict):
        return 0.0
    rates = [float(value) for key, value in missingness_by_feature.items() if str(key).startswith(prefix)]
    return float(np.mean(rates)) if rates else 0.0


def _term_family(term: str) -> str:
    if term.startswith("amp_pc"):
        return "amplitude"
    if term.startswith("alpha_pc"):
        return "alpha"
    if term.startswith("beta_pc"):
        return "beta"
    return "other"


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
