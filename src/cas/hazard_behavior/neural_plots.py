"""Matplotlib figures for low-level neural hazard outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_neural_lowlevel_pca_variance(
    pca_summary: pd.DataFrame,
    *,
    variance_threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.plot(
        pca_summary["component"],
        pca_summary["cumulative_explained_variance"],
        marker="o",
        color="#1f4e79",
        linewidth=2.0,
    )
    axis.axhline(float(variance_threshold), color="#c1121f", linestyle="--", linewidth=1.2)
    selected_rows = pca_summary.loc[pca_summary["selected_for_model"].astype(bool)]
    if not selected_rows.empty:
        selected_component = int(selected_rows["component"].max())
        axis.axvline(selected_component, color="#2a9d8f", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Low-level neural PCA variance retained")
    _save_figure(figure, output_path)


def plot_neural_lowlevel_model_comparison(comparison_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    delta_aic = float(comparison_table.loc[0, "delta_aic"])
    color = "#2a9d8f" if delta_aic < 0.0 else "#e76f51"
    axis.bar(["M3 vs M2"], [delta_aic], color=color)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Delta AIC (child - parent)")
    axis.set_title("Neural low-level model comparison")
    _save_figure(figure, output_path)


def plot_neural_lowlevel_coefficients(summary_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    working = summary_table.loc[
        (summary_table["model_name"].astype(str) == "M3_neural_lowlevel")
        & summary_table["term"].astype(str).str.startswith("neural_pc")
    ].copy()
    if working.empty:
        axis.text(0.5, 0.5, "No neural PC coefficients available.", ha="center", va="center")
        axis.set_axis_off()
        _save_figure(figure, output_path)
        return
    working["estimate"] = pd.to_numeric(working["estimate"], errors="coerce")
    working["conf_low"] = pd.to_numeric(working["conf_low"], errors="coerce")
    working["conf_high"] = pd.to_numeric(working["conf_high"], errors="coerce")
    y_positions = np.arange(len(working))
    xerr = np.vstack(
        [
            working["estimate"].to_numpy(dtype=float) - working["conf_low"].to_numpy(dtype=float),
            working["conf_high"].to_numpy(dtype=float) - working["estimate"].to_numpy(dtype=float),
        ]
    )
    axis.errorbar(
        working["estimate"],
        y_positions,
        xerr=xerr,
        fmt="o",
        color="#1f4e79",
        ecolor="#9ec5e6",
        capsize=4,
    )
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(working["term"].astype(str))
    axis.set_xlabel("Coefficient estimate, beta")
    axis.set_title("Neural PC coefficients in M3")
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
    axis.bar(labels, rates, color=["#6c757d", "#1d3557", "#457b9d", "#2a9d8f"])
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


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
