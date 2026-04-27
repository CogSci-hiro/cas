"""Plot behavioural R GLMM lag-sweep and final-model outputs."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.hazard_behavior.plots import plot_behaviour_delta_bic_by_lag

ACTIVE_LAG_FAMILIES = ("information_rate", "prop_expected")
ACTIVE_FINAL_MODEL_NAMES = ("M_final_glmm", "M_final_plus_expected_glmm")


def plot_behaviour_hazard_results(
    *,
    r_results_dir: Path,
    output_dir: Path,
    timing_control_models_dir: Path | None = None,
    qc_output_dir: Path | None = None,
) -> dict[str, Path]:
    """Create the active behavioural R GLMM lag-inference figure suite."""

    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = output_dir / "data"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    if timing_control_models_dir is None:
        timing_control_models_dir = r_results_dir / "lag_selection"
    if qc_output_dir is None:
        qc_output_dir = output_dir.parent / "qc_plots" / "lag_selection"
    qc_output_dir.mkdir(parents=True, exist_ok=True)

    information_rate_sweep = _read_csv(r_results_dir / "r_glmm_information_rate_lag_sweep.csv")
    prop_expected_sweep = _read_csv(r_results_dir / "r_glmm_prop_expected_lag_sweep.csv")
    final_comparison = _read_csv(r_results_dir / "r_glmm_final_behaviour_model_comparison.csv")
    final_prediction = _read_csv(r_results_dir / "predictions" / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv")
    selected_lags = _read_json(r_results_dir / "r_glmm_selected_behaviour_lags.json")
    lag_selection_path = timing_control_models_dir / "behaviour_timing_control_lag_selection.csv"
    if lag_selection_path.exists():
        pooled_lag_selection = pd.read_csv(lag_selection_path)
    else:
        warnings.warn(
            f"Timing-control lag-selection file was not found: {lag_selection_path}",
            stacklevel=2,
        )
        pooled_lag_selection = pd.DataFrame(columns=["predictor_family", "lag_ms", "delta_bic"])

    lag_sweep = pd.concat([information_rate_sweep, prop_expected_sweep], ignore_index=True, sort=False)
    lag_sweep = filter_active_r_glmm_lag_sweep_rows(lag_sweep)
    final_comparison = filter_active_final_model_comparisons(final_comparison)

    pooled_lag_figure = qc_output_dir / "behaviour_pooled_delta_bic_by_lag.png"
    r_glmm_lag_figure = output_dir / "behaviour_r_glmm_delta_bic_by_lag.png"
    r_glmm_coefficient_figure = output_dir / "behaviour_r_glmm_coefficient_by_lag.png"
    r_glmm_odds_ratio_figure = output_dir / "behaviour_r_glmm_odds_ratio_by_lag.png"
    r_glmm_final_comparison_figure = output_dir / "behaviour_r_glmm_final_model_comparison.png"
    r_glmm_final_prediction_figure = output_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.png"

    plot_behaviour_delta_bic_by_lag(pooled_lag_selection, pooled_lag_figure)
    plot_r_glmm_delta_bic_by_lag(
        lag_sweep,
        selected_lags=selected_lags,
        output_path=r_glmm_lag_figure,
    )
    plot_r_glmm_effect_by_lag(
        lag_sweep,
        value_column="beta",
        lower_column="conf_low",
        upper_column="conf_high",
        y_label="Beta",
        title="Behavioural R GLMM coefficient by lag",
        reference_value=0.0,
        output_path=r_glmm_coefficient_figure,
    )
    plot_r_glmm_effect_by_lag(
        lag_sweep,
        value_column="odds_ratio",
        lower_column="odds_ratio_conf_low",
        upper_column="odds_ratio_conf_high",
        y_label="Odds ratio",
        title="Behavioural R GLMM odds ratio by lag",
        reference_value=1.0,
        output_path=r_glmm_odds_ratio_figure,
    )
    plot_r_glmm_final_model_comparison(final_comparison, r_glmm_final_comparison_figure)
    plot_glmm_prediction_curve(
        final_prediction,
        predictor_value_column="predictor_value",
        x_label="Information rate (z)",
        title="Final R GLMM predicted hazard by information rate",
        output_path=r_glmm_final_prediction_figure,
    )

    write_plot_data(lag_sweep, data_output_dir / "behaviour_r_glmm_delta_bic_by_lag.csv")
    write_plot_data(lag_sweep, data_output_dir / "behaviour_r_glmm_coefficient_by_lag.csv")
    write_plot_data(lag_sweep, data_output_dir / "behaviour_r_glmm_odds_ratio_by_lag.csv")
    write_plot_data(final_comparison, data_output_dir / "behaviour_r_glmm_final_model_comparison.csv")
    write_plot_data(final_prediction, data_output_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv")

    return {
        "behaviour_pooled_delta_bic_by_lag": pooled_lag_figure,
        "behaviour_r_glmm_delta_bic_by_lag": r_glmm_lag_figure,
        "behaviour_r_glmm_coefficient_by_lag": r_glmm_coefficient_figure,
        "behaviour_r_glmm_odds_ratio_by_lag": r_glmm_odds_ratio_figure,
        "behaviour_r_glmm_final_model_comparison": r_glmm_final_comparison_figure,
        "behaviour_r_glmm_final_predicted_hazard_information_rate": r_glmm_final_prediction_figure,
    }


def plot_behaviour_glmm_results(*, r_results_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Backward-compatible wrapper for the active behavioural plotting command."""

    return plot_behaviour_hazard_results(r_results_dir=r_results_dir, output_dir=output_dir)


def plot_r_glmm_delta_bic_by_lag(
    lag_sweep_table: pd.DataFrame,
    *,
    selected_lags: dict[str, object],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    if lag_sweep_table.empty:
        _draw_placeholder(axis, "Behavioural R GLMM lag sweep", "No lag-sweep rows were available.")
        _save_figure(figure, output_path)
        return

    working = lag_sweep_table.copy()
    working["lag_ms"] = pd.to_numeric(working["lag_ms"], errors="coerce")
    working["delta_BIC"] = pd.to_numeric(working["delta_BIC"], errors="coerce")
    working = working.loc[np.isfinite(working["lag_ms"]) & np.isfinite(working["delta_BIC"])].copy()
    color_map = {"information_rate": "#124559", "prop_expected": "#5b8e7d"}
    label_map = {"information_rate": "Information rate", "prop_expected": "Expected cumulative info"}
    for predictor_family in ACTIVE_LAG_FAMILIES:
        subset = working.loc[working["predictor_family"] == predictor_family].sort_values("lag_ms")
        if subset.empty:
            continue
        axis.plot(
            subset["lag_ms"],
            subset["delta_BIC"],
            marker="o",
            linewidth=2.0,
            color=color_map[predictor_family],
            label=label_map[predictor_family],
        )
    selected_rate_lag = selected_lags.get("best_information_rate_lag_ms")
    if selected_rate_lag is not None:
        selected_rate = working.loc[
            (working["predictor_family"] == "information_rate")
            & (working["lag_ms"] == float(selected_rate_lag))
        ]
        if not selected_rate.empty:
            axis.scatter(
                selected_rate["lag_ms"],
                selected_rate["delta_BIC"],
                s=70,
                color="#d62828",
                zorder=5,
                label="Selected information-rate lag",
            )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Lag (ms)")
    axis.set_ylabel("Delta BIC (child - parent)")
    axis.set_title("Behavioural R GLMM lag sweep")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_r_glmm_effect_by_lag(
    lag_sweep_table: pd.DataFrame,
    *,
    value_column: str,
    lower_column: str,
    upper_column: str,
    y_label: str,
    title: str,
    reference_value: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    if lag_sweep_table.empty:
        for axis in axes:
            _draw_placeholder(axis, title, "No lag-sweep rows were available.")
        _save_figure(figure, output_path)
        return

    working = lag_sweep_table.copy()
    for column_name in ("lag_ms", value_column, lower_column, upper_column):
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    color_map = {"information_rate": "#124559", "prop_expected": "#5b8e7d"}
    title_map = {"information_rate": "Information rate", "prop_expected": "Expected cumulative info"}
    for axis, predictor_family in zip(axes, ACTIVE_LAG_FAMILIES, strict=True):
        subset = working.loc[
            (working["predictor_family"] == predictor_family)
            & np.isfinite(working["lag_ms"])
            & np.isfinite(working[value_column])
        ].sort_values("lag_ms")
        if subset.empty:
            _draw_placeholder(axis, title_map[predictor_family], "No finite rows were available.")
            continue
        axis.plot(
            subset["lag_ms"],
            subset[value_column],
            color=color_map[predictor_family],
            linewidth=2.0,
            marker="o",
        )
        if np.isfinite(subset[lower_column]).any() and np.isfinite(subset[upper_column]).any():
            axis.fill_between(
                subset["lag_ms"],
                subset[lower_column],
                subset[upper_column],
                color=color_map[predictor_family],
                alpha=0.2,
            )
        axis.axhline(reference_value, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Lag (ms)")
        axis.set_title(title_map[predictor_family])
    axes[0].set_ylabel(y_label)
    figure.suptitle(title)
    _save_figure(figure, output_path)


def plot_r_glmm_final_model_comparison(comparison_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    if comparison_table.empty:
        _draw_placeholder(axis, "Behavioural R GLMM final comparison", "No final comparison rows were available.")
        _save_figure(figure, output_path)
        return
    working = comparison_table.copy()
    working["delta_BIC"] = pd.to_numeric(working.get("delta_BIC"), errors="coerce")
    working["delta_AIC"] = pd.to_numeric(working.get("delta_AIC"), errors="coerce")
    metric_column = "delta_BIC" if np.isfinite(working["delta_BIC"]).any() else "delta_AIC"
    working = working.loc[np.isfinite(working[metric_column])].copy()
    if working.empty:
        _draw_placeholder(axis, "Behavioural R GLMM final comparison", "No finite comparison rows were available.")
        _save_figure(figure, output_path)
        return
    colors = ["#2a9d8f" if value < 0.0 else "#e76f51" for value in working[metric_column]]
    axis.bar(working["child_model"], working[metric_column], color=colors)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_ylabel(f"{metric_column.replace('_', ' ')} (child - parent)")
    axis.set_title("Behavioural R GLMM final model comparison")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, output_path)


def plot_glmm_prediction_curve(
    prediction_table: pd.DataFrame,
    *,
    predictor_value_column: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    working = prediction_table.copy()
    for column_name in (predictor_value_column, "predicted_probability", "conf_low", "conf_high"):
        if column_name not in working.columns:
            working[column_name] = np.nan
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    working = working.loc[
        np.isfinite(working[predictor_value_column]) & np.isfinite(working["predicted_probability"])
    ].copy()
    if working.empty:
        _draw_placeholder(axis, title, "No prediction rows were available.")
        _save_figure(figure, output_path)
        return

    working = working.sort_values(predictor_value_column).reset_index(drop=True)
    axis.plot(
        working[predictor_value_column],
        working["predicted_probability"],
        color="#124559",
        linewidth=2.2,
    )
    if np.isfinite(working["conf_low"]).any() and np.isfinite(working["conf_high"]).any():
        axis.fill_between(
            working[predictor_value_column],
            working["conf_low"],
            working["conf_high"],
            color="#aec3b0",
            alpha=0.35,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Predicted probability")
    y_columns = ["predicted_probability"]
    if np.isfinite(working["conf_low"]).any():
        y_columns.append("conf_low")
    if np.isfinite(working["conf_high"]).any():
        y_columns.append("conf_high")
    y_values = working[y_columns].to_numpy(dtype=float).ravel()
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_span = max(y_max - y_min, 0.01)
        y_padding = max(y_span * 0.12, 0.005)
        axis.set_ylim(
            max(0.0, y_min - y_padding),
            min(1.0, y_max + y_padding),
        )
    axis.set_title(title)
    if "fixed_effect_only" in working.columns:
        axis.text(
            0.98,
            0.02,
            "Fixed-effect prediction only",
            ha="right",
            va="bottom",
            transform=axis.transAxes,
            fontsize=9,
        )
    _save_figure(figure, output_path)


def filter_active_r_glmm_lag_sweep_rows(lag_sweep_table: pd.DataFrame) -> pd.DataFrame:
    """Retain only the active behavioural R GLMM lag-sweep rows."""

    return lag_sweep_table.loc[lag_sweep_table["predictor_family"].isin(ACTIVE_LAG_FAMILIES)].copy()


def select_best_r_glmm_lag(lag_sweep_table: pd.DataFrame, *, predictor_family: str) -> pd.Series:
    """Select the best converged R GLMM lag by child BIC."""

    working = filter_active_r_glmm_lag_sweep_rows(lag_sweep_table)
    working = working.loc[working["predictor_family"] == predictor_family].copy()
    working["child_BIC"] = pd.to_numeric(working["child_BIC"], errors="coerce")
    working["lag_ms"] = pd.to_numeric(working["lag_ms"], errors="coerce")
    converged = working.get("converged")
    if converged is None:
        working["converged"] = False
    else:
        working["converged"] = converged.astype(bool)
    working = working.loc[working["converged"] & np.isfinite(working["child_BIC"]) & np.isfinite(working["lag_ms"])].copy()
    if working.empty:
        raise ValueError(f"No converged R GLMM lag rows were available for `{predictor_family}`.")
    return working.sort_values(["child_BIC", "lag_ms"], ascending=[True, True]).reset_index(drop=True).iloc[0]


def filter_active_final_model_comparisons(comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Retain only active final R GLMM model-comparison rows."""

    child_model = comparison_table.get("child_model")
    if child_model is None:
        return comparison_table.iloc[0:0].copy()
    return comparison_table.loc[child_model.isin(ACTIVE_FINAL_MODEL_NAMES)].copy()


def filter_active_model_comparisons(comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for active final R GLMM comparisons."""

    return filter_active_final_model_comparisons(comparison_table)


def filter_active_fixed_effects(fixed_effects_table: pd.DataFrame) -> pd.DataFrame:
    """Retain only the current behavioural target coefficients."""

    target_terms = {"z_information_rate_lag_best", "z_prop_expected_cumulative_info_lag_best"}
    return fixed_effects_table.loc[fixed_effects_table["term"].isin(target_terms)].copy()


def write_plot_data(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required behavioural R GLMM result file was not found: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required behavioural R GLMM JSON file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_placeholder(axis: plt.Axes, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
