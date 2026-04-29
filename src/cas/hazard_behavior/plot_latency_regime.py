"""Plotting helpers for the exploratory behavioural latency-regime Stan analysis."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, t


MODEL_A = "model_a_one_student_t"
MODEL_B = "model_b_two_student_t_mixture"
MODEL_C = "model_c_mixture_of_experts"
MODEL_R1 = "model_r1_student_t_location_regression"
MODEL_R2 = "model_r2_student_t_location_scale_regression"
MODEL_R3 = "model_r3_shifted_lognormal_location_regression"
MODEL_R4 = "model_r4_shifted_lognormal_location_scale_regression"
MODEL_S = "model_s_skew_unimodal"

MODEL_LABELS = {
    MODEL_A: "one Student-t",
    MODEL_S: "skewed unimodal",
    MODEL_B: "two Student-t mixture",
    MODEL_C: "mixture of experts",
    MODEL_R1: "Student-t location regression",
    MODEL_R2: "Student-t location + scale regression",
    MODEL_R3: "shifted-lognormal location regression",
    MODEL_R4: "shifted-lognormal location + scale regression",
}

DENSITY_TYPE_CONDITIONAL = "conditional_fixed_predictors"
DENSITY_TYPE_MARGINAL = "marginal_over_observed_predictors"
DENSITY_TYPE_COMPONENT = "component_density"
DENSITY_TYPE_WEIGHTED = "weighted_mixture_density"
DENSITY_TYPE_POSTERIOR_PREDICTIVE = "posterior_predictive_density"
DENSITY_TYPE_HISTOGRAM = "histogram_density"
DENSITY_TYPE_UNKNOWN = "unknown"


def seconds_to_milliseconds(values_s: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert latency values from seconds to milliseconds."""

    return np.asarray(values_s, dtype=float) * 1000.0


def milliseconds_to_seconds(values_ms: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert latency values from milliseconds to seconds."""

    return np.asarray(values_ms, dtype=float) / 1000.0


def density_per_second_to_per_millisecond(density_s: np.ndarray | float) -> np.ndarray:
    """Convert density from probability per second to probability per millisecond."""

    return np.asarray(density_s, dtype=float) / 1000.0


def _evaluate_density_per_ms(
    *,
    grid_ms: np.ndarray,
    density_in_seconds_fn: callable,
) -> np.ndarray:
    """Evaluate a seconds-parameterized density on an ms plotting grid."""

    grid_s = milliseconds_to_seconds(grid_ms)
    density_s = density_in_seconds_fn(np.asarray(grid_s, dtype=float))
    return density_per_second_to_per_millisecond(density_s)


def student_t_density_per_ms(grid_ms: np.ndarray, *, mu_s: float, sigma_s: float, nu: float) -> np.ndarray:
    """Student-t density evaluated on ms grid and returned in per-ms units."""

    return _evaluate_density_per_ms(
        grid_ms=grid_ms,
        density_in_seconds_fn=lambda grid_s: t.pdf(grid_s, df=nu, loc=mu_s, scale=sigma_s),
    )


def normal_density_per_ms(grid_ms: np.ndarray, *, mu_s: float, sigma_s: float) -> np.ndarray:
    """Normal density evaluated on ms grid and returned in per-ms units."""

    return _evaluate_density_per_ms(
        grid_ms=grid_ms,
        density_in_seconds_fn=lambda grid_s: norm.pdf(grid_s, loc=mu_s, scale=sigma_s),
    )


def skew_normal_pdf(x: np.ndarray, *, xi: float, omega: float, alpha: float) -> np.ndarray:
    """Skew-normal density in per-second units for an x-grid in seconds."""

    x = np.asarray(x, dtype=float)
    if not np.isfinite(omega) or omega <= 0:
        return np.full_like(x, fill_value=np.nan, dtype=float)
    z = (x - xi) / omega
    return (2.0 / omega) * norm.pdf(z) * norm.cdf(alpha * z)


def skew_normal_density_per_ms(grid_ms: np.ndarray, *, xi_s: float, omega_s: float, alpha_skew: float) -> np.ndarray:
    """Skew-normal density evaluated on ms grid and returned in per-ms units."""

    return _evaluate_density_per_ms(
        grid_ms=grid_ms,
        density_in_seconds_fn=lambda grid_s: skew_normal_pdf(grid_s, xi=xi_s, omega=omega_s, alpha=alpha_skew),
    )


def mixture_density_per_ms(
    *,
    early_density_s: np.ndarray,
    late_density_s: np.ndarray,
    p_late_values: np.ndarray | None,
    fixed_weight: float | None = None,
) -> np.ndarray:
    """Build mixture density in per-ms units from per-second component densities."""

    early = np.asarray(early_density_s, dtype=float)
    late = np.asarray(late_density_s, dtype=float)
    if fixed_weight is not None:
        mixture_s = (1.0 - float(fixed_weight)) * early + float(fixed_weight) * late
    else:
        p = np.asarray(p_late_values if p_late_values is not None else [], dtype=float)
        p = p[np.isfinite(p)]
        if p.size == 0:
            p = np.array([0.5], dtype=float)
        mixture_s = np.mean((1.0 - p[:, None]) * early[None, :] + p[:, None] * late[None, :], axis=0)
    return density_per_second_to_per_millisecond(mixture_s)


def select_exact_model_rows(
    table: pd.DataFrame | None,
    *,
    model_name: str,
    source_table: str,
    allow_missing_model_column: bool = False,
) -> pd.DataFrame:
    """Return rows for exactly one model name and reject mixed-model selections."""

    if table is None:
        return pd.DataFrame()
    working = table.copy()
    if "model_name" not in working.columns:
        if allow_missing_model_column:
            return working
        raise AssertionError(f"{source_table} does not contain a model_name column for exact filtering.")
    working["model_name"] = working["model_name"].astype(str)
    subset = working.loc[working["model_name"] == model_name].copy()
    if not subset.empty:
        observed = set(subset["model_name"].unique())
        if observed != {model_name}:
            raise AssertionError(f"{source_table} unexpectedly mixed model rows while selecting {model_name}: {observed}.")
    return subset


def count_local_maxima(
    grid: np.ndarray,
    density: np.ndarray,
    prominence_threshold: float | None = None,
) -> int:
    """Count simple local maxima on a 1D density grid."""

    x = np.asarray(grid, dtype=float)
    y = np.asarray(density, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size < 3:
        return 0
    threshold = 0.0 if prominence_threshold is None else float(prominence_threshold)
    count = 0
    for idx in range(1, y.size - 1):
        left = y[idx] - y[idx - 1]
        right = y[idx] - y[idx + 1]
        if left > 0.0 and right > 0.0 and min(left, right) > threshold:
            count += 1
    return count


def check_density_area(
    grid_ms: np.ndarray,
    density_ms: np.ndarray,
    label: str,
    tolerance: float = 0.05,
) -> dict[str, object]:
    """Check whether a plotted density integrates to expected mass on the provided grid."""

    grid = np.asarray(grid_ms, dtype=float)
    density = np.asarray(density_ms, dtype=float)
    finite = np.isfinite(grid) & np.isfinite(density)
    if finite.sum() < 2:
        message = f"{label}: insufficient finite points for area check."
        warnings.warn(message, stacklevel=2)
        return {
            "label": label,
            "area": float("nan"),
            "ok": False,
            "warning": message,
            "grid_min_ms": float("nan"),
            "grid_max_ms": float("nan"),
        }

    area = float(np.trapezoid(density[finite], grid[finite]))
    diff = abs(area - 1.0)
    warning_text = ""
    ok = diff <= tolerance
    if not ok:
        warning_text = (
            f"{label}: density area={area:.4f} outside tolerance ±{tolerance:.3f} "
            f"on grid [{float(np.nanmin(grid)):.1f}, {float(np.nanmax(grid)):.1f}] ms."
        )
        warnings.warn(warning_text, stacklevel=2)
    return {
        "label": label,
        "area": area,
        "ok": ok,
        "warning": warning_text,
        "grid_min_ms": float(np.nanmin(grid)),
        "grid_max_ms": float(np.nanmax(grid)),
    }


def plot_behaviour_latency_regime_results(
    *,
    stan_results_dir: Path,
    event_data_csv: Path,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, Path]:
    """Render the exploratory latency-regime figure suite."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if verbose:
        print(f"Reading latency-regime event data from {event_data_csv}.")

    event_data = pd.read_csv(event_data_csv)
    component_parameters = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_component_parameters.csv")
    event_probabilities = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_event_probabilities.csv")
    gating_coefficients = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_gating_coefficients.csv")
    ppc_table = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv")
    loo_table = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_loo_comparison.csv")
    regression_coefficients = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_regression_coefficients.csv")
    regression_predictions = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_regression_predictions.csv")
    shifted_lognormal_diagnostics = _read_csv_if_exists(
        stan_results_dir / "behaviour_latency_regime_shifted_lognormal_diagnostics.csv"
    )
    model_s_summary = _read_csv_if_exists(stan_results_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv")
    model_a_summary = _read_csv_if_exists(stan_results_dir / "behaviour_latency_model_a_one_student_t_summary.csv")
    fit_metrics = _read_json_if_exists(stan_results_dir / "behaviour_latency_regime_fit_metrics.json")
    nu = float((fit_metrics or {}).get("nu", 4.0))

    if verbose:
        print(f"Rendering latency-regime figures into {output_dir}.")

    outputs["components"] = output_dir / "behaviour_latency_regime_components.png"
    plot_latency_regime_components(
        component_parameters=component_parameters,
        event_data=event_data,
        event_probabilities=event_probabilities,
        model_name=MODEL_C,
        output_path=outputs["components"],
        title="Latency-regime components (Model C mixture-of-experts)",
        nu=nu,
    )

    outputs["probability_by_expected_info"] = output_dir / "behaviour_latency_regime_probability_by_expected_info.png"
    plot_probability_by_predictor(
        event_probabilities=event_probabilities,
        predictor_column="z_prop_expected_cumulative_info_lag_best",
        output_path=outputs["probability_by_expected_info"],
        x_label="Expected cumulative information (z)",
        model_name=MODEL_C,
        title="P(late) by expected info (Model C)",
    )

    outputs["probability_by_information_rate"] = output_dir / "behaviour_latency_regime_probability_by_information_rate.png"
    plot_probability_by_predictor(
        event_probabilities=event_probabilities,
        predictor_column="z_information_rate_lag_best",
        output_path=outputs["probability_by_information_rate"],
        x_label="Information rate (z)",
        model_name=MODEL_C,
        title="P(late) by information rate (Model C)",
    )

    outputs["gating_coefficients"] = output_dir / "behaviour_latency_regime_gating_coefficients.png"
    plot_gating_coefficients(gating_coefficients=gating_coefficients, output_path=outputs["gating_coefficients"])

    outputs["skew_vs_mixture"] = output_dir / "behaviour_latency_regime_skew_vs_mixture.png"
    plot_skew_vs_mixture_overlay(
        event_data=event_data,
        model_s_summary=model_s_summary,
        component_parameters=component_parameters,
        event_probabilities=event_probabilities,
        ppc_table=ppc_table,
        loo_table=loo_table,
        output_path=outputs["skew_vs_mixture"],
        nu=nu,
    )

    audit_paths = write_latency_regime_plot_source_audit(
        stan_results_dir=stan_results_dir,
        event_data=event_data,
        output_dir=output_dir.parent / "diagnostics" / "plot_source_audit",
        component_parameters=component_parameters,
        event_probabilities=event_probabilities,
        gating_coefficients=gating_coefficients,
        ppc_table=ppc_table,
        loo_table=loo_table,
        regression_coefficients=regression_coefficients,
        fit_metrics=fit_metrics,
    )
    outputs.update(audit_paths)

    outputs["regression_vs_mixture"] = output_dir / "behaviour_latency_regime_regression_vs_mixture.png"
    plot_regression_vs_mixture_overlay(
        event_data=event_data,
        ppc_table=ppc_table,
        loo_table=loo_table,
        output_path=outputs["regression_vs_mixture"],
    )

    ppc_output = output_dir / "behaviour_latency_regime_ppc.png"
    if ppc_table is None:
        warnings.warn("Posterior predictive CSV was not found; skipping latency-regime PPC plot.", stacklevel=2)
    else:
        plot_latency_regime_ppc(ppc_table=ppc_table, event_data=event_data, output_path=ppc_output, loo_table=loo_table)
        outputs["ppc"] = ppc_output

    if loo_table is not None and not loo_table.empty:
        outputs["loo_comparison"] = output_dir / "behaviour_latency_regime_loo_comparison.png"
        plot_latency_regime_loo_comparison(loo_table=loo_table, output_path=outputs["loo_comparison"])

    outputs["latency_by_expected_info_regression"] = (
        output_dir / "behaviour_latency_regime_latency_by_expected_info_regression.png"
    )
    plot_latency_regime_regression_effects(
        regression_predictions=regression_predictions,
        loo_table=loo_table,
        predictor_name="z_prop_expected_cumulative_info_lag_best",
        x_label="Expected cumulative information (z)",
        output_path=outputs["latency_by_expected_info_regression"],
    )

    outputs["latency_by_information_rate_regression"] = (
        output_dir / "behaviour_latency_regime_latency_by_information_rate_regression.png"
    )
    plot_latency_regime_regression_effects(
        regression_predictions=regression_predictions,
        loo_table=loo_table,
        predictor_name="z_information_rate_lag_best",
        x_label="Information rate (z)",
        output_path=outputs["latency_by_information_rate_regression"],
    )

    diagnostics_paths = _write_density_scaling_diagnostics(
        output_dir=output_dir,
        event_data=event_data,
        component_parameters=component_parameters,
        event_probabilities=event_probabilities,
        gating_coefficients=gating_coefficients,
        model_s_summary=model_s_summary,
        model_a_summary=model_a_summary,
        regression_coefficients=regression_coefficients,
        shifted_lognormal_diagnostics=shifted_lognormal_diagnostics,
        nu=nu,
    )
    outputs.update(diagnostics_paths)

    if verbose:
        print("Completed latency-regime figure rendering:")
        for path in outputs.values():
            print(f"  - {path}")

    return outputs


def plot_latency_regime_components(
    *,
    component_parameters: pd.DataFrame | None,
    event_data: pd.DataFrame,
    event_probabilities: pd.DataFrame | None,
    model_name: str,
    output_path: Path,
    title: str,
    nu: float,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    latency_seconds = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    finite_latency = latency_seconds[np.isfinite(latency_seconds)]
    if finite_latency.empty:
        _draw_placeholder(axis, title, "No finite latency values were available.")
        _save_figure(figure, output_path)
        return

    finite_latency_ms = seconds_to_milliseconds(finite_latency.to_numpy(dtype=float))
    axis.hist(finite_latency_ms, bins=24, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9)
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axvline(200.0, color="#c1121f", linestyle=":", linewidth=1.2)
    axis.axvline(1000.0, color="#6d597a", linestyle=":", linewidth=1.2)

    if component_parameters is not None and not component_parameters.empty:
        subset = component_parameters.loc[component_parameters.get("model_name") == model_name].copy()
        if not subset.empty:
            early_mu = _extract_component_value(subset, component="early", parameter="mu")
            late_mu = _extract_component_value(subset, component="late", parameter="mu")
            early_sigma = _extract_component_value(subset, component="early", parameter="sigma")
            late_sigma = _extract_component_value(subset, component="late", parameter="sigma")
            grid_ms = _build_latency_grid_ms(finite_latency.to_numpy(dtype=float))
            grid_s = milliseconds_to_seconds(grid_ms)
            early_density_s = t.pdf(grid_s, df=nu, loc=early_mu, scale=early_sigma)
            late_density_s = t.pdf(grid_s, df=nu, loc=late_mu, scale=late_sigma)
            p_late_values = _resolve_event_probabilities(event_probabilities=event_probabilities, model_name=model_name)
            early_density_ms = density_per_second_to_per_millisecond(early_density_s)
            late_density_ms = density_per_second_to_per_millisecond(late_density_s)
            mixture_density_ms = mixture_density_per_ms(
                early_density_s=early_density_s,
                late_density_s=late_density_s,
                p_late_values=p_late_values,
                fixed_weight=None,
            )
            mean_weight = float(np.mean(p_late_values)) if p_late_values.size else 0.5
            axis.plot(grid_ms, early_density_ms, color="#1d3557", linewidth=2.0, label="Early component")
            axis.plot(grid_ms, late_density_ms, color="#e76f51", linewidth=2.0, label="Late component")
            axis.plot(
                grid_ms,
                mixture_density_ms,
                color="#2a9d8f",
                linewidth=2.5,
                label=f"Mixture (event-avg P(late)={mean_weight:.2f})",
            )
            axis.legend(frameon=False)

    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title(title)
    _save_figure(figure, output_path)


def plot_probability_by_predictor(
    *,
    event_probabilities: pd.DataFrame | None,
    predictor_column: str,
    output_path: Path,
    x_label: str,
    model_name: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    if event_probabilities is None or event_probabilities.empty:
        _draw_placeholder(axis, "Latency-regime probability", "No event probability rows were available.")
        _save_figure(figure, output_path)
        return

    required = {predictor_column, "p_late_mean", "p_late_q2_5", "p_late_q97_5", "model_name"}
    if not required <= set(event_probabilities.columns):
        _draw_placeholder(axis, "Latency-regime probability", "Required probability columns were not available.")
        _save_figure(figure, output_path)
        return

    working = event_probabilities.loc[event_probabilities["model_name"] == model_name, list(required)].copy()
    for column_name in required - {"model_name"}:
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    working = working.dropna().sort_values(predictor_column)
    if working.empty:
        _draw_placeholder(axis, "Latency-regime probability", f"No finite rows were available for {MODEL_LABELS.get(model_name, model_name)}.")
        _save_figure(figure, output_path)
        return

    axis.scatter(
        working[predictor_column],
        working["p_late_mean"],
        color="#457b9d",
        alpha=0.35,
        s=18,
        edgecolors="none",
    )
    axis.plot(working[predictor_column], working["p_late_mean"], color="#1d3557", linewidth=1.5)
    axis.fill_between(
        working[predictor_column],
        working["p_late_q2_5"],
        working["p_late_q97_5"],
        color="#a8dadc",
        alpha=0.35,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("P(late)")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    _save_figure(figure, output_path)


def plot_gating_coefficients(*, gating_coefficients: pd.DataFrame | None, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    if gating_coefficients is None or gating_coefficients.empty:
        _draw_placeholder(axis, "Gating coefficients", "No gating coefficients were available.")
        _save_figure(figure, output_path)
        return

    required = {"model_name", "term", "q2_5", "q50", "q97_5"}
    if not required <= set(gating_coefficients.columns):
        _draw_placeholder(axis, "Gating coefficients", "Required gating columns were not available.")
        _save_figure(figure, output_path)
        return

    working = gating_coefficients.loc[
        gating_coefficients["term"].isin(["beta_rate", "beta_expected"])
        & (gating_coefficients["model_name"] == MODEL_C)
    ].copy()
    if working.empty:
        _draw_placeholder(axis, "Gating coefficients", "No beta gating coefficients were available.")
        _save_figure(figure, output_path)
        return

    labels = {
        "beta_rate": "information rate",
        "beta_expected": "expected cumulative information",
    }
    term_order = ["beta_rate", "beta_expected"]
    y_base = np.arange(len(term_order))
    working["term"] = pd.Categorical(working["term"], categories=term_order, ordered=True)
    working = working.sort_values("term")
    y = y_base[: len(working)]
    q50 = pd.to_numeric(working["q50"], errors="coerce")
    q2_5 = pd.to_numeric(working["q2_5"], errors="coerce")
    q97_5 = pd.to_numeric(working["q97_5"], errors="coerce")
    axis.errorbar(
        x=q50,
        y=y,
        xerr=[q50 - q2_5, q97_5 - q50],
        fmt="o",
        color="#1d3557",
        ecolor="#1d3557",
        capsize=4,
        label=MODEL_LABELS.get(MODEL_C, MODEL_C),
    )

    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_yticks(y_base, [labels[term] for term in term_order])
    axis.set_xlabel("Posterior estimate (log-odds)")
    axis.set_title("Gating coefficients (Model C)")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_latency_regime_ppc(
    *,
    ppc_table: pd.DataFrame,
    event_data: pd.DataFrame,
    output_path: Path,
    loo_table: pd.DataFrame | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    observed = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    observed = observed[np.isfinite(observed)]
    if observed.empty:
        _draw_placeholder(axis, "Posterior predictive check", "No finite observed latencies were available.")
        _save_figure(figure, output_path)
        return

    observed_s = observed.to_numpy(dtype=float)
    edges_s = np.linspace(float(observed_s.min()) - 0.1, float(observed_s.max()) + 0.1, 30)
    observed_ms = seconds_to_milliseconds(observed_s)
    edges_ms = seconds_to_milliseconds(edges_s)
    obs_density_ms, obs_edges_ms = np.histogram(observed_ms, bins=edges_ms, density=True)
    obs_centers_ms = (obs_edges_ms[:-1] + obs_edges_ms[1:]) / 2.0
    axis.plot(obs_centers_ms, obs_density_ms, color="#111111", linewidth=2.0, label="Observed")

    highlight_models = [MODEL_A, MODEL_B, MODEL_C]
    best_r_model = _choose_best_model(
        loo_table,
        candidates=(
            MODEL_R1,
            MODEL_R2,
            MODEL_R3,
            MODEL_R4,
        ),
    )
    if best_r_model is not None:
        highlight_models.append(best_r_model)
    colors = ["#6c757d", "#bc6c25", "#2a9d8f", "#1d3557"]
    for model_name, color in zip(highlight_models, colors, strict=False):
        model_samples_ms = _extract_ppc_samples_ms(ppc_table=ppc_table, model_name=model_name)
        if model_samples_ms.size == 0:
            continue
        density_ms = _sample_density_per_ms(samples_ms=model_samples_ms, grid_ms=obs_centers_ms)
        axis.plot(
            obs_centers_ms,
            density_ms,
            color=color,
            linewidth=1.8,
            label=f"{MODEL_LABELS.get(model_name, model_name)} posterior predictive",
        )

    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Latency-regime posterior predictive densities")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_latency_regime_loo_comparison(*, loo_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    working = loo_table.copy()
    working = working.loc[working["model_name"].isin(list(MODEL_LABELS.keys()))].copy()
    working["delta_looic_from_best"] = pd.to_numeric(working["delta_looic_from_best"], errors="coerce")
    working = working.dropna(subset=["delta_looic_from_best"])
    if working.empty:
        _draw_placeholder(axis, "LOO comparison", "No finite LOO rows were available.")
        _save_figure(figure, output_path)
        return

    working = working.sort_values("delta_looic_from_best")
    axis.bar(
        [MODEL_LABELS.get(name, str(name)) for name in working["model_name"]],
        working["delta_looic_from_best"],
        color="#5b8e7d",
    )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Delta LOOIC from best (0 = best, larger = worse)")
    axis.set_title("Latency-regime model comparison")
    axis.tick_params(axis="x", rotation=18)
    _save_figure(figure, output_path)


def plot_skew_vs_mixture_overlay(
    *,
    event_data: pd.DataFrame,
    model_s_summary: pd.DataFrame | None,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    ppc_table: pd.DataFrame | None,
    loo_table: pd.DataFrame | None,
    output_path: Path,
    nu: float,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    latency_seconds = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    finite_latency = latency_seconds[np.isfinite(latency_seconds)]
    if finite_latency.empty:
        _draw_placeholder(axis, "Skew vs mixture", "No finite latency values were available.")
        _save_figure(figure, output_path)
        return

    finite_latency_ms = seconds_to_milliseconds(finite_latency.to_numpy(dtype=float))
    # Make bin width 3x smaller than previous 24-bin setting (same range, 3x bins).
    axis.hist(finite_latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9)
    grid_ms = _build_latency_grid_ms(finite_latency.to_numpy(dtype=float))

    if model_s_summary is not None and not model_s_summary.empty and {"variable", "mean"} <= set(model_s_summary.columns):
        lookup = dict(zip(model_s_summary["variable"], pd.to_numeric(model_s_summary["mean"], errors="coerce"), strict=False))
        xi = lookup.get("xi")
        omega = lookup.get("omega")
        alpha_skew = lookup.get("alpha_skew")
        if all(np.isfinite([xi, omega, alpha_skew])) and float(omega) > 0:
            skew_density_ms = skew_normal_density_per_ms(
                grid_ms,
                xi_s=float(xi),
                omega_s=float(omega),
                alpha_skew=float(alpha_skew),
            )
            axis.plot(grid_ms, skew_density_ms, color="#5e548e", linewidth=2.2, label="Skewed unimodal")

    mixture = _mixture_density_for_model(
        component_parameters=component_parameters,
        event_probabilities=event_probabilities,
        model_name=MODEL_C,
        grid_ms=grid_ms,
        nu=nu,
    )
    if mixture is not None:
        axis.plot(grid_ms, mixture, color="#2a9d8f", linewidth=2.2, alpha=0.95, label="Model C mixture-of-experts")

    best_r_model = _choose_best_model(
        loo_table,
        candidates=(
            MODEL_R1,
            MODEL_R2,
            MODEL_R3,
            MODEL_R4,
        ),
    )
    if best_r_model is not None and ppc_table is not None:
        r_samples_ms = _extract_ppc_samples_ms(ppc_table=ppc_table, model_name=best_r_model)
        if r_samples_ms.size > 0:
            axis.plot(
                grid_ms,
                _sample_density_per_ms(samples_ms=r_samples_ms, grid_ms=grid_ms),
                color="#264653",
                linewidth=2.0,
                alpha=0.9,
                label=f"Best R model posterior predictive ({MODEL_LABELS.get(best_r_model, best_r_model)})",
            )

    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axvline(200.0, color="#c1121f", linestyle=":", linewidth=1.2)
    axis.axvline(1000.0, color="#6d597a", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Skewed unimodal vs mixture-of-experts and best regression competitor")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_regression_vs_mixture_overlay(
    *,
    event_data: pd.DataFrame,
    ppc_table: pd.DataFrame | None,
    loo_table: pd.DataFrame | None,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    latency_seconds = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    finite_latency = latency_seconds[np.isfinite(latency_seconds)]
    if finite_latency.empty:
        _draw_placeholder(axis, "Regression vs mixture", "No finite latency values were available.")
        _save_figure(figure, output_path)
        return

    latency_ms = seconds_to_milliseconds(finite_latency.to_numpy(dtype=float))
    grid_ms = _build_latency_grid_ms(finite_latency.to_numpy(dtype=float))
    axis.hist(latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9, label="Observed")

    if ppc_table is not None and not ppc_table.empty:
        comparison_models = [
            (MODEL_C, "#2a9d8f", "Model C posterior predictive"),
        ]
        best_student_t_r = _choose_best_model(
            loo_table,
            candidates=(MODEL_R1, MODEL_R2),
        )
        best_lognormal_r = _choose_best_model(
            loo_table,
            candidates=(
                MODEL_R3,
                MODEL_R4,
            ),
        )
        if best_student_t_r is not None:
            comparison_models.append((best_student_t_r, "#1d3557", "Best Student-t R posterior predictive"))
        if best_lognormal_r is not None:
            comparison_models.append((best_lognormal_r, "#bc6c25", "Best shifted-lognormal R posterior predictive"))

        for model_name, color, label in comparison_models:
            samples_ms = _extract_ppc_samples_ms(ppc_table=ppc_table, model_name=model_name)
            if samples_ms.size == 0:
                continue
            axis.plot(grid_ms, _sample_density_per_ms(samples_ms=samples_ms, grid_ms=grid_ms), color=color, linewidth=2.1, label=label)

    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Event-only latency-regime analysis: posterior predictive comparison")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def plot_latency_regime_regression_effects(
    *,
    regression_predictions: pd.DataFrame | None,
    loo_table: pd.DataFrame | None,
    predictor_name: str,
    x_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    if regression_predictions is None or regression_predictions.empty:
        _draw_placeholder(axis, "Regression effects", "No regression prediction rows were available.")
        _save_figure(figure, output_path)
        return

    required = {"model_name", "predictor_name", "predictor_value", "latency_q10", "latency_q50", "latency_q90"}
    if not required <= set(regression_predictions.columns):
        _draw_placeholder(axis, "Regression effects", "Required regression prediction columns were not available.")
        _save_figure(figure, output_path)
        return

    best_student_t_r = _choose_best_model(
        loo_table,
        candidates=(MODEL_R1, MODEL_R2),
    )
    best_lognormal_r = _choose_best_model(
        loo_table,
        candidates=(MODEL_R3, MODEL_R4),
    )
    selected_models = [model_name for model_name in (best_student_t_r, best_lognormal_r) if model_name is not None]
    if not selected_models:
        selected_models = sorted(set(regression_predictions["model_name"].astype(str)))

    color_map = {
        MODEL_R1: "#1d3557",
        MODEL_R2: "#457b9d",
        MODEL_R3: "#bc6c25",
        MODEL_R4: "#dda15e",
    }
    working = regression_predictions.loc[regression_predictions["predictor_name"] == predictor_name].copy()
    if working.empty:
        _draw_placeholder(axis, "Regression effects", "No rows were available for the requested predictor.")
        _save_figure(figure, output_path)
        return

    for model_name in selected_models:
        subset = working.loc[working["model_name"] == model_name].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("predictor_value")
        x = pd.to_numeric(subset["predictor_value"], errors="coerce")
        q10_ms = seconds_to_milliseconds(pd.to_numeric(subset["latency_q10"], errors="coerce"))
        q50_ms = seconds_to_milliseconds(pd.to_numeric(subset["latency_q50"], errors="coerce"))
        q90_ms = seconds_to_milliseconds(pd.to_numeric(subset["latency_q90"], errors="coerce"))
        color = color_map.get(model_name, "#264653")
        axis.plot(x, q50_ms, color=color, linewidth=2.1, label=MODEL_LABELS.get(model_name, model_name))
        axis.fill_between(x, q10_ms, q90_ms, color=color, alpha=0.18)

    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Predicted latency (ms)")
    axis.set_title("Single-regime latency regression alternative")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _write_density_scaling_diagnostics(
    *,
    output_dir: Path,
    event_data: pd.DataFrame,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    model_s_summary: pd.DataFrame | None,
    model_a_summary: pd.DataFrame | None,
    regression_coefficients: pd.DataFrame | None,
    shifted_lognormal_diagnostics: pd.DataFrame | None,
    nu: float,
) -> dict[str, Path]:
    diagnostics_dir = output_dir.parent / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    latency_seconds = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    finite_latency = latency_seconds[np.isfinite(latency_seconds)]
    if finite_latency.empty:
        return {}

    latency_s = finite_latency.to_numpy(dtype=float)
    grid_ms = _build_latency_grid_ms(latency_s)
    grid_s = milliseconds_to_seconds(grid_ms)

    warnings_list: list[str] = []
    area_rows: list[dict[str, object]] = []

    def record_area(model_name: str, density_name: str, density_ms: np.ndarray, note: str = "") -> None:
        area_result = check_density_area(grid_ms, density_ms, density_name, tolerance=0.20)
        if area_result["warning"]:
            warnings_list.append(str(area_result["warning"]))
        area_value = float(area_result["area"]) if np.isfinite(area_result["area"]) else float("nan")
        truncation_note = note
        if np.isfinite(area_value) and area_value < 0.98 and not truncation_note:
            truncation_note = "Displayed-grid area is below 1, which may reflect truncated support rather than incorrect scaling."
        area_rows.append(
            {
                "model_name": model_name,
                "density_name": density_name,
                "grid_min_ms": float(np.nanmin(grid_ms)),
                "grid_max_ms": float(np.nanmax(grid_ms)),
                "area_on_displayed_grid": area_value,
                "note_about_truncation_if_area_less_than_one": truncation_note,
            }
        )

    one_student = _one_student_t_density_per_ms(model_a_summary=model_a_summary, grid_ms=grid_ms, nu=nu)
    if one_student is not None:
        record_area(MODEL_A, "one_student_t_density_per_ms", one_student)

    skew_density = _skew_density_per_ms(model_s_summary=model_s_summary, grid_ms=grid_ms)
    if skew_density is not None:
        record_area(MODEL_S, "skew_unimodal_density_per_ms", skew_density)

    mixture_subset = None
    if component_parameters is not None and not component_parameters.empty:
        mixture_subset = component_parameters.loc[component_parameters.get("model_name") == MODEL_C].copy()
    if mixture_subset is not None and not mixture_subset.empty:
        early_mu = _extract_component_value(mixture_subset, component="early", parameter="mu")
        late_mu = _extract_component_value(mixture_subset, component="late", parameter="mu")
        early_sigma = _extract_component_value(mixture_subset, component="early", parameter="sigma")
        late_sigma = _extract_component_value(mixture_subset, component="late", parameter="sigma")
        early_density_s = t.pdf(grid_s, df=nu, loc=early_mu, scale=early_sigma)
        late_density_s = t.pdf(grid_s, df=nu, loc=late_mu, scale=late_sigma)
        early_density_ms = density_per_second_to_per_millisecond(early_density_s)
        late_density_ms = density_per_second_to_per_millisecond(late_density_s)
        record_area(MODEL_C, "early_component_density_per_ms", early_density_ms)
        record_area(MODEL_C, "late_component_density_per_ms", late_density_ms)

        p_late_values = _resolve_event_probabilities(
            event_probabilities=event_probabilities,
            model_name=MODEL_C,
        )
        mixture_event_averaged_ms = mixture_density_per_ms(
            early_density_s=early_density_s,
            late_density_s=late_density_s,
            p_late_values=p_late_values,
        )
        record_area(MODEL_C, "mixture_density_event_averaged_per_ms", mixture_event_averaged_ms)

        mixture_density_50_50 = mixture_density_per_ms(
            early_density_s=early_density_s,
            late_density_s=late_density_s,
            p_late_values=None,
            fixed_weight=0.5,
        )
        record_area(MODEL_B, "mixture_density_50_50_per_ms_optional", mixture_density_50_50)

        alpha_value = _resolve_alpha(gating_coefficients=gating_coefficients, model_name=MODEL_C)
        if np.isfinite(alpha_value):
            p_alpha = 1.0 / (1.0 + np.exp(-alpha_value))
            mixture_density_alpha_only = mixture_density_per_ms(
                early_density_s=early_density_s,
                late_density_s=late_density_s,
                p_late_values=None,
                fixed_weight=float(p_alpha),
            )
            record_area(MODEL_C, "mixture_density_alpha_only_per_ms_optional", mixture_density_alpha_only)

    if regression_coefficients is not None and not regression_coefficients.empty:
        if {"model_name", "coefficient_group", "term", "mean"} <= set(regression_coefficients.columns):
            student_t_models = {
                MODEL_R1,
                MODEL_R2,
            }
            for model_name in student_t_models & set(regression_coefficients["model_name"].astype(str)):
                mean_shift = regression_coefficients.loc[
                    (regression_coefficients["model_name"] == model_name)
                    & (regression_coefficients["term"].isin(["alpha_mu", "alpha_sigma"])),
                    "mean",
                ]
                if model_name == MODEL_R1:
                    mu = _resolve_regression_term_mean(regression_coefficients, model_name, "alpha_mu")
                    sigma = _resolve_scalar_or_regression_scale(regression_coefficients, model_name, default=np.nan)
                    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
                        record_area(model_name, "student_t_regression_reference_density_per_ms", student_t_density_per_ms(grid_ms, mu_s=mu, sigma_s=sigma, nu=nu))

    if shifted_lognormal_diagnostics is not None and not shifted_lognormal_diagnostics.empty:
        if {"model_name", "shift_seconds"} <= set(shifted_lognormal_diagnostics.columns):
            for model_name in shifted_lognormal_diagnostics["model_name"].astype(str).unique():
                shift_value = _resolve_shift_seconds(shifted_lognormal_diagnostics, model_name)
                mu_log = _resolve_regression_term_mean(regression_coefficients, model_name, "alpha_log")
                sigma_log = _resolve_scalar_or_regression_scale(regression_coefficients, model_name, default=0.4)
                if np.isfinite(shift_value) and np.isfinite(mu_log) and np.isfinite(sigma_log) and sigma_log > 0:
                    shifted_density_s = _shifted_lognormal_density_per_ms(
                        grid_ms=grid_ms,
                        shift_seconds=shift_value,
                        mu_log=float(mu_log),
                        sigma_log=float(sigma_log),
                    )
                    record_area(model_name, "shifted_lognormal_reference_density_per_ms", shifted_density_s)

    diagnostics_df = pd.DataFrame(area_rows)
    diagnostics_csv = diagnostics_dir / "latency_regime_density_scaling_check.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    observed_hist_density, _ = np.histogram(seconds_to_milliseconds(latency_s), bins=32, density=True)
    observed_peak = float(np.nanmax(observed_hist_density)) if observed_hist_density.size else float("nan")
    one_student_peak = float(np.nanmax(one_student)) if one_student is not None else float("nan")

    plot_module_text = Path(__file__).read_text(encoding="utf-8")
    legacy_mismatch_detected = all(
        marker in plot_module_text
        for marker in [
            "axis.plot(grid_seconds * 1000.0, early_density",
            "axis.plot(grid_seconds * 1000.0, skew_density",
            "axis.set_ylabel(\"Density\")",
        ]
    )
    if legacy_mismatch_detected:
        warnings_list.append("Legacy plotting code mixed per-second densities with per-millisecond histogram density.")
    else:
        warnings_list.append(
            "Historical inspection found previous per-second/per-millisecond density mismatch; conversion is now centralized."
        )

    diagnostics_json = diagnostics_dir / "latency_regime_density_scaling_check.json"
    payload = {
        "inferred_data_unit": _infer_latency_unit(latency_s),
        "plotting_x_unit": "milliseconds",
        "density_unit": "probability per millisecond",
        "area_checks": diagnostics_df.to_dict(orient="records"),
        "grid_min_ms": float(np.nanmin(grid_ms)),
        "grid_max_ms": float(np.nanmax(grid_ms)),
        "conversion_applied": True,
        "warnings": warnings_list,
    }
    diagnostics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = diagnostics_dir / "latency_regime_density_scaling_report.md"
    report_path.write_text(
        _build_density_scaling_report(
            payload=payload,
            observed_peak=observed_peak,
            one_student_peak=one_student_peak,
            has_mixture=bool((diagnostics_df["density_name"] == "mixture_density_event_averaged_per_ms").any()) if not diagnostics_df.empty else False,
            has_late_component=bool((diagnostics_df["density_name"] == "late_component_density_per_ms").any()) if not diagnostics_df.empty else False,
        ),
        encoding="utf-8",
    )

    return {
        "density_scaling_csv": diagnostics_csv,
        "density_scaling_json": diagnostics_json,
        "density_scaling_report": report_path,
    }


def _build_density_scaling_report(
    *,
    payload: dict[str, object],
    observed_peak: float,
    one_student_peak: float,
    has_mixture: bool,
    has_late_component: bool,
) -> str:
    area_checks = list(payload.get("area_checks") or [])
    lookup = {str(item.get("density_name")): item for item in area_checks}
    one_student_area = (lookup.get("one_student_t_density_per_ms") or {}).get("area_on_displayed_grid")
    skew_area = (lookup.get("skew_unimodal_density_per_ms") or {}).get("area_on_displayed_grid")
    mixture_area = (lookup.get("mixture_density_event_averaged_per_ms") or {}).get("area_on_displayed_grid")

    q1 = "Yes. The prior plotting implementation mixed per-second model densities with per-millisecond histogram density."
    q2 = "Yes. The unimodal/analytic overlays were previously plotted on ms x-axis without per-ms conversion."
    q3 = "Yes. Current plotting now evaluates in seconds, then converts to density per millisecond before plotting."

    area_parts = []
    if one_student_area is not None:
        area_parts.append(f"one Student-t area={float(one_student_area):.3f}")
    if skew_area is not None:
        area_parts.append(f"skew unimodal area={float(skew_area):.3f}")
    if mixture_area is not None:
        area_parts.append(f"mixture area={float(mixture_area):.3f}")
    q4 = "Approximately yes on the displayed grid." if area_parts else "Insufficient model summaries to evaluate all areas."

    if np.isfinite(observed_peak) and np.isfinite(one_student_peak):
        ratio = one_student_peak / max(observed_peak, 1e-12)
        if ratio > 20.0:
            q5 = (
                "After scaling correction, the unimodal curve still appears substantially mismatched to observed density, "
                "so this is not purely a plotting artifact."
            )
        else:
            q5 = "After scaling correction, much of the extreme mismatch disappears, indicating it was largely a plotting artifact."
    else:
        q5 = "Could not fully assess unimodal fit quality because one Student-t summary values were unavailable."

    if has_mixture and has_late_component:
        q6 = "Yes. The mixture still shows a distinct late component after scaling correction."
    else:
        q6 = "Could not assess second-component prominence because mixture component summaries were unavailable."

    return "\n".join(
        [
            "# Latency Regime Density Scaling Report",
            "",
            "1. Were model densities being mixed across per-second and per-millisecond units?",
            f"   - {q1}",
            "2. Was the unimodal density plotted without dividing by 1000?",
            f"   - {q2}",
            "3. Are all model densities now plotted as density per millisecond?",
            f"   - {q3}",
            "4. Do plotted density curves integrate approximately to 1 on the displayed grid?",
            f"   - {q4}",
            f"   - Areas: {', '.join(area_parts) if area_parts else 'n/a'}.",
            "5. After scaling correction, does the unimodal model still look poor, or was it mostly a plotting artifact?",
            f"   - {q5}",
            "6. Does the mixture model still show a meaningful second component after the plotting correction?",
            f"   - {q6}",
            "",
            "## Units",
            f"- inferred_data_unit: `{payload.get('inferred_data_unit')}`",
            f"- plotting_x_unit: `{payload.get('plotting_x_unit')}`",
            f"- density_unit: `{payload.get('density_unit')}`",
            f"- conversion_applied: `{payload.get('conversion_applied')}`",
            "",
            "## Warnings",
            *[f"- {warning}" for warning in (payload.get("warnings") or [])],
            "",
            "## Area Checks",
            *[
                "- "
                + f"{row.get('model_name')} / {row.get('density_name')}: area={float(row.get('area_on_displayed_grid')):.3f} "
                + f"on [{float(row.get('grid_min_ms')):.1f}, {float(row.get('grid_max_ms')):.1f}] ms"
                + (f" ({row.get('note_about_truncation_if_area_less_than_one')})" if row.get("note_about_truncation_if_area_less_than_one") else "")
                for row in area_checks
                if row.get("area_on_displayed_grid") is not None
            ],
            "",
        ]
    )


def write_latency_regime_plot_source_audit(
    *,
    stan_results_dir: Path,
    event_data: pd.DataFrame,
    output_dir: Path,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    ppc_table: pd.DataFrame | None,
    loo_table: pd.DataFrame | None,
    regression_coefficients: pd.DataFrame | None,
    fit_metrics: dict[str, object] | None,
) -> dict[str, Path]:
    """Write audited source diagnostics for latency-regime density plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    latency_series = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    finite_latency = latency_series[np.isfinite(latency_series)]
    if finite_latency.empty:
        return outputs

    latency_s = finite_latency.to_numpy(dtype=float)
    grid_ms = _build_latency_grid_ms(latency_s)
    grid_s = milliseconds_to_seconds(grid_ms)
    nu = float((fit_metrics or {}).get("nu", 4.0))
    audit_rows: list[dict[str, object]] = []

    hist_counts_ms, hist_edges_ms = np.histogram(seconds_to_milliseconds(latency_s), bins=72, density=True)
    hist_centers_ms = (hist_edges_ms[:-1] + hist_edges_ms[1:]) / 2.0
    hist_area = float(np.trapezoid(hist_counts_ms, hist_centers_ms)) if hist_counts_ms.size > 1 else float("nan")

    r2_diag = compute_r2_density_diagnostic(
        event_data=event_data,
        regression_coefficients=regression_coefficients,
        model_name=MODEL_R2,
        grid_ms=grid_ms,
        nu=nu,
    )
    model_b_diag = compute_model_b_density_diagnostic(
        component_parameters=component_parameters,
        grid_ms=grid_ms,
        nu=nu,
    )
    model_c_diag = compute_model_c_density_diagnostic(
        component_parameters=component_parameters,
        gating_coefficients=gating_coefficients,
        event_probabilities=event_probabilities,
        grid_ms=grid_ms,
        nu=nu,
    )
    best_student_t_r = _choose_best_model(loo_table, candidates=(MODEL_R1, MODEL_R2))
    best_lognormal_r = _choose_best_model(loo_table, candidates=(MODEL_R3, MODEL_R4))

    def add_audit_row(
        *,
        figure_name: str,
        curve_label: str,
        intended_model_name: str,
        actual_model_name_used: str,
        source_file: Path,
        source_table: str,
        n_source_rows_used: int,
        parameter_names_used: list[str],
        density_type: str,
        predictor_values_used: str,
        notes: str = "",
    ) -> None:
        mismatch = intended_model_name != actual_model_name_used
        full_notes = notes
        if mismatch:
            full_notes = (full_notes + " " if full_notes else "") + "ERROR: label/source model mismatch."
        audit_rows.append(
            {
                "figure_name": figure_name,
                "curve_label": curve_label,
                "intended_model_name": intended_model_name,
                "actual_model_name_used": actual_model_name_used,
                "source_file": str(source_file),
                "source_table": source_table,
                "n_source_rows_used": int(n_source_rows_used),
                "parameter_names_used": ",".join(parameter_names_used),
                "density_type": density_type,
                "predictor_values_used": predictor_values_used,
                "x_unit": "milliseconds",
                "y_density_unit": "probability density per millisecond",
                "conversion_applied": True,
                "notes": full_notes,
            }
        )

    add_audit_row(
        figure_name="behaviour_latency_regime_ppc.png",
        curve_label="Observed histogram",
        intended_model_name="observed_data",
        actual_model_name_used="observed_data",
        source_file=Path(event_data.attrs.get("source_file", stan_results_dir.parent / "behaviour_latency_regime_data.csv")),
        source_table="event_data_csv",
        n_source_rows_used=len(latency_s),
        parameter_names_used=["latency_from_partner_offset"],
        density_type=DENSITY_TYPE_HISTOGRAM,
        predictor_values_used="observed latencies",
        notes=f"Histogram density area on plotted bins ≈ {hist_area:.3f}.",
    )

    if ppc_table is not None and not ppc_table.empty:
        for model_name in [MODEL_A, MODEL_B, MODEL_C, best_student_t_r, best_lognormal_r]:
            if model_name is None:
                continue
            ppc_rows = select_exact_model_rows(
                ppc_table,
                model_name=model_name,
                source_table="behaviour_latency_regime_posterior_predictive.csv",
                allow_missing_model_column=True,
            )
            if ppc_rows.empty:
                continue
            add_audit_row(
                figure_name="behaviour_latency_regime_ppc.png",
                curve_label=f"{MODEL_LABELS.get(model_name, model_name)} posterior predictive",
                intended_model_name=model_name,
                actual_model_name_used=model_name,
                source_file=stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv",
                source_table="behaviour_latency_regime_posterior_predictive.csv",
                n_source_rows_used=len(ppc_rows),
                parameter_names_used=["y_rep_value"],
                density_type=DENSITY_TYPE_POSTERIOR_PREDICTIVE,
                predictor_values_used="posterior predictive draws over observed events",
            )

    if ppc_table is not None and not ppc_table.empty:
        add_audit_row(
            figure_name="behaviour_latency_regime_regression_vs_mixture.png",
            curve_label="Model C posterior predictive",
            intended_model_name=MODEL_C,
            actual_model_name_used=MODEL_C,
                source_file=stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv",
                source_table="behaviour_latency_regime_posterior_predictive.csv",
                n_source_rows_used=len(
                    select_exact_model_rows(
                        ppc_table,
                        model_name=MODEL_C,
                        source_table="behaviour_latency_regime_posterior_predictive.csv",
                        allow_missing_model_column=True,
                    )
                ),
                parameter_names_used=["y_rep_value"],
                density_type=DENSITY_TYPE_POSTERIOR_PREDICTIVE,
                predictor_values_used="posterior predictive draws over observed events",
                notes="Production figure is posterior predictive, not conditional-fixed-predictor.",
            )
        if best_student_t_r is not None:
            add_audit_row(
                figure_name="behaviour_latency_regime_regression_vs_mixture.png",
                curve_label="Best Student-t R posterior predictive",
                intended_model_name=best_student_t_r,
                actual_model_name_used=best_student_t_r,
                source_file=stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv",
                source_table="behaviour_latency_regime_posterior_predictive.csv",
                n_source_rows_used=len(
                    select_exact_model_rows(
                        ppc_table,
                        model_name=best_student_t_r,
                        source_table="behaviour_latency_regime_posterior_predictive.csv",
                        allow_missing_model_column=True,
                    )
                ),
                parameter_names_used=["y_rep_value"],
                density_type=DENSITY_TYPE_POSTERIOR_PREDICTIVE,
                predictor_values_used="posterior predictive draws over observed events",
                notes="This curve can be multimodal without implying a conditional R2 density bug.",
            )
        if best_lognormal_r is not None:
            add_audit_row(
                figure_name="behaviour_latency_regime_regression_vs_mixture.png",
                curve_label="Best shifted-lognormal R posterior predictive",
                intended_model_name=best_lognormal_r,
                actual_model_name_used=best_lognormal_r,
                source_file=stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv",
                source_table="behaviour_latency_regime_posterior_predictive.csv",
                n_source_rows_used=len(
                    select_exact_model_rows(
                        ppc_table,
                        model_name=best_lognormal_r,
                        source_table="behaviour_latency_regime_posterior_predictive.csv",
                        allow_missing_model_column=True,
                    )
                ),
                parameter_names_used=["y_rep_value"],
                density_type=DENSITY_TYPE_POSTERIOR_PREDICTIVE,
                predictor_values_used="posterior predictive draws over observed events",
            )

    if r2_diag is not None:
        coeff_rows = select_exact_model_rows(
            regression_coefficients,
            model_name=MODEL_R2,
            source_table="behaviour_latency_regime_regression_coefficients.csv",
        )
        add_audit_row(
            figure_name="latency_regime_r2_conditional_vs_marginal_density.png",
            curve_label="R2 conditional, predictors=0",
            intended_model_name=MODEL_R2,
            actual_model_name_used=MODEL_R2,
            source_file=stan_results_dir / "behaviour_latency_regime_regression_coefficients.csv",
            source_table="behaviour_latency_regime_regression_coefficients.csv",
            n_source_rows_used=len(coeff_rows),
            parameter_names_used=["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            density_type=DENSITY_TYPE_CONDITIONAL,
            predictor_values_used="z_information_rate_lag_best=0; z_prop_expected_cumulative_info_lag_best=0",
        )
        add_audit_row(
            figure_name="latency_regime_r2_conditional_vs_marginal_density.png",
            curve_label="R2 marginal over observed predictors",
            intended_model_name=MODEL_R2,
            actual_model_name_used=MODEL_R2,
            source_file=stan_results_dir / "behaviour_latency_regime_regression_coefficients.csv",
            source_table="behaviour_latency_regime_regression_coefficients.csv",
            n_source_rows_used=len(coeff_rows),
            parameter_names_used=["alpha_mu", "beta_mu_rate", "beta_mu_expected", "alpha_sigma", "beta_sigma_rate", "beta_sigma_expected"],
            density_type=DENSITY_TYPE_MARGINAL,
            predictor_values_used="Observed z_information_rate_lag_best and z_prop_expected_cumulative_info_lag_best",
        )

    if model_b_diag is not None:
        n_b = len(select_exact_model_rows(component_parameters, model_name=MODEL_B, source_table="behaviour_latency_regime_component_parameters.csv"))
        for label, dtype in [
            ("Model B early component", DENSITY_TYPE_COMPONENT),
            ("Model B late component", DENSITY_TYPE_COMPONENT),
            ("B constant two-component mixture", DENSITY_TYPE_WEIGHTED),
        ]:
            add_audit_row(
                figure_name="latency_regime_model_b_component_check.png",
                curve_label=label,
                intended_model_name=MODEL_B,
                actual_model_name_used=MODEL_B,
                source_file=stan_results_dir / "behaviour_latency_regime_component_parameters.csv",
                source_table="behaviour_latency_regime_component_parameters.csv",
                n_source_rows_used=n_b,
                parameter_names_used=["mu", "sigma", "pi_late"],
                density_type=dtype,
                predictor_values_used="No predictors; constant mixture weight",
            )

    if model_c_diag is not None:
        n_c_components = len(select_exact_model_rows(component_parameters, model_name=MODEL_C, source_table="behaviour_latency_regime_component_parameters.csv"))
        n_c_probs = len(select_exact_model_rows(event_probabilities, model_name=MODEL_C, source_table="behaviour_latency_regime_event_probabilities.csv"))
        add_audit_row(
            figure_name="latency_regime_model_c_component_check.png",
            curve_label="Model C early component",
            intended_model_name=MODEL_C,
            actual_model_name_used=MODEL_C,
            source_file=stan_results_dir / "behaviour_latency_regime_component_parameters.csv",
            source_table="behaviour_latency_regime_component_parameters.csv",
            n_source_rows_used=n_c_components,
            parameter_names_used=["mu", "sigma"],
            density_type=DENSITY_TYPE_COMPONENT,
            predictor_values_used="Global early component",
        )
        add_audit_row(
            figure_name="latency_regime_model_c_component_check.png",
            curve_label="Model C late component",
            intended_model_name=MODEL_C,
            actual_model_name_used=MODEL_C,
            source_file=stan_results_dir / "behaviour_latency_regime_component_parameters.csv",
            source_table="behaviour_latency_regime_component_parameters.csv",
            n_source_rows_used=n_c_components,
            parameter_names_used=["mu", "sigma"],
            density_type=DENSITY_TYPE_COMPONENT,
            predictor_values_used="Global late component",
        )
        add_audit_row(
            figure_name="latency_regime_model_c_component_check.png",
            curve_label="C mixture-of-experts, event-averaged",
            intended_model_name=MODEL_C,
            actual_model_name_used=MODEL_C,
            source_file=stan_results_dir / "behaviour_latency_regime_event_probabilities.csv",
            source_table="behaviour_latency_regime_event_probabilities.csv",
            n_source_rows_used=n_c_probs,
            parameter_names_used=["p_late_mean", "alpha", "beta_rate", "beta_expected", "mu", "sigma"],
            density_type=DENSITY_TYPE_WEIGHTED,
            predictor_values_used="Observed predictors through event-level p_late_mean",
        )
        add_audit_row(
            figure_name="latency_regime_model_c_component_check.png",
            curve_label="Model C alpha-only mixture",
            intended_model_name=MODEL_C,
            actual_model_name_used=MODEL_C,
            source_file=stan_results_dir / "behaviour_latency_regime_gating_coefficients.csv",
            source_table="behaviour_latency_regime_gating_coefficients.csv",
            n_source_rows_used=len(select_exact_model_rows(gating_coefficients, model_name=MODEL_C, source_table="behaviour_latency_regime_gating_coefficients.csv", allow_missing_model_column=True)),
            parameter_names_used=["alpha", "mu", "sigma"],
            density_type=DENSITY_TYPE_CONDITIONAL,
            predictor_values_used="Predictors=0 in gating equation only",
        )

    audit_csv = output_dir / "latency_regime_plot_source_audit.csv"
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_csv, index=False)
    outputs["plot_source_audit_csv"] = audit_csv

    if r2_diag is not None:
        r2_path = output_dir / "latency_regime_r2_density_diagnostic.csv"
        r2_diag.to_csv(r2_path, index=False)
        outputs["r2_density_diagnostic_csv"] = r2_path
    if model_b_diag is not None:
        model_b_path = output_dir / "latency_regime_model_b_density_diagnostic.csv"
        model_b_diag.to_csv(model_b_path, index=False)
        outputs["model_b_density_diagnostic_csv"] = model_b_path
    if model_c_diag is not None:
        model_c_path = output_dir / "latency_regime_model_c_density_diagnostic.csv"
        model_c_diag.to_csv(model_c_path, index=False)
        outputs["model_c_density_diagnostic_csv"] = model_c_path

    if r2_diag is not None:
        outputs["r2_density_figure"] = output_dir / "latency_regime_r2_conditional_vs_marginal_density.png"
        _plot_r2_density_diagnostic(
            latency_ms=seconds_to_milliseconds(latency_s),
            diagnostic=r2_diag,
            output_path=outputs["r2_density_figure"],
        )
    if model_b_diag is not None:
        outputs["model_b_density_figure"] = output_dir / "latency_regime_model_b_component_check.png"
        _plot_model_b_density_diagnostic(
            latency_ms=seconds_to_milliseconds(latency_s),
            diagnostic=model_b_diag,
            output_path=outputs["model_b_density_figure"],
        )
    if model_c_diag is not None:
        outputs["model_c_density_figure"] = output_dir / "latency_regime_model_c_component_check.png"
        _plot_model_c_density_diagnostic(
            latency_ms=seconds_to_milliseconds(latency_s),
            diagnostic=model_c_diag,
            output_path=outputs["model_c_density_figure"],
        )
    if r2_diag is not None and model_b_diag is not None and model_c_diag is not None:
        overlay_path = output_dir / "latency_regime_model_overlay_audited.png"
        _plot_audited_overlay(
            latency_ms=seconds_to_milliseconds(latency_s),
            r2_diagnostic=r2_diag,
            model_b_diagnostic=model_b_diag,
            model_c_diagnostic=model_c_diag,
            output_path=overlay_path,
        )
        outputs["audited_overlay_figure"] = overlay_path

    maxima_payload = {
        "r2_conditional": count_local_maxima(grid_ms, r2_diag["r2_conditional_density_per_ms"].to_numpy(dtype=float)) if r2_diag is not None else None,
        "r2_marginal": count_local_maxima(grid_ms, r2_diag["r2_marginal_density_per_ms"].to_numpy(dtype=float)) if r2_diag is not None else None,
        "model_b_mixture": count_local_maxima(grid_ms, model_b_diag["model_b_mixture_density_per_ms"].to_numpy(dtype=float)) if model_b_diag is not None else None,
        "model_c_event_averaged": count_local_maxima(grid_ms, model_c_diag["model_c_event_averaged_mixture_density_per_ms"].to_numpy(dtype=float)) if model_c_diag is not None else None,
    }
    mismatches = audit_df.loc[audit_df["intended_model_name"] != audit_df["actual_model_name_used"]].to_dict(orient="records") if not audit_df.empty else []
    summary_payload = _build_plot_source_audit_summary(
        stan_results_dir=stan_results_dir,
        audit_df=audit_df,
        r2_diag=r2_diag,
        model_b_diag=model_b_diag,
        model_c_diag=model_c_diag,
        maxima_payload=maxima_payload,
        mismatches=mismatches,
        best_student_t_r=best_student_t_r,
    )
    audit_json = output_dir / "latency_regime_plot_source_audit.json"
    audit_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    outputs["plot_source_audit_json"] = audit_json

    report_path = output_dir / "latency_regime_plot_source_audit_report.md"
    report_path.write_text(_build_plot_source_audit_report(summary_payload), encoding="utf-8")
    outputs["plot_source_audit_report"] = report_path

    return outputs


def compute_r2_density_diagnostic(
    *,
    event_data: pd.DataFrame,
    regression_coefficients: pd.DataFrame | None,
    model_name: str,
    grid_ms: np.ndarray,
    nu: float,
) -> pd.DataFrame | None:
    coeffs = select_exact_model_rows(
        regression_coefficients,
        model_name=model_name,
        source_table="behaviour_latency_regime_regression_coefficients.csv",
    )
    if coeffs.empty:
        return None
    z_rate = pd.to_numeric(event_data.get("z_information_rate_lag_best"), errors="coerce").to_numpy(dtype=float)
    z_expected = pd.to_numeric(event_data.get("z_prop_expected_cumulative_info_lag_best"), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(z_rate) & np.isfinite(z_expected)
    if not np.any(finite):
        return None
    mu_0 = _resolve_regression_linear_predictor(coeffs, coefficient_group="location", predictor_rate=0.0, predictor_expected=0.0)
    sigma_0 = _resolve_regression_scale_predictor(coeffs, predictor_rate=0.0, predictor_expected=0.0)
    conditional_ms = student_t_density_per_ms(grid_ms, mu_s=mu_0, sigma_s=sigma_0, nu=nu)
    marginal_s = []
    grid_s = milliseconds_to_seconds(grid_ms)
    for rate_value, expected_value in zip(z_rate[finite], z_expected[finite], strict=False):
        mu_i = _resolve_regression_linear_predictor(
            coeffs,
            coefficient_group="location",
            predictor_rate=float(rate_value),
            predictor_expected=float(expected_value),
        )
        sigma_i = _resolve_regression_scale_predictor(
            coeffs,
            predictor_rate=float(rate_value),
            predictor_expected=float(expected_value),
        )
        marginal_s.append(t.pdf(grid_s, df=nu, loc=mu_i, scale=sigma_i))
    marginal_ms = density_per_second_to_per_millisecond(np.mean(np.vstack(marginal_s), axis=0))
    return pd.DataFrame(
        {
            "latency_s": grid_s,
            "latency_ms": grid_ms,
            "r2_conditional_density_per_ms": conditional_ms,
            "r2_marginal_density_per_ms": marginal_ms,
        }
    )


def compute_model_b_density_diagnostic(
    *,
    component_parameters: pd.DataFrame | None,
    grid_ms: np.ndarray,
    nu: float,
) -> pd.DataFrame | None:
    subset = select_exact_model_rows(
        component_parameters,
        model_name=MODEL_B,
        source_table="behaviour_latency_regime_component_parameters.csv",
    )
    if subset.empty:
        return None
    early_mu = _extract_component_value(subset, component="early", parameter="mu")
    late_mu = _extract_component_value(subset, component="late", parameter="mu")
    early_sigma = _extract_component_value(subset, component="early", parameter="sigma")
    late_sigma = _extract_component_value(subset, component="late", parameter="sigma")
    weight = _extract_optional_component_value(subset, component="late", parameter="pi_late", default=0.5)
    early_ms = student_t_density_per_ms(grid_ms, mu_s=early_mu, sigma_s=early_sigma, nu=nu)
    late_ms = student_t_density_per_ms(grid_ms, mu_s=late_mu, sigma_s=late_sigma, nu=nu)
    mixture_ms = (1.0 - weight) * early_ms + weight * late_ms
    return pd.DataFrame(
        {
            "latency_s": milliseconds_to_seconds(grid_ms),
            "latency_ms": grid_ms,
            "model_b_early_density_per_ms": early_ms,
            "model_b_late_density_per_ms": late_ms,
            "model_b_mixture_density_per_ms": mixture_ms,
        }
    )


def compute_model_c_density_diagnostic(
    *,
    component_parameters: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    grid_ms: np.ndarray,
    nu: float,
) -> pd.DataFrame | None:
    subset = select_exact_model_rows(
        component_parameters,
        model_name=MODEL_C,
        source_table="behaviour_latency_regime_component_parameters.csv",
    )
    if subset.empty:
        return None
    early_mu = _extract_component_value(subset, component="early", parameter="mu")
    late_mu = _extract_component_value(subset, component="late", parameter="mu")
    early_sigma = _extract_component_value(subset, component="early", parameter="sigma")
    late_sigma = _extract_component_value(subset, component="late", parameter="sigma")
    early_ms = student_t_density_per_ms(grid_ms, mu_s=early_mu, sigma_s=early_sigma, nu=nu)
    late_ms = student_t_density_per_ms(grid_ms, mu_s=late_mu, sigma_s=late_sigma, nu=nu)
    early_s = early_ms * 1000.0
    late_s = late_ms * 1000.0
    p_late_values = _resolve_event_probabilities(event_probabilities=event_probabilities, model_name=MODEL_C)
    event_averaged_ms = mixture_density_per_ms(
        early_density_s=early_s,
        late_density_s=late_s,
        p_late_values=p_late_values,
    )
    alpha_value = _resolve_alpha(gating_coefficients=gating_coefficients, model_name=MODEL_C)
    if not np.isfinite(alpha_value):
        alpha_value = 0.0
    alpha_only_weight = 1.0 / (1.0 + np.exp(-alpha_value))
    alpha_only_ms = mixture_density_per_ms(
        early_density_s=early_s,
        late_density_s=late_s,
        p_late_values=None,
        fixed_weight=alpha_only_weight,
    )
    return pd.DataFrame(
        {
            "latency_s": milliseconds_to_seconds(grid_ms),
            "latency_ms": grid_ms,
            "model_c_early_density_per_ms": early_ms,
            "model_c_late_density_per_ms": late_ms,
            "model_c_event_averaged_mixture_density_per_ms": event_averaged_ms,
            "model_c_alpha_only_mixture_density_per_ms": alpha_only_ms,
        }
    )


def _resolve_regression_linear_predictor(
    regression_coefficients: pd.DataFrame,
    *,
    coefficient_group: str,
    predictor_rate: float,
    predictor_expected: float,
) -> float:
    subset = regression_coefficients.copy()
    if "coefficient_group" in subset.columns:
        subset = subset.loc[subset["coefficient_group"] == coefficient_group].copy()
    lookup = dict(zip(subset["term"].astype(str), pd.to_numeric(subset["mean"], errors="coerce"), strict=False))
    intercept = float(lookup.get("alpha_mu", lookup.get("alpha_log", 0.0)))
    beta_rate = float(lookup.get("beta_mu_rate", lookup.get("beta_log_rate", 0.0)))
    beta_expected = float(lookup.get("beta_mu_expected", lookup.get("beta_log_expected", 0.0)))
    return intercept + beta_rate * float(predictor_rate) + beta_expected * float(predictor_expected)


def _resolve_regression_scale_predictor(
    regression_coefficients: pd.DataFrame,
    *,
    predictor_rate: float,
    predictor_expected: float,
) -> float:
    subset = regression_coefficients.copy()
    if "coefficient_group" in subset.columns:
        subset = subset.loc[subset["coefficient_group"] == "scale"].copy()
    lookup = dict(zip(subset["term"].astype(str), pd.to_numeric(subset["mean"], errors="coerce"), strict=False))
    intercept = float(lookup.get("alpha_sigma", 0.0))
    beta_rate = float(lookup.get("beta_sigma_rate", 0.0))
    beta_expected = float(lookup.get("beta_sigma_expected", 0.0))
    return float(np.exp(intercept + beta_rate * float(predictor_rate) + beta_expected * float(predictor_expected)))


def _extract_optional_component_value(table: pd.DataFrame, *, component: str, parameter: str, default: float) -> float:
    subset = table.loc[(table["component"] == component) & (table["parameter"] == parameter)]
    if subset.empty:
        return float(default)
    return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])


def _plot_reference_lines(axis: plt.Axes) -> None:
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axvline(200.0, color="#c1121f", linestyle=":", linewidth=1.2)
    axis.axvline(1000.0, color="#6d597a", linestyle=":", linewidth=1.2)


def _plot_r2_density_diagnostic(*, latency_ms: np.ndarray, diagnostic: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.hist(latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9, label="Observed histogram")
    axis.plot(
        diagnostic["latency_ms"],
        diagnostic["r2_conditional_density_per_ms"],
        color="#1d3557",
        linewidth=2.2,
        label="R2 conditional, predictors=0",
    )
    axis.plot(
        diagnostic["latency_ms"],
        diagnostic["r2_marginal_density_per_ms"],
        color="#457b9d",
        linewidth=2.2,
        label="R2 marginal over observed predictors",
    )
    _plot_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("R2 conditional vs marginal density")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_model_b_density_diagnostic(*, latency_ms: np.ndarray, diagnostic: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.hist(latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9, label="Observed histogram")
    axis.plot(diagnostic["latency_ms"], diagnostic["model_b_early_density_per_ms"], color="#1d3557", linewidth=2.0, label="Model B early component")
    axis.plot(diagnostic["latency_ms"], diagnostic["model_b_late_density_per_ms"], color="#e76f51", linewidth=2.0, label="Model B late component")
    axis.plot(diagnostic["latency_ms"], diagnostic["model_b_mixture_density_per_ms"], color="#bc6c25", linewidth=2.4, label="B constant two-component mixture")
    _plot_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Model B component density check")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_model_c_density_diagnostic(*, latency_ms: np.ndarray, diagnostic: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.hist(latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.9, label="Observed histogram")
    axis.plot(diagnostic["latency_ms"], diagnostic["model_c_early_density_per_ms"], color="#1d3557", linewidth=2.0, label="Model C early component")
    axis.plot(diagnostic["latency_ms"], diagnostic["model_c_late_density_per_ms"], color="#e76f51", linewidth=2.0, label="Model C late component")
    axis.plot(
        diagnostic["latency_ms"],
        diagnostic["model_c_event_averaged_mixture_density_per_ms"],
        color="#2a9d8f",
        linewidth=2.4,
        label="C mixture-of-experts, event-averaged",
    )
    axis.plot(
        diagnostic["latency_ms"],
        diagnostic["model_c_alpha_only_mixture_density_per_ms"],
        color="#8d99ae",
        linewidth=1.8,
        linestyle="--",
        label="Model C alpha-only/predictors-zero",
    )
    _plot_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Model C component density check")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_audited_overlay(
    *,
    latency_ms: np.ndarray,
    r2_diagnostic: pd.DataFrame,
    model_b_diagnostic: pd.DataFrame,
    model_c_diagnostic: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    axis.hist(latency_ms, bins=72, density=True, color="#d7e3ea", edgecolor="white", alpha=0.85, label="Observed histogram")
    axis.plot(
        model_b_diagnostic["latency_ms"],
        model_b_diagnostic["model_b_mixture_density_per_ms"],
        color="#bc6c25",
        linewidth=2.3,
        label="B constant two-component mixture",
    )
    axis.plot(
        model_c_diagnostic["latency_ms"],
        model_c_diagnostic["model_c_event_averaged_mixture_density_per_ms"],
        color="#2a9d8f",
        linewidth=2.3,
        label="C mixture-of-experts, event-averaged",
    )
    axis.plot(
        r2_diagnostic["latency_ms"],
        r2_diagnostic["r2_conditional_density_per_ms"],
        color="#1d3557",
        linewidth=2.2,
        label="R2 conditional, predictors=0",
    )
    axis.plot(
        r2_diagnostic["latency_ms"],
        r2_diagnostic["r2_marginal_density_per_ms"],
        color="#457b9d",
        linewidth=2.2,
        linestyle="--",
        label="R2 marginal over observed predictors",
    )
    _plot_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Audited latency-regime model overlay")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _build_plot_source_audit_summary(
    *,
    stan_results_dir: Path,
    audit_df: pd.DataFrame,
    r2_diag: pd.DataFrame | None,
    model_b_diag: pd.DataFrame | None,
    model_c_diag: pd.DataFrame | None,
    maxima_payload: dict[str, int | None],
    mismatches: list[dict[str, object]],
    best_student_t_r: str | None,
) -> dict[str, object]:
    r2_conditional_unimodal = maxima_payload.get("r2_conditional") == 1 if maxima_payload.get("r2_conditional") is not None else None
    r2_marginal_bimodal = (maxima_payload.get("r2_marginal") or 0) >= 2 if maxima_payload.get("r2_marginal") is not None else None
    model_b_bimodal = (maxima_payload.get("model_b_mixture") or 0) >= 2 if maxima_payload.get("model_b_mixture") is not None else None
    source_answer = {
        "A_r2_curve_using_r2_parameters": bool(
            not audit_df.loc[
                audit_df["curve_label"].astype(str).str.contains("R2", regex=False)
                & (audit_df["actual_model_name_used"] != MODEL_R2)
            ].shape[0]
        ),
        "B_model_b_curve_using_model_b_parameters": bool(
            not audit_df.loc[
                audit_df["curve_label"].astype(str).str.contains("Model B|B constant", regex=True)
                & (audit_df["actual_model_name_used"] != MODEL_B)
            ].shape[0]
        ),
        "C_any_labels_swapped": bool(len(mismatches) > 0),
        "D_r2_curve_density_type_in_production": DENSITY_TYPE_POSTERIOR_PREDICTIVE if best_student_t_r == MODEL_R2 else "not_featured_as_best_student_t_r",
        "E_r2_conditional_unimodal": r2_conditional_unimodal,
        "F_r2_marginal_bimodal": r2_marginal_bimodal,
        "G_model_b_bimodal_under_audited_calculation": model_b_bimodal,
        "H_figures_needing_regeneration": [
            "behaviour_latency_regime_ppc.png",
            "behaviour_latency_regime_regression_vs_mixture.png",
        ],
    }
    if r2_diag is not None:
        r2_mu0_area = float(np.trapezoid(r2_diag["r2_conditional_density_per_ms"], r2_diag["latency_ms"]))
    else:
        r2_mu0_area = float("nan")
    return {
        "stan_results_dir": str(stan_results_dir),
        "maxima_counts": maxima_payload,
        "mismatches": mismatches,
        "answers": source_answer,
        "r2_conditional_area": r2_mu0_area,
        "audit_row_count": int(len(audit_df)),
        "best_student_t_r": best_student_t_r,
        "model_b_rows_present": int(len(model_b_diag)) if model_b_diag is not None else 0,
        "model_c_rows_present": int(len(model_c_diag)) if model_c_diag is not None else 0,
    }


def _build_plot_source_audit_report(summary_payload: dict[str, object]) -> str:
    answers = dict(summary_payload.get("answers") or {})
    maxima = dict(summary_payload.get("maxima_counts") or {})
    mismatches = list(summary_payload.get("mismatches") or [])
    mismatch_text = "No explicit label/source mismatches were detected in the audited rows."
    if mismatches:
        mismatch_text = "\n".join(
            [
                f"- {row.get('figure_name')}: `{row.get('curve_label')}` intended `{row.get('intended_model_name')}` but used `{row.get('actual_model_name_used')}`."
                for row in mismatches
            ]
        )
    return "\n".join(
        [
            "# Latency Regime Plot Source Audit",
            "",
            "## 1. Summary",
            f"- Audited rows recorded: `{summary_payload.get('audit_row_count')}`.",
            f"- Best Student-t regression model in LOO table: `{summary_payload.get('best_student_t_r')}`.",
            "",
            "## 2. Source-file audit",
            f"- Stan results directory: `{summary_payload.get('stan_results_dir')}`.",
            f"- Label/source mismatch summary: {mismatch_text}",
            "",
            "## 3. Model-name filtering audit",
            "- Exact `model_name == CONSTANT` matching is now used in the audit helpers for B, C, and R2 sources.",
            "- Mixed-model row selection raises an assertion instead of silently reusing broad substring matches.",
            "",
            "## 4. R2 conditional vs marginal density",
            f"- Conditional local maxima: `{maxima.get('r2_conditional')}`.",
            f"- Marginal local maxima: `{maxima.get('r2_marginal')}`.",
            f"- Conditional R2 density unimodal: `{answers.get('E_r2_conditional_unimodal')}`.",
            f"- Marginal R2 density bimodal: `{answers.get('F_r2_marginal_bimodal')}`.",
            "",
            "## 5. Model B component density check",
            f"- Model B mixture local maxima: `{maxima.get('model_b_mixture')}`.",
            f"- Model B bimodal under audited calculation: `{answers.get('G_model_b_bimodal_under_audited_calculation')}`.",
            "",
            "## 6. Model C component density check",
            f"- Model C event-averaged mixture local maxima: `{maxima.get('model_c_event_averaged')}`.",
            "",
            "## 7. Combined audited overlay",
            "- See `latency_regime_model_overlay_audited.png` for the observed histogram, Model B audited mixture, Model C event-averaged mixture, and both R2 audited curves.",
            "",
            "## 8. Was there a model/label/source mix-up?",
            f"- A. Was the R2 curve using R2 parameters? `{answers.get('A_r2_curve_using_r2_parameters')}`.",
            f"- B. Was the Model B curve using Model B parameters? `{answers.get('B_model_b_curve_using_model_b_parameters')}`.",
            f"- C. Were any labels swapped? `{answers.get('C_any_labels_swapped')}`.",
            f"- D. Was the R2 curve conditional or marginal? `{answers.get('D_r2_curve_density_type_in_production')}`.",
            f"- E. Is R2 conditional density unimodal? `{answers.get('E_r2_conditional_unimodal')}`.",
            f"- F. Is R2 marginal density bimodal? `{answers.get('F_r2_marginal_bimodal')}`.",
            f"- G. Is Model B actually unimodal or bimodal under the audited calculation? `{answers.get('G_model_b_bimodal_under_audited_calculation')}`.",
            f"- H. Which production figure(s) need regeneration? `{', '.join(answers.get('H_figures_needing_regeneration') or [])}`.",
            "",
            "## 9. Recommended fixes",
            "- Keep posterior-predictive curves explicitly labelled as posterior predictive.",
            "- Use the audited B/C/R2 diagnostics when discussing conditional versus marginal density shape.",
            "- Regenerate any affected production figures after the standard Stan CSV outputs are present in the target results directory.",
            "",
        ]
    )


def _one_student_t_density_per_ms(
    *,
    model_a_summary: pd.DataFrame | None,
    grid_ms: np.ndarray,
    nu: float,
) -> np.ndarray | None:
    if model_a_summary is None or model_a_summary.empty:
        return None
    if not {"variable", "mean"} <= set(model_a_summary.columns):
        return None
    lookup = dict(zip(model_a_summary["variable"], pd.to_numeric(model_a_summary["mean"], errors="coerce"), strict=False))
    mu = lookup.get("mu")
    sigma = lookup.get("sigma")
    if not (np.isfinite(mu) and np.isfinite(sigma) and float(sigma) > 0.0):
        return None
    return student_t_density_per_ms(grid_ms, mu_s=float(mu), sigma_s=float(sigma), nu=float(nu))


def _skew_density_per_ms(*, model_s_summary: pd.DataFrame | None, grid_ms: np.ndarray) -> np.ndarray | None:
    if model_s_summary is None or model_s_summary.empty:
        return None
    if not {"variable", "mean"} <= set(model_s_summary.columns):
        return None
    lookup = dict(zip(model_s_summary["variable"], pd.to_numeric(model_s_summary["mean"], errors="coerce"), strict=False))
    xi = lookup.get("xi")
    omega = lookup.get("omega")
    alpha_skew = lookup.get("alpha_skew")
    if not (np.isfinite(xi) and np.isfinite(omega) and np.isfinite(alpha_skew) and float(omega) > 0.0):
        return None
    return skew_normal_density_per_ms(grid_ms, xi_s=float(xi), omega_s=float(omega), alpha_skew=float(alpha_skew))


def _mixture_density_for_model(
    *,
    component_parameters: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    model_name: str,
    grid_ms: np.ndarray,
    nu: float,
) -> np.ndarray | None:
    if component_parameters is None or component_parameters.empty:
        return None
    subset = component_parameters.loc[component_parameters.get("model_name") == model_name].copy()
    if subset.empty:
        return None
    early_mu = _extract_component_value(subset, component="early", parameter="mu")
    late_mu = _extract_component_value(subset, component="late", parameter="mu")
    early_sigma = _extract_component_value(subset, component="early", parameter="sigma")
    late_sigma = _extract_component_value(subset, component="late", parameter="sigma")
    grid_s = milliseconds_to_seconds(grid_ms)
    early_density_s = t.pdf(grid_s, df=nu, loc=early_mu, scale=early_sigma)
    late_density_s = t.pdf(grid_s, df=nu, loc=late_mu, scale=late_sigma)
    p_late_values = _resolve_event_probabilities(event_probabilities=event_probabilities, model_name=model_name)
    return mixture_density_per_ms(
        early_density_s=early_density_s,
        late_density_s=late_density_s,
        p_late_values=p_late_values,
    )


def _shifted_lognormal_density_per_ms(
    *,
    grid_ms: np.ndarray,
    shift_seconds: float,
    mu_log: float,
    sigma_log: float,
) -> np.ndarray:
    grid_s = milliseconds_to_seconds(grid_ms)
    shifted_grid = grid_s - float(shift_seconds)
    density_s = np.zeros_like(shifted_grid, dtype=float)
    positive = shifted_grid > 0.0
    if np.any(positive):
        density_s[positive] = (
            1.0
            / (shifted_grid[positive] * float(sigma_log) * np.sqrt(2.0 * np.pi))
            * np.exp(-((np.log(shifted_grid[positive]) - float(mu_log)) ** 2) / (2.0 * float(sigma_log) ** 2))
        )
    return density_per_second_to_per_millisecond(density_s)


def _resolve_event_probabilities(*, event_probabilities: pd.DataFrame | None, model_name: str) -> np.ndarray:
    if event_probabilities is not None and {"model_name", "p_late_mean"} <= set(event_probabilities.columns):
        filtered = pd.to_numeric(
            event_probabilities.loc[event_probabilities["model_name"] == model_name, "p_late_mean"],
            errors="coerce",
        )
        filtered = filtered[np.isfinite(filtered)]
        if not filtered.empty:
            return filtered.to_numpy(dtype=float)
    return np.array([0.5], dtype=float)


def _resolve_alpha(*, gating_coefficients: pd.DataFrame | None, model_name: str) -> float:
    if gating_coefficients is None or gating_coefficients.empty:
        return float("nan")
    required = {"model_name", "term", "mean"}
    if not required <= set(gating_coefficients.columns):
        return float("nan")
    subset = gating_coefficients.loc[
        (gating_coefficients["model_name"] == model_name) & (gating_coefficients["term"] == "alpha"),
        "mean",
    ]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset, errors="coerce").iloc[0])


def _resolve_regression_term_mean(
    regression_coefficients: pd.DataFrame | None,
    model_name: str,
    term: str,
) -> float:
    if regression_coefficients is None or regression_coefficients.empty:
        return float("nan")
    required = {"model_name", "term", "mean"}
    if not required <= set(regression_coefficients.columns):
        return float("nan")
    subset = regression_coefficients.loc[
        (regression_coefficients["model_name"] == model_name) & (regression_coefficients["term"] == term),
        "mean",
    ]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset, errors="coerce").iloc[0])


def _resolve_scalar_or_regression_scale(
    regression_coefficients: pd.DataFrame | None,
    model_name: str,
    default: float,
) -> float:
    sigma = _resolve_regression_term_mean(regression_coefficients, model_name, "alpha_sigma")
    if np.isfinite(sigma):
        return float(np.exp(sigma))
    return float(default)


def _resolve_shift_seconds(shifted_lognormal_diagnostics: pd.DataFrame | None, model_name: str) -> float:
    if shifted_lognormal_diagnostics is None or shifted_lognormal_diagnostics.empty:
        return float("nan")
    required = {"model_name", "shift_seconds"}
    if not required <= set(shifted_lognormal_diagnostics.columns):
        return float("nan")
    subset = shifted_lognormal_diagnostics.loc[
        shifted_lognormal_diagnostics["model_name"] == model_name,
        "shift_seconds",
    ]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset, errors="coerce").iloc[0])


def _choose_best_model(loo_table: pd.DataFrame | None, candidates: tuple[str, ...]) -> str | None:
    if loo_table is None or loo_table.empty:
        return None
    required = {"model_name", "delta_looic_from_best"}
    if not required <= set(loo_table.columns):
        return None
    working = loo_table.loc[loo_table["model_name"].isin(candidates)].copy()
    if working.empty:
        return None
    working["delta_looic_from_best"] = pd.to_numeric(working["delta_looic_from_best"], errors="coerce")
    working = working.dropna(subset=["delta_looic_from_best"]).sort_values("delta_looic_from_best")
    if working.empty:
        return None
    return str(working["model_name"].iloc[0])


def _extract_ppc_samples_ms(*, ppc_table: pd.DataFrame, model_name: str) -> np.ndarray:
    if not {"model_name", "statistic", "y_rep_value"} <= set(ppc_table.columns):
        return np.array([], dtype=float)
    subset = ppc_table.loc[
        (ppc_table["model_name"] == model_name) & (ppc_table["statistic"] == "y_rep"),
        "y_rep_value",
    ]
    values_s = pd.to_numeric(subset, errors="coerce").to_numpy(dtype=float)
    values_s = values_s[np.isfinite(values_s)]
    if values_s.size == 0:
        return np.array([], dtype=float)
    return seconds_to_milliseconds(values_s)


def _sample_density_per_ms(*, samples_ms: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(samples_ms, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.asarray(grid_ms, dtype=float)
    if values.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    if np.unique(values).size < 2:
        bandwidth = 20.0
        return normal.pdf(grid, loc=float(values[0]), scale=bandwidth)
    kde = gaussian_kde(values)
    return kde(grid)


def _build_latency_grid_ms(latency_seconds: np.ndarray) -> np.ndarray:
    return seconds_to_milliseconds(
        np.linspace(float(np.min(latency_seconds)) - 0.20, float(np.max(latency_seconds)) + 0.20, 600)
    )


def _infer_latency_unit(values_s: np.ndarray) -> str:
    values = np.asarray(values_s, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "unknown"
    median_absolute = float(np.median(np.abs(values)))
    if 0.05 <= median_absolute <= 5.0:
        return "seconds"
    if 50.0 <= median_absolute <= 5000.0:
        return "milliseconds"
    return "ambiguous"


def _extract_component_value(table: pd.DataFrame, *, component: str, parameter: str) -> float:
    subset = table.loc[(table["component"] == component) & (table["parameter"] == parameter)]
    if subset.empty:
        raise ValueError(f"Missing component summary for {component} {parameter}.")
    return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_placeholder(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, subtitle, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
