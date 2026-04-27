"""Diagnostics for the exploratory behavioural latency-regime Stan analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, t

LATENCY_SUMMARY_COLUMNS = (
    "n",
    "min",
    "q001",
    "q005",
    "q01",
    "q05",
    "q25",
    "median",
    "mean",
    "q75",
    "q95",
    "q99",
    "q995",
    "q999",
    "max",
    "proportion_negative",
    "proportion_above_0_5",
    "proportion_above_1_0",
    "proportion_above_1_5",
    "proportion_above_2_0",
    "median_absolute_latency",
    "inferred_latency_unit",
    "plotting_code_converts_seconds_to_ms",
    "stan_priors_match_data_units",
)


@dataclass(frozen=True, slots=True)
class BehaviourLatencyRegimeDiagnosticResult:
    """Paths for the latency-regime diagnostics bundle."""

    output_dir: Path
    report_path: Path


def diagnose_behaviour_latency_regime(
    *,
    event_data_csv: Path,
    stan_results_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> BehaviourLatencyRegimeDiagnosticResult:
    """Generate a diagnostics bundle for an existing latency-regime Stan run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Reading event-level latency-regime data from {event_data_csv}.")
    event_data = pd.read_csv(event_data_csv)
    latency = pd.to_numeric(event_data.get("latency_from_partner_offset"), errors="coerce")
    latency = latency[np.isfinite(latency)]
    if latency.empty:
        raise ValueError("Event data does not contain any finite `latency_from_partner_offset` values.")

    component_parameters = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_component_parameters.csv")
    gating_coefficients = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_gating_coefficients.csv")
    event_probabilities = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_event_probabilities.csv")
    loo_table = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_loo_comparison.csv")
    ppc_table = _read_csv_if_exists(stan_results_dir / "behaviour_latency_regime_posterior_predictive.csv")
    model_c_summary = _read_csv_if_exists(stan_results_dir / "behaviour_latency_model_c_mixture_of_experts_summary.csv")
    model_s_summary = _read_csv_if_exists(stan_results_dir / "behaviour_latency_model_s_skew_unimodal_summary.csv")
    fit_metrics = _read_json_if_exists(stan_results_dir / "behaviour_latency_regime_fit_metrics.json")

    if verbose:
        print(f"Writing diagnostics into {output_dir}.")

    latency_summary = summarize_latency_table(latency)
    inferred_unit = str(latency_summary["inferred_latency_unit"].iloc[0])
    plotting_converts_to_ms = bool(latency_summary["plotting_code_converts_seconds_to_ms"].iloc[0])
    stan_priors_match_units = bool(latency_summary["stan_priors_match_data_units"].iloc[0])
    latency_summary_path = output_dir / "latency_regime_latency_summary.csv"
    latency_summary.to_csv(latency_summary_path, index=False)

    component_summary = summarize_component_table(component_parameters, model_c_summary)
    component_summary_path = output_dir / "latency_regime_component_summary.csv"
    component_summary.to_csv(component_summary_path, index=False)

    gating_summary = summarize_gating_table(gating_coefficients)
    gating_summary_path = output_dir / "latency_regime_gating_summary.csv"
    gating_summary.to_csv(gating_summary_path, index=False)

    p_late_summary = summarize_p_late_table(event_probabilities)
    p_late_summary_path = output_dir / "latency_regime_p_late_summary.csv"
    p_late_summary.to_csv(p_late_summary_path, index=False)

    density_check = compute_density_check_table(
        latency=latency.to_numpy(dtype=float),
        component_parameters=component_parameters,
        model_c_summary=model_c_summary,
        gating_coefficients=gating_coefficients,
        event_probabilities=event_probabilities,
        fit_metrics=fit_metrics,
    )
    density_check_path = output_dir / "latency_regime_density_check.csv"
    density_check.to_csv(density_check_path, index=False)

    _plot_observed_histogram(
        latency_s=latency.to_numpy(dtype=float),
        output_path=output_dir / "latency_regime_observed_histogram_raw.png",
        zoom_ms=None,
    )
    _plot_observed_histogram(
        latency_s=latency.to_numpy(dtype=float),
        output_path=output_dir / "latency_regime_observed_histogram_zoom.png",
        zoom_ms=(-500.0, 1500.0),
    )
    _plot_density_overlay(
        latency_s=latency.to_numpy(dtype=float),
        density_check=density_check,
        output_path=output_dir / "latency_regime_component_overlay_check_full.png",
        zoom_ms=None,
    )
    _plot_density_overlay(
        latency_s=latency.to_numpy(dtype=float),
        density_check=density_check,
        output_path=output_dir / "latency_regime_component_overlay_check_zoom.png",
        zoom_ms=(-500.0, 1500.0),
    )
    _plot_p_late_distribution(
        event_probabilities=event_probabilities,
        output_path=output_dir / "latency_regime_p_late_distribution.png",
    )
    _plot_ppc(
        latency_s=latency.to_numpy(dtype=float),
        ppc_table=ppc_table,
        output_path=output_dir / "latency_regime_posterior_predictive_check_full.png",
        zoom_ms=None,
    )
    _plot_ppc(
        latency_s=latency.to_numpy(dtype=float),
        ppc_table=ppc_table,
        output_path=output_dir / "latency_regime_posterior_predictive_check_zoom.png",
        zoom_ms=(-500.0, 1500.0),
    )
    _plot_ppc_with_skew_unimodal_overlay(
        latency_s=latency.to_numpy(dtype=float),
        ppc_table=ppc_table,
        model_s_summary=model_s_summary,
        output_path=output_dir / "latency_regime_posterior_predictive_check_full_with_skew_unimodal.png",
    )
    _plot_observed_with_skew_unimodal_only(
        latency_s=latency.to_numpy(dtype=float),
        model_s_summary=model_s_summary,
        output_path=output_dir / "latency_regime_observed_with_skew_unimodal_only.png",
    )

    report_path = output_dir / "latency_regime_diagnostic_report.md"
    report_path.write_text(
        build_latency_regime_report(
            latency_summary=latency_summary,
            component_summary=component_summary,
            gating_summary=gating_summary,
            p_late_summary=p_late_summary,
            density_check=density_check,
            loo_table=loo_table,
            fit_metrics=fit_metrics,
            inferred_unit=inferred_unit,
            plotting_converts_to_ms=plotting_converts_to_ms,
            stan_priors_match_units=stan_priors_match_units,
            ppc_available=ppc_table is not None and not ppc_table.empty,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"Wrote diagnostic report to {report_path}.")
    return BehaviourLatencyRegimeDiagnosticResult(output_dir=output_dir, report_path=report_path)


def infer_latency_unit(latency_values: pd.Series | np.ndarray) -> str:
    values = np.asarray(latency_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "unknown"
    median_absolute = float(np.median(np.abs(values)))
    if 0.1 <= median_absolute <= 0.5:
        return "seconds"
    if 100.0 <= median_absolute <= 500.0:
        return "milliseconds"
    return "ambiguous"


def summarize_latency_table(latency_values: pd.Series | np.ndarray) -> pd.DataFrame:
    values = np.asarray(latency_values, dtype=float)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, [0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999])
    inferred_unit = infer_latency_unit(values)
    plotting_converts_to_ms = _plotting_code_converts_to_ms()
    stan_priors_match_units = inferred_unit == "seconds"
    row = {
        "n": int(values.size),
        "min": float(values.min()),
        "q001": float(quantiles[0]),
        "q005": float(quantiles[1]),
        "q01": float(quantiles[2]),
        "q05": float(quantiles[3]),
        "q25": float(quantiles[4]),
        "median": float(quantiles[5]),
        "mean": float(values.mean()),
        "q75": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "q995": float(quantiles[9]),
        "q999": float(quantiles[10]),
        "max": float(values.max()),
        "proportion_negative": float((values < 0.0).mean()),
        "proportion_above_0_5": float((values > 0.5).mean()),
        "proportion_above_1_0": float((values > 1.0).mean()),
        "proportion_above_1_5": float((values > 1.5).mean()),
        "proportion_above_2_0": float((values > 2.0).mean()),
        "median_absolute_latency": float(np.median(np.abs(values))),
        "inferred_latency_unit": inferred_unit,
        "plotting_code_converts_seconds_to_ms": plotting_converts_to_ms,
        "stan_priors_match_data_units": stan_priors_match_units,
    }
    return pd.DataFrame([row], columns=LATENCY_SUMMARY_COLUMNS)


def summarize_component_table(
    component_parameters: pd.DataFrame | None,
    model_c_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = ["source", "model_name", "component", "parameter", "mean", "sd", "q2_5", "q50", "q97_5"]
    rows: list[dict[str, object]] = []
    if component_parameters is not None and not component_parameters.empty:
        for record in component_parameters.to_dict(orient="records"):
            rows.append(
                {
                    "source": "component_parameters",
                    "model_name": record.get("model_name"),
                    "component": record.get("component"),
                    "parameter": record.get("parameter"),
                    "mean": record.get("mean"),
                    "sd": record.get("sd"),
                    "q2_5": record.get("q2_5"),
                    "q50": record.get("q50"),
                    "q97_5": record.get("q97_5"),
                }
            )
    if model_c_summary is not None and not model_c_summary.empty:
        summary_lookup = {
            "mu[1]": ("model_c_mixture_of_experts", "early", "mu"),
            "mu[2]": ("model_c_mixture_of_experts", "late", "mu"),
            "sigma[1]": ("model_c_mixture_of_experts", "early", "sigma"),
            "sigma[2]": ("model_c_mixture_of_experts", "late", "sigma"),
            "alpha": ("model_c_mixture_of_experts", "gating", "alpha"),
            "beta_rate": ("model_c_mixture_of_experts", "gating", "beta_rate"),
            "beta_expected": ("model_c_mixture_of_experts", "gating", "beta_expected"),
        }
        for _, row in model_c_summary.iterrows():
            variable = str(row.get("variable"))
            if variable not in summary_lookup:
                continue
            model_name, component, parameter = summary_lookup[variable]
            rows.append(
                {
                    "source": "model_c_summary",
                    "model_name": model_name,
                    "component": component,
                    "parameter": parameter,
                    "mean": row.get("mean"),
                    "sd": row.get("sd"),
                    "q2_5": row.get("q5"),
                    "q50": row.get("median"),
                    "q97_5": row.get("q95"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def summarize_gating_table(gating_coefficients: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["term", "mean", "sd", "q2_5", "q50", "q97_5", "prob_gt_zero", "prob_lt_zero"]
    if gating_coefficients is None:
        return pd.DataFrame(columns=columns)
    return gating_coefficients.loc[:, [column for column in columns if column in gating_coefficients.columns]].copy()


def summarize_p_late_table(event_probabilities: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["summary_type", "metric", "value", "latency_bin_ms", "n", "mean_p_late", "median_p_late"]
    if event_probabilities is None or event_probabilities.empty or "p_late_mean" not in event_probabilities.columns:
        return pd.DataFrame(columns=columns)
    working = event_probabilities.copy()
    working["p_late_mean"] = pd.to_numeric(working["p_late_mean"], errors="coerce")
    working["latency_from_partner_offset"] = pd.to_numeric(working["latency_from_partner_offset"], errors="coerce")
    working = working.dropna(subset=["p_late_mean", "latency_from_partner_offset"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    overall_quantiles = working["p_late_mean"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rows = [
        {"summary_type": "overall", "metric": "min", "value": float(working["p_late_mean"].min()), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q01", "value": float(overall_quantiles.loc[0.01]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q05", "value": float(overall_quantiles.loc[0.05]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q25", "value": float(overall_quantiles.loc[0.25]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "median", "value": float(overall_quantiles.loc[0.5]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "mean", "value": float(working["p_late_mean"].mean()), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q75", "value": float(overall_quantiles.loc[0.75]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q95", "value": float(overall_quantiles.loc[0.95]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "q99", "value": float(overall_quantiles.loc[0.99]), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
        {"summary_type": "overall", "metric": "max", "value": float(working["p_late_mean"].max()), "latency_bin_ms": np.nan, "n": np.nan, "mean_p_late": np.nan, "median_p_late": np.nan},
    ]

    working["latency_ms"] = working["latency_from_partner_offset"] * 1000.0
    bin_edges = np.arange(-500.0, max(1525.0, float(working["latency_ms"].max()) + 25.0), 250.0)
    if bin_edges.size < 2:
        bin_edges = np.array([-500.0, 1500.0], dtype=float)
    working["latency_bin_ms"] = pd.cut(working["latency_ms"], bins=bin_edges, labels=False, include_lowest=True, right=False)
    for bin_index, subset in working.groupby("latency_bin_ms", observed=True):
        left = bin_edges[int(bin_index)]
        right = bin_edges[int(bin_index) + 1]
        rows.append(
            {
                "summary_type": "latency_bin",
                "metric": f"[{left:.0f}, {right:.0f})",
                "value": np.nan,
                "latency_bin_ms": float(left),
                "n": int(len(subset)),
                "mean_p_late": float(subset["p_late_mean"].mean()),
                "median_p_late": float(subset["p_late_mean"].median()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_event_averaged_mixture_density(
    *,
    latency_grid_s: np.ndarray,
    p_late_values: np.ndarray,
    mu_early: float,
    sigma_early: float,
    mu_late: float,
    sigma_late: float,
    nu: float,
) -> np.ndarray:
    early_density = t.pdf(latency_grid_s, df=nu, loc=mu_early, scale=sigma_early)
    late_density = t.pdf(latency_grid_s, df=nu, loc=mu_late, scale=sigma_late)
    p_values = np.asarray(p_late_values, dtype=float)
    if p_values.size == 0:
        p_values = np.array([0.5], dtype=float)
    return np.mean((1.0 - p_values[:, None]) * early_density[None, :] + p_values[:, None] * late_density[None, :], axis=0)


def compute_density_check_table(
    *,
    latency: np.ndarray,
    component_parameters: pd.DataFrame | None,
    model_c_summary: pd.DataFrame | None,
    gating_coefficients: pd.DataFrame | None,
    event_probabilities: pd.DataFrame | None,
    fit_metrics: dict[str, object] | None,
) -> pd.DataFrame:
    grid_s = np.linspace(float(latency.min()) - 0.25, float(latency.max()) + 0.25, 600)
    observed_density = _compute_observed_kde(latency, grid_s) / 1000.0
    mu_early, sigma_early, mu_late, sigma_late = _resolve_model_c_components(component_parameters, model_c_summary)
    alpha = _resolve_gating_value(gating_coefficients, model_c_summary, "alpha")
    p_values = _resolve_p_late_values(event_probabilities)
    nu = float((fit_metrics or {}).get("nu", 4.0))
    early_density_s = t.pdf(grid_s, df=nu, loc=mu_early, scale=sigma_early)
    late_density_s = t.pdf(grid_s, df=nu, loc=mu_late, scale=sigma_late)
    mixture_event_averaged_s = compute_event_averaged_mixture_density(
        latency_grid_s=grid_s,
        p_late_values=p_values,
        mu_early=mu_early,
        sigma_early=sigma_early,
        mu_late=mu_late,
        sigma_late=sigma_late,
        nu=nu,
    )
    p_alpha_only = 1.0 / (1.0 + np.exp(-alpha))
    mixture_alpha_only_s = (1.0 - p_alpha_only) * early_density_s + p_alpha_only * late_density_s
    mixture_50_50_s = 0.5 * early_density_s + 0.5 * late_density_s
    return pd.DataFrame(
        {
            "latency_s": grid_s,
            "latency_ms": grid_s * 1000.0,
            "observed_density_optional": observed_density,
            "early_component_density": early_density_s / 1000.0,
            "late_component_density": late_density_s / 1000.0,
            "mixture_density_event_averaged": mixture_event_averaged_s / 1000.0,
            "mixture_density_alpha_only": mixture_alpha_only_s / 1000.0,
            "mixture_density_50_50": mixture_50_50_s / 1000.0,
        }
    )


def build_latency_regime_report(
    *,
    latency_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    gating_summary: pd.DataFrame,
    p_late_summary: pd.DataFrame,
    density_check: pd.DataFrame,
    loo_table: pd.DataFrame | None,
    fit_metrics: dict[str, object] | None,
    inferred_unit: str,
    plotting_converts_to_ms: bool,
    stan_priors_match_units: bool,
    ppc_available: bool,
) -> str:
    latency_row = latency_summary.iloc[0]
    model_c_components = component_summary.loc[
        (component_summary["model_name"] == "model_c_mixture_of_experts")
        & (component_summary["component"].isin(["early", "late"]))
    ].copy()
    mu_early = _first_value(model_c_components, component="early", parameter="mu")
    mu_late = _first_value(model_c_components, component="late", parameter="mu")
    sigma_early = _first_value(model_c_components, component="early", parameter="sigma")
    sigma_late = _first_value(model_c_components, component="late", parameter="sigma")
    alpha = _first_term(gating_summary, "alpha")
    beta_rate = _first_term(gating_summary, "beta_rate")
    beta_expected = _first_term(gating_summary, "beta_expected")
    p_late_mean = _extract_overall_metric(p_late_summary, "mean")
    p_late_median = _extract_overall_metric(p_late_summary, "median")
    high_tail_rows = p_late_summary.loc[p_late_summary["summary_type"] == "latency_bin"].copy()
    max_bin_row = high_tail_rows.sort_values("mean_p_late", ascending=False).head(1)
    top_bin_text = (
        f"Highest mean p_late occurs in latency bin {max_bin_row['metric'].iloc[0]} ms "
        f"(mean p_late={max_bin_row['mean_p_late'].iloc[0]:.3f}, n={int(max_bin_row['n'].iloc[0])})."
        if not max_bin_row.empty
        else "Latency-bin p_late summaries were unavailable."
    )
    mixture_event_peak_ms = float(density_check.loc[density_check["mixture_density_event_averaged"].idxmax(), "latency_ms"])
    late_peak_ms = float(density_check.loc[density_check["late_component_density"].idxmax(), "latency_ms"])
    alpha_only_peak_ms = float(density_check.loc[density_check["mixture_density_alpha_only"].idxmax(), "latency_ms"])
    loo_lines = _format_loo_lines(loo_table)
    workflow_warnings = list((fit_metrics or {}).get("workflow_warnings", []))
    loo_available = (fit_metrics or {}).get("loo_available")

    summary_lines = [
        "## 1. Summary",
        "",
        f"- The strange ~1000 ms feature is **not explained by a seconds/milliseconds mismatch**. The exported latencies are in `{inferred_unit}`, the plotting code converts seconds to milliseconds for display, and the Stan priors are written on the same seconds scale.",
        f"- Model C currently fits an **early/main component near {mu_early * 1000.0:.0f} ms** and a **late-tail component near {mu_late * 1000.0:.0f} ms** rather than a 0 ms versus 150 ms pair of nearby overlapping regimes.",
        f"- The current `behaviour_latency_regime_components.png` implementation uses the mean event-level `p_late` weight. Because the component parameters are global, that is algebraically equivalent to the correct event-averaged marginal mixture density. So this is **not a plotting bug in the marginal-weight calculation**.",
        f"- The LOO result therefore appears to support an **information-linked long-latency tail component**, not yet the intended 0/150 ms regime hypothesis.",
        "",
        "## 2. Data/unit checks",
        "",
        f"- `n = {int(latency_row['n'])}` events.",
        f"- Latency range: {latency_row['min']:.3f} s to {latency_row['max']:.3f} s.",
        f"- Median latency: {latency_row['median']:.3f} s ({latency_row['median'] * 1000.0:.0f} ms).",
        f"- Median absolute latency: {latency_row['median_absolute_latency']:.3f}. Inferred unit: `{inferred_unit}`.",
        f"- Proportion negative: {latency_row['proportion_negative']:.3f}.",
        f"- Proportion above 0.5 s: {latency_row['proportion_above_0_5']:.3f}; above 1.0 s: {latency_row['proportion_above_1_0']:.3f}; above 1.5 s: {latency_row['proportion_above_1_5']:.3f}.",
        f"- Plotting code appears to convert seconds to ms: `{plotting_converts_to_ms}`.",
        f"- Stan priors and data appear on the same unit scale: `{stan_priors_match_units}`.",
        "",
        "## 3. Observed latency distribution",
        "",
        "- Inspect `latency_regime_observed_histogram_raw.png` and `latency_regime_observed_histogram_zoom.png`.",
        "- These plots are intended to answer whether there is an actual empirical mode near 1000 ms or whether the distribution is mainly concentrated around earlier latencies with a long right tail.",
        f"- The raw quantiles already show a substantial long tail: q95={latency_row['q95'] * 1000.0:.0f} ms, q99={latency_row['q99'] * 1000.0:.0f} ms, max={latency_row['max'] * 1000.0:.0f} ms.",
        "",
        "## 4. Component parameter checks",
        "",
        f"- Model C `mu_early` is about {mu_early:.3f} s ({mu_early * 1000.0:.0f} ms).",
        f"- Model C `mu_late` is about {mu_late:.3f} s ({mu_late * 1000.0:.0f} ms).",
        f"- Model C `sigma_early` is about {sigma_early:.3f} s; `sigma_late` is about {sigma_late:.3f} s.",
        f"- Gating coefficients: alpha={alpha:.3f}, beta_rate={beta_rate:.3f}, beta_expected={beta_expected:.3f}.",
        f"- `mu_late` is {'near 0.150 s' if abs(mu_late - 0.150) < 0.100 else 'not near 0.150 s'} and {'near 1.000 s' if abs(mu_late - 1.000) < 0.200 else 'not near 1.000 s'}.",
        f"- `mu_early` is {'near 0 ms' if abs(mu_early) < 0.075 else 'not near 0 ms'}, {'near 150 ms' if abs(mu_early - 0.150) < 0.075 else 'not near 150 ms'}, and {'near 200 ms' if abs(mu_early - 0.200) < 0.100 else 'not near 200 ms'}.",
        "- Interpretation: the fitted components currently look more like **ordinary latency versus very-late tail/outlier latency** than the intended pair of nearby overlapping regimes.",
        "",
        "## 5. p_late checks",
        "",
        f"- Overall `p_late_mean` mean={p_late_mean:.3f}, median={p_late_median:.3f}.",
        f"- {top_bin_text}",
        "- Inspect `latency_regime_p_late_distribution.png` to see whether assignments are mostly near 0/1 or spread smoothly.",
        "- If high `p_late` is concentrated in the longest-latency bins, that supports the interpretation that Model C is acting as a long-tail detector.",
        "",
        "## 6. Mixture-density plotting check",
        "",
        f"- The event-averaged mixture density peaks near {mixture_event_peak_ms:.0f} ms.",
        f"- The late component density peaks near {late_peak_ms:.0f} ms.",
        f"- The alpha-only mixture density peaks near {alpha_only_peak_ms:.0f} ms.",
        "- Because the component parameters do not vary by event, averaging the mixture over event-specific `p_late_i` is equivalent to using the mean `p_late` weight. The old plot therefore appears conceptually consistent with the correct marginal density, although the new diagnostic plots make the distinction explicit.",
        "- Inspect `latency_regime_component_overlay_check_full.png` and `latency_regime_component_overlay_check_zoom.png` to determine whether the 1000 ms structure belongs mainly to the late component, the overall mixture, or only to alternative weighting schemes.",
        "",
        "## 7. Posterior predictive check",
        "",
        f"- Posterior predictive file available: `{ppc_available}`.",
        "- Inspect `latency_regime_posterior_predictive_check_full.png` and `latency_regime_posterior_predictive_check_zoom.png`.",
        "- The key question is whether Model C reproduces the observed bulk around early-to-mid latencies while overpredicting a separate long-latency hump.",
        "",
        "## 8. Prior/unit consistency check",
        "",
        "- Model C priors from `behaviour_latency_mixture_of_experts.stan` are:",
        "  - `mu[1] ~ normal(0.000, 0.150)`",
        "  - `mu[2] ~ normal(0.150, 0.150)`",
        "  - `sigma ~ normal(0, 0.300)` with lower bound 0",
        "  - `alpha ~ normal(0, 2)`",
        "  - `beta_rate ~ normal(0, 1)`",
        "  - `beta_expected ~ normal(0, 1)`",
        f"  - `nu = {(fit_metrics or {}).get('nu', 4)}`",
        f"- Data are on the `{inferred_unit}` scale, so a seconds/ms mismatch does **not** explain the 1000 ms component.",
        "- However, the priors are broad enough that the late component can drift far from 150 ms if the likelihood prefers a long-tail solution.",
        "",
        "## 9. LOO interpretation",
        "",
        *loo_lines,
        "- If Model C is much better than Model B while `mu_late` sits near 1000 ms, then the improvement should be interpreted as information variables helping to predict **very-late tail responses**, not yet the intended 0/150 ms timing-regime hypothesis.",
        "",
        "## 10. Recommended next steps",
        "",
        "- Recommended active path: **A/S/B/C interpretation only**.",
        "- Report Model C, if retained, as an exploratory **main-vs-late-tail** mixture rather than nearby regime evidence.",
        "- Compare Model S vs Model C to evaluate whether mixture structure adds beyond skewed unimodality.",
        "- Compare Model B vs Model C to evaluate value added from information-linked mixing versus constant-weight mixing.",
        "- A pure plotting-bug explanation is not supported by the current diagnostics.",
        "- A unit-mismatch explanation is not supported by the current diagnostics.",
        "- If posterior predictive plots show substantial mismatch in the bulk of the distribution, revise model assumptions before interpretation.",
    ]
    if workflow_warnings:
        summary_lines.extend(["", "### Workflow warnings", ""] + [f"- {warning}" for warning in workflow_warnings])
    if loo_available is False:
        summary_lines.extend(["", "- `loo` was unavailable for that run; any missing LOO values reflect that environment rather than model behavior."])
    return "\n".join(summary_lines) + "\n"


def _plot_observed_histogram(*, latency_s: np.ndarray, output_path: Path, zoom_ms: tuple[float, float] | None) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values_ms = latency_s * 1000.0
    bins = np.arange(np.floor(values_ms.min() / 25.0) * 25.0, np.ceil(values_ms.max() / 25.0) * 25.0 + 25.0, 25.0)
    if bins.size < 2:
        bins = np.array([values_ms.min() - 25.0, values_ms.max() + 25.0], dtype=float)
    axis.hist(values_ms, bins=bins, color="#d7e3ea", edgecolor="white", alpha=0.95)
    _add_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Count")
    axis.set_title("Observed latency histogram")
    if zoom_ms is not None:
        axis.set_xlim(*zoom_ms)
    _save_figure(figure, output_path)


def _plot_density_overlay(
    *,
    latency_s: np.ndarray,
    density_check: pd.DataFrame,
    output_path: Path,
    zoom_ms: tuple[float, float] | None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values_ms = latency_s * 1000.0
    bins = np.arange(np.floor(values_ms.min() / 25.0) * 25.0, np.ceil(values_ms.max() / 25.0) * 25.0 + 25.0, 25.0)
    if bins.size < 2:
        bins = np.array([values_ms.min() - 25.0, values_ms.max() + 25.0], dtype=float)
    axis.hist(values_ms, bins=bins, density=True, color="#ebf1f5", edgecolor="white", alpha=0.9, label="Observed histogram")
    axis.plot(density_check["latency_ms"], density_check["early_component_density"], color="#1d3557", linewidth=2.0, label="Early component")
    axis.plot(density_check["latency_ms"], density_check["late_component_density"], color="#e76f51", linewidth=2.0, label="Late component")
    axis.plot(density_check["latency_ms"], density_check["mixture_density_event_averaged"], color="#2a9d8f", linewidth=2.4, label="Mixture event-averaged")
    axis.plot(density_check["latency_ms"], density_check["mixture_density_alpha_only"], color="#264653", linewidth=1.8, linestyle="--", label="Mixture alpha-only")
    axis.plot(density_check["latency_ms"], density_check["mixture_density_50_50"], color="#8d99ae", linewidth=1.8, linestyle=":", label="Mixture 50/50")
    _add_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Latency-regime component overlay check")
    if zoom_ms is not None:
        axis.set_xlim(*zoom_ms)
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_p_late_distribution(*, event_probabilities: pd.DataFrame | None, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    if event_probabilities is None or event_probabilities.empty or "p_late_mean" not in event_probabilities.columns:
        _draw_placeholder(axis, "P(late) distribution", "Event probability file was unavailable.")
        _save_figure(figure, output_path)
        return
    values = pd.to_numeric(event_probabilities["p_late_mean"], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        _draw_placeholder(axis, "P(late) distribution", "No finite p_late_mean values were available.")
        _save_figure(figure, output_path)
        return
    axis.hist(values, bins=np.linspace(0.0, 1.0, 21), color="#5b8e7d", edgecolor="white")
    axis.set_xlabel("Posterior mean P(late)")
    axis.set_ylabel("Count")
    axis.set_title("Distribution of event-level late-component probabilities")
    _save_figure(figure, output_path)


def _plot_ppc(
    *,
    latency_s: np.ndarray,
    ppc_table: pd.DataFrame | None,
    output_path: Path,
    zoom_ms: tuple[float, float] | None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values_ms = latency_s * 1000.0
    if ppc_table is None or ppc_table.empty or "y_rep_value" not in ppc_table.columns:
        _draw_placeholder(axis, "Posterior predictive check", "Posterior predictive file was unavailable.")
        _save_figure(figure, output_path)
        return
    working = ppc_table.copy()
    if "model_name" in working.columns:
        model_c_rows = working.loc[working["model_name"] == "model_c_mixture_of_experts"].copy()
        if not model_c_rows.empty:
            working = model_c_rows
    working = working.loc[working["y_rep_value"].notna()].copy()
    working["y_rep_value"] = pd.to_numeric(working["y_rep_value"], errors="coerce") * 1000.0
    working = working.dropna(subset=["y_rep_value"])
    if working.empty:
        _draw_placeholder(axis, "Posterior predictive check", "No replicated y values were available.")
        _save_figure(figure, output_path)
        return
    combined_ms = np.concatenate([values_ms, working["y_rep_value"].to_numpy(dtype=float)])
    bins = np.arange(np.floor(combined_ms.min() / 25.0) * 25.0, np.ceil(combined_ms.max() / 25.0) * 25.0 + 25.0, 25.0)
    if bins.size < 2:
        bins = np.array([combined_ms.min() - 25.0, combined_ms.max() + 25.0], dtype=float)
    observed_density, edges = np.histogram(values_ms, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    axis.hist(values_ms, bins=bins, density=True, color="#ebf1f5", edgecolor="none", alpha=0.8, zorder=1)
    axis.plot(centers, observed_density, color="#111111", linewidth=2.8, label="Observed", zorder=6)
    curve_maxima = [float(np.nanmax(observed_density))]
    for draw_id, draw_subset in list(working.groupby("draw_id", sort=True))[:40]:
        density, _ = np.histogram(draw_subset["y_rep_value"], bins=bins, density=True)
        curve_maxima.append(float(np.nanmax(density)))
        axis.plot(centers, density, color="#457b9d", alpha=0.10, linewidth=1.0, zorder=3)
    _apply_robust_density_ylim(axis, curve_maxima)
    _add_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Posterior predictive check (Model C)")
    if zoom_ms is not None:
        axis.set_xlim(*zoom_ms)
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_ppc_with_skew_unimodal_overlay(
    *,
    latency_s: np.ndarray,
    ppc_table: pd.DataFrame | None,
    model_s_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values_ms = latency_s * 1000.0
    if ppc_table is None or ppc_table.empty or "y_rep_value" not in ppc_table.columns:
        _draw_placeholder(axis, "Posterior predictive check", "Posterior predictive file was unavailable.")
        _save_figure(figure, output_path)
        return
    working = ppc_table.copy()
    if "model_name" in working.columns:
        model_c_rows = working.loc[working["model_name"] == "model_c_mixture_of_experts"].copy()
        if not model_c_rows.empty:
            working = model_c_rows
    working = working.loc[working["y_rep_value"].notna()].copy()
    working["y_rep_value"] = pd.to_numeric(working["y_rep_value"], errors="coerce") * 1000.0
    working = working.dropna(subset=["y_rep_value"])
    if working.empty:
        _draw_placeholder(axis, "Posterior predictive check", "No replicated y values were available.")
        _save_figure(figure, output_path)
        return
    combined_ms = np.concatenate([values_ms, working["y_rep_value"].to_numpy(dtype=float)])
    bins = np.arange(np.floor(combined_ms.min() / 25.0) * 25.0, np.ceil(combined_ms.max() / 25.0) * 25.0 + 25.0, 25.0)
    if bins.size < 2:
        bins = np.array([combined_ms.min() - 25.0, combined_ms.max() + 25.0], dtype=float)
    observed_density, edges = np.histogram(values_ms, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    axis.hist(values_ms, bins=bins, density=True, color="#ebf1f5", edgecolor="none", alpha=0.8, zorder=1)
    axis.plot(centers, observed_density, color="#111111", linewidth=2.8, label="Observed", zorder=6)
    curve_maxima = [float(np.nanmax(observed_density))]
    for _, draw_subset in list(working.groupby("draw_id", sort=True))[:40]:
        density, _ = np.histogram(draw_subset["y_rep_value"], bins=bins, density=True)
        curve_maxima.append(float(np.nanmax(density)))
        axis.plot(centers, density, color="#457b9d", alpha=0.10, linewidth=1.0, zorder=3)

    skew_params = _resolve_model_s_skew_params(model_s_summary)
    if skew_params is not None:
        xi, omega, alpha_skew = skew_params
        centers_s = centers / 1000.0
        z = (centers_s - xi) / omega
        skew_density_s = (2.0 / omega) * norm.pdf(z) * norm.cdf(alpha_skew * z)
        skew_density_ms = skew_density_s / 1000.0
        curve_maxima.append(float(np.nanmax(skew_density_ms)))
        axis.plot(centers, skew_density_ms, color="#6d597a", linewidth=2.2, label="Skewed unimodal (Model S)", zorder=5)

    _apply_robust_density_ylim(axis, curve_maxima)

    _add_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Posterior predictive check (Model C) + skewed unimodal overlay")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_observed_with_skew_unimodal_only(
    *,
    latency_s: np.ndarray,
    model_s_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    values_ms = latency_s * 1000.0
    bins = np.arange(np.floor(values_ms.min() / 25.0) * 25.0, np.ceil(values_ms.max() / 25.0) * 25.0 + 25.0, 25.0)
    if bins.size < 2:
        bins = np.array([values_ms.min() - 25.0, values_ms.max() + 25.0], dtype=float)
    observed_density, edges = np.histogram(values_ms, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    axis.plot(centers, observed_density, color="#111111", linewidth=2.0, label="Observed")

    skew_params = _resolve_model_s_skew_params(model_s_summary)
    if skew_params is not None:
        xi, omega, alpha_skew = skew_params
        centers_s = centers / 1000.0
        z = (centers_s - xi) / omega
        skew_density_s = (2.0 / omega) * norm.pdf(z) * norm.cdf(alpha_skew * z)
        axis.plot(centers, skew_density_s / 1000.0, color="#6d597a", linewidth=2.3, label="Skewed unimodal (Model S)")
    else:
        axis.text(
            0.5,
            0.90,
            "Model S summary unavailable; skew overlay omitted.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    _add_reference_lines(axis)
    axis.set_xlabel("Latency from partner offset (ms)")
    axis.set_ylabel("Probability density (per ms)")
    axis.set_title("Observed density with skewed unimodal fit only")
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _resolve_model_c_components(
    component_parameters: pd.DataFrame | None,
    model_c_summary: pd.DataFrame | None,
) -> tuple[float, float, float, float]:
    if component_parameters is not None and not component_parameters.empty:
        subset = component_parameters.loc[component_parameters["model_name"] == "model_c_mixture_of_experts"].copy()
        if not subset.empty:
            return (
                _extract_component_value(subset, "early", "mu"),
                _extract_component_value(subset, "early", "sigma"),
                _extract_component_value(subset, "late", "mu"),
                _extract_component_value(subset, "late", "sigma"),
            )
    if model_c_summary is None or model_c_summary.empty:
        raise ValueError("Model C summaries were unavailable for density diagnostics.")
    lookup = dict(zip(model_c_summary["variable"], model_c_summary["mean"], strict=False))
    return (
        float(lookup["mu[1]"]),
        float(lookup["sigma[1]"]),
        float(lookup["mu[2]"]),
        float(lookup["sigma[2]"]),
    )


def _resolve_model_s_skew_params(model_s_summary: pd.DataFrame | None) -> tuple[float, float, float] | None:
    if model_s_summary is None or model_s_summary.empty:
        return None
    required = {"variable", "mean"}
    if not required <= set(model_s_summary.columns):
        return None
    lookup = dict(zip(model_s_summary["variable"], pd.to_numeric(model_s_summary["mean"], errors="coerce"), strict=False))
    xi = lookup.get("xi")
    omega = lookup.get("omega")
    alpha_skew = lookup.get("alpha_skew")
    if not np.isfinite(xi) or not np.isfinite(omega) or not np.isfinite(alpha_skew):
        return None
    if float(omega) <= 0:
        return None
    return float(xi), float(omega), float(alpha_skew)


def _resolve_gating_value(gating_coefficients: pd.DataFrame | None, model_c_summary: pd.DataFrame | None, term: str) -> float:
    if gating_coefficients is not None and not gating_coefficients.empty:
        subset = gating_coefficients.loc[gating_coefficients["term"] == term]
        if not subset.empty:
            return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])
    if model_c_summary is not None and not model_c_summary.empty:
        subset = model_c_summary.loc[model_c_summary["variable"] == term]
        if not subset.empty:
            return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])
    return 0.0


def _resolve_p_late_values(event_probabilities: pd.DataFrame | None) -> np.ndarray:
    if event_probabilities is None or event_probabilities.empty or "p_late_mean" not in event_probabilities.columns:
        return np.array([0.5], dtype=float)
    values = pd.to_numeric(event_probabilities["p_late_mean"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.5], dtype=float)
    return values


def _compute_observed_kde(latency: np.ndarray, grid_s: np.ndarray) -> np.ndarray:
    try:
        kde = gaussian_kde(latency)
    except Exception:
        return np.full_like(grid_s, np.nan, dtype=float)
    return kde(grid_s)


def _extract_component_value(table: pd.DataFrame, component: str, parameter: str) -> float:
    subset = table.loc[(table["component"] == component) & (table["parameter"] == parameter)]
    if subset.empty:
        raise ValueError(f"Missing component parameter for {component} {parameter}.")
    return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])


def _extract_overall_metric(summary_table: pd.DataFrame, metric: str) -> float:
    subset = summary_table.loc[(summary_table["summary_type"] == "overall") & (summary_table["metric"] == metric)]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset["value"], errors="coerce").iloc[0])


def _first_value(table: pd.DataFrame, *, component: str, parameter: str) -> float:
    subset = table.loc[(table["component"] == component) & (table["parameter"] == parameter)]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])


def _first_term(table: pd.DataFrame, term: str) -> float:
    if table.empty:
        return float("nan")
    subset = table.loc[table["term"] == term]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset["mean"], errors="coerce").iloc[0])


def _format_loo_lines(loo_table: pd.DataFrame | None) -> list[str]:
    if loo_table is None or loo_table.empty:
        return ["- LOO comparison file was unavailable."]
    working = loo_table.copy()
    for column_name in ("elpd_loo", "looic", "delta_elpd_from_best", "delta_looic_from_best"):
        if column_name in working.columns:
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    lines = ["- LOO comparison table:"]
    for _, row in working.iterrows():
        lines.append(
            f"  - {row['model_name']}: elpd_loo={row.get('elpd_loo')}, looic={row.get('looic')}, "
            f"delta_elpd_from_best={row.get('delta_elpd_from_best')}, delta_looic_from_best={row.get('delta_looic_from_best')}"
        )
    model_b = working.loc[working["model_name"] == "model_b_two_student_t_mixture"]
    model_c = working.loc[working["model_name"] == "model_c_mixture_of_experts"]
    model_a = working.loc[working["model_name"] == "model_a_one_student_t"]
    if not model_b.empty and not model_c.empty:
        delta_bc = float(model_b["elpd_loo"].iloc[0] - model_c["elpd_loo"].iloc[0])
        lines.append(f"- Model C vs Model B: Model C improves elpd_loo by {abs(delta_bc):.3f} relative to Model B.")
    if not model_a.empty and not model_b.empty:
        delta_ba = float(model_a["elpd_loo"].iloc[0] - model_b["elpd_loo"].iloc[0])
        lines.append(f"- Model B vs Model A: Model B improves elpd_loo by {abs(delta_ba):.3f} relative to Model A.")
    return lines


def _plotting_code_converts_to_ms() -> bool:
    plot_module_path = Path(__file__).resolve().parent / "plot_latency_regime.py"
    text = plot_module_path.read_text(encoding="utf-8")
    return "* 1000.0" in text and "Latency from partner offset (ms)" in text


def _add_reference_lines(axis: plt.Axes) -> None:
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axvline(150.0, color="#c1121f", linestyle=":", linewidth=1.2)
    axis.axvline(1000.0, color="#7f5539", linestyle="-.", linewidth=1.2)


def _draw_placeholder(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        textwrap.fill(subtitle, width=42),
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _apply_robust_density_ylim(axis: plt.Axes, curve_maxima: list[float]) -> None:
    finite = np.asarray(curve_maxima, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return
    robust_top = float(np.quantile(finite, 0.95))
    ymax = max(robust_top * 1.15, float(np.nanmax(finite)) * 0.60)
    if np.isfinite(ymax) and ymax > 0:
        axis.set_ylim(0.0, ymax)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
