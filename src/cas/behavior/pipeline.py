"""Behavioral hazard pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas.behavior._r_backend import BACKEND_ID as R_BACKEND_ID, BACKEND_NOTE as R_BACKEND_NOTE, run_r_lag_selection, run_r_model_bundle
from cas.behavior.comparisons import MODEL_COMPARISONS
from cas.behavior.config import BehaviorHazardConfig, load_behavior_hazard_config
from cas.behavior.diagnostics import bins_by_subject, collinearity_summary, event_rate_summary
from cas.behavior.formulas import formula_metadata, render_fixed_formula, render_formula
from cas.behavior.models import RIDGE_BACKEND_ID, STANDARD_BACKEND_ID, odds_ratio_rows
from cas.behavior.predictors import standardize_predictors
from cas.behavior.risksets import build_anchor_riskset
from cas.behavior.summaries import ensure_behavior_directories
from cas.viz.behavior.lag_selection import plot_lag_selection
from cas.viz.behavior.lag_sensitivity import plot_lag_sensitivity
from cas.viz.behavior.primary_effects import plot_primary_effects
from cas.viz.behavior.qc import plot_qc_bars
from cas.viz.behavior.timing_heatmaps import plot_timing_information_interaction


LOGGER = logging.getLogger(__name__)


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", force=True)
    logging.getLogger("cas.behavior").setLevel(level)


def _log(verbose: bool, message: str, *args: object) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_parquet(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _empty_csv(path: Path, columns: list[str], *, note: str) -> Path:
    frame = pd.DataFrame(columns=columns)
    frame.loc[0, columns[0]] = note
    return _write_csv(frame, path)


def _write_placeholder_figure(path: Path, *, title: str, message: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _riskset_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "fpp": paths["risksets"] / "fpp.parquet",
        "spp": paths["risksets"] / "spp_control.parquet",
        "pooled": paths["risksets"] / "pooled_fpp_spp.parquet",
    }


def _predictor_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "fpp": paths["predictors"] / "fpp_with_lags.parquet",
        "spp": paths["predictors"] / "spp_control_with_lags.parquet",
        "pooled": paths["predictors"] / "pooled_with_lags.parquet",
        "scaling": paths["predictors"] / "standardization_summary.csv",
    }


def _load_predictor_tables(paths: dict[str, Path], config_path: str | Path, *, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    return (
        pd.read_parquet(predictor_paths["fpp"]),
        pd.read_parquet(predictor_paths["spp"]),
        pd.read_parquet(predictor_paths["pooled"]),
    )


def _load_selected_lag_payload(paths: dict[str, Path], config_path: str | Path, *, verbose: bool = False) -> dict[str, object]:
    lag_path = paths["lag_selection"] / "selected_lag.json"
    if not lag_path.exists():
        select_lag(config_path, verbose=verbose)
    return json.loads(lag_path.read_text(encoding="utf-8"))


def _backend_metadata(config: BehaviorHazardConfig) -> dict[str, object]:
    if config.model_backend == "glmm":
        return {
            "model_backend": "glmm",
            "backend_runtime": R_BACKEND_ID,
            "covariance_type": "mixed_model",
            "cluster_variable": None,
            "backend_notes": [R_BACKEND_NOTE],
            "random_effects": ["(1 | dyad_id)", "(1 | subject)"],
        }
    return {
        "model_backend": "glm",
        "backend_runtime": STANDARD_BACKEND_ID,
        "covariance_type": "model_based",
        "cluster_variable": None,
        "backend_notes": ["Behavioral hazard GLM compatibility mode uses fixed-effect binomial glm with no random effects."],
        "random_effects": [],
    }


def _build_model_specs(config: BehaviorHazardConfig, *, selected_lag_ms: int) -> list[dict[str, object]]:
    backend = config.model_backend
    specs: list[dict[str, object]] = []
    rows = [
        ("fpp", "FPP", "M_0"),
        ("fpp", "FPP", "M_1"),
        ("fpp", "FPP", "M_2"),
        ("fpp", "FPP", "M_3"),
        ("fpp", "FPP", "M_4"),
        ("spp", "SPP", "M_0"),
        ("spp", "SPP", "M_1"),
        ("spp", "SPP", "M_2"),
        ("spp", "SPP", "M_3"),
        ("spp", "SPP", "M_4"),
        ("pooled", "pooled_fpp_spp", "M_pooled_main"),
        ("pooled", "pooled_fpp_spp", "M_pooled_anchor_interaction"),
    ]
    for anchor_subset, dataset, model_id in rows:
        formula_fixed = render_fixed_formula(model_id, lag_ms=selected_lag_ms)
        specs.append(
            {
                "anchor_subset": anchor_subset,
                "dataset": dataset,
                "model_id": model_id,
                "formula_fixed": formula_fixed,
                "formula_full": render_formula(model_id, lag_ms=selected_lag_ms, backend=backend),
            }
        )
    return specs


def _build_comparison_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for anchor_subset, pairs in MODEL_COMPARISONS.items():
        for parent_model_id, child_model_id in pairs:
            specs.append(
                {
                    "comparison_id": f"{anchor_subset}__{parent_model_id}__vs__{child_model_id}",
                    "anchor_subset": anchor_subset,
                    "parent_model_id": parent_model_id,
                    "child_model_id": child_model_id,
                }
            )
    return specs


def build_risksets(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    _log(verbose, "[behavior hazard] Building FPP risk set")
    fpp = build_anchor_riskset(config, anchor="FPP", verbose=verbose)
    _log(verbose, "[behavior hazard] Building SPP matched-control risk set")
    spp = build_anchor_riskset(config, anchor="SPP", verbose=verbose)
    pooled = pd.concat([fpp, spp], ignore_index=True, sort=False)
    outputs = _riskset_paths(paths)
    _write_parquet(fpp, outputs["fpp"])
    _write_parquet(spp, outputs["spp"])
    _write_parquet(pooled, outputs["pooled"])
    return outputs


def add_predictors(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    riskset_paths = _riskset_paths(paths)
    if not all(path.exists() for path in riskset_paths.values()):
        build_risksets(config_path, verbose=verbose)
    fpp = pd.read_parquet(riskset_paths["fpp"])
    spp = pd.read_parquet(riskset_paths["spp"])
    pooled = pd.read_parquet(riskset_paths["pooled"])
    fpp_pred, spp_pred, pooled_pred, scaling = standardize_predictors(fpp, spp, pooled, config=config, verbose=verbose)
    outputs = _predictor_paths(paths)
    _write_parquet(fpp_pred, outputs["fpp"])
    _write_parquet(spp_pred, outputs["spp"])
    _write_parquet(pooled_pred, outputs["pooled"])
    _write_csv(scaling, outputs["scaling"])
    return outputs


def select_lag(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    fpp = pd.read_parquet(predictor_paths["fpp"])
    score_path = paths["lag_selection"] / "candidate_lag_scores.csv"
    selected_path = paths["lag_selection"] / "selected_lag.json"
    run_r_lag_selection(
        fpp_table=fpp,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        score_path=score_path,
        selected_path=selected_path,
        lag_sensitivity_path=paths["diagnostics"] / "lag_sensitivity.csv",
        verbose=verbose,
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    _log(verbose, "[behavior hazard] Selected shared lag: %d ms", int(selected["selected_lag_ms"]))
    return {"scores": score_path, "selected": selected_path}


def _write_model_jsons_from_r_outputs(model_metrics: pd.DataFrame, paths: dict[str, Path]) -> None:
    output_dir_by_anchor = {
        "fpp": paths["models_fpp"],
        "spp": paths["models_spp"],
        "pooled": paths["models_pooled"],
    }
    for row in model_metrics.to_dict("records"):
        output_dir = output_dir_by_anchor[str(row["anchor_subset"])]
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            key: row.get(key)
            for key in [
                "anchor_subset",
                "model_id",
                "model_backend",
                "lag_selection_criterion",
                "selected_lag_ms",
                "formula_fixed",
                "formula_full",
                "random_effects",
                "covariance_type",
                "n",
                "k",
                "logLik",
                "AIC",
                "BIC",
                "converged",
                "warnings",
            ]
        }
        _write_json(output_dir / f"{payload['model_id']}.json", payload)


def fit_models(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    selected_lag_ms = int(selected_payload["selected_lag_ms"])
    fpp, spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_model_coefficients.csv"
    run_r_model_bundle(
        fpp_table=fpp,
        spp_table=spp,
        pooled_table=pooled,
        selected_lag_ms=selected_lag_ms,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        model_specs=_build_model_specs(config, selected_lag_ms=selected_lag_ms),
        comparison_specs=[],
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=paths["logs"] / "r_model_comparisons.csv",
        convergence_path=paths["logs"] / "r_convergence_warnings.csv",
        lag_sensitivity_path=None,
        verbose=verbose,
    )
    _write_model_jsons_from_r_outputs(pd.read_csv(model_metrics_path), paths)
    return {"models_root": paths["models"]}


def build_tables(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    selected_lag_ms = int(selected_payload["selected_lag_ms"])
    fpp, spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_table_coefficients.csv"
    comparison_path = paths["logs"] / "r_table_comparisons.csv"
    convergence_path = paths["diagnostics"] / "convergence_warnings.csv"
    run_r_model_bundle(
        fpp_table=fpp,
        spp_table=spp,
        pooled_table=pooled,
        selected_lag_ms=selected_lag_ms,
        candidate_lags_ms=config.candidate_lags_ms,
        model_backend=config.model_backend,
        lag_selection_criterion=config.lag_selection_criterion,
        model_specs=_build_model_specs(config, selected_lag_ms=selected_lag_ms),
        comparison_specs=_build_comparison_specs(),
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=comparison_path,
        convergence_path=convergence_path,
        lag_sensitivity_path=None,
        figure_prediction_path=paths["tables"] / "figure_predictions.csv",
        timing_heatmap_path=paths["tables"] / "timing_heatmap_predictions.csv",
        three_way_path=paths["tables"] / "three_way_heatmap_predictions.csv",
        verbose=verbose,
    )
    model_metrics = pd.read_csv(model_metrics_path)
    coefficients = pd.read_csv(coefficient_path)
    comparisons = pd.read_csv(comparison_path)
    _write_csv(model_metrics, paths["tables"] / "model_summary.csv")
    _write_csv(coefficients, paths["tables"] / "coefficient_summary.csv")
    _write_csv(comparisons, paths["tables"] / "model_comparisons.csv")
    _write_csv(odds_ratio_rows(coefficients), paths["tables"] / "odds_ratios.csv")
    _write_csv(event_rate_summary(pooled), paths["tables"] / "event_rate_summary.csv")
    _write_csv(bins_by_subject(pooled), paths["tables"] / "bins_by_subject.csv")
    collinearity_columns = [
        "z_time_from_partner_onset_s",
        "z_time_from_partner_offset_s",
        "z_time_from_partner_offset_s_squared",
        f"z_information_rate_lag_{selected_lag_ms}",
        f"z_prop_expected_cum_info_lag_{selected_lag_ms}",
    ]
    _write_csv(collinearity_summary(pooled, collinearity_columns), paths["tables"] / "collinearity_summary.csv")
    lag_sensitivity_path = paths["diagnostics"] / "lag_sensitivity.csv"
    if not lag_sensitivity_path.exists():
        _empty_csv(lag_sensitivity_path, ["candidate_lag_ms", "term", "predictor", "estimate", "ci_low", "ci_high", "backend"], note="Lag sensitivity was not available.")
    _empty_csv(paths["diagnostics"] / "loo_subject_summary.csv", ["left_out_subject", "term", "estimate", "ci_low", "ci_high", "notes"], note="Disabled by config or not implemented.")
    summarize_outputs(config_path, verbose=verbose)
    return {"tables": paths["tables"] / "model_comparisons.csv"}


def summarize_outputs(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    backend_meta = _backend_metadata(config)
    summary_path = paths["logs"] / "summary_manifest.json"
    payload = {
        "hazard_root": str(paths["hazard_root"]),
        "tables": str(paths["tables"]),
        "diagnostics": str(paths["diagnostics"]),
        "figures_main": str(paths["figures_main"]),
        "figures_supp": str(paths["figures_supp"]),
        "figures_qc": str(paths["figures_qc"]),
        "selected_lag_ms": int(selected_payload["selected_lag_ms"]),
        "selector_model_id": str(selected_payload["selector_model_id"]),
        "anchor_subset": str(selected_payload["anchor_subset"]),
        "model_backend": backend_meta["model_backend"],
        "lag_selection_criterion": config.lag_selection_criterion,
        "backend": backend_meta["backend_runtime"],
        "covariance_type": backend_meta["covariance_type"],
        "cluster_variable": backend_meta["cluster_variable"],
        "backend_notes": list(backend_meta["backend_notes"]),
        "possible_backends": ["glm", "glmm"],
        "available_runtime_backends": [R_BACKEND_ID, STANDARD_BACKEND_ID, RIDGE_BACKEND_ID],
    }
    _write_json(summary_path, payload)
    return {"summary": summary_path}


def render_figures(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "figure_predictions.csv").exists():
        build_tables(config_path, verbose=verbose)
    lag_scores = pd.read_csv(paths["lag_selection"] / "candidate_lag_scores.csv")
    selected = json.loads((paths["lag_selection"] / "selected_lag.json").read_text(encoding="utf-8"))
    selected_lag_ms = int(selected["selected_lag_ms"])
    primary_predictions = pd.read_csv(paths["tables"] / "figure_predictions.csv")
    timing_predictions = pd.read_csv(paths["tables"] / "timing_heatmap_predictions.csv")
    lag_sensitivity = pd.read_csv(paths["diagnostics"] / "lag_sensitivity.csv")
    outputs: dict[str, Path] = {}
    outputs["lag_selection"] = plot_lag_selection(lag_scores, paths["figures_main"] / "fig01_lag_selection.png")
    outputs["primary_effects"] = plot_primary_effects(primary_predictions, paths["figures_main"] / "fig02_primary_information_effects.png")
    outputs["timing"] = plot_timing_information_interaction(
        timing_predictions.loc[timing_predictions["panel"].astype(str) == "A"].copy(),
        timing_predictions.loc[timing_predictions["panel"].astype(str) == "B"].copy(),
        paths["figures_main"] / "fig03_timing_information_interaction.png",
    )
    if not lag_sensitivity.empty:
        plot_lag_sensitivity(lag_sensitivity, selected_lag_ms=selected_lag_ms, output_path=paths["figures_supp"] / "figS01_lag_sensitivity.png")
    else:
        _write_placeholder_figure(paths["figures_supp"] / "figS01_lag_sensitivity.png", title="Lag sensitivity", message="Lag sensitivity table was unavailable.")
    _write_placeholder_figure(paths["figures_supp"] / "figS02_extra_timing_maps.png", title="Supplementary timing maps", message="Main Figure 3 now uses ridge plots from M_4.")
    _write_placeholder_figure(paths["figures_supp"] / "figS03_three_way_interaction.png", title="Three-way interaction", message="The active behavioral target uses per-anchor M_0–M_4 models plus the pooled omnibus anchor-interaction model, and does not fit an exploratory three-way interaction model.")
    return outputs


def render_qc(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "event_rate_summary.csv").exists():
        build_tables(config_path, verbose=verbose)
    event_rates = pd.read_csv(paths["tables"] / "event_rate_summary.csv")
    bins = pd.read_csv(paths["tables"] / "bins_by_subject.csv")
    collinearity = pd.read_csv(paths["tables"] / "collinearity_summary.csv")
    convergence = pd.read_csv(paths["diagnostics"] / "convergence_warnings.csv")
    outputs = {
        "event_rates": plot_qc_bars(event_rates, x="anchor_type", y="event_rate", title="Event rates", output_path=paths["figures_qc"] / "qc_event_rates.png"),
        "bins_by_subject": plot_qc_bars(bins, x="subject", y="n_bins", title="Bins by subject", output_path=paths["figures_qc"] / "qc_bins_by_subject.png"),
        "collinearity": plot_qc_bars(collinearity, x="term", y="vif", title="Collinearity", output_path=paths["figures_qc"] / "qc_collinearity.png"),
        "model_convergence": plot_qc_bars(
            convergence.assign(count=1).groupby("model_id", as_index=False)["count"].sum() if not convergence.empty else pd.DataFrame({"model_id": [], "count": []}),
            x="model_id",
            y="count",
            title="Model convergence warnings",
            output_path=paths["figures_qc"] / "qc_model_convergence.png",
        ),
    }
    return outputs


def run_behavior_hazard_stage(stage: str, *, config_path: str | Path, verbose: bool = False) -> dict[str, Path]:
    _configure_logging(verbose=verbose)
    _log(verbose, "[behavior hazard] Starting stage: %s", stage)
    if stage == "build-risksets":
        return build_risksets(config_path, verbose=verbose)
    if stage == "add-predictors":
        return add_predictors(config_path, verbose=verbose)
    if stage == "select-lag":
        return select_lag(config_path, verbose=verbose)
    if stage == "fit-models":
        return fit_models(config_path, verbose=verbose)
    if stage == "tables":
        return build_tables(config_path, verbose=verbose)
    if stage == "summarize":
        return summarize_outputs(config_path, verbose=verbose)
    if stage == "figures":
        return render_figures(config_path, verbose=verbose)
    if stage == "qc":
        return render_qc(config_path, verbose=verbose)
    if stage == "all":
        build_risksets(config_path, verbose=verbose)
        add_predictors(config_path, verbose=verbose)
        select_lag(config_path, verbose=verbose)
        fit_models(config_path, verbose=verbose)
        build_tables(config_path, verbose=verbose)
        render_figures(config_path, verbose=verbose)
        render_qc(config_path, verbose=verbose)
        return summarize_outputs(config_path, verbose=verbose)
    raise ValueError(f"Unknown behavior hazard stage: {stage}")
