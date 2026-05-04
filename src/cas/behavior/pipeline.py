"""Behavioral hazard pipeline orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cas.behavior._legacy_support import progress_iterable
from cas.behavior._r_backend import (
    BACKEND_ID as R_BACKEND_ID,
    BACKEND_NOTE as R_BACKEND_NOTE,
    run_r_lag_selection,
    run_r_model_bundle,
)
from cas.behavior.comparisons import MODEL_COMPARISONS
from cas.behavior.config import load_behavior_hazard_config
from cas.behavior.diagnostics import (
    bins_by_subject,
    collinearity_summary,
    convergence_warnings,
    event_rate_summary,
    lag_sensitivity_rows,
)
from cas.behavior.formulas import render_formula
from cas.behavior.lags import select_behavior_lag
from cas.behavior.models import (
    RIDGE_BACKEND_ID,
    STANDARD_BACKEND_ID,
    STANDARD_BACKEND_NOTE,
    coefficient_rows,
    comparison_row,
    fit_registered_model,
    model_metadata,
    odds_ratio_rows,
)
from cas.behavior.predictions import primary_effect_predictions, three_way_predictions, timing_heatmap_predictions
from cas.behavior.predictors import standardize_predictors
from cas.behavior.risksets import build_anchor_riskset
from cas.behavior.summaries import ensure_behavior_directories
from cas.viz.behavior.lag_selection import plot_lag_selection
from cas.viz.behavior.lag_sensitivity import plot_lag_sensitivity
from cas.viz.behavior.primary_effects import plot_primary_effects
from cas.viz.behavior.qc import plot_qc_bars
from cas.viz.behavior.timing_heatmaps import plot_timing_heatmaps


LOGGER = logging.getLogger(__name__)


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", force=True)
    logging.getLogger("cas.behavior").setLevel(level)
    logging.getLogger("cas.behavior._legacy_support").setLevel(level)


def _log(verbose: bool, message: str, *args: object) -> None:
    if verbose:
        LOGGER.info(message, *args)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_parquet(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _empty_csv(path: Path, columns: list[str], *, note: str | None = None) -> Path:
    frame = pd.DataFrame(columns=columns)
    if note:
        frame.loc[0, columns[0]] = note
    return _write_csv(frame, path)


def _write_placeholder_figure(path: Path, *, title: str, message: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _numeric_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(table[column], errors="coerce")


def _finite_variation(series: pd.Series, *, min_unique: int = 2) -> bool:
    finite = series[np.isfinite(series)]
    return bool(finite.nunique(dropna=True) >= min_unique)


def _lag_selection_is_interpretable(scores: pd.DataFrame) -> tuple[bool, str]:
    delta = _numeric_series(scores, "delta_log_likelihood")
    if not _finite_variation(delta):
        return False, "Lag-selection delta log-likelihood values were all non-finite or flat for the active backend."
    return True, ""


def _primary_predictions_are_interpretable(table: pd.DataFrame) -> tuple[bool, str]:
    hazard = _numeric_series(table, "predicted_hazard")
    if not _finite_variation(hazard, min_unique=5):
        return False, "Primary prediction curves collapsed to a constant or near-constant hazard."
    boundary_share = float(np.mean((hazard <= 1e-6) | (hazard >= 1.0 - 1e-6))) if len(hazard) else 1.0
    if boundary_share >= 0.98:
        return False, "Primary prediction curves were saturated at probability boundaries for nearly all plotted points."
    return True, ""


def _heatmap_predictions_are_interpretable(table: pd.DataFrame) -> tuple[bool, str]:
    hazard = _numeric_series(table, "predicted_hazard")
    if not _finite_variation(hazard, min_unique=20):
        return False, "Heatmap predictions did not vary enough to support an interpretable surface."
    boundary_share = float(np.mean((hazard <= 1e-6) | (hazard >= 1.0 - 1e-6))) if len(hazard) else 1.0
    if boundary_share >= 0.98:
        return False, "Heatmap predictions were saturated at probability boundaries for nearly all grid cells."
    return True, ""


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
    _log(verbose, "[behavior hazard] Wrote risk sets to %s", paths["risksets"])
    return outputs


def add_predictors(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    riskset_paths = _riskset_paths(paths)
    if not all(path.exists() for path in riskset_paths.values()):
        build_risksets(config_path, verbose=verbose)
    _log(verbose, "[behavior hazard] Loading risk sets for predictor engineering")
    fpp = pd.read_parquet(riskset_paths["fpp"])
    spp = pd.read_parquet(riskset_paths["spp"])
    pooled = pd.read_parquet(riskset_paths["pooled"])
    _log(verbose, "[behavior hazard] Adding candidate lags and standardizing predictors")
    fpp_pred, spp_pred, pooled_pred, scaling = standardize_predictors(
        fpp,
        spp,
        pooled,
        config=config,
        verbose=verbose,
    )
    outputs = _predictor_paths(paths)
    _write_parquet(fpp_pred, outputs["fpp"])
    _write_parquet(spp_pred, outputs["spp"])
    _write_parquet(pooled_pred, outputs["pooled"])
    _write_csv(scaling, outputs["scaling"])
    _log(verbose, "[behavior hazard] Wrote predictor tables to %s", paths["predictors"])
    return outputs


def select_lag(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    fpp = pd.read_parquet(predictor_paths["fpp"])
    pooled = pd.read_parquet(predictor_paths["pooled"])
    _log(verbose, "[behavior hazard] Selecting lag across %d candidates", len(config.candidate_lags_ms))
    score_path = paths["lag_selection"] / "candidate_lag_scores.csv"
    selected_path = paths["lag_selection"] / "selected_lag.json"
    run_r_lag_selection(
        fpp_table=fpp,
        pooled_table=pooled,
        candidate_lags_ms=config.candidate_lags_ms,
        score_path=score_path,
        selected_path=selected_path,
        family_summary_path=paths["lag_selection"] / "family_lag_summary.csv",
        family_rankings_path=paths["lag_selection"] / "family_lag_rankings.csv",
        model_diagnostics_path=paths["lag_selection"] / "family_model_diagnostics.csv",
        selector_comparison_path=paths["lag_selection"] / "lag_selector_comparison.csv",
        lag_sensitivity_path=paths["diagnostics"] / "lag_sensitivity.csv",
        verbose=verbose,
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    baseline_metadata = {
        "model_id": "A0_timing",
        "backend": R_BACKEND_ID,
        "notes": [R_BACKEND_NOTE],
    }
    _write_json(paths["lag_selection_models"] / "A0_timing.json", baseline_metadata)
    for lag_ms in config.candidate_lags_ms:
        _write_json(
            paths["lag_selection_models"] / f"A3_joint_information_lag_{int(lag_ms)}.json",
            {
                "model_id": "A3_joint_information",
                "candidate_lag_ms": int(lag_ms),
                "backend": R_BACKEND_ID,
                "notes": [R_BACKEND_NOTE],
            },
        )
    _log(
        verbose,
        "[behavior hazard] Selected family-wise lags: A=%d ms, B=%d ms, C=%d ms",
        int(selected["best_lag_A_ms"]),
        int(selected["best_lag_B_ms"]),
        int(selected["best_lag_C_ms"]),
    )
    return {"scores": score_path, "selected": selected_path}


def _load_selected_lag_payload(
    paths: dict[str, Path],
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> dict[str, object]:
    lag_path = paths["lag_selection"] / "selected_lag.json"
    if not lag_path.exists():
        select_lag(config_path, verbose=verbose)
    return json.loads(lag_path.read_text(encoding="utf-8"))


def _load_selected_lag(paths: dict[str, Path], config_path: str | Path, *, verbose: bool = False) -> int:
    selected = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    return int(selected["selected_lag_ms"])


def _family_lags(selected_payload: dict[str, object]) -> dict[str, int]:
    family_lags_raw = dict(selected_payload.get("family_lags") or {})
    if family_lags_raw:
        return {str(key): int(value) for key, value in family_lags_raw.items()}
    if "selected_lag_ms" in selected_payload:
        selected_lag_ms = int(selected_payload["selected_lag_ms"])
        return {"A": selected_lag_ms, "B": selected_lag_ms, "C": selected_lag_ms}
    return {
        "A": int(selected_payload["best_lag_A_ms"]),
        "B": int(selected_payload["best_lag_B_ms"]),
        "C": int(selected_payload["best_lag_C_ms"]),
    }


def _load_predictor_tables(
    paths: dict[str, Path],
    config_path: str | Path,
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictor_paths = _predictor_paths(paths)
    if not predictor_paths["fpp"].exists():
        add_predictors(config_path, verbose=verbose)
    return (
        pd.read_parquet(predictor_paths["fpp"]),
        pd.read_parquet(predictor_paths["spp"]),
        pd.read_parquet(predictor_paths["pooled"]),
    )


def _r_final_model_specs(_config: object, family_lags: dict[str, int]) -> list[dict[str, object]]:
    model_specs: list[dict[str, object]] = []
    spec_rows = [
        ("primary_fpp", "FPP", "A0_timing", "A", family_lags["A"]),
        ("primary_fpp", "FPP", "A1_information_rate", "A", family_lags["A"]),
        ("primary_fpp", "FPP", "A2_expected_cum_info", "A", family_lags["A"]),
        ("primary_fpp", "FPP", "A3_joint_information", "A", family_lags["A"]),
        ("fpp_spp_control", "pooled_fpp_spp", "B1_shared_information", "B", family_lags["B"]),
        ("fpp_spp_control", "pooled_fpp_spp", "B2_anchor_x_information", "B", family_lags["B"]),
        ("timing_moderation", "FPP", "C1_onset_x_rate", "C", family_lags["C"]),
        ("timing_moderation", "FPP", "C2_offset_x_rate", "C", family_lags["C"]),
    ]
    for group_name, dataset_name, model_id, family_name, lag_ms in spec_rows:
        model_specs.append(
            {
                "group": group_name,
                "dataset": dataset_name,
                "model_id": model_id,
                "public_model_id": model_id,
                "family": family_name,
                "lag_ms": int(lag_ms),
                "formula": render_formula(model_id, lag_ms=lag_ms),
            }
        )
    return model_specs


def _r_table_model_specs(_config: object, family_lags: dict[str, int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_specs = _r_final_model_specs(_config, family_lags)
    c_family_lag = int(family_lags["C"])
    model_specs.append(
        {
            "group": "timing_moderation",
            "dataset": "FPP",
            "model_id": "A3_joint_information__C_family_reference",
            "public_model_id": "A3_joint_information",
            "family": "C",
            "lag_ms": c_family_lag,
            "formula": render_formula("A3_joint_information", lag_ms=c_family_lag),
        }
    )
    comparison_specs: list[dict[str, object]] = []
    for reduced_id, full_id in MODEL_COMPARISONS["primary_fpp"]:
        comparison_specs.append(
            {
                "group": "primary_fpp",
                "dataset": "FPP",
                "family": "A",
                "reduced": reduced_id,
                "full": full_id,
                "public_reduced": reduced_id,
                "public_full": full_id,
                "lag_ms": int(family_lags["A"]),
            }
        )
    for reduced_id, full_id in MODEL_COMPARISONS["fpp_spp_control"]:
        comparison_specs.append(
            {
                "group": "fpp_spp_control",
                "dataset": "pooled_fpp_spp",
                "family": "B",
                "reduced": reduced_id,
                "full": full_id,
                "public_reduced": reduced_id,
                "public_full": full_id,
                "lag_ms": int(family_lags["B"]),
            }
        )
    comparison_specs.extend(
        [
            {
                "group": "timing_moderation",
                "dataset": "FPP",
                "family": "C",
                "reduced": "A3_joint_information__C_family_reference",
                "full": "C1_onset_x_rate",
                "public_reduced": "A3_joint_information",
                "public_full": "C1_onset_x_rate",
                "lag_ms": c_family_lag,
            },
            {
                "group": "timing_moderation",
                "dataset": "FPP",
                "family": "C",
                "reduced": "A3_joint_information__C_family_reference",
                "full": "C2_offset_x_rate",
                "public_reduced": "A3_joint_information",
                "public_full": "C2_offset_x_rate",
                "lag_ms": c_family_lag,
            },
        ]
    )
    return model_specs, comparison_specs


def _write_model_jsons_from_r_outputs(
    *,
    model_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    output_dir_by_group = {
        "primary_fpp": paths["models_primary"],
        "fpp_spp_control": paths["models_control"],
        "timing_moderation": paths["models_timing"],
        "exploratory": paths["models_exploratory"],
    }
    coefficient_groups = {
        model_id: frame.copy()
        for model_id, frame in coefficients.groupby("model_id", sort=False)
    }
    for row in model_metrics.to_dict("records"):
        group_name = str(row.get("group", ""))
        model_id = str(row.get("model_id", ""))
        output_dir = output_dir_by_group[group_name]
        output_dir.mkdir(parents=True, exist_ok=True)
        coef_frame = coefficient_groups.get(model_id, pd.DataFrame())
        payload = {
            "model_id": str(row.get("public_model_id", model_id)),
            "family": str(row.get("family", "")),
            "requested_formula": row.get("formula"),
            "fitted_formula": row.get("formula"),
            "n_rows": int(row.get("n_rows", 0) or 0),
            "n_events": int(row.get("n_events", 0) or 0),
            "log_likelihood": row.get("log_likelihood"),
            "aic": row.get("aic"),
            "bic": row.get("bic"),
            "converged": bool(row.get("converged", False)),
            "backend": row.get("backend", R_BACKEND_ID),
            "random_effects_requested": True,
            "random_effects_applied": True,
            "regularization_alpha": None,
            "selected_lag_ms": int(row.get("lag_ms", coef_frame["selected_lag_ms"].iloc[0] if not coef_frame.empty else 0)),
            "notes": [str(row.get("notes", R_BACKEND_NOTE))],
            "warnings": [w for w in [str(row.get("warnings", ""))] if w],
            "coefficients": coef_frame.to_dict(orient="records"),
        }
        _write_json(output_dir / f"{payload['model_id']}.json", payload)


def fit_models(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    family_lags = _family_lags(selected_payload)
    fpp, _spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    model_specs = _r_final_model_specs(config, family_lags)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_model_coefficients.csv"
    run_r_model_bundle(
        fpp_table=fpp,
        pooled_table=pooled,
        selected_lags=family_lags,
        candidate_lags_ms=config.candidate_lags_ms,
        model_specs=model_specs,
        comparison_specs=[],
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=paths["logs"] / "r_model_comparisons.csv",
        convergence_path=paths["logs"] / "r_convergence_warnings.csv",
        lag_sensitivity_path=None,
        verbose=verbose,
    )
    model_metrics = pd.read_csv(model_metrics_path)
    coefficients = pd.read_csv(coefficient_path)
    _write_model_jsons_from_r_outputs(model_metrics=model_metrics, coefficients=coefficients, paths=paths)
    return {"models_root": paths["models"]}


def build_tables(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    selected_payload = _load_selected_lag_payload(paths, config_path, verbose=verbose)
    family_lags = _family_lags(selected_payload)
    fpp, _spp, pooled = _load_predictor_tables(paths, config_path, verbose=verbose)
    required_model_jsons = [
        paths["models_primary"] / "A0_timing.json",
        paths["models_primary"] / "A1_information_rate.json",
        paths["models_primary"] / "A2_expected_cum_info.json",
        paths["models_primary"] / "A3_joint_information.json",
        paths["models_control"] / "B1_shared_information.json",
        paths["models_control"] / "B2_anchor_x_information.json",
        paths["models_timing"] / "C1_onset_x_rate.json",
        paths["models_timing"] / "C2_offset_x_rate.json",
    ]
    if not all(path.exists() for path in required_model_jsons):
        _log(verbose, "[behavior hazard] Model JSON outputs were missing; fitting models before table generation")
        fit_models(config_path, verbose=verbose)
    model_specs, comparison_specs = _r_table_model_specs(config, family_lags)
    model_metrics_path = paths["logs"] / "r_model_metrics.csv"
    coefficient_path = paths["logs"] / "r_table_coefficients.csv"
    comparison_path = paths["logs"] / "r_table_comparisons.csv"
    convergence_path = paths["diagnostics"] / "convergence_warnings.csv"
    lag_sensitivity_path = paths["diagnostics"] / "lag_sensitivity.csv"
    figure_prediction_path = paths["tables"] / "figure_predictions.csv"
    timing_heatmap_path = paths["tables"] / "timing_heatmap_predictions.csv"
    three_way_path = paths["tables"] / "three_way_heatmap_predictions.csv"
    _log(verbose, "[behavior hazard] Building diagnostics and prediction tables with one R GLMM pass")
    run_r_model_bundle(
        fpp_table=fpp,
        pooled_table=pooled,
        selected_lags=family_lags,
        candidate_lags_ms=config.candidate_lags_ms,
        model_specs=model_specs,
        comparison_specs=comparison_specs,
        model_metrics_path=model_metrics_path,
        coefficient_path=coefficient_path,
        comparison_path=comparison_path,
        convergence_path=convergence_path,
        lag_sensitivity_path=None,
        figure_prediction_path=figure_prediction_path,
        timing_heatmap_path=timing_heatmap_path,
        three_way_path=three_way_path,
        verbose=verbose,
    )
    coefficients = pd.read_csv(coefficient_path)
    coefficients = coefficients.loc[~coefficients["model_id"].astype(str).str.contains("__C_family_reference", regex=False)].copy()
    comparisons = pd.read_csv(comparison_path)
    _write_csv(comparisons, paths["tables"] / "model_comparisons.csv")
    _write_csv(coefficients, paths["tables"] / "coefficient_summary.csv")
    _write_csv(odds_ratio_rows(coefficients), paths["tables"] / "odds_ratios.csv")
    _write_csv(event_rate_summary(pooled), paths["tables"] / "event_rate_summary.csv")
    _write_csv(bins_by_subject(pooled), paths["tables"] / "bins_by_subject.csv")
    collinearity_columns = [
        "z_time_from_partner_onset_s",
        "z_time_from_partner_offset_s",
        "z_time_from_partner_offset_s_squared",
        f"z_information_rate_lag_{family_lags['A']}",
        f"z_prop_expected_cum_info_lag_{family_lags['A']}",
    ]
    _write_csv(collinearity_summary(pooled, collinearity_columns), paths["tables"] / "collinearity_summary.csv")
    if not lag_sensitivity_path.exists():
        _empty_csv(
            lag_sensitivity_path,
            ["candidate_lag_ms", "term", "predictor", "estimate", "ci_low", "ci_high", "backend"],
            note="Lag sensitivity was not available from lag selection.",
        )

    if config.diagnostics_enabled("leave_one_subject_out"):
        _empty_csv(
            paths["diagnostics"] / "loo_subject_summary.csv",
            ["left_out_subject", "term", "estimate", "ci_low", "ci_high", "notes"],
            note="Leave-one-subject-out is configured but not implemented for the current R GLMM backend.",
        )
    else:
        _empty_csv(
            paths["diagnostics"] / "loo_subject_summary.csv",
            ["left_out_subject", "term", "estimate", "ci_low", "ci_high", "notes"],
            note="Disabled by config: behavior.hazard.diagnostics.leave_one_subject_out = false",
        )
    summarize_outputs(config_path, verbose=verbose)
    return {"tables": paths["tables"] / "model_comparisons.csv"}


def summarize_outputs(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "model_comparisons.csv").exists():
        build_tables(config_path, verbose=verbose)
    summary_path = paths["logs"] / "summary_manifest.json"
    payload = {
        "hazard_root": str(paths["hazard_root"]),
        "tables": str(paths["tables"]),
        "diagnostics": str(paths["diagnostics"]),
        "figures_main": str(paths["figures_main"]),
        "figures_supp": str(paths["figures_supp"]),
        "figures_qc": str(paths["figures_qc"]),
        "backend": R_BACKEND_ID,
        "backend_notes": [R_BACKEND_NOTE],
        "possible_backends": [R_BACKEND_ID, STANDARD_BACKEND_ID, RIDGE_BACKEND_ID],
    }
    _write_json(summary_path, payload)
    _log(verbose, "[behavior hazard] Wrote summary manifest to %s", summary_path)
    return {"summary": summary_path}


def render_figures(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "figure_predictions.csv").exists():
        build_tables(config_path, verbose=verbose)
    _log(verbose, "[behavior hazard] Rendering main and supplementary figures")
    lag_scores = pd.read_csv(paths["lag_selection"] / "candidate_lag_scores.csv")
    selected = json.loads((paths["lag_selection"] / "selected_lag.json").read_text(encoding="utf-8"))
    lag_ms = int(selected["selected_lag_ms"])
    primary_predictions = pd.read_csv(paths["tables"] / "figure_predictions.csv")
    timing_grid = pd.read_csv(paths["tables"] / "timing_heatmap_predictions.csv")
    onset_grid = timing_grid.loc[timing_grid["panel"].astype(str) == "A"].copy()
    offset_grid = timing_grid.loc[timing_grid["panel"].astype(str) == "B"].copy()
    lag_sensitivity = pd.read_csv(paths["diagnostics"] / "lag_sensitivity.csv")
    event_rates = pd.read_csv(paths["tables"] / "event_rate_summary.csv")
    bins = pd.read_csv(paths["tables"] / "bins_by_subject.csv")
    collinearity = pd.read_csv(paths["tables"] / "collinearity_summary.csv")
    convergence = pd.read_csv(paths["diagnostics"] / "convergence_warnings.csv")
    three_way_table = pd.read_csv(paths["tables"] / "three_way_heatmap_predictions.csv")

    lag_ok, lag_message = _lag_selection_is_interpretable(lag_scores)
    primary_ok, primary_message = _primary_predictions_are_interpretable(primary_predictions)
    timing_ok, timing_message = _heatmap_predictions_are_interpretable(timing_grid)
    outputs = {
        "lag_selection": (
            plot_lag_selection(lag_scores, paths["figures_main"] / "fig01_lag_selection.png")
            if lag_ok
            else _write_placeholder_figure(
                paths["figures_main"] / "fig01_lag_selection.png",
                title="Behavioral information lag selection",
                message=lag_message,
            )
        ),
        "primary_effects": (
            plot_primary_effects(primary_predictions, paths["figures_main"] / "fig02_primary_information_effects.png")
            if primary_ok
            else _write_placeholder_figure(
                paths["figures_main"] / "fig02_primary_information_effects.png",
                title="Primary information effects",
                message=primary_message,
            )
        ),
        "timing_heatmaps": (
            plot_timing_heatmaps(onset_grid, offset_grid, paths["figures_main"] / "fig03_timing_information_heatmaps.png")
            if timing_ok
            else _write_placeholder_figure(
                paths["figures_main"] / "fig03_timing_information_heatmaps.png",
                title="Timing-dependent information effects",
                message=timing_message,
            )
        ),
    }
    plot_lag_sensitivity(lag_sensitivity, selected_lag_ms=lag_ms, output_path=paths["figures_supp"] / "figS01_lag_sensitivity.png")
    if timing_ok:
        plot_timing_heatmaps(onset_grid, offset_grid, paths["figures_supp"] / "figS02_extra_timing_maps.png")
    else:
        _write_placeholder_figure(
            paths["figures_supp"] / "figS02_extra_timing_maps.png",
            title="Supplementary timing maps",
            message=timing_message,
        )
    if config.figure_enabled("main", "three_way_interaction") and not three_way_table.empty:
        plot_timing_heatmaps(
            three_way_table.loc[three_way_table["anchor_type"].astype(str) == "FPP"].assign(panel="A"),
            three_way_table.loc[three_way_table["anchor_type"].astype(str) == "SPP"].assign(panel="B"),
            paths["figures_main"] / "fig04_three_way_interaction.png",
        )
    elif config.figure_enabled("main", "three_way_interaction"):
        _write_placeholder_figure(
            paths["figures_main"] / "fig04_three_way_interaction.png",
            title="Three-way interaction",
            message="Exploratory three-way figure not available because exploratory_three_way is disabled or produced no prediction grid.",
        )
    if config.figure_enabled("supplementary", "three_way_interaction") and not three_way_table.empty:
        plot_timing_heatmaps(
            three_way_table.loc[three_way_table["anchor_type"].astype(str) == "FPP"].assign(panel="A"),
            three_way_table.loc[three_way_table["anchor_type"].astype(str) == "SPP"].assign(panel="B"),
            paths["figures_supp"] / "figS03_three_way_interaction.png",
        )
    elif config.figure_enabled("supplementary", "three_way_interaction"):
        _write_placeholder_figure(
            paths["figures_supp"] / "figS03_three_way_interaction.png",
            title="Supplementary three-way interaction",
            message="Exploratory three-way figure not available because exploratory_three_way is disabled or produced no prediction grid.",
        )
    if config.figure_enabled("supplementary", "leave_one_subject_out"):
        loo_path = paths["diagnostics"] / "loo_subject_summary.csv"
        if loo_path.exists():
            loo_table = pd.read_csv(loo_path)
            if {"left_out_subject", "estimate"} <= set(loo_table.columns) and not loo_table.empty:
                plot_qc_bars(
                    loo_table.head(20),
                    x="left_out_subject",
                    y="estimate",
                    title="Leave-one-subject-out",
                    output_path=paths["figures_supp"] / "figS04_leave_one_subject_out.png",
                )
            else:
                _write_placeholder_figure(
                    paths["figures_supp"] / "figS04_leave_one_subject_out.png",
                    title="Leave-one-subject-out",
                    message="Leave-one-subject-out diagnostics are disabled or unavailable for the current backend.",
                )
    return outputs


def render_qc(config_path: str | Path, *, verbose: bool = False) -> dict[str, Path]:
    config = load_behavior_hazard_config(config_path)
    paths = ensure_behavior_directories(config)
    if not (paths["tables"] / "event_rate_summary.csv").exists():
        build_tables(config_path, verbose=verbose)
    _log(verbose, "[behavior hazard] Rendering QC figures")
    event_rates = pd.read_csv(paths["tables"] / "event_rate_summary.csv")
    bins = pd.read_csv(paths["tables"] / "bins_by_subject.csv")
    collinearity = pd.read_csv(paths["tables"] / "collinearity_summary.csv")
    convergence = pd.read_csv(paths["diagnostics"] / "convergence_warnings.csv")
    outputs = {
        "event_rates": plot_qc_bars(event_rates, x="anchor_type", y="event_rate", title="Event rates", output_path=paths["figures_qc"] / "qc_event_rates.png"),
        "bins_by_subject": plot_qc_bars(bins, x="subject", y="n_bins", title="Bins by subject", output_path=paths["figures_qc"] / "qc_bins_by_subject.png"),
        "collinearity": plot_qc_bars(collinearity, x="term", y="vif", title="Collinearity", output_path=paths["figures_qc"] / "qc_collinearity.png"),
        "model_convergence": plot_qc_bars(
            convergence.assign(count=1).groupby("model_id", as_index=False)["count"].sum()
            if not convergence.empty
            else pd.DataFrame({"model_id": [], "count": []}),
            x="model_id",
            y="count",
            title="Model convergence warnings",
            output_path=paths["figures_qc"] / "qc_model_convergence.png",
        ),
    }
    return outputs


def run_behavior_hazard_stage(
    stage: str,
    *,
    config_path: str | Path,
    verbose: bool = False,
) -> dict[str, Path]:
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
