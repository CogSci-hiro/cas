from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

from cas.neural_hazard import fpp_spp_pipeline as base
from cas.neural_hazard.renyi import compute_renyi_entropy, discover_state_probability_columns


@dataclass(frozen=True, slots=True)
class NeuralHazardFppSppRenyiAlphaConfig:
    raw: dict
    config_path: Path


@dataclass(frozen=True, slots=True)
class NeuralHazardFppSppRenyiAlphaResult:
    out_dir: Path
    summary_json_path: Path


def load_neural_hazard_fpp_spp_renyi_alpha_config(path: Path) -> NeuralHazardFppSppRenyiAlphaConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected mapping config.")
    return NeuralHazardFppSppRenyiAlphaConfig(raw=payload, config_path=path.resolve())


def _safe_alpha_label(alpha: float) -> str:
    return f"alpha_{str(alpha).replace('.', 'p')}"


def _resolve_with_base(path_text: str, *, project_root: Path, derivatives_root: Path | None) -> Path:
    return base._resolve_config_path(path_text, project_root=project_root, derivatives_root=derivatives_root)


def _resolve_out(path_text: str, *, project_root: Path, derivatives_root: Path | None) -> Path:
    return base._resolve_output_path(path_text, project_root=project_root, derivatives_root=derivatives_root)


def _load_neural_features_with_state_probabilities(
    *,
    neural_features_path: Path,
    raw_config: dict,
    project_root: Path,
    derivatives_root: Path | None,
    scratch_dir: Path,
) -> pd.DataFrame:
    neural = base._load_table(neural_features_path)
    if discover_state_probability_columns(neural):
        return neural

    glhmm_path_text = str(raw_config.get("paths", {}).get("glhmm_output_dir", "models/glhmm"))
    glhmm_output_dir = _resolve_with_base(
        glhmm_path_text,
        project_root=project_root,
        derivatives_root=derivatives_root,
    )
    if not glhmm_output_dir.exists():
        raise ValueError(
            "Missing HMM state probability columns required for Renyi entropy, and the fallback "
            f"GLHMM output directory was not found: {glhmm_output_dir}"
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="renyi_enriched_neural_features_",
        suffix=".parquet",
        dir=scratch_dir,
        delete=False,
    ) as handle:
        rebuilt_path = Path(handle.name)
    try:
        base.build_entropy_features_table_from_glhmm_output(
            glhmm_output_dir,
            rebuilt_path,
            instability_window_s=base.DEFAULT_INSTABILITY_WINDOW_S,
        )
        neural = base._load_table(rebuilt_path)
    finally:
        rebuilt_path.unlink(missing_ok=True)

    prob_cols = discover_state_probability_columns(neural)
    if not prob_cols:
        raise ValueError(
            "Rebuilt GLHMM entropy features still did not contain HMM state probability columns required "
            "for Renyi entropy."
        )
    return neural


def _compute_renyi_entropy_allow_missing_rows(
    state_probabilities: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
    alpha_one_tolerance: float,
) -> np.ndarray:
    probs = np.asarray(state_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("state_probabilities must be a 2D array (n_samples, n_states).")
    out = np.full(probs.shape[0], np.nan, dtype=float)
    valid_rows = np.isfinite(probs).all(axis=1)
    if not np.any(valid_rows):
        return out
    out[valid_rows] = compute_renyi_entropy(
        probs[valid_rows],
        alpha,
        epsilon=epsilon,
        alpha_one_tolerance=alpha_one_tolerance,
    )
    return out


def _mk_base_cfg(
    raw: dict,
    *,
    project_root: Path,
    derivatives_root: Path | None,
    out_dir: Path,
    lag_grid_ms: tuple[int, ...] | None = None,
) -> base.NeuralHazardFppSppConfig:
    p = raw["paths"]
    a = raw["analysis"]
    circular_shift = dict(raw.get("circular_shift") or {})
    return base.NeuralHazardFppSppConfig(
        fpp_risk_set_path=_resolve_with_base(
            str(p["fpp_risk_set"]), project_root=project_root, derivatives_root=derivatives_root
        ),
        spp_risk_set_path=_resolve_with_base(
            str(p["spp_risk_set"]), project_root=project_root, derivatives_root=derivatives_root
        ),
        neural_features_path=_resolve_with_base(
            str(p["neural_features"]), project_root=project_root, derivatives_root=derivatives_root
        ),
        out_dir=out_dir,
        bin_width_s=float(a["bin_width_s"]),
        lag_grid_ms=tuple(int(x) for x in (lag_grid_ms if lag_grid_ms is not None else a["lag_grid_ms"])),
        pca_input_columns=tuple(raw["pca"]["input_columns"]),
        nearest_merge_tolerance_s=float(a["bin_width_s"]) / 2.0,
        timing_zscore_scope=str(a.get("timing_zscore_scope", "global")),
        neural_zscore_scope=str(a.get("feature_zscore_scope", "subject_run")),
        pca_pc1_warning_threshold=float(raw["pca"].get("minimum_pc1_explained_variance", 0.40)),
        pca_n_components=int(raw["pca"].get("n_components", 1)),
        lag_selection_criterion="delta_loglik",
        avoid_zero_lag=bool(a.get("avoid_zero_lag", True)),
        minimum_circular_shift_duration_s=float(circular_shift.get("minimum_shift_s", 5.0)),
        n_circular_shift_permutations=int(circular_shift.get("n_permutations", 100)),
        circular_shift_seed=int(circular_shift.get("random_seed", 12345)),
        overwrite=True,
        verbose=True,
    )


def _ensure_default_config(raw: dict) -> dict:
    payload = json.loads(json.dumps(raw))
    renyi = payload.setdefault("renyi", {})
    circular_shift = payload.setdefault("circular_shift", {})
    diagnostics = payload.setdefault("diagnostics", {})
    motor = payload.setdefault("motor_proximal_sensitivity", {})
    plots = payload.setdefault("plots", {})

    renyi.setdefault(
        "alpha_grid",
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
    )
    renyi.setdefault("alpha_one_tolerance", 1.0e-6)
    renyi.setdefault("probability_epsilon", 1.0e-12)
    renyi.setdefault("primary_selection_metric", "m2_vs_m1_delta_loglik")
    renyi.setdefault("include_shannon_baseline", True)
    renyi.setdefault("edge_warning_enabled", True)
    renyi.setdefault("fixed_lag_sensitivity_ms", [50, 200])

    circular_shift.setdefault("enabled", True)
    circular_shift.setdefault("n_permutations", 100)
    circular_shift.setdefault("minimum_shift_s", 5.0)
    circular_shift.setdefault("random_seed", 12345)
    circular_shift.setdefault("run_for_best_alpha_only", True)
    circular_shift.setdefault("run_for_shannon", True)
    circular_shift.setdefault("shift_scope", "subject_run")
    circular_shift.setdefault("shift_entropy_only", True)
    circular_shift.setdefault("p_value_mode_interaction_beta", "two_sided")

    diagnostics.setdefault("enabled", True)

    motor.setdefault("enabled", True)
    motor.setdefault("minimum_lag_ms", 150)
    motor.setdefault("run_circular_shift_for_best", False)

    plots.setdefault("alpha_axis_scale", "linear")
    return payload


def _make_neural_alpha_table(
    neural: pd.DataFrame,
    *,
    entropy: np.ndarray,
    pca_columns: tuple[str, ...],
) -> pd.DataFrame:
    columns = ["subject_id", "run_id", "time_s", *[column for column in pca_columns if column in neural.columns]]
    out = neural.loc[:, list(dict.fromkeys(columns))].copy()
    out["entropy"] = np.asarray(entropy, dtype=float)
    return out


def _model_warning_text(model: base.FittedGlmModel) -> str:
    parts = [*model.warnings_text]
    if model.error_message:
        parts.append("ERROR: " + model.error_message)
    return " | ".join(part for part in parts if part)


def _shannon_validation_row(
    *,
    alpha: float,
    entropy: np.ndarray,
    neural: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    del tolerance
    has_existing = "entropy" in neural.columns
    existing = pd.to_numeric(neural["entropy"], errors="coerce").to_numpy(dtype=float) if has_existing else np.array([])
    if not has_existing:
        return {
            "alpha": float(alpha),
            "existing_entropy_present": False,
            "n_compared": 0,
            "max_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "mean_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "median_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "correlation_alpha1_vs_existing_entropy": np.nan,
        }
    finite = np.isfinite(entropy) & np.isfinite(existing)
    if not finite.any():
        return {
            "alpha": float(alpha),
            "existing_entropy_present": True,
            "n_compared": 0,
            "max_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "mean_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "median_abs_difference_alpha1_vs_existing_entropy": np.nan,
            "correlation_alpha1_vs_existing_entropy": np.nan,
        }
    diff = np.abs(entropy[finite] - existing[finite])
    corr = np.nan
    if finite.sum() >= 2:
        corr = float(np.corrcoef(entropy[finite], existing[finite])[0, 1])
    return {
        "alpha": float(alpha),
        "existing_entropy_present": True,
        "n_compared": int(finite.sum()),
        "max_abs_difference_alpha1_vs_existing_entropy": float(np.nanmax(diff)),
        "mean_abs_difference_alpha1_vs_existing_entropy": float(np.nanmean(diff)),
        "median_abs_difference_alpha1_vs_existing_entropy": float(np.nanmedian(diff)),
        "correlation_alpha1_vs_existing_entropy": corr,
    }


def _describe_series(
    *,
    alpha: float,
    alpha_label: str,
    variable: str,
    values: pd.Series,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {
            "alpha": float(alpha),
            "alpha_label": alpha_label,
            "variable": variable,
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }
    return {
        "alpha": float(alpha),
        "alpha_label": alpha_label,
        "variable": variable,
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "q01": float(finite.quantile(0.01)),
        "q05": float(finite.quantile(0.05)),
        "q25": float(finite.quantile(0.25)),
        "median": float(finite.quantile(0.50)),
        "q75": float(finite.quantile(0.75)),
        "q95": float(finite.quantile(0.95)),
        "q99": float(finite.quantile(0.99)),
        "max": float(finite.max()),
        "skew": float(stats.skew(finite.to_numpy(dtype=float), bias=False)) if finite.size >= 3 else np.nan,
        "kurtosis": float(stats.kurtosis(finite.to_numpy(dtype=float), fisher=True, bias=False))
        if finite.size >= 4
        else np.nan,
    }


def _fit_alpha_on_lag_grid(
    *,
    alpha: float,
    alpha_label: str,
    neural_alpha: pd.DataFrame,
    riskset: pd.DataFrame,
    cfg: base.NeuralHazardFppSppConfig,
) -> dict[str, Any]:
    merged = base._merge_lagged_neural_features(riskset, neural_alpha, cfg)
    merged = base._zscore_all_features(merged, cfg)
    merged, pca_variance, pca_warnings = base._add_instability_pc1_by_lag(merged, cfg)
    lag_selection = base._run_fpp_lag_selection(merged, cfg)
    lag_selection["alpha"] = float(alpha)
    lag_selection["alpha_label"] = alpha_label
    selected_lag_ms = base._select_entropy_lag(lag_selection, cfg)
    selected_row = lag_selection.loc[pd.to_numeric(lag_selection["lag_ms"], errors="coerce") == int(selected_lag_ms)].iloc[0]
    pooled = base._fit_pooled_models(merged, selected_lag_ms)
    m2_table = pooled["coefficients"]["M2_entropy"]
    entropy_term = f"entropy_lag_{selected_lag_ms}ms_z"
    interaction_term = f"C(anchor_type)[T.SPP]:{entropy_term}"
    model_table = pooled["model_comparison"].copy()
    model_rows = model_table.loc[
        model_table["row_type"].astype(str) == "model",
        ["model_name", "n_rows", "n_events", "log_likelihood", "aic", "bic", "converged", "warnings"],
    ].copy()
    pc1_variance = np.nan
    if not pca_variance.empty:
        hit = pca_variance.loc[
            (pd.to_numeric(pca_variance["lag_ms"], errors="coerce") == int(selected_lag_ms))
            & (pd.to_numeric(pca_variance["component"], errors="coerce") == 1)
        ]
        if not hit.empty:
            pc1_variance = float(hit["explained_variance_ratio"].iloc[0])
    return {
        "alpha": float(alpha),
        "alpha_label": alpha_label,
        "merged": merged,
        "lag_selection": lag_selection,
        "selected_lag_ms": int(selected_lag_ms),
        "selected_lag_row": {
            "alpha": float(alpha),
            "alpha_label": alpha_label,
            "selected_lag_ms": int(selected_lag_ms),
            "selection_metric": "delta_loglik",
            "selected_metric_value": float(selected_row["delta_loglik"]),
            "avoid_zero_lag_applied": bool(cfg.avoid_zero_lag),
        },
        "pca_variance": pca_variance,
        "pca_warnings": pca_warnings,
        "pooled": pooled,
        "coefficients": {
            name: table.assign(alpha=float(alpha), alpha_label=alpha_label, selected_lag_ms=int(selected_lag_ms))
            for name, table in pooled["coefficients"].items()
        },
        "model_rows": model_rows.assign(alpha=float(alpha), alpha_label=alpha_label, selected_lag_ms=int(selected_lag_ms)),
        "pairwise_rows": pd.DataFrame(
            [
                {"comparison": "M1_instability vs M0_timing", **pooled["pairwise"]["M1_instability_vs_M0_timing"]},
                {"comparison": "M2_renyi_entropy vs M1_instability", **pooled["pairwise"]["M2_entropy_vs_M1_instability"]},
            ]
        ).assign(alpha=float(alpha), alpha_label=alpha_label, selected_lag_ms=int(selected_lag_ms)),
        "summary_row": {
            "alpha": float(alpha),
            "alpha_label": alpha_label,
            "selected_lag_ms": int(selected_lag_ms),
            "pc1_explained_variance_selected_lag": pc1_variance,
            "entropy_beta_fpp": float(base._extract_term_stats(pooled["models"]["M2_entropy"], entropy_term)["estimate"]),
            "entropy_interaction_beta_spp_minus_fpp": float(
                base._extract_term_stats(pooled["models"]["M2_entropy"], interaction_term)["estimate"]
            ),
            "m2_vs_m1_delta_loglik": float(pooled["pairwise"]["M2_entropy_vs_M1_instability"]["delta_loglik"]),
            "m2_vs_m1_lrt_p_value": float(pooled["pairwise"]["M2_entropy_vs_M1_instability"]["p_value"]),
            "m2_aic": float(model_rows.loc[model_rows["model_name"] == "M2_entropy", "aic"].iloc[0]),
            "m2_bic": float(model_rows.loc[model_rows["model_name"] == "M2_entropy", "bic"].iloc[0]),
            "n_rows": int(model_rows.loc[model_rows["model_name"] == "M2_entropy", "n_rows"].iloc[0]),
            "n_events": int(model_rows.loc[model_rows["model_name"] == "M2_entropy", "n_events"].iloc[0]),
            "converged": bool(model_rows.loc[model_rows["model_name"] == "M2_entropy", "converged"].iloc[0]),
            "warnings": str(model_rows.loc[model_rows["model_name"] == "M2_entropy", "warnings"].iloc[0]),
        },
        "selected_lag_z_stats_series": merged[f"entropy_lag_{selected_lag_ms}ms_z"],
        "selected_lag_z_series_name": f"selected-lag z-scored renyi entropy",
        "selected_lag_z_column": f"entropy_lag_{selected_lag_ms}ms_z",
    }


def _fit_alpha_at_fixed_lag(
    *,
    alpha: float,
    alpha_label: str,
    neural_alpha: pd.DataFrame,
    riskset: pd.DataFrame,
    cfg_template: base.NeuralHazardFppSppConfig,
    fixed_lag_ms: int,
) -> dict[str, Any]:
    cfg = _mk_base_cfg(
        {
            "paths": {
                "fpp_risk_set": str(cfg_template.fpp_risk_set_path),
                "spp_risk_set": str(cfg_template.spp_risk_set_path),
                "neural_features": str(cfg_template.neural_features_path),
            },
            "analysis": {
                "bin_width_s": cfg_template.bin_width_s,
                "lag_grid_ms": [int(fixed_lag_ms)],
                "avoid_zero_lag": cfg_template.avoid_zero_lag,
                "timing_zscore_scope": cfg_template.timing_zscore_scope,
                "feature_zscore_scope": cfg_template.neural_zscore_scope,
            },
            "pca": {
                "input_columns": list(cfg_template.pca_input_columns),
                "minimum_pc1_explained_variance": cfg_template.pca_pc1_warning_threshold,
                "n_components": cfg_template.pca_n_components,
            },
            "circular_shift": {
                "minimum_shift_s": cfg_template.minimum_circular_shift_duration_s,
                "n_permutations": cfg_template.n_circular_shift_permutations,
                "random_seed": cfg_template.circular_shift_seed,
            },
        },
        project_root=Path(".").resolve(),
        derivatives_root=None,
        out_dir=cfg_template.out_dir,
        lag_grid_ms=(int(fixed_lag_ms),),
    )
    merged = base._merge_lagged_neural_features(riskset, neural_alpha, cfg)
    merged = base._zscore_all_features(merged, cfg)
    merged, _, _ = base._add_instability_pc1_by_lag(merged, cfg)
    pooled = base._fit_pooled_models(merged, int(fixed_lag_ms))
    m2 = pooled["models"]["M2_entropy"]
    entropy_term = f"entropy_lag_{fixed_lag_ms}ms_z"
    interaction_term = f"C(anchor_type)[T.SPP]:{entropy_term}"
    m2_row = pooled["model_comparison"].loc[
        (pooled["model_comparison"]["row_type"].astype(str) == "model")
        & (pooled["model_comparison"]["model_name"].astype(str) == "M2_entropy")
    ].iloc[0]
    return {
        "alpha": float(alpha),
        "alpha_label": alpha_label,
        "m2_vs_m1_delta_loglik": float(pooled["pairwise"]["M2_entropy_vs_M1_instability"]["delta_loglik"]),
        "m2_vs_m1_lrt_p_value": float(pooled["pairwise"]["M2_entropy_vs_M1_instability"]["p_value"]),
        "m2_aic": float(m2_row["aic"]),
        "m2_bic": float(m2_row["bic"]),
        "entropy_beta_fpp": float(base._extract_term_stats(m2, entropy_term)["estimate"]),
        "entropy_interaction_beta_spp_minus_fpp": float(base._extract_term_stats(m2, interaction_term)["estimate"]),
        "n_rows": int(m2_row["n_rows"]),
        "n_events": int(m2_row["n_events"]),
        "converged": bool(m2_row["converged"]),
        "warnings": str(m2_row["warnings"]),
    }


def _permutation_p_value_interaction(observed: float, null_values: np.ndarray, *, mode: str) -> float:
    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0:
        return float("nan")
    if mode == "two_sided":
        return float((1 + np.count_nonzero(np.abs(finite) >= abs(observed))) / (1 + finite.size))
    if mode == "greater":
        return float((1 + np.count_nonzero(finite >= observed)) / (1 + finite.size))
    if mode == "less":
        return float((1 + np.count_nonzero(finite <= observed)) / (1 + finite.size))
    raise ValueError("Unsupported p_value_mode_interaction_beta: " + str(mode))


def _run_renyi_circular_shift_null(
    *,
    alpha: float,
    alpha_label: str,
    selected_lag_ms: int,
    observed_fit: dict[str, Any],
    riskset_enriched: pd.DataFrame,
    neural_alpha: pd.DataFrame,
    cfg: base.NeuralHazardFppSppConfig,
    p_value_mode_interaction_beta: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(cfg.circular_shift_seed)
    rows: list[dict[str, Any]] = []
    observed_m2 = observed_fit["pooled"]["models"]["M2_entropy"]
    entropy_term = f"entropy_lag_{selected_lag_ms}ms_z"
    interaction_term = f"C(anchor_type)[T.SPP]:{entropy_term}"
    for permutation_id in range(int(cfg.n_circular_shift_permutations)):
        shifted_neural = base._circular_shift_entropy(neural_alpha, config=cfg, rng=rng)
        shifted = base._merge_selected_entropy_shift(riskset_enriched, shifted_neural, int(selected_lag_ms), cfg)
        shifted = base._add_groupwise_zscore(
            shifted,
            source_column=f"entropy_lag_{selected_lag_ms}ms",
            output_column=entropy_term,
            scope=cfg.neural_zscore_scope,
        )
        pair = base._fit_shifted_entropy_models(shifted, int(selected_lag_ms))
        m1 = pair["models"]["M1_instability"]
        m2 = pair["models"]["M2_entropy_shifted"]
        rows.append(
            {
                "permutation_id": int(permutation_id),
                "alpha": float(alpha),
                "alpha_label": alpha_label,
                "selected_lag_ms": int(selected_lag_ms),
                "m1_loglik": m1.log_likelihood,
                "m2_loglik": m2.log_likelihood,
                "delta_loglik": pair["comparison"]["delta_loglik"],
                "entropy_beta_fpp": float(base._extract_term_stats(m2, entropy_term)["estimate"]),
                "entropy_interaction_beta_spp_minus_fpp": float(base._extract_term_stats(m2, interaction_term)["estimate"]),
                "converged": bool(m1.converged and m2.converged),
                "warnings": " | ".join(part for part in [_model_warning_text(m1), _model_warning_text(m2)] if part),
            }
        )
    null_table = pd.DataFrame(rows)
    null_delta = pd.to_numeric(null_table["delta_loglik"], errors="coerce").to_numpy(dtype=float)
    null_interaction = pd.to_numeric(
        null_table["entropy_interaction_beta_spp_minus_fpp"], errors="coerce"
    ).to_numpy(dtype=float)
    observed_delta = float(observed_fit["pooled"]["pairwise"]["M2_entropy_vs_M1_instability"]["delta_loglik"])
    observed_interaction = float(base._extract_term_stats(observed_m2, interaction_term)["estimate"])
    finite_delta = null_delta[np.isfinite(null_delta)]
    finite_interaction = null_interaction[np.isfinite(null_interaction)]
    summary = {
        "alpha": float(alpha),
        "alpha_label": alpha_label,
        "selected_lag_ms": int(selected_lag_ms),
        "n_permutations_requested": int(cfg.n_circular_shift_permutations),
        "n_permutations_successful": int(finite_delta.size),
        "observed_delta_loglik": observed_delta,
        "null_mean_delta_loglik": float(np.nanmean(null_delta)) if finite_delta.size else np.nan,
        "null_sd_delta_loglik": float(np.nanstd(null_delta, ddof=0)) if finite_delta.size else np.nan,
        "permutation_p_value_delta_loglik": float((1 + np.count_nonzero(finite_delta >= observed_delta)) / (1 + finite_delta.size))
        if finite_delta.size
        else np.nan,
        "observed_entropy_interaction_beta_spp_minus_fpp": observed_interaction,
        "null_mean_entropy_interaction_beta_spp_minus_fpp": float(np.nanmean(null_interaction))
        if finite_interaction.size
        else np.nan,
        "null_sd_entropy_interaction_beta_spp_minus_fpp": float(np.nanstd(null_interaction, ddof=0))
        if finite_interaction.size
        else np.nan,
        "permutation_p_value_interaction_beta": _permutation_p_value_interaction(
            observed_interaction,
            null_interaction,
            mode=p_value_mode_interaction_beta,
        ),
        "p_value_mode_interaction_beta": p_value_mode_interaction_beta,
    }
    return null_table, summary


def _build_correlation_outputs(
    *,
    raw_series_map: dict[str, pd.Series],
    selected_lag_series_map: dict[str, pd.Series],
    alpha_lookup: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frame = pd.DataFrame(raw_series_map).corr()
    lag_frame = pd.DataFrame(selected_lag_series_map).corr()
    long_rows: list[dict[str, Any]] = []
    for correlation_type, frame in (
        ("raw_same_time", raw_frame),
        ("selected_lag_z", lag_frame),
    ):
        for alpha_i_label in frame.index:
            for alpha_j_label in frame.columns:
                long_rows.append(
                    {
                        "correlation_type": correlation_type,
                        "alpha_i": float(alpha_lookup[alpha_i_label]),
                        "alpha_i_label": alpha_i_label,
                        "alpha_j": float(alpha_lookup[alpha_j_label]),
                        "alpha_j_label": alpha_j_label,
                        "correlation": float(frame.loc[alpha_i_label, alpha_j_label]),
                    }
                )
    return raw_frame, lag_frame, pd.DataFrame(long_rows)


def _plot_distribution_boxplot(selected_lag_series_map: dict[str, pd.Series], out_path: Path) -> None:
    plt.figure(figsize=(10, 4.8))
    labels = list(selected_lag_series_map.keys())
    data = [pd.to_numeric(selected_lag_series_map[label], errors="coerce").dropna().to_numpy(dtype=float) for label in labels]
    if not any(len(values) for values in data):
        plt.text(0.5, 0.5, "No finite selected-lag entropy values available.", ha="center", va="center")
        plt.gca().set_axis_off()
    else:
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Selected-lag z-scored entropy")
        plt.title("Entropy distributions by alpha")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_correlation_heatmap(frame: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.4, 5.4))
    if frame.empty:
        plt.text(0.5, 0.5, "No correlation matrix available.", ha="center", va="center")
        plt.gca().set_axis_off()
    else:
        matrix = frame.to_numpy(dtype=float)
        image = plt.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        plt.xticks(np.arange(len(frame.columns)), frame.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(frame.index)), frame.index)
        plt.title("Cross-alpha entropy correlation")
        plt.colorbar(image, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_alpha_curve(summary: pd.DataFrame, out_path: Path, *, title: str, axis_scale: str, best_alpha: float) -> None:
    plt.figure(figsize=(6.8, 4.4))
    plt.plot(summary["alpha"], summary["m2_vs_m1_delta_loglik"], marker="o")
    plt.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    plt.axvline(float(best_alpha), color="tab:red", linestyle=":", linewidth=1.2)
    if axis_scale == "log":
        plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("M2 vs M1 delta log-likelihood")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_same_lag_curves(table: pd.DataFrame, out_path: Path, *, axis_scale: str) -> None:
    plt.figure(figsize=(7.4, 4.8))
    if table.empty:
        plt.text(0.5, 0.5, "No same-lag comparisons available.", ha="center", va="center")
        plt.gca().set_axis_off()
    else:
        table = table.copy()
        table["series_label"] = table["fixed_lag_source"].astype(str) + ":" + table["fixed_lag_ms"].astype(int).astype(str) + "ms"
        for series_label, frame in table.groupby("series_label", sort=False):
            frame = frame.sort_values("alpha", kind="mergesort")
            plt.plot(frame["alpha"], frame["m2_vs_m1_delta_loglik"], marker="o", label=series_label)
        if axis_scale == "log":
            plt.xscale("log")
        plt.xlabel("alpha")
        plt.ylabel("M2 vs M1 delta log-likelihood")
        plt.title("Same-lag alpha comparison")
        plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _safe_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def run_neural_hazard_fpp_spp_renyi_alpha_pipeline(
    config: NeuralHazardFppSppRenyiAlphaConfig,
) -> NeuralHazardFppSppRenyiAlphaResult:
    raw = _ensure_default_config(config.raw)
    paths = raw["paths"]
    renyi = raw["renyi"]
    diagnostics_cfg = dict(raw.get("diagnostics") or {})
    circular_shift_cfg = dict(raw.get("circular_shift") or {})
    motor_cfg = dict(raw.get("motor_proximal_sensitivity") or {})
    plots_cfg = dict(raw.get("plots") or {})
    config_path = config.config_path.resolve()
    project_root = config_path.parent.parent
    derivatives_root = base._load_derivatives_root(project_root / "config" / "paths.yaml")
    out_dir = _resolve_out(str(paths["out_dir"]), project_root=project_root, derivatives_root=derivatives_root)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    alpha_grid = [float(alpha) for alpha in renyi["alpha_grid"]]
    alpha_lookup = {_safe_alpha_label(alpha): float(alpha) for alpha in alpha_grid}
    base_cfg = _mk_base_cfg(
        raw,
        project_root=project_root,
        derivatives_root=derivatives_root,
        out_dir=out_dir,
    )
    riskset = base._load_and_combine_risk_sets(base_cfg)
    neural = _load_neural_features_with_state_probabilities(
        neural_features_path=base_cfg.neural_features_path,
        raw_config=raw,
        project_root=project_root,
        derivatives_root=derivatives_root,
        scratch_dir=out_dir / "tables",
    )
    prob_cols = discover_state_probability_columns(neural)
    if not prob_cols:
        raise ValueError("Missing HMM state probability columns required for Renyi entropy.")

    entropy_by_alpha: dict[float, np.ndarray] = {}
    neural_alpha_tables: dict[float, pd.DataFrame] = {}
    descriptives_rows: list[dict[str, Any]] = []
    shannon_validation_rows: list[dict[str, Any]] = []
    lag_rows: list[pd.DataFrame] = []
    selected_lags_rows: list[dict[str, Any]] = []
    coeff_rows: list[pd.DataFrame] = []
    model_rows: list[pd.DataFrame] = []
    pairwise_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    selected_lag_series_map: dict[str, pd.Series] = {}
    raw_series_map: dict[str, pd.Series] = {}
    search_results: dict[float, dict[str, Any]] = {}

    for alpha in alpha_grid:
        alpha_label = _safe_alpha_label(alpha)
        entropy = _compute_renyi_entropy_allow_missing_rows(
            neural[prob_cols].to_numpy(dtype=float),
            alpha=float(alpha),
            epsilon=float(renyi.get("probability_epsilon", 1e-12)),
            alpha_one_tolerance=float(renyi.get("alpha_one_tolerance", 1e-6)),
        )
        entropy_by_alpha[float(alpha)] = entropy
        neural_alpha = _make_neural_alpha_table(
            neural,
            entropy=entropy,
            pca_columns=tuple(raw["pca"]["input_columns"]),
        )
        neural_alpha_tables[float(alpha)] = neural_alpha
        raw_series_map[alpha_label] = pd.Series(entropy)
        descriptives_rows.append(
            _describe_series(alpha=float(alpha), alpha_label=alpha_label, variable="raw renyi entropy", values=pd.Series(entropy))
        )
        if abs(float(alpha) - 1.0) < float(renyi.get("alpha_one_tolerance", 1e-6)):
            shannon_validation_rows.append(
                _shannon_validation_row(
                    alpha=float(alpha),
                    entropy=entropy,
                    neural=neural,
                    tolerance=float(renyi.get("alpha_one_tolerance", 1e-6)),
                )
            )
        fit = _fit_alpha_on_lag_grid(
            alpha=float(alpha),
            alpha_label=alpha_label,
            neural_alpha=neural_alpha,
            riskset=riskset,
            cfg=base_cfg,
        )
        search_results[float(alpha)] = fit
        lag_rows.append(fit["lag_selection"])
        selected_lags_rows.append(fit["selected_lag_row"])
        coeff_rows.extend(fit["coefficients"].values())
        model_rows.append(fit["model_rows"])
        pairwise_rows.append(fit["pairwise_rows"])
        summary_rows.append(fit["summary_row"])
        selected_lag_series_map[alpha_label] = pd.to_numeric(fit["selected_lag_z_stats_series"], errors="coerce")
        descriptives_rows.append(
            _describe_series(
                alpha=float(alpha),
                alpha_label=alpha_label,
                variable=fit["selected_lag_z_series_name"],
                values=fit["selected_lag_z_stats_series"],
            )
        )

    shannon_validation = pd.DataFrame(shannon_validation_rows)
    if shannon_validation.empty:
        shannon_validation = pd.DataFrame(
            [
                {
                    "alpha": 1.0,
                    "existing_entropy_present": False,
                    "n_compared": 0,
                    "max_abs_difference_alpha1_vs_existing_entropy": np.nan,
                    "mean_abs_difference_alpha1_vs_existing_entropy": np.nan,
                    "median_abs_difference_alpha1_vs_existing_entropy": np.nan,
                    "correlation_alpha1_vs_existing_entropy": np.nan,
                }
            ]
        )
    shannon_validation.to_csv(out_dir / "tables" / "renyi_shannon_validation.csv", index=False)

    lag_table = pd.concat(lag_rows, ignore_index=True)
    lag_table.to_csv(out_dir / "tables" / "renyi_alpha_lag_selection.csv", index=False)
    pd.DataFrame(selected_lags_rows).to_csv(out_dir / "tables" / "renyi_alpha_selected_lags.csv", index=False)
    pd.concat(coeff_rows, ignore_index=True).to_csv(out_dir / "tables" / "renyi_alpha_coefficients.csv", index=False)
    pd.concat(model_rows, ignore_index=True).to_csv(out_dir / "tables" / "renyi_alpha_model_comparison.csv", index=False)
    pd.concat(pairwise_rows, ignore_index=True).to_csv(out_dir / "tables" / "renyi_alpha_pairwise_model_comparison.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary["rank_by_delta_loglik"] = summary["m2_vs_m1_delta_loglik"].rank(ascending=False, method="min").astype(int)
    summary["rank_by_aic"] = summary["m2_aic"].rank(ascending=True, method="min").astype(int)
    summary["rank_by_bic"] = summary["m2_bic"].rank(ascending=True, method="min").astype(int)
    summary = summary.sort_values("m2_vs_m1_delta_loglik", ascending=False, kind="mergesort").reset_index(drop=True)
    summary.to_csv(out_dir / "tables" / "renyi_alpha_search_summary.csv", index=False)

    best = summary.iloc[0]
    best_alpha = float(best["alpha"])
    best_alpha_label = str(best["alpha_label"])
    shannon_row = summary.loc[np.isclose(summary["alpha"], 1.0)]
    shannon_delta = float(shannon_row["m2_vs_m1_delta_loglik"].iloc[0]) if not shannon_row.empty else np.nan
    shannon_selected_lag_ms = int(shannon_row["selected_lag_ms"].iloc[0]) if not shannon_row.empty else None

    pd.DataFrame(
        [
            {
                "best_alpha": best_alpha,
                "best_alpha_label": best_alpha_label,
                "selected_lag_ms": int(best["selected_lag_ms"]),
                "selection_metric": "m2_vs_m1_delta_loglik",
                "selection_metric_value": float(best["m2_vs_m1_delta_loglik"]),
                "entropy_beta_fpp": float(best["entropy_beta_fpp"]),
                "entropy_interaction_beta_spp_minus_fpp": float(best["entropy_interaction_beta_spp_minus_fpp"]),
                "shannon_delta_loglik": shannon_delta,
                "best_minus_shannon_delta_loglik": float(best["m2_vs_m1_delta_loglik"] - shannon_delta)
                if np.isfinite(shannon_delta)
                else np.nan,
            }
        ]
    ).to_csv(out_dir / "tables" / "renyi_best_alpha.csv", index=False)

    descriptives = pd.DataFrame(descriptives_rows)
    descriptives.to_csv(out_dir / "tables" / "renyi_entropy_descriptives.csv", index=False)

    raw_corr, selected_lag_corr, corr_long = _build_correlation_outputs(
        raw_series_map=raw_series_map,
        selected_lag_series_map=selected_lag_series_map,
        alpha_lookup=alpha_lookup,
    )
    raw_corr.to_csv(out_dir / "tables" / "renyi_alpha_entropy_correlation_raw.csv", index=True)
    selected_lag_corr.to_csv(out_dir / "tables" / "renyi_alpha_entropy_correlation_selected_lag_z.csv", index=True)
    corr_long.to_csv(out_dir / "tables" / "renyi_alpha_entropy_correlation_long.csv", index=False)

    edge_warning_enabled = bool(renyi.get("edge_warning_enabled", True))
    at_lower_edge = bool(np.isclose(best_alpha, min(alpha_grid)))
    at_upper_edge = bool(np.isclose(best_alpha, max(alpha_grid)))
    edge_warning_message = ""
    if edge_warning_enabled and (at_lower_edge or at_upper_edge):
        edge_warning_message = (
            f"Best alpha {best_alpha:g} lies on the explored grid edge; extend the grid before treating the optimum as stable."
        )
    pd.DataFrame(
        [
            {
                "best_alpha": best_alpha,
                "min_alpha": float(min(alpha_grid)),
                "max_alpha": float(max(alpha_grid)),
                "at_lower_edge": at_lower_edge,
                "at_upper_edge": at_upper_edge,
                "warning": edge_warning_message,
            }
        ]
    ).to_csv(out_dir / "tables" / "renyi_alpha_edge_warnings.csv", index=False)

    same_lag_rows: list[dict[str, Any]] = []
    fixed_lag_specs: list[tuple[str, int]] = [("best_alpha_selected_lag", int(best["selected_lag_ms"]))]
    if shannon_selected_lag_ms is not None:
        fixed_lag_specs.append(("shannon_selected_lag", int(shannon_selected_lag_ms)))
    for lag_ms in renyi.get("fixed_lag_sensitivity_ms", []):
        fixed_lag_specs.append(("config_fixed_lag", int(lag_ms)))
    seen_fixed = set()
    for fixed_lag_source, fixed_lag_ms in fixed_lag_specs:
        key = (fixed_lag_source, int(fixed_lag_ms))
        if key in seen_fixed:
            continue
        seen_fixed.add(key)
        for alpha in alpha_grid:
            alpha_label = _safe_alpha_label(alpha)
            row = _fit_alpha_at_fixed_lag(
                alpha=float(alpha),
                alpha_label=alpha_label,
                neural_alpha=neural_alpha_tables[float(alpha)],
                riskset=riskset,
                cfg_template=base_cfg,
                fixed_lag_ms=int(fixed_lag_ms),
            )
            row["fixed_lag_source"] = fixed_lag_source
            row["fixed_lag_ms"] = int(fixed_lag_ms)
            same_lag_rows.append(row)
    same_lag_table = pd.DataFrame(same_lag_rows)
    same_lag_table = same_lag_table[
        [
            "fixed_lag_source",
            "fixed_lag_ms",
            "alpha",
            "alpha_label",
            "m2_vs_m1_delta_loglik",
            "m2_vs_m1_lrt_p_value",
            "m2_aic",
            "m2_bic",
            "entropy_beta_fpp",
            "entropy_interaction_beta_spp_minus_fpp",
            "n_rows",
            "n_events",
            "converged",
            "warnings",
        ]
    ]
    same_lag_table.to_csv(out_dir / "tables" / "renyi_same_lag_model_comparison.csv", index=False)

    motor_minimum_lag_ms = int(motor_cfg.get("minimum_lag_ms", 150))
    restricted_grid = tuple(
        int(lag_ms)
        for lag_ms in motor_cfg.get(
            "lag_grid_ms",
            [lag for lag in base_cfg.lag_grid_ms if int(lag) >= motor_minimum_lag_ms],
        )
    )
    motor_alpha_grid = [float(alpha) for alpha in motor_cfg.get("alpha_grid", alpha_grid)]
    motor_lag_rows: list[pd.DataFrame] = []
    motor_selected_rows: list[dict[str, Any]] = []
    motor_summary_rows: list[dict[str, Any]] = []
    if bool(motor_cfg.get("enabled", True)) and restricted_grid:
        motor_base_cfg = _mk_base_cfg(
            raw,
            project_root=project_root,
            derivatives_root=derivatives_root,
            out_dir=out_dir,
            lag_grid_ms=restricted_grid,
        )
        for alpha in motor_alpha_grid:
            alpha_label = _safe_alpha_label(alpha)
            fit = _fit_alpha_on_lag_grid(
                alpha=float(alpha),
                alpha_label=alpha_label,
                neural_alpha=neural_alpha_tables[float(alpha)],
                riskset=riskset,
                cfg=motor_base_cfg,
            )
            motor_lag_rows.append(fit["lag_selection"])
            motor_selected_rows.append(fit["selected_lag_row"])
            motor_summary_rows.append(fit["summary_row"])
    motor_lag_table = pd.concat(motor_lag_rows, ignore_index=True) if motor_lag_rows else pd.DataFrame()
    motor_lag_table.to_csv(out_dir / "tables" / "renyi_motor_exclusion_lag_selection.csv", index=False)
    motor_selected_table = pd.DataFrame(motor_selected_rows)
    motor_selected_table.to_csv(out_dir / "tables" / "renyi_motor_exclusion_selected_lags.csv", index=False)
    motor_summary = pd.DataFrame(motor_summary_rows)
    if not motor_summary.empty:
        motor_summary["rank_by_delta_loglik"] = motor_summary["m2_vs_m1_delta_loglik"].rank(ascending=False, method="min").astype(int)
        motor_summary["rank_by_aic"] = motor_summary["m2_aic"].rank(ascending=True, method="min").astype(int)
        motor_summary["rank_by_bic"] = motor_summary["m2_bic"].rank(ascending=True, method="min").astype(int)
        motor_summary = motor_summary.sort_values("m2_vs_m1_delta_loglik", ascending=False, kind="mergesort").reset_index(drop=True)
    motor_summary.to_csv(out_dir / "tables" / "renyi_motor_exclusion_alpha_search_summary.csv", index=False)
    motor_best_row = motor_summary.iloc[0] if not motor_summary.empty else None
    motor_shannon_row = motor_summary.loc[np.isclose(motor_summary["alpha"], 1.0)] if not motor_summary.empty else pd.DataFrame()
    motor_best_table = pd.DataFrame(
        [
            {
                "best_alpha": float(motor_best_row["alpha"]) if motor_best_row is not None else np.nan,
                "best_alpha_label": str(motor_best_row["alpha_label"]) if motor_best_row is not None else "",
                "selected_lag_ms": int(motor_best_row["selected_lag_ms"]) if motor_best_row is not None else np.nan,
                "selection_metric": "m2_vs_m1_delta_loglik",
                "selection_metric_value": float(motor_best_row["m2_vs_m1_delta_loglik"]) if motor_best_row is not None else np.nan,
                "entropy_beta_fpp": float(motor_best_row["entropy_beta_fpp"]) if motor_best_row is not None else np.nan,
                "entropy_interaction_beta_spp_minus_fpp": float(motor_best_row["entropy_interaction_beta_spp_minus_fpp"])
                if motor_best_row is not None
                else np.nan,
                "shannon_delta_loglik": float(motor_shannon_row["m2_vs_m1_delta_loglik"].iloc[0]) if not motor_shannon_row.empty else np.nan,
                "best_minus_shannon_delta_loglik": (
                    float(motor_best_row["m2_vs_m1_delta_loglik"] - motor_shannon_row["m2_vs_m1_delta_loglik"].iloc[0])
                    if (motor_best_row is not None and not motor_shannon_row.empty)
                    else np.nan
                ),
            }
        ]
    )
    motor_best_table.to_csv(out_dir / "tables" / "renyi_motor_exclusion_best_alpha.csv", index=False)

    circular_null_rows: list[pd.DataFrame] = []
    circular_summary_rows: list[dict[str, Any]] = []
    circular_summaries_by_alpha: dict[float, dict[str, Any]] = {}
    if bool(circular_shift_cfg.get("enabled", True)):
        targets: list[tuple[float, str]] = []
        if bool(circular_shift_cfg.get("run_for_best_alpha_only", True)):
            targets.append((best_alpha, best_alpha_label))
        if bool(circular_shift_cfg.get("run_for_shannon", True)) and 1.0 in search_results:
            targets.append((1.0, _safe_alpha_label(1.0)))
        seen_targets: set[str] = set()
        for alpha, alpha_label in targets:
            if alpha_label in seen_targets:
                continue
            seen_targets.add(alpha_label)
            fit = search_results[float(alpha)]
            null_table, null_summary = _run_renyi_circular_shift_null(
                alpha=float(alpha),
                alpha_label=alpha_label,
                selected_lag_ms=int(fit["selected_lag_ms"]),
                observed_fit=fit,
                riskset_enriched=fit["merged"],
                neural_alpha=neural_alpha_tables[float(alpha)],
                cfg=base_cfg,
                p_value_mode_interaction_beta=str(circular_shift_cfg.get("p_value_mode_interaction_beta", "two_sided")),
            )
            circular_null_rows.append(null_table)
            circular_summary_rows.append(null_summary)
            circular_summaries_by_alpha[float(alpha)] = null_summary
    circular_null_table = pd.concat(circular_null_rows, ignore_index=True) if circular_null_rows else pd.DataFrame(
        columns=[
            "permutation_id",
            "alpha",
            "alpha_label",
            "selected_lag_ms",
            "m1_loglik",
            "m2_loglik",
            "delta_loglik",
            "entropy_beta_fpp",
            "entropy_interaction_beta_spp_minus_fpp",
            "converged",
            "warnings",
        ]
    )
    circular_null_table.to_csv(out_dir / "tables" / "renyi_circular_shift_null.csv", index=False)
    circular_summary_table = pd.DataFrame(circular_summary_rows)
    circular_summary_table.to_csv(out_dir / "tables" / "renyi_circular_shift_summary.csv", index=False)

    best_fit = search_results[best_alpha]
    base._plot_predicted_hazard(
        best_fit["pooled"]["models"]["M2_entropy"],
        int(best_fit["selected_lag_ms"]),
        out_dir / "figures" / "renyi_best_alpha_predicted_hazard_by_anchor_type.png",
    )
    _plot_alpha_curve(
        summary,
        out_dir / "figures" / "renyi_alpha_search_delta_loglik.png",
        title="Rényi alpha search",
        axis_scale=str(plots_cfg.get("alpha_axis_scale", "linear")),
        best_alpha=best_alpha,
    )
    _plot_alpha_curve(
        summary,
        out_dir / "figures" / "renyi_alpha_search_delta_loglik_extended.png",
        title="Extended Rényi alpha search",
        axis_scale=str(plots_cfg.get("alpha_axis_scale", "linear")),
        best_alpha=best_alpha,
    )
    _plot_distribution_boxplot(
        selected_lag_series_map,
        out_dir / "figures" / "renyi_entropy_distribution_by_alpha.png",
    )
    _plot_correlation_heatmap(
        raw_corr,
        out_dir / "figures" / "renyi_alpha_entropy_correlation_heatmap.png",
    )
    _plot_same_lag_curves(
        same_lag_table,
        out_dir / "figures" / "renyi_same_lag_delta_loglik.png",
        axis_scale=str(plots_cfg.get("alpha_axis_scale", "linear")),
    )
    if not motor_summary.empty:
        _plot_alpha_curve(
            motor_summary,
            out_dir / "figures" / "renyi_motor_exclusion_alpha_search_delta_loglik.png",
            title="Motor-proximal exclusion alpha search",
            axis_scale=str(plots_cfg.get("alpha_axis_scale", "linear")),
            best_alpha=float(motor_summary.iloc[0]["alpha"]),
        )
    else:
        plt.figure(figsize=(6.8, 4.4))
        plt.text(0.5, 0.5, "Motor exclusion was disabled or had no eligible lags.", ha="center", va="center")
        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "renyi_motor_exclusion_alpha_search_delta_loglik.png", dpi=200)
        plt.close()
    best_null_summary = circular_summaries_by_alpha.get(best_alpha)
    if best_null_summary is not None:
        base._plot_null_histogram(
            circular_null_table.loc[circular_null_table["alpha_label"].astype(str) == best_alpha_label].copy(),
            float(best_null_summary["observed_delta_loglik"]),
            out_dir / "figures" / "renyi_best_alpha_circular_shift_delta_loglik.png",
        )
    else:
        plt.figure(figsize=(6.8, 4.4))
        plt.text(0.5, 0.5, "Circular-shift null disabled.", ha="center", va="center")
        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "renyi_best_alpha_circular_shift_delta_loglik.png", dpi=200)
        plt.close()
    shannon_null_summary = circular_summaries_by_alpha.get(1.0)
    if shannon_null_summary is not None:
        base._plot_null_histogram(
            circular_null_table.loc[circular_null_table["alpha_label"].astype(str) == _safe_alpha_label(1.0)].copy(),
            float(shannon_null_summary["observed_delta_loglik"]),
            out_dir / "figures" / "renyi_shannon_circular_shift_delta_loglik.png",
        )
    else:
        plt.figure(figsize=(6.8, 4.4))
        plt.text(0.5, 0.5, "Shannon circular-shift null unavailable.", ha="center", va="center")
        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "renyi_shannon_circular_shift_delta_loglik.png", dpi=200)
        plt.close()

    same_lag_best_alpha_selected_best = same_lag_table.loc[
        (same_lag_table["fixed_lag_source"].astype(str) == "best_alpha_selected_lag")
        & np.isclose(same_lag_table["alpha"], best_alpha),
        "m2_vs_m1_delta_loglik",
    ]
    same_lag_best_alpha_selected_shannon = same_lag_table.loc[
        (same_lag_table["fixed_lag_source"].astype(str) == "best_alpha_selected_lag")
        & np.isclose(same_lag_table["alpha"], 1.0),
        "m2_vs_m1_delta_loglik",
    ]
    same_lag_shannon_selected_best = same_lag_table.loc[
        (same_lag_table["fixed_lag_source"].astype(str) == "shannon_selected_lag")
        & np.isclose(same_lag_table["alpha"], best_alpha),
        "m2_vs_m1_delta_loglik",
    ]
    same_lag_shannon_selected_shannon = same_lag_table.loc[
        (same_lag_table["fixed_lag_source"].astype(str) == "shannon_selected_lag")
        & np.isclose(same_lag_table["alpha"], 1.0),
        "m2_vs_m1_delta_loglik",
    ]

    caveat = "Renyi alpha search is exploratory unless alpha selection is validated on held-out data or by nested resampling."
    caveats = [caveat]
    if int(best["selected_lag_ms"]) < 150:
        caveats.append(
            "The selected lag is motor-proximal; interpret as immediate pre-action dynamics unless motor-proximal exclusion sensitivity supports the same pattern."
        )

    summary_json = {
        "alpha_grid": alpha_grid,
        "best_alpha": best_alpha,
        "best_alpha_label": best_alpha_label,
        "best_alpha_selected_lag_ms": int(best["selected_lag_ms"]),
        "best_alpha_delta_loglik": float(best["m2_vs_m1_delta_loglik"]),
        "shannon_selected_lag_ms": shannon_selected_lag_ms,
        "shannon_delta_loglik": _safe_float_or_none(shannon_delta),
        "best_minus_shannon_delta_loglik": _safe_float_or_none(
            float(best["m2_vs_m1_delta_loglik"] - shannon_delta) if np.isfinite(shannon_delta) else np.nan
        ),
        "best_alpha_entropy_beta_fpp": float(best["entropy_beta_fpp"]),
        "best_alpha_entropy_interaction_beta_spp_minus_fpp": float(best["entropy_interaction_beta_spp_minus_fpp"]),
        "circular_shift_enabled": bool(circular_shift_cfg.get("enabled", True)),
        "best_alpha_circular_shift_p_value_delta_loglik": _safe_float_or_none(
            None if best_null_summary is None else best_null_summary["permutation_p_value_delta_loglik"]
        ),
        "best_alpha_circular_shift_p_value_interaction_beta": _safe_float_or_none(
            None if best_null_summary is None else best_null_summary["permutation_p_value_interaction_beta"]
        ),
        "shannon_circular_shift_p_value_delta_loglik": _safe_float_or_none(
            None if shannon_null_summary is None else shannon_null_summary["permutation_p_value_delta_loglik"]
        ),
        "shannon_circular_shift_p_value_interaction_beta": _safe_float_or_none(
            None if shannon_null_summary is None else shannon_null_summary["permutation_p_value_interaction_beta"]
        ),
        "best_alpha_circular_shift_n_successful": None if best_null_summary is None else int(best_null_summary["n_permutations_successful"]),
        "shannon_circular_shift_n_successful": None if shannon_null_summary is None else int(shannon_null_summary["n_permutations_successful"]),
        "alpha_edge_warning": bool(edge_warning_message),
        "alpha_edge_warning_message": edge_warning_message or None,
        "entropy_correlation_alpha_0p25_vs_alpha_1p0": _safe_float_or_none(
            raw_corr.loc[_safe_alpha_label(0.25), _safe_alpha_label(1.0)]
            if _safe_alpha_label(0.25) in raw_corr.index and _safe_alpha_label(1.0) in raw_corr.columns
            else np.nan
        ),
        "entropy_correlation_best_alpha_vs_shannon": _safe_float_or_none(
            raw_corr.loc[best_alpha_label, _safe_alpha_label(1.0)]
            if best_alpha_label in raw_corr.index and _safe_alpha_label(1.0) in raw_corr.columns
            else np.nan
        ),
        "same_lag_best_alpha_selected_lag_best_alpha": _safe_float_or_none(
            same_lag_best_alpha_selected_best.iloc[0] if not same_lag_best_alpha_selected_best.empty else np.nan
        ),
        "same_lag_best_alpha_selected_lag_shannon": _safe_float_or_none(
            same_lag_best_alpha_selected_shannon.iloc[0] if not same_lag_best_alpha_selected_shannon.empty else np.nan
        ),
        "same_lag_shannon_selected_lag_best_alpha": _safe_float_or_none(
            same_lag_shannon_selected_best.iloc[0] if not same_lag_shannon_selected_best.empty else np.nan
        ),
        "same_lag_shannon_selected_lag_shannon": _safe_float_or_none(
            same_lag_shannon_selected_shannon.iloc[0] if not same_lag_shannon_selected_shannon.empty else np.nan
        ),
        "motor_exclusion_enabled": bool(motor_cfg.get("enabled", True)),
        "motor_exclusion_minimum_lag_ms": motor_minimum_lag_ms,
        "motor_exclusion_best_alpha": None if motor_best_row is None else float(motor_best_row["alpha"]),
        "motor_exclusion_best_alpha_label": None if motor_best_row is None else str(motor_best_row["alpha_label"]),
        "motor_exclusion_best_alpha_selected_lag_ms": None if motor_best_row is None else int(motor_best_row["selected_lag_ms"]),
        "motor_exclusion_best_alpha_delta_loglik": None if motor_best_row is None else float(motor_best_row["m2_vs_m1_delta_loglik"]),
        "motor_exclusion_shannon_delta_loglik": None
        if motor_shannon_row.empty
        else float(motor_shannon_row["m2_vs_m1_delta_loglik"].iloc[0]),
        "motor_exclusion_best_minus_shannon_delta_loglik": None
        if (motor_best_row is None or motor_shannon_row.empty)
        else float(motor_best_row["m2_vs_m1_delta_loglik"] - motor_shannon_row["m2_vs_m1_delta_loglik"].iloc[0]),
        "motor_exclusion_entropy_beta_fpp": None if motor_best_row is None else float(motor_best_row["entropy_beta_fpp"]),
        "motor_exclusion_entropy_interaction_beta_spp_minus_fpp": None
        if motor_best_row is None
        else float(motor_best_row["entropy_interaction_beta_spp_minus_fpp"]),
        "caveat": caveat,
        "caveats": caveats,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    return NeuralHazardFppSppRenyiAlphaResult(out_dir=out_dir, summary_json_path=summary_path)
