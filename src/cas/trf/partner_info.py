"""Partner-turn information-tracking TRF analysis.

This analysis is a continuous partner-turn information-tracking TRF. It is
explicitly distinct from the SPP-response and SPP-onset-control analyses: the
predictors track continuous information dynamics during partner speech and the
post-offset planning interval that follows each partner IPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
from scipy.stats import wilcoxon

from cas.annotations.autocorrect import normalize_tier_name
from cas.annotations.io import load_textgrid
from cas.behavior._legacy_support import read_surprisal_tables, resolve_surprisal_paths
from cas.preprocessing.config import build_preprocessing_run_paths, resolve_preprocessing_output_layout
from cas.trf.nested_cv import loro_nested_cv_design_grid
from cas.trf.prepare import canonical_dyad_id, other_speaker_label, prepare_trf_runs
from cas.viz.lmeeeg import plot_joint_model_weights

matplotlib.use("Agg")

from matplotlib import pyplot as plt


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class PartnerInfoSubjectResult:
    subject_id: str
    runs: list[int]
    channel_names: list[str]
    lag_times_s: np.ndarray
    target_results: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ConversationWindow:
    run: int
    anchor_time_s: float
    duration_s: float


def lag_window_samples(*, start_ms: float, stop_ms: float, sfreq_hz: float) -> np.ndarray:
    """Convert a lag window in milliseconds into inclusive integer samples."""

    if sfreq_hz <= 0.0:
        raise ValueError("sfreq_hz must be positive.")
    start = int(np.rint((float(start_ms) / 1000.0) * float(sfreq_hz)))
    stop = int(np.rint((float(stop_ms) / 1000.0) * float(sfreq_hz)))
    if stop < start:
        raise ValueError("stop_ms must be greater than or equal to start_ms.")
    return np.arange(start, stop + 1, dtype=int)


def zscore_predictor(values: np.ndarray) -> np.ndarray:
    """Return a finite z-scored 1D predictor array."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=float)
    mean = float(np.nanmean(array[finite_mask]))
    std = float(np.nanstd(array[finite_mask]))
    if std <= 0.0:
        out = np.zeros_like(array, dtype=float)
        out[~finite_mask] = 0.0
        return out
    out = (array - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out


def build_information_rate_predictor(
    token_table: pd.DataFrame,
    *,
    n_samples: int,
    sfreq_hz: float,
    sigma_ms: float,
    surprisal_column: str = "surprisal",
) -> np.ndarray:
    """Build the duration-filled, smoothed information-rate predictor."""

    raw = np.zeros(int(n_samples), dtype=float)
    if raw.size == 0:
        return raw
    for token in token_table.to_dict("records"):
        surprisal = float(token.get(surprisal_column, np.nan))
        onset = float(token["onset"])
        offset = float(token["offset"])
        duration = float(offset - onset)
        if not np.isfinite(surprisal) or not np.isfinite(duration) or duration <= 0.0:
            continue
        onset_index = max(0, int(np.rint(onset * sfreq_hz)))
        offset_index = min(raw.shape[0], int(np.rint(offset * sfreq_hz)))
        if offset_index <= onset_index:
            offset_index = min(raw.shape[0], onset_index + 1)
        raw[onset_index:offset_index] = surprisal / duration
    return zscore_predictor(_smooth_continuous(raw, sfreq_hz=sfreq_hz, sigma_ms=sigma_ms))


def build_prop_expected_cumulative_info_predictor(
    token_table: pd.DataFrame,
    *,
    n_samples: int,
    sfreq_hz: float,
    sigma_ms: float,
    surprisal_column: str = "surprisal",
    expected_total_column: str = "expected_total_info",
) -> np.ndarray:
    """Build the smoothed proportional expected cumulative information predictor."""

    raw = np.zeros(int(n_samples), dtype=float)
    if raw.size == 0 or token_table.empty:
        return raw

    for _, ipu_tokens in token_table.groupby("partner_ipu_id", sort=False):
        sorted_tokens = ipu_tokens.sort_values(["onset", "offset"], kind="mergesort").reset_index(drop=True)
        expected_total = float(sorted_tokens.iloc[0][expected_total_column])
        if not np.isfinite(expected_total) or expected_total <= 0.0:
            continue
        cumulative = 0.0
        for token_index, token in sorted_tokens.iterrows():
            surprisal = float(token[surprisal_column])
            cumulative += surprisal if np.isfinite(surprisal) else 0.0
            onset_index = max(0, int(np.rint(float(token["onset"]) * sfreq_hz)))
            if token_index + 1 < len(sorted_tokens):
                next_change_s = float(sorted_tokens.iloc[token_index + 1]["onset"])
            else:
                next_change_s = float(sorted_tokens.iloc[-1]["offset"])
            stop_index = min(raw.shape[0], int(np.rint(next_change_s * sfreq_hz)))
            if stop_index <= onset_index:
                stop_index = min(raw.shape[0], onset_index + 1)
            raw[onset_index:stop_index] = cumulative / expected_total
    return zscore_predictor(_smooth_continuous(raw, sfreq_hz=sfreq_hz, sigma_ms=sigma_ms))


def build_partner_info_model_specs(
    control_predictors: Sequence[str],
    models_config: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Return the four-model predictor specification used by the analysis."""

    controls = [str(value) for value in control_predictors]
    if models_config:
        explicit_specs: dict[str, list[str]] = {}
        for model_name in ("N0", "N1", "N2", "N3"):
            model_cfg = dict(models_config.get(model_name) or {})
            predictor_names = [str(value) for value in (model_cfg.get("predictors") or [])]
            if not predictor_names:
                raise ValueError(f"partner_info_trf config model {model_name} is missing predictors.")
            explicit_specs[model_name] = predictor_names
        return explicit_specs
    return {
        "N0": controls,
        "N1": controls + ["information_rate"],
        "N2": controls + ["prop_expected_cumulative_info"],
        "N3": controls + ["information_rate", "prop_expected_cumulative_info"],
    }


def fit_partner_info_subject(
    *,
    config_path: str | Path,
    subject_id: str,
    project_root: str | Path,
    runs: list[int] | None = None,
) -> PartnerInfoSubjectResult:
    """Fit the partner-turn information-tracking TRF for one subject."""

    project_root_path = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    config_root = _discover_config_root(config_path.parent)
    paths_config = _load_paths_config(config_root)

    analysis_cfg = dict(config.get("analysis") or {})
    targets_cfg = dict(config.get("targets") or {})
    models_cfg = dict(config.get("models") or {})
    smoothing_cfg = dict(config.get("smoothing") or {})
    trf_cfg = dict(config.get("trf") or {})
    cv_cfg = dict(config.get("cv") or {})
    predictor_cfg = dict(config.get("predictors") or {})

    requested_runs = (
        list(runs)
        if runs is not None
        else list(range(1, int(cv_cfg.get("n_runs", 0)) + 1))
    )
    if not requested_runs:
        raise ValueError("No runs requested for partner_info_trf.")

    modelling_sfreq_hz = float(targets_cfg["modelling_sampling_rate_hz"])
    if modelling_sfreq_hz >= 64.0:
        raise ValueError("partner_info_trf requires modelling_sampling_rate_hz below 64 Hz.")

    control_predictors = [str(value) for value in predictor_cfg.get("controls", [])]
    model_specs = build_partner_info_model_specs(control_predictors, models_cfg)
    target_kinds = [str(value) for value in targets_cfg.get("include", [])]
    sigma_grid_ms = [float(value) for value in smoothing_cfg.get("sigma_ms_grid", [])]
    ridge_alpha_grid = [float(value) for value in trf_cfg.get("ridge_alpha_grid", [])]
    conversation_duration_s = float((config.get("inputs") or {}).get("conversation_duration_s", 240.0))
    lag_samples = lag_window_samples(
        start_ms=float(trf_cfg["lag_start_ms"]),
        stop_ms=float(trf_cfg["lag_stop_ms"]),
        sfreq_hz=modelling_sfreq_hz,
    )
    lag_times_s = lag_samples.astype(float) / modelling_sfreq_hz

    behaviour_config_path = _resolve_path(
        str((config.get("inputs") or {}).get("behavior_config", "config/behavior/hazard.yaml")),
        project_root=project_root_path,
        config_root=config_root,
    )
    behaviour_config = _load_yaml(behaviour_config_path)
    surprisal_glob = str(((behaviour_config.get("inputs") or {}).get("surprisal_tsv", ""))).strip()
    if not surprisal_glob:
        raise ValueError("behavior_config must define inputs.surprisal_tsv.")
    surprisal_paths = tuple(resolve_surprisal_paths(surprisal_glob))
    surprisal_table, _ = read_surprisal_tables(
        surprisal_paths,
        unmatched_surprisal_strategy="drop",
    )

    verbose = bool((config.get("logging") or {}).get("verbose", False))
    target_results: dict[str, dict[str, Any]] = {}
    channel_names_reference: list[str] | None = None
    conversation_windows = [
        _resolve_conversation_window(
            subject_id=subject_id,
            run=int(run),
            duration_s=conversation_duration_s,
            paths_config=paths_config,
        )
        for run in requested_runs
    ]

    for target_kind in target_kinds:
        target_runs, channel_names, target_sfreq_hz = _load_target_runs(
            subject_id=subject_id,
            runs=requested_runs,
            target_kind=target_kind,
            config=config,
            paths_config=paths_config,
            project_root=project_root_path,
            config_root=config_root,
            conversation_windows=conversation_windows,
        )
        if channel_names_reference is None:
            channel_names_reference = channel_names

        predictor_designs_by_sigma: dict[float, dict[str, list[np.ndarray]]] = {}
        diagnostic_rows: list[dict[str, Any]] = []
        for sigma_ms in sigma_grid_ms:
            predictor_runs_by_name = _build_partner_info_predictor_runs(
                subject_id=subject_id,
                runs=requested_runs,
                surprisal_table=surprisal_table,
                config=config,
                paths_config=paths_config,
                modelling_sfreq_hz=modelling_sfreq_hz,
                sigma_ms=float(sigma_ms),
                target_runs=target_runs,
                target_sfreq_hz=target_sfreq_hz,
                conversation_windows=conversation_windows,
            )
            predictor_designs_by_sigma[float(sigma_ms)] = predictor_runs_by_name
            if "information_rate" in predictor_runs_by_name and "prop_expected_cumulative_info" in predictor_runs_by_name:
                for run_index, run in enumerate(requested_runs):
                    full_design = np.column_stack(
                        [predictor_runs_by_name[name][run_index] for name in model_specs["N3"]]
                    )
                    diagnostic_rows.extend(
                        _diagnostic_rows_from_design(
                            design=full_design,
                            predictor_names=model_specs["N3"],
                            subject_id=subject_id,
                            target_kind=target_kind,
                            run=int(run),
                            sigma_ms=float(sigma_ms),
                        )
                    )

        model_results: dict[str, dict[str, Any]] = {}
        for model_name, predictor_names in model_specs.items():
            design_grid: list[tuple[dict[str, object], list[np.ndarray]]] = []
            for sigma_ms in sigma_grid_ms:
                predictor_runs = _stack_named_predictors(
                    predictor_names=predictor_names,
                    predictor_runs_by_name=predictor_designs_by_sigma[float(sigma_ms)],
                )
                X_runs, Y_runs = prepare_trf_runs(
                    eeg_runs=target_runs,
                    predictor_runs=predictor_runs,
                    eeg_sfreq=target_sfreq_hz,
                    predictor_sfreq=modelling_sfreq_hz,
                    target_sfreq=modelling_sfreq_hz,
                    tmin_s=float(lag_times_s[0]),
                    tmax_s=float(lag_times_s[-1]),
                )
                design_grid.append(({"sigma_ms": float(sigma_ms)}, X_runs))

            fold_scores, fold_coefficients = loro_nested_cv_design_grid(
                X_runs_grid=design_grid,
                Y_runs=Y_runs,
                alphas=ridge_alpha_grid,
                srate=modelling_sfreq_hz,
                tmin_s=float(lag_times_s[0]),
                tmax_s=float(lag_times_s[-1]),
                fit_intercept=bool(trf_cfg.get("fit_intercept", False)),
                scoring=str(cv_cfg.get("scoring_metric", "corr")),
                standardize_X=bool(trf_cfg.get("standardize_X", False)),
                standardize_Y=bool(trf_cfg.get("standardize_Y", False)),
                verbose=verbose,
            )
            model_results[model_name] = {
                "predictors": list(predictor_names),
                "fold_scores": fold_scores,
                "coefficients": np.stack([np.asarray(value, dtype=float) for value in fold_coefficients], axis=0),
            }

        target_results[target_kind] = {
            "target_kind": target_kind,
            "predictor_diagnostics": diagnostic_rows,
            "models": model_results,
        }

    if channel_names_reference is None:
        raise ValueError(f"No EEG channels could be resolved for {subject_id}.")

    if verbose:
        print(
            f"[partner-info-trf] subject={subject_id} analysis={analysis_cfg.get('name', 'partner_info_trf')} "
            f"targets={target_kinds} runs={requested_runs}",
            flush=True,
        )

    return PartnerInfoSubjectResult(
        subject_id=subject_id,
        runs=requested_runs,
        channel_names=channel_names_reference,
        lag_times_s=np.asarray(lag_times_s, dtype=float),
        target_results=target_results,
    )


def write_partner_info_subject_outputs(
    *,
    result: PartnerInfoSubjectResult,
    summary_json: str | Path,
    coefficients_npz: str | Path,
) -> None:
    """Write subject-level JSON and NPZ outputs."""

    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "subject": result.subject_id,
        "runs": [int(value) for value in result.runs],
        "channel_names": list(result.channel_names),
        "lag_times_s": [float(value) for value in result.lag_times_s],
        "targets": {},
    }
    coefficient_payload: dict[str, Any] = {
        "subject": result.subject_id,
        "channel_names": np.asarray(result.channel_names, dtype=object),
        "lag_times_s": np.asarray(result.lag_times_s, dtype=float),
    }
    for target_kind, target_result in result.target_results.items():
        summary_payload["targets"][target_kind] = {
            "predictor_diagnostics": list(target_result["predictor_diagnostics"]),
            "models": {},
        }
        for model_name, model_result in target_result["models"].items():
            summary_payload["targets"][target_kind]["models"][model_name] = {
                "predictors": list(model_result["predictors"]),
                "fold_scores": list(model_result["fold_scores"]),
            }
            coefficient_payload[f"{target_kind}_{model_name}_predictors"] = np.asarray(
                model_result["predictors"],
                dtype=object,
            )
            coefficient_payload[f"{target_kind}_{model_name}_coefficients"] = np.asarray(
                model_result["coefficients"],
                dtype=float,
            )
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    coefficients_path = Path(coefficients_npz)
    coefficients_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(coefficients_path, **coefficient_payload)


def summarize_partner_info_group(
    *,
    subject_summary_paths: Sequence[str | Path],
    subject_coefficient_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Aggregate subject-level outputs into group tables and kernel payloads."""

    if len(subject_summary_paths) != len(subject_coefficient_paths):
        raise ValueError("Expected matched subject summary and coefficient file lists.")

    subject_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    kernel_accumulator: dict[tuple[str, str], list[np.ndarray]] = {}
    channel_names: list[str] | None = None
    lag_times_s: np.ndarray | None = None

    for summary_path, coefficient_path in zip(subject_summary_paths, subject_coefficient_paths):
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        coefficients = np.load(Path(coefficient_path), allow_pickle=True)
        subject_id = str(payload["subject"])
        if channel_names is None:
            channel_names = [str(value) for value in payload["channel_names"]]
        if lag_times_s is None:
            lag_times_s = np.asarray(payload["lag_times_s"], dtype=float)

        for target_kind, target_payload in dict(payload["targets"]).items():
            diagnostic_rows.extend(list(target_payload.get("predictor_diagnostics", [])))
            model_mean_by_name: dict[str, float] = {}
            delta_by_name: dict[str, float] = {}
            model_fold_tables: dict[str, dict[int, dict[str, Any]]] = {}
            for model_name, model_payload in dict(target_payload["models"]).items():
                fold_scores = list(model_payload["fold_scores"])
                model_mean_by_name[model_name] = float(np.nanmean([fold["mean_score"] for fold in fold_scores]))
                model_fold_tables[model_name] = {int(fold["test_run"]): fold for fold in fold_scores}
                for fold in fold_scores:
                    fold_rows.append(
                        {
                            "subject": subject_id,
                            "target": target_kind,
                            "model": model_name,
                            "test_run": int(fold["test_run"]),
                            "mean_r": float(fold["mean_score"]),
                            "selected_alpha": float(fold["selected_alpha"]),
                            "selected_sigma_ms": float(dict(fold["selected_design"]).get("sigma_ms", np.nan)),
                        }
                    )
            for comparison_name, left_model, right_model in _comparison_specs():
                deltas: list[float] = []
                common_runs = sorted(
                    set(model_fold_tables[left_model]).intersection(model_fold_tables[right_model])
                )
                for run in common_runs:
                    left_fold = model_fold_tables[left_model][run]
                    right_fold = model_fold_tables[right_model][run]
                    delta_value = float(left_fold["mean_score"] - right_fold["mean_score"])
                    deltas.append(delta_value)
                    comparison_rows.append(
                        {
                            "subject": subject_id,
                            "target": target_kind,
                            "comparison": comparison_name,
                            "test_run": int(run),
                            "delta_r": delta_value,
                        }
                    )
                delta_by_name[comparison_name] = float(np.nanmean(deltas)) if deltas else np.nan

            subject_row = {
                "subject": subject_id,
                "target": target_kind,
            }
            for model_name, mean_score in model_mean_by_name.items():
                subject_row[f"{model_name}_mean_r"] = mean_score
            for comparison_name, delta_value in delta_by_name.items():
                subject_row[f"delta_r_{comparison_name}"] = delta_value
            subject_rows.append(subject_row)

            full_predictors = [str(value) for value in coefficients[f"{target_kind}_N3_predictors"].tolist()]
            full_coefficients = np.asarray(coefficients[f"{target_kind}_N3_coefficients"], dtype=float)
            mean_kernel = np.nanmean(full_coefficients, axis=0)
            for predictor_name in ("information_rate", "prop_expected_cumulative_info"):
                predictor_index = full_predictors.index(predictor_name)
                kernel_accumulator.setdefault((target_kind, predictor_name), []).append(
                    np.asarray(mean_kernel[:, predictor_index, :].T, dtype=float)
                )

    subject_table = pd.DataFrame(subject_rows).sort_values(["target", "subject"]).reset_index(drop=True)
    fold_table = pd.DataFrame(fold_rows).sort_values(["target", "model", "subject", "test_run"]).reset_index(drop=True)
    comparison_table = pd.DataFrame(comparison_rows).sort_values(
        ["target", "comparison", "subject", "test_run"]
    ).reset_index(drop=True)
    diagnostic_table = pd.DataFrame(diagnostic_rows)

    stats_by_target: dict[str, dict[str, Any]] = {}
    for target_kind in sorted(subject_table["target"].astype(str).unique().tolist()):
        stats_by_target[target_kind] = {}
        target_slice = subject_table.loc[subject_table["target"].astype(str) == target_kind].copy()
        for comparison_name, _, _ in _comparison_specs():
            values = pd.to_numeric(target_slice[f"delta_r_{comparison_name}"], errors="coerce").to_numpy(dtype=float)
            stats_by_target[target_kind][comparison_name] = _signed_rank_summary(values)

    mean_kernels = {
        f"{target_kind}__{predictor_name}": np.nanmean(np.stack(kernel_list, axis=0), axis=0)
        for (target_kind, predictor_name), kernel_list in kernel_accumulator.items()
    }

    return {
        "subject_table": subject_table,
        "fold_table": fold_table,
        "comparison_table": comparison_table,
        "diagnostic_table": diagnostic_table,
        "mean_kernels": mean_kernels,
        "channel_names": list(channel_names or []),
        "lag_times_s": np.asarray(lag_times_s if lag_times_s is not None else [], dtype=float),
        "stats": stats_by_target,
    }


def write_partner_info_group_outputs(
    *,
    summary: Mapping[str, Any],
    summary_json: str | Path,
    subject_csv: str | Path,
    fold_csv: str | Path,
    comparison_csv: str | Path,
    diagnostics_csv: str | Path,
    model_comparison_png: str | Path,
    model_comparison_pdf: str | Path,
    kernel_dir: str | Path,
    sigma_png: str | Path,
    sigma_pdf: str | Path,
    alpha_png: str | Path,
    alpha_pdf: str | Path,
    fold_scores_png: str | Path,
    fold_scores_pdf: str | Path,
    predictor_corr_png: str | Path,
    predictor_corr_pdf: str | Path,
    predictor_variance_png: str | Path,
    predictor_variance_pdf: str | Path,
) -> None:
    """Write aggregated tables and figures."""

    subject_table = summary["subject_table"]
    fold_table = summary["fold_table"]
    comparison_table = summary["comparison_table"]
    diagnostic_table = summary["diagnostic_table"]
    stats_by_target = dict(summary["stats"])
    channel_names = [str(value) for value in summary["channel_names"]]
    lag_times_s = np.asarray(summary["lag_times_s"], dtype=float)

    Path(subject_csv).parent.mkdir(parents=True, exist_ok=True)
    subject_table.to_csv(subject_csv, index=False)
    Path(fold_csv).parent.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(fold_csv, index=False)
    Path(comparison_csv).parent.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(comparison_csv, index=False)
    Path(diagnostics_csv).parent.mkdir(parents=True, exist_ok=True)
    diagnostic_table.to_csv(diagnostics_csv, index=False)

    _plot_model_comparison(
        subject_table=subject_table,
        stats_by_target=stats_by_target,
        output_paths=(Path(model_comparison_png), Path(model_comparison_pdf)),
    )
    _plot_hyperparameter_summary(
        fold_table=fold_table,
        value_column="selected_sigma_ms",
        ylabel="Selected sigma (ms)",
        output_paths=(Path(sigma_png), Path(sigma_pdf)),
    )
    _plot_hyperparameter_summary(
        fold_table=fold_table,
        value_column="selected_alpha",
        ylabel="Selected ridge alpha",
        output_paths=(Path(alpha_png), Path(alpha_pdf)),
    )
    _plot_fold_score_summary(
        fold_table=fold_table,
        output_paths=(Path(fold_scores_png), Path(fold_scores_pdf)),
    )
    _plot_predictor_correlation_heatmap(
        diagnostic_table=diagnostic_table,
        output_paths=(Path(predictor_corr_png), Path(predictor_corr_pdf)),
    )
    _plot_predictor_variance_summary(
        diagnostic_table=diagnostic_table,
        output_paths=(Path(predictor_variance_png), Path(predictor_variance_pdf)),
    )

    kernel_directory = Path(kernel_dir)
    kernel_directory.mkdir(parents=True, exist_ok=True)
    for kernel_key, kernel_array in dict(summary["mean_kernels"]).items():
        target_kind, predictor_name = kernel_key.split("__", 1)
        label = "R" if predictor_name == "information_rate" else "P"
        output_stem = kernel_directory / f"{target_kind}_{label.lower()}_kernel_joint"
        plot_joint_model_weights(
            np.asarray(kernel_array, dtype=float),
            times=lag_times_s,
            channel_names=channel_names,
            output_stem=output_stem,
            title=f"Partner-info TRF kernel | {label} x {target_kind}",
            formats=("png", "pdf"),
            dpi=300,
            line_width=2.5,
        )

    summary_payload = {
        "stats": stats_by_target,
        "subject_csv": str(subject_csv),
        "fold_csv": str(fold_csv),
        "comparison_csv": str(comparison_csv),
        "diagnostics_csv": str(diagnostics_csv),
        "kernel_dir": str(kernel_directory),
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")


def _comparison_specs() -> list[tuple[str, str, str]]:
    return [
        ("N1_vs_N0", "N1", "N0"),
        ("N2_vs_N0", "N2", "N0"),
        ("N3_vs_N0", "N3", "N0"),
        ("N3_vs_N1", "N3", "N1"),
        ("N3_vs_N2", "N3", "N2"),
    ]


def _signed_rank_summary(values: np.ndarray) -> dict[str, Any]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"n_subjects": 0, "statistic": np.nan, "pvalue": np.nan, "mean_delta_r": np.nan}
    if np.allclose(finite_values, 0.0):
        return {"n_subjects": int(finite_values.size), "statistic": 0.0, "pvalue": 1.0, "mean_delta_r": 0.0}
    test = wilcoxon(finite_values, alternative="greater", zero_method="wilcox")
    return {
        "n_subjects": int(finite_values.size),
        "statistic": float(test.statistic),
        "pvalue": float(test.pvalue),
        "mean_delta_r": float(np.nanmean(finite_values)),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping at {path}.")
    return payload


def _discover_config_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "paths.yaml").exists():
            return candidate
    return start


def _load_paths_config(config_root: Path) -> dict[str, Any]:
    return _load_yaml(_discover_config_root(config_root) / "paths.yaml")


def _resolve_path(path_like: str | Path, *, project_root: Path, config_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    repo_candidate = (_discover_config_root(config_root).parent / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    project_candidate = (project_root / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (config_root / path).resolve()


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


def _load_target_runs(
    *,
    subject_id: str,
    runs: Sequence[int],
    target_kind: str,
    config: Mapping[str, Any],
    paths_config: Mapping[str, Any],
    project_root: Path,
    config_root: Path,
    conversation_windows: Sequence[ConversationWindow],
) -> tuple[list[np.ndarray], list[str], float]:
    _configure_mne_runtime()
    import mne

    preprocessing_config_path = _resolve_path(
        str((config.get("inputs") or {}).get("preprocessing_config", "config/preprocessing.yaml")),
        project_root=project_root,
        config_root=config_root,
    )
    preprocessing_config = _load_yaml(preprocessing_config_path)
    preprocessing_layout = resolve_preprocessing_output_layout(
        paths_config,
        preprocessing_config,
        base_dir=project_root,
    )

    target_runs: list[np.ndarray] = []
    channel_names: list[str] | None = None
    sfreq_hz: float | None = None
    for run, conversation_window in zip(runs, conversation_windows):
        raw_path = build_preprocessing_run_paths(
            layout=preprocessing_layout,
            subject=subject_id,
            task="conversation",
            run=str(int(run)),
            dyad_id=canonical_dyad_id(subject_id),
        ).eeg_path
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR").pick("eeg")
        crop_stop_s = min(
            float(conversation_window.anchor_time_s + conversation_window.duration_s),
            float(raw.times[-1]) + (1.0 / float(raw.info["sfreq"])),
        )
        raw.crop(
            tmin=float(conversation_window.anchor_time_s),
            tmax=crop_stop_s,
            include_tmax=False,
        )
        if channel_names is None:
            channel_names = [str(value) for value in raw.ch_names]
        data = raw.get_data()
        sfreq_hz = float(raw.info["sfreq"])
        if target_kind == "raw":
            target = data.T
        else:
            target = _continuous_band_power(
                data=data,
                sfreq_hz=sfreq_hz,
                target_kind=target_kind,
                config=config,
            ).T
        target_runs.append(np.asarray(target, dtype=float))
    if channel_names is None or sfreq_hz is None:
        raise ValueError(f"No target data could be loaded for {subject_id}.")
    return target_runs, channel_names, sfreq_hz


def _continuous_band_power(
    *,
    data: np.ndarray,
    sfreq_hz: float,
    target_kind: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    _configure_mne_runtime()
    import mne

    tf_cfg = dict(config.get("time_frequency") or {})
    method = str(tf_cfg.get("method", "morlet"))
    if method != "morlet":
        raise ValueError(f"Unsupported time-frequency method '{method}'.")
    band_cfg = dict((tf_cfg.get("bands") or {}).get(target_kind) or {})
    if not band_cfg:
        raise ValueError(f"Missing time-frequency band config for target '{target_kind}'.")
    fmin = float(band_cfg["fmin"])
    fmax = float(band_cfg["fmax"])
    morlet_cfg = dict(tf_cfg.get("morlet") or {})
    step_hz = float(morlet_cfg.get("freq_step_hz", 1.0))
    frequencies = np.arange(fmin, fmax + (0.5 * step_hz), step_hz, dtype=float)
    if frequencies.size == 0:
        frequencies = np.asarray([0.5 * (fmin + fmax)], dtype=float)
    n_cycles_cfg = morlet_cfg.get("n_cycles", 6)
    if isinstance(n_cycles_cfg, (int, float)):
        n_cycles = np.full(frequencies.shape, float(n_cycles_cfg), dtype=float)
    else:
        n_cycles = np.asarray(n_cycles_cfg, dtype=float)
    decim_cfg = morlet_cfg.get("decim")
    decim = 1 if decim_cfg in (None, "null") else int(decim_cfg)
    power = mne.time_frequency.tfr_array_morlet(
        data[np.newaxis, :, :],
        sfreq=sfreq_hz,
        freqs=frequencies,
        n_cycles=n_cycles,
        output="power",
        use_fft=bool(morlet_cfg.get("use_fft", True)),
        decim=decim,
        zero_mean=True,
        verbose="ERROR",
    )
    return np.asarray(power.mean(axis=2)[0], dtype=float)


def _build_partner_info_predictor_runs(
    *,
    subject_id: str,
    runs: Sequence[int],
    surprisal_table: pd.DataFrame,
    config: Mapping[str, Any],
    paths_config: Mapping[str, Any],
    modelling_sfreq_hz: float,
    sigma_ms: float,
    target_runs: Sequence[np.ndarray],
    target_sfreq_hz: float,
    conversation_windows: Sequence[ConversationWindow],
) -> dict[str, list[np.ndarray]]:
    predictor_runs_by_name: dict[str, list[np.ndarray]] = {
        "acoustic_envelope": [],
        "word_onset_impulse": [],
        "partner_onset_impulse": [],
        "partner_offset_impulse": [],
        "time_from_partner_onset": [],
        "time_from_partner_offset": [],
        "time_from_partner_offset_squared": [],
        "information_rate": [],
        "prop_expected_cumulative_info": [],
    }
    partner_subject_id = canonical_dyad_id(subject_id)  # placeholder to keep mypy quiet
    del partner_subject_id
    partner_label = other_speaker_label(subject_id)
    dyad_id = canonical_dyad_id(subject_id)
    for run_index, (run, conversation_window) in enumerate(zip(runs, conversation_windows)):
        duration_s = float(target_runs[run_index].shape[0]) / float(target_sfreq_hz)
        n_samples = max(2, int(np.rint(duration_s * modelling_sfreq_hz)))
        sample_times_s = np.arange(n_samples, dtype=float) / float(modelling_sfreq_hz)
        run_tokens = surprisal_table.loc[
            (surprisal_table["dyad_id"].astype(str) == dyad_id)
            & (surprisal_table["run"].astype(str) == str(int(run)))
            & (surprisal_table["speaker"].astype(str) == partner_label)
        ].copy()
        run_tokens = run_tokens.sort_values(["onset", "offset"], kind="mergesort").reset_index(drop=True)
        ipu_table = _load_partner_ipus_from_annotations(
            subject_id=subject_id,
            run=int(run),
            dyad_id=dyad_id,
            partner_label=partner_label,
            paths_config=paths_config,
        )
        token_table = _annotate_tokens_with_ipu_information(run_tokens, ipu_table)

        envelope, _ = _load_partner_envelope_run(
            subject_id=subject_id,
            run=int(run),
            paths_config=paths_config,
            target_n_samples=n_samples,
        )
        predictor_runs_by_name["acoustic_envelope"].append(zscore_predictor(envelope))
        predictor_runs_by_name["word_onset_impulse"].append(
            zscore_predictor(_build_impulse_from_times(token_table["onset"].to_numpy(dtype=float), n_samples, modelling_sfreq_hz))
        )
        predictor_runs_by_name["partner_onset_impulse"].append(
            zscore_predictor(
                _build_impulse_from_times(
                    pd.to_numeric(ipu_table["partner_ipu_onset"], errors="coerce").to_numpy(dtype=float),
                    n_samples,
                    modelling_sfreq_hz,
                )
            )
        )
        predictor_runs_by_name["partner_offset_impulse"].append(
            zscore_predictor(
                _build_impulse_from_times(
                    pd.to_numeric(ipu_table["partner_ipu_offset"], errors="coerce").to_numpy(dtype=float),
                    n_samples,
                    modelling_sfreq_hz,
                )
            )
        )
        onset_time, offset_time, offset_time_sq = _build_timing_controls(
            ipu_table=ipu_table,
            sample_times_s=sample_times_s,
        )
        predictor_runs_by_name["time_from_partner_onset"].append(zscore_predictor(onset_time))
        predictor_runs_by_name["time_from_partner_offset"].append(zscore_predictor(offset_time))
        predictor_runs_by_name["time_from_partner_offset_squared"].append(zscore_predictor(offset_time_sq))
        predictor_runs_by_name["information_rate"].append(
            build_information_rate_predictor(
                token_table,
                n_samples=n_samples,
                sfreq_hz=modelling_sfreq_hz,
                sigma_ms=sigma_ms,
            )
        )
        predictor_runs_by_name["prop_expected_cumulative_info"].append(
            build_prop_expected_cumulative_info_predictor(
                token_table,
                n_samples=n_samples,
                sfreq_hz=modelling_sfreq_hz,
                sigma_ms=sigma_ms,
            )
        )
    return predictor_runs_by_name


def _load_partner_envelope_run(
    *,
    subject_id: str,
    run: int,
    paths_config: Mapping[str, Any],
    target_n_samples: int,
) -> tuple[np.ndarray, float]:
    partner_subject_id = _canonical_partner_id(subject_id)
    envelope_path = (
        Path(str(paths_config["features_root"]))
        / "envelope"
        / partner_subject_id
        / f"{partner_subject_id}_task-conversation_run-{int(run)}_envelope.npy"
    )
    values = np.asarray(np.load(envelope_path, allow_pickle=False), dtype=float).reshape(-1)
    summary_path = envelope_path.with_name(envelope_path.name.replace("_envelope.npy", "_envelope.summary.json"))
    source_sfreq_hz = 100.0
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        source_sfreq_hz = float(payload.get("sampling_rate_hz", source_sfreq_hz))
    resampled = _resample_1d(values, source_length=values.shape[0], target_length=int(target_n_samples))
    return resampled, source_sfreq_hz


def _resolve_conversation_window(
    *,
    subject_id: str,
    run: int,
    duration_s: float,
    paths_config: Mapping[str, Any],
) -> ConversationWindow:
    _configure_mne_runtime()
    import mne

    bids_root = Path(str(paths_config["bids_root"]))
    candidates = [
        bids_root / subject_id / "eeg" / f"{subject_id}_task-conversation_run-{run}_eeg.edf",
        bids_root / subject_id / "eeg" / f"{subject_id}_task-conversation_run-{run}_eeg.fif",
    ]
    raw_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if raw_path is None:
        raise FileNotFoundError(f"No raw EEG file found for {subject_id} run {run}.")

    raw = mne.io.read_raw(raw_path, preload=False, verbose="ERROR")
    anchor_kwargs: dict[str, Any] = {"shortest_event": 1, "verbose": "ERROR"}
    if "Status" in raw.ch_names:
        anchor_kwargs["stim_channel"] = "Status"
    anchor_events = mne.find_events(raw, **anchor_kwargs)
    if anchor_events.size == 0:
        raise ValueError(f"No conversation-start trigger found for {subject_id} run {run}.")

    anchor_sample = int(anchor_events[0, 0])
    anchor_time_s = float((anchor_sample - raw.first_samp) / raw.info["sfreq"])
    return ConversationWindow(
        run=int(run),
        anchor_time_s=anchor_time_s,
        duration_s=float(duration_s),
    )

def _load_partner_ipus_from_annotations(
    *,
    subject_id: str,
    run: int,
    dyad_id: str,
    partner_label: str,
    paths_config: Mapping[str, Any],
) -> pd.DataFrame:
    annotation_path = (
        Path(str(paths_config["annotations_dir"]))
        / f"{canonical_dyad_id(subject_id)}_run-{int(run)}_combined.TextGrid"
    )
    textgrid = load_textgrid(annotation_path)
    target_tier_name = f"ipu-{partner_label}"

    rows: list[dict[str, Any]] = []
    for tier in textgrid.tiers:
        normalized_name = normalize_tier_name(tier.name).value
        if normalized_name != target_tier_name:
            continue
        for interval_index, interval in enumerate(tier.intervals, start=1):
            if not str(interval.text).strip():
                continue
            onset = float(interval.xmin)
            offset = float(interval.xmax)
            if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset:
                continue
            rows.append(
                {
                    "dyad_id": str(dyad_id),
                    "run": str(int(run)),
                    "speaker": str(partner_label),
                    "partner_ipu_id": f"{dyad_id}|run-{int(run)}|{partner_label}|ipu-{interval_index:05d}",
                    "partner_ipu_class": f"{dyad_id}|run-{int(run)}|{partner_label}|ipu-{interval_index:05d}",
                    "partner_ipu_onset": onset,
                    "partner_ipu_offset": offset,
                    "anchor_source": "annotation_ipu_tier",
                }
            )
        break

    if not rows:
        return pd.DataFrame(
            columns=[
                "dyad_id",
                "run",
                "speaker",
                "partner_ipu_id",
                "partner_ipu_class",
                "partner_ipu_onset",
                "partner_ipu_offset",
                "partner_ipu_duration",
                "next_partner_ipu_onset",
                "anchor_source",
            ]
        )

    ipu_table = pd.DataFrame(rows).sort_values(
        ["partner_ipu_onset", "partner_ipu_offset"],
        kind="mergesort",
    ).reset_index(drop=True)
    ipu_table["partner_ipu_duration"] = (
        pd.to_numeric(ipu_table["partner_ipu_offset"], errors="coerce")
        - pd.to_numeric(ipu_table["partner_ipu_onset"], errors="coerce")
    )
    ipu_table["next_partner_ipu_onset"] = (
        pd.to_numeric(ipu_table["partner_ipu_onset"], errors="coerce").shift(-1)
    )
    return ipu_table


def _annotate_tokens_with_ipu_information(
    token_table: pd.DataFrame,
    ipu_table: pd.DataFrame,
) -> pd.DataFrame:
    if token_table.empty:
        out = token_table.copy()
        out["partner_ipu_id"] = pd.Series(dtype="object")
        out["expected_total_info"] = pd.Series(dtype="float64")
        return out
    rows: list[dict[str, Any]] = []
    grouped_ipus = list(ipu_table.to_dict("records"))
    for token in token_table.to_dict("records"):
        match = None
        onset = float(token["onset"])
        offset = float(token["offset"])
        for ipu in grouped_ipus:
            if onset >= float(ipu["partner_ipu_onset"]) - EPSILON and offset <= float(ipu["partner_ipu_offset"]) + EPSILON:
                match = ipu
                break
        row = dict(token)
        row["partner_ipu_id"] = None if match is None else str(match["partner_ipu_id"])
        rows.append(row)
    annotated = pd.DataFrame(rows)
    if annotated["partner_ipu_id"].notna().any():
        totals = (
            annotated.loc[annotated["partner_ipu_id"].notna()]
            .groupby("partner_ipu_id", sort=False)["surprisal"]
            .sum(min_count=1)
            .rename("expected_total_info")
            .reset_index()
        )
        annotated = annotated.merge(totals, on="partner_ipu_id", how="left")
    else:
        annotated["expected_total_info"] = np.nan
    return annotated


def _build_impulse_from_times(event_times_s: np.ndarray, n_samples: int, sfreq_hz: float) -> np.ndarray:
    impulse = np.zeros(int(n_samples), dtype=float)
    finite = np.asarray(event_times_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return impulse
    sample_indices = np.rint(finite * float(sfreq_hz)).astype(int)
    keep = (sample_indices >= 0) & (sample_indices < impulse.shape[0])
    if np.any(keep):
        np.add.at(impulse, sample_indices[keep], 1.0)
    return impulse


def _build_timing_controls(
    *,
    ipu_table: pd.DataFrame,
    sample_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    onset_time = np.zeros(sample_times_s.shape[0], dtype=float)
    offset_time = np.zeros(sample_times_s.shape[0], dtype=float)
    for ipu in ipu_table.to_dict("records"):
        onset = float(ipu["partner_ipu_onset"])
        offset = float(ipu["partner_ipu_offset"])
        segment_end = ipu.get("next_partner_ipu_onset")
        segment_end = float(segment_end) if pd.notna(segment_end) else float(sample_times_s[-1] + (sample_times_s[1] - sample_times_s[0]))
        mask = (sample_times_s >= onset) & (sample_times_s < segment_end)
        onset_time[mask] = sample_times_s[mask] - onset
        offset_time[mask] = sample_times_s[mask] - offset
    return onset_time, offset_time, offset_time**2


def _stack_named_predictors(
    *,
    predictor_names: Sequence[str],
    predictor_runs_by_name: Mapping[str, Sequence[np.ndarray]],
) -> list[np.ndarray]:
    n_runs = len(next(iter(predictor_runs_by_name.values())))
    stacked_runs: list[np.ndarray] = []
    for run_index in range(n_runs):
        columns = [np.asarray(predictor_runs_by_name[name][run_index], dtype=float).reshape(-1, 1) for name in predictor_names]
        common_length = min(column.shape[0] for column in columns)
        stacked_runs.append(np.concatenate([column[:common_length] for column in columns], axis=1))
    return stacked_runs


def _smooth_continuous(values: np.ndarray, *, sfreq_hz: float, sigma_ms: float) -> np.ndarray:
    sigma_samples = (float(sigma_ms) / 1000.0) * float(sfreq_hz)
    if sigma_samples <= 0.0:
        return np.asarray(values, dtype=float)
    return np.asarray(gaussian_filter1d(np.asarray(values, dtype=float), sigma=sigma_samples, mode="nearest"), dtype=float)


def _resample_1d(values: np.ndarray, *, source_length: int, target_length: int) -> np.ndarray:
    if source_length == target_length:
        return np.asarray(values, dtype=float)
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length and target_length must be positive.")
    from math import gcd

    up = int(target_length)
    down = int(source_length)
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor
    return np.asarray(resample_poly(np.asarray(values, dtype=float), up, down, axis=0), dtype=float)[:target_length]


def _diagnostic_rows_from_design(
    *,
    design: np.ndarray,
    predictor_names: Sequence[str],
    subject_id: str,
    target_kind: str,
    run: int,
    sigma_ms: float,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(np.asarray(design, dtype=float), columns=[str(value) for value in predictor_names])
    rows: list[dict[str, Any]] = []
    corr = frame.corr().fillna(0.0)
    for predictor_name in frame.columns:
        rows.append(
            {
                "subject": subject_id,
                "target": target_kind,
                "run": int(run),
                "sigma_ms": float(sigma_ms),
                "predictor": str(predictor_name),
                "mean": float(pd.to_numeric(frame[predictor_name], errors="coerce").mean()),
                "std": float(pd.to_numeric(frame[predictor_name], errors="coerce").std(ddof=0)),
                "missing_fraction": float(pd.to_numeric(frame[predictor_name], errors="coerce").isna().mean()),
                "mean_abs_corr": float(corr[predictor_name].drop(labels=[predictor_name], errors="ignore").abs().mean()),
            }
        )
    return rows


def _plot_model_comparison(
    *,
    subject_table: pd.DataFrame,
    stats_by_target: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_paths: tuple[Path, Path],
) -> None:
    comparisons = [name for name, _, _ in _comparison_specs()]
    targets = sorted(subject_table["target"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), squeeze=False, sharey=True)
    for axis, target_kind in zip(axes[0], targets):
        target_slice = subject_table.loc[subject_table["target"].astype(str) == target_kind]
        data = [
            pd.to_numeric(target_slice[f"delta_r_{comparison_name}"], errors="coerce").dropna().to_numpy(dtype=float)
            for comparison_name in comparisons
        ]
        parts = axis.violinplot(data, positions=np.arange(1, len(comparisons) + 1), showmeans=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
            body.set_facecolor("#5a7d9a")
        axis.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        for position, values in enumerate(data, start=1):
            if values.size:
                jitter = np.linspace(-0.08, 0.08, num=values.size)
                axis.scatter(np.full(values.shape, position, dtype=float) + jitter, values, s=18, color="0.15", alpha=0.7)
            stats = dict(stats_by_target.get(target_kind, {}).get(comparisons[position - 1], {}))
            if values.size:
                y_anchor = float(np.nanmax(values)) + 0.01
            else:
                y_anchor = 0.01
            axis.text(position, y_anchor, _pvalue_stars(float(stats.get("pvalue", np.nan))), ha="center", va="bottom", fontsize=12)
        axis.set_xticks(np.arange(1, len(comparisons) + 1))
        axis.set_xticklabels([value.replace("_vs_", " - ") for value in comparisons], rotation=30, ha="right")
        axis.set_title(target_kind)
        axis.set_ylabel("Delta Pearson r")
    fig.suptitle("Partner-turn information-tracking TRF model comparison", fontsize=14)
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hyperparameter_summary(
    *,
    fold_table: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_paths: tuple[Path, Path],
) -> None:
    targets = sorted(fold_table["target"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False, sharey=False)
    for axis, target_kind in zip(axes[0], targets):
        target_slice = fold_table.loc[fold_table["target"].astype(str) == target_kind]
        model_names = sorted(target_slice["model"].astype(str).unique().tolist())
        data = [
            pd.to_numeric(
                target_slice.loc[target_slice["model"].astype(str) == model_name, value_column],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            for model_name in model_names
        ]
        axis.boxplot(data, labels=model_names)
        axis.set_title(target_kind)
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fold_score_summary(
    *,
    fold_table: pd.DataFrame,
    output_paths: tuple[Path, Path],
) -> None:
    targets = sorted(fold_table["target"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False, sharey=True)
    for axis, target_kind in zip(axes[0], targets):
        target_slice = fold_table.loc[fold_table["target"].astype(str) == target_kind]
        model_names = sorted(target_slice["model"].astype(str).unique().tolist())
        data = [
            pd.to_numeric(
                target_slice.loc[target_slice["model"].astype(str) == model_name, "mean_r"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            for model_name in model_names
        ]
        axis.violinplot(data, positions=np.arange(1, len(model_names) + 1), showmeans=True, widths=0.85)
        axis.set_xticks(np.arange(1, len(model_names) + 1))
        axis.set_xticklabels(model_names)
        axis.set_title(target_kind)
        axis.set_ylabel("Fold Pearson r")
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_predictor_correlation_heatmap(
    *,
    diagnostic_table: pd.DataFrame,
    output_paths: tuple[Path, Path],
) -> None:
    if diagnostic_table.empty:
        return
    summary = (
        diagnostic_table.groupby(["target", "predictor"], dropna=False)["mean_abs_corr"]
        .mean()
        .reset_index()
        .pivot(index="predictor", columns="target", values="mean_abs_corr")
        .fillna(0.0)
    )
    fig, axis = plt.subplots(figsize=(6, max(4, 0.45 * len(summary.index))))
    image = axis.imshow(summary.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(summary.shape[1]))
    axis.set_xticklabels(summary.columns.tolist(), rotation=30, ha="right")
    axis.set_yticks(np.arange(summary.shape[0]))
    axis.set_yticklabels(summary.index.tolist())
    axis.set_title("Mean absolute predictor correlation")
    fig.colorbar(image, ax=axis, label="|r|")
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_predictor_variance_summary(
    *,
    diagnostic_table: pd.DataFrame,
    output_paths: tuple[Path, Path],
) -> None:
    if diagnostic_table.empty:
        return
    summary = (
        diagnostic_table.groupby(["predictor"], dropna=False)[["std", "missing_fraction"]]
        .mean()
        .sort_values("std", ascending=False)
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.4 * len(summary))))
    axes[0].barh(summary["predictor"], summary["std"], color="#5a7d9a")
    axes[0].invert_yaxis()
    axes[0].set_title("Mean predictor SD")
    axes[1].barh(summary["predictor"], summary["missing_fraction"], color="#c97b63")
    axes[1].invert_yaxis()
    axes[1].set_title("Mean missing fraction")
    fig.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pvalue_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n.s."
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def _canonical_partner_id(subject_id: str) -> str:
    subject_number = int(str(subject_id).replace("sub-", "", 1))
    partner_number = subject_number + 1 if subject_number % 2 == 1 else subject_number - 1
    return f"sub-{partner_number:03d}"
