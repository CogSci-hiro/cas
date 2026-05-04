"""Helpers for the SPP-onset TRF control analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cas.trf.prepare import (
    build_impulse_predictor,
    load_events_table,
    resolve_predictor_paths,
    resolve_speaker_value,
    select_subject_run_events,
)
from trf.nested_cv import loro_nested_cv
from trf.prepare import prepare_trf_runs


def _progress_iter(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False)
    except ModuleNotFoundError:
        return iterable


@dataclass(frozen=True, slots=True)
class SubjectControlResult:
    subject_id: str
    runs: list[int]
    channel_names: list[str]
    times_s: np.ndarray
    model_results: dict[str, dict[str, Any]]


def _load_array(path: str | Path, *, label: str) -> np.ndarray:
    values = np.asarray(np.load(Path(path), allow_pickle=False), dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"{label} must be 1D or 2D, got shape {values.shape}.")
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping at {path}.")
    return payload


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


def _discover_config_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "paths.yaml").exists():
            return candidate
    return start


def _load_paths_config(config_root: Path) -> dict[str, Any]:
    paths_path = _discover_config_root(config_root) / "paths.yaml"
    return _load_yaml(paths_path)


def _load_eeg_channel_names(*, subject_id: str, run: int, paths_config: dict[str, Any]) -> list[str]:
    import os

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import mne

    bids_root = Path(paths_config["bids_root"])
    candidates = [
        bids_root / subject_id / "eeg" / f"{subject_id}_task-conversation_run-{run}_eeg.edf",
        bids_root / subject_id / "eeg" / f"{subject_id}_task-conversation_run-{run}_eeg.fif",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".edf":
            raw = mne.io.read_raw_edf(candidate, preload=False, verbose="ERROR")
        else:
            raw = mne.io.read_raw_fif(candidate, preload=False, verbose="ERROR")
        raw.pick("eeg")
        return [str(name) for name in raw.ch_names]
    raise FileNotFoundError(f"No raw EEG file found for {subject_id} run {run}.")


def _build_impulse_predictor_array(
    *,
    predictor_cfg: dict[str, Any],
    subject_id: str,
    run: int,
    eeg_run: np.ndarray,
    target_sfreq_hz: float,
    events_table: pd.DataFrame,
    dyad_table: pd.DataFrame | None,
) -> np.ndarray:
    speaker_column = str(predictor_cfg["speaker_column"])
    time_column = str(predictor_cfg["time_column"])
    speaker_role = str(predictor_cfg["speaker_role"])
    speaker_value = resolve_speaker_value(subject_id, speaker_role)
    run_events = select_subject_run_events(
        events_table=events_table,
        subject_id=subject_id,
        run=run,
        dyad_table=dyad_table,
    )
    if speaker_column not in run_events.columns:
        raise ValueError(
            f"Impulse predictor speaker column '{speaker_column}' is missing from the events table."
        )
    if time_column not in run_events.columns:
        raise ValueError(
            f"Impulse predictor time column '{time_column}' is missing from the events table."
        )
    selected_times = pd.to_numeric(
        run_events.loc[run_events[speaker_column].astype(str) == speaker_value, time_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    impulse = build_impulse_predictor(
        n_samples=int(eeg_run.shape[0]),
        sfreq_hz=float(target_sfreq_hz),
        event_times_s=selected_times,
    )
    return impulse[:, np.newaxis]


def build_named_predictor_runs(
    *,
    trf_config: dict[str, Any],
    subject_id: str,
    runs: list[int],
    project_root: Path,
    config_root: Path,
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[str]]:
    """Build EEG runs plus named predictor runs for one subject."""

    trf_section = dict(trf_config.get("trf") or {})
    predictor_definitions = dict(trf_section.get("predictor_definitions") or {})
    if not predictor_definitions:
        raise ValueError("TRF control config must define trf.predictor_definitions.")

    paths_config = _load_paths_config(config_root)
    events_csv = _resolve_path(
        str((trf_section.get("events") or {}).get("csv_path", "")),
        project_root=project_root,
        config_root=config_root,
    )
    target_path_template = str((trf_section.get("target") or {}).get("path", ""))
    if not target_path_template:
        raise ValueError("TRF control config must define trf.target.path.")

    events_table = load_events_table(events_csv)
    eeg_array_root = Path(paths_config["eeg_array_root"])
    target_sfreq_hz = float((trf_section.get("timing") or {})["target_sfreq_hz"])
    verbose = bool((trf_section.get("logging") or {}).get("verbose", False))

    eeg_runs: list[np.ndarray] = []
    predictor_runs_by_name = {name: [] for name in predictor_definitions}
    channel_names: list[str] | None = None

    run_iterator = _progress_iter(
        runs,
        enabled=verbose,
        desc=f"{subject_id} runs",
        total=len(runs),
    )
    for run in run_iterator:
        if verbose:
            print(f"[trf-control] preparing subject={subject_id} run={int(run)}")
        eeg_path = _resolve_path(
            eeg_array_root / target_path_template.format(subject=subject_id, run=run),
            project_root=project_root,
            config_root=config_root,
        )
        eeg_run = _load_array(eeg_path, label=f"EEG run {run}")
        eeg_runs.append(eeg_run)
        if verbose:
            print(
                f"[trf-control]  EEG run={int(run)} path={eeg_path} "
                f"shape={tuple(int(value) for value in eeg_run.shape)}"
            )
        if channel_names is None:
            channel_names = _load_eeg_channel_names(
                subject_id=subject_id,
                run=int(run),
                paths_config=paths_config,
            )

        for predictor_name, predictor_cfg in predictor_definitions.items():
            predictor_kind = str(predictor_cfg.get("kind", "continuous"))
            if predictor_kind == "continuous":
                resolved = resolve_predictor_paths(
                    subject_id,
                    run,
                    [dict(name=predictor_name, **predictor_cfg)],
                    None,
                    feature_root=paths_config["features_root"],
                )
                predictor_array = _load_array(
                    _resolve_path(
                        resolved[predictor_name],
                        project_root=project_root,
                        config_root=config_root,
                    ),
                    label=predictor_name,
                )
            elif predictor_kind == "impulse":
                predictor_array = _build_impulse_predictor_array(
                    predictor_cfg=predictor_cfg,
                    subject_id=subject_id,
                    run=int(run),
                    eeg_run=eeg_run,
                    target_sfreq_hz=target_sfreq_hz,
                    events_table=events_table,
                    dyad_table=None,
                )
            else:
                raise ValueError(
                    f"Unsupported TRF predictor kind '{predictor_kind}' for '{predictor_name}'."
                )
            predictor_runs_by_name[predictor_name].append(predictor_array)
            if verbose:
                print(
                    f"[trf-control]  predictor={predictor_name} kind={predictor_kind} "
                    f"run={int(run)} shape={tuple(int(value) for value in predictor_array.shape)}"
                )

    if channel_names is None:
        raise ValueError(f"No channel names could be resolved for {subject_id}.")
    return eeg_runs, predictor_runs_by_name, channel_names


def _stack_predictors_for_model(
    predictor_names: list[str],
    predictor_runs_by_name: dict[str, list[np.ndarray]],
) -> list[np.ndarray]:
    n_runs = len(next(iter(predictor_runs_by_name.values())))
    stacked_runs: list[np.ndarray] = []
    for run_index in range(n_runs):
        run_arrays = [
            np.asarray(predictor_runs_by_name[predictor_name][run_index], dtype=float)
            for predictor_name in predictor_names
        ]
        run_lengths = [int(array.shape[0]) for array in run_arrays]
        common_length = min(run_lengths)
        if common_length <= 1:
            raise ValueError(
                f"Predictor run {run_index + 1} is too short after alignment: {run_lengths}"
            )
        if len(set(run_lengths)) > 1:
            print(
                f"[trf-control] aligning predictor run {run_index + 1} "
                f"from lengths={run_lengths} to common_length={common_length}"
            )
            run_arrays = [array[:common_length] for array in run_arrays]
        stacked_runs.append(np.concatenate(run_arrays, axis=1))
    return stacked_runs


def fit_spp_onset_control_subject(
    *,
    config_path: str | Path,
    subject_id: str,
    project_root: str | Path,
    runs: list[int] | None = None,
) -> SubjectControlResult:
    """Fit the null and full control-analysis TRFs for one subject."""

    project_root_path = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    trf_config = _load_yaml(config_path)
    config_root = _discover_config_root(config_path.parent)
    trf_section = dict(trf_config.get("trf") or {})
    timing_cfg = dict(trf_section.get("timing") or {})
    cv_cfg = dict(trf_section.get("cv") or {})
    model_cfg = dict(trf_section.get("model") or {})
    models_cfg = dict(trf_section.get("models") or {})
    if not models_cfg:
        raise ValueError("TRF control config must define trf.models.")

    requested_runs = (
        list(runs)
        if runs is not None
        else list(range(1, int(cv_cfg.get("n_runs", 0)) + 1))
    )
    if not requested_runs:
        raise ValueError("No runs requested for the TRF control analysis.")

    verbose = bool((trf_section.get("logging") or {}).get("verbose", False))
    if verbose:
        print(
            f"[trf-control] subject fit start: subject={subject_id} "
            f"runs={requested_runs} models={list(models_cfg)}"
        )

    eeg_runs, predictor_runs_by_name, channel_names = build_named_predictor_runs(
        trf_config=trf_config,
        subject_id=subject_id,
        runs=requested_runs,
        project_root=project_root_path,
        config_root=config_root,
    )

    tmin_s = float(timing_cfg["tmin_s"])
    tmax_s = float(timing_cfg["tmax_s"])
    target_sfreq_hz = float(timing_cfg["target_sfreq_hz"])
    lag_samples = np.arange(
        int(np.rint(tmin_s * target_sfreq_hz)),
        int(np.rint(tmax_s * target_sfreq_hz)) + 1,
        dtype=int,
    )
    times_s = lag_samples.astype(float) / target_sfreq_hz

    model_results: dict[str, dict[str, Any]] = {}
    model_iterator = _progress_iter(
        list(models_cfg.items()),
        enabled=verbose,
        desc=f"{subject_id} models",
        total=len(models_cfg),
    )
    for model_name, model_definition in model_iterator:
        predictor_names = [str(name) for name in model_definition.get("predictors", [])]
        if not predictor_names:
            raise ValueError(f"TRF model '{model_name}' must define at least one predictor.")
        if verbose:
            print(
                f"[trf-control] fitting subject={subject_id} model={model_name} "
                f"predictors={predictor_names}"
            )
        predictor_runs = _stack_predictors_for_model(predictor_names, predictor_runs_by_name)
        X_runs, Y_runs = prepare_trf_runs(
            eeg_runs=eeg_runs,
            predictor_runs=predictor_runs,
            eeg_sfreq=target_sfreq_hz,
            predictor_sfreq=target_sfreq_hz,
            target_sfreq=target_sfreq_hz,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
        )
        fold_scores, fold_coefficients = loro_nested_cv(
            X_runs=X_runs,
            Y_runs=Y_runs,
            alphas=[float(alpha) for alpha in model_cfg["alphas"]],
            srate=target_sfreq_hz,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            fit_intercept=bool(model_cfg.get("fit_intercept", False)),
            scoring=str((trf_section.get("scoring") or {}).get("metric", "corr")),
            standardize_X=bool(model_cfg.get("standardize_X", True)),
            standardize_Y=bool(model_cfg.get("standardize_Y", False)),
            verbose=bool((trf_section.get("logging") or {}).get("verbose", False)),
        )
        if verbose:
            mean_scores = [float(fold["mean_score"]) for fold in fold_scores]
            print(
                f"[trf-control] completed subject={subject_id} model={model_name} "
                f"mean_score={float(np.nanmean(mean_scores)):.6f}"
            )
        model_results[model_name] = {
            "predictors": predictor_names,
            "fold_scores": fold_scores,
            "coefficients": np.stack([np.asarray(value, dtype=float) for value in fold_coefficients], axis=0),
        }

    if verbose:
        print(f"[trf-control] subject fit complete: subject={subject_id}")

    return SubjectControlResult(
        subject_id=subject_id,
        runs=requested_runs,
        channel_names=channel_names,
        times_s=np.asarray(times_s, dtype=float),
        model_results=model_results,
    )


def subject_result_to_json_payload(result: SubjectControlResult) -> dict[str, Any]:
    """Serialize a subject-level control result for JSON output."""

    payload = {
        "subject": result.subject_id,
        "runs": [int(run) for run in result.runs],
        "channel_names": list(result.channel_names),
        "times_s": [float(value) for value in result.times_s],
        "models": {},
    }
    for model_name, model_result in result.model_results.items():
        payload["models"][model_name] = {
            "predictors": list(model_result["predictors"]),
            "fold_scores": list(model_result["fold_scores"]),
        }
    return payload


def write_subject_control_outputs(
    *,
    result: SubjectControlResult,
    summary_json: str | Path,
    coefficients_npz: str | Path,
) -> None:
    """Write subject-level control outputs."""

    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[trf-control] writing summary JSON: {summary_path}")
    summary_path.write_text(
        json.dumps(subject_result_to_json_payload(result), indent=2) + "\n",
        encoding="utf-8",
    )

    coefficient_payload: dict[str, Any] = {
        "subject": result.subject_id,
        "times_s": np.asarray(result.times_s, dtype=float),
        "channel_names": np.asarray(result.channel_names, dtype=object),
    }
    for model_name, model_result in result.model_results.items():
        coefficient_payload[f"{model_name}_predictors"] = np.asarray(
            model_result["predictors"],
            dtype=object,
        )
        coefficient_payload[f"{model_name}_coefficients"] = np.asarray(
            model_result["coefficients"],
            dtype=float,
        )

    coefficient_path = Path(coefficients_npz)
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[trf-control] writing coefficient NPZ: {coefficient_path}")
    np.savez(coefficient_path, **coefficient_payload)


def summarize_spp_onset_control_group(
    *,
    subject_summary_paths: list[str | Path],
    subject_coefficient_paths: list[str | Path],
    full_model_name: str = "full",
    null_model_name: str = "null",
    kernel_predictor: str = "spp_onset",
) -> dict[str, Any]:
    """Aggregate subject-level control outputs into a group summary."""

    if len(subject_summary_paths) != len(subject_coefficient_paths):
        raise ValueError("Expected matched subject summary and coefficient file lists.")

    print(
        f"[trf-control] aggregating group results: n_subjects={len(subject_summary_paths)} "
        f"full_model={full_model_name} null_model={null_model_name}"
    )
    subject_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    kernel_arrays: list[np.ndarray] = []
    channel_names: list[str] | None = None
    times_s: np.ndarray | None = None

    for summary_path, coefficient_path in zip(subject_summary_paths, subject_coefficient_paths):
        print(
            f"[trf-control] reading subject outputs: "
            f"summary={Path(summary_path).name} coef={Path(coefficient_path).name}"
        )
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        coefficients = np.load(Path(coefficient_path), allow_pickle=True)
        subject_id = str(payload["subject"])

        full_scores = payload["models"][full_model_name]["fold_scores"]
        null_scores = payload["models"][null_model_name]["fold_scores"]
        for full_fold, null_fold in zip(full_scores, null_scores):
            full_channel_scores = np.asarray(full_fold["channel_scores"], dtype=float)
            null_channel_scores = np.asarray(null_fold["channel_scores"], dtype=float)
            fold_rows.append(
                {
                    "subject": subject_id,
                    "test_run": int(full_fold["test_run"]),
                    "full_mean_r": float(np.nanmean(full_channel_scores)),
                    "null_mean_r": float(np.nanmean(null_channel_scores)),
                    "delta_mean_r": float(np.nanmean(full_channel_scores - null_channel_scores)),
                }
            )

        subject_delta = float(
            np.nanmean(
                np.asarray([row["delta_mean_r"] for row in fold_rows if row["subject"] == subject_id], dtype=float)
            )
        )
        subject_rows.append(
            {
                "subject": subject_id,
                "full_mean_r": float(np.nanmean([fold["mean_score"] for fold in full_scores])),
                "null_mean_r": float(np.nanmean([fold["mean_score"] for fold in null_scores])),
                "delta_mean_r": subject_delta,
            }
        )

        loaded_times = np.asarray(coefficients["times_s"], dtype=float)
        loaded_channel_names = [str(value) for value in coefficients["channel_names"].tolist()]
        full_predictors = [str(value) for value in coefficients[f"{full_model_name}_predictors"].tolist()]
        full_coefficients = np.asarray(coefficients[f"{full_model_name}_coefficients"], dtype=float)
        predictor_index = full_predictors.index(kernel_predictor)
        kernel = np.nanmean(full_coefficients[:, :, predictor_index, :], axis=0).T
        kernel_arrays.append(np.asarray(kernel, dtype=float))

        if channel_names is None:
            channel_names = loaded_channel_names
        if times_s is None:
            times_s = loaded_times

    subject_frame = pd.DataFrame(subject_rows).sort_values("subject").reset_index(drop=True)
    fold_frame = pd.DataFrame(fold_rows).sort_values(["subject", "test_run"]).reset_index(drop=True)
    deltas = subject_frame["delta_mean_r"].to_numpy(dtype=float)

    if deltas.size == 0:
        statistic = np.nan
        pvalue = np.nan
    elif np.allclose(deltas, 0.0):
        statistic = 0.0
        pvalue = 1.0
    else:
        test = wilcoxon(deltas, alternative="greater", zero_method="wilcox")
        statistic = float(test.statistic)
        pvalue = float(test.pvalue)

    mean_kernel = np.nanmean(np.stack(kernel_arrays, axis=0), axis=0)
    print(
        f"[trf-control] group aggregation complete: "
        f"mean_delta_r={float(np.nanmean(deltas)) if deltas.size else float('nan'):.6f} "
        f"pvalue={float(pvalue) if np.isfinite(pvalue) else float('nan'):.6g}"
    )
    return {
        "subject_table": subject_frame,
        "fold_table": fold_frame,
        "kernel": mean_kernel,
        "channel_names": list(channel_names or []),
        "times_s": np.asarray(times_s if times_s is not None else [], dtype=float),
        "stats": {
            "test": "wilcoxon_signed_rank",
            "alternative": "greater",
            "n_subjects": int(subject_frame.shape[0]),
            "statistic": statistic,
            "pvalue": pvalue,
            "mean_delta_r": float(np.nanmean(deltas)) if deltas.size else np.nan,
            "median_delta_r": float(np.nanmedian(deltas)) if deltas.size else np.nan,
        },
    }
