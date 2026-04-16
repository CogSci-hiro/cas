from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def _infer_sfreq(times: np.ndarray) -> float:
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if times.size < 2:
        return 1.0
    deltas = np.diff(times.astype(float))
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return 1.0
    delta = float(np.median(finite))
    if delta <= 0:
        return 1.0
    return 1.0 / delta


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "item"


def _extract_overlapping_channels(message: str) -> list[str]:
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return []
    channel_line = lines[-1]
    return [name.strip() for name in channel_line.split(",") if name.strip()]


def _ensure_2d_map(array: np.ndarray, *, channel_names: list[str], times: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected a 2D array for lmeEEG plotting.")
    expected = (len(channel_names), len(times))
    if data.shape == expected:
        return data
    if data.T.shape == expected:
        return data.T
    raise ValueError(f"Array shape {data.shape} does not match expected {expected}.")


def _pick_peak_indices(score: np.ndarray, *, n_peaks: int, min_separation: int) -> list[int]:
    ranking = np.argsort(np.asarray(score, dtype=float))[::-1]
    peaks: list[int] = []
    for index in ranking:
        if not np.isfinite(score[index]):
            continue
        if any(abs(int(index) - existing) < min_separation for existing in peaks):
            continue
        peaks.append(int(index))
        if len(peaks) >= n_peaks:
            break
    return sorted(peaks)


def _resolve_joint_times(
    beta_map: np.ndarray,
    time_array: np.ndarray,
    *,
    joint_times: str | Sequence[float],
    significance_mask: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(joint_times, str) and joint_times != "peaks":
        raise ValueError(f"Unsupported joint_times mode: {joint_times}")
    if not isinstance(joint_times, str):
        return np.asarray(joint_times, dtype=float)

    score = np.nanmax(np.abs(np.asarray(beta_map, dtype=float)), axis=0)
    if significance_mask is not None:
        significant_times = np.any(np.asarray(significance_mask, dtype=bool), axis=0)
        if np.any(significant_times):
            score = np.where(significant_times, score, -np.inf)
    min_separation = max(1, int(round(time_array.shape[0] / 10.0)))
    peak_indices = _pick_peak_indices(score, n_peaks=3, min_separation=min_separation)
    if not peak_indices:
        fallback = np.linspace(0, max(0, time_array.shape[0] - 1), num=min(3, time_array.shape[0]), dtype=int)
        peak_indices = sorted({int(index) for index in fallback.tolist()})
    return time_array[np.asarray(peak_indices, dtype=int)]


def _contiguous_true_spans(mask: np.ndarray, time_array: np.ndarray) -> list[tuple[float, float]]:
    boolean_mask = np.asarray(mask, dtype=bool)
    if boolean_mask.ndim != 1 or boolean_mask.shape[0] != time_array.shape[0]:
        raise ValueError("mask must be a 1D boolean array aligned to time_array")
    if not np.any(boolean_mask):
        return []

    if time_array.shape[0] > 1:
        finite_deltas = np.diff(np.asarray(time_array, dtype=float))
        finite_deltas = finite_deltas[np.isfinite(finite_deltas) & (finite_deltas > 0)]
        half_step = float(np.median(finite_deltas)) / 2.0 if finite_deltas.size else 0.0
    else:
        half_step = 0.0

    spans: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, flag in enumerate(boolean_mask):
        if flag and start_index is None:
            start_index = index
        if not flag and start_index is not None:
            spans.append((float(time_array[start_index]) - half_step, float(time_array[index - 1]) + half_step))
            start_index = None
    if start_index is not None:
        spans.append((float(time_array[start_index]) - half_step, float(time_array[-1]) + half_step))
    return spans


def _find_joint_timeseries_axis(figure: plt.Figure, *, time_array: np.ndarray):
    time_min = float(np.nanmin(time_array))
    time_max = float(np.nanmax(time_array))
    matching_axes = []
    fallback_axes = []
    for axis in figure.axes:
        if not axis.lines:
            continue
        x_min, x_max = sorted(float(value) for value in axis.get_xlim())
        if x_min <= time_min and x_max >= time_max:
            matching_axes.append(axis)
        fallback_axes.append(axis)
    if matching_axes:
        return max(matching_axes, key=lambda axis: len(axis.lines))
    if fallback_axes:
        return max(fallback_axes, key=lambda axis: len(axis.lines))
    return None


def plot_joint_model_weights(
    array: np.ndarray,
    *,
    times: np.ndarray,
    channel_names: list[str],
    output_stem: str | Path,
    title: str,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
    joint_times: str | Sequence[float] = "peaks",
    significance_mask: np.ndarray | None = None,
) -> list[Path]:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    import mne

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    data = _ensure_2d_map(np.asarray(array), channel_names=list(channel_names), times=np.asarray(times))
    time_array = np.asarray(times, dtype=float)
    channel_names = list(channel_names)
    significance_array = None
    if significance_mask is not None:
        significance_array = _ensure_2d_map(
            np.asarray(significance_mask, dtype=bool),
            channel_names=channel_names,
            times=time_array,
        ).astype(bool)

    info = mne.create_info(channel_names, sfreq=_infer_sfreq(time_array), ch_types=["eeg"] * len(channel_names))
    try:
        info.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass

    evoked = mne.EvokedArray(data, info, tmin=float(time_array[0]), nave=1, comment=title or "lmeeeg")
    selected_joint_times = _resolve_joint_times(
        data,
        time_array,
        joint_times=joint_times,
        significance_mask=significance_array,
    )
    topomap_args: dict[str, object] = {}
    try:
        figure = evoked.plot_joint(times=selected_joint_times, title=title, show=False, topomap_args=topomap_args)
    except ValueError as exc:
        if "overlapping positions" not in str(exc):
            raise
        overlapping_channels = _extract_overlapping_channels(str(exc))
        keep_channels = [name for name in evoked.ch_names if name not in overlapping_channels]
        if not keep_channels:
            raise
        keep_indices = [evoked.ch_names.index(name) for name in keep_channels]
        reduced_topomap_args = dict(topomap_args)
        figure = evoked.copy().pick(keep_channels).plot_joint(
            times=selected_joint_times,
            title=title,
            show=False,
            topomap_args=reduced_topomap_args,
        )

    for axis in figure.axes:
        for line in axis.lines:
            line.set_linewidth(float(line_width))
    if significance_array is not None and np.any(significance_array):
        significant_time_mask = np.any(significance_array, axis=0)
        main_axis = _find_joint_timeseries_axis(figure, time_array=time_array)
        if main_axis is not None:
            for start_time, end_time in _contiguous_true_spans(significant_time_mask, time_array):
                main_axis.axvspan(start_time, end_time, color="0.85", alpha=0.7, zorder=1.5)

    written: list[Path] = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def build_lmeeeg_qc_manifest_from_model_payloads(
    *,
    out_dir: str | Path,
    model_payloads: dict[str, dict[str, object]],
    manifest_path: str | Path,
    significance_masks: dict[str, dict[str, np.ndarray]] | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    del manifest_path
    out_dir = Path(out_dir)
    significance_masks = significance_masks or {}

    plots: list[dict[str, object]] = []
    for model_name, payload in sorted(model_payloads.items()):
        channel_names = [str(name) for name in payload["channel_names"]]
        times = np.asarray(payload["times"], dtype=float)
        column_names = [str(name) for name in payload["column_names"]]
        measures = (
            ("beta", "betas", np.asarray(payload["betas"], dtype=float)),
            ("t_value", "t_values", np.asarray(payload["t_values"], dtype=float)),
        )
        for measure_name, measure_dirname, measure_stack in measures:
            if measure_stack.ndim != 3:
                continue
            for column_index, column_name in enumerate(column_names):
                if column_index >= measure_stack.shape[0]:
                    break
                column_map = _ensure_2d_map(
                    measure_stack[column_index],
                    channel_names=channel_names,
                    times=times,
                )
                if not np.isfinite(column_map).any():
                    continue
                significance_mask = significance_masks.get(model_name, {}).get(column_name)
                output_stem = out_dir / "figures" / "lmeeeg" / measure_dirname / model_name / _sanitize_token(column_name)
                written_paths = plot_joint_model_weights(
                    column_map,
                    times=times,
                    channel_names=channel_names,
                    output_stem=output_stem,
                    title=f"lmeEEG {measure_name} | {model_name} | {column_name}",
                    formats=formats,
                    dpi=dpi,
                    line_width=2.5,
                    significance_mask=significance_mask,
                )
                plots.append(
                    {
                        "model_name": model_name,
                        "kernel": column_name,
                        "measure": measure_name,
                        "has_significance_overlay": significance_mask is not None and bool(np.any(significance_mask)),
                        "files": [str(path) for path in written_paths],
                    }
                )

    return {"status": "ok", "plot_count": len(plots), "plots": plots}


def build_lmeeeg_qc_manifest_from_stats(
    *,
    out_dir: str | Path,
    stats_root: str | Path,
    manifest_path: str | Path,
    model_axes: dict[str, dict[str, object]],
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    del manifest_path
    out_dir = Path(out_dir)
    stats_root = Path(stats_root)

    plots: list[dict[str, object]] = []
    for summary_path in sorted(stats_root.glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = str(summary.get("model_name") or summary_path.parent.parent.name)
        effect_name = str(summary.get("effect") or summary_path.parent.name)
        observed_value = summary.get("observed_statistic")
        corrected_p_value = summary.get("corrected_p_values")
        if not isinstance(observed_value, str) or not isinstance(corrected_p_value, str) or model_name not in model_axes:
            continue

        observed_path = Path(observed_value)
        corrected_p_path = Path(corrected_p_value)
        if not observed_path.exists() or not corrected_p_path.exists():
            continue

        axes = model_axes[model_name]
        times = np.asarray(axes["times"], dtype=float)
        channel_names = [str(name) for name in axes["channel_names"]]
        observed_map = _ensure_2d_map(np.asarray(np.load(observed_path), dtype=float), channel_names=channel_names, times=times)
        corrected_p_map = _ensure_2d_map(
            np.asarray(np.load(corrected_p_path), dtype=float),
            channel_names=channel_names,
            times=times,
        )
        significance_mask = corrected_p_map < 0.05

        stem = out_dir / "figures" / "lmeeeg" / "statistics" / model_name / _sanitize_token(effect_name)
        written = plot_joint_model_weights(
            observed_map,
            times=times,
            channel_names=channel_names,
            output_stem=stem,
            title=f"lmeEEG observed statistic | {model_name} | {effect_name}",
            formats=formats,
            dpi=dpi,
            line_width=2.5,
            significance_mask=significance_mask,
        )
        plots.append(
            {
                "model_name": model_name,
                "kernel": effect_name,
                "source": "stats_fallback",
                "has_significant_samples": bool(np.any(significance_mask)),
                "min_corrected_p": summary.get("min_corrected_p"),
                "n_significant_p_lt_0_05": summary.get("n_significant_p_lt_0_05"),
                "files": [str(path) for path in written],
            }
        )

    return {"status": "ok", "plot_count": len(plots), "plots": plots}
