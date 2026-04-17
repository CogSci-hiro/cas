"""Post-fit QC plots for the causal TDE-HMM pipeline.

This module consumes saved artifacts from :mod:`cas.hmm.tde_hmm` and writes a
set of QC figures plus a small markdown summary report.
"""

from __future__ import annotations

import inspect
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from cas.hmm.tde_hmm import compute_state_entropy

# ---------------------------------------------------------------------------
# Constants

DEFAULT_DPI = 150
DEFAULT_EPSILON = 1e-12
DEFAULT_EXAMPLE_COUNT = 3
DEFAULT_TIME_TO_EVENT_BIN_WIDTH_S = 0.1
DEFAULT_TIME_TO_EVENT_HORIZON_S = 5.0
DEFAULT_FPP_WINDOW_S = (-2.0, 0.0)
DEFAULT_CHUNK_START_WINDOW_S = (0.0, 1.0)
FIGURE_SUFFIXES = (".png", ".pdf")
REQUIRED_ENTROPY_COLUMNS = ("sample", "time_s", "state_entropy")
REQUIRED_GAMMA_INDEX_COLUMNS = ("processed_index", "subject", "run", "original_sample")
REQUIRED_EVENT_COLUMNS = ("subject", "run", "onset_s", "onset_sample")
REQUIRED_CHUNK_COLUMNS = (
    "subject",
    "run",
    "chunk_id",
    "original_start_sample",
    "original_stop_sample",
)
REQUIRED_MODEL_SELECTION_COLUMNS = ("k", "free_energy")
REQUIRED_FREE_ENERGY_TRACE_COLUMNS = ("cycle", "free_energy")
REQUIRED_INIT_FREE_ENERGY_COLUMNS = ("initialization", "free_energy")
REQUIRED_PCA_COLUMNS = ("component", "explained_variance_ratio")


# ---------------------------------------------------------------------------
# Dataclasses


@dataclass(slots=True)
class QcInputs:
    """Container for loaded QC artifacts.

    Parameters
    ----------
    output_dir : Path
        Root directory containing HMM outputs.
    selected_k : int
        Selected number of states.
    fit_summary : dict[str, Any]
        Parsed fit summary payload.
    model_selection : pandas.DataFrame
        Per-K model selection summary.
    chunks : pandas.DataFrame
        Chunk metadata table.
    gamma : numpy.ndarray
        Posterior state probabilities with shape ``(n_samples, k)``.
    state_entropy_processed : numpy.ndarray
        Samplewise entropy on the processed timeline.
    entropy_by_run : dict[tuple[str, str], pandas.DataFrame]
        Original-timeline entropy traces keyed by ``(subject, run)``.
    warnings : list[str]
        Accumulated warnings for skipped optional inputs or plots.
    """

    output_dir: Path
    selected_k: int
    fit_summary: dict[str, Any]
    model_selection: pd.DataFrame
    chunks: pd.DataFrame
    gamma: np.ndarray
    state_entropy_processed: np.ndarray
    entropy_by_run: dict[tuple[str, str], pd.DataFrame]
    warnings: list[str] = field(default_factory=list)
    sampling_rate_hz: float | None = None
    gamma_index: pd.DataFrame | None = None
    pca_explained_variance: pd.DataFrame | None = None
    training_free_energy: pd.DataFrame | None = None
    init_free_energy: pd.DataFrame | None = None
    viterbi: np.ndarray | None = None
    fpp_events: pd.DataFrame | None = None
    model: Any | None = None
    paths: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation helpers


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(sorted(missing))}.")


def _validate_gamma(gamma: np.ndarray, *, label: str = "gamma") -> np.ndarray:
    values = np.asarray(gamma, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"`{label}` must be a 2D array.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"`{label}` must contain at least one sample and one state.")
    if not np.isfinite(values).all():
        raise ValueError(f"`{label}` contains NaN or infinite values.")
    if np.any(values < 0.0):
        raise ValueError(f"`{label}` contains negative values.")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"`{label}` rows must sum to one within tolerance.")
    return values


def _validate_entropy(entropy: np.ndarray, *, expected_length: int) -> np.ndarray:
    values = np.asarray(entropy, dtype=float).reshape(-1)
    if values.shape[0] != expected_length:
        raise ValueError(
            "`state_entropy_processed` length does not match the number of processed samples."
        )
    if not np.isfinite(values).all():
        raise ValueError("`state_entropy_processed` contains NaN or infinite values.")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _collect_entropy_csvs(output_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    entropy_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    for csv_path in sorted(output_dir.glob("subject-*_run-*_state_entropy.csv")):
        frame = pd.read_csv(csv_path)
        _require_columns(frame, REQUIRED_ENTROPY_COLUMNS, label=f"entropy file {csv_path.name}")
        subject_value = str(csv_path.name.split("_run-")[0].removeprefix("subject-"))
        run_value = str(csv_path.name.split("_run-")[1].removesuffix("_state_entropy.csv"))
        entropy_by_run[(subject_value, run_value)] = frame.copy()
    return entropy_by_run


def _append_missing_warning(
    warnings: list[str],
    *,
    label: str,
    path: Path,
) -> None:
    warnings.append(f"Missing optional input `{label}` at `{path}`; skipped related QC outputs.")


def _primary_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.png"


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for suffix in FIGURE_SUFFIXES:
        output_path = output_dir / f"{stem}{suffix}"
        figure.savefig(output_path, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


def _boxplot(axis: plt.Axes, values: list[np.ndarray], labels: list[str]) -> None:
    parameters = inspect.signature(axis.boxplot).parameters
    if "tick_labels" in parameters:
        axis.boxplot(values, tick_labels=labels)
        return
    axis.boxplot(values, labels=labels)


def _coerce_run_sort_columns(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = frame.copy()
    sortable["subject"] = sortable["subject"].astype(str)
    sortable["run"] = sortable["run"].astype(str)
    return sortable.sort_values(["subject", "run"], kind="mergesort").reset_index(drop=True)


def _infer_selected_k(fit_summary: dict[str, Any]) -> int:
    if "selected_k" not in fit_summary:
        raise ValueError("`fit_summary.json` does not contain `selected_k`.")
    selected_k = int(fit_summary["selected_k"])
    if selected_k <= 0:
        raise ValueError("`selected_k` must be positive.")
    return selected_k


def _normalize_entropy(entropy: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.zeros_like(np.asarray(entropy, dtype=float))
    return np.asarray(entropy, dtype=float) / math.log(float(k))


def compute_normalized_state_entropy(gamma: np.ndarray) -> np.ndarray:
    """Compute normalized samplewise state entropy.

    Parameters
    ----------
    gamma : numpy.ndarray
        Posterior state probabilities with shape ``(n_samples, k)``.

    Returns
    -------
    numpy.ndarray
        Entropy normalized by ``log(k)``.
    """

    posterior = _validate_gamma(gamma)
    entropy = compute_state_entropy(posterior)
    return _normalize_entropy(entropy, posterior.shape[1])


def compute_chunk_durations(chunks: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    """Compute chunk durations in samples and seconds.

    Parameters
    ----------
    chunks : pandas.DataFrame
        Chunk table with inclusive-exclusive original sample bounds.
    sampling_rate_hz : float
        Sampling rate in Hz.

    Returns
    -------
    pandas.DataFrame
        Copy of ``chunks`` with ``duration_samples`` and ``duration_seconds``.
    """

    _require_columns(chunks, REQUIRED_CHUNK_COLUMNS, label="chunks")
    if sampling_rate_hz <= 0:
        raise ValueError("`sampling_rate_hz` must be positive.")
    durations = chunks.copy()
    durations["duration_samples"] = (
        durations["original_stop_sample"].to_numpy(dtype=int)
        - durations["original_start_sample"].to_numpy(dtype=int)
    )
    if np.any(durations["duration_samples"].to_numpy(dtype=int) <= 0):
        raise ValueError("All chunk durations must be positive.")
    durations["duration_seconds"] = durations["duration_samples"].to_numpy(dtype=float) / sampling_rate_hz
    return durations


def compute_delta_free_energy(model_selection: pd.DataFrame) -> pd.DataFrame:
    """Compute best free-energy deltas across consecutive K values."""

    _require_columns(model_selection, REQUIRED_MODEL_SELECTION_COLUMNS, label="model_selection")
    ordered = model_selection.sort_values("k", kind="mergesort").reset_index(drop=True).copy()
    ordered["delta_free_energy"] = ordered["free_energy"].diff()
    return ordered


def compute_empirical_transition_matrix(
    viterbi: np.ndarray,
    n_states: int | None = None,
) -> np.ndarray:
    """Estimate an empirical transition matrix from a hard state path."""

    labels = np.asarray(viterbi, dtype=int).reshape(-1)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("`viterbi` must be a non-empty 1D array.")
    if np.any(labels < 0):
        raise ValueError("`viterbi` cannot contain negative state labels.")
    state_count = int(n_states if n_states is not None else labels.max() + 1)
    counts = np.zeros((state_count, state_count), dtype=float)
    for start_state, stop_state in zip(labels[:-1], labels[1:], strict=False):
        counts[int(start_state), int(stop_state)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        transitions = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0.0)
    return transitions


def _extract_transition_matrix_from_model(model: Any, n_states: int) -> np.ndarray | None:
    if model is None:
        return None
    candidate_attributes = ("P", "trans_prob", "transition_matrix", "transmat_", "A")
    for attribute in candidate_attributes:
        if not hasattr(model, attribute):
            continue
        values = np.asarray(getattr(model, attribute), dtype=float)
        if values.shape == (n_states, n_states) and np.isfinite(values).all():
            row_sums = values.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0.0)
    return None


def _run_key(subject: str, run: str) -> tuple[str, str]:
    return str(subject), str(run)


def _choose_example_runs(
    entropy_by_run: dict[tuple[str, str], pd.DataFrame],
    n_examples: int,
) -> list[tuple[str, str]]:
    summary_rows: list[dict[str, Any]] = []
    for (subject, run), frame in entropy_by_run.items():
        entropy_values = frame["state_entropy"].to_numpy(dtype=float)
        valid_values = entropy_values[np.isfinite(entropy_values)]
        if valid_values.size == 0:
            continue
        duration_s = float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0]) if len(frame) > 1 else 0.0
        summary_rows.append(
            {
                "subject": subject,
                "run": run,
                "duration_s": duration_s,
                "entropy_std": float(np.std(valid_values)),
            }
        )
    if not summary_rows:
        return []
    summary = pd.DataFrame(summary_rows)
    median_std = float(summary["entropy_std"].median())
    summary["std_distance"] = np.abs(summary["entropy_std"] - median_std)
    chosen = summary.sort_values(
        ["std_distance", "duration_s", "subject", "run"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).head(n_examples)
    return [(_run_key(row["subject"], row["run"])) for row in chosen.to_dict("records")]


def _collect_aligned_series(
    entropy_frame: pd.DataFrame,
    anchors: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    entropy_values = entropy_frame["state_entropy"].to_numpy(dtype=float)
    aligned = np.full((anchors.shape[0], offsets.shape[0]), np.nan, dtype=float)
    for anchor_index, anchor_sample in enumerate(anchors):
        sample_indices = anchor_sample + offsets
        in_bounds = (sample_indices >= 0) & (sample_indices < entropy_values.shape[0])
        aligned[anchor_index, in_bounds] = entropy_values[sample_indices[in_bounds]]
    return aligned


def _mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_counts = np.sum(np.isfinite(values), axis=0)
    mean_values = np.full(values.shape[1], np.nan, dtype=float)
    std_values = np.full(values.shape[1], np.nan, dtype=float)
    for column_index in range(values.shape[1]):
        column_values = values[:, column_index]
        finite_values = column_values[np.isfinite(column_values)]
        if finite_values.size == 0:
            continue
        mean_values[column_index] = float(np.mean(finite_values))
        if finite_values.size > 1:
            std_values[column_index] = float(np.std(finite_values, ddof=1))
    sem_values = np.divide(
        std_values,
        np.sqrt(valid_counts),
        out=np.full_like(std_values, np.nan),
        where=valid_counts > 1,
    )
    return mean_values, sem_values


def _load_optional_model(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Public loaders


def load_qc_inputs(output_dir: Path, selected_k: int | None = None) -> QcInputs:
    """Load available QC inputs from an HMM output directory.

    Parameters
    ----------
    output_dir : Path
        HMM output directory produced by the fitting step.
    selected_k : int | None, optional
        Selected number of states. When omitted, this is inferred from
        ``fit_summary.json``.

    Returns
    -------
    QcInputs
        Loaded and validated QC inputs.
    """

    resolved_output_dir = Path(output_dir).resolve()
    if not resolved_output_dir.exists():
        raise FileNotFoundError(f"HMM output directory does not exist: {resolved_output_dir}")

    fit_summary_path = resolved_output_dir / "fit_summary.json"
    model_selection_path = resolved_output_dir / "model_selection.csv"
    chunks_path = resolved_output_dir / "chunks.csv"
    if not fit_summary_path.exists():
        raise FileNotFoundError(f"Missing required file: {fit_summary_path}")
    if not model_selection_path.exists():
        raise FileNotFoundError(f"Missing required file: {model_selection_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing required file: {chunks_path}")

    fit_summary = _load_json(fit_summary_path)
    inferred_selected_k = _infer_selected_k(fit_summary)
    chosen_k = inferred_selected_k if selected_k is None else int(selected_k)
    if chosen_k <= 0:
        raise ValueError("`selected_k` must be positive.")

    gamma_path = resolved_output_dir / f"gamma_k{chosen_k}.npy"
    entropy_path = resolved_output_dir / f"state_entropy_processed_k{chosen_k}.npy"
    if not gamma_path.exists():
        raise FileNotFoundError(f"Missing required file: {gamma_path}")
    if not entropy_path.exists():
        raise FileNotFoundError(f"Missing required file: {entropy_path}")

    gamma = _validate_gamma(np.load(gamma_path))
    if gamma.shape[1] != chosen_k:
        raise ValueError(
            f"Gamma state dimension ({gamma.shape[1]}) does not match selected_k ({chosen_k})."
        )
    state_entropy_processed = _validate_entropy(np.load(entropy_path), expected_length=gamma.shape[0])
    model_selection = pd.read_csv(model_selection_path)
    _require_columns(model_selection, REQUIRED_MODEL_SELECTION_COLUMNS, label="model_selection")
    chunks = pd.read_csv(chunks_path)
    _require_columns(chunks, REQUIRED_CHUNK_COLUMNS, label="chunks")
    entropy_by_run = _collect_entropy_csvs(resolved_output_dir)

    warnings: list[str] = []
    gamma_index_path = resolved_output_dir / "subject_run_gamma_index.csv"
    gamma_index = _load_optional_csv(gamma_index_path)
    if gamma_index is None:
        _append_missing_warning(warnings, label="subject_run_gamma_index.csv", path=gamma_index_path)
    else:
        _require_columns(gamma_index, REQUIRED_GAMMA_INDEX_COLUMNS, label="subject_run_gamma_index.csv")
        if len(gamma_index) != gamma.shape[0]:
            raise ValueError(
                "subject_run_gamma_index.csv length does not match the number of processed samples."
            )

    pca_path = resolved_output_dir / "pca_explained_variance.csv"
    pca_explained_variance = _load_optional_csv(pca_path)
    if pca_explained_variance is None:
        _append_missing_warning(warnings, label="pca_explained_variance.csv", path=pca_path)
    else:
        _require_columns(pca_explained_variance, REQUIRED_PCA_COLUMNS, label="pca_explained_variance.csv")

    training_free_energy_path = resolved_output_dir / f"training_free_energy_k{chosen_k}.csv"
    training_free_energy = _load_optional_csv(training_free_energy_path)
    if training_free_energy is None:
        _append_missing_warning(
            warnings,
            label=f"training_free_energy_k{chosen_k}.csv",
            path=training_free_energy_path,
        )
    else:
        _require_columns(
            training_free_energy,
            REQUIRED_FREE_ENERGY_TRACE_COLUMNS,
            label=f"training_free_energy_k{chosen_k}.csv",
        )

    init_free_energy_path = resolved_output_dir / f"init_free_energy_k{chosen_k}.csv"
    init_free_energy = _load_optional_csv(init_free_energy_path)
    if init_free_energy is None:
        _append_missing_warning(
            warnings,
            label=f"init_free_energy_k{chosen_k}.csv",
            path=init_free_energy_path,
        )
    else:
        _require_columns(
            init_free_energy,
            REQUIRED_INIT_FREE_ENERGY_COLUMNS,
            label=f"init_free_energy_k{chosen_k}.csv",
        )

    viterbi_path = resolved_output_dir / f"viterbi_k{chosen_k}.npy"
    viterbi: np.ndarray | None = None
    if viterbi_path.exists():
        viterbi = np.asarray(np.load(viterbi_path), dtype=int).reshape(-1)
        if viterbi.shape[0] != gamma.shape[0]:
            raise ValueError("Saved viterbi path length does not match gamma length.")
    else:
        _append_missing_warning(warnings, label=f"viterbi_k{chosen_k}.npy", path=viterbi_path)

    fpp_events_path = resolved_output_dir / "fpp_events.csv"
    fpp_events = _load_optional_csv(fpp_events_path)
    if fpp_events is None:
        _append_missing_warning(warnings, label="fpp_events.csv", path=fpp_events_path)
    else:
        _require_columns(fpp_events, REQUIRED_EVENT_COLUMNS, label="fpp_events.csv")

    model_path = resolved_output_dir / f"hmm_k{chosen_k}.pkl"
    model = _load_optional_model(model_path)
    if model is None:
        _append_missing_warning(warnings, label=f"hmm_k{chosen_k}.pkl", path=model_path)

    sampling_rate_hz = None
    if "sampling_rate_hz" in fit_summary:
        sampling_rate_hz = float(fit_summary["sampling_rate_hz"])
        if sampling_rate_hz <= 0:
            raise ValueError("`sampling_rate_hz` in fit_summary must be positive.")

    return QcInputs(
        output_dir=resolved_output_dir,
        selected_k=chosen_k,
        fit_summary=fit_summary,
        model_selection=model_selection,
        chunks=_coerce_run_sort_columns(chunks),
        gamma=gamma,
        state_entropy_processed=state_entropy_processed,
        entropy_by_run=entropy_by_run,
        warnings=warnings,
        sampling_rate_hz=sampling_rate_hz,
        gamma_index=_coerce_run_sort_columns(gamma_index) if gamma_index is not None else None,
        pca_explained_variance=pca_explained_variance,
        training_free_energy=training_free_energy,
        init_free_energy=init_free_energy,
        viterbi=viterbi,
        fpp_events=_coerce_run_sort_columns(fpp_events) if fpp_events is not None else None,
        model=model,
        paths={
            "fit_summary": fit_summary_path,
            "model_selection": model_selection_path,
            "chunks": chunks_path,
            "gamma": gamma_path,
            "state_entropy_processed": entropy_path,
            "gamma_index": gamma_index_path,
            "pca_explained_variance": pca_path,
            "training_free_energy": training_free_energy_path,
            "init_free_energy": init_free_energy_path,
            "viterbi": viterbi_path,
            "fpp_events": fpp_events_path,
            "model": model_path,
        },
    )


# ---------------------------------------------------------------------------
# Plotting helpers


def plot_chunk_length_histogram(qc_inputs: QcInputs, output_dir: Path) -> Path:
    """Plot a histogram of chunk durations in seconds."""

    if qc_inputs.sampling_rate_hz is None:
        raise ValueError("Chunk-duration plotting requires `sampling_rate_hz` in fit_summary.")
    durations = compute_chunk_durations(qc_inputs.chunks, qc_inputs.sampling_rate_hz)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(durations["duration_seconds"].to_numpy(dtype=float), bins="auto")
    axis.set_title("Chunk Duration Distribution")
    axis.set_xlabel("Chunk duration (s)")
    axis.set_ylabel("Count")
    minimum_chunk_duration_s = qc_inputs.fit_summary.get("minimum_chunk_duration_s")
    if minimum_chunk_duration_s is not None:
        axis.axvline(float(minimum_chunk_duration_s), linestyle="--", label="Minimum chunk duration")
    causal_history_ms = qc_inputs.fit_summary.get("causal_history_ms")
    if causal_history_ms is not None:
        axis.axvline(float(causal_history_ms) / 1000.0, linestyle=":", label="Causal history window")
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_chunk_length_histogram")
    return _primary_path(output_dir, "qc_chunk_length_histogram")


def plot_pca_explained_variance(qc_inputs: QcInputs, output_dir: Path) -> list[Path]:
    """Plot scree and cumulative explained variance curves if PCA outputs exist."""

    if qc_inputs.pca_explained_variance is None:
        qc_inputs.warnings.append("PCA explained variance was unavailable; skipped PCA QC plots.")
        return []

    frame = qc_inputs.pca_explained_variance.sort_values("component", kind="mergesort").reset_index(drop=True)
    written_paths: list[Path] = []

    scree_figure, scree_axis = plt.subplots(figsize=(8, 5))
    scree_axis.plot(
        frame["component"].to_numpy(dtype=int),
        frame["explained_variance_ratio"].to_numpy(dtype=float),
        marker="o",
    )
    scree_axis.set_title("PCA Scree Plot")
    scree_axis.set_xlabel("Component")
    scree_axis.set_ylabel("Explained variance ratio")
    scree_figure.tight_layout()
    _save_figure(scree_figure, output_dir, "qc_pca_scree")
    written_paths.append(_primary_path(output_dir, "qc_pca_scree"))

    cumulative_figure, cumulative_axis = plt.subplots(figsize=(8, 5))
    cumulative_axis.plot(
        frame["component"].to_numpy(dtype=int),
        np.cumsum(frame["explained_variance_ratio"].to_numpy(dtype=float)),
        marker="o",
    )
    cumulative_axis.set_title("PCA Cumulative Explained Variance")
    cumulative_axis.set_xlabel("Component")
    cumulative_axis.set_ylabel("Cumulative explained variance ratio")
    cumulative_figure.tight_layout()
    _save_figure(cumulative_figure, output_dir, "qc_pca_cumulative_variance")
    written_paths.append(_primary_path(output_dir, "qc_pca_cumulative_variance"))

    return written_paths


def compute_fractional_occupancy(gamma: np.ndarray) -> np.ndarray:
    """Compute global fractional occupancy for each state."""

    posterior = _validate_gamma(gamma)
    return posterior.mean(axis=0)


def plot_state_fractional_occupancy(qc_inputs: QcInputs, output_dir: Path) -> Path:
    """Plot global state fractional occupancy."""

    occupancy = compute_fractional_occupancy(qc_inputs.gamma)
    figure, axis = plt.subplots(figsize=(8, 5))
    state_indices = np.arange(qc_inputs.selected_k, dtype=int)
    axis.bar(state_indices, occupancy)
    axis.set_title("Global State Fractional Occupancy")
    axis.set_xlabel("State")
    axis.set_ylabel("Fractional occupancy")
    axis.set_xticks(state_indices)
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_state_fractional_occupancy")
    return _primary_path(output_dir, "qc_state_fractional_occupancy")


def compute_subject_run_fractional_occupancy(
    gamma: np.ndarray,
    gamma_index: pd.DataFrame,
) -> pd.DataFrame:
    """Compute subject-run fractional occupancy values in long format."""

    posterior = _validate_gamma(gamma)
    _require_columns(gamma_index, REQUIRED_GAMMA_INDEX_COLUMNS, label="subject_run_gamma_index.csv")
    if len(gamma_index) != posterior.shape[0]:
        raise ValueError("Gamma and gamma_index must have the same number of rows.")

    working_index = _coerce_run_sort_columns(gamma_index).reset_index(drop=True)
    occupancy_rows: list[dict[str, Any]] = []
    for (subject, run), run_frame in working_index.groupby(["subject", "run"], sort=False):
        processed_indices = run_frame["processed_index"].to_numpy(dtype=int)
        run_gamma = posterior[processed_indices]
        run_occupancy = compute_fractional_occupancy(run_gamma)
        for state_index, occupancy_value in enumerate(run_occupancy):
            occupancy_rows.append(
                {
                    "subject": str(subject),
                    "run": str(run),
                    "subject_run": f"{subject} | {run}",
                    "state": int(state_index),
                    "fractional_occupancy": float(occupancy_value),
                }
            )
    return pd.DataFrame(occupancy_rows)


def plot_subject_run_state_occupancy_heatmap(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot subject-run fractional occupancy as a heatmap."""

    if qc_inputs.gamma_index is None:
        qc_inputs.warnings.append("Gamma index mapping was unavailable; skipped subject-run occupancy heatmap.")
        return None

    occupancy = compute_subject_run_fractional_occupancy(qc_inputs.gamma, qc_inputs.gamma_index)
    occupancy.to_csv(output_dir / "qc_subject_run_fractional_occupancy.csv", index=False)

    pivot = (
        occupancy.pivot(index="subject_run", columns="state", values="fractional_occupancy")
        .sort_index(kind="mergesort")
        .reindex(columns=np.arange(qc_inputs.selected_k, dtype=int))
    )
    figure, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(pivot.index) + 2)))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    axis.set_title("Subject-Run State Fractional Occupancy")
    axis.set_xlabel("State")
    axis.set_ylabel("Subject | Run")
    axis.set_xticks(np.arange(qc_inputs.selected_k, dtype=int))
    axis.set_yticks(np.arange(len(pivot.index), dtype=int))
    axis.set_yticklabels(pivot.index.tolist())
    figure.colorbar(image, ax=axis, label="Fractional occupancy")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_subject_run_state_occupancy_heatmap")
    return _primary_path(output_dir, "qc_subject_run_state_occupancy_heatmap")


def compute_viterbi_from_gamma(gamma: np.ndarray) -> np.ndarray:
    """Compute a hard state path from posterior probabilities."""

    posterior = _validate_gamma(gamma)
    return np.argmax(posterior, axis=1).astype(int)


def compute_dwell_times(
    viterbi: np.ndarray,
    sampling_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Compute dwell episodes from a hard state path.

    Parameters
    ----------
    viterbi : numpy.ndarray
        Hard state labels with shape ``(n_samples,)``.
    sampling_rate_hz : float | None, optional
        Sampling rate in Hz. When provided, ``dwell_seconds`` is filled.

    Returns
    -------
    pandas.DataFrame
        One row per dwell episode.
    """

    labels = np.asarray(viterbi, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("`viterbi` must be non-empty.")
    if np.any(labels < 0):
        raise ValueError("`viterbi` cannot contain negative labels.")
    rows: list[dict[str, Any]] = []
    start = 0
    current_state = int(labels[0])
    for index in range(1, labels.shape[0] + 1):
        reached_end = index == labels.shape[0]
        state_changed = not reached_end and int(labels[index]) != current_state
        if not reached_end and not state_changed:
            continue
        stop = index
        dwell_samples = stop - start
        row = {
            "state": current_state,
            "start": start,
            "stop": stop,
            "dwell_samples": dwell_samples,
            "dwell_seconds": (
                np.nan if sampling_rate_hz is None else float(dwell_samples) / float(sampling_rate_hz)
            ),
        }
        rows.append(row)
        if not reached_end:
            start = index
            current_state = int(labels[index])
    return pd.DataFrame(rows)


def plot_state_dwell_time_distributions(qc_inputs: QcInputs, output_dir: Path) -> list[Path]:
    """Plot dwell-time histograms for each state."""

    viterbi = qc_inputs.viterbi if qc_inputs.viterbi is not None else compute_viterbi_from_gamma(qc_inputs.gamma)
    dwell_times = compute_dwell_times(viterbi, sampling_rate_hz=qc_inputs.sampling_rate_hz)
    dwell_times.to_csv(output_dir / "qc_dwell_times.csv", index=False)

    output_paths: list[Path] = []
    seconds_column = (
        "dwell_seconds" if qc_inputs.sampling_rate_hz is not None else "dwell_samples"
    )
    x_label = "Dwell time (s)" if qc_inputs.sampling_rate_hz is not None else "Dwell time (samples)"
    for state_index in range(qc_inputs.selected_k):
        state_values = dwell_times.loc[dwell_times["state"] == state_index, seconds_column].to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(state_values[np.isfinite(state_values)], bins="auto")
        axis.set_title(f"State {state_index} Dwell-Time Distribution")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Count")
        figure.tight_layout()
        stem = f"qc_dwell_time_state-{state_index}"
        _save_figure(figure, output_dir, stem)
        output_paths.append(_primary_path(output_dir, stem))
    return output_paths


def plot_transition_matrix(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot the transition matrix from the fitted model or empirical Viterbi counts."""

    transition_matrix = _extract_transition_matrix_from_model(qc_inputs.model, qc_inputs.selected_k)
    if transition_matrix is None:
        viterbi = qc_inputs.viterbi if qc_inputs.viterbi is not None else compute_viterbi_from_gamma(qc_inputs.gamma)
        transition_matrix = compute_empirical_transition_matrix(viterbi, n_states=qc_inputs.selected_k)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(transition_matrix, interpolation="nearest", aspect="auto")
    axis.set_title("State Transition Matrix")
    axis.set_xlabel("To state")
    axis.set_ylabel("From state")
    state_ticks = np.arange(qc_inputs.selected_k, dtype=int)
    axis.set_xticks(state_ticks)
    axis.set_yticks(state_ticks)
    figure.colorbar(image, ax=axis, label="Transition probability")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_transition_matrix")
    return _primary_path(output_dir, "qc_transition_matrix")


def plot_gamma_entropy_examples(
    qc_inputs: QcInputs,
    output_dir: Path,
    n_examples: int = DEFAULT_EXAMPLE_COUNT,
) -> list[Path]:
    """Plot representative entropy traces and, when possible, gamma images."""

    example_keys = _choose_example_runs(qc_inputs.entropy_by_run, n_examples)
    output_paths: list[Path] = []
    for subject, run in example_keys:
        entropy_frame = qc_inputs.entropy_by_run[(subject, run)]
        entropy_figure, entropy_axis = plt.subplots(figsize=(10, 4))
        entropy_axis.plot(
            entropy_frame["time_s"].to_numpy(dtype=float),
            entropy_frame["state_entropy"].to_numpy(dtype=float),
        )
        entropy_axis.set_title(f"State Entropy Example: subject={subject}, run={run}")
        entropy_axis.set_xlabel("Time (s)")
        entropy_axis.set_ylabel("State entropy")
        entropy_figure.tight_layout()
        entropy_stem = f"qc_entropy_example_subject-{subject}_run-{run}"
        _save_figure(entropy_figure, output_dir, entropy_stem)
        output_paths.append(_primary_path(output_dir, entropy_stem))

        if qc_inputs.gamma_index is None:
            continue
        run_gamma_index = qc_inputs.gamma_index.loc[
            (qc_inputs.gamma_index["subject"].astype(str) == subject)
            & (qc_inputs.gamma_index["run"].astype(str) == run)
        ].sort_values("processed_index", kind="mergesort")
        if run_gamma_index.empty:
            continue
        processed_indices = run_gamma_index["processed_index"].to_numpy(dtype=int)
        run_gamma = qc_inputs.gamma[processed_indices].T
        if "time_s" in run_gamma_index.columns:
            x_values = run_gamma_index["time_s"].to_numpy(dtype=float)
        elif qc_inputs.sampling_rate_hz is not None:
            x_values = processed_indices.astype(float) / qc_inputs.sampling_rate_hz
        else:
            x_values = np.arange(processed_indices.shape[0], dtype=float)
        gamma_figure, gamma_axis = plt.subplots(figsize=(10, 4))
        image = gamma_axis.imshow(
            run_gamma,
            aspect="auto",
            interpolation="nearest",
            extent=[float(x_values[0]), float(x_values[-1]), qc_inputs.selected_k - 0.5, -0.5],
        )
        gamma_axis.set_title(f"Gamma Example: subject={subject}, run={run}")
        gamma_axis.set_xlabel("Processed time (s)" if qc_inputs.sampling_rate_hz is not None else "Processed sample")
        gamma_axis.set_ylabel("State")
        gamma_axis.set_yticks(np.arange(qc_inputs.selected_k, dtype=int))
        gamma_figure.colorbar(image, ax=gamma_axis, label="Gamma")
        gamma_figure.tight_layout()
        gamma_stem = f"qc_gamma_example_subject-{subject}_run-{run}"
        _save_figure(gamma_figure, output_dir, gamma_stem)
        output_paths.append(_primary_path(output_dir, gamma_stem))
    return output_paths


def plot_entropy_distribution(qc_inputs: QcInputs, output_dir: Path) -> list[Path]:
    """Plot entropy distributions on the processed and original timelines."""

    written_paths: list[Path] = []
    entropy_values = qc_inputs.state_entropy_processed
    normalized_entropy = _normalize_entropy(entropy_values, qc_inputs.selected_k)

    histogram_figure, histogram_axis = plt.subplots(figsize=(8, 5))
    histogram_axis.hist(entropy_values, bins="auto")
    histogram_axis.set_title("Processed State Entropy Distribution")
    histogram_axis.set_xlabel("State entropy")
    histogram_axis.set_ylabel("Count")
    histogram_figure.tight_layout()
    _save_figure(histogram_figure, output_dir, "qc_state_entropy_histogram")
    written_paths.append(_primary_path(output_dir, "qc_state_entropy_histogram"))

    normalized_figure, normalized_axis = plt.subplots(figsize=(8, 5))
    normalized_axis.hist(normalized_entropy, bins="auto")
    normalized_axis.set_title("Normalized State Entropy Distribution")
    normalized_axis.set_xlabel("Normalized state entropy")
    normalized_axis.set_ylabel("Count")
    normalized_figure.tight_layout()
    _save_figure(normalized_figure, output_dir, "qc_state_entropy_normalized_histogram")
    written_paths.append(_primary_path(output_dir, "qc_state_entropy_normalized_histogram"))

    run_labels: list[str] = []
    run_values: list[np.ndarray] = []
    for (subject, run), frame in sorted(qc_inputs.entropy_by_run.items()):
        values = frame["state_entropy"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        run_labels.append(f"{subject} | {run}")
        run_values.append(values)
    if run_values:
        by_run_figure, by_run_axis = plt.subplots(figsize=(max(8, 0.6 * len(run_labels) + 2), 5))
        _boxplot(by_run_axis, run_values, run_labels)
        by_run_axis.set_title("State Entropy by Subject-Run")
        by_run_axis.set_xlabel("Subject | Run")
        by_run_axis.set_ylabel("State entropy")
        by_run_axis.tick_params(axis="x", rotation=90)
        by_run_figure.tight_layout()
        _save_figure(by_run_figure, output_dir, "qc_state_entropy_by_subject_run")
        written_paths.append(_primary_path(output_dir, "qc_state_entropy_by_subject_run"))
    else:
        qc_inputs.warnings.append("No valid run-level entropy files were available for the by-run boxplot.")

    return written_paths


def plot_entropy_aligned_to_fpp_onset(
    qc_inputs: QcInputs,
    output_dir: Path,
    window_s: tuple[float, float] = DEFAULT_FPP_WINDOW_S,
) -> Path | None:
    """Plot event-locked entropy aligned to FPP onset."""

    if qc_inputs.fpp_events is None:
        qc_inputs.warnings.append("FPP events were unavailable; skipped entropy alignment to FPP onset.")
        return None
    if qc_inputs.sampling_rate_hz is None:
        qc_inputs.warnings.append("Sampling rate was unavailable; skipped entropy alignment to FPP onset.")
        return None

    start_offset = int(round(window_s[0] * qc_inputs.sampling_rate_hz))
    stop_offset = int(round(window_s[1] * qc_inputs.sampling_rate_hz))
    if stop_offset <= start_offset:
        raise ValueError("FPP alignment window must have positive width.")
    offsets = np.arange(start_offset, stop_offset, dtype=int)
    aligned_blocks: list[np.ndarray] = []
    for row in qc_inputs.fpp_events.to_dict("records"):
        key = _run_key(row["subject"], row["run"])
        if key not in qc_inputs.entropy_by_run:
            continue
        entropy_frame = qc_inputs.entropy_by_run[key]
        aligned = _collect_aligned_series(
            entropy_frame,
            np.asarray([int(row["onset_sample"])], dtype=int),
            offsets,
        )
        aligned_blocks.append(aligned)
    if not aligned_blocks:
        qc_inputs.warnings.append("No FPP events could be aligned to available entropy timelines.")
        return None
    aligned_values = np.vstack(aligned_blocks)
    mean_values, sem_values = _mean_and_sem(aligned_values)
    time_axis_s = offsets.astype(float) / qc_inputs.sampling_rate_hz

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(time_axis_s, mean_values, label="Mean entropy")
    axis.fill_between(time_axis_s, mean_values - sem_values, mean_values + sem_values, alpha=0.3, label="SEM")
    axis.set_title("Entropy Aligned to FPP Onset")
    axis.set_xlabel("Time relative to FPP onset (s)")
    axis.set_ylabel("State entropy")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_entropy_aligned_to_fpp_onset")
    return _primary_path(output_dir, "qc_entropy_aligned_to_fpp_onset")


def plot_entropy_vs_time_to_event(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot binned mean entropy against time to the next FPP event."""

    if qc_inputs.fpp_events is None:
        qc_inputs.warnings.append("FPP events were unavailable; skipped entropy versus time-to-event plot.")
        return None
    if qc_inputs.sampling_rate_hz is None:
        qc_inputs.warnings.append("Sampling rate was unavailable; skipped entropy versus time-to-event plot.")
        return None

    paired_rows: list[tuple[float, float]] = []
    for key, entropy_frame in qc_inputs.entropy_by_run.items():
        run_events = qc_inputs.fpp_events.loc[
            (qc_inputs.fpp_events["subject"].astype(str) == key[0])
            & (qc_inputs.fpp_events["run"].astype(str) == key[1])
        ].sort_values("onset_sample", kind="mergesort")
        if run_events.empty:
            continue
        entropy_values = entropy_frame["state_entropy"].to_numpy(dtype=float)
        next_event_samples = run_events["onset_sample"].to_numpy(dtype=int)
        previous_start = 0
        for event_sample in next_event_samples:
            start_sample = max(previous_start, event_sample - int(round(DEFAULT_TIME_TO_EVENT_HORIZON_S * qc_inputs.sampling_rate_hz)))
            sample_indices = np.arange(start_sample, event_sample, dtype=int)
            valid_mask = np.isfinite(entropy_values[sample_indices])
            if np.any(valid_mask):
                time_to_event_s = (event_sample - sample_indices[valid_mask]).astype(float) / qc_inputs.sampling_rate_hz
                entropy_segment = entropy_values[sample_indices[valid_mask]]
                paired_rows.extend(zip(time_to_event_s, entropy_segment, strict=False))
            previous_start = event_sample + 1
    if not paired_rows:
        qc_inputs.warnings.append("No valid at-risk entropy samples were available for the time-to-event plot.")
        return None

    paired = pd.DataFrame(paired_rows, columns=["time_to_event_s", "state_entropy"])
    max_time_s = float(min(DEFAULT_TIME_TO_EVENT_HORIZON_S, paired["time_to_event_s"].max()))
    bin_edges = np.arange(0.0, max_time_s + DEFAULT_TIME_TO_EVENT_BIN_WIDTH_S, DEFAULT_TIME_TO_EVENT_BIN_WIDTH_S)
    paired["time_bin"] = pd.cut(paired["time_to_event_s"], bins=bin_edges, include_lowest=True, right=False)
    binned = paired.groupby("time_bin", observed=True)["state_entropy"].mean().reset_index()
    if binned.empty:
        qc_inputs.warnings.append("Time-to-event binning produced no data.")
        return None
    centers = np.asarray(
        [
            interval.left + (interval.right - interval.left) / 2.0
            for interval in binned["time_bin"].to_list()
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(centers, binned["state_entropy"].to_numpy(dtype=float), marker="o")
    axis.set_title("Entropy vs. Time to Next FPP")
    axis.set_xlabel("Time to next FPP (s)")
    axis.set_ylabel("Mean state entropy")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_entropy_vs_time_to_event")
    return _primary_path(output_dir, "qc_entropy_vs_time_to_event")


def plot_entropy_aligned_to_chunk_start(
    qc_inputs: QcInputs,
    output_dir: Path,
    window_s: tuple[float, float] = DEFAULT_CHUNK_START_WINDOW_S,
) -> Path | None:
    """Plot entropy aligned to chunk starts."""

    if qc_inputs.sampling_rate_hz is None:
        qc_inputs.warnings.append("Sampling rate was unavailable; skipped entropy alignment to chunk start.")
        return None
    offsets = np.arange(
        int(round(window_s[0] * qc_inputs.sampling_rate_hz)),
        int(round(window_s[1] * qc_inputs.sampling_rate_hz)),
        dtype=int,
    )
    if offsets.size == 0:
        raise ValueError("Chunk-start window produced zero samples.")

    aligned_blocks: list[np.ndarray] = []
    for row in qc_inputs.chunks.to_dict("records"):
        key = _run_key(row["subject"], row["run"])
        if key not in qc_inputs.entropy_by_run:
            continue
        entropy_frame = qc_inputs.entropy_by_run[key]
        anchor = int(row["original_start_sample"])
        aligned_blocks.append(
            _collect_aligned_series(
                entropy_frame,
                np.asarray([anchor], dtype=int),
                offsets,
            )
        )
    if not aligned_blocks:
        qc_inputs.warnings.append("Chunk-start alignment was not possible from the available entropy files.")
        return None
    aligned_values = np.vstack(aligned_blocks)
    mean_values, sem_values = _mean_and_sem(aligned_values)
    time_axis_s = offsets.astype(float) / qc_inputs.sampling_rate_hz

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(time_axis_s, mean_values, label="Mean entropy")
    axis.fill_between(time_axis_s, mean_values - sem_values, mean_values + sem_values, alpha=0.3, label="SEM")
    axis.set_title("Entropy Aligned to Chunk Start")
    axis.set_xlabel("Time from chunk start (s)")
    axis.set_ylabel("State entropy")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_entropy_aligned_to_chunk_start")
    return _primary_path(output_dir, "qc_entropy_aligned_to_chunk_start")


def plot_free_energy_training_curve(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot free energy across training cycles."""

    if qc_inputs.training_free_energy is None:
        qc_inputs.warnings.append("Training free-energy trace was unavailable; skipped training curve.")
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        qc_inputs.training_free_energy["cycle"].to_numpy(dtype=float),
        qc_inputs.training_free_energy["free_energy"].to_numpy(dtype=float),
        marker="o",
    )
    axis.set_title(f"Training Free Energy (K={qc_inputs.selected_k})")
    axis.set_xlabel("Cycle")
    axis.set_ylabel("Free energy")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_free_energy_training_curve")
    return _primary_path(output_dir, "qc_free_energy_training_curve")


def plot_free_energy_initializations(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot free energy across random initializations."""

    if qc_inputs.init_free_energy is None:
        qc_inputs.warnings.append("Initialization free-energy trace was unavailable; skipped init plot.")
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        qc_inputs.init_free_energy["initialization"].to_numpy(dtype=float),
        qc_inputs.init_free_energy["free_energy"].to_numpy(dtype=float),
        marker="o",
        linestyle="None",
    )
    axis.set_title(f"Initialization Free Energy (K={qc_inputs.selected_k})")
    axis.set_xlabel("Initialization")
    axis.set_ylabel("Free energy")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_free_energy_initializations")
    return _primary_path(output_dir, "qc_free_energy_initializations")


def plot_free_energy_vs_k(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot best free energy against K."""

    model_selection = qc_inputs.model_selection.sort_values("k", kind="mergesort").reset_index(drop=True)
    if model_selection.empty:
        qc_inputs.warnings.append("Model selection table was empty; skipped free energy versus K plot.")
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(model_selection["k"].to_numpy(dtype=int), model_selection["free_energy"].to_numpy(dtype=float), marker="o")
    selected_rows = model_selection.loc[model_selection["k"].astype(int) == qc_inputs.selected_k]
    if not selected_rows.empty:
        selected_row = selected_rows.iloc[0]
        axis.annotate(
            f"selected K={qc_inputs.selected_k}",
            xy=(float(selected_row["k"]), float(selected_row["free_energy"])),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_title("Free Energy vs. Number of States")
    axis.set_xlabel("K")
    axis.set_ylabel("Free energy")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_free_energy_vs_k")
    return _primary_path(output_dir, "qc_free_energy_vs_k")


def plot_delta_free_energy_vs_k(qc_inputs: QcInputs, output_dir: Path) -> Path | None:
    """Plot free-energy deltas across consecutive K values."""

    delta_frame = compute_delta_free_energy(qc_inputs.model_selection)
    if delta_frame.shape[0] < 2:
        qc_inputs.warnings.append("Need at least two K values to plot delta free energy.")
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        delta_frame["k"].to_numpy(dtype=int)[1:],
        delta_frame["delta_free_energy"].to_numpy(dtype=float)[1:],
        marker="o",
    )
    axis.set_title("Delta Free Energy vs. Number of States")
    axis.set_xlabel("K")
    axis.set_ylabel("Delta free energy from previous K")
    figure.tight_layout()
    _save_figure(figure, output_dir, "qc_delta_free_energy_vs_k")
    return _primary_path(output_dir, "qc_delta_free_energy_vs_k")


# ---------------------------------------------------------------------------
# Reporting


def write_qc_summary_report(qc_inputs: QcInputs, generated_paths: list[Path], output_dir: Path) -> Path:
    """Write a markdown QC summary report."""

    occupancy = compute_fractional_occupancy(qc_inputs.gamma)
    entropy_values = qc_inputs.state_entropy_processed
    normalized_entropy = _normalize_entropy(entropy_values, qc_inputs.selected_k)
    report_lines = [
        "# TDE-HMM QC Report",
        "",
        f"- Selected K: `{qc_inputs.selected_k}`",
        f"- Total chunks: `{len(qc_inputs.chunks)}`",
        f"- Total processed samples: `{qc_inputs.gamma.shape[0]}`",
        "",
        "## Fractional Occupancy",
        "",
    ]
    for state_index, occupancy_value in enumerate(occupancy):
        report_lines.append(f"- State {state_index}: `{occupancy_value:.6f}`")
    report_lines.extend(
        [
            "",
            "## Entropy Summary",
            "",
            f"- Mean entropy: `{float(np.mean(entropy_values)):.6f}`",
            f"- Std entropy: `{float(np.std(entropy_values)):.6f}`",
            f"- Mean normalized entropy: `{float(np.mean(normalized_entropy)):.6f}`",
            f"- Std normalized entropy: `{float(np.std(normalized_entropy)):.6f}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if qc_inputs.warnings:
        report_lines.extend(f"- {warning}" for warning in qc_inputs.warnings)
    else:
        report_lines.append("- None")
    report_lines.extend(
        [
            "",
            "## Generated Figures",
            "",
        ]
    )
    report_lines.extend(f"- `{path}`" for path in generated_paths)
    report_path = output_dir / "qc_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# High-level runners


def _run_all_qc_plots_from_inputs(
    qc_inputs: QcInputs,
    qc_output_dir: Path,
    *,
    n_examples: int = DEFAULT_EXAMPLE_COUNT,
) -> dict[str, Path | list[Path] | None]:
    qc_output_dir.mkdir(parents=True, exist_ok=True)
    generated_artifacts: dict[str, Path | list[Path] | None] = {}

    generated_artifacts["chunk_length_histogram"] = plot_chunk_length_histogram(qc_inputs, qc_output_dir)
    generated_artifacts["pca_explained_variance"] = plot_pca_explained_variance(qc_inputs, qc_output_dir)
    generated_artifacts["state_fractional_occupancy"] = plot_state_fractional_occupancy(qc_inputs, qc_output_dir)
    generated_artifacts["subject_run_state_occupancy_heatmap"] = plot_subject_run_state_occupancy_heatmap(
        qc_inputs,
        qc_output_dir,
    )
    generated_artifacts["dwell_time_distributions"] = plot_state_dwell_time_distributions(qc_inputs, qc_output_dir)
    generated_artifacts["transition_matrix"] = plot_transition_matrix(qc_inputs, qc_output_dir)
    generated_artifacts["gamma_entropy_examples"] = plot_gamma_entropy_examples(
        qc_inputs,
        qc_output_dir,
        n_examples=n_examples,
    )
    generated_artifacts["entropy_distribution"] = plot_entropy_distribution(qc_inputs, qc_output_dir)
    generated_artifacts["entropy_aligned_to_fpp_onset"] = plot_entropy_aligned_to_fpp_onset(qc_inputs, qc_output_dir)
    generated_artifacts["entropy_vs_time_to_event"] = plot_entropy_vs_time_to_event(qc_inputs, qc_output_dir)
    generated_artifacts["entropy_aligned_to_chunk_start"] = plot_entropy_aligned_to_chunk_start(
        qc_inputs,
        qc_output_dir,
    )
    generated_artifacts["free_energy_training_curve"] = plot_free_energy_training_curve(qc_inputs, qc_output_dir)
    generated_artifacts["free_energy_initializations"] = plot_free_energy_initializations(qc_inputs, qc_output_dir)
    generated_artifacts["free_energy_vs_k"] = plot_free_energy_vs_k(qc_inputs, qc_output_dir)
    generated_artifacts["delta_free_energy_vs_k"] = plot_delta_free_energy_vs_k(qc_inputs, qc_output_dir)

    generated_paths: list[Path] = []
    for artifact in generated_artifacts.values():
        if artifact is None:
            continue
        if isinstance(artifact, list):
            generated_paths.extend(path for path in artifact if path is not None)
        else:
            generated_paths.append(artifact)
    generated_artifacts["report"] = write_qc_summary_report(qc_inputs, generated_paths, qc_output_dir)
    return generated_artifacts


def run_all_qc_plots(output_dir: Path, selected_k: int | None = None) -> dict[str, Path | list[Path] | None]:
    """Load QC inputs, create the ``qc`` directory, and write all applicable QC outputs."""

    qc_inputs = load_qc_inputs(output_dir=output_dir, selected_k=selected_k)
    qc_output_dir = Path(output_dir).resolve() / "qc"
    return _run_all_qc_plots_from_inputs(qc_inputs, qc_output_dir)


__all__ = [
    "QcInputs",
    "compute_chunk_durations",
    "compute_delta_free_energy",
    "compute_empirical_transition_matrix",
    "compute_dwell_times",
    "compute_fractional_occupancy",
    "compute_normalized_state_entropy",
    "compute_subject_run_fractional_occupancy",
    "compute_viterbi_from_gamma",
    "load_qc_inputs",
    "plot_chunk_length_histogram",
    "plot_delta_free_energy_vs_k",
    "plot_entropy_aligned_to_chunk_start",
    "plot_entropy_aligned_to_fpp_onset",
    "plot_entropy_distribution",
    "plot_entropy_vs_time_to_event",
    "plot_free_energy_initializations",
    "plot_free_energy_training_curve",
    "plot_free_energy_vs_k",
    "plot_gamma_entropy_examples",
    "plot_pca_explained_variance",
    "plot_state_dwell_time_distributions",
    "plot_state_fractional_occupancy",
    "plot_subject_run_state_occupancy_heatmap",
    "plot_transition_matrix",
    "run_all_qc_plots",
    "write_qc_summary_report",
]
