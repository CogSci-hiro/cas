"""Causal TDE-HMM feature extraction for source-level EEG.

This module fits a group-level causal time-delay embedded HMM with the
``glhmm`` package and exports samplewise state entropy as the default neural
feature for downstream hazard modeling.

Examples
--------
>>> from pathlib import Path
>>> from cas.hmm.tde_hmm import TdeHmmConfig, fit_tde_hmm_pipeline
>>> config = TdeHmmConfig(candidate_k=(4, 5), pca_n_components=16, verbose=False)
>>> result = fit_tde_hmm_pipeline(
...     manifest_path=Path("runs.csv"),
...     output_dir=Path("out/hmm"),
...     config=config,
... )
>>> result.selected_k in {4, 5}
True
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from glhmm import preproc
from glhmm.glhmm import glhmm
from scipy import signal
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants

DEFAULT_SUBJECT_COLUMN = "subject"
DEFAULT_RUN_COLUMN = "run"
DEFAULT_SOURCE_ARRAY_COLUMN = "source_path"
DEFAULT_SPEECH_INTERVALS_COLUMN = "speech_path"
DEFAULT_ONSET_COLUMN = "onset_s"
DEFAULT_OFFSET_COLUMN = "offset_s"
DEFAULT_INPUT_SAMPLING_RATE_COLUMN = "input_sampling_rate_hz"
DEFAULT_EPSILON = 1e-12
DEFAULT_EMPTY_STATE_THRESHOLD = 1e-3
DEFAULT_MODEL_SELECTION_TOLERANCE = 1e-6
SECONDS_PER_MILLISECOND = 1e-3
MINIMUM_LAG_COUNT = 1
JSON_INDENT = 2

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses


@dataclass(frozen=True, slots=True)
class TdeHmmConfig:
    """Configuration for causal TDE-HMM fitting.

    Parameters
    ----------
    sampling_rate_hz : float, default=128.0
        Target sampling rate after optional downsampling.
    causal_history_ms : float, default=100.0
        Width of the causal embedding window in milliseconds.
    speech_guard_pre_s : float, default=0.2
        Guard interval added before each speaking interval.
    speech_guard_post_s : float, default=0.2
        Guard interval added after each speaking interval.
    minimum_chunk_duration_s : float, default=0.5
        Minimum contiguous non-speaking duration retained for fitting.
    candidate_k : tuple[int, ...], default=(4, 5, 6, 7)
        Candidate state counts considered during model selection.
    n_initializations : int, default=5
        Number of GLHMM random initializations.
    standardize_per_run : bool, default=True
        Whether to z-score each feature within each run before masking.
    pca_n_components : int | None, default=None
        Optional PCA dimensionality reduction applied after embedding.
    random_seed : int, default=123
        Base random seed for reproducible fitting.
    covariance_type : str, default="full"
        GLHMM covariance type passed through to the package.
    verbose : bool, default=True
        Whether to emit informative log messages.
    """

    sampling_rate_hz: float = 128.0
    causal_history_ms: float = 100.0
    speech_guard_pre_s: float = 0.2
    speech_guard_post_s: float = 0.2
    minimum_chunk_duration_s: float = 0.5
    candidate_k: tuple[int, ...] = (4, 5, 6, 7)
    n_initializations: int = 5
    standardize_per_run: bool = True
    pca_n_components: int | None = None
    random_seed: int = 123
    covariance_type: str = "full"
    verbose: bool = True

    def validate(self) -> None:
        """Validate configuration values."""

        if self.sampling_rate_hz <= 0:
            raise ValueError("`sampling_rate_hz` must be positive.")
        if self.causal_history_ms < 0:
            raise ValueError("`causal_history_ms` must be non-negative.")
        if self.speech_guard_pre_s < 0 or self.speech_guard_post_s < 0:
            raise ValueError("Speech guard durations must be non-negative.")
        if self.minimum_chunk_duration_s <= 0:
            raise ValueError("`minimum_chunk_duration_s` must be positive.")
        if not self.candidate_k:
            raise ValueError("`candidate_k` must contain at least one candidate state count.")
        if any(k <= 0 for k in self.candidate_k):
            raise ValueError("All candidate K values must be positive integers.")
        if self.n_initializations <= 0:
            raise ValueError("`n_initializations` must be positive.")
        if self.pca_n_components is not None and self.pca_n_components <= 0:
            raise ValueError("`pca_n_components` must be positive when provided.")


@dataclass(frozen=True, slots=True)
class TdeHmmPipelineResult:
    """Summary returned by :func:`fit_tde_hmm_pipeline`."""

    output_dir: Path
    selected_k: int
    model_path: Path
    model_selection_path: Path
    gamma_path: Path
    entropy_processed_path: Path
    chunk_table_path: Path
    fit_summary_path: Path
    entropy_csv_paths: tuple[Path, ...]
    total_chunks: int
    total_samples_kept: int
    total_processed_samples: int


# ---------------------------------------------------------------------------
# Validation helpers


def _validate_2d_array(array: np.ndarray, *, label: str) -> np.ndarray:
    """Validate and coerce a finite 2D float array."""

    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one sample.")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return values


def _validate_indices(indices: np.ndarray, *, n_samples: int) -> np.ndarray:
    """Validate session indices in Python slice convention."""

    values = np.asarray(indices, dtype=int)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("`indices` must have shape (n_segments, 2).")
    if values.shape[0] == 0:
        raise ValueError("`indices` must contain at least one segment.")
    previous_stop = 0
    for start, stop in values:
        if start < 0 or stop < 0:
            raise ValueError("Indices must be non-negative.")
        if stop <= start:
            raise ValueError(f"Invalid segment [{start}, {stop}); stop must exceed start.")
        if stop > n_samples:
            raise ValueError(f"Index stop {stop} exceeds data length {n_samples}.")
        if start < previous_stop:
            raise ValueError("Indices must be sorted and non-overlapping.")
        previous_stop = stop
    return values


def _safe_standardize(values: np.ndarray) -> np.ndarray:
    """Z-score columns with zero-variance protection."""

    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


def _resolve_logging(verbose: bool) -> None:
    """Configure module-level logging once."""

    if LOGGER.handlers:
        LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
        return

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _progress(
    iterable,
    *,
    total: int | None = None,
    desc: str,
    enabled: bool,
):
    """Wrap an iterable in a tqdm progress bar when enabled."""

    if not enabled:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
    )


# ---------------------------------------------------------------------------
# Public helpers


def compute_causal_lags(sampling_rate_hz: float, causal_history_ms: float) -> np.ndarray:
    """Compute contiguous causal lags ending at the current sample.

    Parameters
    ----------
    sampling_rate_hz : float
        Sampling rate in Hz.
    causal_history_ms : float
        Causal history window in milliseconds.

    Returns
    -------
    numpy.ndarray
        Integer lags such as ``[-12, -11, ..., -1, 0]``.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("`sampling_rate_hz` must be positive.")
    if causal_history_ms < 0:
        raise ValueError("`causal_history_ms` must be non-negative.")

    history_samples = int(
        math.floor(causal_history_ms * SECONDS_PER_MILLISECOND * sampling_rate_hz + DEFAULT_EPSILON)
    )
    lags = np.arange(-history_samples, 1, dtype=int)
    if lags.size < MINIMUM_LAG_COUNT:
        raise ValueError("At least one lag must be included.")
    if np.any(lags > 0):
        raise ValueError("Causal lags must all be <= 0.")
    if 0 not in lags:
        raise ValueError("Causal lags must include 0.")
    if not np.array_equal(lags, np.arange(lags[0], lags[-1] + 1, dtype=int)):
        raise ValueError("Causal lags must be contiguous.")
    return lags


def build_non_speaking_mask(
    n_samples: int,
    *,
    sample_times_s: np.ndarray | None = None,
    sampling_rate_hz: float | None = None,
    speech_intervals_s: np.ndarray | list[list[float]] | list[tuple[float, float]],
    guard_pre_s: float,
    guard_post_s: float,
) -> np.ndarray:
    """Build a boolean mask that keeps only non-speaking samples.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the run.
    sample_times_s : numpy.ndarray | None, optional
        Optional per-sample time stamps.
    sampling_rate_hz : float | None, optional
        Sampling rate used when ``sample_times_s`` is not supplied.
    speech_intervals_s : array-like of shape (n_intervals, 2)
        Speaking intervals in seconds.
    guard_pre_s : float
        Pre-onset exclusion guard in seconds.
    guard_post_s : float
        Post-offset exclusion guard in seconds.

    Returns
    -------
    numpy.ndarray
        Boolean mask where ``True`` means the sample is retained.
    """

    if n_samples <= 0:
        raise ValueError("`n_samples` must be positive.")
    if guard_pre_s < 0 or guard_post_s < 0:
        raise ValueError("Guard intervals must be non-negative.")
    if sample_times_s is None and sampling_rate_hz is None:
        raise ValueError("Provide either `sample_times_s` or `sampling_rate_hz`.")

    keep_mask = np.ones(n_samples, dtype=bool)
    intervals = np.asarray(speech_intervals_s, dtype=float)
    if intervals.size == 0:
        return keep_mask
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("`speech_intervals_s` must have shape (n_intervals, 2).")

    if sample_times_s is not None:
        sample_times = np.asarray(sample_times_s, dtype=float)
        if sample_times.shape != (n_samples,):
            raise ValueError("`sample_times_s` must have shape (n_samples,).")
        for onset_s, offset_s in intervals:
            if not np.isfinite([onset_s, offset_s]).all():
                raise ValueError("Speech intervals must be finite.")
            if offset_s < onset_s:
                raise ValueError(
                    f"Invalid speech interval [{onset_s}, {offset_s}]; offset must be >= onset."
                )
            start_s = onset_s - guard_pre_s
            stop_s = offset_s + guard_post_s
            keep_mask[(sample_times >= start_s) & (sample_times < stop_s)] = False
        return keep_mask

    assert sampling_rate_hz is not None
    if sampling_rate_hz <= 0:
        raise ValueError("`sampling_rate_hz` must be positive.")

    for onset_s, offset_s in intervals:
        if not np.isfinite([onset_s, offset_s]).all():
            raise ValueError("Speech intervals must be finite.")
        if offset_s < onset_s:
            raise ValueError(
                f"Invalid speech interval [{onset_s}, {offset_s}]; offset must be >= onset."
            )
        start_index = max(
            0,
            int(math.ceil((onset_s - guard_pre_s) * sampling_rate_hz - DEFAULT_EPSILON)),
        )
        stop_index = min(
            n_samples,
            int(math.ceil((offset_s + guard_post_s) * sampling_rate_hz - DEFAULT_EPSILON)),
        )
        keep_mask[start_index:stop_index] = False
    return keep_mask


def split_valid_samples_into_chunks(
    keep_mask: np.ndarray,
    minimum_chunk_samples: int,
) -> list[tuple[int, int]]:
    """Split a keep-mask into contiguous chunks.

    Parameters
    ----------
    keep_mask : numpy.ndarray
        Boolean array where ``True`` indicates a retained sample.
    minimum_chunk_samples : int
        Minimum chunk length to retain.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive-exclusive chunks in Python slice convention.
    """

    values = np.asarray(keep_mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("`keep_mask` must be one-dimensional.")
    if minimum_chunk_samples <= 0:
        raise ValueError("`minimum_chunk_samples` must be positive.")

    chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    for sample_index, keep_sample in enumerate(values):
        if keep_sample and chunk_start is None:
            chunk_start = sample_index
        if not keep_sample and chunk_start is not None:
            if sample_index - chunk_start >= minimum_chunk_samples:
                chunks.append((chunk_start, sample_index))
            chunk_start = None
    if chunk_start is not None and len(values) - chunk_start >= minimum_chunk_samples:
        chunks.append((chunk_start, len(values)))
    return chunks


def concatenate_chunks_and_build_indices(
    source_data: np.ndarray,
    chunks: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Concatenate chunks and build GLHMM-style segment indices.

    Parameters
    ----------
    source_data : numpy.ndarray
        Run data with shape ``(n_samples, n_features)``.
    chunks : list[tuple[int, int]]
        Chunk list in original sample coordinates.

    Returns
    -------
    tuple
        ``(concatenated_data, indices, mapping_table)``.
    """

    data = _validate_2d_array(source_data, label="source_data")
    if not chunks:
        raise ValueError("At least one chunk is required for concatenation.")

    concatenated_segments: list[np.ndarray] = []
    indices_rows: list[tuple[int, int]] = []
    mapping_rows: list[dict[str, Any]] = []
    concat_start = 0
    for chunk_id, (start_sample, stop_sample) in enumerate(chunks):
        if start_sample < 0 or stop_sample > data.shape[0] or stop_sample <= start_sample:
            raise ValueError(
                f"Invalid chunk [{start_sample}, {stop_sample}) for data length {data.shape[0]}."
            )
        chunk_data = data[start_sample:stop_sample]
        chunk_length = stop_sample - start_sample
        concat_stop = concat_start + chunk_length
        concatenated_segments.append(chunk_data)
        indices_rows.append((concat_start, concat_stop))
        mapping_rows.extend(
            {
                "concat_index": concat_start + offset,
                "original_sample_index": start_sample + offset,
                "chunk_id": chunk_id,
            }
            for offset in range(chunk_length)
        )
        concat_start = concat_stop

    concatenated_data = np.vstack(concatenated_segments)
    indices = np.asarray(indices_rows, dtype=int)
    mapping_table = pd.DataFrame(mapping_rows)
    return concatenated_data, indices, mapping_table


def standardize_data_per_run(
    source_data: np.ndarray,
    *,
    run_labels: np.ndarray | None = None,
    run_boundaries: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Z-score features within each run.

    Parameters
    ----------
    source_data : numpy.ndarray
        Data matrix of shape ``(n_samples, n_features)``.
    run_labels : numpy.ndarray | None, optional
        Run label per sample.
    run_boundaries : list[tuple[int, int]] | None, optional
        Inclusive-exclusive boundaries for each run.

    Returns
    -------
    numpy.ndarray
        Standardized data with the same shape as the input.
    """

    data = _validate_2d_array(source_data, label="source_data")
    if (run_labels is None) == (run_boundaries is None):
        raise ValueError("Provide exactly one of `run_labels` or `run_boundaries`.")

    standardized = np.empty_like(data)
    if run_labels is not None:
        labels = np.asarray(run_labels)
        if labels.shape != (data.shape[0],):
            raise ValueError("`run_labels` must have one label per sample.")
        for label in pd.unique(labels):
            mask = labels == label
            standardized[mask] = _safe_standardize(data[mask])
        return standardized

    assert run_boundaries is not None
    current_stop = 0
    for start, stop in run_boundaries:
        if start != current_stop and start < current_stop:
            raise ValueError("`run_boundaries` must be sorted and non-overlapping.")
        if stop <= start:
            raise ValueError(f"Invalid run boundary [{start}, {stop}).")
        if stop > data.shape[0]:
            raise ValueError("Run boundary exceeds available samples.")
        standardized[start:stop] = _safe_standardize(data[start:stop])
        current_stop = stop
    return standardized


def preprocess_data_for_glhmm(
    concatenated_data: np.ndarray,
    indices: np.ndarray,
    config: TdeHmmConfig,
    *,
    sampling_rate_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Preprocess concatenated data for GLHMM.

    Notes
    -----
    The installed ``glhmm.preproc.build_data_tde`` helper does not support
    one-sided lag windows ending at lag ``0``. For the required causal TDE
    setup we therefore implement the embedding locally while still using the
    installed GLHMM package for model fitting and optional PCA.
    """

    config.validate()
    data = _validate_2d_array(concatenated_data, label="concatenated_data")
    validated_indices = _validate_indices(indices, n_samples=data.shape[0])
    effective_sampling_rate_hz = float(config.sampling_rate_hz if sampling_rate_hz is None else sampling_rate_hz)
    lags = compute_causal_lags(effective_sampling_rate_hz, config.causal_history_ms)

    if config.pca_n_components is not None:
        processed_data, processed_indices, processed_mapping = _build_causal_tde_data_with_incremental_pca(
            data,
            validated_indices,
            lags,
            pca_n_components=config.pca_n_components,
        )
    else:
        processed_data, processed_indices, processed_mapping = _build_causal_tde_data(
            data,
            validated_indices,
            lags,
        )
    if processed_data.size == 0:
        raise ValueError("Causal embedding produced no valid samples. Increase chunk duration or reduce history.")

    if config.pca_n_components is None:
        processed_data = _safe_standardize(processed_data)
    preprocessing_log: dict[str, Any] = {
        "sampling_rate_hz": effective_sampling_rate_hz,
        "causal_history_ms": config.causal_history_ms,
        "lags": lags.tolist(),
        "n_input_samples": int(data.shape[0]),
        "n_processed_samples_before_pca": int(processed_data.shape[0]),
        "used_glhmm_build_data_tde": False,
        "embedding_note": (
            "Implemented causal TDE locally because glhmm.preproc.build_data_tde "
            "fails when the maximum lag is 0."
        ),
    }

    if config.pca_n_components is not None:
        preprocessing_log["pca_n_components"] = int(processed_data.shape[1])
        preprocessing_log["pca_backend"] = "incremental"
    else:
        preprocessing_log["pca_n_components"] = None

    preprocessing_log["n_processed_samples"] = int(processed_data.shape[0])
    return processed_data, processed_indices, preprocessing_log, processed_mapping


def fit_one_glhmm_model(
    processed_data: np.ndarray,
    processed_indices: np.ndarray,
    k: int,
    config: TdeHmmConfig,
) -> tuple[glhmm, np.ndarray, np.ndarray | None, float]:
    """Fit one GLHMM model for a specified number of states."""

    if k <= 0:
        raise ValueError("`k` must be positive.")

    data = _validate_2d_array(processed_data, label="processed_data")
    indices = _validate_indices(processed_indices, n_samples=data.shape[0])
    best_result: tuple[glhmm, np.ndarray, np.ndarray | None, float] | None = None
    initialization_iterator = _progress(
        range(config.n_initializations),
        total=config.n_initializations,
        desc=f"K={k} initializations",
        enabled=config.verbose,
    )
    for initialization_index in initialization_iterator:
        np.random.seed(config.random_seed + initialization_index)
        start_time_s = time.perf_counter()
        LOGGER.info(
            "Starting K=%d initialization %d/%d",
            k,
            initialization_index + 1,
            config.n_initializations,
        )
        hmm = glhmm(
            K=k,
            covtype=config.covariance_type,
            model_mean="state",
            model_beta="no",
        )
        options = {
            "cyc": 100,
            "cyc_to_go_under_th": 10,
            "initrep": 0,
            "verbose": config.verbose,
        }
        gamma, xi, free_energy_history = hmm.train(X=None, Y=data, indices=indices, options=options)
        if gamma.size == 0:
            gamma, xi, _ = hmm.decode(X=None, Y=data, indices=indices)
        free_energy = _extract_final_free_energy(free_energy_history)
        elapsed_s = time.perf_counter() - start_time_s
        LOGGER.info(
            "Completed K=%d initialization %d/%d in %.1fs (free energy %.6f)",
            k,
            initialization_index + 1,
            config.n_initializations,
            elapsed_s,
            free_energy,
        )
        current_result = (hmm, gamma, xi if xi.size > 0 else None, free_energy)
        if best_result is None or free_energy < best_result[3]:
            best_result = current_result
    if best_result is None:
        raise RuntimeError(f"No successful GLHMM fits were produced for K={k}.")
    return best_result


def compute_state_entropy(gamma: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Compute samplewise Shannon entropy from posterior state probabilities."""

    posterior = np.asarray(gamma, dtype=float)
    if posterior.ndim != 2:
        raise ValueError("`gamma` must be a 2D array of shape (n_samples, n_states).")
    if posterior.shape[1] == 0:
        raise ValueError("`gamma` must contain at least one state.")
    if np.any(posterior < 0):
        raise ValueError("`gamma` cannot contain negative probabilities.")
    stabilized = np.clip(posterior, epsilon, 1.0)
    return -np.sum(stabilized * np.log(stabilized), axis=1)


def map_processed_feature_to_original_timeline(
    processed_feature: np.ndarray,
    processed_to_concat_mapping: pd.DataFrame,
    concat_to_original_mapping: pd.DataFrame,
    original_n_samples: int,
) -> np.ndarray:
    """Map a processed samplewise feature back to original sample coordinates."""

    feature = np.asarray(processed_feature, dtype=float)
    if feature.ndim != 1:
        raise ValueError("`processed_feature` must be one-dimensional.")
    if original_n_samples <= 0:
        raise ValueError("`original_n_samples` must be positive.")

    processed_mapping = processed_to_concat_mapping.reset_index(drop=True)
    concat_mapping = concat_to_original_mapping.reset_index(drop=True)
    if len(processed_mapping) != feature.shape[0]:
        raise ValueError("Processed mapping length must match the feature length.")

    merged = processed_mapping.merge(
        concat_mapping[["concat_index", "original_sample_index"]],
        on="concat_index",
        how="left",
        validate="one_to_one",
    )
    if merged["original_sample_index"].isna().any():
        raise ValueError("Processed-to-concat mapping contains indices not present in concat mapping.")

    timeline = np.full(original_n_samples, np.nan, dtype=float)
    original_indices = merged["original_sample_index"].to_numpy(dtype=int)
    timeline[original_indices] = feature
    return timeline


def evaluate_candidate_k_values(
    processed_data: np.ndarray,
    processed_indices: np.ndarray,
    config: TdeHmmConfig,
) -> tuple[pd.DataFrame, int, dict[int, tuple[glhmm, np.ndarray, np.ndarray | None, float]]]:
    """Fit and evaluate candidate state counts."""

    rows: list[dict[str, Any]] = []
    successful_rows: list[dict[str, Any]] = []
    fit_cache: dict[int, tuple[glhmm, np.ndarray, np.ndarray | None, float]] = {}

    for k in _progress(
        config.candidate_k,
        total=len(config.candidate_k),
        desc="Model selection",
        enabled=config.verbose,
    ):
        try:
            fit_result = fit_one_glhmm_model(processed_data, processed_indices, k, config)
            _, gamma, _, free_energy = fit_result
            fit_cache[int(k)] = fit_result
            occupancy = gamma.mean(axis=0)
            empty_states = int(np.sum(occupancy < DEFAULT_EMPTY_STATE_THRESHOLD))
            row = {
                "k": int(k),
                "fit_success": True,
                "free_energy": float(free_energy),
                "mean_occupancy": float(np.mean(occupancy)),
                "minimum_occupancy": float(np.min(occupancy)),
                "n_effectively_empty_states": empty_states,
                "is_degenerate": bool(empty_states > 0),
            }
            successful_rows.append(row)
        except Exception as exc:  # pragma: no cover - exercised through summary table path
            row = {
                "k": int(k),
                "fit_success": False,
                "free_energy": math.nan,
                "mean_occupancy": math.nan,
                "minimum_occupancy": math.nan,
                "n_effectively_empty_states": math.nan,
                "is_degenerate": True,
                "error": str(exc),
            }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    non_degenerate = [
        row
        for row in successful_rows
        if not bool(row["is_degenerate"]) and np.isfinite(row["free_energy"])
    ]
    if non_degenerate:
        best_free_energy = min(row["free_energy"] for row in non_degenerate)
        eligible = [
            row for row in non_degenerate if abs(row["free_energy"] - best_free_energy) <= DEFAULT_MODEL_SELECTION_TOLERANCE
        ]
        selected_k = int(min(row["k"] for row in eligible))
    else:
        successful = [row for row in successful_rows if np.isfinite(row["free_energy"])]
        if not successful:
            raise RuntimeError("All candidate HMM fits failed.")
        best_free_energy = min(row["free_energy"] for row in successful)
        eligible = [
            row for row in successful if abs(row["free_energy"] - best_free_energy) <= DEFAULT_MODEL_SELECTION_TOLERANCE
        ]
        selected_k = int(min(row["k"] for row in eligible))
    return summary, selected_k, fit_cache


# ---------------------------------------------------------------------------
# Pipeline


def fit_tde_hmm_pipeline(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: TdeHmmConfig,
    source_array_column: str = DEFAULT_SOURCE_ARRAY_COLUMN,
    speech_intervals_column: str = DEFAULT_SPEECH_INTERVALS_COLUMN,
    subject_column: str = DEFAULT_SUBJECT_COLUMN,
    run_column: str = DEFAULT_RUN_COLUMN,
    input_sampling_rate_column: str | None = DEFAULT_INPUT_SAMPLING_RATE_COLUMN,
) -> TdeHmmPipelineResult:
    """Run the full causal TDE-HMM feature extraction pipeline."""

    config.validate()
    _resolve_logging(config.verbose)
    manifest = _load_manifest(manifest_path)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    required_columns = {
        source_array_column,
        speech_intervals_column,
        subject_column,
        run_column,
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing required columns: {missing}.")

    concat_blocks: list[np.ndarray] = []
    concat_indices_rows: list[tuple[int, int]] = []
    concat_mapping_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    effective_sampling_rates_hz: list[float] = []
    concat_cursor = 0
    chunk_id_offset = 0

    manifest_rows = manifest.reset_index(drop=True)
    for run_id, row in _progress(
        manifest_rows.iterrows(),
        total=len(manifest_rows),
        desc="Loading runs",
        enabled=config.verbose,
    ):
        subject = str(row[subject_column])
        run = str(row[run_column])
        source_path = _resolve_input_path(manifest_path, row[source_array_column])
        speech_path = _resolve_input_path(manifest_path, row[speech_intervals_column])
        input_sampling_rate_hz = _resolve_input_sampling_rate(
            row,
            input_sampling_rate_column=input_sampling_rate_column,
            target_sampling_rate_hz=config.sampling_rate_hz,
        )

        LOGGER.info("Loading subject=%s run=%s from %s", subject, run, source_path)
        source_data, inferred_sampling_rate_hz = load_source_data(source_path)
        speech_intervals = load_speech_intervals(speech_path)
        if input_sampling_rate_column and input_sampling_rate_column in row.index and pd.notna(row[input_sampling_rate_column]):
            effective_input_sampling_rate_hz = input_sampling_rate_hz
        elif inferred_sampling_rate_hz is not None:
            effective_input_sampling_rate_hz = inferred_sampling_rate_hz
        else:
            effective_input_sampling_rate_hz = input_sampling_rate_hz
        effective_target_sampling_rate_hz = _resolve_effective_target_sampling_rate(
            input_sampling_rate_hz=effective_input_sampling_rate_hz,
            requested_sampling_rate_hz=config.sampling_rate_hz,
        )
        resampled_data, resampled_intervals = _downsample_run_if_needed(
            source_data,
            speech_intervals,
            input_sampling_rate_hz=effective_input_sampling_rate_hz,
            target_sampling_rate_hz=effective_target_sampling_rate_hz,
        )

        if config.standardize_per_run:
            run_data = standardize_data_per_run(
                resampled_data,
                run_boundaries=[(0, resampled_data.shape[0])],
            )
        else:
            run_data = resampled_data.copy()

        keep_mask = build_non_speaking_mask(
            run_data.shape[0],
            sampling_rate_hz=effective_target_sampling_rate_hz,
            speech_intervals_s=resampled_intervals,
            guard_pre_s=config.speech_guard_pre_s,
            guard_post_s=config.speech_guard_post_s,
        )
        minimum_chunk_samples = int(round(config.minimum_chunk_duration_s * effective_target_sampling_rate_hz))
        minimum_chunk_samples = max(1, minimum_chunk_samples)
        chunks = split_valid_samples_into_chunks(keep_mask, minimum_chunk_samples)
        if not chunks:
            raise ValueError(
                f"No non-speaking chunks survived for subject={subject} run={run}. "
                "Check speech intervals, guard bands, and minimum chunk duration."
            )

        concatenated_run_data, run_indices, run_mapping = concatenate_chunks_and_build_indices(run_data, chunks)
        run_mapping = run_mapping.assign(
            run_id=run_id,
            subject=subject,
            run=run,
            chunk_id=lambda frame: frame["chunk_id"] + chunk_id_offset,
        )
        for local_chunk_id, (start_sample, stop_sample) in enumerate(chunks):
            local_concat_start = int(run_indices[local_chunk_id, 0])
            local_concat_stop = int(run_indices[local_chunk_id, 1])
            global_concat_start = concat_cursor + local_concat_start
            global_concat_stop = concat_cursor + local_concat_stop
            chunk_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "chunk_id": chunk_id_offset + local_chunk_id,
                    "original_start_sample": start_sample,
                    "original_stop_sample": stop_sample,
                    "concat_start_sample": global_concat_start,
                    "concat_stop_sample": global_concat_stop,
                }
            )
            concat_indices_rows.append((global_concat_start, global_concat_stop))
        concat_blocks.append(concatenated_run_data)
        concat_mapping_rows.extend(
            {
                "concat_index": concat_cursor + int(mapping_row["concat_index"]),
                "original_sample_index": int(mapping_row["original_sample_index"]),
                "chunk_id": int(mapping_row["chunk_id"]),
                "run_id": int(mapping_row["run_id"]),
                "subject": str(mapping_row["subject"]),
                "run": str(mapping_row["run"]),
            }
            for mapping_row in run_mapping.to_dict("records")
        )
        run_records.append(
            {
                "run_id": run_id,
                "subject": subject,
                "run": run,
                "n_samples": int(run_data.shape[0]),
                "sampling_rate_hz": float(effective_target_sampling_rate_hz),
            }
        )
        effective_sampling_rates_hz.append(float(effective_target_sampling_rate_hz))
        concat_cursor += concatenated_run_data.shape[0]
        chunk_id_offset += len(chunks)

    unique_sampling_rates_hz = sorted({round(rate, 9) for rate in effective_sampling_rates_hz})
    if len(unique_sampling_rates_hz) != 1:
        raise ValueError(
            "TDE-HMM currently requires a common effective sampling rate across runs, "
            f"but found: {unique_sampling_rates_hz}."
        )
    common_sampling_rate_hz = float(unique_sampling_rates_hz[0])

    concatenated_data = np.vstack(concat_blocks)
    concat_indices = np.asarray(concat_indices_rows, dtype=int)
    concat_mapping = pd.DataFrame(concat_mapping_rows).sort_values("concat_index").reset_index(drop=True)

    LOGGER.info("Preprocessing %d concatenated samples across %d chunks", concatenated_data.shape[0], len(chunk_rows))
    processed_data, processed_indices, preprocessing_log, processed_mapping = preprocess_data_for_glhmm(
        concatenated_data,
        concat_indices,
        config,
        sampling_rate_hz=common_sampling_rate_hz,
    )
    processed_mapping = processed_mapping.merge(
        concat_mapping[["concat_index", "run_id", "subject", "run", "chunk_id"]],
        on=["concat_index", "chunk_id"],
        how="left",
        validate="one_to_one",
    )

    chunk_table = pd.DataFrame(chunk_rows).sort_values("chunk_id").reset_index(drop=True)
    processed_bounds = (
        processed_mapping.groupby("chunk_id")["processed_index"]
        .agg(processed_start_sample="min", processed_stop_sample="max")
        .reset_index()
    )
    processed_bounds["processed_stop_sample"] = processed_bounds["processed_stop_sample"] + 1
    chunk_table = chunk_table.merge(processed_bounds, on="chunk_id", how="left", validate="one_to_one")

    LOGGER.info("Evaluating candidate K values: %s", ", ".join(str(k) for k in config.candidate_k))
    model_selection_summary, selected_k, fit_cache = evaluate_candidate_k_values(
        processed_data,
        processed_indices,
        config,
    )
    LOGGER.info("Selected K=%d", selected_k)

    hmm, gamma, _, final_free_energy = fit_cache[selected_k]
    state_entropy = compute_state_entropy(gamma)

    model_path = output_path / f"hmm_k{selected_k}.pkl"
    gamma_path = output_path / f"gamma_k{selected_k}.npy"
    entropy_processed_path = output_path / f"state_entropy_processed_k{selected_k}.npy"
    model_selection_path = output_path / "model_selection.csv"
    chunk_table_path = output_path / "chunks.csv"
    fit_summary_path = output_path / "fit_summary.json"

    with model_path.open("wb") as handle:
        pickle.dump(hmm, handle)
    np.save(gamma_path, gamma)
    np.save(entropy_processed_path, state_entropy)
    model_selection_summary.to_csv(model_selection_path, index=False)
    chunk_table.to_csv(chunk_table_path, index=False)

    entropy_csv_paths: list[Path] = []
    for run_record in _progress(
        run_records,
        total=len(run_records),
        desc="Writing outputs",
        enabled=config.verbose,
    ):
        run_id = int(run_record["run_id"])
        original_n_samples = int(run_record["n_samples"])
        run_concat_mapping = concat_mapping.loc[concat_mapping["run_id"] == run_id].copy()
        run_processed_mapping = processed_mapping.loc[processed_mapping["run_id"] == run_id].copy()
        run_entropy = state_entropy[run_processed_mapping["processed_index"].to_numpy(dtype=int)]
        entropy_timeline = map_processed_feature_to_original_timeline(
            run_entropy,
            run_processed_mapping[["processed_index", "concat_index", "chunk_id"]].reset_index(drop=True),
            run_concat_mapping[["concat_index", "original_sample_index", "chunk_id"]].reset_index(drop=True),
            original_n_samples,
        )
        time_s = np.arange(original_n_samples, dtype=float) / float(run_record["sampling_rate_hz"])
        entropy_frame = pd.DataFrame(
            {
                "sample": np.arange(original_n_samples, dtype=int),
                "time_s": time_s,
                "state_entropy": entropy_timeline,
            }
        )
        entropy_csv_path = output_path / (
            f"subject-{_sanitize_token(str(run_record['subject']))}"
            f"_run-{_sanitize_token(str(run_record['run']))}_state_entropy.csv"
        )
        entropy_frame.to_csv(entropy_csv_path, index=False)
        entropy_csv_paths.append(entropy_csv_path)

    fit_summary = {
        "selected_k": selected_k,
        "candidate_k": list(config.candidate_k),
        "lags": preprocessing_log["lags"],
        "sampling_rate_hz": common_sampling_rate_hz,
        "requested_sampling_rate_hz": config.sampling_rate_hz,
        "causal_history_ms": config.causal_history_ms,
        "speech_guard_pre_s": config.speech_guard_pre_s,
        "speech_guard_post_s": config.speech_guard_post_s,
        "minimum_chunk_duration_s": config.minimum_chunk_duration_s,
        "number_of_chunks": int(len(chunk_table)),
        "total_samples_kept": int(concatenated_data.shape[0]),
        "total_processed_samples": int(processed_data.shape[0]),
        "free_energy_per_k": {
            str(int(row["k"])): (
                None if not np.isfinite(row["free_energy"]) else float(row["free_energy"])
            )
            for row in model_selection_summary.to_dict("records")
        },
        "selected_model_free_energy": float(final_free_energy),
        "config": asdict(config),
    }
    fit_summary_path.write_text(
        json.dumps(fit_summary, indent=JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TdeHmmPipelineResult(
        output_dir=output_path,
        selected_k=selected_k,
        model_path=model_path,
        model_selection_path=model_selection_path,
        gamma_path=gamma_path,
        entropy_processed_path=entropy_processed_path,
        chunk_table_path=chunk_table_path,
        fit_summary_path=fit_summary_path,
        entropy_csv_paths=tuple(entropy_csv_paths),
        total_chunks=int(len(chunk_table)),
        total_samples_kept=int(concatenated_data.shape[0]),
        total_processed_samples=int(processed_data.shape[0]),
    )


# ---------------------------------------------------------------------------
# IO helpers


def load_source_data(path: Path) -> tuple[np.ndarray, float | None]:
    """Load source or preprocessed EEG data from disk.

    Parameters
    ----------
    path : Path
        Input path. Supported formats are ``.npy`` and ``.fif``.

    Returns
    -------
    tuple
        Data array with shape ``(n_samples, n_features)`` and an optional
        inferred sampling rate in Hz.
    """

    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Expected a NumPy array at {path}.")
        return _validate_2d_array(array, label=f"source array {path}"), None

    if suffix == ".fif":
        _configure_mne_runtime()
        import mne

        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        data = raw.get_data().T
        return _validate_2d_array(data, label=f"preprocessed EEG {path}"), float(raw.info["sfreq"])

    raise ValueError(f"Unsupported source data format for {path}. Expected .npy or .fif.")


def load_speech_intervals(path: Path) -> np.ndarray:
    """Load speech intervals from CSV."""

    frame = pd.read_csv(path)
    required_columns = {DEFAULT_ONSET_COLUMN, DEFAULT_OFFSET_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Speech intervals file {path} is missing required columns: {missing}.")
    return frame[[DEFAULT_ONSET_COLUMN, DEFAULT_OFFSET_COLUMN]].to_numpy(dtype=float)


def _configure_mne_runtime() -> None:
    """Configure MNE before import on Python stacks that fail with numba caching."""

    os.environ["MNE_USE_NUMBA"] = "false"
    try:
        from lmeeeg.backends.correction._regression import configure_mne_runtime
    except Exception:
        return
    configure_mne_runtime()


# ---------------------------------------------------------------------------
# Internal helpers


def _build_causal_tde_data(
    data: np.ndarray,
    indices: np.ndarray,
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build causal time-delay embedded data without cross-chunk leakage."""

    min_lag = int(np.min(lags))
    max_lag = int(np.max(lags))
    if max_lag != 0:
        raise ValueError("Causal embedding expects the rightmost lag to be 0.")
    history_samples = -min_lag
    if history_samples < 0:
        raise ValueError("History sample count must be non-negative.")

    embedded_chunks: list[np.ndarray] = []
    processed_indices_rows: list[tuple[int, int]] = []
    mapping_rows: list[dict[str, int]] = []
    processed_cursor = 0
    n_features = data.shape[1]

    for chunk_id, (start, stop) in enumerate(indices):
        chunk = data[start:stop]
        chunk_length = stop - start
        if chunk_length <= history_samples:
            continue

        processed_start = processed_cursor
        current_times = np.arange(start + history_samples, stop, dtype=int)
        embedded = np.empty((current_times.size, n_features * lags.size), dtype=float)
        for lag_position, lag in enumerate(lags):
            source_indices = current_times + lag
            feature_slice = slice(lag_position * n_features, (lag_position + 1) * n_features)
            embedded[:, feature_slice] = data[source_indices]
        embedded_chunks.append(embedded)

        processed_stop = processed_start + embedded.shape[0]
        processed_indices_rows.append((processed_start, processed_stop))
        mapping_rows.extend(
            {
                "processed_index": processed_start + offset,
                "concat_index": int(concat_index),
                "chunk_id": chunk_id,
            }
            for offset, concat_index in enumerate(current_times)
        )
        processed_cursor = processed_stop

    if not embedded_chunks:
        return np.empty((0, data.shape[1] * lags.size), dtype=float), np.empty((0, 2), dtype=int), pd.DataFrame(
            columns=["processed_index", "concat_index", "chunk_id"]
        )

    return (
        np.vstack(embedded_chunks),
        np.asarray(processed_indices_rows, dtype=int),
        pd.DataFrame(mapping_rows),
    )


def _build_causal_tde_data_with_incremental_pca(
    data: np.ndarray,
    indices: np.ndarray,
    lags: np.ndarray,
    *,
    pca_n_components: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build causal TDE data with streamed standardization and incremental PCA."""

    n_embedded_features = data.shape[1] * lags.size
    if pca_n_components > n_embedded_features:
        raise ValueError(
            "`pca_n_components` cannot exceed the number of embedded features "
            f"({n_embedded_features})."
        )

    embedded_sample_count = 0
    sum_vector = np.zeros(n_embedded_features, dtype=np.float64)
    sumsq_vector = np.zeros(n_embedded_features, dtype=np.float64)
    for embedded_chunk, _, _ in _progress(
        _iterate_causal_tde_chunks(data, indices, lags),
        total=indices.shape[0],
        desc="Embedding pass 1/3",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        embedded_sample_count += embedded_chunk.shape[0]
        sum_vector += embedded_chunk.sum(axis=0, dtype=np.float64)
        sumsq_vector += np.square(embedded_chunk, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if embedded_sample_count == 0:
        return (
            np.empty((0, pca_n_components), dtype=np.float32),
            np.empty((0, 2), dtype=int),
            pd.DataFrame(columns=["processed_index", "concat_index", "chunk_id"]),
        )

    mean_vector = sum_vector / embedded_sample_count
    variance_vector = (sumsq_vector / embedded_sample_count) - np.square(mean_vector)
    variance_vector = np.maximum(variance_vector, 0.0)
    std_vector = np.sqrt(variance_vector)
    std_vector[std_vector == 0.0] = 1.0

    batch_size = max(2048, pca_n_components * 10)
    incremental_pca = IncrementalPCA(n_components=pca_n_components, batch_size=batch_size)
    pca_batch: list[np.ndarray] = []
    pca_batch_rows = 0
    for embedded_chunk, _, _ in _progress(
        _iterate_causal_tde_chunks(data, indices, lags),
        total=indices.shape[0],
        desc="Embedding pass 2/3",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        standardized_chunk = ((embedded_chunk - mean_vector) / std_vector).astype(np.float32, copy=False)
        pca_batch.append(standardized_chunk)
        pca_batch_rows += standardized_chunk.shape[0]
        if pca_batch_rows >= batch_size:
            incremental_pca.partial_fit(np.vstack(pca_batch))
            pca_batch = []
            pca_batch_rows = 0
    if pca_batch:
        stacked = np.vstack(pca_batch)
        if stacked.shape[0] >= pca_n_components:
            incremental_pca.partial_fit(stacked)
        else:
            raise ValueError(
                "Not enough embedded samples to fit IncrementalPCA with "
                f"{pca_n_components} components."
            )

    processed_sample_count = embedded_sample_count
    processed_data = np.empty((processed_sample_count, pca_n_components), dtype=np.float32)
    processed_indices_rows: list[tuple[int, int]] = []
    mapping_rows: list[dict[str, int]] = []
    processed_cursor = 0
    for embedded_chunk, current_times, chunk_id in _progress(
        _iterate_causal_tde_chunks(data, indices, lags),
        total=indices.shape[0],
        desc="Embedding pass 3/3",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        standardized_chunk = ((embedded_chunk - mean_vector) / std_vector).astype(np.float32, copy=False)
        transformed_chunk = incremental_pca.transform(standardized_chunk).astype(np.float32, copy=False)
        chunk_start = processed_cursor
        chunk_stop = chunk_start + transformed_chunk.shape[0]
        processed_data[chunk_start:chunk_stop] = transformed_chunk
        processed_indices_rows.append((chunk_start, chunk_stop))
        mapping_rows.extend(
            {
                "processed_index": chunk_start + offset,
                "concat_index": int(concat_index),
                "chunk_id": chunk_id,
            }
            for offset, concat_index in enumerate(current_times)
        )
        processed_cursor = chunk_stop

    return processed_data, np.asarray(processed_indices_rows, dtype=int), pd.DataFrame(mapping_rows)


def _iterate_causal_tde_chunks(
    data: np.ndarray,
    indices: np.ndarray,
    lags: np.ndarray,
):
    """Yield embedded chunks without crossing segment boundaries."""

    min_lag = int(np.min(lags))
    max_lag = int(np.max(lags))
    if max_lag != 0:
        raise ValueError("Causal embedding expects the rightmost lag to be 0.")
    history_samples = -min_lag
    n_features = data.shape[1]

    for chunk_id, (start, stop) in enumerate(indices):
        chunk_length = stop - start
        if chunk_length <= history_samples:
            continue
        current_times = np.arange(start + history_samples, stop, dtype=int)
        embedded_chunk = np.empty((current_times.size, n_features * lags.size), dtype=np.float32)
        for lag_position, lag in enumerate(lags):
            source_indices = current_times + lag
            feature_slice = slice(lag_position * n_features, (lag_position + 1) * n_features)
            embedded_chunk[:, feature_slice] = data[source_indices]
        yield embedded_chunk, current_times, chunk_id


def _downsample_run_if_needed(
    source_data: np.ndarray,
    speech_intervals: np.ndarray,
    *,
    input_sampling_rate_hz: float,
    target_sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a run and keep speech intervals in seconds unchanged."""

    if input_sampling_rate_hz <= 0:
        raise ValueError("`input_sampling_rate_hz` must be positive.")
    if target_sampling_rate_hz <= 0:
        raise ValueError("`target_sampling_rate_hz` must be positive.")
    if math.isclose(input_sampling_rate_hz, target_sampling_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        return source_data, speech_intervals
    if input_sampling_rate_hz < target_sampling_rate_hz:
        LOGGER.warning(
            "Input sampling rate %.6f Hz is below requested %.6f Hz; skipping upsampling and using the native rate.",
            input_sampling_rate_hz,
            target_sampling_rate_hz,
        )
        return source_data, speech_intervals

    up, down = _rational_resample_factors(target_sampling_rate_hz, input_sampling_rate_hz)
    LOGGER.info(
        "Downsampling run from %.6f Hz to %.6f Hz with polyphase factors up=%d down=%d",
        input_sampling_rate_hz,
        target_sampling_rate_hz,
        up,
        down,
    )
    resampled = signal.resample_poly(source_data, up=up, down=down, axis=0)
    return np.asarray(resampled, dtype=float), speech_intervals


def _resolve_effective_target_sampling_rate(
    *,
    input_sampling_rate_hz: float,
    requested_sampling_rate_hz: float,
) -> float:
    """Resolve the sampling rate actually used downstream."""

    if input_sampling_rate_hz < requested_sampling_rate_hz:
        return float(input_sampling_rate_hz)
    return float(requested_sampling_rate_hz)


def _extract_final_free_energy(free_energy_history: np.ndarray | list[float] | float) -> float:
    """Extract the final finite free-energy value."""

    values = np.asarray(free_energy_history, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("GLHMM fit did not return a finite free-energy value.")
    return float(finite_values[-1])


def _load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the manifest CSV."""

    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return frame


def _resolve_input_path(manifest_path: Path, value: Any) -> Path:
    """Resolve a manifest path value relative to the manifest location."""

    resolved = Path(str(value))
    if not resolved.is_absolute():
        resolved = manifest_path.resolve().parent / resolved
    return resolved.resolve()


def _resolve_input_sampling_rate(
    row: pd.Series,
    *,
    input_sampling_rate_column: str | None,
    target_sampling_rate_hz: float,
) -> float:
    """Resolve the input sampling rate for a run."""

    if input_sampling_rate_column and input_sampling_rate_column in row.index and pd.notna(row[input_sampling_rate_column]):
        input_sampling_rate_hz = float(row[input_sampling_rate_column])
        if input_sampling_rate_hz <= 0:
            raise ValueError(
                f"Manifest column `{input_sampling_rate_column}` must contain positive sampling rates."
            )
        return input_sampling_rate_hz
    return float(target_sampling_rate_hz)


def _rational_resample_factors(target_hz: float, source_hz: float) -> tuple[int, int]:
    """Compute integer polyphase resampling factors."""

    source_ns = int(round(source_hz * 1_000_000))
    target_ns = int(round(target_hz * 1_000_000))
    common_divisor = math.gcd(source_ns, target_ns)
    up = target_ns // common_divisor
    down = source_ns // common_divisor
    if up <= 0 or down <= 0:
        raise ValueError("Invalid resampling factors.")
    return up, down


def _sanitize_token(value: str) -> str:
    """Sanitize an identifier for filenames."""

    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
