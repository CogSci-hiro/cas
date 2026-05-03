"""Pooled FPP-vs-SPP entropy hazard interaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as glm
from tqdm.auto import tqdm
import yaml

TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"
DEFAULT_INSTABILITY_WINDOW_S = 0.25
RISKSET_REQUIRED_COLUMNS = (
    "episode_id",
    "dyad_id",
    "subject_id",
    "run_id",
    "bin_start_s",
    "bin_end_s",
    "event_bin",
    "time_from_partner_onset_s",
    "time_from_partner_offset_s",
    "time_within_run_s",
)
NEURAL_REQUIRED_COLUMNS = ("subject_id", "run_id", "time_s", "entropy")
TIMING_Z_COLUMNS = (
    "time_from_partner_onset_s_z",
    "time_from_partner_offset_s_z",
    "time_within_run_s_z",
)

glm.SET_USE_BIC_LLF(True)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NeuralHazardFppSppConfig:
    """Configuration for the FPP-vs-SPP neural hazard interaction pipeline."""

    fpp_risk_set_path: Path
    spp_risk_set_path: Path
    neural_features_path: Path
    out_dir: Path
    bin_width_s: float
    lag_grid_ms: tuple[int, ...]
    pca_input_columns: tuple[str, ...]
    nearest_merge_tolerance_s: float
    timing_zscore_scope: str = "global"
    neural_zscore_scope: str = "subject_run"
    pca_pc1_warning_threshold: float = 0.40
    pca_n_components: int = 1
    lag_selection_criterion: str = "delta_loglik"
    avoid_zero_lag: bool = True
    minimum_circular_shift_duration_s: float = 1.0
    n_circular_shift_permutations: int = 100
    circular_shift_seed: int = 12345
    circular_shift_interaction_test: str = "one_sided_negative"
    event_triggered_window_start_s: float = -1.5
    event_triggered_window_end_s: float = 0.5
    overwrite: bool = True
    verbose: bool = True
    dyads_csv_path: Path | None = None

    def validate(self) -> None:
        """Validate config values before running the pipeline."""

        if self.bin_width_s <= 0.0:
            raise ValueError("`bin_width_s` must be positive.")
        if not self.lag_grid_ms:
            raise ValueError("`lag_grid_ms` must contain at least one lag.")
        if self.nearest_merge_tolerance_s <= 0.0:
            raise ValueError("`nearest_merge_tolerance_s` must be positive.")
        if self.timing_zscore_scope not in {"global", "subject_run"}:
            raise ValueError("`timing_zscore_scope` must be `global` or `subject_run`.")
        if self.neural_zscore_scope != "subject_run":
            raise ValueError("`neural_zscore_scope` currently must be `subject_run`.")
        if self.pca_n_components < 1:
            raise ValueError("`pca_n_components` must be at least 1.")
        if self.lag_selection_criterion not in {"delta_loglik", "delta_aic"}:
            raise ValueError("`lag_selection_criterion` must be `delta_loglik` or `delta_aic`.")
        if self.minimum_circular_shift_duration_s < 0.0:
            raise ValueError("`minimum_circular_shift_duration_s` must be non-negative.")
        if self.n_circular_shift_permutations < 0:
            raise ValueError("`n_circular_shift_permutations` must be non-negative.")
        if self.circular_shift_interaction_test not in {"one_sided_negative", "two_sided"}:
            raise ValueError(
                "`circular_shift_interaction_test` must be `one_sided_negative` or `two_sided`."
            )
        if self.event_triggered_window_end_s <= self.event_triggered_window_start_s:
            raise ValueError("Event-triggered window end must be greater than start.")


@dataclass(frozen=True, slots=True)
class NeuralHazardFppSppResult:
    """High-level output handles for the completed pipeline."""

    out_dir: Path
    model_comparison_path: Path
    coefficients_path: Path
    circular_shift_summary_path: Path
    summary_json_path: Path


def build_entropy_features_table_from_glhmm_output(
    glhmm_output_dir: Path,
    output_path: Path,
    *,
    instability_window_s: float = DEFAULT_INSTABILITY_WINDOW_S,
) -> Path:
    """Aggregate per-run GLHMM entropy CSV exports and derived instability features."""

    from cas.hmm.tde_hmm import load_source_data

    input_dir = glhmm_output_dir.resolve()
    csv_paths = sorted(input_dir.glob("subject-*_run-*_state_entropy.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per-run state entropy CSVs were found in {input_dir}")

    fit_summary_path = input_dir / "fit_summary.json"
    chunks_path = input_dir / "chunks.csv"
    input_manifest_path = input_dir / "input_manifest.csv"
    if not fit_summary_path.exists():
        raise FileNotFoundError(f"Missing GLHMM fit summary: {fit_summary_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing GLHMM chunks table: {chunks_path}")
    if not input_manifest_path.exists():
        raise FileNotFoundError(f"Missing GLHMM input manifest: {input_manifest_path}")

    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    selected_k = int(fit_summary["selected_k"])
    gamma_path = input_dir / f"gamma_k{selected_k}.npy"
    if not gamma_path.exists():
        raise FileNotFoundError(f"Missing GLHMM posterior probabilities: {gamma_path}")

    gamma = np.asarray(np.load(gamma_path), dtype=float)
    if gamma.ndim != 2:
        raise ValueError(f"Expected 2D gamma array at {gamma_path}")

    chunk_table = pd.read_csv(chunks_path)
    input_manifest = pd.read_csv(input_manifest_path)
    manifest_columns = {"subject", "run", "source_path"}
    if not manifest_columns.issubset(input_manifest.columns):
        raise ValueError(
            "GLHMM input manifest is missing required columns: "
            + ", ".join(sorted(manifest_columns.difference(input_manifest.columns)))
        )

    if float(instability_window_s) <= 0.0:
        raise ValueError("`instability_window_s` must be positive.")

    lag_span_samples = _infer_glhmm_lag_span_samples(fit_summary)
    LOGGER.info(
        "Aggregating %d per-run entropy CSVs and instability features from %s",
        int(len(csv_paths)),
        input_dir,
    )
    frames: list[pd.DataFrame] = []
    manifest_by_key = {
        (str(row["subject"]), str(row["run"])): dict(row)
        for row in input_manifest.to_dict(orient="records")
    }
    for csv_path in _progress(
        csv_paths,
        total=len(csv_paths),
        desc="Aggregating entropy CSVs",
        enabled=True,
    ):
        name = csv_path.name
        if "_run-" not in name or not name.endswith("_state_entropy.csv"):
            continue
        subject_part, run_part = name.removesuffix("_state_entropy.csv").split("_run-", 1)
        subject_id = subject_part.removeprefix("subject-")
        run_id = run_part
        frame = pd.read_csv(csv_path)
        required = {"sample", "time_s", "state_entropy"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Entropy CSV is missing required columns: {csv_path}")
        manifest_row = manifest_by_key.get((str(subject_id), str(run_id)))
        if manifest_row is None:
            raise ValueError(
                f"Could not find source_path in input_manifest.csv for subject={subject_id}, run={run_id}"
            )

        entropy_time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        entropy_values = pd.to_numeric(frame["state_entropy"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(entropy_time_s).all():
            raise ValueError(f"Entropy CSV contains non-finite time_s values: {csv_path}")

        source_data, source_sampling_rate_hz = load_source_data(Path(str(manifest_row["source_path"])))
        if source_sampling_rate_hz is None or not np.isfinite(source_sampling_rate_hz):
            source_sampling_rate_hz = float(fit_summary["sampling_rate_hz"])
        eeg_features = _compute_eeg_instability_features(
            source_data=source_data,
            source_sampling_rate_hz=float(source_sampling_rate_hz),
            target_time_s=entropy_time_s,
            window_duration_s=float(instability_window_s),
        )
        posterior_features = _compute_hmm_instability_features(
            gamma=gamma,
            chunk_table=chunk_table,
            subject_id=str(subject_id),
            run_id=str(run_id),
            original_n_samples=int(len(frame)),
            lag_span_samples=lag_span_samples,
            sampling_rate_hz=float(fit_summary["sampling_rate_hz"]),
            target_time_s=entropy_time_s,
            window_duration_s=float(instability_window_s),
        )
        state_probabilities = _compute_hmm_state_probabilities(
            gamma=gamma,
            chunk_table=chunk_table,
            subject_id=str(subject_id),
            run_id=str(run_id),
            original_n_samples=int(len(frame)),
            lag_span_samples=lag_span_samples,
            sampling_rate_hz=float(fit_summary["sampling_rate_hz"]),
            target_time_s=entropy_time_s,
        )
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": str(subject_id),
                    "run_id": str(run_id),
                    "time_s": entropy_time_s,
                    "entropy": entropy_values,
                    **eeg_features,
                    **posterior_features,
                    **state_probabilities,
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["subject_id", "run_id", "time_s"], kind="mergesort"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        combined.to_parquet(output_path, index=False)
    else:
        combined.to_csv(output_path, index=False)
    LOGGER.info("Wrote aggregated entropy features to %s (%d rows).", output_path, int(len(combined)))
    return output_path


def _compute_hmm_state_probabilities(
    *,
    gamma: np.ndarray,
    chunk_table: pd.DataFrame,
    subject_id: str,
    run_id: str,
    original_n_samples: int,
    lag_span_samples: int,
    sampling_rate_hz: float,
    target_time_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct run-wise posterior state probabilities at target times."""

    run_chunks = chunk_table.loc[
        (chunk_table["subject"].astype(str) == str(subject_id))
        & (chunk_table["run"].astype(str) == str(run_id))
    ].sort_values("original_start_sample", kind="mergesort")
    n_states = int(gamma.shape[1])
    timelines = np.full((original_n_samples, n_states), np.nan, dtype=float)
    if run_chunks.empty:
        return {
            f"state_probability_{state_idx + 1}": np.full(len(target_time_s), np.nan, dtype=float)
            for state_idx in range(n_states)
        }
    for chunk in run_chunks.to_dict(orient="records"):
        processed_start = int(chunk["processed_start_sample"])
        processed_stop = int(chunk["processed_stop_sample"])
        posterior = np.asarray(gamma[processed_start:processed_stop], dtype=float)
        if posterior.ndim != 2 or posterior.shape[0] == 0:
            continue
        original_start = int(chunk["original_start_sample"])
        mapped_original_start = int(original_start) + int(lag_span_samples)
        mapped_original_stop = mapped_original_start + posterior.shape[0]
        if mapped_original_stop > original_n_samples:
            raise ValueError(
                "HMM posterior mapping exceeded the original run length for "
                f"subject={subject_id}, run={run_id}."
            )
        timelines[mapped_original_start:mapped_original_stop, :] = posterior

    source_time_s = np.arange(original_n_samples, dtype=float) / float(sampling_rate_hz)
    out: dict[str, np.ndarray] = {}
    for state_idx in range(n_states):
        out[f"state_probability_{state_idx + 1}"] = _interpolate_to_target_times(
            source_time_s,
            timelines[:, state_idx],
            target_time_s,
        )
    return out


def _infer_glhmm_lag_span_samples(fit_summary: dict[str, Any]) -> int:
    """Infer the causal lag span in processed samples from saved GLHMM metadata."""

    lags = [int(value) for value in fit_summary.get("lags", [])]
    if not lags:
        raise ValueError("GLHMM fit summary did not contain saved lags.")
    if max(lags) != 0:
        raise ValueError("Expected causal GLHMM lags ending at 0.")
    if lags != list(range(min(lags), max(lags) + 1)):
        raise ValueError("Expected contiguous causal GLHMM lags for processed-to-original reconstruction.")
    return int(max(lags) - min(lags))


def _compute_eeg_instability_features(
    *,
    source_data: np.ndarray,
    source_sampling_rate_hz: float,
    target_time_s: np.ndarray,
    window_duration_s: float,
) -> dict[str, np.ndarray]:
    """Compute EEG-derived instability features and resample them to the target timeline."""

    data = np.asarray(source_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D source EEG array with shape (n_samples, n_channels).")
    if data.shape[0] == 0:
        return {
            "signal_variance": np.full(len(target_time_s), np.nan, dtype=float),
            "broadband_power": np.full(len(target_time_s), np.nan, dtype=float),
        }
    source_time_s = np.arange(data.shape[0], dtype=float) / float(source_sampling_rate_hz)
    global_field_power = np.nanstd(data, axis=1, ddof=0)
    mean_squared_amplitude = np.nanmean(np.square(data), axis=1)
    source_window_samples = max(1, int(round(float(window_duration_s) * float(source_sampling_rate_hz))))
    signal_variance_raw = _rolling_nanvar(global_field_power, source_window_samples)
    broadband_power_raw = _rolling_nanmean(mean_squared_amplitude, source_window_samples)
    return {
        "signal_variance": _interpolate_to_target_times(source_time_s, signal_variance_raw, target_time_s),
        "broadband_power": _interpolate_to_target_times(source_time_s, broadband_power_raw, target_time_s),
    }


def _compute_hmm_instability_features(
    *,
    gamma: np.ndarray,
    chunk_table: pd.DataFrame,
    subject_id: str,
    run_id: str,
    original_n_samples: int,
    lag_span_samples: int,
    sampling_rate_hz: float,
    target_time_s: np.ndarray,
    window_duration_s: float,
) -> dict[str, np.ndarray]:
    """Compute HMM-derived instability features aligned to original run coordinates."""

    run_chunks = chunk_table.loc[
        (chunk_table["subject"].astype(str) == str(subject_id))
        & (chunk_table["run"].astype(str) == str(run_id))
    ].sort_values("original_start_sample", kind="mergesort")
    hard_state_timeline = np.full(original_n_samples, np.nan, dtype=float)
    switching_timeline = np.full(original_n_samples, np.nan, dtype=float)
    posterior_volatility_timeline = np.full(original_n_samples, np.nan, dtype=float)
    if run_chunks.empty:
        return {
            "state_switching_rate": np.full(len(target_time_s), np.nan, dtype=float),
            "model_badness": np.full(len(target_time_s), np.nan, dtype=float),
        }
    for chunk in run_chunks.to_dict(orient="records"):
        processed_start = int(chunk["processed_start_sample"])
        processed_stop = int(chunk["processed_stop_sample"])
        original_start = int(chunk["original_start_sample"])
        posterior = np.asarray(gamma[processed_start:processed_stop], dtype=float)
        if posterior.ndim != 2 or posterior.shape[0] == 0:
            continue
        hard_state = np.argmax(posterior, axis=1).astype(float)
        switching = np.full(posterior.shape[0], np.nan, dtype=float)
        volatility = np.full(posterior.shape[0], np.nan, dtype=float)
        if posterior.shape[0] > 1:
            switching[1:] = (hard_state[1:] != hard_state[:-1]).astype(float)
            volatility[1:] = 0.5 * np.sum(np.abs(np.diff(posterior, axis=0)), axis=1)
        mapped_original_start = int(original_start) + int(lag_span_samples)
        mapped_original_stop = mapped_original_start + posterior.shape[0]
        if mapped_original_stop > original_n_samples:
            raise ValueError(
                "HMM posterior mapping exceeded the original run length for "
                f"subject={subject_id}, run={run_id}."
            )
        hard_state_timeline[mapped_original_start:mapped_original_stop] = hard_state
        switching_timeline[mapped_original_start:mapped_original_stop] = switching
        posterior_volatility_timeline[mapped_original_start:mapped_original_stop] = volatility
    timeline_window_samples = max(1, int(round(float(window_duration_s) * float(sampling_rate_hz))))
    switching_rate = _rolling_nanmean(switching_timeline, timeline_window_samples)
    model_badness = _rolling_nanmean(posterior_volatility_timeline, timeline_window_samples)
    source_time_s = np.arange(original_n_samples, dtype=float) / float(sampling_rate_hz)
    return {
        "state_switching_rate": _interpolate_to_target_times(source_time_s, switching_rate, target_time_s),
        "model_badness": _interpolate_to_target_times(source_time_s, model_badness, target_time_s),
    }


def _rolling_nanmean(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Compute a centered rolling mean while ignoring NaN values."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array for rolling mean.")
    if len(array) == 0:
        return np.asarray(array, dtype=float)
    kernel = np.ones(int(max(window_samples, 1)), dtype=float)
    valid = np.isfinite(array).astype(float)
    weighted = np.where(np.isfinite(array), array, 0.0)
    numer = np.convolve(weighted, kernel, mode="same")
    denom = np.convolve(valid, kernel, mode="same")
    out = np.full(len(array), np.nan, dtype=float)
    mask = denom > 0.0
    out[mask] = numer[mask] / denom[mask]
    return out


def _rolling_nanvar(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Compute a centered rolling variance while ignoring NaN values."""

    array = np.asarray(values, dtype=float)
    mean = _rolling_nanmean(array, window_samples)
    mean_sq = _rolling_nanmean(np.square(array), window_samples)
    variance = mean_sq - np.square(mean)
    variance[variance < 0.0] = 0.0
    variance[~np.isfinite(mean)] = np.nan
    return variance


def _interpolate_to_target_times(
    source_time_s: np.ndarray,
    source_values: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """Interpolate a one-dimensional source series to target times, preserving NaN gaps conservatively."""

    x = np.asarray(source_time_s, dtype=float)
    y = np.asarray(source_values, dtype=float)
    target = np.asarray(target_time_s, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    out = np.full(len(target), np.nan, dtype=float)
    if np.count_nonzero(valid) == 0:
        return out
    if np.count_nonzero(valid) == 1:
        out[:] = y[valid][0]
        return out
    valid_x = x[valid]
    valid_y = y[valid]
    in_bounds = (target >= valid_x.min()) & (target <= valid_x.max())
    out[in_bounds] = np.interp(target[in_bounds], valid_x, valid_y)
    return out


@dataclass(frozen=True, slots=True)
class FittedGlmModel:
    """A fitted GLM plus metadata used for downstream comparison tables."""

    model_name: str
    formula: str
    result: Any | None
    n_rows: int
    n_events: int
    converged: bool
    warnings_text: list[str]
    error_message: str | None

    @property
    def log_likelihood(self) -> float:
        return float(getattr(self.result, "llf", np.nan)) if self.result is not None else float("nan")

    @property
    def aic(self) -> float:
        return float(getattr(self.result, "aic", np.nan)) if self.result is not None else float("nan")

    @property
    def bic(self) -> float:
        return float(getattr(self.result, "bic", np.nan)) if self.result is not None else float("nan")

    @property
    def df_model(self) -> int:
        if self.result is None:
            return 0
        params = getattr(self.result, "params", None)
        return int(len(params)) if params is not None else 0


def load_neural_hazard_fpp_spp_config(path: Path) -> NeuralHazardFppSppConfig:
    """Load the YAML config for the FPP-vs-SPP neural hazard pipeline."""

    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in config file: {config_path}")
    project_root = config_path.parent.parent
    derivatives_root = _load_derivatives_root(project_root / "config" / "paths.yaml")
    input_payload = dict(payload.get("input") or {})
    output_payload = dict(payload.get("output") or {})
    lag_payload = dict(payload.get("lags") or {})
    pca_payload = dict(payload.get("pca") or {})
    merge_payload = dict(payload.get("merge") or {})
    zscore_payload = dict(payload.get("zscore") or {})
    selection_payload = dict(payload.get("lag_selection") or {})
    null_payload = dict(payload.get("circular_shift_null") or {})
    plot_payload = dict(payload.get("plots") or {})
    misc_payload = dict(payload.get("misc") or {})
    config = NeuralHazardFppSppConfig(
        fpp_risk_set_path=_resolve_config_path(
            str(input_payload["fpp_risk_set_path"]),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        spp_risk_set_path=_resolve_config_path(
            str(input_payload["spp_risk_set_path"]),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        neural_features_path=_resolve_config_path(
            str(input_payload["neural_features_path"]),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        dyads_csv_path=(
            _resolve_config_path(
                str(input_payload.get("dyads_csv_path", "config/dyads.csv")),
                project_root=project_root,
                derivatives_root=derivatives_root,
            )
            if input_payload.get("dyads_csv_path", "config/dyads.csv") is not None
            else None
        ),
        out_dir=_resolve_output_path(
            str(output_payload["out_dir"]),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        bin_width_s=float(payload.get("bin_width_s", 0.05)),
        lag_grid_ms=tuple(int(value) for value in lag_payload.get("grid_ms", [0, 50, 100, 150, 200, 300, 500])),
        pca_input_columns=tuple(str(value) for value in pca_payload.get("input_columns", [])),
        nearest_merge_tolerance_s=float(
            merge_payload.get("nearest_tolerance_s", payload.get("bin_width_s", 0.05) / 2.0)
        ),
        timing_zscore_scope=str(zscore_payload.get("timing_scope", "global")),
        neural_zscore_scope=str(zscore_payload.get("neural_scope", "subject_run")),
        pca_pc1_warning_threshold=float(pca_payload.get("pc1_warning_threshold", 0.40)),
        pca_n_components=int(pca_payload.get("n_components", 1)),
        lag_selection_criterion=str(selection_payload.get("criterion", "delta_loglik")),
        avoid_zero_lag=bool(selection_payload.get("avoid_zero_lag", True)),
        minimum_circular_shift_duration_s=float(null_payload.get("minimum_shift_duration_s", 1.0)),
        n_circular_shift_permutations=int(null_payload.get("n_permutations", 100)),
        circular_shift_seed=int(null_payload.get("seed", 12345)),
        circular_shift_interaction_test=str(
            null_payload.get("interaction_test", "one_sided_negative")
        ),
        event_triggered_window_start_s=float(plot_payload.get("event_triggered_window_start_s", -1.5)),
        event_triggered_window_end_s=float(plot_payload.get("event_triggered_window_end_s", 0.5)),
        overwrite=bool(misc_payload.get("overwrite", True)),
        verbose=bool(misc_payload.get("verbose", True)),
    )
    config.validate()
    return config


def _load_derivatives_root(paths_config_path: Path) -> Path | None:
    if not paths_config_path.exists():
        return None
    payload = yaml.safe_load(paths_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return None
    value = payload.get("derivatives_root")
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).resolve()


def _resolve_config_path(
    path_text: str,
    *,
    project_root: Path,
    derivatives_root: Path | None,
) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    if derivatives_root is not None:
        derivative_candidate = (derivatives_root / candidate).resolve()
        if derivative_candidate.exists() or str(candidate).startswith("results/"):
            return derivative_candidate
    return project_candidate


def _resolve_output_path(
    path_text: str,
    *,
    project_root: Path,
    derivatives_root: Path | None,
) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    if derivatives_root is not None and str(candidate).startswith("reports/"):
        return (derivatives_root / candidate).resolve()
    return (project_root / candidate).resolve()


def run_neural_hazard_fpp_spp_pipeline(config: NeuralHazardFppSppConfig) -> NeuralHazardFppSppResult:
    """Run the full pooled FPP-vs-SPP entropy hazard interaction pipeline."""

    config.validate()
    _configure_logging(verbose=config.verbose)
    LOGGER.info("Starting neural hazard FPP-vs-SPP pipeline.")
    LOGGER.info("FPP risk set: %s", config.fpp_risk_set_path)
    LOGGER.info("SPP risk set: %s", config.spp_risk_set_path)
    LOGGER.info("Neural features: %s", config.neural_features_path)
    LOGGER.info(
        "Lag grid (ms): %s | circular-shift permutations: %d",
        ", ".join(str(lag_ms) for lag_ms in config.lag_grid_ms),
        int(config.n_circular_shift_permutations),
    )
    output_dirs = _prepare_output_directories(config.out_dir, overwrite=config.overwrite)
    riskset = _load_and_combine_risk_sets(config)
    LOGGER.info("Loaded pooled risk set: %d rows, %d events.", int(len(riskset)), int(riskset["event_bin"].sum()))
    _write_table(
        _summarize_risk_set(riskset),
        output_dirs[TABLES_DIRNAME] / "risk_set_summary.csv",
    )
    neural = _load_neural_features(config)
    LOGGER.info("Loaded neural features: %d rows, %d columns.", int(len(neural)), int(len(neural.columns)))
    enriched = _merge_lagged_neural_features(riskset, neural, config)
    enriched = _zscore_all_features(enriched, config)
    enriched, pca_variance, pca_warnings = _add_instability_pc1_by_lag(enriched, config)
    if not pca_warnings.empty:
        _write_table(pca_warnings, output_dirs[TABLES_DIRNAME] / "pca_warnings.csv")
    lag_selection = _run_fpp_lag_selection(enriched, config)
    _write_table(lag_selection, output_dirs[TABLES_DIRNAME] / "fpp_lag_selection.csv")
    if lag_selection.empty or int(pd.to_numeric(lag_selection["n_rows"], errors="coerce").fillna(0).max()) <= 0:
        raise ValueError(
            "FPP lag selection produced no complete-case rows at any lag. "
            "The neural features table likely failed to align to the FPP risk set."
        )
    selected_lag_ms = _select_entropy_lag(lag_selection, config)
    LOGGER.info("Selected entropy lag: %d ms.", int(selected_lag_ms))
    selected_lag_table = pd.DataFrame(
        [{"selected_lag_ms": int(selected_lag_ms), "criterion": config.lag_selection_criterion}]
    )
    _write_table(selected_lag_table, output_dirs[TABLES_DIRNAME] / "selected_entropy_lag.csv")
    pooled_result = _fit_pooled_models(enriched, selected_lag_ms)
    _write_table(
        pooled_result["coefficients"]["M0_timing"],
        output_dirs[TABLES_DIRNAME] / "M0_timing_coefficients.csv",
    )
    _write_table(
        pooled_result["coefficients"]["M1_instability"],
        output_dirs[TABLES_DIRNAME] / "M1_instability_coefficients.csv",
    )
    coefficients_path = _write_table(
        pooled_result["coefficients"]["M2_entropy"],
        output_dirs[TABLES_DIRNAME] / "M2_entropy_coefficients.csv",
    )
    model_comparison_path = _write_table(
        pooled_result["model_comparison"],
        output_dirs[TABLES_DIRNAME] / "model_comparison.csv",
    )
    null_distribution, null_summary = _run_circular_shift_null(
        riskset=enriched,
        neural=neural,
        selected_lag_ms=selected_lag_ms,
        observed_models=pooled_result["models"],
        observed_model_comparison=pooled_result["pairwise"],
        config=config,
    )
    LOGGER.info(
        "Circular-shift null finished: observed delta_loglik=%.6f, permutation p=%.6f",
        float(null_summary["observed_delta_loglik"]),
        float(null_summary["permutation_p_value_delta_loglik"]),
    )
    _write_table(null_distribution, output_dirs[TABLES_DIRNAME] / "circular_shift_null.csv")
    circular_shift_summary_path = _write_table(
        pd.DataFrame([null_summary]),
        output_dirs[TABLES_DIRNAME] / "circular_shift_summary.csv",
    )
    _plot_lag_selection(lag_selection, output_dirs[FIGURES_DIRNAME] / "fpp_lag_selection_delta_loglik.png")
    _plot_pca_variance(pca_variance, output_dirs[FIGURES_DIRNAME] / "pca_explained_variance_by_lag.png")
    _plot_predicted_hazard(
        pooled_result["models"]["M2_entropy"],
        selected_lag_ms,
        output_dirs[FIGURES_DIRNAME] / "predicted_hazard_by_entropy_anchor_type.png",
    )
    _plot_null_histogram(
        null_distribution,
        pooled_result["pairwise"]["M2_entropy_vs_M1_instability"]["delta_loglik"],
        output_dirs[FIGURES_DIRNAME] / "circular_shift_null_delta_loglik.png",
    )
    _plot_event_triggered_feature(
        enriched,
        f"entropy_lag_{selected_lag_ms}ms_z",
        "Lagged entropy (z)",
        config,
        output_dirs[FIGURES_DIRNAME] / "event_triggered_entropy_fpp_vs_spp.png",
    )
    _plot_event_triggered_feature(
        enriched,
        f"instability_pc1_lag_{selected_lag_ms}ms_z",
        "Instability PC1 (z)",
        config,
        output_dirs[FIGURES_DIRNAME] / "event_triggered_instability_pc1_fpp_vs_spp.png",
    )
    summary_json_path = _write_summary_json(
        riskset=enriched,
        selected_lag_ms=selected_lag_ms,
        pca_variance=pca_variance,
        models=pooled_result["models"],
        pairwise=pooled_result["pairwise"],
        null_summary=null_summary,
        out_path=config.out_dir / "summary.json",
    )
    LOGGER.info("Finished neural hazard FPP-vs-SPP pipeline. Outputs written to %s", config.out_dir)
    return NeuralHazardFppSppResult(
        out_dir=config.out_dir,
        model_comparison_path=model_comparison_path,
        coefficients_path=coefficients_path,
        circular_shift_summary_path=circular_shift_summary_path,
        summary_json_path=summary_json_path,
    )


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    if LOGGER.handlers:
        LOGGER.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _progress(iterable, *, total: int | None = None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def _prepare_output_directories(root: Path, *, overwrite: bool) -> dict[str, Path]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {root}")
    tables_dir = root / TABLES_DIRNAME
    figures_dir = root / FIGURES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, TABLES_DIRNAME: tables_dir, FIGURES_DIRNAME: figures_dir}


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format for {path}")


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _load_and_combine_risk_sets(config: NeuralHazardFppSppConfig) -> pd.DataFrame:
    dyad_mapping = _load_dyad_speaker_mapping(config.dyads_csv_path)
    fpp = _load_table(config.fpp_risk_set_path).copy()
    spp = _load_table(config.spp_risk_set_path).copy()
    fpp = _normalize_risk_set(fpp, anchor_type="FPP", dyad_mapping=dyad_mapping)
    spp = _normalize_risk_set(spp, anchor_type="SPP", dyad_mapping=dyad_mapping)
    combined = pd.concat([fpp, spp], ignore_index=True, sort=False)
    combined["anchor_type"] = pd.Categorical(
        combined["anchor_type"].astype(str),
        categories=["FPP", "SPP"],
        ordered=True,
    )
    combined["event_bin"] = pd.to_numeric(combined["event_bin"], errors="coerce")
    if combined["event_bin"].isna().any() or not combined["event_bin"].isin([0, 1]).all():
        raise ValueError("Risk-set `event_bin` values must be 0/1.")
    combined["event_bin"] = combined["event_bin"].astype(int)
    combined["bin_center_s"] = (
        pd.to_numeric(combined["bin_start_s"], errors="coerce")
        + pd.to_numeric(combined["bin_end_s"], errors="coerce")
    ) / 2.0
    return combined.sort_values(
        ["subject_id", "run_id", "episode_id", "bin_start_s"],
        kind="mergesort",
    ).reset_index(drop=True)


DYAD_ID_PATTERN = re.compile(r"^dyad-(?P<number>\d+)$", flags=re.IGNORECASE)


def _load_dyad_speaker_mapping(dyads_csv_path: Path | None) -> dict[str, dict[str, str]]:
    """Load optional dyad speaker-to-subject mappings from CSV."""

    if dyads_csv_path is None or not dyads_csv_path.exists():
        if dyads_csv_path is not None:
            LOGGER.warning("Dyads CSV was not found; falling back to dyad-number speaker mapping: %s", dyads_csv_path)
        return {}
    dyads = pd.read_csv(dyads_csv_path)
    if not {"dyad_id", "subject_id"}.issubset(dyads.columns):
        LOGGER.warning(
            "Dyads CSV is missing `dyad_id`/`subject_id`; falling back to dyad-number speaker mapping: %s",
            dyads_csv_path,
        )
        return {}
    mapping: dict[str, dict[str, str]] = {}
    for dyad_id, frame in dyads.groupby("dyad_id", sort=False):
        subject_ids = sorted(str(value) for value in frame["subject_id"].tolist())
        if len(subject_ids) != 2:
            continue
        mapping[str(dyad_id)] = {"A": subject_ids[0], "B": subject_ids[1]}
    LOGGER.info("Loaded dyad speaker mapping for %d dyads from %s", int(len(mapping)), dyads_csv_path)
    return mapping


def _resolve_subject_id(
    *,
    dyad_id: str,
    speaker_label: str,
    dyad_mapping: dict[str, dict[str, str]],
) -> str | None:
    """Resolve canonical subject IDs from dyad/speaker labels."""

    speaker = str(speaker_label).strip().upper()
    if speaker in {"A", "B"}:
        mapped = dyad_mapping.get(str(dyad_id), {}).get(speaker)
        if mapped is not None:
            return mapped
        match = DYAD_ID_PATTERN.match(str(dyad_id).strip())
        if match is None:
            return None
        dyad_number = int(match.group("number"))
        subject_number = (2 * dyad_number) - 1 if speaker == "A" else 2 * dyad_number
        return f"sub-{subject_number:03d}"
    if speaker.startswith("SUB-"):
        return speaker.lower()
    return str(speaker_label)


def _normalize_risk_set(
    table: pd.DataFrame,
    *,
    anchor_type: str,
    dyad_mapping: dict[str, dict[str, str]],
) -> pd.DataFrame:
    working = table.copy()
    working["anchor_type"] = anchor_type
    if "time_within_run_s" not in working.columns:
        if "bin_start_s" in working.columns and "bin_end_s" in working.columns:
            working["time_within_run_s"] = (
                pd.to_numeric(working["bin_start_s"], errors="coerce")
                + pd.to_numeric(working["bin_end_s"], errors="coerce")
            ) / 2.0
        elif "bin_start_s" in working.columns:
            working["time_within_run_s"] = pd.to_numeric(working["bin_start_s"], errors="coerce")
    missing = [column for column in RISKSET_REQUIRED_COLUMNS if column not in working.columns]
    if missing:
        raise ValueError(
            f"Risk set for anchor `{anchor_type}` is missing required columns: {', '.join(missing)}"
        )
    for column in ("subject_id", "run_id", "episode_id", "dyad_id"):
        working[column] = working[column].astype(str)
    original_subject = working["subject_id"].astype(str)
    canonical_subject = [
        _resolve_subject_id(
            dyad_id=str(dyad_id),
            speaker_label=str(subject_id),
            dyad_mapping=dyad_mapping,
        )
        for dyad_id, subject_id in zip(working["dyad_id"], original_subject, strict=False)
    ]
    canonical_subject_series = pd.Series(canonical_subject, index=working.index, dtype="object")
    unresolved_mask = canonical_subject_series.isna()
    if unresolved_mask.any():
        unresolved_examples = (
            working.loc[unresolved_mask, ["dyad_id", "subject_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        LOGGER.warning(
            "Could not canonicalize %d `%s` subject IDs; keeping original values. Examples: %s",
            int(unresolved_mask.sum()),
            anchor_type,
            unresolved_examples,
        )
    working["subject_id_original"] = original_subject
    working["subject_id"] = canonical_subject_series.where(~unresolved_mask, original_subject).astype(str)
    return working


def _summarize_risk_set(riskset: pd.DataFrame) -> pd.DataFrame:
    return (
        riskset.groupby(["anchor_type", "subject_id", "run_id", "event_bin"], observed=True)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["anchor_type", "subject_id", "run_id", "event_bin"], kind="mergesort")
        .reset_index(drop=True)
    )


def _load_neural_features(config: NeuralHazardFppSppConfig) -> pd.DataFrame:
    neural = _load_table(config.neural_features_path).copy()
    missing = [column for column in NEURAL_REQUIRED_COLUMNS if column not in neural.columns]
    if missing:
        raise ValueError(
            "Neural features table is missing required columns: " + ", ".join(missing)
        )
    requested_pca_columns = tuple(
        column for column in config.pca_input_columns if column not in {"entropy", "max_state_probability"}
    )
    if requested_pca_columns:
        present_pca_columns = [column for column in requested_pca_columns if column in neural.columns]
        if not present_pca_columns:
            raise ValueError(
                "Neural features table does not contain any of the configured instability PCA columns. "
                f"Requested: {', '.join(requested_pca_columns)}. "
                f"Available columns: {', '.join(str(column) for column in neural.columns)}. "
                "The final FPP-vs-SPP pipeline requires real instability controls and will not substitute zeros."
            )
    neural["subject_id"] = neural["subject_id"].astype(str)
    neural["run_id"] = neural["run_id"].astype(str)
    neural["time_s"] = pd.to_numeric(neural["time_s"], errors="coerce")
    if not np.isfinite(neural["time_s"]).all():
        raise ValueError("Neural features table contains non-finite `time_s` values.")
    required_features = ["entropy"]
    required_features.extend(
        [column for column in config.pca_input_columns if column in neural.columns and column != "entropy"]
    )
    missing_entropy = [column for column in ["entropy"] if column not in neural.columns]
    if missing_entropy:
        raise ValueError("Neural features table must include `entropy`.")
    for column in required_features:
        neural[column] = pd.to_numeric(neural[column], errors="coerce")
    keep_columns = ["subject_id", "run_id", "time_s", *required_features]
    return neural.loc[:, list(dict.fromkeys(keep_columns))].sort_values(
        ["subject_id", "run_id", "time_s"], kind="mergesort"
    ).reset_index(drop=True)


def _merge_lagged_neural_features(
    riskset: pd.DataFrame,
    neural: pd.DataFrame,
    config: NeuralHazardFppSppConfig,
) -> pd.DataFrame:
    merged = riskset.copy()
    merge_features = ["entropy", *[column for column in config.pca_input_columns if column in neural.columns]]
    merge_features = [column for column in merge_features if column != "max_state_probability"]
    source = neural.sort_values(["subject_id", "run_id", "time_s"], kind="mergesort").reset_index(drop=True)
    LOGGER.info("Merging lagged neural features for %d lags.", int(len(config.lag_grid_ms)))
    for lag_ms in _progress(
        config.lag_grid_ms,
        total=len(config.lag_grid_ms),
        desc="Merging lags",
        enabled=config.verbose,
    ):
        lagged = _merge_groupwise_asof(
            left=merged,
            right=source,
            lookup_time_s=-(float(lag_ms) / 1000.0),
            feature_names=merge_features,
            tolerance_s=config.nearest_merge_tolerance_s,
        )
        for feature_name in merge_features:
            merged[f"{feature_name}_lag_{lag_ms}ms"] = lagged[feature_name]
    return merged


def _merge_groupwise_asof(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    lookup_time_s: float,
    feature_names: list[str],
    tolerance_s: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    right_groups = {
        (str(subject_id), str(run_id)): group.sort_values("time_s", kind="mergesort").reset_index(drop=True)
        for (subject_id, run_id), group in right.groupby(["subject_id", "run_id"], sort=False, observed=True)
    }
    for (subject_id, run_id), group in left.groupby(["subject_id", "run_id"], sort=False, observed=True):
        left_group = group.copy().sort_values("bin_center_s", kind="mergesort").reset_index()
        left_group["_lookup_time_s"] = pd.to_numeric(left_group["bin_center_s"], errors="coerce") + float(
            lookup_time_s
        )
        right_group = right_groups.get((str(subject_id), str(run_id)))
        if right_group is None or right_group.empty:
            for feature_name in feature_names:
                left_group[feature_name] = np.nan
            rows.append(left_group)
            continue
        right_group = right_group.rename(columns={"time_s": "_lookup_time_s"})
        merged_group = pd.merge_asof(
            left_group.sort_values("_lookup_time_s", kind="mergesort"),
            right_group.loc[:, ["_lookup_time_s", *feature_names]].sort_values(
                "_lookup_time_s", kind="mergesort"
            ),
            on="_lookup_time_s",
            direction="nearest",
            tolerance=tolerance_s,
        )
        rows.append(merged_group)
    if not rows:
        return pd.DataFrame(index=left.index, columns=feature_names)
    combined = pd.concat(rows, ignore_index=True, sort=False).set_index("index")
    combined = combined.reindex(left.index)
    return combined.loc[:, feature_names]


def _zscore_all_features(table: pd.DataFrame, config: NeuralHazardFppSppConfig) -> pd.DataFrame:
    out = table.copy()
    for lag_ms in config.lag_grid_ms:
        out = _add_groupwise_zscore(
            out,
            source_column=f"entropy_lag_{lag_ms}ms",
            output_column=f"entropy_lag_{lag_ms}ms_z",
            scope=config.neural_zscore_scope,
        )
        for feature_name in config.pca_input_columns:
            lagged_column = f"{feature_name}_lag_{lag_ms}ms"
            if lagged_column in out.columns:
                out = _add_groupwise_zscore(
                    out,
                    source_column=lagged_column,
                    output_column=f"{lagged_column}_z",
                    scope=config.neural_zscore_scope,
                )
    for column_name in ("time_from_partner_onset_s", "time_from_partner_offset_s", "time_within_run_s"):
        out = _add_groupwise_zscore(
            out,
            source_column=column_name,
            output_column=f"{column_name}_z",
            scope=config.timing_zscore_scope,
        )
    return out


def _add_groupwise_zscore(
    table: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    scope: str,
) -> pd.DataFrame:
    out = table.copy()
    numeric = pd.to_numeric(out[source_column], errors="coerce")
    if scope == "global":
        out[output_column] = _safe_zscore_series(numeric)
        return out
    if scope == "subject_run":
        grouped = numeric.groupby([out["subject_id"], out["run_id"]], observed=True)
        out[output_column] = grouped.transform(_safe_zscore_series)
        return out
    raise ValueError(f"Unsupported z-score scope: {scope}")


def _safe_zscore_series(values: pd.Series) -> pd.Series:
    mean = values.mean()
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0.0:
        if values.notna().any():
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return pd.Series(np.full(len(values), np.nan, dtype=float), index=values.index)
    return (values - mean) / sd


def _add_instability_pc1_by_lag(
    table: pd.DataFrame,
    config: NeuralHazardFppSppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = table.copy()
    variance_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    LOGGER.info("Computing instability PCA for %d lags.", int(len(config.lag_grid_ms)))
    for lag_ms in _progress(
        config.lag_grid_ms,
        total=len(config.lag_grid_ms),
        desc="PCA by lag",
        enabled=config.verbose,
    ):
        lag_columns = [
            f"{feature_name}_lag_{lag_ms}ms_z"
            for feature_name in config.pca_input_columns
            if feature_name != "entropy" and f"{feature_name}_lag_{lag_ms}ms_z" in out.columns
        ]
        if not lag_columns:
            raise ValueError(
                "No configured PCA input columns were available after lagging and z-scoring. "
                f"Lag {int(lag_ms)} ms requested columns derived from: {', '.join(config.pca_input_columns)}. "
                "The final FPP-vs-SPP pipeline requires real instability controls and will not substitute zeros."
            )
        fit_table = out.loc[:, lag_columns].apply(pd.to_numeric, errors="coerce")
        fit_mask = fit_table.notna().all(axis=1)
        if not fit_mask.any():
            raise ValueError(
                "No complete rows were available to fit instability PCA. "
                f"Lag {int(lag_ms)} ms columns: {', '.join(lag_columns)}."
            )
        pca = PCA(n_components=min(config.pca_n_components, len(lag_columns)))
        scores = pca.fit_transform(fit_table.loc[fit_mask, lag_columns].to_numpy(dtype=float))
        pc1_scores = np.full(len(out), np.nan, dtype=float)
        pc1_scores[fit_mask.to_numpy()] = scores[:, 0]
        out[f"instability_pc1_lag_{lag_ms}ms"] = pc1_scores
        out = _add_groupwise_zscore(
            out,
            source_column=f"instability_pc1_lag_{lag_ms}ms",
            output_column=f"instability_pc1_lag_{lag_ms}ms_z",
            scope="global",
        )
        loadings = pd.DataFrame(
            {
                "lag_ms": int(lag_ms),
                "feature": lag_columns,
                "loading_pc1": pca.components_[0, :],
            }
        )
        _write_table(loadings, config.out_dir / TABLES_DIRNAME / f"pca_loadings_lag_{lag_ms}ms.csv")
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        variance_table = pd.DataFrame(
            {
                "lag_ms": int(lag_ms),
                "component": np.arange(1, len(pca.explained_variance_ratio_) + 1, dtype=int),
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance_ratio": cumulative,
            }
        )
        _write_table(variance_table, config.out_dir / TABLES_DIRNAME / f"pca_variance_lag_{lag_ms}ms.csv")
        variance_rows.extend(variance_table.to_dict(orient="records"))
        pc1_variance = float(pca.explained_variance_ratio_[0])
        if pc1_variance < config.pca_pc1_warning_threshold:
            warning_rows.append(
                {
                    "lag_ms": int(lag_ms),
                    "warning": (
                        f"PC1 explained variance ratio {pc1_variance:.4f} was below the threshold "
                        f"{config.pca_pc1_warning_threshold:.4f}."
                    ),
                }
            )
    return out, pd.DataFrame(variance_rows), pd.DataFrame(warning_rows)


def _run_fpp_lag_selection(table: pd.DataFrame, config: NeuralHazardFppSppConfig) -> pd.DataFrame:
    fpp = table.loc[table["anchor_type"].astype(str) == "FPP"].copy()
    rows: list[dict[str, Any]] = []
    baseline_terms = " + ".join([*TIMING_Z_COLUMNS, "C(run_id)"])
    LOGGER.info("Running FPP-only lag selection across %d lags.", int(len(config.lag_grid_ms)))
    for lag_ms in _progress(
        config.lag_grid_ms,
        total=len(config.lag_grid_ms),
        desc="FPP lag selection",
        enabled=config.verbose,
    ):
        entropy_column = f"entropy_lag_{lag_ms}ms_z"
        required_columns = ["event_bin", "run_id", *TIMING_Z_COLUMNS, entropy_column]
        fit_table = _complete_case_subset(fpp, required_columns)
        model_m0 = _fit_glm(
            fit_table,
            model_name="M0_FPP",
            formula=f"event_bin ~ {baseline_terms}",
        )
        model_m1 = _fit_glm(
            fit_table,
            model_name=f"M1_FPP_{lag_ms}ms",
            formula=f"event_bin ~ {baseline_terms} + {entropy_column}",
        )
        coefficient = _extract_term_stats(model_m1, entropy_column)
        rows.append(
            {
                "lag_ms": int(lag_ms),
                "n_rows": model_m1.n_rows,
                "n_events": model_m1.n_events,
                "m0_loglik": model_m0.log_likelihood,
                "m1_loglik": model_m1.log_likelihood,
                "delta_loglik": model_m1.log_likelihood - model_m0.log_likelihood,
                "m0_aic": model_m0.aic,
                "m1_aic": model_m1.aic,
                "delta_aic": model_m0.aic - model_m1.aic,
                "entropy_beta": coefficient["estimate"],
                "entropy_se": coefficient["standard_error"],
                "entropy_z": coefficient["statistic"],
                "entropy_p": coefficient["p_value"],
            }
        )
    return pd.DataFrame(rows).sort_values(["lag_ms"], kind="mergesort").reset_index(drop=True)


def _complete_case_subset(table: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    working = table.copy()
    mask = np.ones(len(working), dtype=bool)
    for column in required_columns:
        if column in {"anchor_type", "subject_id", "run_id", "episode_id", "dyad_id"}:
            mask &= working[column].notna().to_numpy()
        else:
            mask &= np.isfinite(pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float))
    subset = working.loc[mask].copy()
    subset["event_bin"] = pd.to_numeric(subset["event_bin"], errors="coerce").astype(int)
    return subset


def _fit_glm(table: pd.DataFrame, *, model_name: str, formula: str) -> FittedGlmModel:
    if table.empty:
        return FittedGlmModel(model_name, formula, None, 0, 0, False, [], "No complete-case rows were available.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = sm.GLM.from_formula(formula=formula, data=table, family=sm.families.Binomial()).fit()
        except Exception as error:
            return FittedGlmModel(
                model_name=model_name,
                formula=formula,
                result=None,
                n_rows=int(len(table)),
                n_events=int(table["event_bin"].sum()),
                converged=False,
                warnings_text=[str(item.message) for item in caught],
                error_message=str(error),
            )
    warning_text = [str(item.message) for item in caught]
    converged = bool(getattr(result, "converged", True))
    error_message = None
    for metric_name in ("llf", "aic", "bic"):
        metric_value = float(getattr(result, metric_name, np.nan))
        if not np.isfinite(metric_value):
            converged = False
            error_message = "Model returned non-finite fit statistics."
            break
    return FittedGlmModel(
        model_name=model_name,
        formula=formula,
        result=result,
        n_rows=int(len(table)),
        n_events=int(table["event_bin"].sum()),
        converged=converged,
        warnings_text=warning_text,
        error_message=error_message,
    )


def _extract_term_stats(model: FittedGlmModel, term: str) -> dict[str, float]:
    if model.result is None or term not in model.result.params.index:
        return {"estimate": np.nan, "standard_error": np.nan, "statistic": np.nan, "p_value": np.nan}
    return {
        "estimate": float(model.result.params[term]),
        "standard_error": float(model.result.bse[term]),
        "statistic": float(model.result.tvalues[term]),
        "p_value": float(model.result.pvalues[term]),
    }


def _select_entropy_lag(lag_selection: pd.DataFrame, config: NeuralHazardFppSppConfig) -> int:
    working = lag_selection.copy()
    if config.lag_selection_criterion == "delta_loglik":
        chosen = working.sort_values(["delta_loglik", "lag_ms"], ascending=[False, True], kind="mergesort")
    else:
        chosen = working.sort_values(["delta_aic", "lag_ms"], ascending=[False, True], kind="mergesort")
    selected = int(chosen.iloc[0]["lag_ms"])
    if selected == 0 and config.avoid_zero_lag:
        nonzero = chosen.loc[pd.to_numeric(chosen["lag_ms"], errors="coerce") != 0]
        if not nonzero.empty:
            selected = int(nonzero.iloc[0]["lag_ms"])
    return selected


def _fit_pooled_models(table: pd.DataFrame, selected_lag_ms: int) -> dict[str, Any]:
    entropy_column = f"entropy_lag_{selected_lag_ms}ms_z"
    instability_column = f"instability_pc1_lag_{selected_lag_ms}ms_z"
    anchor_term = "C(anchor_type)"
    base_formula = (
        "event_bin ~ "
        f"{anchor_term}"
        " + time_from_partner_onset_s_z"
        " + time_from_partner_offset_s_z"
        f" + {anchor_term}:time_from_partner_onset_s_z"
        f" + {anchor_term}:time_from_partner_offset_s_z"
        " + time_within_run_s_z"
        " + C(run_id)"
    )
    formulas = {
        "M0_timing": base_formula,
        "M1_instability": (
            f"{base_formula} + {instability_column} + {anchor_term}:{instability_column}"
        ),
        "M2_entropy": (
            f"{base_formula} + {instability_column} + {anchor_term}:{instability_column}"
            f" + {entropy_column} + {anchor_term}:{entropy_column}"
        ),
    }
    required_columns = [
        "event_bin",
        "anchor_type",
        "run_id",
        *TIMING_Z_COLUMNS,
        instability_column,
        entropy_column,
    ]
    fit_table = _complete_case_subset(table, required_columns)
    LOGGER.info("Fitting pooled models on %d complete-case rows.", int(len(fit_table)))
    if fit_table.empty:
        available_subject_run = (
            table.loc[:, ["subject_id", "run_id"]]
            .drop_duplicates()
            .sort_values(["subject_id", "run_id"], kind="mergesort")
        )
        observed_anchor_counts = (
            table.groupby("anchor_type", observed=True)["event_bin"]
            .agg(n_rows="size", n_events="sum")
            .reset_index()
        )
        raise ValueError(
            "No complete-case rows were available for the pooled M0/M1/M2 models after neural merging. "
            f"Observed subject/run combinations: {len(available_subject_run)}. "
            f"Anchor counts: {observed_anchor_counts.to_dict(orient='records')}. "
            "This usually means the neural features table did not align to the risk-set subject_id/run_id/time axes."
        )
    models = {name: _fit_glm(fit_table, model_name=name, formula=formula) for name, formula in formulas.items()}
    coefficients = {name: _coefficient_table(model) for name, model in models.items()}
    pairwise = {
        "M1_instability_vs_M0_timing": _compare_nested(models["M1_instability"], models["M0_timing"]),
        "M2_entropy_vs_M1_instability": _compare_nested(models["M2_entropy"], models["M1_instability"]),
    }
    model_rows = [_model_row(model) for model in models.values()]
    comparison_rows = [
        {"row_type": "comparison", "model_name": key, **value}
        for key, value in pairwise.items()
    ]
    comparison_table = pd.DataFrame([*model_rows, *comparison_rows])
    return {
        "models": models,
        "coefficients": coefficients,
        "pairwise": pairwise,
        "model_comparison": comparison_table,
    }


def _coefficient_table(model: FittedGlmModel) -> pd.DataFrame:
    if model.result is None:
        return pd.DataFrame(
            columns=[
                "model_name",
                "term",
                "estimate",
                "standard_error",
                "statistic",
                "p_value",
                "conf_low",
                "conf_high",
                "odds_ratio",
                "odds_ratio_conf_low",
                "odds_ratio_conf_high",
            ]
        )
    conf = model.result.conf_int()
    table = pd.DataFrame(
        {
            "model_name": model.model_name,
            "term": model.result.params.index.astype(str),
            "estimate": np.asarray(model.result.params, dtype=float),
            "standard_error": np.asarray(model.result.bse, dtype=float),
            "statistic": np.asarray(model.result.tvalues, dtype=float),
            "p_value": np.asarray(model.result.pvalues, dtype=float),
            "conf_low": np.asarray(conf.iloc[:, 0], dtype=float),
            "conf_high": np.asarray(conf.iloc[:, 1], dtype=float),
        }
    )
    table["odds_ratio"] = np.exp(table["estimate"])
    table["odds_ratio_conf_low"] = np.exp(table["conf_low"])
    table["odds_ratio_conf_high"] = np.exp(table["conf_high"])
    return table


def _model_row(model: FittedGlmModel) -> dict[str, Any]:
    return {
        "row_type": "model",
        "model_name": model.model_name,
        "n_rows": model.n_rows,
        "n_events": model.n_events,
        "log_likelihood": model.log_likelihood,
        "aic": model.aic,
        "bic": model.bic,
        "converged": model.converged,
        "warnings": " | ".join([*model.warnings_text, *(["ERROR: " + model.error_message] if model.error_message else [])]),
    }


def _compare_nested(child: FittedGlmModel, parent: FittedGlmModel) -> dict[str, Any]:
    delta_loglik = child.log_likelihood - parent.log_likelihood
    lrt = 2.0 * delta_loglik
    df_difference = max(child.df_model - parent.df_model, 0)
    p_value = float(stats.chi2.sf(lrt, df_difference)) if df_difference > 0 and np.isfinite(lrt) else np.nan
    return {
        "delta_loglik": delta_loglik,
        "likelihood_ratio_statistic": lrt,
        "df_difference": df_difference,
        "p_value": p_value,
        "delta_aic": parent.aic - child.aic,
        "delta_bic": parent.bic - child.bic,
        "n_rows": child.n_rows,
        "n_events": child.n_events,
        "log_likelihood": child.log_likelihood,
        "aic": child.aic,
        "bic": child.bic,
        "converged": child.converged and parent.converged,
        "warnings": " | ".join([*parent.warnings_text, *child.warnings_text]),
    }


def _run_circular_shift_null(
    *,
    riskset: pd.DataFrame,
    neural: pd.DataFrame,
    selected_lag_ms: int,
    observed_models: dict[str, FittedGlmModel],
    observed_model_comparison: dict[str, dict[str, Any]],
    config: NeuralHazardFppSppConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(config.circular_shift_seed)
    rows: list[dict[str, Any]] = []
    observed_entropy_term = f"entropy_lag_{selected_lag_ms}ms_z"
    interaction_term = f"C(anchor_type)[T.SPP]:{observed_entropy_term}"
    observed_m2 = observed_models["M2_entropy"]
    observed_delta = float(observed_model_comparison["M2_entropy_vs_M1_instability"]["delta_loglik"])
    observed_interaction = _extract_term_stats(observed_m2, interaction_term)["estimate"]
    observed_entropy_beta = _extract_term_stats(observed_m2, observed_entropy_term)["estimate"]
    LOGGER.info(
        "Running circular-shift null with %d permutations at selected lag %d ms.",
        int(config.n_circular_shift_permutations),
        int(selected_lag_ms),
    )
    for permutation_id in _progress(
        range(config.n_circular_shift_permutations),
        total=config.n_circular_shift_permutations,
        desc="Circular-shift null",
        enabled=config.verbose and config.n_circular_shift_permutations > 1,
    ):
        shifted_neural = _circular_shift_entropy(neural, config=config, rng=rng)
        shifted = _merge_selected_entropy_shift(riskset, shifted_neural, selected_lag_ms, config)
        shifted = _add_groupwise_zscore(
            shifted,
            source_column=f"entropy_lag_{selected_lag_ms}ms",
            output_column=f"entropy_lag_{selected_lag_ms}ms_z",
            scope=config.neural_zscore_scope,
        )
        pair = _fit_shifted_entropy_models(shifted, selected_lag_ms)
        m2_model = pair["models"]["M2_entropy_shifted"]
        rows.append(
            {
                "permutation_id": int(permutation_id),
                "selected_lag_ms": int(selected_lag_ms),
                "m1_loglik": pair["models"]["M1_instability"].log_likelihood,
                "m2_loglik": m2_model.log_likelihood,
                "delta_loglik": pair["comparison"]["delta_loglik"],
                "entropy_beta_fpp": _extract_term_stats(m2_model, observed_entropy_term)["estimate"],
                "entropy_interaction_beta": _extract_term_stats(m2_model, interaction_term)["estimate"],
                "converged": pair["models"]["M1_instability"].converged and m2_model.converged,
                "warnings": " | ".join(
                    [*pair["models"]["M1_instability"].warnings_text, *m2_model.warnings_text]
                ),
            }
        )
    null_table = pd.DataFrame(rows)
    null_delta = pd.to_numeric(null_table["delta_loglik"], errors="coerce").to_numpy(dtype=float)
    null_interaction = pd.to_numeric(null_table["entropy_interaction_beta"], errors="coerce").to_numpy(dtype=float)
    summary = {
        "observed_delta_loglik": observed_delta,
        "null_mean_delta_loglik": float(np.nanmean(null_delta)) if len(null_delta) else np.nan,
        "null_sd_delta_loglik": float(np.nanstd(null_delta, ddof=0)) if len(null_delta) else np.nan,
        "permutation_p_value_delta_loglik": _upper_tail_p_value(observed_delta, null_delta),
        "observed_entropy_interaction_beta": observed_interaction,
        "null_mean_entropy_interaction_beta": float(np.nanmean(null_interaction)) if len(null_interaction) else np.nan,
        "null_sd_entropy_interaction_beta": float(np.nanstd(null_interaction, ddof=0)) if len(null_interaction) else np.nan,
        "permutation_p_value_interaction_beta": _interaction_p_value(
            observed_interaction,
            null_interaction,
            test_mode=config.circular_shift_interaction_test,
        ),
        "observed_entropy_beta_fpp": observed_entropy_beta,
    }
    return null_table, summary


def _circular_shift_entropy(
    neural: pd.DataFrame,
    *,
    config: NeuralHazardFppSppConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    shifted = neural.copy()
    shifted = shifted.sort_values(["subject_id", "run_id", "time_s"], kind="mergesort").reset_index(drop=True)
    shifted_entropy = np.full(len(shifted), np.nan, dtype=float)
    for _, group in shifted.groupby(["subject_id", "run_id"], sort=False, observed=True):
        values = pd.to_numeric(group["entropy"], errors="coerce").to_numpy(dtype=float, copy=True)
        finite = np.isfinite(values)
        if finite.sum() <= 1:
            shifted_entropy[group.index.to_numpy()] = values
            continue
        group_times = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(dtype=float)
        diffs = np.diff(group_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        median_step = float(np.median(diffs)) if diffs.size else config.bin_width_s
        min_steps = max(1, int(math.ceil(config.minimum_circular_shift_duration_s / max(median_step, 1.0e-9))))
        valid_offsets = np.arange(min_steps, len(values), dtype=int)
        if valid_offsets.size == 0:
            shifted_entropy[group.index.to_numpy()] = values
            continue
        offset = int(rng.choice(valid_offsets))
        shifted_entropy[group.index.to_numpy()] = np.roll(values, offset)
    shifted["entropy"] = shifted_entropy
    return shifted


def _merge_selected_entropy_shift(
    riskset: pd.DataFrame,
    shifted_neural: pd.DataFrame,
    selected_lag_ms: int,
    config: NeuralHazardFppSppConfig,
) -> pd.DataFrame:
    merged = riskset.copy()
    lagged = _merge_groupwise_asof(
        left=merged,
        right=shifted_neural,
        lookup_time_s=-(float(selected_lag_ms) / 1000.0),
        feature_names=["entropy"],
        tolerance_s=config.nearest_merge_tolerance_s,
    )
    merged[f"entropy_lag_{selected_lag_ms}ms"] = pd.to_numeric(lagged["entropy"], errors="coerce")
    return merged


def _fit_shifted_entropy_models(table: pd.DataFrame, selected_lag_ms: int) -> dict[str, Any]:
    entropy_column = f"entropy_lag_{selected_lag_ms}ms_z"
    instability_column = f"instability_pc1_lag_{selected_lag_ms}ms_z"
    anchor_term = "C(anchor_type)"
    m1_formula = (
        "event_bin ~ "
        f"{anchor_term}"
        " + time_from_partner_onset_s_z"
        " + time_from_partner_offset_s_z"
        f" + {anchor_term}:time_from_partner_onset_s_z"
        f" + {anchor_term}:time_from_partner_offset_s_z"
        " + time_within_run_s_z"
        " + C(run_id)"
        f" + {instability_column} + {anchor_term}:{instability_column}"
    )
    m2_formula = f"{m1_formula} + {entropy_column} + {anchor_term}:{entropy_column}"
    required_columns = [
        "event_bin",
        "anchor_type",
        "run_id",
        *TIMING_Z_COLUMNS,
        instability_column,
        entropy_column,
    ]
    fit_table = _complete_case_subset(table, required_columns)
    m1 = _fit_glm(fit_table, model_name="M1_instability", formula=m1_formula)
    m2 = _fit_glm(fit_table, model_name="M2_entropy_shifted", formula=m2_formula)
    return {"models": {"M1_instability": m1, "M2_entropy_shifted": m2}, "comparison": _compare_nested(m2, m1)}


def _upper_tail_p_value(observed: float, null_values: np.ndarray) -> float:
    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0:
        return float("nan")
    return float((1 + np.count_nonzero(finite >= observed)) / (1 + finite.size))


def _interaction_p_value(observed: float, null_values: np.ndarray, *, test_mode: str) -> float:
    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0:
        return float("nan")
    if test_mode == "one_sided_negative":
        return float((1 + np.count_nonzero(finite <= observed)) / (1 + finite.size))
    return float((1 + np.count_nonzero(np.abs(finite) >= abs(observed))) / (1 + finite.size))


def _plot_lag_selection(table: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.8, 4.4))
    plt.plot(table["lag_ms"], table["delta_loglik"], marker="o")
    plt.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    plt.xlabel("Lag (ms)")
    plt.ylabel("Delta log-likelihood")
    plt.title("FPP entropy lag selection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_pca_variance(table: pd.DataFrame, out_path: Path) -> None:
    if "component" not in table.columns:
        plotted = pd.DataFrame()
    else:
        plotted = table.loc[pd.to_numeric(table["component"], errors="coerce") == 1].copy()
    plt.figure(figsize=(6.8, 4.4))
    if plotted.empty:
        plt.text(0.5, 0.5, "No PCA variance rows available.", ha="center", va="center")
        plt.gca().set_axis_off()
    else:
        plt.plot(plotted["lag_ms"], plotted["explained_variance_ratio"], marker="o")
        plt.xlabel("Lag (ms)")
        plt.ylabel("PC1 explained variance ratio")
        plt.title("Instability PCA variance by lag")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_predicted_hazard(model: FittedGlmModel, selected_lag_ms: int, out_path: Path) -> None:
    entropy_column = f"entropy_lag_{selected_lag_ms}ms_z"
    instability_column = f"instability_pc1_lag_{selected_lag_ms}ms_z"
    entropy_grid = np.linspace(-2.5, 2.5, 101)
    reference_run_id = "1"
    if model.result is not None:
        model_frame = getattr(getattr(model.result, "model", None), "data", None)
        frame = getattr(model_frame, "frame", None)
        if isinstance(frame, pd.DataFrame) and "run_id" in frame.columns and not frame.empty:
            reference_run_id = str(frame["run_id"].iloc[0])
    rows: list[pd.DataFrame] = []
    for anchor in ("FPP", "SPP"):
        rows.append(
            pd.DataFrame(
                {
                    "anchor_type": anchor,
                    "time_from_partner_onset_s_z": 0.0,
                    "time_from_partner_offset_s_z": 0.0,
                    "time_within_run_s_z": 0.0,
                    "run_id": reference_run_id,
                    instability_column: 0.0,
                    entropy_column: entropy_grid,
                }
            )
        )
    prediction_table = pd.concat(rows, ignore_index=True)
    prediction_table["anchor_type"] = pd.Categorical(
        prediction_table["anchor_type"],
        categories=["FPP", "SPP"],
        ordered=True,
    )
    predicted = np.asarray(model.result.predict(prediction_table), dtype=float) if model.result is not None else np.full(len(prediction_table), np.nan)
    prediction_table["predicted_probability"] = predicted
    plt.figure(figsize=(6.8, 4.4))
    for anchor in ("FPP", "SPP"):
        subset = prediction_table.loc[prediction_table["anchor_type"].astype(str) == anchor]
        plt.plot(subset[entropy_column], subset["predicted_probability"], label=anchor)
    plt.xlabel("Lagged entropy (z)")
    plt.ylabel("Predicted event probability")
    plt.title("Predicted hazard by entropy and anchor type")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_null_histogram(null_table: pd.DataFrame, observed_delta_loglik: float, out_path: Path) -> None:
    values = pd.to_numeric(null_table["delta_loglik"], errors="coerce")
    plt.figure(figsize=(6.8, 4.4))
    plt.hist(values.dropna(), bins=20)
    plt.axvline(observed_delta_loglik, color="tab:red", linestyle="--", linewidth=1.5)
    plt.xlabel("Delta log-likelihood")
    plt.ylabel("Count")
    plt.title("Circular-shift null for entropy contribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_event_triggered_feature(
    table: pd.DataFrame,
    feature_column: str,
    ylabel: str,
    config: NeuralHazardFppSppConfig,
    out_path: Path,
) -> None:
    aligned = _build_event_triggered_table(table, feature_column, config)
    plt.figure(figsize=(7.0, 4.4))
    if aligned.empty:
        plt.text(0.5, 0.5, "No event-aligned rows available.", ha="center", va="center")
        plt.gca().set_axis_off()
    else:
        for anchor in ("FPP", "SPP"):
            subset = aligned.loc[aligned["anchor_type"].astype(str) == anchor].copy()
            if subset.empty:
                continue
            aggregated = (
                subset.groupby("relative_time_s", observed=True)[feature_column]
                .mean()
                .reset_index()
                .sort_values("relative_time_s", kind="mergesort")
            )
            plt.plot(aggregated["relative_time_s"], aggregated[feature_column], label=anchor)
        plt.axvline(0.0, color="0.3", linestyle="--", linewidth=1.0)
        plt.xlabel("Time from event onset (s)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " aligned to event onset")
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _build_event_triggered_table(
    table: pd.DataFrame,
    feature_column: str,
    config: NeuralHazardFppSppConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, episode in table.groupby("episode_id", sort=False, observed=True):
        event_rows = episode.loc[pd.to_numeric(episode["event_bin"], errors="coerce") == 1].copy()
        if event_rows.empty:
            continue
        event_time = float(event_rows["bin_center_s"].iloc[0])
        aligned = episode.loc[:, ["anchor_type", "episode_id", "bin_center_s", feature_column]].copy()
        aligned["relative_time_s"] = pd.to_numeric(aligned["bin_center_s"], errors="coerce") - event_time
        aligned = aligned.loc[
            (aligned["relative_time_s"] >= config.event_triggered_window_start_s)
            & (aligned["relative_time_s"] <= config.event_triggered_window_end_s)
        ].copy()
        rows.append(aligned)
    if not rows:
        return pd.DataFrame(columns=["anchor_type", "episode_id", "relative_time_s", feature_column])
    return pd.concat(rows, ignore_index=True, sort=False)


def _write_summary_json(
    *,
    riskset: pd.DataFrame,
    selected_lag_ms: int,
    pca_variance: pd.DataFrame,
    models: dict[str, FittedGlmModel],
    pairwise: dict[str, dict[str, Any]],
    null_summary: dict[str, Any],
    out_path: Path,
) -> Path:
    m2 = models["M2_entropy"]
    entropy_column = f"entropy_lag_{selected_lag_ms}ms_z"
    interaction_term = f"C(anchor_type)[T.SPP]:{entropy_column}"
    if {"component", "lag_ms", "explained_variance_ratio"}.issubset(pca_variance.columns):
        selected_pc1 = pca_variance.loc[pd.to_numeric(pca_variance["component"], errors="coerce") == 1].copy()
        selected_pc1 = selected_pc1.loc[
            pd.to_numeric(selected_pc1["lag_ms"], errors="coerce") == int(selected_lag_ms)
        ]
    else:
        selected_pc1 = pd.DataFrame()
    payload = {
        "selected_lag_ms": int(selected_lag_ms),
        "n_rows": int(len(riskset)),
        "n_events_total": int(riskset["event_bin"].sum()),
        "n_events_fpp": int(riskset.loc[riskset["anchor_type"].astype(str) == "FPP", "event_bin"].sum()),
        "n_events_spp": int(riskset.loc[riskset["anchor_type"].astype(str) == "SPP", "event_bin"].sum()),
        "pca_pc1_explained_variance_selected_lag": (
            float(selected_pc1["explained_variance_ratio"].iloc[0]) if not selected_pc1.empty else np.nan
        ),
        "entropy_beta_fpp": _extract_term_stats(m2, entropy_column)["estimate"],
        "entropy_interaction_beta_spp_minus_fpp": _extract_term_stats(m2, interaction_term)["estimate"],
        "m2_vs_m1_delta_loglik": float(pairwise["M2_entropy_vs_M1_instability"]["delta_loglik"]),
        "m2_vs_m1_lrt_p_value": float(pairwise["M2_entropy_vs_M1_instability"]["p_value"]),
        "circular_shift_p_value_delta_loglik": float(null_summary["permutation_p_value_delta_loglik"]),
        "circular_shift_p_value_interaction_beta": float(
            null_summary["permutation_p_value_interaction_beta"]
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
