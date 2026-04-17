"""IO helpers for reusing existing TDE-HMM outputs and writing hazard artifacts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cas.hazard.config import HazardAnalysisConfig, config_to_metadata_dict
from cas.hazard.entropy import compute_posterior_entropy, normalize_entropy

LOGGER = logging.getLogger(__name__)
JSON_INDENT = 2
ENTROPY_FILE_PATTERN = re.compile(r"^subject-(?P<subject>.+?)_run-(?P<run>.+?)_state_entropy\.csv$")


@dataclass(frozen=True, slots=True)
class HmmReuseInputs:
    """Existing TDE-HMM outputs reused by the hazard analysis."""

    output_dir: Path
    selected_k: int
    gamma: np.ndarray
    entropy_processed: np.ndarray
    entropy_processed_raw: np.ndarray
    fit_summary: dict[str, Any]
    chunk_table: pd.DataFrame
    input_manifest: pd.DataFrame
    entropy_by_run: dict[tuple[str, str], pd.DataFrame]
    sampling_rate_hz: float
    entropy_time_step_seconds: float
    alignment_strategy: str
    alignment_assumptions: list[str]


def load_reused_tde_hmm_outputs(config: HazardAnalysisConfig) -> HmmReuseInputs:
    """Load existing TDE-HMM outputs without triggering any refit."""

    output_dir = config.input.tde_hmm_results_dir.resolve()
    fit_summary_path = output_dir / "fit_summary.json"
    chunk_table_path = output_dir / "chunks.csv"
    input_manifest_path = output_dir / "input_manifest.csv"

    if not output_dir.exists():
        raise FileNotFoundError(f"TDE-HMM results directory does not exist: {output_dir}")
    if not fit_summary_path.exists():
        raise FileNotFoundError(f"Missing existing TDE-HMM summary: {fit_summary_path}")
    if not chunk_table_path.exists():
        raise FileNotFoundError(f"Missing existing TDE-HMM chunk table: {chunk_table_path}")
    if not input_manifest_path.exists():
        raise FileNotFoundError(f"Missing existing TDE-HMM input manifest: {input_manifest_path}")

    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    selected_k = int(fit_summary["selected_k"])
    gamma_path = output_dir / f"gamma_k{selected_k}.npy"
    if not gamma_path.exists():
        raise FileNotFoundError(f"Missing existing posterior probabilities: {gamma_path}")

    gamma = np.asarray(np.load(gamma_path), dtype=float)
    if gamma.ndim != 2:
        raise ValueError("Saved gamma must be 2D.")
    if gamma.shape[1] != selected_k:
        raise ValueError("Saved gamma column count does not match selected_k.")

    entropy_processed_raw = compute_posterior_entropy(gamma, epsilon=config.entropy.epsilon)
    entropy_processed = (
        normalize_entropy(entropy_processed_raw, selected_k)
        if config.entropy.normalize_by_log_k
        else entropy_processed_raw.copy()
    )

    chunk_table = pd.read_csv(chunk_table_path)
    input_manifest = pd.read_csv(input_manifest_path)
    sampling_rate_hz = float(fit_summary["sampling_rate_hz"])
    entropy_by_run, alignment_strategy, alignment_assumptions = _load_or_reconstruct_entropy_timelines(
        output_dir=output_dir,
        chunk_table=chunk_table,
        entropy_processed=entropy_processed,
        input_manifest=input_manifest,
        fit_summary=fit_summary,
    )
    entropy_time_step_seconds = _infer_entropy_time_step_seconds(
        entropy_by_run=entropy_by_run,
        sampling_rate_hz=sampling_rate_hz,
    )

    LOGGER.info("Reusing existing TDE-HMM outputs from %s", output_dir)
    LOGGER.info(
        "Loaded posterior probabilities with K=%d and %d processed samples",
        selected_k,
        gamma.shape[0],
    )
    LOGGER.info("Inferred entropy time step %.6f s", entropy_time_step_seconds)

    return HmmReuseInputs(
        output_dir=output_dir,
        selected_k=selected_k,
        gamma=gamma,
        entropy_processed=entropy_processed,
        entropy_processed_raw=entropy_processed_raw,
        fit_summary=fit_summary,
        chunk_table=chunk_table,
        input_manifest=input_manifest,
        entropy_by_run=entropy_by_run,
        sampling_rate_hz=sampling_rate_hz,
        entropy_time_step_seconds=entropy_time_step_seconds,
        alignment_strategy=alignment_strategy,
        alignment_assumptions=alignment_assumptions,
    )


def load_events_table(events_table_path: Path) -> pd.DataFrame:
    """Load the existing canonical paired events table."""

    resolved_path = events_table_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Events table does not exist: {resolved_path}")
    events_table = pd.read_csv(resolved_path)
    if events_table.empty:
        raise ValueError(f"Events table is empty: {resolved_path}")
    return events_table


def load_pairing_issues_table(pairing_issues_table_path: Path | None) -> pd.DataFrame | None:
    """Load the existing pairing-issues table when available."""

    if pairing_issues_table_path is None:
        return None
    resolved_path = pairing_issues_table_path.resolve()
    if not resolved_path.exists():
        LOGGER.warning("Configured pairing-issues CSV is missing: %s", resolved_path)
        return None
    issues_table = pd.read_csv(resolved_path)
    return issues_table if not issues_table.empty else None


def load_dyad_table(dyads_csv_path: Path | None) -> pd.DataFrame | None:
    """Load the optional dyad table used for event-to-subject mapping."""

    if dyads_csv_path is None:
        return None
    resolved_path = dyads_csv_path.resolve()
    if not resolved_path.exists():
        LOGGER.warning("Configured dyads CSV is missing; falling back to inferred dyad mapping: %s", resolved_path)
        return None
    return pd.read_csv(resolved_path)


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    """Create the output directory conservatively."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files and overwrite is false: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_hazard_table(hazard_table: pd.DataFrame, output_dir: Path) -> Path:
    """Write the person-period hazard table."""

    output_path = output_dir / "hazard_table.csv"
    hazard_table.to_csv(output_path, index=False)
    return output_path


def write_entropy_alignment_table(aligned_entropy_table: pd.DataFrame, output_dir: Path) -> Path:
    """Write entropy aligned to partner onset."""

    output_path = output_dir / "entropy_aligned_to_partner_onset.csv"
    aligned_entropy_table.to_csv(output_path, index=False)
    return output_path


def write_coefficients_table(coefficients: pd.DataFrame, output_dir: Path) -> Path:
    """Write model coefficients."""

    output_path = output_dir / "coefficients.csv"
    coefficients.to_csv(output_path, index=False)
    return output_path


def write_model_summary(summary_text: str, output_dir: Path) -> Path:
    """Write the model summary text."""

    output_path = output_dir / "model_summary.txt"
    output_path.write_text(summary_text.rstrip() + "\n", encoding="utf-8")
    return output_path


def write_json_payload(payload: dict[str, Any], output_path: Path) -> Path:
    """Write a JSON artifact with stable formatting."""

    output_path.write_text(
        json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_note_file(note_text: str, output_path: Path) -> Path:
    """Write a plain-text note file."""

    output_path.write_text(note_text.rstrip() + "\n", encoding="utf-8")
    return output_path


def build_analysis_metadata(
    *,
    config: HazardAnalysisConfig,
    hmm_inputs: HmmReuseInputs,
    event_table_columns: list[str],
    pairing_issues_columns: list[str] | None,
    warnings: list[str],
    subject_mapping_assumptions: list[str],
) -> dict[str, Any]:
    """Build a transparent metadata summary for the analysis."""

    lags = list(hmm_inputs.fit_summary.get("lags", []))
    lag_span_samples = int(max(lags) - min(lags)) if lags else None
    return {
        "config": config_to_metadata_dict(config),
        "reused_tde_hmm_outputs": {
            "output_dir": str(hmm_inputs.output_dir),
            "selected_k": hmm_inputs.selected_k,
            "sampling_rate_hz": hmm_inputs.sampling_rate_hz,
            "entropy_time_step_seconds": hmm_inputs.entropy_time_step_seconds,
            "processed_samples": int(hmm_inputs.gamma.shape[0]),
            "lags": lags,
            "lag_span_samples": lag_span_samples,
            "alignment_strategy": hmm_inputs.alignment_strategy,
            "alignment_assumptions": hmm_inputs.alignment_assumptions,
            "causal_right_edge_alignment_assumed": True,
        },
        "events_table": {
            "path": str(config.input.events_table_path),
            "columns": event_table_columns,
        },
        "pairing_issues_table": {
            "path": (
                None
                if config.input.pairing_issues_table_path is None
                else str(config.input.pairing_issues_table_path)
            ),
            "columns": pairing_issues_columns,
        },
        "subject_mapping_assumptions": subject_mapping_assumptions,
        "hazard_clock": "time since partner onset",
        "warnings": warnings,
    }


def _load_or_reconstruct_entropy_timelines(
    *,
    output_dir: Path,
    chunk_table: pd.DataFrame,
    entropy_processed: np.ndarray,
    input_manifest: pd.DataFrame,
    fit_summary: dict[str, Any],
) -> tuple[dict[tuple[str, str], pd.DataFrame], str, list[str]]:
    """Load per-run entropy CSVs, or reconstruct them conservatively if needed."""

    entropy_by_run = _collect_entropy_csvs(output_dir=output_dir)
    if entropy_by_run:
        return entropy_by_run, "per_run_entropy_csv", [
            "Used existing per-run state entropy CSV exports for run-level time alignment.",
            "Per-run entropy CSVs were generated from the existing TDE-HMM fit outputs already on disk.",
            "Causal right-edge alignment is assumed from the saved TDE-HMM metadata and implementation.",
        ]

    reconstructed = _reconstruct_entropy_timelines_from_chunks(
        chunk_table=chunk_table,
        entropy_processed=entropy_processed,
        input_manifest=input_manifest,
        fit_summary=fit_summary,
    )
    return reconstructed, "reconstructed_from_chunks", [
        "Per-run entropy CSVs were missing, so run-level timelines were reconstructed from chunks.csv and fit_summary.json.",
        "Reconstruction assumes contiguous causal lags ending at 0 and right-edge alignment to the original sample index.",
    ]


def _collect_entropy_csvs(output_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Collect existing run-level entropy CSV exports."""

    entropy_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    for path in sorted(output_dir.glob("subject-*_run-*_state_entropy.csv")):
        match = ENTROPY_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        required_columns = {"sample", "time_s", "state_entropy"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"Entropy CSV is missing required columns: {path}")
        subject_id = str(match.group("subject"))
        run_id = str(match.group("run"))
        entropy_by_run[(subject_id, run_id)] = frame.sort_values("time_s").reset_index(drop=True)
    return entropy_by_run


def _reconstruct_entropy_timelines_from_chunks(
    *,
    chunk_table: pd.DataFrame,
    entropy_processed: np.ndarray,
    input_manifest: pd.DataFrame,
    fit_summary: dict[str, Any],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Reconstruct run-level entropy timelines from chunk metadata."""

    lags = [int(value) for value in fit_summary.get("lags", [])]
    if not lags:
        raise ValueError("Cannot reconstruct run-level entropy without saved lags.")
    if max(lags) != 0:
        raise ValueError("Fallback reconstruction requires lags ending at 0.")
    if lags != list(range(min(lags), max(lags) + 1)):
        raise ValueError("Fallback reconstruction requires contiguous lags.")

    lag_span_samples = int(max(lags) - min(lags))
    sampling_rate_hz = float(fit_summary["sampling_rate_hz"])
    entropy_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    for manifest_row in input_manifest.to_dict("records"):
        subject_id = str(manifest_row["subject"])
        run_id = str(manifest_row["run"])
        run_chunks = chunk_table.loc[
            (chunk_table["subject"].astype(str) == subject_id)
            & (chunk_table["run"].astype(str) == run_id)
        ].sort_values("original_start_sample", kind="mergesort")
        if run_chunks.empty:
            continue
        original_stop_sample = int(run_chunks["original_stop_sample"].max())
        entropy_timeline = np.full(original_stop_sample, np.nan, dtype=float)
        for chunk in run_chunks.to_dict("records"):
            processed_start = int(chunk["processed_start_sample"])
            processed_stop = int(chunk["processed_stop_sample"])
            original_start = int(chunk["original_start_sample"])
            original_stop = int(chunk["original_stop_sample"])
            processed_values = entropy_processed[processed_start:processed_stop]
            mapped_original_start = original_start + lag_span_samples
            mapped_original_stop = mapped_original_start + processed_values.shape[0]
            if mapped_original_stop > original_stop:
                raise ValueError("Chunk reconstruction exceeded the original chunk bounds.")
            entropy_timeline[mapped_original_start:mapped_original_stop] = processed_values
        frame = pd.DataFrame(
            {
                "sample": np.arange(entropy_timeline.shape[0], dtype=int),
                "time_s": np.arange(entropy_timeline.shape[0], dtype=float) / sampling_rate_hz,
                "state_entropy": entropy_timeline,
            }
        )
        entropy_by_run[(subject_id, run_id)] = frame
    return entropy_by_run


def _infer_entropy_time_step_seconds(
    *,
    entropy_by_run: dict[tuple[str, str], pd.DataFrame],
    sampling_rate_hz: float,
) -> float:
    """Infer the entropy time step from saved run-level timelines."""

    for frame in entropy_by_run.values():
        diffs = np.diff(frame["time_s"].to_numpy(dtype=float))
        finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if finite_diffs.size > 0:
            return float(np.median(finite_diffs))
    return 1.0 / sampling_rate_hz
