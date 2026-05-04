"""Time-resolved lagged induced-power lmeEEG bridge analysis.

This module implements a causal lagged bridge between behavioural information
rate dynamics and pre-action induced alpha/beta power using one mixed model per
band x neural time bin x information-predictor bin.

Usage example
-------------
>>> from pathlib import Path
>>> cfg = load_info_rate_induced_lmeeg_config(Path("config/induced/info_rate_induced_lmeeg.yaml"))  # doctest: +SKIP
>>> result = run_info_rate_induced_lmeeg_pipeline(cfg)  # doctest: +SKIP
>>> result.model_results_path.exists()  # doctest: +SKIP
True
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any
import warnings

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from tqdm.auto import tqdm
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from cas.source_dics.io import discover_epoch_records, configure_mne_runtime, load_epochs

LOGGER = logging.getLogger(__name__)

MODEL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "induced_power_z",
    "anchor_type",
    "information_rate_bin_z",
    "prop_expected_cumulative_info_bin_z",
    "planned_response_duration_z",
    "planned_response_total_information_z",
    "time_from_partner_onset_z",
    "time_from_partner_offset_z",
    "run_z",
    "time_within_run_z",
    "subject",
)


@dataclass(frozen=True, slots=True)
class InfoRateInducedLmEEGConfig:
    """Configuration for the info-rate induced-power bridge analysis."""

    config_path: Path
    induced_source_epochs_dir: Path
    behaviour_riskset_path: Path
    out_dir: Path
    analysis_name: str
    anchor_window_start_s: float
    anchor_window_end_s: float
    neural_bin_width_s: float
    info_bin_width_s: float
    min_causal_lag_s: float
    max_causal_lag_s: float | None
    anchor_onset_round_decimals: int
    metadata_query: str | None
    bands: dict[str, np.ndarray]
    n_cycles: dict[str, np.ndarray]
    baseline: dict[str, Any]
    controls: dict[str, Any]
    io: dict[str, Any]
    plotting: dict[str, Any]
    modeling: dict[str, Any]
    cluster_correction: dict[str, Any]
    verbose: bool
    progress: bool


@dataclass(frozen=True, slots=True)
class InfoRateInducedLmEEGResult:
    """Top-level output handles for this analysis run."""

    out_dir: Path
    model_results_path: Path
    diagnostics_path: Path
    trial_table_path: Path
    model_input_path: Path


def _load_paths_yaml(project_root: Path) -> dict[str, Any]:
    paths_path = project_root / "config" / "paths.yaml"
    if not paths_path.exists():
        return {}
    payload = yaml.safe_load(paths_path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _discover_config_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "paths.yaml").exists():
            return candidate
    return start


def _resolve_project_root_from_config(config_path: Path) -> Path:
    return _discover_config_root(config_path.parent).parent.resolve()


def _resolve_path(
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
        # `derivatives_root` already points at the derivatives root for this
        # project, so normalize legacy "derivatives/..." relative paths.
        if candidate.parts and candidate.parts[0] == "derivatives":
            trimmed = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path(".")
            derivative_candidate = (derivatives_root / trimmed).resolve()
        if derivative_candidate.exists() or str(candidate).startswith(("reports/", "results/", "derivatives/")):
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
    if derivatives_root is not None:
        return (derivatives_root / candidate).resolve()
    return (project_root / candidate).resolve()


def _parse_freq_grid(payload: dict[str, Any]) -> np.ndarray:
    if "values" in payload:
        values = np.asarray(payload["values"], dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Morlet frequency values must be a non-empty 1D sequence.")
        return values
    start = float(payload["start_hz"])
    end = float(payload["end_hz"])
    step = float(payload.get("step_hz", 1.0))
    if step <= 0.0:
        raise ValueError("Morlet frequency step must be positive.")
    # Include end-point by construction.
    n_steps = int(round((end - start) / step))
    if n_steps < 0:
        raise ValueError("Expected end_hz >= start_hz for Morlet grid.")
    values = start + np.arange(n_steps + 1, dtype=float) * step
    values = values[values <= end + 1.0e-9]
    if values.size == 0:
        raise ValueError("Morlet frequency grid resolved to an empty list.")
    return values


def _resolve_n_cycles(freqs: np.ndarray, payload: Any) -> np.ndarray:
    if isinstance(payload, (int, float)):
        return np.full(freqs.shape, float(payload), dtype=float)
    if isinstance(payload, list):
        values = np.asarray(payload, dtype=float)
        if values.shape != freqs.shape:
            raise ValueError(
                "When Morlet n_cycles is a list, it must match the frequency grid length."
            )
        return values
    if isinstance(payload, dict):
        mode = str(payload.get("mode", "frequency_divisor"))
        if mode != "frequency_divisor":
            raise ValueError("Unsupported Morlet n_cycles mode. Use `frequency_divisor`.")
        divisor = float(payload.get("divisor", 2.0))
        if divisor <= 0.0:
            raise ValueError("Morlet n_cycles frequency divisor must be positive.")
        return freqs / divisor
    raise ValueError("Unsupported Morlet n_cycles config. Use scalar, list, or mapping.")


def load_info_rate_induced_lmeeg_config(config_path: Path) -> InfoRateInducedLmEEGConfig:
    """Load and validate the bridge-analysis YAML config.

    Usage example
    -------------
    >>> from pathlib import Path
    >>> cfg = load_info_rate_induced_lmeeg_config(Path("config/induced/info_rate_induced_lmeeg.yaml"))  # doctest: +SKIP
    >>> cfg.analysis_name  # doctest: +SKIP
    'info_rate_induced_lmeeg'
    """

    resolved_path = config_path.resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {resolved_path}.")

    project_root = _resolve_project_root_from_config(resolved_path)
    paths_cfg = _load_paths_yaml(project_root)
    derivatives_root = None
    derivatives_value = paths_cfg.get("derivatives_root")
    if isinstance(derivatives_value, str) and derivatives_value.strip():
        derivatives_root = Path(derivatives_value).resolve()

    input_payload = dict(payload.get("input") or {})
    output_payload = dict(payload.get("output") or {})
    windows_payload = dict(payload.get("windows") or {})
    morlet_payload = dict(payload.get("morlet") or {})
    model_payload = dict(payload.get("model") or {})
    controls_payload = dict(payload.get("controls") or {})
    io_payload = dict(payload.get("io") or {})
    plot_payload = dict(payload.get("plotting") or {})
    cluster_payload = dict(payload.get("cluster_correction") or {})

    band_payload = dict(morlet_payload.get("bands") or {})
    if not band_payload:
        raise ValueError("morlet.bands must define at least one band.")

    bands: dict[str, np.ndarray] = {}
    n_cycles: dict[str, np.ndarray] = {}
    n_cycles_payload = morlet_payload.get("n_cycles", {"mode": "frequency_divisor", "divisor": 2.0})
    for band_name, grid_payload in band_payload.items():
        if not isinstance(grid_payload, dict):
            raise ValueError(f"morlet.bands.{band_name} must be a mapping.")
        band_freqs = _parse_freq_grid(dict(grid_payload))
        bands[str(band_name)] = band_freqs
        n_cycles[str(band_name)] = _resolve_n_cycles(band_freqs, n_cycles_payload)

    baseline_payload = dict((morlet_payload.get("baseline") or {}))
    baseline_enabled = bool(baseline_payload.get("enabled", False))
    baseline_interval = baseline_payload.get("interval_s")
    if baseline_enabled:
        if not isinstance(baseline_interval, (list, tuple)) or len(baseline_interval) != 2:
            raise ValueError("morlet.baseline.interval_s must be a 2-item list when baseline is enabled.")

    cfg = InfoRateInducedLmEEGConfig(
        config_path=resolved_path,
        induced_source_epochs_dir=_resolve_path(
            str(input_payload.get("induced_source_epochs_dir", "induced_source_epochs")),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        behaviour_riskset_path=_resolve_path(
            str(
                input_payload.get(
                    "behaviour_riskset_path",
                    "behavior/hazard/risksets/pooled_fpp_spp.parquet",
                )
            ),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        out_dir=_resolve_output_path(
            str(output_payload.get("out_dir", "results/info_rate_induced_lmeeg")),
            project_root=project_root,
            derivatives_root=derivatives_root,
        ),
        analysis_name=str(payload.get("analysis_name", "info_rate_induced_lmeeg")),
        anchor_window_start_s=float((windows_payload.get("anchor_window_s") or {}).get("start", -1.5)),
        anchor_window_end_s=float((windows_payload.get("anchor_window_s") or {}).get("end", 0.0)),
        neural_bin_width_s=float(windows_payload.get("neural_bin_width_s", 0.050)),
        info_bin_width_s=float(windows_payload.get("info_bin_width_s", 0.050)),
        min_causal_lag_s=float(windows_payload.get("min_causal_lag_s", 0.050)),
        max_causal_lag_s=(
            None
            if windows_payload.get("max_causal_lag_s") is None
            else float(windows_payload.get("max_causal_lag_s"))
        ),
        anchor_onset_round_decimals=int(windows_payload.get("anchor_onset_round_decimals", 3)),
        metadata_query=(
            None
            if input_payload.get("metadata_query") in {None, ""}
            else str(input_payload.get("metadata_query"))
        ),
        bands=bands,
        n_cycles=n_cycles,
        baseline={
            "enabled": baseline_enabled,
            "interval_s": tuple(baseline_interval) if isinstance(baseline_interval, (list, tuple)) else None,
            "mode": str(baseline_payload.get("mode", "logratio")),
        },
        controls=controls_payload,
        io=io_payload,
        plotting=plot_payload,
        modeling=model_payload,
        cluster_correction=cluster_payload,
        verbose=bool(payload.get("verbose", True)),
        progress=bool(payload.get("progress", True)),
    )

    if cfg.anchor_window_end_s <= cfg.anchor_window_start_s:
        raise ValueError("anchor_window_s.end must be greater than anchor_window_s.start.")
    if cfg.neural_bin_width_s <= 0.0 or cfg.info_bin_width_s <= 0.0:
        raise ValueError("Bin widths must be positive.")
    if cfg.min_causal_lag_s < 0.0:
        raise ValueError("min_causal_lag_s must be non-negative.")
    if cfg.max_causal_lag_s is not None and cfg.max_causal_lag_s <= 0.0:
        raise ValueError("max_causal_lag_s must be positive when provided.")

    return cfg


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _progress(iterable, *, total: int | None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def _read_table(path: Path) -> pd.DataFrame:
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
    if path.suffix.lower() == ".parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path


def _anchor_type_from_metadata(metadata: pd.DataFrame) -> pd.Series:
    if "anchor_type" in metadata.columns:
        values = metadata["anchor_type"].astype(str).str.strip().str.lower()
    elif "event_family" in metadata.columns:
        values = metadata["event_family"].astype(str).str.strip().str.lower()
    elif "event_type" in metadata.columns:
        inferred = metadata["event_type"].astype(str).str.extract(r"(fpp|spp)", expand=False)
        values = inferred.astype(str).str.strip().str.lower()
    else:
        raise ValueError("Could not infer anchor_type from epochs metadata.")
    invalid = sorted(set(values.dropna().unique()) - {"fpp", "spp"})
    if invalid:
        raise ValueError(f"Unexpected anchor_type labels in metadata: {invalid}")
    return values


def _select_self_onset_anchor_rows(metadata: pd.DataFrame, *, metadata_query: str | None) -> pd.DataFrame:
    selected = metadata.copy()
    if "event_lock" in selected.columns:
        selected = selected.loc[selected["event_lock"].astype(str).str.lower() == "onset"].copy()
    if "event_role" in selected.columns:
        selected = selected.loc[selected["event_role"].astype(str).str.lower() == "self"].copy()

    selected["anchor_type"] = _anchor_type_from_metadata(selected)
    selected = selected.loc[selected["anchor_type"].isin(["fpp", "spp"])].copy()

    if metadata_query:
        mask = selected.eval(metadata_query)
        selected = selected.loc[np.asarray(mask, dtype=bool)].copy()

    return selected


def _resolve_anchor_onset_column(metadata: pd.DataFrame) -> str:
    for candidate in ("event_onset_conversation_s", "event_latency_conversation_s", "anchor_onset_s"):
        if candidate in metadata.columns:
            return candidate
    # Fallback to anchor-specific columns when present.
    if {"fpp_onset", "spp_onset"}.intersection(metadata.columns):
        return "__anchor_specific__"
    raise ValueError(
        "Could not resolve anchor onset column from metadata. "
        "Expected one of: event_onset_conversation_s, event_latency_conversation_s, anchor_onset_s, fpp_onset/spp_onset."
    )


def _resolve_anchor_specific_series(metadata: pd.DataFrame, *, anchor_type_col: str, fpp_col: str, spp_col: str) -> pd.Series:
    if fpp_col not in metadata.columns or spp_col not in metadata.columns:
        return pd.Series(np.nan, index=metadata.index, dtype=float)
    anchor = metadata[anchor_type_col].astype(str).str.lower()
    values = np.where(anchor == "fpp", metadata[fpp_col], metadata[spp_col])
    return pd.to_numeric(pd.Series(values, index=metadata.index), errors="coerce")


def _ensure_numeric(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rename(column_name)


def _build_trial_key(
    *,
    subject: pd.Series,
    run: pd.Series,
    anchor_type: pd.Series,
    anchor_onset_s: pd.Series,
    decimals: int,
) -> pd.Series:
    onset = pd.to_numeric(anchor_onset_s, errors="coerce").round(decimals)
    return (
        subject.astype(str)
        + "|"
        + run.astype(str)
        + "|"
        + anchor_type.astype(str)
        + "|"
        + onset.astype(str)
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source_name: str, source_path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source_name}: {', '.join(sorted(missing))}. "
            f"Expected source file: {source_path}"
        )


def _trial_metadata_from_epochs(
    metadata: pd.DataFrame,
    *,
    record_subject: str,
    record_run: str,
    anchor_onset_decimals: int,
    config: InfoRateInducedLmEEGConfig,
) -> pd.DataFrame:
    working = metadata.copy()
    working["anchor_type"] = _anchor_type_from_metadata(working)

    subject = working.get("subject_id", pd.Series([record_subject] * len(working), index=working.index)).astype(str)
    run = pd.to_numeric(
        working.get("run", pd.Series([record_run] * len(working), index=working.index)),
        errors="coerce",
    )

    anchor_onset_col = _resolve_anchor_onset_column(working)
    if anchor_onset_col == "__anchor_specific__":
        anchor_onset = _resolve_anchor_specific_series(
            working,
            anchor_type_col="anchor_type",
            fpp_col="fpp_onset",
            spp_col="spp_onset",
        )
    else:
        anchor_onset = pd.to_numeric(working[anchor_onset_col], errors="coerce")

    planned_duration = _resolve_anchor_specific_series(
        working,
        anchor_type_col="anchor_type",
        fpp_col="fpp_duration",
        spp_col="spp_duration",
    )
    if planned_duration.isna().all() and "duration" in working.columns:
        planned_duration = pd.to_numeric(working["duration"], errors="coerce")
    if planned_duration.isna().all() and not {"fpp_duration", "spp_duration", "duration"}.intersection(working.columns):
        raise ValueError(
            "Missing required planned-response duration columns in epochs metadata. "
            "Expected at least one of: fpp_duration, spp_duration, duration."
        )

    partner_onset = _resolve_anchor_specific_series(
        working,
        anchor_type_col="anchor_type",
        fpp_col="spp_onset",
        spp_col="fpp_onset",
    )
    partner_offset = _resolve_anchor_specific_series(
        working,
        anchor_type_col="anchor_type",
        fpp_col="spp_offset",
        spp_col="fpp_offset",
    )
    if partner_onset.isna().all() and "partner_ipu_onset" in working.columns:
        partner_onset = pd.to_numeric(working["partner_ipu_onset"], errors="coerce")
    if partner_offset.isna().all() and "partner_ipu_offset" in working.columns:
        partner_offset = pd.to_numeric(working["partner_ipu_offset"], errors="coerce")
    if partner_onset.isna().all() and not {"spp_onset", "fpp_onset", "partner_ipu_onset"}.intersection(working.columns):
        raise ValueError(
            "Missing required partner-onset columns in epochs metadata. "
            "Expected at least one of: spp_onset/fpp_onset (anchor-specific) or partner_ipu_onset."
        )
    if partner_offset.isna().all() and not {"spp_offset", "fpp_offset", "partner_ipu_offset"}.intersection(working.columns):
        raise ValueError(
            "Missing required partner-offset columns in epochs metadata. "
            "Expected at least one of: spp_offset/fpp_offset (anchor-specific) or partner_ipu_offset."
        )

    planned_info_candidates = [
        "upcoming_utterance_information_content",
        "planned_response_total_information",
        "response_total_information",
        "expected_total_info",
        "actual_total_info",
    ]
    planned_info = pd.Series(np.nan, index=working.index, dtype=float)
    for candidate in planned_info_candidates:
        if candidate in working.columns:
            planned_info = pd.to_numeric(working[candidate], errors="coerce")
            if planned_info.notna().any():
                break

    time_within_run = pd.to_numeric(
        working.get(
            "time_within_run",
            working.get("event_onset_conversation_s", anchor_onset),
        ),
        errors="coerce",
    )

    out = pd.DataFrame(
        {
            "subject": subject,
            "run": run,
            "anchor_type": working["anchor_type"].astype(str).str.lower(),
            "anchor_onset_s": anchor_onset,
            "partner_onset_s": partner_onset,
            "partner_offset_s": partner_offset,
            "planned_response_duration": planned_duration,
            "planned_response_total_information": planned_info,
            "time_within_run": time_within_run,
        }
    )
    out["time_from_partner_onset"] = out["anchor_onset_s"] - out["partner_onset_s"]
    out["time_from_partner_offset"] = out["anchor_onset_s"] - out["partner_offset_s"]
    out["trial_id"] = _build_trial_key(
        subject=out["subject"],
        run=out["run"],
        anchor_type=out["anchor_type"],
        anchor_onset_s=out["anchor_onset_s"],
        decimals=anchor_onset_decimals,
    )

    required = [
        "subject",
        "run",
        "anchor_type",
        "anchor_onset_s",
        "planned_response_duration",
        "planned_response_total_information",
        "time_from_partner_onset",
        "time_from_partner_offset",
        "time_within_run",
    ]
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise ValueError("Internal trial metadata build error: missing columns " + ", ".join(missing))

    return out


def _bin_edges(start_s: float, end_s: float, width_s: float) -> np.ndarray:
    n_bins = int(math.floor((end_s - start_s) / width_s + 1.0e-9))
    edges = start_s + np.arange(n_bins + 1, dtype=float) * width_s
    if edges[-1] < end_s - 1.0e-9:
        edges = np.append(edges, end_s)
    return edges


def _summarize_zscore(values: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    if (not np.isfinite(sd)) or sd == 0.0:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index), mean, sd
    return (numeric - mean) / sd, mean, sd


def _baseline_suffix(config: InfoRateInducedLmEEGConfig) -> str:
    if not bool(config.baseline.get("enabled", False)):
        return ""
    interval = config.baseline.get("interval_s") or (None, None)
    mode = str(config.baseline.get("mode", "unknown"))
    start, end = interval
    return f"_baseline-{mode}_{start}_{end}"


def _extract_morlet_induced_trials(config: InfoRateInducedLmEEGConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load pooled epochs and extract single-trial induced Morlet power.

    Returns
    -------
    induced_trials:
        One row per trial x band x neural bin with unscaled induced power and
        trial metadata controls.
    missingness_table:
        Per-column missingness summary from trial metadata.
    extraction_summary:
        Metadata summary for reproducibility.
    """

    LOGGER.info("Loading data: discovering induced source epochs in %s", config.induced_source_epochs_dir)
    records = discover_epoch_records(config.induced_source_epochs_dir)
    if not records:
        raise FileNotFoundError(f"No epoch FIF files found under {config.induced_source_epochs_dir}")

    import mne

    configure_mne_runtime()

    trial_rows: list[pd.DataFrame] = []
    expected_time_axis: np.ndarray | None = None
    expected_channel_count: int | None = None

    neural_edges = _bin_edges(config.anchor_window_start_s, config.anchor_window_end_s, config.neural_bin_width_s)
    neural_bins = list(zip(neural_edges[:-1], neural_edges[1:], strict=True))

    for record in _progress(
        records,
        total=len(records),
        desc="Extracting Morlet induced power",
        enabled=config.progress,
    ):
        epochs = load_epochs(record.epochs_path) if record.epochs_path is not None else None
        if epochs is None:
            continue
        if epochs.metadata is None:
            raise ValueError(f"Epochs file has no metadata: {record.epochs_path}")

        selected_metadata = _select_self_onset_anchor_rows(
            epochs.metadata.reset_index(drop=True),
            metadata_query=config.metadata_query,
        )
        if selected_metadata.empty:
            continue

        selected_indices = selected_metadata.index.to_numpy(dtype=int)
        selected_epochs = epochs[selected_indices]
        selected_metadata = selected_metadata.reset_index(drop=True)

        try:
            trial_meta = _trial_metadata_from_epochs(
                selected_metadata,
                record_subject=record.subject_id,
                record_run=record.run_id,
                anchor_onset_decimals=config.anchor_onset_round_decimals,
                config=config,
            )
        except ValueError as exc:
            raise ValueError(
                f"{exc} Expected source file: {record.epochs_path}"
            ) from exc

        epoch_times = np.asarray(selected_epochs.times, dtype=float)
        if expected_time_axis is None:
            expected_time_axis = epoch_times
        else:
            if not np.allclose(epoch_times, expected_time_axis):
                raise ValueError("Epoch time axes differ across files; cannot pool for consistent binning.")

        if epoch_times[0] > config.anchor_window_start_s or epoch_times[-1] < config.anchor_window_end_s:
            raise ValueError(
                "Epochs do not cover requested anchor window "
                f"[{config.anchor_window_start_s}, {config.anchor_window_end_s}] s. "
                f"Observed epoch bounds: [{epoch_times[0]}, {epoch_times[-1]}] s in {record.epochs_path}."
            )

        data = selected_epochs.get_data(copy=True)
        sfreq = float(selected_epochs.info["sfreq"])
        n_epochs, n_channels, _ = data.shape
        if expected_channel_count is None:
            expected_channel_count = int(n_channels)

        for band_name, freqs in config.bands.items():
            n_cycles = config.n_cycles[band_name]
            power = mne.time_frequency.tfr_array_morlet(
                data,
                sfreq=sfreq,
                freqs=freqs,
                n_cycles=n_cycles,
                output="power",
                use_fft=True,
                n_jobs=1,
                decim=1,
                zero_mean=True,
                verbose="ERROR",
            )
            # Shape: (n_epochs, n_channels, n_freqs, n_times).
            band_power = power.mean(axis=2)

            if bool(config.baseline.get("enabled", False)):
                baseline_interval = config.baseline.get("interval_s")
                baseline_mode = str(config.baseline.get("mode", "logratio"))
                band_power = mne.baseline.rescale(
                    band_power,
                    times=epoch_times,
                    baseline=baseline_interval,
                    mode=baseline_mode,
                    copy=True,
                )

            if bool(config.modeling.get("average_channels", True)):
                band_power = band_power.mean(axis=1)
            else:
                # Conservative fallback to grand mean to keep a scalar DV per trial/bin.
                band_power = band_power.mean(axis=1)

            for bin_start, bin_end in neural_bins:
                mask = (epoch_times >= bin_start - 1.0e-9) & (epoch_times < bin_end - 1.0e-9)
                if not np.any(mask):
                    continue
                binned = band_power[:, mask].mean(axis=1)
                rows = trial_meta.copy()
                rows["band"] = band_name
                rows["neural_bin_start_s"] = float(bin_start)
                rows["neural_bin_end_s"] = float(bin_end)
                rows["neural_bin_center_s"] = float(0.5 * (bin_start + bin_end))
                rows["induced_power"] = binned.astype(float)
                trial_rows.append(rows)

    if not trial_rows:
        raise ValueError("No eligible self-onset FPP/SPP epochs were available for induced Morlet extraction.")

    induced_trials = pd.concat(trial_rows, ignore_index=True, sort=False)
    missingness = pd.DataFrame(
        {
            "column": induced_trials.columns,
            "n_missing": [int(induced_trials[column].isna().sum()) for column in induced_trials.columns],
            "n_rows": int(len(induced_trials)),
            "fraction_missing": [float(induced_trials[column].isna().mean()) for column in induced_trials.columns],
            "source": "induced_trials",
        }
    )

    extraction_summary = {
        "n_records": int(len(records)),
        "n_rows": int(len(induced_trials)),
        "n_trials_unique": int(induced_trials["trial_id"].nunique()),
        "bands": sorted(induced_trials["band"].astype(str).unique().tolist()),
        "neural_bins": int(induced_trials[["neural_bin_start_s", "neural_bin_end_s"]].drop_duplicates().shape[0]),
        "baseline": config.baseline,
        "channel_count": int(expected_channel_count or 0),
    }

    return induced_trials, missingness, extraction_summary


def _build_info_bins_table(config: InfoRateInducedLmEEGConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Loading behavioural riskset from %s", config.behaviour_riskset_path)
    riskset = _read_table(config.behaviour_riskset_path)

    required = [
        "subject_id",
        "run_id",
        "anchor_type",
        "bin_start_s",
        "bin_end_s",
        "information_rate",
        "prop_expected_cumulative_info",
    ]
    anchor_onset_candidates = [
        "anchor_onset_s",
        "own_fpp_onset",
        "fpp_onset",
        "own_spp_onset",
        "spp_onset",
    ]
    if not any(column in riskset.columns for column in anchor_onset_candidates):
        raise ValueError(
            "Behaviour riskset is missing an anchor-onset column. "
            f"Expected one of: {', '.join(anchor_onset_candidates)}. "
            f"Expected source file: {config.behaviour_riskset_path}"
        )
    _require_columns(
        riskset,
        required,
        source_name="behaviour riskset",
        source_path=config.behaviour_riskset_path,
    )

    table = riskset.copy()
    table["subject"] = table["subject_id"].astype(str)
    table["run"] = pd.to_numeric(table["run_id"], errors="coerce")
    table["anchor_type"] = table["anchor_type"].astype(str).str.lower()
    table = table.loc[table["anchor_type"].isin(["fpp", "spp"])].copy()

    anchor_onset = pd.Series(np.nan, index=table.index, dtype=float)
    for candidate in anchor_onset_candidates:
        if candidate in table.columns:
            candidate_values = pd.to_numeric(table[candidate], errors="coerce")
            if candidate_values.notna().any():
                anchor_onset = candidate_values
                LOGGER.info("Using `%s` as behaviour anchor-onset column.", candidate)
                break
    if anchor_onset.isna().all():
        raise ValueError(
            "Behaviour riskset anchor-onset column was present but all values were NaN. "
            f"Checked columns: {', '.join([c for c in anchor_onset_candidates if c in table.columns])}. "
            f"Expected source file: {config.behaviour_riskset_path}"
        )

    table["anchor_onset_s"] = anchor_onset
    table["info_bin_start_s"] = pd.to_numeric(table["bin_start_s"], errors="coerce") - anchor_onset
    table["info_bin_end_s"] = pd.to_numeric(table["bin_end_s"], errors="coerce") - anchor_onset
    table["info_bin_center_s"] = 0.5 * (table["info_bin_start_s"] + table["info_bin_end_s"])
    table["information_rate_bin"] = pd.to_numeric(table["information_rate"], errors="coerce")
    table["prop_expected_cumulative_info_bin"] = pd.to_numeric(
        table["prop_expected_cumulative_info"], errors="coerce"
    )
    planned_info_candidates = [
        "planned_response_total_information",
        "upcoming_utterance_information_content",
        "response_total_information",
        "expected_total_info",
        "actual_total_info",
    ]
    planned_info = pd.Series(np.nan, index=table.index, dtype=float)
    for candidate in planned_info_candidates:
        if candidate in table.columns:
            candidate_values = pd.to_numeric(table[candidate], errors="coerce")
            if candidate_values.notna().any():
                planned_info = candidate_values
                LOGGER.info("Using `%s` as planned-response total-information source.", candidate)
                break
    table["planned_response_total_information"] = planned_info
    table["trial_id"] = _build_trial_key(
        subject=table["subject"],
        run=table["run"],
        anchor_type=table["anchor_type"],
        anchor_onset_s=table["anchor_onset_s"],
        decimals=config.anchor_onset_round_decimals,
    )

    # Keep only bins in requested anchor-relative window.
    window_mask = (
        (table["info_bin_start_s"] >= config.anchor_window_start_s - 1.0e-9)
        & (table["info_bin_end_s"] <= config.anchor_window_end_s + 1.0e-9)
    )
    table = table.loc[window_mask].copy()

    grouped = (
        table.groupby(
            [
                "trial_id",
                "subject",
                "run",
                "anchor_type",
                "anchor_onset_s",
                "info_bin_start_s",
                "info_bin_end_s",
                "info_bin_center_s",
            ],
            as_index=False,
            sort=False,
            observed=True,
        )
        .agg(
            information_rate_bin=("information_rate_bin", "mean"),
            prop_expected_cumulative_info_bin=("prop_expected_cumulative_info_bin", "mean"),
            planned_response_total_information=("planned_response_total_information", "mean"),
        )
    )

    missingness = pd.DataFrame(
        {
            "column": grouped.columns,
            "n_missing": [int(grouped[column].isna().sum()) for column in grouped.columns],
            "n_rows": int(len(grouped)),
            "fraction_missing": [float(grouped[column].isna().mean()) for column in grouped.columns],
            "source": "behaviour_bins",
        }
    )

    if grouped.empty:
        raise ValueError(
            "Behaviour riskset did not yield any information bins inside the requested anchor window."
        )

    return grouped, missingness


def _build_model_input_table(
    *,
    induced_trials: pd.DataFrame,
    info_bins: pd.DataFrame,
    config: InfoRateInducedLmEEGConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Building causal bins and joining neural/behaviour tables")

    join_columns = ["trial_id", "subject", "run", "anchor_type"]
    merged = induced_trials.merge(
        info_bins,
        how="inner",
        on=join_columns,
        suffixes=("", "_info"),
    )
    if merged.empty:
        raise ValueError(
            "No rows remained after joining induced trials with behaviour bins. "
            "This usually indicates mismatched trial keys between neural and behavioural tables "
            "(subject/run/anchor_type/anchor_onset)."
        )

    # Causal constraints.
    causal_mask = merged["info_bin_end_s"] <= (
        merged["neural_bin_start_s"] - float(config.min_causal_lag_s)
    )
    if config.max_causal_lag_s is not None:
        causal_mask &= (
            merged["neural_bin_center_s"] - merged["info_bin_center_s"]
            <= float(config.max_causal_lag_s)
        )
    causal = merged.loc[causal_mask].copy()
    causal["causal_lag_s"] = causal["neural_bin_center_s"] - causal["info_bin_center_s"]

    if causal.empty:
        raise ValueError("No causal neural/predictor bin pairs satisfied the configured lag constraints.")

    # Standardize continuous columns globally (within this analysis table).
    zscore_specs = [
        ("induced_power", "induced_power_z"),
        ("information_rate_bin", "information_rate_bin_z"),
        ("prop_expected_cumulative_info_bin", "prop_expected_cumulative_info_bin_z"),
        ("planned_response_duration", "planned_response_duration_z"),
        ("planned_response_total_information", "planned_response_total_information_z"),
        ("time_from_partner_onset", "time_from_partner_onset_z"),
        ("time_from_partner_offset", "time_from_partner_offset_z"),
        ("run", "run_z"),
        ("time_within_run", "time_within_run_z"),
    ]

    scaling_rows: list[dict[str, Any]] = []
    for source, target in zscore_specs:
        z_values, mean, sd = _summarize_zscore(causal[source])
        causal[target] = z_values
        scaling_rows.append({"column": source, "z_column": target, "mean": mean, "sd": sd, "scope": "global"})

    causal["anchor_type"] = pd.Categorical(
        causal["anchor_type"].astype(str).str.lower(),
        categories=["spp", "fpp"],
        ordered=True,
    )

    model_missingness = pd.DataFrame(
        {
            "column": causal.columns,
            "n_missing": [int(causal[column].isna().sum()) for column in causal.columns],
            "n_rows": int(len(causal)),
            "fraction_missing": [float(causal[column].isna().mean()) for column in causal.columns],
            "source": "model_input_causal",
        }
    )

    # Complete-case rows for modeling.
    needed = list(MODEL_REQUIRED_COLUMNS)
    complete_mask = causal[needed].notna().all(axis=1)
    model_input = causal.loc[complete_mask].copy()

    if model_input.empty:
        raise ValueError("No complete-case rows remained after model column filtering.")

    return model_input, pd.DataFrame(scaling_rows), model_missingness


def _fit_model_grid(model_input: pd.DataFrame, *, config: InfoRateInducedLmEEGConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Fitting models")

    formula = (
        "induced_power_z ~ anchor_type * information_rate_bin_z"
        " + anchor_type * prop_expected_cumulative_info_bin_z"
        " + planned_response_duration_z"
        " + planned_response_total_information_z"
        " + time_from_partner_onset_z"
        " + time_from_partner_offset_z"
        " + run_z"
        " + time_within_run_z"
    )

    min_rows = int(config.modeling.get("min_rows_per_cell", 20))
    min_subjects = int(config.modeling.get("min_subjects_per_cell", 2))
    maxiter = int(config.modeling.get("maxiter", 200))

    cell_keys = [
        "band",
        "neural_bin_start_s",
        "neural_bin_end_s",
        "neural_bin_center_s",
        "info_bin_start_s",
        "info_bin_end_s",
        "info_bin_center_s",
    ]

    grouped = list(model_input.groupby(cell_keys, sort=True, observed=True))
    result_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for cell_key, cell in _progress(
        grouped,
        total=len(grouped),
        desc="Model grid",
        enabled=config.progress,
    ):
        (
            band,
            neural_start,
            neural_end,
            neural_center,
            info_start,
            info_end,
            info_center,
        ) = cell_key

        n_rows = int(len(cell))
        n_subjects = int(cell["subject"].nunique())
        warning_messages: list[str] = []
        converged = False
        error_message = ""

        if n_rows < min_rows or n_subjects < min_subjects:
            error_message = (
                f"Skipped cell: n_rows={n_rows} (min {min_rows}), "
                f"n_subjects={n_subjects} (min {min_subjects})."
            )
            diag_rows.append(
                {
                    "band": band,
                    "neural_bin_start_s": float(neural_start),
                    "neural_bin_end_s": float(neural_end),
                    "neural_bin_center_s": float(neural_center),
                    "info_bin_start_s": float(info_start),
                    "info_bin_end_s": float(info_end),
                    "info_bin_center_s": float(info_center),
                    "n_observations": n_rows,
                    "n_subjects": n_subjects,
                    "converged": False,
                    "warning_summary": error_message,
                }
            )
            continue

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = mixedlm(formula=formula, data=cell, groups=cell["subject"], re_formula="1")
                fitted = model.fit(reml=False, method="lbfgs", maxiter=maxiter, disp=False)
            converged = bool(getattr(fitted, "converged", True))
            warning_messages = [str(w.message) for w in caught if str(w.message)]

            params = fitted.params
            bse = fitted.bse
            tvalues = fitted.tvalues
            pvalues = fitted.pvalues
            conf_int = fitted.conf_int()

            for term in params.index:
                estimate = float(params.get(term, np.nan))
                standard_error = float(bse.get(term, np.nan))
                statistic = float(tvalues.get(term, np.nan))
                p_value = float(pvalues.get(term, np.nan))
                if term in conf_int.index:
                    conf_low = float(conf_int.loc[term, 0])
                    conf_high = float(conf_int.loc[term, 1])
                else:
                    conf_low = np.nan
                    conf_high = np.nan
                result_rows.append(
                    {
                        "band": str(band),
                        "neural_bin_start_s": float(neural_start),
                        "neural_bin_end_s": float(neural_end),
                        "neural_bin_center_s": float(neural_center),
                        "info_bin_start_s": float(info_start),
                        "info_bin_end_s": float(info_end),
                        "info_bin_center_s": float(info_center),
                        "causal_lag_s": float(neural_center - info_center),
                        "term": str(term),
                        "estimate": estimate,
                        "standard_error": standard_error,
                        "statistic": statistic,
                        "p_value": p_value,
                        "conf_low": conf_low,
                        "conf_high": conf_high,
                        "n_observations": n_rows,
                        "n_subjects": n_subjects,
                        "converged": converged,
                        "warning_summary": " | ".join(warning_messages),
                    }
                )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        diag_rows.append(
            {
                "band": str(band),
                "neural_bin_start_s": float(neural_start),
                "neural_bin_end_s": float(neural_end),
                "neural_bin_center_s": float(neural_center),
                "info_bin_start_s": float(info_start),
                "info_bin_end_s": float(info_end),
                "info_bin_center_s": float(info_center),
                "n_observations": n_rows,
                "n_subjects": n_subjects,
                "converged": bool(converged),
                "warning_summary": " | ".join([*warning_messages, error_message]).strip(" |"),
            }
        )

    result_columns = [
        "band",
        "neural_bin_start_s",
        "neural_bin_end_s",
        "neural_bin_center_s",
        "info_bin_start_s",
        "info_bin_end_s",
        "info_bin_center_s",
        "causal_lag_s",
        "term",
        "estimate",
        "standard_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
        "n_observations",
        "n_subjects",
        "converged",
        "warning_summary",
    ]
    diag_columns = [
        "band",
        "neural_bin_start_s",
        "neural_bin_end_s",
        "neural_bin_center_s",
        "info_bin_start_s",
        "info_bin_end_s",
        "info_bin_center_s",
        "n_observations",
        "n_subjects",
        "converged",
        "warning_summary",
    ]
    result_table = pd.DataFrame(result_rows, columns=result_columns)
    diagnostics = pd.DataFrame(diag_rows, columns=diag_columns)
    return result_table, diagnostics


def _term_map(term: str) -> str:
    return term.replace(" ", "")


def _find_term(df: pd.DataFrame, preferred: list[str]) -> str | None:
    if df.empty or "term" not in df.columns:
        return None
    terms = df["term"].astype(str).tolist()
    normalized = {_term_map(term): term for term in terms}
    for candidate in preferred:
        if _term_map(candidate) in normalized:
            return normalized[_term_map(candidate)]
    if preferred:
        probe = preferred[0]
        predictor_token = None
        if "prop_expected_cumulative_info_bin_z" in probe:
            predictor_token = "prop_expected_cumulative_info_bin_z"
        elif "information_rate_bin_z" in probe:
            predictor_token = "information_rate_bin_z"
        if predictor_token is not None:
            for term in terms:
                if predictor_token in term and "anchor_type" in term and "fpp" in term:
                    return term
    return None


def _build_grid_frame(
    model_results: pd.DataFrame,
    *,
    band: str,
    term: str,
    value_col: str,
    neural_bins: np.ndarray,
    info_bins: np.ndarray,
) -> np.ndarray:
    subset = model_results.loc[
        (model_results["band"].astype(str) == str(band))
        & (model_results["term"].astype(str) == str(term))
    ].copy()
    if subset.empty:
        return np.full((len(neural_bins), len(info_bins)), np.nan, dtype=float)

    pivot = subset.pivot_table(
        index="neural_bin_center_s",
        columns="info_bin_center_s",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=neural_bins, columns=info_bins)
    return pivot.to_numpy(dtype=float)


def _plot_tmap(
    *,
    values: np.ndarray,
    neural_bins: np.ndarray,
    info_bins: np.ndarray,
    output_stem: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    p_values: np.ndarray | None,
    formats: tuple[str, ...],
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    masked = np.ma.masked_invalid(values)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    extent = [
        float(info_bins.min()),
        float(info_bins.max()),
        float(neural_bins.min()),
        float(neural_bins.max()),
    ]
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        norm=norm,
    )

    # Mark unmodelled cells in light gray.
    if np.any(~np.isfinite(values)):
        invalid = ~np.isfinite(values)
        y_idx, x_idx = np.where(invalid)
        for yi, xi in zip(y_idx.tolist(), x_idx.tolist(), strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (
                        info_bins[xi] - (info_bins[1] - info_bins[0]) / 2.0 if len(info_bins) > 1 else info_bins[xi],
                        neural_bins[yi] - (neural_bins[1] - neural_bins[0]) / 2.0 if len(neural_bins) > 1 else neural_bins[yi],
                    ),
                    (info_bins[1] - info_bins[0]) if len(info_bins) > 1 else 0.05,
                    (neural_bins[1] - neural_bins[0]) if len(neural_bins) > 1 else 0.05,
                    facecolor="0.85",
                    edgecolor="none",
                    alpha=0.8,
                    zorder=1.5,
                )
            )

    if p_values is not None and p_values.shape == values.shape:
        with np.errstate(invalid="ignore"):
            sig = np.where(np.isfinite(p_values) & (p_values < 0.05), 1.0, np.nan)
        if np.isfinite(sig).any():
            ax.contour(
                sig,
                levels=[1.0],
                colors="k",
                linewidths=0.8,
                origin="lower",
                extent=extent,
            )

    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Statistic (t/z)")
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(output_stem.with_suffix(f".{fmt}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_basic_qc(
    *,
    model_input: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    # 1. Number of observations per cell.
    if not diagnostics.empty:
        grouped = list(diagnostics.groupby("band", sort=True))
        fig, axes = plt.subplots(1, max(1, len(grouped)), figsize=(8 * max(1, len(grouped)), 6), squeeze=False)
        for axis, (band_name, subset) in zip(axes.flatten(), grouped, strict=True):
            pivot = subset.pivot_table(
                index="neural_bin_center_s",
                columns="info_bin_center_s",
                values="n_observations",
                aggfunc="mean",
            )
            image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
            axis.set_title(f"{band_name} N observations")
            axis.set_xlabel("Information-rate predictor bin relative to anchor onset (s)")
            axis.set_ylabel("Neural power bin relative to anchor onset (s)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="N")
        fig.tight_layout()
        stem = out_dir / "n_observations_by_band_neural_bin_info_bin"
        for fmt in formats:
            fig.savefig(stem.with_suffix(f".{fmt}"), dpi=220)
        plt.close(fig)

    # 3. Predictor distributions.
    predictors = [
        "information_rate_bin_z",
        "prop_expected_cumulative_info_bin_z",
        "planned_response_duration_z",
        "planned_response_total_information_z",
        "time_from_partner_onset_z",
        "time_from_partner_offset_z",
        "run_z",
        "time_within_run_z",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes_flat = axes.flatten()
    for axis, column in zip(axes_flat, predictors, strict=True):
        values = pd.to_numeric(model_input[column], errors="coerce")
        axis.hist(values[np.isfinite(values)], bins=40, color="#1f77b4", alpha=0.85)
        axis.set_title(column)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"predictor_distributions.{fmt}", dpi=220)
    plt.close(fig)

    # 4. Induced power distribution by band and anchor type.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        float(pd.to_numeric(model_input["induced_power_z"], errors="coerce").min()),
        float(pd.to_numeric(model_input["induced_power_z"], errors="coerce").max()),
        45,
    )
    for (band, anchor), subset in model_input.groupby(["band", "anchor_type"], sort=True, observed=True):
        label = f"{band}:{anchor}"
        values = pd.to_numeric(subset["induced_power_z"], errors="coerce")
        ax.hist(values[np.isfinite(values)], bins=bins, alpha=0.35, label=label)
    ax.set_title("Induced power (z) by band and anchor type")
    ax.set_xlabel("induced_power_z")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"induced_power_distribution_by_anchor_type.{fmt}", dpi=220)
    plt.close(fig)

    # 5. Correlation matrix.
    corr_cols = [
        "information_rate_bin_z",
        "prop_expected_cumulative_info_bin_z",
        "planned_response_duration_z",
        "planned_response_total_information_z",
        "time_from_partner_onset_z",
        "time_from_partner_offset_z",
        "run_z",
        "time_within_run_z",
    ]
    corr = model_input[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Predictor correlation matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"predictor_correlation_matrix.{fmt}", dpi=220)
    plt.close(fig)

    # 6. Convergence heatmap.
    if not diagnostics.empty:
        grouped = list(diagnostics.groupby("band", sort=True))
        fig, axes = plt.subplots(1, max(1, len(grouped)), figsize=(8 * max(1, len(grouped)), 6), squeeze=False)
        for axis, (band_name, subset) in zip(axes.flatten(), grouped, strict=True):
            pivot = subset.pivot_table(
                index="neural_bin_center_s",
                columns="info_bin_center_s",
                values="converged",
                aggfunc="mean",
            )
            image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
            axis.set_title(f"{band_name} convergence")
            axis.set_xlabel("Information-rate predictor bin relative to anchor onset (s)")
            axis.set_ylabel("Neural power bin relative to anchor onset (s)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Converged fraction")
        fig.tight_layout()
        stem = out_dir / "model_convergence_heatmap"
        for fmt in formats:
            fig.savefig(stem.with_suffix(f".{fmt}"), dpi=220)
        plt.close(fig)


def _write_config_snapshot(config: InfoRateInducedLmEEGConfig, out_dir: Path) -> Path:
    payload = {
        "analysis_name": config.analysis_name,
        "input": {
            "induced_source_epochs_dir": str(config.induced_source_epochs_dir),
            "behaviour_riskset_path": str(config.behaviour_riskset_path),
            "metadata_query": config.metadata_query,
        },
        "output": {"out_dir": str(config.out_dir)},
        "windows": {
            "anchor_window_s": {
                "start": config.anchor_window_start_s,
                "end": config.anchor_window_end_s,
            },
            "neural_bin_width_s": config.neural_bin_width_s,
            "info_bin_width_s": config.info_bin_width_s,
            "min_causal_lag_s": config.min_causal_lag_s,
            "max_causal_lag_s": config.max_causal_lag_s,
        },
        "morlet": {
            "bands": {name: freqs.tolist() for name, freqs in config.bands.items()},
            "n_cycles": {name: cycles.tolist() for name, cycles in config.n_cycles.items()},
            "baseline": config.baseline,
        },
        "controls": config.controls,
        "plotting": config.plotting,
        "modeling": config.modeling,
        "cluster_correction": config.cluster_correction,
        "verbose": config.verbose,
        "progress": config.progress,
    }
    path = out_dir / "config_used.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_info_rate_induced_lmeeg_pipeline(config: InfoRateInducedLmEEGConfig) -> InfoRateInducedLmEEGResult:
    """Run the full info-rate induced-power lagged bridge analysis."""

    configure_mne_runtime()
    _configure_logging(verbose=config.verbose)

    LOGGER.info("Starting %s", config.analysis_name)
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Step: loading data and extracting Morlet induced power")
    induced_trials, missingness_induced, extraction_summary = _extract_morlet_induced_trials(config)

    LOGGER.info("Step: loading behavioural bins")
    info_bins, missingness_info = _build_info_bins_table(config)

    LOGGER.info("Step: building causal bins")
    model_input, zscore_summary, missingness_model = _build_model_input_table(
        induced_trials=induced_trials,
        info_bins=info_bins,
        config=config,
    )

    LOGGER.info("Step: fitting models")
    model_results, diagnostics = _fit_model_grid(model_input, config=config)

    # Save tables.
    baseline_suffix = _baseline_suffix(config)
    trial_ext = str(config.io.get("table_format", "parquet")).strip().lower()
    if trial_ext not in {"parquet", "csv"}:
        trial_ext = "parquet"

    trial_table_path = out_dir / f"induced_power_trials{baseline_suffix}.{trial_ext}"
    model_input_path = out_dir / f"model_input_binned{baseline_suffix}.{trial_ext}"
    model_results_path = out_dir / "model_results_long.csv"
    diagnostics_path = out_dir / "model_diagnostics.csv"

    _write_table(induced_trials, trial_table_path)
    _write_table(model_input, model_input_path)
    model_results.to_csv(model_results_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    missingness = pd.concat([missingness_induced, missingness_info, missingness_model], ignore_index=True, sort=False)
    missingness.to_csv(out_dir / "missingness_summary.csv", index=False)
    zscore_summary.to_csv(out_dir / "preprocessing_zscore_summary.csv", index=False)

    # Save map-ready arrays (for optional later correction).
    term_candidates = {
        "anchor_x_information_rate": {
            "preferred_terms": [
                "anchor_type[T.fpp]:information_rate_bin_z",
                "information_rate_bin_z:anchor_type[T.fpp]",
            ],
            "required": True,
        },
        "anchor_x_prop_expected_cum_info": {
            "preferred_terms": [
                "anchor_type[T.fpp]:prop_expected_cumulative_info_bin_z",
                "prop_expected_cumulative_info_bin_z:anchor_type[T.fpp]",
            ],
            "required": True,
        },
        "information_rate_main": {
            "preferred_terms": ["information_rate_bin_z"],
            "required": False,
        },
        "prop_expected_cum_info_main": {
            "preferred_terms": ["prop_expected_cumulative_info_bin_z"],
            "required": False,
        },
    }

    info_bins_axis = np.sort(model_input["info_bin_center_s"].unique())
    neural_bins_axis = np.sort(model_input["neural_bin_center_s"].unique())
    formats = tuple(config.plotting.get("formats", ["png", "pdf"]))
    xlabel = "information-rate predictor bin relative to anchor onset (s)"
    ylabel = "neural power bin relative to anchor onset (s)"

    for band in sorted(model_input["band"].astype(str).unique().tolist()):
        band_results = model_results.loc[model_results["band"].astype(str) == band].copy()
        for map_name, spec in term_candidates.items():
            term = _find_term(band_results, preferred=list(spec["preferred_terms"]))
            if term is None:
                if bool(spec["required"]):
                    LOGGER.warning(
                        "No matching term found for required map `%s` in band `%s`; writing NaN placeholder map.",
                        map_name,
                        band,
                    )
                    stat_grid = np.full((len(neural_bins_axis), len(info_bins_axis)), np.nan, dtype=float)
                    p_grid = np.full((len(neural_bins_axis), len(info_bins_axis)), np.nan, dtype=float)
                else:
                    continue
            else:
                stat_grid = _build_grid_frame(
                    model_results,
                    band=band,
                    term=term,
                    value_col="statistic",
                    neural_bins=neural_bins_axis,
                    info_bins=info_bins_axis,
                )
                p_grid = _build_grid_frame(
                    model_results,
                    band=band,
                    term=term,
                    value_col="p_value",
                    neural_bins=neural_bins_axis,
                    info_bins=info_bins_axis,
                )

            np.save(
                out_dir / f"{band}_{map_name}_statistic_map.npy",
                stat_grid,
            )
            np.save(
                out_dir / f"{band}_{map_name}_pvalue_map.npy",
                p_grid,
            )

            output_stem = out_dir / f"{band}_{map_name}_tmap"
            _plot_tmap(
                values=stat_grid,
                neural_bins=neural_bins_axis,
                info_bins=info_bins_axis,
                output_stem=output_stem,
                title=f"{band.upper()} {map_name.replace('_', ' ')}",
                xlabel=xlabel,
                ylabel=ylabel,
                p_values=p_grid if bool(config.plotting.get("show_uncorrected_p05_contours", True)) else None,
                formats=formats,
            )

    LOGGER.info("Step: plotting QC")
    _plot_basic_qc(
        model_input=model_input,
        diagnostics=diagnostics,
        out_dir=out_dir,
        formats=tuple(config.plotting.get("formats", ["png", "pdf"])),
    )

    _write_config_snapshot(config, out_dir)
    (out_dir / "extraction_summary.json").write_text(
        json.dumps(extraction_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if bool(config.cluster_correction.get("enabled", False)):
        (out_dir / "cluster_correction_todo.txt").write_text(
            "Cluster correction requested but not implemented for this bridge map yet. "
            "Saved map arrays (*.npy) can be used for a later cluster-based pass.\n",
            encoding="utf-8",
        )

    LOGGER.info("Finished %s. Outputs: %s", config.analysis_name, out_dir)
    return InfoRateInducedLmEEGResult(
        out_dir=out_dir,
        model_results_path=model_results_path,
        diagnostics_path=diagnostics_path,
        trial_table_path=trial_table_path,
        model_input_path=model_input_path,
    )
    if "planned_response_total_information_info" in merged.columns:
        merged["planned_response_total_information"] = pd.to_numeric(
            merged["planned_response_total_information"],
            errors="coerce",
        ).fillna(
            pd.to_numeric(merged["planned_response_total_information_info"], errors="coerce")
        )

    if pd.to_numeric(merged["planned_response_total_information"], errors="coerce").isna().all():
        raise ValueError(
            "Missing required planned_response_total_information after joining neural and behaviour tables. "
            "Expected in epochs metadata or behaviour riskset (e.g., expected_total_info/actual_total_info). "
            f"Expected source file: {config.behaviour_riskset_path}"
        )
