"""Pooled lmeEEG analysis pipeline for CAS.

Handles both evoked and induced modalities. For induced models, iterates
over all frequency bands specified in config (``induced_epochs.bands``).
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import importlib
import logging
import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from patsy import dmatrix

LOGGER = logging.getLogger(__name__)

_RANDOM_EFFECT_RE = re.compile(r"\+\s*\(\s*1\s*\|\s*([^)]+?)\s*\)")


def _configure_mne_runtime() -> None:
    """Configure MNE for sandboxed and reproducible inference runs.

    MNE's optional numba-accelerated cluster helpers currently fail on this
    Python 3.13 stack during TFCE/cluster inference, so force the pure-Python
    fallback before importing any MNE modules.
    """
    from lmeeeg.backends.correction._regression import configure_mne_runtime

    configure_mne_runtime()
    os.environ["MNE_USE_NUMBA"] = "false"


def _patch_mne_cluster_level() -> None:
    """Force pure-Python cluster helper functions even if MNE was imported early."""
    _configure_mne_runtime()

    try:
        cluster_level = importlib.import_module("mne.stats.cluster_level")
    except Exception:
        return

    def _masked_sum(x, c):
        return np.sum(x[c])

    def _masked_sum_power(x, c, t_power):
        return np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)

    cluster_level._masked_sum = _masked_sum
    cluster_level._masked_sum_power = _masked_sum_power


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_lmeeeg_config(config_path: str | Path) -> dict[str, Any]:
    """Load the lmeeeg configuration from a YAML file.

    Supports both wrapped (``lmeeeg:`` top-level key) and flat layouts.
    If an ``induced_epochs`` section exists at the top level it is merged in.
    """
    path = Path(config_path)
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    section = dict(payload.get("lmeeeg", payload))
    if "induced_epochs" in payload and "induced_epochs" not in section:
        section["induced_epochs"] = payload["induced_epochs"]
    return section


# ---------------------------------------------------------------------------
# Epoch / metadata helpers
# ---------------------------------------------------------------------------


def load_epochs_with_metadata(
    epochs_path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
):
    """Load MNE epochs and an aligned metadata table."""
    _configure_mne_runtime()
    import mne

    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose="ERROR")
    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
    elif epochs.metadata is not None:
        metadata = epochs.metadata.copy().reset_index(drop=True)
    else:
        metadata = pd.DataFrame(index=range(len(epochs)))
    return epochs, metadata


def select_epochs_from_config(
    epochs,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    """Apply ``selection.event_type`` and ``selection.metadata_query`` filters."""
    selection = dict(config.get("selection") or {})
    event_type = selection.get("event_type")
    metadata_query = selection.get("metadata_query")

    if event_type is not None and "event_type" in metadata.columns:
        mask = metadata["event_type"].astype(str) == str(event_type)
        epochs = epochs[mask.to_numpy()]
        metadata = metadata.loc[mask].reset_index(drop=True)

    if metadata_query:
        mask = metadata.eval(metadata_query)
        epochs = epochs[mask.to_numpy()]
        metadata = metadata.loc[mask].reset_index(drop=True)

    if len(epochs) == 0:
        raise ValueError(
            f"Epoch selection resulted in zero rows (event_type={event_type!r}, "
            f"metadata_query={metadata_query!r})."
        )
    return epochs, metadata


def _augment_lmeeeg_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Derive convenience columns used by lmeEEG models.

    * ``duration_s`` — resolved from ``fpp_duration`` or ``spp_duration``
      depending on the row's ``event_family``.
    * ``fpp_class_1``, ``fpp_class_2``, ``spp_class_1``, ``spp_class_2`` —
      label hierarchy splits.
    """
    df = metadata.copy()

    # duration_s from event_family + family-specific duration column
    if "event_family" in df.columns:
        duration = pd.Series(np.nan, index=df.index, dtype=float)
        for family in ("fpp", "spp"):
            col = f"{family}_duration"
            if col in df.columns:
                mask = df["event_family"].astype(str) == family
                duration.loc[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce")
        df["duration_s"] = duration

    # label class splits: FPP_RFC_DECL -> class_1=RFC, class_2=DECL
    for prefix in ("fpp", "spp"):
        col = f"{prefix}_label"
        if col not in df.columns:
            continue
        parts = df[col].astype(str).str.split("_", expand=True)
        if parts.shape[1] >= 3:
            df[f"{prefix}_class_1"] = parts[1]
            df[f"{prefix}_class_2"] = parts[2]

    return df


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_EPOCHS_PATH_RE = re.compile(
    r"sub-(?P<subject>[^_/]+)_task-(?P<task>[^_/]+)_run-(?P<run>[^_/]+)"
)


def _row_from_epochs_path(epochs_path: str | Path) -> dict[str, str]:
    """Extract subject_id, task, and run from a BIDS-style epochs filename."""
    match = _EPOCHS_PATH_RE.search(str(epochs_path))
    if match is None:
        raise ValueError(f"Cannot parse subject/task/run from epochs path: {epochs_path}")
    return {
        "subject_id": f"sub-{match.group('subject')}",
        "task": match.group("task"),
        "run": match.group("run"),
    }


def _resolve_pooled_source_paths(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    epochs_path: str | Path,
    band_name: str | None = None,
) -> tuple[Path, Path | None]:
    """Resolve the epochs and optional metadata paths for one input file.

    For evoked models the original *epochs_path* is returned unchanged.
    For induced models the path is redirected to the pre-computed induced
    epochs directory for the requested *band_name*.
    """
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    modality = str(model_cfg.get("modality", "evoked")).strip().lower()

    if modality == "induced" and band_name is not None:
        out_dir = Path(runtime_config["paths"]["out_dir"])
        row = _row_from_epochs_path(epochs_path)
        subject_id = row["subject_id"]
        induced_dir = out_dir / "induced_epochs" / band_name / subject_id
        return (induced_dir / "epochs-time_s.fif", induced_dir / "metadata-time_s.csv")

    return (Path(epochs_path), None)


# ---------------------------------------------------------------------------
# Band resolution
# ---------------------------------------------------------------------------


def _resolve_induced_band_names(config: dict[str, Any]) -> list[str]:
    """Return the list of frequency band names from config."""
    induced_cfg = dict(config.get("induced_epochs") or {})
    bands = induced_cfg.get("bands")
    if bands is None:
        return ["theta"]
    if not isinstance(bands, list) or not bands:
        raise ValueError("`induced_epochs.bands` must be a non-empty list.")
    return [str(b) for b in bands]


# ---------------------------------------------------------------------------
# Trial data
# ---------------------------------------------------------------------------


def build_lmeeeg_trial_data_from_arrays(
    *,
    eeg_data: np.ndarray,
    metadata_df: pd.DataFrame | None = None,
    channel_names: list[str],
    times: np.ndarray,
):
    """Build a lightweight trial-data namespace for downstream fitting."""
    from types import SimpleNamespace

    return SimpleNamespace(
        eeg_data=np.asarray(eeg_data, dtype=float),
        channel_names=list(channel_names),
        times=np.asarray(times, dtype=float),
        trial_metadata=metadata_df,
    )


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------


def _prepare_model_inputs(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    eeg_data: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply model eligibility rules and align rows with the fixed-effects design."""
    from lmeeeg.core.formulas import parse_mixed_formula

    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    _, formula_rhs, group_column = _resolve_model_formula(
        runtime_config,
        model_name=model_name,
    )

    prepared = metadata.reset_index(drop=True).copy()
    prepared = _apply_model_design(prepared, runtime_config, model_name=model_name)
    keep_mask = pd.Series(True, index=prepared.index, dtype=bool)

    eligibility = dict(model_cfg.get("eligibility") or {})
    for column in eligibility.get("not_null") or []:
        if column in prepared.columns:
            keep_mask &= prepared[column].notna()

    for column, minimum in dict(eligibility.get("min_values") or {}).items():
        if column in prepared.columns:
            values = pd.to_numeric(prepared[column], errors="coerce")
            keep_mask &= values.notna() & (values >= float(minimum))

    prepared = prepared.loc[keep_mask].reset_index(drop=True)
    eeg_prepared = np.asarray(eeg_data)[keep_mask.to_numpy()]

    for column in model_cfg.get("standardize") or []:
        if column not in prepared.columns:
            continue
        values = pd.to_numeric(prepared[column], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            prepared[column] = 0.0
        else:
            prepared[column] = (values - mean) / std

    formula = f"y ~ {formula_rhs} + (1|{group_column})"
    parsed_formula = parse_mixed_formula(formula)
    design_metadata = prepared.copy()
    design_metadata["y"] = 0.0
    fixed_design = dmatrix(
        parsed_formula.fixed_formula.split("~", maxsplit=1)[1],
        design_metadata,
        return_type="dataframe",
    )

    retained_index = fixed_design.index.to_numpy(dtype=int)
    prepared = prepared.iloc[retained_index].reset_index(drop=True)
    eeg_prepared = eeg_prepared[retained_index]

    if len(prepared) == 0:
        raise ValueError(f"Model '{model_name}' has zero eligible observations after metadata cleanup.")

    return eeg_prepared, prepared


# ---------------------------------------------------------------------------
# Fitting and inference
# ---------------------------------------------------------------------------


def _fit_one_model(
    runtime_config: dict[str, Any],
    trial_data,
    *,
    model_name: str,
    band_name: str | None = None,
) -> dict[str, Any]:
    """Fit a mass-univariate LME model and save outputs."""
    from lmeeeg import fit_lmm_mass_univariate

    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    formula, _, group_column = _resolve_model_formula(runtime_config, model_name=model_name)

    metadata = trial_data.trial_metadata.reset_index(drop=True).copy()
    variable_types: dict[str, str] = {}
    for col in metadata.columns:
        if col == group_column:
            variable_types[col] = "group"
        elif pd.api.types.is_numeric_dtype(metadata[col]):
            variable_types[col] = "numeric"
        else:
            variable_types[col] = "categorical"

    fit_result = fit_lmm_mass_univariate(
        eeg=trial_data.eeg_data,
        metadata=metadata,
        formula=formula,
        variable_types=variable_types,
    )

    # Persist artifacts
    output_dir = _model_output_dir(runtime_config, model_name=model_name, band_name=band_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    column_names = list(fit_result.design_spec.fixed_column_names)
    betas = np.stack(
        [np.asarray(fit_result.ols_betas[c], dtype=float) for c in column_names],
        axis=0,
    )
    t_values = np.stack(
        [np.asarray(fit_result.ols_t_values[c], dtype=float) for c in column_names],
        axis=0,
    )

    np.save(output_dir / "betas.npy", betas)
    np.save(output_dir / "t_values.npy", t_values)
    np.save(output_dir / "times.npy", trial_data.times)
    (output_dir / "channel_names.json").write_text(
        json.dumps(list(trial_data.channel_names)) + "\n", encoding="utf-8"
    )
    (output_dir / "column_names.json").write_text(
        json.dumps(column_names) + "\n", encoding="utf-8"
    )
    normalized_column_names = [_normalize_effect_name(name) for name in column_names]
    (output_dir / "normalized_column_names.json").write_text(
        json.dumps(normalized_column_names) + "\n", encoding="utf-8"
    )
    effect_name_map = {
        "raw_to_normalized": dict(zip(column_names, normalized_column_names, strict=True)),
        "normalized_to_raw": dict(zip(normalized_column_names, column_names, strict=True)),
    }
    (output_dir / "effect_name_map.json").write_text(
        json.dumps(effect_name_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    per_effect_outputs: dict[str, dict[str, str]] = {}
    for raw_name, normalized_name in zip(column_names, normalized_column_names, strict=True):
        beta_path = output_dir / f"{normalized_name}_beta.npy"
        t_path = output_dir / f"{normalized_name}_t_values.npy"
        np.save(beta_path, np.asarray(fit_result.ols_betas[raw_name], dtype=float))
        np.save(t_path, np.asarray(fit_result.ols_t_values[raw_name], dtype=float))
        per_effect_outputs[normalized_name] = {
            "raw_name": raw_name,
            "beta": str(beta_path),
            "t_values": str(t_path),
        }

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "n_trials_used": int(trial_data.eeg_data.shape[0]),
        "n_channels": len(trial_data.channel_names),
        "n_times": int(trial_data.times.shape[0]),
        "betas_shape": list(betas.shape),
        "per_effect_outputs": per_effect_outputs,
        "summary_output": str(output_dir / "summary.json"),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Attach fit_result for inference
    summary["_fit_result"] = fit_result
    return summary


def _resolve_test_effects(fixed_column_names: list[str], predictor: str) -> list[str]:
    """Map a configured predictor to exact fixed-effect columns."""
    if predictor in fixed_column_names:
        return [predictor]

    for name in fixed_column_names:
        if _normalize_effect_name(name) == predictor:
            return [name]

    prefix = f"{predictor}["
    matches = [name for name in fixed_column_names if name.startswith(prefix)]
    if matches:
        return matches

    prefix_matches = [
        name for name in fixed_column_names if _normalize_effect_name(name).startswith(predictor)
    ]
    if prefix_matches:
        return prefix_matches

    raise ValueError(
        f"Unknown effect '{predictor}'. Available fixed effects: {', '.join(fixed_column_names)}"
    )


def _write_stat_map_csv(
    values: np.ndarray,
    *,
    channel_names: list[str],
    times: np.ndarray,
    output_path: Path,
    value_column: str,
) -> None:
    """Write a channel x time map as a long-form CSV."""
    value_array = np.asarray(values, dtype=float)
    if value_array.shape != (len(channel_names), int(times.shape[0])):
        raise ValueError(
            "Stat map shape does not match channel/time axes: "
            f"{value_array.shape} vs ({len(channel_names)}, {int(times.shape[0])})"
        )

    frame = pd.DataFrame(
        {
            value_column: value_array.reshape(-1),
            "channel": np.repeat(np.asarray(channel_names, dtype=object), int(times.shape[0])),
            "time": np.tile(np.asarray(times, dtype=float), len(channel_names)),
        }
    )
    frame.to_csv(output_path, index=False)


def _run_permutation_inference(
    fit_result,
    *,
    effect_name: str,
    correction: str,
    n_permutations: int,
    seed: int,
    tail: int,
    threshold: float | dict[str, float] | None,
    adjacency,
):
    """Run permutation inference and return a skipped status on backend failures."""
    if correction in {"cluster", "tfce"}:
        _patch_mne_cluster_level()
    from lmeeeg import permute_fixed_effect

    try:
        return {
            "status": "ok",
            "inference_result": permute_fixed_effect(
                fit_result,
                effect=effect_name,
                correction=correction,
                n_permutations=n_permutations,
                seed=seed,
                tail=tail,
                threshold=threshold,
                adjacency=adjacency,
            ),
            "correction": correction,
        }
    except Exception as error:
        error_traceback = traceback.format_exc()
        LOGGER.exception(
            "Inference failed for effect=%s with correction=%s; skipping effect. Original error: %s",
            effect_name,
            correction,
            error,
        )
        return {
            "status": "skipped",
            "correction": correction,
            "error": f"{type(error).__name__}: {error}",
            "traceback": error_traceback,
        }


def _run_model_inference(
    runtime_config: dict[str, Any],
    trial_data,
    *,
    model_name: str,
    band_name: str | None = None,
    fit_result=None,
) -> list[dict[str, Any]]:
    """Run permutation-based inference for each test predictor."""
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    test_cfg = dict(lmeeeg_cfg.get("test") or {})
    test_predictors = list(model_cfg.get("test_predictors") or [])

    if not test_predictors:
        return []

    output_dir = _model_output_dir(runtime_config, model_name=model_name, band_name=band_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    correction = str(test_cfg.get("method", "cluster")).lower()
    n_permutations = int(test_cfg.get("n_permutations", 1024))
    tail = int(test_cfg.get("tail", 0))
    seed = int(test_cfg.get("seed", 0))

    threshold: float | dict[str, float] | None = None
    if correction == "tfce":
        tfce_cfg = test_cfg.get("tfce_threshold") or {}
        threshold = {
            "start": float(tfce_cfg.get("start", 0.0)),
            "step": float(tfce_cfg.get("step", 0.2)),
        }
    elif correction == "cluster":
        from scipy.stats import t as t_dist

        alpha = float(test_cfg.get("cluster_forming_alpha", 0.05))
        n_obs = int(trial_data.eeg_data.shape[0])
        n_predictors = len(model_cfg.get("formula", "").split("+"))
        dof = max(n_obs - n_predictors - 1, 1)
        if tail == 0:
            threshold = float(t_dist.ppf(1 - alpha / 2, df=dof))
        else:
            threshold = float(t_dist.ppf(1 - alpha, df=dof))

    adjacency = None
    if correction in {"cluster", "tfce"}:
        adjacency = _build_adjacency(
            trial_data.channel_names,
            int(trial_data.times.shape[0]),
            test_cfg,
        )

    # Reuse the main fit when available so inference does not refit the same model.
    if fit_result is None:
        fit_result = _refit_for_inference(runtime_config, trial_data, model_name=model_name)

    fixed_column_names = list(fit_result.design_spec.fixed_column_names)
    results: list[dict[str, Any]] = []
    for predictor in test_predictors:
        effect_names = _resolve_test_effects(fixed_column_names, predictor)
        for effect_name in effect_names:
            LOGGER.info(
                "Running inference: model=%s band=%s predictor=%s effect=%s",
                model_name,
                band_name,
                predictor,
                effect_name,
            )
            inference_status = _run_permutation_inference(
                fit_result,
                effect_name=effect_name,
                correction=correction,
                n_permutations=n_permutations,
                seed=seed,
                tail=tail,
                threshold=threshold,
                adjacency=adjacency,
            )

            if inference_status["status"] != "ok":
                results.append(
                    {
                        "effect": effect_name,
                        "requested_predictor": predictor,
                        "status": "skipped",
                        "correction": inference_status["correction"],
                        "error": inference_status["error"],
                    }
                )
                continue

            inference_result = inference_status["inference_result"]
            used_correction = inference_status["correction"]

            safe_effect_name = _normalize_effect_name(effect_name)
            observed_path = output_dir / f"{safe_effect_name}_observed.npy"
            corrected_p_path = output_dir / f"{safe_effect_name}_corrected_p.npy"
            corrected_p_csv_path = output_dir / f"{safe_effect_name}_corrected_p.csv"
            np.save(observed_path, inference_result.observed_statistic)
            np.save(corrected_p_path, inference_result.corrected_p_values)
            _write_stat_map_csv(
                inference_result.corrected_p_values,
                channel_names=list(trial_data.channel_names),
                times=np.asarray(trial_data.times, dtype=float),
                output_path=corrected_p_csv_path,
                value_column="p_values",
            )

            corrected_p = np.asarray(inference_result.corrected_p_values, dtype=float)
            results.append(
                {
                    "effect": effect_name,
                    "normalized_effect": safe_effect_name,
                    "requested_predictor": predictor,
                    "status": "ok",
                    "correction": used_correction,
                    "observed_statistic": str(observed_path),
                    "corrected_p_values": str(corrected_p_path),
                    "corrected_p_values_csv": str(corrected_p_csv_path),
                    "min_corrected_p": float(np.nanmin(corrected_p)) if corrected_p.size else 1.0,
                    "n_significant_p_lt_0_05": int(np.sum(corrected_p < 0.05)),
                }
            )

    return results


def _refit_for_inference(runtime_config, trial_data, *, model_name):
    """Return the lmeeeg FitResult object needed by permute_fixed_effect."""
    from lmeeeg import fit_lmm_mass_univariate

    formula, _, group_column = _resolve_model_formula(runtime_config, model_name=model_name)

    metadata = trial_data.trial_metadata.reset_index(drop=True).copy()
    variable_types: dict[str, str] = {}
    for col in metadata.columns:
        if col == group_column:
            variable_types[col] = "group"
        elif pd.api.types.is_numeric_dtype(metadata[col]):
            variable_types[col] = "numeric"
        else:
            variable_types[col] = "categorical"

    return fit_lmm_mass_univariate(
        eeg=trial_data.eeg_data,
        metadata=metadata,
        formula=formula,
        variable_types=variable_types,
    )


def _build_adjacency(channel_names: list[str], n_times: int, test_cfg: dict[str, Any]):
    """Build an explicit spatiotemporal adjacency matrix for cluster tests."""
    adjacency_type = str(test_cfg.get("adjacency", "none")).lower()
    if adjacency_type == "none":
        return None

    _patch_mne_cluster_level()
    import mne

    montage_name = str(test_cfg.get("montage", "biosemi64"))
    info = mne.create_info(list(channel_names), sfreq=256.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="warn")
    spatial_adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    return mne.stats.combine_adjacency(int(n_times), spatial_adjacency)


def _model_output_dir(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    band_name: str | None = None,
) -> Path:
    """Return the output directory for one model (optionally band-specific)."""
    out_dir = Path(runtime_config["paths"]["out_dir"])
    analysis_name = str((runtime_config.get("lmeeeg") or {}).get("analysis_name", "")).strip()
    lmeeeg_root = out_dir / "lmeeeg"
    if analysis_name:
        lmeeeg_root = lmeeeg_root / analysis_name
    if band_name is not None:
        return lmeeeg_root / "induced" / band_name / model_name
    return lmeeeg_root / model_name


def _resolve_project_out_dir(output_dir: str | Path) -> Path:
    """Return the project-level OUT_DIR from an lmeeeg analysis output path."""
    output_dir = Path(output_dir)
    if output_dir.name == "lmeeeg":
        return output_dir.parent
    if output_dir.parent.name == "lmeeeg":
        return output_dir.parent.parent
    return output_dir.parent


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pooled_lmeeeg_analysis(
    epochs_paths: list[str | Path],
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run pooled lmeEEG analysis across multiple epoch files.

    For induced models every frequency band listed in
    ``induced_epochs.bands`` is fitted and tested independently.
    """
    config = load_lmeeeg_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = _resolve_project_out_dir(output_dir)

    runtime_config: dict[str, Any] = {
        "paths": {"out_dir": str(out_dir)},
        "lmeeeg": config,
        "runtime": {"config_path": str(config_path)},
    }
    if "induced_epochs" in config:
        runtime_config["induced_epochs"] = config["induced_epochs"]

    models_cfg = dict(config.get("models") or {})
    model_summaries: list[dict[str, Any]] = []

    for model_name, model_cfg in models_cfg.items():
        model_cfg = dict(model_cfg or {})
        modality = str(model_cfg.get("modality", "evoked")).strip().lower()

        if modality == "induced":
            band_names = _resolve_induced_band_names(config)
            for band_name in band_names:
                summary = _run_pooled_model(
                    runtime_config=runtime_config,
                    model_name=model_name,
                    epochs_paths=epochs_paths,
                    band_name=band_name,
                )
                model_summaries.append(summary)
        else:
            summary = _run_pooled_model(
                runtime_config=runtime_config,
                model_name=model_name,
                epochs_paths=epochs_paths,
                band_name=None,
            )
            model_summaries.append(summary)

    analysis_summary: dict[str, Any] = {
        "status": "ok",
        "epochs_paths": [str(p) for p in epochs_paths],
        "output_dir": str(output_dir),
        "n_files_input": len(epochs_paths),
        "models": model_summaries,
    }

    summary_path = output_dir / "lmeeeg_analysis_summary.json"
    summary_path.write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return analysis_summary


def _run_pooled_model(
    *,
    runtime_config: dict[str, Any],
    model_name: str,
    epochs_paths: list[str | Path],
    band_name: str | None,
) -> dict[str, Any]:
    """Load, pool, fit, and infer for one model (and optionally one band)."""
    pooled_eeg: list[np.ndarray] = []
    pooled_metadata: list[pd.DataFrame] = []
    ref_ch_names: list[str] | None = None
    ref_times: np.ndarray | None = None

    for ep_path in epochs_paths:
        source_path, metadata_csv = _resolve_pooled_source_paths(
            runtime_config,
            model_name=model_name,
            epochs_path=ep_path,
            band_name=band_name,
        )

        epochs, metadata = load_epochs_with_metadata(source_path, metadata_csv=metadata_csv)
        metadata = _augment_lmeeeg_metadata(metadata)

        # Ensure subject_id is present
        if "subject_id" not in metadata.columns:
            try:
                row = _row_from_epochs_path(ep_path)
                metadata["subject_id"] = row["subject_id"]
            except ValueError:
                pass

        eeg = epochs.get_data(copy=True)
        pooled_eeg.append(eeg)
        pooled_metadata.append(metadata)

        ch_names = list(epochs.ch_names)
        times = np.asarray(epochs.times, dtype=float)
        if ref_ch_names is None:
            ref_ch_names = ch_names
            ref_times = times
        else:
            if ch_names != ref_ch_names:
                raise ValueError("Channel mismatch across pooled epochs.")
            if not np.allclose(times, ref_times):
                raise ValueError("Time axis mismatch across pooled epochs.")

    if not pooled_eeg:
        raise ValueError("No epochs were loaded.")

    combined_eeg = np.concatenate(pooled_eeg, axis=0)
    combined_metadata = pd.concat(pooled_metadata, axis=0, ignore_index=True)

    # Apply selection (event_type, metadata_query) before building trial data
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    merged_selection = dict(lmeeeg_cfg.get("selection") or {})
    merged_selection.update(dict(model_cfg.get("selection") or {}))
    selection_config = {"selection": merged_selection}

    if merged_selection:
        from types import SimpleNamespace

        class _IndexableArray:
            """Thin wrapper so select_epochs_from_config can index eeg data."""

            def __init__(self, arr):
                self._arr = arr
                self.selection = np.arange(arr.shape[0], dtype=int)

            def __len__(self):
                return self._arr.shape[0]

            def __getitem__(self, item):
                idx = np.asarray(item)
                if idx.dtype == bool:
                    idx = np.flatnonzero(idx)
                new = _IndexableArray(self._arr[idx])
                new.selection = self.selection[idx]
                return new

        wrapper = _IndexableArray(combined_eeg)
        wrapper, combined_metadata = select_epochs_from_config(
            wrapper, combined_metadata, selection_config
        )
        combined_eeg = wrapper._arr

    combined_eeg, combined_metadata = _prepare_model_inputs(
        runtime_config,
        model_name=model_name,
        eeg_data=combined_eeg,
        metadata=combined_metadata,
    )
    _write_model_design_artifacts(
        runtime_config,
        model_name=model_name,
        band_name=band_name,
        design_table=combined_metadata,
    )

    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=combined_eeg,
        metadata_df=combined_metadata,
        channel_names=ref_ch_names,
        times=ref_times,
    )

    fit_summary = _fit_one_model(
        runtime_config, trial_data, model_name=model_name, band_name=band_name
    )

    inference = _run_model_inference(
        runtime_config,
        trial_data,
        model_name=model_name,
        band_name=band_name,
        fit_result=fit_summary.get("_fit_result"),
    )

    # Strip internal objects before serialising
    fit_summary.pop("_fit_result", None)

    entry: dict[str, Any] = {
        "model_name": model_name,
        "fit": fit_summary,
        "inference": inference,
    }
    if band_name is not None:
        entry["band_name"] = band_name
    return entry


def _resolve_model_formula(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> tuple[str, str, str]:
    """Return the backend formula, fixed-effect RHS, and random intercept group."""
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    raw_formula = str(model_cfg.get("formula", "~ 1")).strip()
    rhs = raw_formula.split("~", maxsplit=1)[-1].strip() if "~" in raw_formula else raw_formula
    group_column = str(
        model_cfg.get("random_effect_group")
        or (lmeeeg_cfg.get("random_effects") or {}).get("group", "subject_id")
    ).strip()
    random_match = _RANDOM_EFFECT_RE.search(rhs)
    if random_match is not None:
        group_column = random_match.group(1).strip()
        rhs = _RANDOM_EFFECT_RE.sub("", rhs)
        rhs = re.sub(r"\s+", " ", rhs).strip()
        rhs = re.sub(r"\+\s*$", "", rhs).strip()
    formula_rhs = rhs or "1"
    return f"y ~ {formula_rhs} + (1|{group_column})", formula_rhs, group_column


def _normalize_effect_name(name: str) -> str:
    """Normalize patsy/lmeEEG column names to stable file-safe effect names."""
    text = str(name).strip()
    text = re.sub(r"\[T\.([^\]]+)\]", r"\1", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return text or "effect"


def _apply_model_design(
    metadata: pd.DataFrame,
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> pd.DataFrame:
    """Apply config-driven design-table mappings, validation, and z-scoring."""
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    design_cfg = dict(lmeeeg_cfg.get("design") or {})
    design_cfg.update(dict(model_cfg.get("design") or {}))
    if not design_cfg:
        return metadata

    prepared = metadata.copy()

    for target, source in dict(design_cfg.get("column_mapping") or {}).items():
        if source in prepared.columns:
            prepared[target] = prepared[source]

    for column, mapping in dict(design_cfg.get("value_mapping") or {}).items():
        if column not in prepared.columns:
            continue
        values = prepared[column]
        mapped = values.map(mapping)
        prepared[column] = mapped.where(mapped.notna(), values)

    required_columns = [str(value) for value in design_cfg.get("required_columns") or []]
    missing_columns = [column for column in required_columns if column not in prepared.columns]
    if missing_columns:
        raise ValueError(
            f"Model '{model_name}' is missing required design columns: {', '.join(sorted(missing_columns))}"
        )

    invalid_reasons: list[str] = []
    _require_non_missing(prepared, "subject", invalid_reasons)
    _require_non_missing(prepared, "run", invalid_reasons)

    pair_position = "pair_position"
    if pair_position in prepared.columns:
        levels = prepared[pair_position].astype(str)
        valid_levels = {"FPP", "SPP"}
        invalid_levels = sorted(
            {value for value in levels.dropna().unique().tolist() if value not in valid_levels}
        )
        if invalid_levels:
            invalid_reasons.append(
                f"`pair_position` contains invalid levels: {', '.join(invalid_levels)}"
            )
        observed_levels = {value for value in levels.dropna().unique().tolist() if value in valid_levels}
        if observed_levels != valid_levels:
            invalid_reasons.append(
                f"`pair_position` must contain both FPP and SPP after filtering; observed: {', '.join(sorted(observed_levels)) or 'none'}"
            )

    for column in list(((design_cfg.get("predictors") or {}).get("continuous") or [])):
        if column not in prepared.columns:
            continue
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.isna().any():
            invalid_reasons.append(f"`{column}` must be numeric and non-missing.")
        prepared[column] = numeric

    _require_finite(prepared, "latency", invalid_reasons, label="`latency` must be finite.")
    _require_positive(prepared, "event_duration", invalid_reasons, label="`event_duration` must be positive.")
    _require_finite(
        prepared,
        "time_within_run",
        invalid_reasons,
        label="`time_within_run` must be finite.",
    )

    if invalid_reasons:
        raise ValueError(
            f"Model '{model_name}' design validation failed:\n- " + "\n- ".join(invalid_reasons)
        )

    for source, target in dict(design_cfg.get("zscore") or {}).items():
        if source not in prepared.columns:
            continue
        values = pd.to_numeric(prepared[source], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            prepared[target] = 0.0
        else:
            prepared[target] = (values - mean) / std

    reference_levels = {
        str(column): str(value)
        for column, value in dict(design_cfg.get("reference_levels") or {}).items()
    }
    for column in list(((design_cfg.get("predictors") or {}).get("categorical") or [])):
        if column not in prepared.columns:
            continue
        series = prepared[column].astype(str)
        if column in reference_levels:
            reference = reference_levels[column]
            categories = [reference] + [value for value in pd.unique(series) if value != reference]
            if reference not in categories:
                raise ValueError(
                    f"Model '{model_name}' expected reference level {reference!r} for {column!r}."
                )
            prepared[column] = pd.Categorical(series, categories=categories, ordered=True)
        else:
            prepared[column] = pd.Categorical(series)

    return prepared


def _require_non_missing(frame: pd.DataFrame, column: str, reasons: list[str]) -> None:
    if column not in frame.columns:
        return
    if frame[column].isna().any():
        reasons.append(f"`{column}` must be non-missing.")


def _require_finite(frame: pd.DataFrame, column: str, reasons: list[str], *, label: str) -> None:
    if column not in frame.columns:
        return
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        reasons.append(label)
    frame[column] = values


def _require_positive(frame: pd.DataFrame, column: str, reasons: list[str], *, label: str) -> None:
    if column not in frame.columns:
        return
    values = pd.to_numeric(frame[column], errors="coerce")
    if (values <= 0).any() or values.isna().any():
        reasons.append(label)
    frame[column] = values


def _write_model_design_artifacts(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    band_name: str | None,
    design_table: pd.DataFrame,
) -> None:
    """Persist the auditable design table and run metadata for one fitted model."""
    output_dir = _model_output_dir(runtime_config, model_name=model_name, band_name=band_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_table.to_csv(output_dir / "design_table.csv", index=False)

    _, formula_rhs, group_column = _resolve_model_formula(runtime_config, model_name=model_name)
    metadata_payload = {
        "analysis_name": str((runtime_config.get("lmeeeg") or {}).get("analysis_name", "")).strip()
        or None,
        "model_name": model_name,
        "band_name": band_name,
        "config_path": str(((runtime_config.get("runtime") or {}).get("config_path")) or ""),
        "formula": f"power ~ {formula_rhs} + (1 | {group_column})",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "event_table": str(((runtime_config.get("lmeeeg") or {}).get("event_table")) or ""),
        "n_events": int(len(design_table)),
        "n_subjects": int(design_table["subject"].nunique()) if "subject" in design_table.columns else None,
        "n_runs": int(design_table["run"].nunique()) if "run" in design_table.columns else None,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:
        return None
