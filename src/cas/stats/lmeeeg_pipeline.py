"""Pooled lmeEEG analysis pipeline for CAS.

Handles both evoked and induced modalities. For induced models, iterates
over all frequency bands specified in config (``induced_epochs.bands``).
"""

from __future__ import annotations

from dataclasses import dataclass
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
_SPP_CONF_RE = re.compile(r"^SPP_CONF_", re.IGNORECASE)
_SPP_DISC_RE = re.compile(r"^SPP_DISC_", re.IGNORECASE)
_DEFAULT_DURATION_COLUMN = "duration_s"
_DEFAULT_DURATION_BINS_ACTION = "raise"


@dataclass(frozen=True)
class PreparedModelInputs:
    """Prepared metadata, EEG rows, and resolved model specification."""

    eeg_data: np.ndarray
    metadata: pd.DataFrame
    formula: str
    formula_rhs: str
    group_column: str
    duration_artifacts: dict[str, Any]
    term_tests: list[dict[str, Any]]


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


def _emit_status(message: str) -> None:
    """Log and print a status line for verbose pipeline runs."""
    LOGGER.info(message)
    print(message, flush=True)


def _zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((numeric - mean) / std).astype(float)


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

    df = derive_fpp_spp_conf_disc_class(df)
    return df


def derive_fpp_spp_conf_disc_class(
    metadata: pd.DataFrame,
    *,
    drop_other: bool = False,
) -> pd.DataFrame:
    """Annotate rows with FPP/SPP_CONF/SPP_DISC class and duration covariates."""
    df = metadata.copy()

    family = _derive_anchor_family(df)
    if family is None:
        return df

    duration = _resolve_anchor_duration(df, family)

    class_3 = pd.Series("OTHER", index=df.index, dtype=object)
    family_upper = family.str.upper()
    spp_labels = df["spp_label"].astype(str) if "spp_label" in df.columns else pd.Series("", index=df.index, dtype=object)

    class_3.loc[family_upper == "FPP"] = "FPP"
    spp_mask = family_upper == "SPP"
    class_3.loc[spp_mask & spp_labels.str.match(_SPP_CONF_RE)] = "SPP_CONF"
    class_3.loc[spp_mask & spp_labels.str.match(_SPP_DISC_RE)] = "SPP_DISC"

    df["class_3"] = class_3
    df["duration"] = duration

    positive_duration = duration.where(duration > 0.0)
    df["log_duration"] = np.log(positive_duration)
    df["log_duration_within_class"] = (
        df.groupby("class_3", observed=False)["log_duration"].transform(lambda s: s - s.mean())
    )

    if drop_other:
        df = df.loc[df["class_3"] != "OTHER"].reset_index(drop=True)

    counts = df["class_3"].value_counts(dropna=False).to_dict()
    LOGGER.info("Derived class_3 counts: %s", counts)
    return df


def _derive_anchor_family(metadata: pd.DataFrame) -> pd.Series | None:
    if "event_family" in metadata.columns:
        return metadata["event_family"].astype(str)
    if "part" in metadata.columns:
        return metadata["part"].astype(str)
    if "anchor_type" in metadata.columns:
        return metadata["anchor_type"].astype(str)
    return None


def _resolve_anchor_duration(metadata: pd.DataFrame, family: pd.Series) -> pd.Series:
    duration = pd.Series(np.nan, index=metadata.index, dtype=float)

    for generic_column in ("duration", "duration_s"):
        if generic_column in metadata.columns:
            duration = pd.to_numeric(metadata[generic_column], errors="coerce")
            if duration.notna().any():
                break

    family_upper = family.str.upper()
    for anchor_name, column_name in (("FPP", "fpp_duration"), ("SPP", "spp_duration")):
        if column_name not in metadata.columns:
            continue
        mask = family_upper == anchor_name
        resolved = pd.to_numeric(metadata.loc[mask, column_name], errors="coerce")
        duration.loc[mask] = resolved.where(resolved.notna(), duration.loc[mask])

    if duration.isna().all():
        raise ValueError(
            "Unable to derive duration: expected one of `duration`, `duration_s`, "
            "`fpp_duration`, or `spp_duration`."
        )

    return duration


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
        input_cfg = dict(lmeeeg_cfg.get("input") or {})
        induced_subdir = str(input_cfg.get("induced_epochs_subdir", "induced_epochs")).strip() or "induced_epochs"
        induced_dir = out_dir / induced_subdir / band_name / subject_id
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
        eeg_data=np.asarray(eeg_data, dtype=np.float32),
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
    prepared = _prepare_model_inputs_detailed(
        runtime_config,
        model_name=model_name,
        eeg_data=eeg_data,
        metadata=metadata,
    )
    return prepared.eeg_data, prepared.metadata


def _prepare_model_inputs_detailed(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    eeg_data: np.ndarray,
    metadata: pd.DataFrame,
) -> PreparedModelInputs:
    """Apply model eligibility rules and align rows with the fixed-effects design."""
    from lmeeeg.core.formulas import parse_mixed_formula

    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})

    prepared = metadata.reset_index(drop=True).copy()
    prepared = _apply_model_design(prepared, runtime_config, model_name=model_name)
    prepared, duration_artifacts = _apply_duration_controls(
        prepared,
        runtime_config,
        model_name=model_name,
    )
    formula, formula_rhs, group_column = _resolve_model_formula(
        runtime_config,
        model_name=model_name,
        duration_artifacts=duration_artifacts,
    )
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
        prepared[column] = _zscore_series(prepared[column])

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

    term_tests = _resolve_term_tests(
        runtime_config,
        model_name=model_name,
        duration_artifacts=duration_artifacts,
        metadata=prepared,
    )
    return PreparedModelInputs(
        eeg_data=eeg_prepared,
        metadata=prepared,
        formula=formula,
        formula_rhs=formula_rhs,
        group_column=group_column,
        duration_artifacts=duration_artifacts,
        term_tests=term_tests,
    )


def _resolve_duration_controls(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    return dict(model_cfg.get("duration_controls") or {})


def _duration_feature_enabled(feature_cfg: dict[str, Any]) -> bool:
    return bool(feature_cfg.get("enabled", False))


def _resolve_primary_class_column(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    metadata: pd.DataFrame | None = None,
) -> str | None:
    controls_cfg = _resolve_duration_controls(runtime_config, model_name=model_name)
    common_support_cfg = dict(controls_cfg.get("common_support") or {})
    class_column = str(common_support_cfg.get("class_column", "")).strip()
    if class_column:
        return class_column

    model_cfg = dict((dict(runtime_config.get("lmeeeg") or {}).get("models") or {}).get(model_name) or {})
    term_tests_cfg = dict(model_cfg.get("term_tests") or {})
    class_cfg = dict(term_tests_cfg.get("class_effect") or {})
    for value in class_cfg.get("terms") or []:
        candidate = str(value).strip()
        if candidate:
            return candidate

    for candidate in ("spp_class_1", "class_3"):
        if metadata is not None and candidate in metadata.columns:
            return candidate
    return None


def _resolve_duration_support_mask(
    duration_values: pd.Series,
    *,
    controls_cfg: dict[str, Any],
) -> tuple[pd.Series, list[str]]:
    messages: list[str] = []
    values = pd.to_numeric(duration_values, errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype=float))
    keep_mask = pd.Series(finite_mask, index=duration_values.index, dtype=bool)

    if (~keep_mask).any():
        messages.append(f"Dropping {(~keep_mask).sum()} rows with non-finite duration values.")

    positive_mask = values > 0.0
    nonpositive_mask = ~positive_mask
    if nonpositive_mask.any():
        log_cfg = dict(controls_cfg.get("log_duration") or {})
        spline_cfg = dict(controls_cfg.get("spline_duration") or {})
        bins_enabled = _duration_feature_enabled(dict(controls_cfg.get("duration_bins") or {}))
        common_support_enabled = _duration_feature_enabled(dict(controls_cfg.get("common_support") or {}))
        spline_source = str(spline_cfg.get("source", "")).strip().lower()
        log_offset = float(log_cfg.get("offset_s", 0.0))
        can_keep_nonpositive = (
            _duration_feature_enabled(log_cfg)
            and log_offset > 0.0
            and not bins_enabled
            and not common_support_enabled
            and spline_source in {"", "log_duration", "z_log_duration", str(log_cfg.get("output_column", "")).strip().lower()}
            and ((values + log_offset) > 0.0).all()
        )
        if can_keep_nonpositive:
            messages.append(
                "Keeping non-positive duration rows because log-duration control uses a positive offset "
                "and no raw-duration controls require strict positivity."
            )
        else:
            dropped_nonpositive = int((keep_mask & nonpositive_mask).sum())
            keep_mask &= positive_mask
            messages.append(f"Dropping {dropped_nonpositive} rows with non-positive duration values.")

    return keep_mask, messages


def _apply_common_support(
    metadata: pd.DataFrame,
    *,
    duration_column: str,
    class_column: str,
    common_support_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lower_quantile = float(common_support_cfg.get("lower_quantile", 0.05))
    upper_quantile = float(common_support_cfg.get("upper_quantile", 0.95))
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("common_support quantiles must satisfy 0 <= lower < upper <= 1.")

    duration = pd.to_numeric(metadata[duration_column], errors="coerce")
    grouped = metadata.assign(_duration=duration).groupby(class_column, observed=False)
    lower_by_class = grouped["_duration"].quantile(lower_quantile)
    upper_by_class = grouped["_duration"].quantile(upper_quantile)
    lower = float(lower_by_class.max())
    upper = float(upper_by_class.min())
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError(
            f"Common-support bounds do not overlap for {class_column!r}: lower={lower}, upper={upper}."
        )

    before_counts = metadata[class_column].astype(str).value_counts(dropna=False).to_dict()
    keep_mask = duration.between(lower, upper, inclusive="both")
    filtered = metadata.loc[keep_mask].reset_index(drop=True)
    after_counts = filtered[class_column].astype(str).value_counts(dropna=False).to_dict()
    return filtered, {
        "class_column": class_column,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "lower_bound": lower,
        "upper_bound": upper,
        "counts_before": {str(key): int(value) for key, value in before_counts.items()},
        "counts_after": {str(key): int(value) for key, value in after_counts.items()},
    }


def _apply_duration_controls(
    metadata: pd.DataFrame,
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    controls_cfg = _resolve_duration_controls(runtime_config, model_name=model_name)
    duration_column = str(controls_cfg.get("duration_column", _DEFAULT_DURATION_COLUMN)).strip() or _DEFAULT_DURATION_COLUMN
    class_column = _resolve_primary_class_column(
        runtime_config,
        model_name=model_name,
        metadata=metadata,
    )
    artifacts: dict[str, Any] = {
        "duration_column": duration_column,
        "class_column": class_column,
        "enabled_controls": [],
        "generated_columns": [],
        "formula_terms": [],
        "term_aliases": {},
        "transforms": [],
        "common_support": None,
    }
    if duration_column not in metadata.columns:
        if any(
            _duration_feature_enabled(dict(controls_cfg.get(key) or {}))
            for key in ("log_duration", "spline_duration", "duration_bins", "common_support")
        ):
            raise ValueError(
                f"Configured duration controls for model {model_name!r} require "
                f"duration column {duration_column!r}, but it is missing."
            )
        return metadata, artifacts

    prepared = metadata.copy()
    keep_mask, messages = _resolve_duration_support_mask(prepared[duration_column], controls_cfg=controls_cfg)
    for message in messages:
        _emit_status(message)
    prepared = prepared.loc[keep_mask].reset_index(drop=True)

    log_cfg = dict(controls_cfg.get("log_duration") or {})
    if _duration_feature_enabled(log_cfg):
        output_column = str(log_cfg.get("output_column", "z_log_duration")).strip() or "z_log_duration"
        offset_s = float(log_cfg.get("offset_s", 0.0))
        duration_values = pd.to_numeric(prepared[duration_column], errors="coerce")
        transformed = np.log(duration_values + offset_s)
        prepared[output_column] = (
            _zscore_series(transformed) if bool(log_cfg.get("standardize", False)) else transformed
        )
        artifacts["enabled_controls"].append("log_duration")
        artifacts["generated_columns"].append(output_column)
        artifacts["formula_terms"].append(output_column)
        artifacts["term_aliases"][output_column] = [output_column]
        artifacts["transforms"].append(
            {
                "name": "log_duration",
                "source_column": duration_column,
                "output_columns": [output_column],
                "offset_s": offset_s,
                "standardize": bool(log_cfg.get("standardize", False)),
            }
        )

    spline_cfg = dict(controls_cfg.get("spline_duration") or {})
    if _duration_feature_enabled(spline_cfg):
        output_prefix = str(spline_cfg.get("output_prefix", "spline_duration")).strip() or "spline_duration"
        source_name = str(spline_cfg.get("source", duration_column)).strip() or duration_column
        source_lookup = {
            duration_column: pd.to_numeric(prepared[duration_column], errors="coerce"),
            "raw_duration": pd.to_numeric(prepared[duration_column], errors="coerce"),
        }
        for generated_column in artifacts["generated_columns"]:
            source_lookup[generated_column] = pd.to_numeric(prepared[generated_column], errors="coerce")
        if source_name not in source_lookup:
            raise ValueError(
                f"Spline duration source {source_name!r} is unavailable for model {model_name!r}."
            )
        basis = dmatrix(
            (
                "bs(x, df="
                f"{int(spline_cfg.get('df', 4))}, degree={int(spline_cfg.get('degree', 3))}, "
                f"include_intercept={bool(spline_cfg.get('include_intercept', False))}) - 1"
            ),
            {"x": source_lookup[source_name]},
            return_type="dataframe",
        )
        basis_values = np.asarray(basis, dtype=float)
        if bool(spline_cfg.get("orthogonalize_to_source", True)):
            nuisance_columns = [np.ones(len(prepared), dtype=float), np.asarray(source_lookup[source_name], dtype=float)]
            for generated_column in artifacts["generated_columns"]:
                nuisance_columns.append(
                    pd.to_numeric(prepared[generated_column], errors="coerce").to_numpy(dtype=float)
                )
            basis_values = _residualize_columns(
                basis_values,
                nuisance=np.column_stack(nuisance_columns),
            )
            basis_rank = int(np.linalg.matrix_rank(basis_values))
            if basis_rank <= 0:
                raise ValueError(
                    f"Spline duration basis for model {model_name!r} has no remaining variation after "
                    f"orthogonalization to source {source_name!r}."
                )
            if basis_rank < basis_values.shape[1]:
                _emit_status(
                    f"Reduced spline basis for model {model_name} from {basis_values.shape[1]} "
                    f"to {basis_rank} columns after orthogonalizing against {source_name}."
                )
            basis_values, _ = np.linalg.qr(basis_values, mode="reduced")
            basis_values = basis_values[:, :basis_rank]
        basis_columns: list[str] = []
        for column_index in range(1, basis_values.shape[1] + 1):
            target_name = f"{output_prefix}_{column_index}"
            values = basis_values[:, column_index - 1]
            prepared[target_name] = (
                _zscore_series(values) if bool(spline_cfg.get("standardize", False)) else values
            )
            basis_columns.append(target_name)
        artifacts["enabled_controls"].append("spline_duration")
        artifacts["generated_columns"].extend(basis_columns)
        artifacts["formula_terms"].extend(basis_columns)
        artifacts["term_aliases"][output_prefix] = basis_columns
        artifacts["transforms"].append(
            {
                "name": "spline_duration",
                "source_column": source_name,
                "output_columns": basis_columns,
                "df": int(spline_cfg.get("df", 4)),
                "degree": int(spline_cfg.get("degree", 3)),
                "include_intercept": bool(spline_cfg.get("include_intercept", False)),
                "orthogonalize_to_source": bool(spline_cfg.get("orthogonalize_to_source", True)),
                "standardize": bool(spline_cfg.get("standardize", False)),
            }
        )

    bins_cfg = dict(controls_cfg.get("duration_bins") or {})
    if _duration_feature_enabled(bins_cfg):
        output_column = str(bins_cfg.get("output_column", "duration_bin")).strip() or "duration_bin"
        labels_prefix = str(bins_cfg.get("labels_prefix", "dur_q")).strip() or "dur_q"
        q = int(bins_cfg.get("n_bins", 5))
        binned = pd.qcut(
            pd.to_numeric(prepared[duration_column], errors="coerce"),
            q=q,
            labels=None,
            duplicates="drop",
        )
        observed_bin_count = int(len(binned.cat.categories))
        categories = [f"{labels_prefix}{index}" for index in range(1, observed_bin_count + 1)]
        min_unique_bins = int(bins_cfg.get("min_unique_bins", 3))
        if len(categories) < min_unique_bins:
            action = str(
                bins_cfg.get("on_insufficient_unique_bins", _DEFAULT_DURATION_BINS_ACTION)
            ).strip().lower()
            message = (
                f"Duration binning for model {model_name!r} produced only {len(categories)} unique bins; "
                f"minimum is {min_unique_bins}."
            )
            if action == "skip":
                _emit_status(message + " Skipping duration-bin control.")
            else:
                raise ValueError(message)
        else:
            prepared[output_column] = binned.cat.rename_categories(categories)
            artifacts["enabled_controls"].append("duration_bins")
            artifacts["generated_columns"].append(output_column)
            artifacts["formula_terms"].append(output_column)
            artifacts["term_aliases"][output_column] = [output_column]
            artifacts["transforms"].append(
                {
                    "name": "duration_bins",
                    "source_column": duration_column,
                    "output_columns": [output_column],
                    "n_bins_requested": q,
                    "n_bins_observed": len(categories),
                    "strategy": str(bins_cfg.get("strategy", "quantile")),
                    "labels": categories,
                }
            )

    common_support_cfg = dict(controls_cfg.get("common_support") or {})
    if _duration_feature_enabled(common_support_cfg):
        if class_column is None or class_column not in prepared.columns:
            raise ValueError("Common-support filtering requires a valid class column.")
        prepared, support_summary = _apply_common_support(
            prepared,
            duration_column=duration_column,
            class_column=class_column,
            common_support_cfg=common_support_cfg,
        )
        artifacts["common_support"] = support_summary
        _emit_status(
            "Common support filtering retained "
            f"{len(prepared)} rows within [{support_summary['lower_bound']:.6f}, "
            f"{support_summary['upper_bound']:.6f}] for {class_column}."
        )

    return prepared, artifacts


def _formula_rhs_terms(formula_rhs: str) -> list[str]:
    return [term.strip() for term in formula_rhs.split("+") if term.strip()]


def _merge_formula_terms(formula_rhs: str, extra_terms: list[str]) -> str:
    merged_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in _formula_rhs_terms(formula_rhs) + list(extra_terms):
        normalized = term.strip()
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        merged_terms.append(normalized)
    return " + ".join(merged_terms) if merged_terms else "1"


def _residualize_columns(
    values: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
) -> np.ndarray:
    """Project columns onto the orthogonal complement of nuisance columns."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix of columns to residualize.")
    if nuisance is None:
        return matrix
    nuisance_matrix = np.asarray(nuisance, dtype=float)
    if nuisance_matrix.ndim == 1:
        nuisance_matrix = nuisance_matrix[:, None]
    if nuisance_matrix.shape[0] != matrix.shape[0]:
        raise ValueError("Nuisance and values must have the same number of rows.")
    if nuisance_matrix.shape[1] == 0:
        return matrix
    nuisance_pinv = np.linalg.pinv(nuisance_matrix)
    return matrix - nuisance_matrix @ (nuisance_pinv @ matrix)


def _resolve_term_tests(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    duration_artifacts: dict[str, Any],
    metadata: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    configured_groups = dict(model_cfg.get("term_tests") or {})
    if not configured_groups:
        test_predictors = _resolve_configured_test_predictors(runtime_config, model_name=model_name)
        return [
            {
                "group_name": predictor,
                "contrast_type": _infer_contrast_type(predictor, metadata=metadata),
                "terms": [predictor],
            }
            for predictor in test_predictors
        ]

    resolved: list[dict[str, Any]] = []
    for group_name, payload in configured_groups.items():
        group_cfg = dict(payload or {})
        if not bool(group_cfg.get("enabled", False)):
            continue
        terms = [str(value).strip() for value in group_cfg.get("terms") or [] if str(value).strip()]
        available_terms: list[str] = []
        for term in terms:
            mapped_terms = duration_artifacts.get("term_aliases", {}).get(term)
            if mapped_terms is not None:
                available_terms.append(term)
            elif metadata is None or term in metadata.columns:
                available_terms.append(term)
        if not available_terms:
            continue
        contrast_type = str(group_cfg.get("contrast_type", "")).strip()
        if not contrast_type:
            contrast_type = _infer_contrast_type(
                available_terms[0],
                metadata=metadata,
                mapped_columns=duration_artifacts.get("term_aliases", {}).get(available_terms[0]),
            )
        resolved.append(
            {
                "group_name": str(group_name),
                "contrast_type": contrast_type,
                "terms": available_terms,
            }
        )
    return resolved


def _infer_contrast_type(
    term: str,
    *,
    metadata: pd.DataFrame | None,
    mapped_columns: list[str] | None = None,
) -> str:
    if mapped_columns and len(mapped_columns) > 1:
        return "grouped_continuous"
    if metadata is not None and term in metadata.columns:
        return "continuous" if pd.api.types.is_numeric_dtype(metadata[term]) else "categorical"
    if "spline" in term:
        return "grouped_continuous"
    if "bin" in term or "class" in term:
        return "categorical"
    return "continuous"


# ---------------------------------------------------------------------------
# Fitting and inference
# ---------------------------------------------------------------------------


def _fit_one_model(
    runtime_config: dict[str, Any],
    trial_data,
    *,
    model_name: str,
    band_name: str | None = None,
    formula_override: str | None = None,
    group_column_override: str | None = None,
    term_tests_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit a mass-univariate LME model and save outputs."""
    from lmeeeg import fit_lmm_mass_univariate

    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    formula, _, group_column = _resolve_model_formula(runtime_config, model_name=model_name)
    if formula_override is not None:
        formula = formula_override
    if group_column_override is not None:
        group_column = group_column_override

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
    pd.DataFrame({"design_column": column_names}).to_csv(
        output_dir / "design_columns.csv",
        index=False,
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

    term_tests = term_tests_override or _resolve_term_tests(
        runtime_config,
        model_name=model_name,
        duration_artifacts={},
        metadata=metadata,
    )
    resolved_term_tests = _resolve_term_test_effects(column_names, term_tests)
    test_predictors = _resolve_configured_test_predictors(runtime_config, model_name=model_name)
    resolved_test_effects: dict[str, list[str]] = {}
    for predictor in test_predictors:
        try:
            resolved_test_effects[predictor] = [
                _normalize_effect_name(name)
                for name in _resolve_test_effects(column_names, predictor)
            ]
        except ValueError:
            resolved_test_effects[predictor] = []

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "formula": formula.replace("y ~", "power ~", 1),
        "band_name": band_name,
        "test_predictors": test_predictors,
        "contrast_of_interest": _resolve_contrast_of_interest(runtime_config, model_name=model_name),
        "resolved_test_effects": resolved_test_effects,
        "term_tests": resolved_term_tests,
        "n_trials_used": int(trial_data.eeg_data.shape[0]),
        "n_channels": len(trial_data.channel_names),
        "n_times": int(trial_data.times.shape[0]),
        "betas_shape": list(betas.shape),
        "per_effect_outputs": per_effect_outputs,
        "summary_output": str(output_dir / "summary.json"),
    }
    (output_dir / "model_formula.txt").write_text(
        summary["formula"] + "\n",
        encoding="utf-8",
    )
    (output_dir / "term_tests.json").write_text(
        json.dumps(resolved_term_tests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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


def _resolve_term_test_effects(
    fixed_column_names: list[str],
    term_tests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for group in term_tests:
        resolved_terms: list[dict[str, Any]] = []
        for term in group.get("terms") or []:
            try:
                resolved_effects = _resolve_test_effects(fixed_column_names, str(term))
            except ValueError:
                resolved_effects = []
            resolved_terms.append(
                {
                    "term": str(term),
                    "effects": [_normalize_effect_name(value) for value in resolved_effects],
                    "raw_effects": resolved_effects,
                }
            )
        resolved.append(
            {
                "group_name": str(group.get("group_name", "")),
                "contrast_type": str(group.get("contrast_type", "")),
                "terms": resolved_terms,
            }
        )
    return resolved


def _resolve_configured_test_predictors(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> list[str]:
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    return [str(value) for value in model_cfg.get("test_predictors") or []]


def _resolve_contrast_of_interest(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
) -> str | None:
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    if model_cfg.get("contrast_of_interest") not in {None, ""}:
        return str(model_cfg.get("contrast_of_interest"))
    design_cfg = dict(lmeeeg_cfg.get("design") or {})
    if design_cfg.get("contrast_of_interest") not in {None, ""}:
        return str(design_cfg.get("contrast_of_interest"))
    test_predictors = _resolve_configured_test_predictors(runtime_config, model_name=model_name)
    if len(test_predictors) == 1:
        return test_predictors[0]
    return None


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
    formula_override: str | None = None,
    term_tests_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run permutation-based inference for configured term groups."""
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    test_cfg = dict(lmeeeg_cfg.get("test") or {})
    contrast_of_interest = _resolve_contrast_of_interest(runtime_config, model_name=model_name)
    term_tests = term_tests_override or _resolve_term_tests(
        runtime_config,
        model_name=model_name,
        duration_artifacts={},
        metadata=trial_data.trial_metadata,
    )

    if not term_tests:
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
        threshold = None

    adjacency = None
    if correction in {"cluster", "tfce"}:
        adjacency = _build_adjacency(
            trial_data.channel_names,
            int(trial_data.times.shape[0]),
            test_cfg,
        )

    # Reuse the main fit when available so inference does not refit the same model.
    if fit_result is None:
        fit_result = _refit_for_inference(
            runtime_config,
            trial_data,
            model_name=model_name,
            formula_override=formula_override,
        )

    fixed_column_names = list(fit_result.design_spec.fixed_column_names)
    if correction == "cluster":
        from scipy.stats import t as t_dist

        alpha = float(test_cfg.get("cluster_forming_alpha", 0.05))
        n_obs = int(trial_data.eeg_data.shape[0])
        n_predictors = max(len(fixed_column_names) - 1, 1)
        dof = max(n_obs - n_predictors - 1, 1)
        if tail == 0:
            threshold = float(t_dist.ppf(1 - alpha / 2, df=dof))
        else:
            threshold = float(t_dist.ppf(1 - alpha, df=dof))

    results: list[dict[str, Any]] = []
    for group in term_tests:
        group_name = str(group.get("group_name", ""))
        contrast_type = str(group.get("contrast_type", ""))
        for term in group.get("terms") or []:
            effect_names = _resolve_test_effects(fixed_column_names, str(term))
            for effect_name in effect_names:
                _emit_status(
                    "Running cluster testing for "
                    f"term_group={group_name} term={term} effect={effect_name}"
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
                            "normalized_effect": _normalize_effect_name(effect_name),
                            "requested_predictor": str(term),
                            "term": str(term),
                            "term_group": group_name,
                            "contrast_type": contrast_type,
                            "contrast_of_interest": contrast_of_interest,
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
                summary_metrics = _summarize_inference_result(
                    observed_statistic=np.asarray(inference_result.observed_statistic, dtype=float),
                    corrected_p_values=corrected_p,
                    channel_names=list(trial_data.channel_names),
                    times=np.asarray(trial_data.times, dtype=float),
                )
                results.append(
                    {
                        "effect": effect_name,
                        "normalized_effect": safe_effect_name,
                        "requested_predictor": str(term),
                        "term": str(term),
                        "term_group": group_name,
                        "contrast_type": contrast_type,
                        "contrast_of_interest": contrast_of_interest,
                        "status": "ok",
                        "correction": used_correction,
                        "observed_statistic": str(observed_path),
                        "corrected_p_values": str(corrected_p_path),
                        "corrected_p_values_csv": str(corrected_p_csv_path),
                        "min_corrected_p": float(np.nanmin(corrected_p)) if corrected_p.size else 1.0,
                        "n_significant_p_lt_0_05": int(np.sum(corrected_p < 0.05)),
                        **summary_metrics,
                    }
                )
                _emit_status(f"Saved inference outputs to {corrected_p_path.parent}")

    _write_inference_summary_artifacts(
        output_dir,
        results=results,
        model_name=model_name,
        band_name=band_name,
    )
    return results


def _summarize_inference_result(
    *,
    observed_statistic: np.ndarray,
    corrected_p_values: np.ndarray,
    channel_names: list[str],
    times: np.ndarray,
) -> dict[str, Any]:
    from scipy import ndimage

    if not np.isfinite(observed_statistic).any():
        return {
            "n_clusters": 0,
            "cluster_p_min": 1.0,
            "peak_abs_t_or_stat": None,
            "peak_time": None,
            "peak_channel": None,
        }
    peak_index = np.unravel_index(int(np.nanargmax(np.abs(observed_statistic))), observed_statistic.shape)
    significant_mask = np.asarray(corrected_p_values < 0.05, dtype=bool)
    labeled_mask, n_clusters = ndimage.label(significant_mask)
    cluster_p_min = float(np.nanmin(corrected_p_values[significant_mask])) if significant_mask.any() else 1.0
    return {
        "n_clusters": int(n_clusters),
        "cluster_p_min": cluster_p_min,
        "peak_abs_t_or_stat": float(np.abs(observed_statistic[peak_index])),
        "peak_time": float(times[int(peak_index[1])]),
        "peak_channel": str(channel_names[int(peak_index[0])]),
    }


def _write_inference_summary_artifacts(
    output_dir: Path,
    *,
    results: list[dict[str, Any]],
    model_name: str,
    band_name: str | None,
) -> None:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        row = {
            "model_name": model_name,
            "band_name": band_name,
            "term": result.get("term"),
            "term_group": result.get("term_group"),
            "effect": result.get("effect"),
            "normalized_effect": result.get("normalized_effect"),
            "contrast_type": result.get("contrast_type"),
            "status": result.get("status"),
            "n_clusters": result.get("n_clusters"),
            "cluster_p_min": result.get("cluster_p_min"),
            "peak_abs_t_or_stat": result.get("peak_abs_t_or_stat"),
            "peak_time": result.get("peak_time"),
            "peak_channel": result.get("peak_channel"),
            "observed_statistic": result.get("observed_statistic"),
            "corrected_p_values": result.get("corrected_p_values"),
            "corrected_p_values_csv": result.get("corrected_p_values_csv"),
        }
        rows.append(row)
        group_name = str(result.get("term_group", ""))
        group_entry = grouped.setdefault(
            group_name,
            {
                "term_group": group_name,
                "contrast_type": result.get("contrast_type"),
                "effects": [],
                "n_clusters": 0,
                "cluster_p_min": 1.0,
                "peak_abs_t_or_stat": 0.0,
                "peak_time": None,
                "peak_channel": None,
                "output_paths": [],
            },
        )
        group_entry["effects"].append(result.get("effect"))
        group_entry["output_paths"].append(
            {
                "effect": result.get("effect"),
                "observed_statistic": result.get("observed_statistic"),
                "corrected_p_values": result.get("corrected_p_values"),
                "corrected_p_values_csv": result.get("corrected_p_values_csv"),
            }
        )
        if result.get("status") == "ok":
            group_entry["n_clusters"] += int(result.get("n_clusters", 0) or 0)
            group_entry["cluster_p_min"] = min(
                float(group_entry["cluster_p_min"]),
                float(result.get("cluster_p_min", 1.0) or 1.0),
            )
            if float(result.get("peak_abs_t_or_stat", 0.0) or 0.0) >= float(
                group_entry["peak_abs_t_or_stat"]
            ):
                group_entry["peak_abs_t_or_stat"] = float(result.get("peak_abs_t_or_stat", 0.0) or 0.0)
                group_entry["peak_time"] = result.get("peak_time")
                group_entry["peak_channel"] = result.get("peak_channel")

    pd.DataFrame(rows).to_csv(output_dir / "inference_summary.csv", index=False)
    (output_dir / "inference_summary.json").write_text(
        json.dumps(
            {
                "effects": rows,
                "groups": list(grouped.values()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _refit_for_inference(runtime_config, trial_data, *, model_name, formula_override: str | None = None):
    """Return the lmeeeg FitResult object needed by permute_fixed_effect."""
    from lmeeeg import fit_lmm_mass_univariate

    formula, _, group_column = _resolve_model_formula(runtime_config, model_name=model_name)
    if formula_override is not None:
        formula = formula_override

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
    """Build a spatial EEG adjacency matrix for cluster/TFCE tests."""
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
    return spatial_adjacency


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
    _emit_status("Running SPP amplitude lmeEEG duration-control analysis")
    _emit_status(f"Loaded lmeEEG config: {Path(config_path)}")
    _emit_status(f"Writing lmeEEG outputs under: {output_dir}")

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
        _emit_status(f"Starting lmeEEG branch: model={model_name} modality={modality}")

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
    lmeeeg_cfg = dict(runtime_config.get("lmeeeg") or {})
    model_cfg = dict((lmeeeg_cfg.get("models") or {}).get(model_name) or {})
    merged_selection = dict(lmeeeg_cfg.get("selection") or {})
    merged_selection.update(dict(model_cfg.get("selection") or {}))
    selection_config = {"selection": merged_selection}
    _emit_status(
        f"Preparing pooled lmeEEG model {model_name} "
        f"({'band=' + band_name if band_name is not None else 'evoked'})"
    )
    selection_audit: dict[str, Any] = {
        "n_rows_before_selection": 0,
        "n_rows_after_selection": 0,
        "class_3_counts_before_selection": {},
        "class_3_counts_after_selection": {},
    }

    for ep_path in epochs_paths:
        source_path, metadata_csv = _resolve_pooled_source_paths(
            runtime_config,
            model_name=model_name,
            epochs_path=ep_path,
            band_name=band_name,
        )

        epochs, metadata = load_epochs_with_metadata(source_path, metadata_csv=metadata_csv)
        metadata = _augment_lmeeeg_metadata(metadata)
        selection_audit["n_rows_before_selection"] += int(len(metadata))
        if "class_3" in metadata.columns:
            counts_before = metadata["class_3"].value_counts(dropna=False).to_dict()
            for key, value in counts_before.items():
                selection_audit["class_3_counts_before_selection"][str(key)] = (
                    selection_audit["class_3_counts_before_selection"].get(str(key), 0) + int(value)
                )

        # Ensure subject_id is present
        if "subject_id" not in metadata.columns:
            try:
                row = _row_from_epochs_path(ep_path)
                metadata["subject_id"] = row["subject_id"]
            except ValueError:
                pass

        eeg = np.asarray(epochs.get_data(copy=False), dtype=np.float32)

        # Apply event/metadata selection per source file so we do not pool
        # rows that will immediately be discarded downstream.
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

            wrapper = _IndexableArray(eeg)
            try:
                wrapper, metadata = select_epochs_from_config(wrapper, metadata, selection_config)
            except ValueError as error:
                if "zero rows" in str(error):
                    _emit_status(
                        f"Skipping {source_path} because selection produced zero rows for model {model_name}."
                    )
                    continue
                raise
            eeg = wrapper._arr

        selection_audit["n_rows_after_selection"] += int(len(metadata))
        if "class_3" in metadata.columns:
            counts_after = metadata["class_3"].value_counts(dropna=False).to_dict()
            for key, value in counts_after.items():
                selection_audit["class_3_counts_after_selection"][str(key)] = (
                    selection_audit["class_3_counts_after_selection"].get(str(key), 0) + int(value)
                )

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

    combined_eeg = np.concatenate(pooled_eeg, axis=0, dtype=np.float32)
    combined_metadata = pd.concat(pooled_metadata, axis=0, ignore_index=True)

    prepared_inputs = _prepare_model_inputs_detailed(
        runtime_config,
        model_name=model_name,
        eeg_data=combined_eeg,
        metadata=combined_metadata,
    )
    combined_eeg = prepared_inputs.eeg_data
    combined_metadata = prepared_inputs.metadata
    class_column = str(prepared_inputs.duration_artifacts.get("class_column") or "")
    _emit_status(f"Rows retained for {model_name}: {len(combined_metadata)}")
    if class_column and class_column in combined_metadata.columns:
        trial_counts = combined_metadata[class_column].astype(str).value_counts(dropna=False).sort_index()
        _emit_status(f"Trial counts by {class_column}: {trial_counts.to_dict()}")
        duration_summary = _build_duration_summary_table(
            combined_metadata,
            duration_column=str(prepared_inputs.duration_artifacts.get("duration_column") or _DEFAULT_DURATION_COLUMN),
            class_column=class_column,
        )
        _emit_status(
            "Duration summary by class: "
            + duration_summary.to_dict(orient="records").__repr__()
        )
    _emit_status(
        "Enabled duration controls: "
        f"{prepared_inputs.duration_artifacts.get('enabled_controls') or ['none']}"
    )
    _emit_status(
        "Generated duration columns: "
        f"{prepared_inputs.duration_artifacts.get('generated_columns') or ['none']}"
    )
    _emit_status(f"Final model formula: {prepared_inputs.formula.replace('y ~', 'power ~', 1)}")
    _emit_status(f"Configured term tests: {prepared_inputs.term_tests}")
    _write_model_design_artifacts(
        runtime_config,
        model_name=model_name,
        band_name=band_name,
        design_table=combined_metadata,
        formula=prepared_inputs.formula.replace("y ~", "power ~", 1),
        duration_artifacts=prepared_inputs.duration_artifacts,
        term_tests=prepared_inputs.term_tests,
        selection_audit={
            **selection_audit,
            "dropped_other_rows": int(
                selection_audit["class_3_counts_before_selection"].get("OTHER", 0)
                - selection_audit["class_3_counts_after_selection"].get("OTHER", 0)
            ),
        },
    )

    trial_data = build_lmeeeg_trial_data_from_arrays(
        eeg_data=combined_eeg,
        metadata_df=combined_metadata,
        channel_names=ref_ch_names,
        times=ref_times,
    )

    fit_summary = _fit_one_model(
        runtime_config,
        trial_data,
        model_name=model_name,
        band_name=band_name,
        formula_override=prepared_inputs.formula,
        group_column_override=prepared_inputs.group_column,
        term_tests_override=prepared_inputs.term_tests,
    )

    inference = _run_model_inference(
        runtime_config,
        trial_data,
        model_name=model_name,
        band_name=band_name,
        fit_result=fit_summary.get("_fit_result"),
        formula_override=prepared_inputs.formula,
        term_tests_override=prepared_inputs.term_tests,
    )

    # Strip internal objects before serialising
    fit_summary.pop("_fit_result", None)

    entry: dict[str, Any] = {
        "model_name": model_name,
        "test_predictors": _resolve_configured_test_predictors(runtime_config, model_name=model_name),
        "contrast_of_interest": _resolve_contrast_of_interest(runtime_config, model_name=model_name),
        "term_tests": prepared_inputs.term_tests,
        "fit": fit_summary,
        "inference": inference,
    }
    if band_name is not None:
        entry["band_name"] = band_name
    output_location = fit_summary.get(
        "output_dir",
        str(_model_output_dir(runtime_config, model_name=model_name, band_name=band_name)),
    )
    _emit_status(
        f"Completed model {model_name}"
        + (f" ({band_name})" if band_name is not None else "")
        + f"; outputs written to {output_location}"
    )
    return entry


def _resolve_model_formula(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    duration_artifacts: dict[str, Any] | None = None,
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
    if duration_artifacts is not None:
        formula_rhs = _merge_formula_terms(
            formula_rhs,
            [str(value) for value in duration_artifacts.get("formula_terms") or []],
        )
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
    if "run" in prepared.columns:
        run_numeric = pd.to_numeric(prepared["run"], errors="coerce")
        if run_numeric.isna().any():
            invalid_reasons.append("`run` must be numeric and non-missing.")
        prepared["run"] = run_numeric

    categorical_levels_cfg = dict(design_cfg.get("categorical_levels") or {})
    if "pair_position" in prepared.columns and "pair_position" not in categorical_levels_cfg:
        categorical_levels_cfg["pair_position"] = {
            "allowed": ["FPP", "SPP"],
            "required": ["FPP", "SPP"],
        }

    for column, level_cfg in categorical_levels_cfg.items():
        if column not in prepared.columns:
            continue
        if isinstance(level_cfg, dict):
            allowed_levels = [str(value) for value in level_cfg.get("allowed") or []]
            required_levels = [str(value) for value in level_cfg.get("required") or allowed_levels]
        else:
            allowed_levels = [str(value) for value in level_cfg]
            required_levels = list(allowed_levels)

        values = prepared[column].astype(str)
        if allowed_levels:
            allowed_set = set(allowed_levels)
            invalid_levels = sorted(
                {value for value in values.dropna().unique().tolist() if value not in allowed_set}
            )
            if invalid_levels:
                invalid_reasons.append(
                    f"`{column}` contains invalid levels: {', '.join(invalid_levels)}"
                )

        if required_levels:
            observed_levels = set(values.dropna().unique().tolist())
            missing_levels = [value for value in required_levels if value not in observed_levels]
            if missing_levels:
                invalid_reasons.append(
                    f"`{column}` is missing required levels after filtering: {', '.join(missing_levels)}"
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
        if column == "run":
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


def _build_duration_summary_table(
    design_table: pd.DataFrame,
    *,
    duration_column: str,
    class_column: str,
) -> pd.DataFrame:
    if duration_column not in design_table.columns or class_column not in design_table.columns:
        return pd.DataFrame(
            columns=[class_column, "n", "mean", "median", "sd", "q05", "q25", "q75", "q95"]
        )

    rows: list[dict[str, Any]] = []
    for class_value, frame in design_table.groupby(class_column, observed=False):
        values = pd.to_numeric(frame[duration_column], errors="coerce").dropna()
        rows.append(
            {
                class_column: str(class_value),
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else None,
                "median": float(values.median()) if not values.empty else None,
                "sd": float(values.std(ddof=0)) if not values.empty else None,
                "q05": float(values.quantile(0.05)) if not values.empty else None,
                "q25": float(values.quantile(0.25)) if not values.empty else None,
                "q75": float(values.quantile(0.75)) if not values.empty else None,
                "q95": float(values.quantile(0.95)) if not values.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _write_duration_qc_plots(
    *,
    output_dir: Path,
    design_table: pd.DataFrame,
    duration_column: str | None,
    class_column: str | None,
) -> None:
    if not duration_column or not class_column:
        return
    if duration_column not in design_table.columns or class_column not in design_table.columns:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        _emit_status(f"Skipping duration QC plots because matplotlib is unavailable: {error}")
        return

    try:
        duration = pd.to_numeric(design_table[duration_column], errors="coerce")
        classes = design_table[class_column].astype(str)

        fig, ax = plt.subplots(figsize=(6, 4))
        for class_value, frame in design_table.assign(_duration=duration, _class=classes).groupby("_class", observed=False):
            ax.hist(frame["_duration"].dropna(), bins=20, alpha=0.45, label=str(class_value))
        ax.set_xlabel(duration_column)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "duration_histogram_by_class.png", dpi=150)
        plt.close(fig)

        log_column_candidates = [column for column in design_table.columns if "log" in column and "duration" in column]
        if log_column_candidates:
            log_column = log_column_candidates[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            for class_value, frame in design_table.assign(_class=classes).groupby("_class", observed=False):
                ax.hist(pd.to_numeric(frame[log_column], errors="coerce").dropna(), bins=20, alpha=0.45, label=str(class_value))
            ax.set_xlabel(log_column)
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "log_duration_histogram_by_class.png", dpi=150)
            plt.close(fig)

        bin_columns = [column for column in design_table.columns if "bin" in column and "duration" in column]
        if bin_columns:
            bin_counts = (
                design_table.groupby([class_column, bin_columns[0]], observed=False).size().reset_index(name="n_trials")
            )
            pivot = bin_counts.pivot(index=bin_columns[0], columns=class_column, values="n_trials").fillna(0.0)
            fig, ax = plt.subplots(figsize=(6, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(output_dir / "duration_bin_counts_by_class.png", dpi=150)
            plt.close(fig)
    except Exception as error:
        _emit_status(f"Skipping duration QC plots after plotting error: {error}")


def _summarize_design_table(
    design_table: pd.DataFrame,
    *,
    duration_column: str | None = None,
    class_column: str | None = None,
    duration_artifacts: dict[str, Any] | None = None,
    selection_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if "class_3" in design_table.columns:
        summary["class_3_counts"] = {
            str(key): int(value)
            for key, value in design_table["class_3"].value_counts(dropna=False).to_dict().items()
        }

    if duration_column and class_column:
        summary["duration_summary_by_class"] = _build_duration_summary_table(
            design_table,
            duration_column=duration_column,
            class_column=class_column,
        ).to_dict(orient="records")
        summary["trial_counts_by_class"] = (
            design_table[class_column].astype(str).value_counts(dropna=False).sort_index().to_dict()
            if class_column in design_table.columns
            else {}
        )

    if "duration" in design_table.columns and "class_3" in design_table.columns:
        duration_stats: dict[str, dict[str, float | int | None]] = {}
        for class_name, frame in design_table.groupby("class_3", observed=False):
            values = pd.to_numeric(frame["duration"], errors="coerce").dropna()
            duration_stats[str(class_name)] = {
                "count": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else None,
                "median": float(values.median()) if not values.empty else None,
                "std": float(values.std(ddof=0)) if not values.empty else None,
            }
        summary["duration_by_class_3"] = duration_stats

    if "log_duration_within_class" in design_table.columns:
        centered = pd.to_numeric(design_table["log_duration_within_class"], errors="coerce")
        summary["log_duration_within_class"] = {
            "mean": float(centered.mean()) if centered.notna().any() else None,
            "std": float(centered.std(ddof=0)) if centered.notna().any() else None,
            "min": float(centered.min()) if centered.notna().any() else None,
            "max": float(centered.max()) if centered.notna().any() else None,
        }
        if "class_3" in design_table.columns:
            summary["log_duration_within_class_mean_by_class_3"] = {
                str(key): float(value)
                for key, value in (
                    design_table.groupby("class_3", observed=False)["log_duration_within_class"]
                    .mean()
                    .fillna(0.0)
                    .to_dict()
                    .items()
                )
            }

    if selection_audit:
        summary["selection_audit"] = selection_audit
    if duration_artifacts:
        summary["duration_artifacts"] = duration_artifacts

    return summary


def _write_model_design_artifacts(
    runtime_config: dict[str, Any],
    *,
    model_name: str,
    band_name: str | None,
    design_table: pd.DataFrame,
    formula: str | None = None,
    duration_artifacts: dict[str, Any] | None = None,
    term_tests: list[dict[str, Any]] | None = None,
    selection_audit: dict[str, Any] | None = None,
) -> None:
    """Persist the auditable design table and run metadata for one fitted model."""
    output_dir = _model_output_dir(runtime_config, model_name=model_name, band_name=band_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_table.to_csv(output_dir / "design_table.csv", index=False)

    if formula is None:
        _, formula_rhs, group_column = _resolve_model_formula(runtime_config, model_name=model_name)
        formula_text = f"power ~ {formula_rhs} + (1 | {group_column})"
    else:
        formula_text = formula
        _, formula_rhs, group_column = _resolve_model_formula(runtime_config, model_name=model_name)
    class_column = str((duration_artifacts or {}).get("class_column") or "")
    duration_column = str((duration_artifacts or {}).get("duration_column") or "")
    metadata_payload = {
        "analysis_name": str((runtime_config.get("lmeeeg") or {}).get("analysis_name", "")).strip()
        or None,
        "model_name": model_name,
        "band_name": band_name,
        "config_path": str(((runtime_config.get("runtime") or {}).get("config_path")) or ""),
        "formula": formula_text,
        "test_predictors": _resolve_configured_test_predictors(runtime_config, model_name=model_name),
        "contrast_of_interest": _resolve_contrast_of_interest(runtime_config, model_name=model_name),
        "term_tests": term_tests or [],
        "duration_column": duration_column or None,
        "class_column": class_column or None,
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
    if formula_text:
        (output_dir / "model_formula.txt").write_text(formula_text + "\n", encoding="utf-8")
    if term_tests is not None:
        (output_dir / "term_tests.json").write_text(
            json.dumps(term_tests, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if class_column and class_column in design_table.columns:
        design_table[class_column].astype(str).value_counts(dropna=False).sort_index().rename_axis(class_column).reset_index(
            name="n_trials"
        ).to_csv(output_dir / "trial_counts_by_class.csv", index=False)
    else:
        pd.DataFrame(columns=["class", "n_trials"]).to_csv(output_dir / "trial_counts_by_class.csv", index=False)
    if duration_column and class_column:
        _build_duration_summary_table(
            design_table,
            duration_column=duration_column,
            class_column=class_column,
        ).to_csv(output_dir / "duration_summary_by_class.csv", index=False)
    else:
        pd.DataFrame(columns=["class", "n", "mean", "median", "sd", "q05", "q25", "q75", "q95"]).to_csv(
            output_dir / "duration_summary_by_class.csv",
            index=False,
        )
    _write_duration_qc_plots(
        output_dir=output_dir,
        design_table=design_table,
        duration_column=duration_column or None,
        class_column=class_column or None,
    )
    (output_dir / "design_summary.json").write_text(
        json.dumps(
            _summarize_design_table(
                design_table,
                duration_column=duration_column or None,
                class_column=class_column or None,
                duration_artifacts=duration_artifacts,
                selection_audit=selection_audit,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
