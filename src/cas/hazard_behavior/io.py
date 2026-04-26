"""IO helpers for behavioural hazard analysis."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.schema import normalize_events_schema, normalize_surprisal_schema

JSON_INDENT = 2


def resolve_surprisal_paths(path_or_glob: str | Path) -> tuple[Path, ...]:
    """Resolve one or more surprisal TSV paths from a file, directory, or glob."""

    path_text = str(path_or_glob)
    candidate_path = Path(path_text)
    if candidate_path.exists():
        if candidate_path.is_dir():
            return tuple(sorted(candidate_path.rglob("*.tsv")))
        return (candidate_path,)
    return tuple(sorted(Path(path) for path in glob.glob(path_text, recursive=True)))


def read_events_table(events_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and normalize the events table."""

    table = pd.read_csv(events_path)
    if table.empty:
        raise ValueError(f"Events table is empty: {events_path}")
    normalized = normalize_events_schema(table)
    return normalized.table, normalized.warnings


def read_surprisal_tables(
    surprisal_paths: tuple[Path, ...],
    *,
    unmatched_surprisal_strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load, normalize, and combine surprisal TSV files."""

    if not surprisal_paths:
        raise FileNotFoundError("No surprisal TSV files were found.")

    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    for path in surprisal_paths:
        table = pd.read_csv(path, sep="\t")
        normalized = normalize_surprisal_schema(table)
        warnings.extend(normalized.warnings)
        frames.append(normalized.table.assign(source_path=str(path)))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["onset"] = pd.to_numeric(combined["onset"], errors="coerce")
    combined["duration"] = pd.to_numeric(combined["duration"], errors="coerce")
    combined["offset"] = combined["onset"] + combined["duration"]
    if "surprisal" in combined.columns:
        combined["surprisal"] = pd.to_numeric(combined["surprisal"], errors="coerce")
    if "word_index" in combined.columns:
        combined["word_index"] = pd.to_numeric(combined["word_index"], errors="coerce")

    combined = combined.loc[
        combined["onset"].notna()
        & combined["duration"].notna()
        & (combined["duration"] >= 0.0)
        & combined["speaker"].astype(str).isin({"A", "B"})
    ].copy()

    if "alignment_status" in combined.columns:
        combined["alignment_status"] = combined["alignment_status"].astype(str)
        if unmatched_surprisal_strategy == "drop":
            combined = combined.loc[combined["alignment_status"] == "ok"].copy()
        elif unmatched_surprisal_strategy == "zero":
            unmatched_mask = combined["alignment_status"] != "ok"
            combined.loc[unmatched_mask & combined["surprisal"].isna(), "surprisal"] = 0.0
        elif unmatched_surprisal_strategy == "keep_nan":
            pass

    if unmatched_surprisal_strategy != "keep_nan":
        combined = combined.loc[combined["surprisal"].notna() & combined["surprisal"].map(pd.notna)].copy()
    combined = combined.sort_values(
        [column for column in ["dyad_id", "run", "speaker", "onset", "offset", "word_index"] if column in combined],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined, warnings


def ensure_output_directories(config: BehaviourHazardConfig) -> dict[str, Path]:
    """Create output directories and return their paths."""

    root = config.out_dir.resolve()
    if root.exists() and any(root.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {root}")
    directories = {
        "root": root,
        "riskset": root / "riskset",
        "models": root / "models",
        "models_exports": root / "models" / "exports",
        "models_lag_selection": root / "models" / "lag_selection",
        "models_predictions": root / "models" / "predictions",
        "figures": root / "figures",
        "qc_plots": root / "qc_plots",
        "qc_plots_lag_selection": root / "qc_plots" / "lag_selection",
        "logs": root / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_table(table: pd.DataFrame, path: Path, *, sep: str = "\t") -> Path:
    """Write a table with stable defaults."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep=sep, index=False)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_config_and_warnings(
    *,
    config: BehaviourHazardConfig,
    warnings: list[str],
    logs_dir: Path,
) -> tuple[Path, Path]:
    """Persist configuration and warnings metadata."""

    config_path = logs_dir / "hazard_behavior_config.json"
    warnings_path = logs_dir / "hazard_behavior_warnings.txt"
    config_path.write_text(config.to_json() + "\n", encoding="utf-8")
    warnings_path.write_text("\n".join(warnings).rstrip() + ("\n" if warnings else ""), encoding="utf-8")
    return config_path, warnings_path
