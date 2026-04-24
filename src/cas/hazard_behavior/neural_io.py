"""IO helpers for low-level neural hazard features."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

NEURAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dyad_id": ("dyad_id", "dyad", "recording_id"),
    "run": ("run",),
    "speaker": ("speaker", "participant_speaker", "subject", "subject_id"),
    "time": ("time", "timestamp", "sample_time", "onset"),
}
FEATURE_FAMILY_PREFIXES = {
    "amplitude": "amp_",
    "alpha": "alpha_",
    "beta": "beta_",
}


def resolve_neural_feature_paths(path_or_glob_values: tuple[str | Path, ...]) -> tuple[Path, ...]:
    """Resolve one or more neural feature paths from file, directory, or glob inputs."""

    resolved: list[Path] = []
    for value in path_or_glob_values:
        path_text = str(value)
        candidate_path = Path(path_text)
        if candidate_path.exists():
            if candidate_path.is_dir():
                resolved.extend(sorted(candidate_path.rglob("*.tsv")))
                resolved.extend(sorted(candidate_path.rglob("*.csv")))
            else:
                resolved.append(candidate_path)
            continue
        resolved.extend(sorted(Path(path) for path in glob.glob(path_text, recursive=True)))
    return tuple(dict.fromkeys(path.resolve() for path in resolved))


def read_neural_feature_tables(
    paths: tuple[Path, ...],
    *,
    time_column: str,
    speaker_column: str,
) -> pd.DataFrame:
    """Read and combine tabular neural feature inputs."""

    if not paths:
        raise FileNotFoundError("No neural feature files were provided.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            table = pd.read_csv(path, sep="\t")
        elif suffix == ".csv":
            table = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported neural feature format for {path}. Expected CSV or TSV for the first-pass neural pipeline."
            )
        normalized = normalise_neural_schema(table, time_column=time_column, speaker_column=speaker_column)
        frames.append(normalized.assign(source_path=str(path)))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["time"] = pd.to_numeric(combined["time"], errors="coerce")
    combined["dyad_id"] = combined["dyad_id"].astype(str)
    combined["run"] = combined["run"].astype(str)
    combined["speaker"] = combined["speaker"].astype(str)
    combined = combined.loc[np.isfinite(combined["time"])].copy()  # type: ignore[name-defined]
    combined = combined.sort_values(["dyad_id", "run", "speaker", "time"], kind="mergesort").reset_index(drop=True)
    return combined


def normalise_neural_schema(
    table: pd.DataFrame,
    *,
    time_column: str,
    speaker_column: str,
) -> pd.DataFrame:
    """Normalize a neural feature table to the hazard schema."""

    rename_map: dict[str, str] = {}
    required_columns = {
        "dyad_id": _find_neural_column(table, "dyad_id"),
        "run": _find_neural_column(table, "run"),
        "speaker": _find_neural_column(table, "speaker", explicit_name=speaker_column),
        "time": _find_neural_column(table, "time", explicit_name=time_column),
    }
    missing = [canonical for canonical, matched in required_columns.items() if matched is None]
    if missing:
        available = ", ".join(str(column) for column in table.columns)
        raise ValueError(
            "Neural feature table is missing required columns "
            f"{missing}. Available columns: {available}"
        )
    for canonical, matched in required_columns.items():
        if matched != canonical:
            rename_map[str(matched)] = canonical
    return table.rename(columns=rename_map).copy()


def select_neural_feature_columns(
    table: pd.DataFrame,
    *,
    feature_prefixes: tuple[str, ...],
    include_amplitude: bool,
    include_alpha: bool,
    include_beta: bool,
) -> list[str]:
    """Select supported low-level neural feature columns by prefix."""

    allowed_prefixes: list[str] = []
    prefix_set = tuple(feature_prefixes)
    if include_amplitude:
        allowed_prefixes.extend(prefix for prefix in prefix_set if prefix.startswith(FEATURE_FAMILY_PREFIXES["amplitude"]))
    if include_alpha:
        allowed_prefixes.extend(prefix for prefix in prefix_set if prefix.startswith(FEATURE_FAMILY_PREFIXES["alpha"]))
    if include_beta:
        allowed_prefixes.extend(prefix for prefix in prefix_set if prefix.startswith(FEATURE_FAMILY_PREFIXES["beta"]))
    if not allowed_prefixes:
        raise ValueError("No neural feature prefixes are enabled. Enable at least one of amplitude, alpha, or beta.")
    feature_columns = [
        str(column)
        for column in table.columns
        if str(column) not in {"dyad_id", "run", "speaker", "time", "source_path"}
        and any(str(column).startswith(prefix) for prefix in allowed_prefixes)
    ]
    if not feature_columns:
        raise ValueError(
            "No neural feature columns matched the requested prefixes: "
            + ", ".join(sorted(allowed_prefixes))
        )
    return sorted(feature_columns)


def _find_neural_column(table: pd.DataFrame, canonical_name: str, *, explicit_name: str | None = None) -> str | None:
    if explicit_name and explicit_name in table.columns:
        return explicit_name
    for candidate in NEURAL_COLUMN_ALIASES.get(canonical_name, ()):
        if candidate in table.columns:
            return candidate
    return None
