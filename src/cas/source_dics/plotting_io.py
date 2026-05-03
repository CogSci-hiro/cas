"""IO and schema-adaptation helpers for source-level lmeEEG plotting.

Usage example
-------------
>>> from pathlib import Path
>>> import pandas as pd
>>> _ = pd.DataFrame(
...     {
...         "band": ["alpha"],
...         "predictor": ["duration"],
...         "source_id": ["lh:12"],
...         "time": [-0.5],
...         "t_value": [2.1],
...         "p_value": [0.03],
...     }
... ).to_csv("/tmp/source_stats.csv", index=False)
>>> adapted = load_source_statistics_table(Path("/tmp"))
>>> sorted(adapted.columns)[:4]
['band', 'cluster_id', 'cluster_p_value', 'hemi']
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Final

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

STANDARD_COLUMNS: Final[tuple[str, ...]] = (
    "band",
    "predictor",
    "hemi",
    "source_id",
    "time",
    "t_value",
    "p_value",
    "p_value_corrected",
    "significant",
    "cluster_id",
    "cluster_p_value",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("cluster_significant", "roi", "label")

SOURCE_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("source_id", "vertex", "source", "source", "node")
TIME_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("time", "time_s", "latency", "latency_s")
T_VALUE_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("t_value", "t", "t_stat", "tvalue", "stat")
P_VALUE_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("p_value", "p", "p_uncorrected", "uncorrected_p")
P_VALUE_CORRECTED_CANDIDATES: Final[tuple[str, ...]] = (
    "p_value_corrected",
    "p_corrected",
    "corrected_p",
    "p_fdr",
    "p_adj",
)


@dataclass(frozen=True, slots=True)
class SourceStatisticsLoadResult:
    """Normalized source-statistics payload.

    Attributes
    ----------
    table
        Standardized source-level statistics table.
    files_loaded
        Number of files read and merged.
    rows_loaded
        Number of normalized rows retained.
    """

    table: pd.DataFrame
    files_loaded: int
    rows_loaded: int


def _find_first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        matched = lookup.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _parse_hemi_and_source_id(raw_value: object) -> tuple[str | None, str]:
    token = str(raw_value)
    stripped = token.strip()
    if not stripped:
        return None, ""

    lowered = stripped.lower()
    if lowered.startswith("ctx-lh-"):
        return "lh", stripped[7:]
    if lowered.startswith("ctx-rh-"):
        return "rh", stripped[7:]
    if lowered.startswith("lh:"):
        return "lh", stripped[3:]
    if lowered.startswith("rh:"):
        return "rh", stripped[3:]
    if lowered.startswith("lh."):
        return "lh", stripped[3:]
    if lowered.startswith("rh."):
        return "rh", stripped[3:]
    if lowered.endswith("-lh"):
        return "lh", stripped[:-3]
    if lowered.endswith("-rh"):
        return "rh", stripped[:-3]

    if re.fullmatch(r"-?\d+", stripped):
        return None, stripped
    return None, stripped


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "t", "yes", "y"})


def _adapt_source_statistics_frame(frame: pd.DataFrame, *, file_path: Path) -> pd.DataFrame:
    source_column = _find_first_column(list(frame.columns), SOURCE_COLUMN_CANDIDATES)
    time_column = _find_first_column(list(frame.columns), TIME_COLUMN_CANDIDATES)
    t_value_column = _find_first_column(list(frame.columns), T_VALUE_COLUMN_CANDIDATES)

    missing = []
    if "band" not in frame.columns:
        missing.append("band")
    if "predictor" not in frame.columns:
        missing.append("predictor")
    if source_column is None:
        missing.append("source_id/vertex")
    if time_column is None:
        missing.append("time")
    if t_value_column is None:
        missing.append("t_value")
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {', '.join(missing)}")

    out = pd.DataFrame()
    out["band"] = frame["band"].astype(str)
    out["predictor"] = frame["predictor"].astype(str)

    parsed = frame[source_column].map(_parse_hemi_and_source_id)
    parsed_hemi = parsed.map(lambda item: item[0])
    parsed_source = parsed.map(lambda item: item[1])

    if "hemi" in frame.columns:
        hemi_series = frame["hemi"].where(frame["hemi"].notna(), parsed_hemi)
    elif "hemisphere" in frame.columns:
        hemi_series = frame["hemisphere"].where(frame["hemisphere"].notna(), parsed_hemi)
    else:
        hemi_series = parsed_hemi

    out["hemi"] = hemi_series.fillna("unknown").astype(str)
    out["source_id"] = parsed_source.astype(str)
    out["time"] = pd.to_numeric(frame[time_column], errors="coerce")
    out["t_value"] = pd.to_numeric(frame[t_value_column], errors="coerce")

    p_value_column = _find_first_column(list(frame.columns), P_VALUE_COLUMN_CANDIDATES)
    p_value_corrected_column = _find_first_column(list(frame.columns), P_VALUE_CORRECTED_CANDIDATES)

    out["p_value"] = (
        pd.to_numeric(frame[p_value_column], errors="coerce")
        if p_value_column is not None
        else np.nan
    )
    out["p_value_corrected"] = (
        pd.to_numeric(frame[p_value_corrected_column], errors="coerce")
        if p_value_corrected_column is not None
        else np.nan
    )

    if "significant" in frame.columns:
        out["significant"] = _as_bool_series(frame["significant"])
    else:
        out["significant"] = False

    out["cluster_id"] = frame["cluster_id"] if "cluster_id" in frame.columns else np.nan
    out["cluster_p_value"] = (
        pd.to_numeric(frame["cluster_p_value"], errors="coerce")
        if "cluster_p_value" in frame.columns
        else np.nan
    )
    out["cluster_significant"] = (
        _as_bool_series(frame["cluster_significant"])
        if "cluster_significant" in frame.columns
        else False
    )

    out["roi"] = frame["roi"] if "roi" in frame.columns else np.nan
    if "label" in frame.columns:
        out["label"] = frame["label"]
    elif "roi" in frame.columns:
        out["label"] = frame["roi"]
    else:
        out["label"] = np.nan

    out = out.replace({"": np.nan})
    out = out.dropna(subset=["band", "predictor", "source_id", "time", "t_value"]).reset_index(drop=True)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_source_statistics_table(
    statistics_dir: Path,
    *,
    predictors: set[str] | None = None,
    bands: set[str] | None = None,
) -> SourceStatisticsLoadResult:
    """Load and normalize source-level lmeEEG statistics tables.

    Parameters
    ----------
    statistics_dir
        Directory containing tidy or semi-tidy source-level statistics files.
    predictors
        Optional predictor filter.
    bands
        Optional frequency-band filter.

    Returns
    -------
    SourceStatisticsLoadResult
        Standardized statistics table and load counts.

    Usage example
    -------------
    >>> from pathlib import Path
    >>> result = load_source_statistics_table(Path("/tmp"))  # doctest: +SKIP
    >>> set(result.table.columns).issuperset({"band", "predictor", "t_value"})  # doctest: +SKIP
    True
    """

    if not statistics_dir.exists():
        raise FileNotFoundError(f"Statistics directory does not exist: {statistics_dir}")

    candidate_files: list[Path] = []
    candidate_files.extend(sorted(statistics_dir.rglob("*.parquet")))
    candidate_files.extend(sorted(statistics_dir.rglob("*.csv")))
    candidate_files.extend(sorted(statistics_dir.rglob("*.csv.gz")))

    if not candidate_files:
        raise FileNotFoundError(f"No statistics files found under {statistics_dir}")

    normalized_frames: list[pd.DataFrame] = []
    for path in candidate_files:
        try:
            frame = _read_table(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping unreadable statistics file %s: %s", path, exc)
            continue

        if frame.empty:
            continue

        try:
            adapted = _adapt_source_statistics_frame(frame, file_path=path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping file with incompatible schema %s: %s", path, exc)
            continue

        if predictors is not None:
            adapted = adapted.loc[adapted["predictor"].isin(predictors)]
        if bands is not None:
            adapted = adapted.loc[adapted["band"].isin(bands)]
        if adapted.empty:
            continue

        normalized_frames.append(adapted)

    if not normalized_frames:
        raise ValueError(
            "No compatible source-level statistics rows were loaded after schema adaptation and filtering."
        )

    table = pd.concat(normalized_frames, ignore_index=True)
    table = table.sort_values(["band", "predictor", "source_id", "time"], kind="mergesort").reset_index(drop=True)
    return SourceStatisticsLoadResult(
        table=table,
        files_loaded=len(normalized_frames),
        rows_loaded=int(len(table)),
    )
