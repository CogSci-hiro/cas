from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_epochs(epochs, output_path: Path) -> Path:
    """Write epochs to FIF."""
    _configure_mne_runtime()
    _ensure_parent_dir(output_path)
    epochs.save(output_path, overwrite=True, verbose="ERROR")
    return output_path


def write_epoch_metadata(metadata_df: pd.DataFrame, output_path: Path) -> Path:
    """Write epoch metadata as CSV."""
    _ensure_parent_dir(output_path)
    metadata_df.to_csv(output_path, index=False)
    return output_path


def write_epoch_events_array(events_array: np.ndarray, output_path: Path) -> Path:
    """Write the integer events array used for epoching."""
    _ensure_parent_dir(output_path)
    np.save(output_path, np.asarray(events_array, dtype=int))
    return output_path


def write_epoch_summary(summary: Mapping[str, Any], output_path: Path) -> Path:
    """Write a JSON epoching summary."""
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path

