from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class InducedEpochBandPaths:
    epochs_output_path: Path
    metadata_output_path: Path
    events_array_output_path: Path
    summary_output_path: Path


def build_induced_epoch_band_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    band_name: str,
    root_subdir: str = "induced_epochs",
) -> InducedEpochBandPaths:
    out_dir = Path(str((config.get("paths") or {}).get("out_dir", "")))
    if not str(out_dir):
        raise ValueError("config.paths.out_dir is required.")
    subject_id = str(row["subject_id"])
    band_dir = out_dir / root_subdir / str(band_name) / subject_id
    return InducedEpochBandPaths(
        epochs_output_path=band_dir / "epochs-time_s.fif",
        metadata_output_path=band_dir / "metadata-time_s.csv",
        events_array_output_path=band_dir / "events-time_s.npy",
        summary_output_path=band_dir / "epoching_summary-time_s.json",
    )
