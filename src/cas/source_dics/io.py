"""IO helpers for the source-level DICS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

_EPOCH_FILE_RE = re.compile(
    r"sub-(?P<subject>[^_/]+)_task-(?P<task>[^_/]+)_run-(?P<run>[^_/]+).*_epo\.fif$"
)


def configure_mne_runtime() -> None:
    """Configure MNE runtime safeguards for this repo's Python stack.

    Usage example
    -------------
    >>> configure_mne_runtime()
    """

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


@dataclass(frozen=True, slots=True)
class EpochRecord:
    subject_id: str
    run_id: str
    task: str
    epochs_path: Path | None = None
    preprocessed_eeg_path: Path | None = None
    raw_eeg_path: Path | None = None


def discover_epoch_records(epochs_dir: Path, *, subjects: set[str] | None = None) -> list[EpochRecord]:
    pattern = str(epochs_dir / "**" / "*_epo.fif")
    records: list[EpochRecord] = []
    for path in sorted(Path(epochs_dir).glob("**/*_epo.fif")):
        match = _EPOCH_FILE_RE.search(path.name)
        if match is None:
            continue
        subject_id = f"sub-{match.group('subject')}"
        if subjects is not None and subject_id not in subjects:
            continue
        records.append(
            EpochRecord(
                subject_id=subject_id,
                run_id=match.group("run"),
                task=match.group("task"),
                epochs_path=path.resolve(),
            )
        )
    return records


_PREPROCESSED_FILE_RE = re.compile(
    r".*/sub-(?P<subject>[^/]+)/task-(?P<task>[^/]+)/run-(?P<run>[^/]+)/preprocessed_eeg\.fif$"
)


def discover_preprocessed_records(
    preprocessed_eeg_root: Path,
    *,
    bids_root: Path,
    subjects: set[str] | None = None,
) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    for path in sorted(preprocessed_eeg_root.glob("sub-*/task-*/run-*/preprocessed_eeg.fif")):
        match = _PREPROCESSED_FILE_RE.match(str(path))
        if match is None:
            continue
        subject_id = f"sub-{match.group('subject')}"
        if subjects is not None and subject_id not in subjects:
            continue
        task = match.group("task")
        run_id = match.group("run")
        raw_eeg_path = bids_root / subject_id / "eeg" / f"{subject_id}_task-{task}_run-{run_id}_eeg.edf"
        if not raw_eeg_path.exists():
            raw_eeg_path = bids_root / subject_id / "eeg" / f"{subject_id}_task-{task}_run-{run_id}_eeg.fif"
        if not raw_eeg_path.exists():
            continue
        records.append(
            EpochRecord(
                subject_id=subject_id,
                run_id=run_id,
                task=task,
                preprocessed_eeg_path=path.resolve(),
                raw_eeg_path=raw_eeg_path.resolve(),
            )
        )
    return records


def load_epochs(epochs_path: Path):
    configure_mne_runtime()
    import mne

    return mne.read_epochs(epochs_path, preload=True, verbose="ERROR")


def load_events_table(events_path: Path) -> pd.DataFrame:
    return pd.read_csv(events_path)


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_table(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        table.to_csv(output_path, index=False)
    elif output_path.suffixes[-2:] == [".csv", ".gz"]:
        table.to_csv(output_path, index=False, compression="gzip")
    elif output_path.suffix == ".parquet":
        table.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported table output path: {output_path}")
    return output_path


def build_record_stem(record: EpochRecord, *, anchor_type: str, band_name: str) -> str:
    return f"{record.subject_id}_run-{record.run_id}_anchor-{anchor_type}_band-{band_name}"
