"""CSV IO helpers for canonical event extraction outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from cas.events.models import FppSppEvent, PairingIssue

EVENTS_CSV_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "run",
    "file_path",
    "part",
    "response",
    "speaker_fpp",
    "speaker_spp",
    "fpp_label",
    "spp_label",
    "fpp_onset",
    "fpp_offset",
    "spp_onset",
    "spp_offset",
    "fpp_duration",
    "spp_duration",
    "latency",
    "pair_id",
)

PAIRING_ISSUES_CSV_COLUMNS: tuple[str, ...] = (
    "file_path",
    "recording_id",
    "run",
    "fpp_tier",
    "fpp_index",
    "fpp_label",
    "fpp_onset",
    "fpp_offset",
    "issue_code",
    "message",
    "n_candidates",
)


def write_events_csv(events: Iterable[FppSppEvent], output_path: Path) -> None:
    """Write canonical paired events to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVENTS_CSV_COLUMNS))
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_csv_row())


def write_pairing_issues_csv(issues: Iterable[PairingIssue], output_path: Path) -> None:
    """Write extraction issues to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRING_ISSUES_CSV_COLUMNS))
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.to_csv_row())
