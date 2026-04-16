"""Event extraction helpers used during preprocessing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import mne
import numpy as np
import pandas as pd

from cas.events import ExtractionConfig, extract_events_from_textgrid

LOGGER = logging.getLogger(__name__)

EVENTS_TSV_COLUMNS: tuple[str, ...] = (
    "event_index",
    "source",
    "sample",
    "onset_s",
    "duration_s",
    "event_id",
    "label",
    "pair_id",
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
    "latency",
)


@dataclass(frozen=True, slots=True)
class ExtractedEvents:
    """Tabular events plus lightweight provenance metadata."""

    rows: list[dict[str, str | int | float]]
    source: str


def extract_events(
    raw: mne.io.BaseRaw,
    *,
    annotation_path: str | Path | None = None,
    annotation_pairing_margin_s: float = 1.0,
) -> ExtractedEvents:
    """Extract events from annotations first, falling back to MNE events."""

    annotation_rows = _extract_annotation_events(
        raw,
        annotation_path=annotation_path,
        annotation_pairing_margin_s=annotation_pairing_margin_s,
    )
    if annotation_rows:
        return ExtractedEvents(rows=annotation_rows, source="annotations")

    fallback_rows = _extract_mne_events(raw)
    return ExtractedEvents(rows=fallback_rows, source="mne")


def write_events_tsv(rows: list[dict[str, str | int | float]], output_path: str | Path) -> Path:
    """Write preprocessing events as a TSV file."""

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVENTS_TSV_COLUMNS), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column_name: row.get(column_name, "") for column_name in EVENTS_TSV_COLUMNS})
    return resolved_output_path


def _extract_annotation_events(
    raw: mne.io.BaseRaw,
    *,
    annotation_path: str | Path | None,
    annotation_pairing_margin_s: float,
) -> list[dict[str, str | int | float]]:
    if annotation_path is None:
        return []

    resolved_annotation_path = Path(annotation_path)
    if not resolved_annotation_path.exists():
        LOGGER.info("Annotation file not found at %s; falling back to MNE events.", resolved_annotation_path)
        return []

    try:
        result = extract_events_from_textgrid(
            resolved_annotation_path,
            ExtractionConfig(pairing_margin_s=annotation_pairing_margin_s),
        )
    except Exception as exc:
        LOGGER.warning(
            "Annotation event extraction failed for %s; falling back to MNE events: %s",
            resolved_annotation_path,
            exc,
        )
        return []

    if not result.events:
        LOGGER.info(
            "No annotation-derived events were found in %s; falling back to MNE events.",
            resolved_annotation_path,
        )
        return []

    sampling_rate_hz = float(raw.info["sfreq"])
    rows: list[dict[str, str | int | float]] = []
    for event_index, event in enumerate(result.events, start=1):
        onset_s = float(event.fpp_offset)
        rows.append(
            {
                "event_index": event_index,
                "source": "annotations",
                "sample": int(round(onset_s * sampling_rate_hz)),
                "onset_s": onset_s,
                "duration_s": 0.0,
                "event_id": event_index,
                "label": f"{event.part}:{event.response}",
                "pair_id": event.pair_id,
                "part": event.part,
                "response": event.response,
                "speaker_fpp": event.speaker_fpp,
                "speaker_spp": event.speaker_spp,
                "fpp_label": event.fpp_label,
                "spp_label": event.spp_label,
                "fpp_onset": event.fpp_onset,
                "fpp_offset": event.fpp_offset,
                "spp_onset": event.spp_onset,
                "spp_offset": event.spp_offset,
                "latency": event.latency,
            }
        )
    return rows


def _extract_mne_events(raw: mne.io.BaseRaw) -> list[dict[str, str | int | float]]:
    try:
        found_events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    except Exception as exc:
        LOGGER.warning("MNE event detection failed; writing an empty events table: %s", exc)
        return []

    sampling_rate_hz = float(raw.info["sfreq"])
    rows: list[dict[str, str | int | float]] = []
    for event_index, event_triplet in enumerate(found_events, start=1):
        sample = int(event_triplet[0])
        event_id = int(event_triplet[2])
        rows.append(
            {
                "event_index": event_index,
                "source": "mne",
                "sample": sample,
                "onset_s": sample / sampling_rate_hz,
                "duration_s": 0.0,
                "event_id": event_id,
                "label": f"event_{event_id}",
                "pair_id": "",
                "part": "",
                "response": "",
                "speaker_fpp": "",
                "speaker_spp": "",
                "fpp_label": "",
                "spp_label": "",
                "fpp_onset": "",
                "fpp_offset": "",
                "spp_onset": "",
                "spp_offset": "",
                "latency": "",
            }
        )
    return rows


def extract_events_table(
    raw: mne.io.BaseRaw,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract a tabular event representation using MNE annotations or triggers."""

    events_cfg = dict((config.get("preprocessing") or {}).get("events") or {})
    stim_channel = events_cfg.get("stim_channel")
    shortest_event = int(events_cfg.get("shortest_event", 1))
    consecutive = bool(events_cfg.get("consecutive", True))

    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except ValueError:
        events = np.empty((0, 3), dtype=int)
        event_id = {}

    if events.size == 0:
        try:
            kwargs: dict[str, Any] = {
                "shortest_event": shortest_event,
                "consecutive": consecutive,
                "verbose": "ERROR",
            }
            if stim_channel:
                kwargs["stim_channel"] = stim_channel
            events = mne.find_events(raw, **kwargs)
        except ValueError:
            events = np.empty((0, 3), dtype=int)
        event_id = {}

    if events.size == 0:
        return pd.DataFrame(columns=["sample", "time_s", "previous", "event_id"]), event_id

    sampling_rate_hz = float(raw.info["sfreq"])
    table = pd.DataFrame(
        {
            "sample": events[:, 0].astype(int),
            "time_s": events[:, 0].astype(float) / sampling_rate_hz,
            "previous": events[:, 1].astype(int),
            "event_id": events[:, 2].astype(int),
        }
    )
    return table, event_id
