#!/usr/bin/env python3
"""Extract FPP-SPP pairs from Praat TextGrid files into a single CSV.

This script scans a directory tree of files such as::

    /Users/hiro/Datasets/anais/dyad-003_run-1_combined.TextGrid

and outputs one CSV row per matched FPP-SPP pair.

Matching rule
-------------
For each FPP event, find unused SPP events whose onset falls in the window

    [FPP offset - 1.0 s, FPP offset + 1.0 s]

and select the SPP whose onset is closest to the FPP offset. Once an SPP is
matched, it cannot be used again.

Important note
--------------
You originally wrote the window as ``(FPP offset - 1, FPP onset + 1)``.
That would often produce an odd or even inverted window for events longer than
1 second. Because the next rule says to choose the onset closest to the FPP
offset, this script uses the more coherent interpretation centered on the FPP
offset: ``[offset - 1, offset + 1]``.

Token counting
--------------
Token counts are computed from the corresponding ``palign-A`` and ``palign-B``
word-alignment tiers by counting aligned intervals that overlap the event.
The script ignores empty labels and the common non-lexical markers ``#`` and
``@``.

Usage example
-------------
    python textgrid_fpp_spp_pairs.py \
        --input-dir /Users/hiro/Datasets/anais \
        --output-csv /Users/hiro/Datasets/anais/fpp_spp_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Parsing constants
ENCODING_CANDIDATES: Tuple[str, ...] = ("utf-16", "utf-8", "utf-8-sig", "latin-1")
TIER_SPLIT_PATTERN = re.compile(r"^\s*item\s*\[\d+\]:", re.MULTILINE)
NAME_PATTERN = re.compile(r'^\s*name\s*=\s*"([^"]*)"', re.MULTILINE)
INTERVAL_PATTERN = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*([\d.eE+-]+)\s*"
    r"xmax\s*=\s*([\d.eE+-]+)\s*"
    r'text\s*=\s*"([^"]*)"',
    re.DOTALL,
)
FILENAME_PATTERN = re.compile(r"dyad-(?P<dyad>\d+)_run-(?P<run>\d+)_combined\.TextGrid$")
SPEAKER_TIER_PATTERN = re.compile(r"(?P<prefix>actions?|palign|ipu)[-\s]+(?P<speaker>[ab])", re.IGNORECASE)
LABEL_SEPARATOR_PATTERN = re.compile(r"[_\s]+")
LABEL_COMPONENT_TYPO_MAP = {
    "?": "",
    "C": "C",
    "ONF": "CONF",
    "C ONF": "CONF",
    "CON": "CONF",
    "CONC": "CONF",
    "CONF": "CONF",
    "DIS": "DISC",
    "DISC": "DISC",
    "DSIC": "DISC",
    "RFC": "RFC",
}


# Matching constants
WINDOW_LEFT_S = 1.0
WINDOW_RIGHT_S = 1.0
TOKEN_IGNORE_LABELS = {"", "#", "@"}


@dataclass(frozen=True)
class Interval:
    """Simple interval container.

    Parameters
    ----------
    onset_s
        Interval onset in seconds.
    offset_s
        Interval offset in seconds.
    text
        Praat interval label.
    """

    onset_s: float
    offset_s: float
    text: str

    @property
    def duration_s(self) -> float:
        """Return interval duration in seconds."""
        return self.offset_s - self.onset_s


@dataclass(frozen=True)
class Event:
    """Conversation event extracted from an ``actions`` tier.

    Parameters
    ----------
    event_type
        High-level prefix such as ``FPP`` or ``SPP``.
    class_1
        First lower-level class, e.g. ``RFC`` from ``FPP_RFC_DECL``.
    class_2
        Second lower-level class, e.g. ``DECL`` from ``FPP_RFC_DECL``.
    """

    speaker_id: str
    onset_s: float
    offset_s: float
    label_full: str
    event_type: str
    class_1: str
    class_2: str

    @property
    def duration_s(self) -> float:
        """Return event duration in seconds."""
        return self.offset_s - self.onset_s


def read_textgrid_text(path: Path) -> str:
    """Read a TextGrid file with a small encoding fallback cascade.

    Parameters
    ----------
    path
        Path to the TextGrid file.

    Returns
    -------
    str
        Decoded file contents.
    """
    raw_bytes = path.read_bytes()
    for encoding in ENCODING_CANDIDATES:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw_bytes, 0, 1, f"Could not decode {path}")


def parse_interval_tiers(text: str) -> Dict[str, List[Interval]]:
    """Parse IntervalTier contents from a Praat ooTextFile TextGrid.

    Parameters
    ----------
    text
        Full decoded TextGrid text.

    Returns
    -------
    dict of str to list of Interval
        Mapping from tier name to all parsed intervals.
    """
    tiers: Dict[str, List[Interval]] = {}
    chunks = TIER_SPLIT_PATTERN.split(text)

    for chunk in chunks:
        name_match = NAME_PATTERN.search(chunk)
        if name_match is None:
            continue

        tier_name = canonicalize_tier_name(name_match.group(1))
        intervals: List[Interval] = []
        for onset_text, offset_text, label_text in INTERVAL_PATTERN.findall(chunk):
            intervals.append(
                Interval(
                    onset_s=float(onset_text),
                    offset_s=float(offset_text),
                    text=label_text.strip(),
                )
            )

        if intervals:
            tiers.setdefault(tier_name, []).extend(intervals)

    return tiers


def canonicalize_tier_name(tier_name: str) -> str:
    """Normalize safe tier-name variants to one canonical spelling."""
    compact_name = " ".join(tier_name.strip().split())
    if not compact_name:
        return tier_name

    speaker_match = SPEAKER_TIER_PATTERN.fullmatch(compact_name)
    if speaker_match is not None:
        prefix = speaker_match.group("prefix").lower()
        speaker_id = speaker_match.group("speaker").upper()
        if prefix in {"action", "actions"}:
            return f"actions {speaker_id}"
        return f"{prefix}-{speaker_id}"

    if compact_name.lower() == "comment":
        return "comment"

    return compact_name


def parse_label_components(label: str) -> Tuple[str, str, str]:
    """Split labels like ``FPP_RFC_DECL`` into components.

    Parameters
    ----------
    label
        Full annotation label.

    Returns
    -------
    tuple of str
        ``(event_type, class_1, class_2)``.
    """
    normalized_label = normalize_action_label(label)
    parts = normalized_label.split("_")
    event_type = parts[0] if len(parts) >= 1 else ""
    class_1 = parts[1] if len(parts) >= 2 else ""
    class_2 = "_".join(parts[2:]) if len(parts) >= 3 else ""
    return event_type, class_1, class_2


def normalize_action_label(label: str) -> str:
    """Normalize known label typos without changing intended classes."""
    normalized_label = label.strip().replace("è", "_")
    if not normalized_label:
        return normalized_label

    parts = normalized_label.split("_")
    normalized_parts: List[str] = []
    for part in parts:
        cleaned_part = " ".join(part.split()).upper()
        canonical_part = LABEL_COMPONENT_TYPO_MAP.get(cleaned_part, cleaned_part)
        if canonical_part:
            normalized_parts.extend(
                token for token in LABEL_SEPARATOR_PATTERN.split(canonical_part) if token
            )
    return "_".join(normalized_parts)


def extract_action_events(tiers: Dict[str, List[Interval]]) -> List[Event]:
    """Extract only FPP and SPP events from ``actions A/B`` tiers.

    Parameters
    ----------
    tiers
        Parsed TextGrid tier dictionary.

    Returns
    -------
    list of Event
        All FPP/SPP events from both speakers.
    """
    events: List[Event] = []

    for speaker_id in ("A", "B"):
        tier_name = f"actions {speaker_id}"
        for interval in tiers.get(tier_name, []):
            label = normalize_action_label(interval.text)
            if not label:
                continue

            event_type, class_1, class_2 = parse_label_components(label)
            if event_type not in {"FPP", "SPP"}:
                continue

            events.append(
                Event(
                    speaker_id=speaker_id,
                    onset_s=interval.onset_s,
                    offset_s=interval.offset_s,
                    label_full=label,
                    event_type=event_type,
                    class_1=class_1,
                    class_2=class_2,
                )
            )

    events.sort(key=lambda event: (event.onset_s, event.offset_s, event.speaker_id))
    return events


def interval_overlap_s(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return overlap duration between two closed-open intervals."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def count_overlapping_tokens(word_intervals: Sequence[Interval], onset_s: float, offset_s: float) -> int:
    """Count lexical tokens overlapping an event interval.

    Parameters
    ----------
    word_intervals
        Intervals from ``palign-A`` or ``palign-B``.
    onset_s
        Event onset.
    offset_s
        Event offset.

    Returns
    -------
    int
        Number of overlapping lexical tokens.
    """
    n_tokens = 0
    for interval in word_intervals:
        token = interval.text.strip()
        if token in TOKEN_IGNORE_LABELS:
            continue
        if interval_overlap_s(onset_s, offset_s, interval.onset_s, interval.offset_s) > 0.0:
            n_tokens += 1
    return n_tokens


def find_best_spp_match(fpp_event: Event, candidate_spp_events: Sequence[Tuple[int, Event]]) -> Optional[int]:
    """Return the index of the best available SPP match for one FPP.

    Parameters
    ----------
    fpp_event
        Current FPP event.
    candidate_spp_events
        Sequence of ``(index, spp_event)`` pairs for currently unused SPPs.

    Returns
    -------
    int | None
        Index into the original SPP list if a candidate exists, else ``None``.
    """
    window_start_s = fpp_event.offset_s - WINDOW_LEFT_S
    window_end_s = fpp_event.offset_s + WINDOW_RIGHT_S

    best_key: Optional[Tuple[float, float, int]] = None
    best_index: Optional[int] = None

    for spp_index, spp_event in candidate_spp_events:
        onset_s = spp_event.onset_s
        if onset_s < window_start_s or onset_s > window_end_s:
            continue

        # Primary criterion: onset closest to the FPP offset.
        # Tie-breaker 1: earlier onset.
        # Tie-breaker 2: earlier original index for deterministic behavior.
        score_key = (abs(onset_s - fpp_event.offset_s), onset_s, spp_index)
        if best_key is None or score_key < best_key:
            best_key = score_key
            best_index = spp_index

    return best_index


def parse_dyad_and_run(path: Path) -> Tuple[str, str]:
    """Extract dyad and run IDs from the filename.

    Parameters
    ----------
    path
        TextGrid path.

    Returns
    -------
    tuple of str
        ``(dyad_id, run_id)``.
    """
    match = FILENAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            "Filename does not match expected pattern "
            f"'dyad-XXX_run-Y_combined.TextGrid': {path.name}"
        )
    return match.group("dyad"), match.group("run")


def infer_subject_ids(dyad_id: str) -> Dict[str, str]:
    """Infer A/B subject IDs from the dyad ID.

    Dyad numbering is 1-based, with speaker A assigned the odd participant
    index and speaker B assigned the even participant index.
    """
    dyad_number = int(dyad_id)
    subject_a_number = dyad_number * 2 - 1
    subject_b_number = dyad_number * 2
    width = max(3, len(dyad_id))
    return {
        "A": f"subject-{subject_a_number:0{width}d}",
        "B": f"subject-{subject_b_number:0{width}d}",
    }


def pair_events_for_file(path: Path) -> List[Dict[str, object]]:
    """Create one output row per matched FPP-SPP pair for a single file.

    Parameters
    ----------
    path
        Input TextGrid path.

    Returns
    -------
    list of dict
        CSV-ready rows for this file.
    """
    dyad_id, run_id = parse_dyad_and_run(path)
    subject_ids = infer_subject_ids(dyad_id)
    tiers = parse_interval_tiers(read_textgrid_text(path))
    events = extract_action_events(tiers)

    palign_tiers: Dict[str, List[Interval]] = {
        "A": tiers.get("palign-A", []),
        "B": tiers.get("palign-B", []),
    }

    fpp_events = [event for event in events if event.event_type == "FPP"]
    spp_events = [event for event in events if event.event_type == "SPP"]

    unused_spp_indices = set(range(len(spp_events)))
    rows: List[Dict[str, object]] = []

    for fpp_event in fpp_events:
        candidates = [(index, spp_events[index]) for index in sorted(unused_spp_indices)]
        matched_index = find_best_spp_match(fpp_event, candidates)
        if matched_index is None:
            continue

        spp_event = spp_events[matched_index]
        unused_spp_indices.remove(matched_index)

        fpp_tokens = count_overlapping_tokens(
            word_intervals=palign_tiers.get(fpp_event.speaker_id, []),
            onset_s=fpp_event.onset_s,
            offset_s=fpp_event.offset_s,
        )
        spp_tokens = count_overlapping_tokens(
            word_intervals=palign_tiers.get(spp_event.speaker_id, []),
            onset_s=spp_event.onset_s,
            offset_s=spp_event.offset_s,
        )

        rows.append(
            {
                "dyad_id": dyad_id,
                "run": run_id,
                "fpp_speaker_id": subject_ids[fpp_event.speaker_id],
                "spp_speaker_id": subject_ids[spp_event.speaker_id],
                "fpp_onset_s": fpp_event.onset_s,
                "fpp_offset_s": fpp_event.offset_s,
                "spp_onset_s": spp_event.onset_s,
                "spp_offset_s": spp_event.offset_s,
                "fpp_duration_s": fpp_event.duration_s,
                "spp_duration_s": spp_event.duration_s,
                "response_latency_s": spp_event.onset_s - fpp_event.offset_s,
                "fpp_n_tokens": fpp_tokens,
                "spp_n_tokens": spp_tokens,
                "fpp_class_1": fpp_event.class_1,
                "fpp_class_2": fpp_event.class_2,
                "spp_class_1": spp_event.class_1,
                "spp_class_2": spp_event.class_2,
            }
        )

    return rows


def discover_textgrid_paths(input_dir: Path) -> List[Path]:
    """Find matching TextGrid files in a directory tree.

    Parameters
    ----------
    input_dir
        Root directory containing TextGrid files.

    Returns
    -------
    list of Path
        Sorted matching file paths.
    """
    paths = sorted(input_dir.rglob("dyad-*_run-*_combined.TextGrid"))
    return [path for path in paths if path.is_file()]


def write_rows_to_csv(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    """Write extracted rows to CSV.

    Parameters
    ----------
    rows
        CSV-ready rows.
    output_csv
        Destination CSV path.
    """
    fieldnames = [
        "dyad_id",
        "run",
        "fpp_speaker_id",
        "spp_speaker_id",
        "fpp_onset_s",
        "fpp_offset_s",
        "spp_onset_s",
        "spp_offset_s",
        "fpp_duration_s",
        "spp_duration_s",
        "response_latency_s",
        "fpp_n_tokens",
        "spp_n_tokens",
        "fpp_class_1",
        "fpp_class_2",
        "spp_class_1",
        "spp_class_2",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory tree containing dyad-*_run-*_combined.TextGrid files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to the output CSV file.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    textgrid_paths = discover_textgrid_paths(arguments.input_dir)
    if not textgrid_paths:
        raise FileNotFoundError(
            f"No files matching 'dyad-*_run-*_combined.TextGrid' found in {arguments.input_dir}"
        )

    all_rows: List[Dict[str, object]] = []
    for textgrid_path in textgrid_paths:
        all_rows.extend(pair_events_for_file(textgrid_path))

    write_rows_to_csv(all_rows, arguments.output_csv)
    print(f"Wrote {len(all_rows)} matched FPP-SPP pairs to {arguments.output_csv}")


if __name__ == "__main__":
    main()
