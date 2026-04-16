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
Token counts are computed from the corresponding word-alignment tiers by
counting aligned intervals that overlap the event. The script ignores empty
labels and common non-lexical/alignment markers such as ``#``, ``@``, ``sil``,
``sp``, ``spn``, and ``pau``.

Syllable counting
-----------------
Syllable counts are read from per-subject CSV files in a separate directory,
for example::

    /Users/hiro/Projects/active/diapix-annotations/EEG/annotations/syllable_v1/sub-001_run-1_syllable.csv

For each event, the script counts syllable intervals in the corresponding
subject/run CSV that overlap the event interval, then computes speech rate as::

    n_syllables / duration_s

Safety improvements
-------------------
Compared with the earlier version, this script now:

1. Refuses to silently merge duplicate tier names after canonicalization.
2. Prints tier diagnostics when requested.
3. Applies a heuristic check to detect phone-like tiers being used as word tiers.
4. Ignores common silence/alignment labels in token counting.

Usage example
-------------
    python textgrid_fpp_spp_pairs.py
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple


# User configuration
INPUT_DIR = Path("/Users/hiro/Datasets/anais")
OUTPUT_CSV = Path("/Users/hiro/Datasets/anais/fpp_spp_pairs.csv")
SYLLABLE_DIR = Path("/Users/hiro/Projects/active/diapix-annotations/EEG/annotations/syllable_v1")
DEBUG_TIER_SELECTION = False

# Syllable CSV schema
# Syllable CSV schema (headerless CSV)
# Example row:
#   SyllAlign,4.33,4.96272,A/-l-R
SYLLABLE_ONSET_COLUMN_INDEX = 1
SYLLABLE_OFFSET_COLUMN_INDEX = 2
SYLLABLE_TIER_COLUMN_INDEX = 0
SYLLABLE_LABEL_COLUMN_INDEX = 3
SYLLABLE_MIN_COLUMNS = 4
PREFERRED_SYLLABLE_TIER_NAMES: Tuple[str, ...] = ("SyllAlign", "SyllClassAlign")

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
SYLLABLE_FILENAME_PATTERN = re.compile(r"sub-(?P<subject>\d+)_run-(?P<run>\d+)_syllable\.csv$")
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

# Token-counting constants
TOKEN_IGNORE_LABELS = {
    "",
    "#",
    "@",
    "sil",
    "sp",
    "spn",
    "pau",
    "<sil>",
    "<sp>",
    "SIL",
    "SP",
    "SPN",
    "PAU",
}
PHONE_TIER_SHORT_LABEL_MAX_CHARS = 2
PHONE_TIER_SHORT_LABEL_FRACTION_THRESHOLD = 0.60
PHONE_TIER_MEAN_DURATION_THRESHOLD_S = 0.15


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
        """Return interval duration in seconds.

        Usage example
        -------------
            interval = Interval(1.0, 1.4, "hello")
            duration_s = interval.duration_s
        """
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

    Usage example
    -------------
        event = Event(
            speaker_id="A",
            onset_s=1.0,
            offset_s=1.8,
            label_full="FPP_RFC_DECL",
            event_type="FPP",
            class_1="RFC",
            class_2="DECL",
        )
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
        """Return event duration in seconds.

        Usage example
        -------------
            duration_s = event.duration_s
        """
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

    Usage example
    -------------
        text = read_textgrid_text(Path("example.TextGrid"))
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

    Notes
    -----
    This function now refuses to silently merge duplicate tier names after
    canonicalization. That makes duplicate or ambiguously named tiers visible
    immediately instead of inflating downstream counts.

    Usage example
    -------------
        tiers = parse_interval_tiers(text)
    """
    tiers: Dict[str, List[Interval]] = {}
    chunks = TIER_SPLIT_PATTERN.split(text)

    for chunk in chunks:
        name_match = NAME_PATTERN.search(chunk)
        if name_match is None:
            continue

        original_tier_name = name_match.group(1)
        tier_name = canonicalize_tier_name(original_tier_name)

        intervals: List[Interval] = []
        for onset_text, offset_text, label_text in INTERVAL_PATTERN.findall(chunk):
            intervals.append(
                Interval(
                    onset_s=float(onset_text),
                    offset_s=float(offset_text),
                    text=label_text.strip(),
                )
            )

        if not intervals:
            continue

        if tier_name in tiers:
            raise ValueError(
                "Duplicate tier name after canonicalization: "
                f"original={original_tier_name!r}, canonical={tier_name!r}"
            )

        tiers[tier_name] = intervals

    return tiers


def canonicalize_tier_name(tier_name: str) -> str:
    """Normalize safe tier-name variants to one canonical spelling.

    Usage example
    -------------
        canonical_name = canonicalize_tier_name("palign A")
    """
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

    Usage example
    -------------
        event_type, class_1, class_2 = parse_label_components("FPP_RFC_DECL")
    """
    normalized_label = normalize_action_label(label)
    parts = normalized_label.split("_")
    event_type = parts[0] if len(parts) >= 1 else ""
    class_1 = parts[1] if len(parts) >= 2 else ""
    class_2 = "_".join(parts[2:]) if len(parts) >= 3 else ""
    return event_type, class_1, class_2


def normalize_action_label(label: str) -> str:
    """Normalize known label typos without changing intended classes.

    Usage example
    -------------
        normalized_label = normalize_action_label("FPPèRFCèDECL")
    """
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

    Usage example
    -------------
        events = extract_action_events(tiers)
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
    """Return overlap duration between two closed-open intervals.

    Usage example
    -------------
        overlap_s = interval_overlap_s(0.0, 1.0, 0.5, 1.5)
    """
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_token_label(token_label: str) -> str:
    """Normalize token labels for lexical filtering.

    Parameters
    ----------
    token_label
        Raw interval text.

    Returns
    -------
    str
        Normalized label.

    Usage example
    -------------
        normalized_label = normalize_token_label(" sil ")
    """
    return token_label.strip()


def get_non_ignored_labels(intervals: Sequence[Interval]) -> List[str]:
    """Return normalized labels excluding known non-lexical markers.

    Parameters
    ----------
    intervals
        Alignment intervals.

    Returns
    -------
    list of str
        Non-ignored labels.

    Usage example
    -------------
        labels = get_non_ignored_labels(intervals)
    """
    kept_labels: List[str] = []
    for interval in intervals:
        normalized_label = normalize_token_label(interval.text)
        if normalized_label in TOKEN_IGNORE_LABELS:
            continue
        kept_labels.append(normalized_label)
    return kept_labels


def looks_like_phone_tier(intervals: Sequence[Interval]) -> bool:
    """Heuristically detect whether a tier looks phone-like rather than word-like.

    The heuristic flags a tier as phone-like when both conditions hold:

    1. A large fraction of non-ignored labels are very short.
    2. Mean interval duration is very short.

    Parameters
    ----------
    intervals
        Candidate alignment intervals.

    Returns
    -------
    bool
        True if the tier looks phone-like.

    Usage example
    -------------
        is_phone_like = looks_like_phone_tier(intervals)
    """
    if not intervals:
        return False

    non_ignored_labels = get_non_ignored_labels(intervals)
    if not non_ignored_labels:
        return False

    short_label_fraction = sum(
        len(label) <= PHONE_TIER_SHORT_LABEL_MAX_CHARS for label in non_ignored_labels
    ) / len(non_ignored_labels)

    mean_duration_s = mean(interval.duration_s for interval in intervals)

    return (
        short_label_fraction >= PHONE_TIER_SHORT_LABEL_FRACTION_THRESHOLD
        and mean_duration_s <= PHONE_TIER_MEAN_DURATION_THRESHOLD_S
    )


def debug_print_tier_summary(path: Path, tiers: Dict[str, List[Interval]], palign_tiers: Dict[str, List[Interval]]) -> None:
    """Print a compact diagnostic summary for tier selection.

    Parameters
    ----------
    path
        Current TextGrid path.
    tiers
        All parsed tiers.
    palign_tiers
        Selected alignment tiers for speakers A and B.

    Usage example
    -------------
        debug_print_tier_summary(path, tiers, palign_tiers)
    """
    print(f"\n=== Tier diagnostics for {path.name} ===")
    print("All parsed tiers:")
    for tier_name in sorted(tiers):
        print(f"  {tier_name}: {len(tiers[tier_name])} intervals")

    for speaker_id in ("A", "B"):
        intervals = palign_tiers[speaker_id]
        non_ignored_labels = get_non_ignored_labels(intervals)
        unique_labels = sorted(set(non_ignored_labels))
        sample_labels = unique_labels[:30]
        mean_duration_s = mean(interval.duration_s for interval in intervals) if intervals else 0.0

        print(f"\nSelected palign-{speaker_id}:")
        print(f"  n_intervals = {len(intervals)}")
        print(f"  mean_interval_duration_s = {mean_duration_s:.4f}")
        print(f"  n_non_ignored_labels = {len(non_ignored_labels)}")
        print(f"  sample_labels = {sample_labels}")
        print(f"  phone_like = {looks_like_phone_tier(intervals)}")


def count_overlapping_tokens(word_intervals: Sequence[Interval], onset_s: float, offset_s: float) -> int:
    """Count lexical tokens overlapping an event interval.

    Parameters
    ----------
    word_intervals
        Intervals from a word-alignment tier.
    onset_s
        Event onset.
    offset_s
        Event offset.

    Returns
    -------
    int
        Number of overlapping lexical tokens.

    Notes
    -----
    This function counts overlapping *intervals* from the selected alignment
    tier after filtering known non-lexical markers. It does not attempt to
    merge split words or deduplicate intervals across tiers.

    Usage example
    -------------
        n_tokens = count_overlapping_tokens(intervals, 1.0, 2.0)
    """
    n_tokens = 0
    for interval in word_intervals:
        token = normalize_token_label(interval.text)
        if token in TOKEN_IGNORE_LABELS:
            continue
        if interval_overlap_s(onset_s, offset_s, interval.onset_s, interval.offset_s) > 0.0:
            n_tokens += 1
    return n_tokens


def count_overlapping_syllables(
    syllable_intervals: Sequence[Interval],
    onset_s: float,
    offset_s: float,
) -> int:
    """Count syllable intervals overlapping an event interval.

    Parameters
    ----------
    syllable_intervals
        Intervals from a per-subject syllable CSV.
    onset_s
        Event onset.
    offset_s
        Event offset.

    Returns
    -------
    int
        Number of overlapping syllable intervals.

    Usage example
    -------------
        n_syllables = count_overlapping_syllables(intervals, 1.0, 2.0)
    """
    n_syllables = 0
    for interval in syllable_intervals:
        if not interval.text.strip():
            continue
        if interval_overlap_s(onset_s, offset_s, interval.onset_s, interval.offset_s) > 0.0:
            n_syllables += 1
    return n_syllables


def compute_speech_rate_syllables_per_s(n_syllables: int, duration_s: float) -> float:
    """Compute speech rate in syllables per second.

    Usage example
    -------------
        speech_rate = compute_speech_rate_syllables_per_s(5, 1.25)
    """
    if duration_s <= 0.0:
        return 0.0
    return n_syllables / duration_s


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

    Usage example
    -------------
        matched_index = find_best_spp_match(fpp_event, candidates)
    """
    window_start_s = fpp_event.offset_s - WINDOW_LEFT_S
    window_end_s = fpp_event.offset_s + WINDOW_RIGHT_S

    best_key: Optional[Tuple[float, float, int]] = None
    best_index: Optional[int] = None

    for spp_index, spp_event in candidate_spp_events:
        onset_s = spp_event.onset_s
        if onset_s < window_start_s or onset_s > window_end_s:
            continue

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

    Usage example
    -------------
        dyad_id, run_id = parse_dyad_and_run(path)
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

    Usage example
    -------------
        subject_ids = infer_subject_ids("003")
    """
    dyad_number = int(dyad_id)
    subject_a_number = dyad_number * 2 - 1
    subject_b_number = dyad_number * 2
    width = max(3, len(dyad_id))
    return {
        "A": f"subject-{subject_a_number:0{width}d}",
        "B": f"subject-{subject_b_number:0{width}d}",
    }


def subject_id_to_syllable_stem(subject_id: str) -> str:
    """Convert ``subject-001`` to ``sub-001`` for syllable CSV lookup.

    Usage example
    -------------
        stem = subject_id_to_syllable_stem("subject-001")
    """
    return subject_id.replace("subject-", "sub-")


def select_alignment_tier(tiers: Dict[str, List[Interval]], speaker_id: str) -> List[Interval]:
    """Select the alignment tier for one speaker and validate it.

    Parameters
    ----------
    tiers
        Parsed tier mapping.
    speaker_id
        ``"A"`` or ``"B"``.

    Returns
    -------
    list of Interval
        Selected alignment intervals.

    Raises
    ------
    KeyError
        If the expected tier is missing.
    ValueError
        If the selected tier appears phone-like.

    Usage example
    -------------
        alignment_intervals = select_alignment_tier(tiers, "A")
    """
    tier_name = f"palign-{speaker_id}"
    if tier_name not in tiers:
        available_tier_names = ", ".join(sorted(tiers))
        raise KeyError(
            f"Required tier {tier_name!r} not found. Available tiers: {available_tier_names}"
        )

    selected_intervals = tiers[tier_name]

    if looks_like_phone_tier(selected_intervals):
        sample_labels = sorted(set(get_non_ignored_labels(selected_intervals)))[:30]
        raise ValueError(
            f"Tier {tier_name!r} appears phone-like rather than word-like. "
            f"Sample labels: {sample_labels}"
        )

    return selected_intervals


def parse_syllable_csv(path: Path) -> List[Interval]:
    """Parse one headerless syllable CSV into intervals.

    Parameters
    ----------
    path
        Path to a subject/run syllable CSV.

    Returns
    -------
    list of Interval
        Parsed syllable intervals.

    Usage example
    -------------
        intervals = parse_syllable_csv(path)
    """
    intervals_by_tier: Dict[str, List[Interval]] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as file_handle:
        reader = csv.reader(file_handle)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < SYLLABLE_MIN_COLUMNS:
                raise ValueError(
                    f"Syllable CSV {path} has too few columns on row {row_number}: {row}"
                )

            tier_name = row[SYLLABLE_TIER_COLUMN_INDEX].strip()
            onset_text = row[SYLLABLE_ONSET_COLUMN_INDEX].strip()
            offset_text = row[SYLLABLE_OFFSET_COLUMN_INDEX].strip()
            label_text = row[SYLLABLE_LABEL_COLUMN_INDEX].strip()

            if not onset_text or not offset_text:
                continue

            try:
                interval = Interval(
                    onset_s=float(onset_text),
                    offset_s=float(offset_text),
                    text=label_text,
                )
                intervals_by_tier.setdefault(tier_name, []).append(interval)
            except ValueError as error:
                raise ValueError(
                    f"Could not parse syllable timing in {path} on row {row_number}: {row}"
                ) from error

    for tier_name in PREFERRED_SYLLABLE_TIER_NAMES:
        if tier_name in intervals_by_tier:
            return intervals_by_tier[tier_name]

    available_tier_names = ", ".join(sorted(intervals_by_tier)) or "<none>"
    raise ValueError(
        f"Could not find a supported syllable tier in {path}. "
        f"Available tier names: {available_tier_names}"
    )


def discover_syllable_csv_paths(syllable_dir: Path) -> List[Path]:
    """Find syllable CSV files in a directory tree.

    Parameters
    ----------
    syllable_dir
        Root directory containing ``sub-*_run-*_syllable.csv`` files.

    Returns
    -------
    list of Path
        Sorted matching CSV paths.

    Usage example
    -------------
        paths = discover_syllable_csv_paths(SYLLABLE_DIR)
    """
    paths = sorted(syllable_dir.rglob("sub-*_run-*_syllable.csv"))
    return [path for path in paths if path.is_file()]


def build_syllable_interval_index(syllable_dir: Path) -> Dict[Tuple[str, str], List[Interval]]:
    """Build an index from ``(subject_id, run_id)`` to syllable intervals.

    Parameters
    ----------
    syllable_dir
        Root directory containing syllable CSV files.

    Returns
    -------
    dict
        Mapping from ``(sub-XXX, run)`` to syllable intervals.

    Usage example
    -------------
        syllable_index = build_syllable_interval_index(SYLLABLE_DIR)
    """
    syllable_index: Dict[Tuple[str, str], List[Interval]] = {}

    for path in discover_syllable_csv_paths(syllable_dir):
        match = SYLLABLE_FILENAME_PATTERN.search(path.name)
        if match is None:
            continue

        subject_id = f"sub-{match.group('subject')}"
        run_id = match.group("run")
        key = (subject_id, run_id)

        if key in syllable_index:
            raise ValueError(f"Duplicate syllable CSV for {key}: {path}")

        syllable_index[key] = parse_syllable_csv(path)

    return syllable_index


def pair_events_for_file(
    path: Path,
    syllable_index: Dict[Tuple[str, str], List[Interval]],
    debug_tier_selection: bool = False,
) -> List[Dict[str, object]]:
    """Create one output row per matched FPP-SPP pair for a single file.

    Parameters
    ----------
    path
        Input TextGrid path.
    syllable_index
        Mapping from ``(sub-XXX, run)`` to syllable intervals.
    debug_tier_selection
        Whether to print tier diagnostics.

    Returns
    -------
    list of dict
        CSV-ready rows for this file.

    Usage example
    -------------
        rows = pair_events_for_file(path, syllable_index, debug_tier_selection=True)
    """
    dyad_id, run_id = parse_dyad_and_run(path)
    subject_ids = infer_subject_ids(dyad_id)
    tiers = parse_interval_tiers(read_textgrid_text(path))
    events = extract_action_events(tiers)

    palign_tiers: Dict[str, List[Interval]] = {
        "A": select_alignment_tier(tiers, "A"),
        "B": select_alignment_tier(tiers, "B"),
    }

    if debug_tier_selection:
        debug_print_tier_summary(path, tiers, palign_tiers)

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

        fpp_subject_id = subject_ids[fpp_event.speaker_id]
        spp_subject_id = subject_ids[spp_event.speaker_id]

        fpp_syllable_key = (subject_id_to_syllable_stem(fpp_subject_id), run_id)
        spp_syllable_key = (subject_id_to_syllable_stem(spp_subject_id), run_id)

        if fpp_syllable_key not in syllable_index:
            raise KeyError(
                f"Missing syllable CSV for FPP subject/run {fpp_syllable_key}. "
                f"Expected file like {fpp_syllable_key[0]}_run-{run_id}_syllable.csv in {SYLLABLE_DIR}"
            )
        if spp_syllable_key not in syllable_index:
            raise KeyError(
                f"Missing syllable CSV for SPP subject/run {spp_syllable_key}. "
                f"Expected file like {spp_syllable_key[0]}_run-{run_id}_syllable.csv in {SYLLABLE_DIR}"
            )

        fpp_syllables = count_overlapping_syllables(
            syllable_intervals=syllable_index[fpp_syllable_key],
            onset_s=fpp_event.onset_s,
            offset_s=fpp_event.offset_s,
        )
        spp_syllables = count_overlapping_syllables(
            syllable_intervals=syllable_index[spp_syllable_key],
            onset_s=spp_event.onset_s,
            offset_s=spp_event.offset_s,
        )

        fpp_speech_rate = compute_speech_rate_syllables_per_s(
            n_syllables=fpp_syllables,
            duration_s=fpp_event.duration_s,
        )
        spp_speech_rate = compute_speech_rate_syllables_per_s(
            n_syllables=spp_syllables,
            duration_s=spp_event.duration_s,
        )

        rows.append(
            {
                "dyad_id": dyad_id,
                "run": run_id,
                "fpp_speaker_id": fpp_subject_id,
                "spp_speaker_id": spp_subject_id,
                "fpp_onset_s": fpp_event.onset_s,
                "fpp_offset_s": fpp_event.offset_s,
                "spp_onset_s": spp_event.onset_s,
                "spp_offset_s": spp_event.offset_s,
                "fpp_duration_s": fpp_event.duration_s,
                "spp_duration_s": spp_event.duration_s,
                "response_latency_s": spp_event.onset_s - fpp_event.offset_s,
                "fpp_n_tokens": fpp_tokens,
                "spp_n_tokens": spp_tokens,
                "fpp_n_syllables": fpp_syllables,
                "spp_n_syllables": spp_syllables,
                "fpp_speech_rate_syllables_per_s": fpp_speech_rate,
                "spp_speech_rate_syllables_per_s": spp_speech_rate,
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

    Usage example
    -------------
        paths = discover_textgrid_paths(Path("/data"))
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

    Usage example
    -------------
        write_rows_to_csv(rows, Path("output.csv"))
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
        "fpp_n_syllables",
        "spp_n_syllables",
        "fpp_speech_rate_syllables_per_s",
        "spp_speech_rate_syllables_per_s",
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


def main() -> None:
    """CLI entry point.

    Usage example
    -------------
        main()
    """
    textgrid_paths = discover_textgrid_paths(INPUT_DIR)
    if not textgrid_paths:
        raise FileNotFoundError(
            f"No files matching 'dyad-*_run-*_combined.TextGrid' found in {INPUT_DIR}"
        )

    syllable_index = build_syllable_interval_index(SYLLABLE_DIR)
    if not syllable_index:
        raise FileNotFoundError(
            f"No files matching 'sub-*_run-*_syllable.csv' found in {SYLLABLE_DIR}"
        )

    all_rows: List[Dict[str, object]] = []
    for textgrid_path in textgrid_paths:
        all_rows.extend(
            pair_events_for_file(
                path=textgrid_path,
                syllable_index=syllable_index,
                debug_tier_selection=DEBUG_TIER_SELECTION,
            )
        )

    write_rows_to_csv(all_rows, OUTPUT_CSV)
    print(f"Wrote {len(all_rows)} matched FPP-SPP pairs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
