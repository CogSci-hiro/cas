#!/usr/bin/env python3
"""Identify IPUs with at least N overlapping tokens in Praat TextGrid files.

This script scans a directory tree for files named like::

    dyad-003_run-1_combined.TextGrid

and reports any IPU interval whose overlapping token count is at least a chosen
threshold (default: 60).

Assumptions
-----------
1. IPUs are stored in tiers named ``ipu-A`` and ``ipu-B``.
2. Token intervals are stored in tiers named ``palign-A`` and ``palign-B``.
3. Token count for an IPU is the number of non-empty, non-silence token
   intervals in the corresponding palign tier that overlap the IPU.

Important
---------
This script is intentionally simple and diagnostic. It does not try to infer
whether the palign tier is word-like or phone-like. Its purpose is to surface
suspiciously large IPUs so you can inspect the underlying files and timestamps.

Usage example
-------------
    python find_long_ipus.py \
        --input-dir /Users/hiro/Datasets/anais \
        --output-csv /Users/hiro/Datasets/anais/ipu_60plus.csv \
        --min-tokens 60 \
        --print-matches
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ###########################################################################
# Parsing constants
# ###########################################################################

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
SPEAKER_TIER_PATTERN = re.compile(
    r"(?P<prefix>actions?|palign|ipu)[-\s]+(?P<speaker>[ab])",
    re.IGNORECASE,
)

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


# ###########################################################################
# Data containers
# ###########################################################################

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

    Usage example
    -------------
        interval = Interval(1.0, 1.5, "hello")
    """

    onset_s: float
    offset_s: float
    text: str

    @property
    def duration_s(self) -> float:
        """Return interval duration in seconds.

        Usage example
        -------------
            duration_s = interval.duration_s
        """
        return self.offset_s - self.onset_s


# ###########################################################################
# TextGrid parsing
# ###########################################################################

def read_textgrid_text(path: Path) -> str:
    """Read a TextGrid file with encoding fallback.

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


def canonicalize_tier_name(tier_name: str) -> str:
    """Normalize tier-name variants to one canonical spelling.

    Examples
    --------
    ``ipu A`` -> ``ipu-A``
    ``palign A`` -> ``palign-A``
    ``actions a`` -> ``actions A``

    Usage example
    -------------
        canonical_name = canonicalize_tier_name("ipu A")
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

    return compact_name


def parse_interval_tiers(text: str) -> Dict[str, List[Interval]]:
    """Parse IntervalTiers from a Praat ooTextFile TextGrid.

    Parameters
    ----------
    text
        Full TextGrid text.

    Returns
    -------
    dict of str to list of Interval
        Mapping from tier name to intervals.

    Notes
    -----
    This version refuses to silently merge duplicate canonicalized tier names.

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


# ###########################################################################
# Counting helpers
# ###########################################################################

def interval_overlap_s(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Return overlap duration between two closed-open intervals.

    Usage example
    -------------
        overlap_s = interval_overlap_s(0.0, 1.0, 0.5, 1.5)
    """
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalize_token_label(token_label: str) -> str:
    """Normalize token labels before lexical filtering.

    Usage example
    -------------
        normalized_label = normalize_token_label(" sil ")
    """
    return token_label.strip()


def count_overlapping_tokens(
    token_intervals: Sequence[Interval],
    onset_s: float,
    offset_s: float,
) -> int:
    """Count overlapping non-ignored token intervals.

    Parameters
    ----------
    token_intervals
        Intervals from the corresponding ``palign`` tier.
    onset_s
        IPU onset.
    offset_s
        IPU offset.

    Returns
    -------
    int
        Number of overlapping token intervals.

    Usage example
    -------------
        n_tokens = count_overlapping_tokens(token_intervals, 1.0, 3.0)
    """
    n_tokens = 0
    for interval in token_intervals:
        token_label = normalize_token_label(interval.text)
        if token_label in TOKEN_IGNORE_LABELS:
            continue
        if interval_overlap_s(onset_s, offset_s, interval.onset_s, interval.offset_s) > 0.0:
            n_tokens += 1
    return n_tokens


# ###########################################################################
# File processing
# ###########################################################################

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


def find_long_ipus_in_file(path: Path, min_tokens: int) -> List[Dict[str, object]]:
    """Find IPUs in one file whose token count reaches the threshold.

    Parameters
    ----------
    path
        Input TextGrid path.
    min_tokens
        Minimum token count threshold.

    Returns
    -------
    list of dict
        CSV-ready result rows.

    Usage example
    -------------
        rows = find_long_ipus_in_file(Path("file.TextGrid"), 60)
    """
    tiers = parse_interval_tiers(read_textgrid_text(path))

    rows: List[Dict[str, object]] = []

    for speaker_id in ("A", "B"):
        ipu_tier_name = f"ipu-{speaker_id}"
        palign_tier_name = f"palign-{speaker_id}"

        ipu_intervals = tiers.get(ipu_tier_name, [])
        token_intervals = tiers.get(palign_tier_name, [])

        if not ipu_intervals or not token_intervals:
            continue

        for ipu_interval in ipu_intervals:
            n_tokens = count_overlapping_tokens(
                token_intervals=token_intervals,
                onset_s=ipu_interval.onset_s,
                offset_s=ipu_interval.offset_s,
            )

            if n_tokens >= min_tokens:
                rows.append(
                    {
                        "file_name": path.name,
                        "file_path": str(path),
                        "speaker_id": speaker_id,
                        "ipu_onset_s": ipu_interval.onset_s,
                        "ipu_offset_s": ipu_interval.offset_s,
                        "ipu_duration_s": ipu_interval.duration_s,
                        "n_tokens": n_tokens,
                        "ipu_label": ipu_interval.text,
                    }
                )

    return rows


def write_rows_to_csv(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    """Write result rows to CSV.

    Parameters
    ----------
    rows
        Result rows.
    output_csv
        Destination CSV path.

    Usage example
    -------------
        write_rows_to_csv(rows, Path("output.csv"))
    """
    fieldnames = [
        "file_name",
        "file_path",
        "speaker_id",
        "ipu_onset_s",
        "ipu_offset_s",
        "ipu_duration_s",
        "n_tokens",
        "ipu_label",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ###########################################################################
# CLI
# ###########################################################################

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Usage example
    -------------
        parser = build_argument_parser()
    """
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
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=60,
        help="Minimum token count threshold. Default: 60.",
    )
    parser.add_argument(
        "--print-matches",
        action="store_true",
        help="Print matching file/timestamp rows to stdout.",
    )
    return parser


def main() -> None:
    """CLI entry point.

    Usage example
    -------------
        main()
    """
    parser = build_argument_parser()
    arguments = parser.parse_args()

    textgrid_paths = discover_textgrid_paths(arguments.input_dir)
    if not textgrid_paths:
        raise FileNotFoundError(
            f"No files matching 'dyad-*_run-*_combined.TextGrid' found in {arguments.input_dir}"
        )

    all_rows: List[Dict[str, object]] = []

    for textgrid_path in textgrid_paths:
        all_rows.extend(find_long_ipus_in_file(textgrid_path, arguments.min_tokens))

    write_rows_to_csv(all_rows, arguments.output_csv)

    if arguments.print_matches:
        for row in all_rows:
            print(
                f"{row['file_name']}, "
                f"speaker={row['speaker_id']}, "
                f"onset={row['ipu_onset_s']:.3f}, "
                f"offset={row['ipu_offset_s']:.3f}, "
                f"duration={row['ipu_duration_s']:.3f}, "
                f"n_tokens={row['n_tokens']}"
            )

    print(
        f"Wrote {len(all_rows)} IPUs with at least {arguments.min_tokens} tokens "
        f"to {arguments.output_csv}"
    )


if __name__ == "__main__":
    main()