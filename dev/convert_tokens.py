#!/usr/bin/env python3

"""
Convert SPPAS palign token annotations into dyad-level token tables.

Input:
    Flat directory containing files like:
        sub-001_run-1_palign.csv

Expected CSV structure:
    tier,start,end,annotation

Output:
    <out_dir>/
        dyad-001_tokens.csv
        dyad-002_tokens.csv
        ...
        words.csv

Per-dyad CSV columns:
    run,token,speaker,start,end

words.csv columns:
    word,count

Usage example:

    python palign_to_dyad_tokens.py \
        --in_dir "/path/to/palign" \
        --out_dir "/path/to/output"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


TOKENS_TIER: str = "TokensAlign"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SPPAS palign annotations to dyad token tables."
    )

    parser.add_argument(
        "--in_dir",
        required=True,
        type=Path,
        help="Directory containing palign CSV files.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Directory to save dyad token outputs.",
    )

    return parser.parse_args()


def subject_to_dyad(subject_id: int) -> str:
    """
    Convert subject ID to dyad ID.

    Example:
        001 -> dyad-001
        002 -> dyad-001
        003 -> dyad-002
    """
    dyad_number: int = (subject_id + 1) // 2
    return f"dyad-{dyad_number:03d}"


def subject_to_speaker(subject_id: int) -> str:
    """
    Assign speaker label.

    Odd subject -> A
    Even subject -> B
    """
    return "A" if subject_id % 2 == 1 else "B"


def extract_metadata(filename: str) -> tuple[int, int]:
    """
    Extract subject ID and run number from filename.

    Expected format:
        sub-001_run-1_palign.csv
    """
    match = re.match(r"sub-(\d+)_run-(\d+)_palign\.csv", filename)

    if match is None:
        raise ValueError(f"Unexpected filename format: {filename}")

    subject_id = int(match.group(1))
    run_id = int(match.group(2))

    return subject_id, run_id


def read_tokens_from_file(
    filepath: Path,
    speaker: str,
    run_id: int,
    token_counter: Counter,
) -> List[Dict]:
    """
    Extract TokensAlign tier rows from CSV.

    Returns:
        list of token dictionaries
    """
    tokens: List[Dict] = []

    with filepath.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) != 4:
                continue

            tier, start, end, annotation = row

            if tier != TOKENS_TIER:
                continue

            token = annotation.strip()

            if token == "":
                continue

            tokens.append(
                {
                    "run": run_id,
                    "token": token,
                    "speaker": speaker,
                    "start": float(start),
                    "end": float(end),
                }
            )

            token_counter[token] += 1

    return tokens


def write_dyad_outputs(
    dyad_data: Dict[str, List[Dict]],
    out_dir: Path,
) -> None:
    """
    Write dyad-level token CSV files.
    """
    for dyad_id, rows in dyad_data.items():
        output_path = out_dir / f"{dyad_id}_tokens.csv"

        rows_sorted = sorted(
            rows,
            key=lambda x: (x["run"], x["start"], x["end"]),
        )

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run", "token", "speaker", "start", "end"],
            )

            writer.writeheader()
            writer.writerows(rows_sorted)


def write_words_csv(
    token_counter: Counter,
    out_dir: Path,
) -> None:
    """
    Write words.csv with alphabetical ordering.
    """
    output_path = out_dir / "words.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["word", "count"])

        for word in sorted(token_counter.keys()):
            writer.writerow([word, token_counter[word]])


def main() -> None:
    """Main entry point."""
    args = parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    dyad_data: Dict[str, List[Dict]] = defaultdict(list)
    token_counter: Counter = Counter()

    for filepath in sorted(in_dir.glob("sub-*_run-*_palign.csv")):
        subject_id, run_id = extract_metadata(filepath.name)

        dyad_id = subject_to_dyad(subject_id)
        speaker = subject_to_speaker(subject_id)

        tokens = read_tokens_from_file(
            filepath=filepath,
            speaker=speaker,
            run_id=run_id,
            token_counter=token_counter,
        )

        dyad_data[dyad_id].extend(tokens)

    write_dyad_outputs(
        dyad_data=dyad_data,
        out_dir=out_dir,
    )

    write_words_csv(
        token_counter=token_counter,
        out_dir=out_dir,
    )

    print("Done.")
    print(f"Dyad files written to: {out_dir}")
    print(f"Unique words: {len(token_counter)}")


if __name__ == "__main__":
    main()