#!/usr/bin/env python3
"""
Convert SPPAS palign token annotations into dyad-level token tables while
preserving original SPPAS token structure and adding clean rendering fields
for language-model inference.

Key behavior
------------
- Keeps original token rows and timestamps unchanged.
- Splits underscore-joined multiword tokens for LM rendering:
    du_coup -> ["du", "coup"]
- Skips configured special symbols from LM rendering but keeps them in output:
    @, *
- Keeps silence markers (#) in the output but does not render them to the LM.
- Preserves hyphenated incomplete forms as lexical tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKENS_TIER: str = "TokensAlign"
SILENCE_TOKEN: str = "#"

DEFAULT_SKIP_SYMBOL_TOKENS: tuple[str, ...] = (
    "@",
    "*",
)

PUNCTUATION_ONLY_PATTERN = re.compile(r"^[^\wÀ-ÿ]+$", re.UNICODE)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SPPAS palign annotations to dyad token tables with LM rendering fields."
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
    parser.add_argument(
        "--skip_symbol_tokens",
        nargs="*",
        default=list(DEFAULT_SKIP_SYMBOL_TOKENS),
        help="Tokens that should remain in the output CSV but be skipped from LM rendering.",
    )
    parser.add_argument(
        "--write_words_in_order",
        action="store_true",
        help="Also write one words-in-order CSV per dyad, using rendered LM pieces in transcript order.",
    )
    return parser.parse_args()


def subject_to_dyad(subject_id: int) -> str:
    """Convert subject ID to dyad ID."""
    dyad_number: int = (subject_id + 1) // 2
    return f"dyad-{dyad_number:03d}"


def subject_to_speaker(subject_id: int) -> str:
    """Assign speaker label from subject ID."""
    return "A" if subject_id % 2 == 1 else "B"


def extract_metadata(filename: str) -> tuple[int, int]:
    """Extract subject ID and run number from filename."""
    match = re.match(r"sub-(\d+)_run-(\d+)_palign\.csv", filename)
    if match is None:
        raise ValueError(f"Unexpected filename format: {filename}")
    subject_id = int(match.group(1))
    run_id = int(match.group(2))
    return subject_id, run_id


def classify_token(token_text: str, skip_symbol_tokens: set[str]) -> str:
    """Classify a raw SPPAS token."""
    stripped_token = token_text.strip()

    if stripped_token == SILENCE_TOKEN:
        return "silence"

    if stripped_token in skip_symbol_tokens:
        return "skip_symbol"

    if "_" in stripped_token:
        return "underscore_multiword"

    if PUNCTUATION_ONLY_PATTERN.fullmatch(stripped_token) is not None:
        return "punctuation_only"

    return "lexical"


def token_to_rendered_pieces(token_text: str, skip_symbol_tokens: set[str]) -> list[str]:
    """
    Convert one original token to the text pieces seen by the LM.

    Rules
    -----
    - "#" -> []
    - configured skip symbols such as "@" or "*" -> []
    - underscore tokens -> split on "_"
    - hyphenated incomplete forms are kept as-is
    - everything else is kept as-is
    """
    stripped_token = token_text.strip()

    if stripped_token == "":
        return []

    if stripped_token == SILENCE_TOKEN:
        return []

    if stripped_token in skip_symbol_tokens:
        return []

    if "_" in stripped_token:
        return [
            piece.strip()
            for piece in stripped_token.split("_")
            if piece.strip() != ""
        ]

    return [stripped_token]


def read_tokens_from_file(
    filepath: Path,
    speaker: str,
    run_id: int,
    token_counter: Counter[str],
    skip_symbol_tokens: set[str],
) -> list[dict[str, Any]]:
    """Extract token rows from one SPPAS palign file."""
    rows: list[dict[str, Any]] = []

    with filepath.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)

        for row in reader:
            if len(row) != 4:
                continue

            tier_name, start_text, end_text, annotation_text = row

            if tier_name != TOKENS_TIER:
                continue

            token_text = annotation_text.strip()
            if token_text == "":
                continue

            start_s = float(start_text)
            end_s = float(end_text)

            rendered_pieces = token_to_rendered_pieces(
                token_text=token_text,
                skip_symbol_tokens=skip_symbol_tokens,
            )
            token_kind = classify_token(
                token_text=token_text,
                skip_symbol_tokens=skip_symbol_tokens,
            )

            rows.append(
                {
                    "run": run_id,
                    "token": token_text,
                    "speaker": speaker,
                    "start": start_s,
                    "end": end_s,
                    "token_kind": token_kind,
                    "render_for_lm": bool(len(rendered_pieces) > 0),
                    "rendered_text": " ".join(rendered_pieces),
                    "rendered_piece_count": len(rendered_pieces),
                    "rendered_pieces_json": json.dumps(rendered_pieces, ensure_ascii=False),
                }
            )

            token_counter[token_text] += 1

    return rows


def write_dyad_outputs(
    dyad_data: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Write dyad-level token CSV files."""
    fieldnames = [
        "run",
        "token",
        "speaker",
        "start",
        "end",
        "token_kind",
        "render_for_lm",
        "rendered_text",
        "rendered_piece_count",
        "rendered_pieces_json",
    ]

    for dyad_id, rows in dyad_data.items():
        output_path = out_dir / f"{dyad_id}_tokens.csv"

        rows_sorted = sorted(
            rows,
            key=lambda row: (
                int(row["run"]),
                float(row["start"]),
                float(row["end"]),
            ),
        )

        with output_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)


def iter_rendered_words_in_order(rows: list[dict[str, Any]]):
    """Yield rendered lexical pieces in transcript order."""
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            int(row["run"]),
            float(row["start"]),
            float(row["end"]),
        ),
    )

    for row in rows_sorted:
        rendered_pieces = json.loads(str(row["rendered_pieces_json"]))
        for rendered_piece in rendered_pieces:
            yield str(rendered_piece)


def write_words_in_order_csvs(
    dyad_data: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Write one transcript-order words CSV per dyad."""
    for dyad_id, rows in dyad_data.items():
        output_path = out_dir / f"{dyad_id}_words_in_order.csv"

        with output_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["word"])

            for rendered_word in iter_rendered_words_in_order(rows):
                writer.writerow([rendered_word])


def write_words_vocab_csv(
    token_counter: Counter[str],
    out_dir: Path,
) -> None:
    """Write vocabulary-style words CSV with alphabetical ordering."""
    output_path = out_dir / "words_vocab.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["word", "count"])

        for word in sorted(token_counter.keys()):
            writer.writerow([word, token_counter[word]])


def main() -> None:
    """Main CLI entry point."""
    arguments = parse_args()

    input_dir: Path = arguments.in_dir
    output_dir: Path = arguments.out_dir
    skip_symbol_tokens: set[str] = set(arguments.skip_symbol_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)

    dyad_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_counter: Counter[str] = Counter()

    for filepath in sorted(input_dir.glob("sub-*_run-*_palign.csv")):
        subject_id, run_id = extract_metadata(filepath.name)
        dyad_id = subject_to_dyad(subject_id)
        speaker = subject_to_speaker(subject_id)

        token_rows = read_tokens_from_file(
            filepath=filepath,
            speaker=speaker,
            run_id=run_id,
            token_counter=token_counter,
            skip_symbol_tokens=skip_symbol_tokens,
        )

        dyad_data[dyad_id].extend(token_rows)

    write_dyad_outputs(
        dyad_data=dyad_data,
        out_dir=output_dir,
    )
    write_words_vocab_csv(
        token_counter=token_counter,
        out_dir=output_dir,
    )

    if arguments.write_words_in_order:
        write_words_in_order_csvs(
            dyad_data=dyad_data,
            out_dir=output_dir,
        )

    print("Done.")
    print(f"Dyad token files written to: {output_dir}")
    print(f"Original token vocabulary size: {len(token_counter)}")
    print(f"Skipped LM symbols: {sorted(skip_symbol_tokens)}")


if __name__ == "__main__":
    main()