#!/usr/bin/env python3
"""Minimal preflight checks for claire_surprisal.py inputs.

Goal
----
Catch common input problems early so the GPU job does not hard-fail after
startup. This script is intentionally simple and conservative.

Exit codes
----------
0 -> inputs look safe enough to try
1 -> one or more blocking problems found

Usage
-----
python preflight_surprisal_inputs.py \
    --tokens_csv tokens.csv \
    --words_csv words.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_TOKEN_COLUMNS = {"token", "speaker", "start", "end"}
REQUIRED_WORD_COLUMNS = {"word"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens_csv", type=Path, required=True)
    parser.add_argument("--words_csv", type=Path, required=True)
    return parser.parse_args()


def add_error(errors: List[str], message: str) -> None:
    """Append one blocking error."""
    errors.append(message)


def add_warning(warnings: List[str], message: str) -> None:
    """Append one non-blocking warning."""
    warnings.append(message)


def validate_tokens_df(tokens_df: pd.DataFrame, errors: List[str], warnings: List[str]) -> None:
    """Run blocking and non-blocking checks on the token table."""
    missing_columns = REQUIRED_TOKEN_COLUMNS.difference(tokens_df.columns)
    if missing_columns:
        add_error(errors, f"tokens_csv missing required columns: {sorted(missing_columns)}")
        return

    if len(tokens_df) == 0:
        add_error(errors, "tokens_csv has zero rows.")
        return

    # Coerce timing columns to numeric
    for column_name in ("start", "end"):
        coerced = pd.to_numeric(tokens_df[column_name], errors="coerce")
        n_bad = int(coerced.isna().sum())
        if n_bad > 0:
            add_error(errors, f"tokens_csv column '{column_name}' has {n_bad} non-numeric values.")
        tokens_df[column_name] = coerced

    if errors:
        return

    token_text = tokens_df["token"].astype(str)
    speaker_text = tokens_df["speaker"].astype(str)

    n_empty_tokens = int(token_text.str.strip().eq("").sum())
    if n_empty_tokens > 0:
        add_warning(warnings, f"tokens_csv contains {n_empty_tokens} empty token strings.")

    n_empty_speakers = int(speaker_text.str.strip().eq("").sum())
    if n_empty_speakers > 0:
        add_error(errors, f"tokens_csv contains {n_empty_speakers} empty speaker labels.")

    n_bad_intervals = int((tokens_df["end"] < tokens_df["start"]).sum())
    if n_bad_intervals > 0:
        add_error(errors, f"tokens_csv contains {n_bad_intervals} rows where end < start.")

    n_missing_time = int(tokens_df["start"].isna().sum() + tokens_df["end"].isna().sum())
    if n_missing_time > 0:
        add_error(errors, f"tokens_csv contains {n_missing_time} missing start/end values.")

    lexical_mask = token_text.str.strip().ne("#") & token_text.str.strip().ne("")
    n_lexical = int(lexical_mask.sum())
    if n_lexical == 0:
        add_error(errors, "tokens_csv contains no lexical tokens after excluding '#' and empty strings.")

    # Soft sanity checks only
    n_negative_times = int(((tokens_df["start"] < 0) | (tokens_df["end"] < 0)).sum())
    if n_negative_times > 0:
        add_warning(warnings, f"tokens_csv contains {n_negative_times} rows with negative timestamps.")

    n_speakers = int(speaker_text.nunique())
    if n_speakers == 0:
        add_error(errors, "tokens_csv has no speaker labels.")
    elif n_speakers > 8:
        add_warning(warnings, f"tokens_csv has {n_speakers} unique speakers; verify labels are clean.")

    # Check sortedness; not blocking because the script can still run, but output may be odd
    sorted_df = tokens_df.sort_values(["start", "end"]).reset_index(drop=True)
    same_order = tokens_df.reset_index(drop=True)[["token", "speaker", "start", "end"]].equals(
        sorted_df[["token", "speaker", "start", "end"]]
    )
    if not same_order:
        add_warning(warnings, "tokens_csv is not globally sorted by start/end time.")

    # Large prompt risk hint
    if n_lexical > 15000:
        add_warning(
            warnings,
            f"tokens_csv has {n_lexical} lexical tokens. This may be slow or exceed comfortable context limits.",
        )


def validate_words_df(words_df: pd.DataFrame, errors: List[str], warnings: List[str]) -> None:
    """Run blocking and non-blocking checks on the canonical word table."""
    missing_columns = REQUIRED_WORD_COLUMNS.difference(words_df.columns)
    if missing_columns:
        add_error(errors, f"words_csv missing required columns: {sorted(missing_columns)}")
        return

    if len(words_df) == 0:
        add_error(errors, "words_csv has zero rows.")
        return

    word_text = words_df["word"].astype(str)
    n_empty_words = int(word_text.str.strip().eq("").sum())
    if n_empty_words > 0:
        add_warning(warnings, f"words_csv contains {n_empty_words} empty canonical words.")

    # Heuristic only: if words.csv looks like a vocabulary list with counts rather than transcript order
    has_count_column = "count" in words_df.columns
    is_sorted_alpha = word_text.fillna("").str.lower().tolist() == sorted(word_text.fillna("").str.lower().tolist())
    if has_count_column and is_sorted_alpha:
        add_warning(
            warnings,
            "words_csv looks like an alphabetical vocabulary list with counts, not a transcript-order canonical word list.",
        )


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    if not args.tokens_csv.exists():
        add_error(errors, f"tokens_csv not found: {args.tokens_csv}")

    if not args.words_csv.exists():
        add_error(errors, f"words_csv not found: {args.words_csv}")

    if errors:
        print("PRECHECK FAILED")
        for message in errors:
            print(f"[ERROR] {message}")
        sys.exit(1)

    try:
        tokens_df = pd.read_csv(args.tokens_csv)
    except Exception as exc:  # noqa: BLE001
        add_error(errors, f"Could not read tokens_csv: {exc}")
        tokens_df = None

    try:
        words_df = pd.read_csv(args.words_csv)
    except Exception as exc:  # noqa: BLE001
        add_error(errors, f"Could not read words_csv: {exc}")
        words_df = None

    if tokens_df is not None:
        validate_tokens_df(tokens_df=tokens_df, errors=errors, warnings=warnings)

    if words_df is not None:
        validate_words_df(words_df=words_df, errors=errors, warnings=warnings)

    if errors:
        print("PRECHECK FAILED")
        for message in errors:
            print(f"[ERROR] {message}")
        for message in warnings:
            print(f"[WARN]  {message}")
        sys.exit(1)

    print("PRECHECK OK")
    for message in warnings:
        print(f"[WARN]  {message}")
    sys.exit(0)


if __name__ == "__main__":
    main()