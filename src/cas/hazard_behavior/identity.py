"""Helpers for canonical dyad-specific participant-speaker identities."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_participant_speaker_id(
    data: pd.DataFrame,
    *,
    dyad_col: str = "dyad_id",
    speaker_col: str = "speaker",
    output_col: str = "participant_speaker_id",
) -> dict[str, Any]:
    """Validate that a participant-speaker identity column is dyad-specific."""

    missing_columns = [column_name for column_name in (dyad_col, speaker_col) if column_name not in data.columns]
    if missing_columns:
        raise ValueError(
            "Participant-speaker identity validation requires columns: "
            + ", ".join(missing_columns)
        )

    working = data.copy()
    dyad = _coerce_required_identity_component(working, column_name=dyad_col)
    speaker = _coerce_required_identity_component(working, column_name=speaker_col)
    expected = dyad + "_" + speaker

    n_unique_dyad = int(dyad.nunique(dropna=True))
    n_unique_speaker = int(speaker.nunique(dropna=True))
    n_unique_pairs = int(expected.nunique(dropna=True))

    if output_col not in working.columns:
        return {
            "n_unique_dyad_id": n_unique_dyad,
            "n_unique_speaker": n_unique_speaker,
            "n_unique_dyad_speaker_pairs": n_unique_pairs,
            "n_unique_participant_speaker_id": 0,
            "participant_speaker_id_valid": False,
            "participant_speaker_id_column_used": output_col,
            "participant_speaker_id_matches_canonical": False,
        }

    output = _coerce_required_identity_component(working, column_name=output_col)
    canonical_values = expected.astype(str)
    output_values = output.astype(str)
    output_unique_values = set(output_values.dropna().unique().tolist())
    collapsed_labels = output_unique_values <= {"A", "B"} if output_unique_values else False
    n_unique_output = int(output.nunique(dropna=True))
    matches_canonical = bool(output_values.equals(canonical_values))
    is_valid = bool(
        not collapsed_labels
        and n_unique_output >= n_unique_pairs
        and matches_canonical
    )
    return {
        "n_unique_dyad_id": n_unique_dyad,
        "n_unique_speaker": n_unique_speaker,
        "n_unique_dyad_speaker_pairs": n_unique_pairs,
        "n_unique_participant_speaker_id": n_unique_output,
        "participant_speaker_id_valid": is_valid,
        "participant_speaker_id_column_used": output_col,
        "participant_speaker_id_matches_canonical": matches_canonical,
    }


def ensure_participant_speaker_id(
    data: pd.DataFrame,
    dyad_col: str = "dyad_id",
    speaker_col: str = "speaker",
    output_col: str = "participant_speaker_id",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create or validate a canonical dyad-specific participant-speaker ID column."""

    counts = validate_participant_speaker_id(
        data,
        dyad_col=dyad_col,
        speaker_col=speaker_col,
        output_col=output_col,
    )
    working = data.copy()
    canonical = _coerce_required_identity_component(working, column_name=dyad_col) + "_" + _coerce_required_identity_component(
        working,
        column_name=speaker_col,
    )
    if output_col not in working.columns or overwrite:
        working[output_col] = canonical
        return working
    if counts["participant_speaker_id_valid"]:
        return working
    if counts["n_unique_participant_speaker_id"] < counts["n_unique_dyad_speaker_pairs"]:
        raise ValueError(
            "Invalid participant-speaker identity: "
            f"`{output_col}` has {counts['n_unique_participant_speaker_id']} unique values, "
            f"but dyad/speaker pairs imply {counts['n_unique_dyad_speaker_pairs']}. "
            "This usually means participant identities were collapsed to plain A/B labels. "
            "Pass overwrite=True to replace the invalid values with canonical dyad-specific IDs."
        )
    raise ValueError(
        "Invalid participant-speaker identity: "
        f"`{output_col}` does not match the canonical `{dyad_col}_{speaker_col}` form. "
        "Pass overwrite=True to standardize it."
    )


def _coerce_required_identity_component(data: pd.DataFrame, *, column_name: str) -> pd.Series:
    values = data[column_name]
    if values.isna().any():
        raise ValueError(f"Participant-speaker identity requires non-missing values in `{column_name}`.")
    text = values.astype(str).str.strip()
    if (text == "").any():
        raise ValueError(f"Participant-speaker identity requires non-empty values in `{column_name}`.")
    return text
