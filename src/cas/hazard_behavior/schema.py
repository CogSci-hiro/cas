"""Schema normalization helpers for behavioural hazard inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EVENT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dyad_id": ("dyad_id", "dyad", "recording_id"),
    "run": ("run",),
    "participant_speaker": ("participant_speaker", "speaker", "speaker_fpp"),
    "partner_speaker": ("partner_speaker", "speaker_spp"),
    "fpp_onset": ("fpp_onset",),
    "fpp_offset": ("fpp_offset",),
    "fpp_label": ("fpp_label", "fpp_type"),
    "spp_onset": ("spp_onset", "partner_ipu_onset", "previous_partner_onset"),
    "spp_offset": ("spp_offset", "partner_ipu_offset", "previous_partner_offset"),
    "spp_label": ("spp_label", "spp_type", "partner_ipu_class"),
    "pair_id": ("pair_id",),
    "subject_id": ("subject_id", "participant_id", "subject"),
}

SURPRISAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dyad_id": ("dyad_id", "dyad", "recording_id"),
    "run": ("run",),
    "speaker": ("speaker",),
    "onset": ("onset",),
    "duration": ("duration",),
    "word": ("word",),
    "normalized_word": ("normalized_word",),
    "surprisal": ("surprisal",),
    "alignment_status": ("alignment_status",),
    "word_index": ("word_index",),
    "lm_token_id": ("lm_token_id",),
    "source_interval_id": ("source_interval_id",),
}


@dataclass(frozen=True, slots=True)
class SchemaNormalizationResult:
    """Normalized schema output plus provenance notes."""

    table: pd.DataFrame
    used_columns: dict[str, str]
    warnings: list[str]


def normalize_events_schema(events_table: pd.DataFrame) -> SchemaNormalizationResult:
    """Normalize event columns to the behavioural hazard schema.

    Usage example
    -------------
        normalized = normalize_events_schema(events_df)
        events = normalized.table
    """

    return _normalize_schema(
        table=events_table,
        aliases=EVENT_COLUMN_ALIASES,
        required={"dyad_id", "run", "participant_speaker", "fpp_onset"},
        table_label="events",
    )


def normalize_surprisal_schema(surprisal_table: pd.DataFrame) -> SchemaNormalizationResult:
    """Normalize surprisal columns to the behavioural hazard schema."""

    return _normalize_schema(
        table=surprisal_table,
        aliases=SURPRISAL_COLUMN_ALIASES,
        required={"dyad_id", "run", "speaker", "onset", "duration", "surprisal"},
        table_label="surprisal",
    )


def _normalize_schema(
    *,
    table: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
    required: set[str],
    table_label: str,
) -> SchemaNormalizationResult:
    used_columns: dict[str, str] = {}
    warnings: list[str] = []
    rename_map: dict[str, str] = {}

    for canonical_name, candidate_names in aliases.items():
        matched_name = next((name for name in candidate_names if name in table.columns), None)
        if matched_name is None:
            continue
        used_columns[canonical_name] = matched_name
        if matched_name != canonical_name:
            rename_map[matched_name] = canonical_name
            warnings.append(
                f"Normalized {table_label} column `{matched_name}` to canonical name `{canonical_name}`."
            )

    missing = sorted(required - set(used_columns))
    if missing:
        available = ", ".join(sorted(str(column) for column in table.columns))
        raise ValueError(
            f"{table_label.capitalize()} table is missing required columns {missing}. "
            f"Available columns: {available}"
        )

    normalized = table.rename(columns=rename_map).copy()
    return SchemaNormalizationResult(table=normalized, used_columns=used_columns, warnings=warnings)
