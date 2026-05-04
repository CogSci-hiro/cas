"""Internal support code migrated from deprecated behavioral surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
import glob
import logging
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

UnmatchedSurprisalStrategy = Literal["drop", "zero", "keep_nan"]
TokenAvailability = Literal["onset", "offset"]
ExpectedInfoGroup = Literal["partner_ipu_class", "partner_role", "global"]
EpisodeAnchor = Literal["partner_ipu"]

EPSILON = 1.0e-12
FLOAT_TOLERANCE = 1.0e-9
DEFAULT_PARTNER_IPU_CLASS = "unknown"
SUPPORTED_SPEAKERS = {"A", "B"}
ACTIVE_LAGGED_INFORMATION_FEATURES = (
    "information_rate",
    "prop_expected_cumulative_info",
)
REQUIRED_FINAL_COLUMNS = [
    "episode_id",
    "dyad_id",
    "subject_id",
    "run_id",
    "anchor_type",
    "bin_start_s",
    "bin_end_s",
    "event_bin",
    "time_from_partner_onset_s",
    "time_from_partner_offset_s",
    "information_rate",
    "prop_expected_cumulative_info",
]
EXPECTED_ANCHOR_TYPES = {"fpp", "spp"}
LOGGER = logging.getLogger(__name__)

EVENT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dyad_id": ("dyad_id", "dyad", "recording_id"),
    "run": ("run",),
    "participant_speaker_id": ("participant_speaker_id",),
    "participant_speaker": ("participant_speaker", "speaker", "speaker_fpp", "fpp_speaker"),
    "partner_speaker": ("partner_speaker", "speaker_spp"),
    "fpp_onset": ("fpp_onset", "onset"),
    "fpp_offset": ("fpp_offset", "offset"),
    "fpp_label": ("fpp_label", "fpp_type", "label", "type"),
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


def progress_iterable(
    iterable: Iterable,
    *,
    total: int | None,
    description: str,
    enabled: bool = True,
) -> Iterator:
    if not enabled:
        yield from iterable
        return
    yield from tqdm(
        iterable,
        total=total,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )


@dataclass(frozen=True, slots=True)
class BehaviourHazardConfig:
    events_path: Path
    surprisal_paths: tuple[Path, ...]
    out_dir: Path
    bin_size_s: float = 0.050
    information_rate_window_s: float = 0.500
    minimum_episode_duration_s: float = 0.100
    unmatched_surprisal_strategy: UnmatchedSurprisalStrategy = "drop"
    include_quadratic_offset_timing: bool = True
    ipu_gap_threshold_s: float = 0.300
    max_followup_s: float = 6.0
    episode_anchor: EpisodeAnchor = "partner_ipu"
    include_censored: bool = True
    token_availability: TokenAvailability = "onset"
    expected_info_group: ExpectedInfoGroup = "partner_ipu_class"
    target_fpp_label_prefix: str = "FPP_"
    require_partner_offset_before_fpp: bool = False
    partner_offset_fpp_tolerance_s: float = 0.020
    overlapping_episode_strategy: Literal["exclude", "truncate", "keep"] = "keep"
    cluster_column: str | None = None
    overwrite: bool = False
    save_riskset: bool = True
    clip_proportions: bool = False
    clip_range: tuple[float, float] = (0.0, 1.5)
    lag_grid_ms: tuple[int, ...] = (0, 50, 100, 150, 200, 300, 500, 700, 1000)
    lagged_feature_fill_value: float = 0.0
    default_output_prefix: str = "hazard_behavior_fpp"
    default_expected_info_group_column: str = "partner_ipu_class"
    default_model_family: str = "binomial_glm"
    drop_unmatched_surprisal: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surprisal_paths", tuple(Path(path) for path in self.surprisal_paths))
        object.__setattr__(self, "drop_unmatched_surprisal", self.unmatched_surprisal_strategy == "drop")
        self.validate()

    def validate(self) -> None:
        if self.bin_size_s <= 0.0:
            raise ValueError("`bin_size_s` must be positive.")
        if self.information_rate_window_s <= 0.0:
            raise ValueError("`information_rate_window_s` must be positive.")
        if self.minimum_episode_duration_s <= 0.0:
            raise ValueError("`minimum_episode_duration_s` must be positive.")
        if self.ipu_gap_threshold_s < 0.0:
            raise ValueError("`ipu_gap_threshold_s` must be non-negative.")
        if self.max_followup_s <= 0.0:
            raise ValueError("`max_followup_s` must be positive.")
        if self.episode_anchor != "partner_ipu":
            raise ValueError("`episode_anchor` must be `partner_ipu` for the active behavioural hazard pipeline.")
        if self.unmatched_surprisal_strategy not in {"drop", "zero", "keep_nan"}:
            raise ValueError("`unmatched_surprisal_strategy` must be one of drop, zero, keep_nan.")
        if self.token_availability not in {"onset", "offset"}:
            raise ValueError("`token_availability` must be one of onset, offset.")
        if self.expected_info_group not in {"partner_ipu_class", "partner_role", "global"}:
            raise ValueError("`expected_info_group` must be one of partner_ipu_class, partner_role, global.")
        if self.partner_offset_fpp_tolerance_s < 0.0:
            raise ValueError("`partner_offset_fpp_tolerance_s` must be non-negative.")
        if self.overlapping_episode_strategy not in {"exclude", "truncate", "keep"}:
            raise ValueError("`overlapping_episode_strategy` must be one of exclude, truncate, keep.")
        lower, upper = self.clip_range
        if lower >= upper:
            raise ValueError("`clip_range` must be an increasing interval.")
        if not self.target_fpp_label_prefix:
            raise ValueError("`target_fpp_label_prefix` must be non-empty.")
        if not self.lag_grid_ms:
            raise ValueError("`lag_grid_ms` must contain at least one lag.")
        if any(int(lag_ms) < 0 for lag_ms in self.lag_grid_ms):
            raise ValueError("`lag_grid_ms` must contain only non-negative integers.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events_path"] = str(self.events_path)
        payload["surprisal_paths"] = [str(path) for path in self.surprisal_paths]
        payload["out_dir"] = str(self.out_dir)
        return payload


@dataclass(frozen=True, slots=True)
class SchemaNormalizationResult:
    table: pd.DataFrame
    used_columns: dict[str, str]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class EpisodeBuildResult:
    episodes: pd.DataFrame
    candidate_episodes: pd.DataFrame
    excluded_episodes: pd.DataFrame
    validation_qc: dict[str, object]
    warnings: list[str]
    used_partner_anchor: str
    partner_ipu_table: pd.DataFrame
    event_rows_debug: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RiskSetResult:
    riskset_table: pd.DataFrame
    episode_summary: pd.DataFrame
    warnings: list[str]
    event_qc: dict[str, object]


def resolve_surprisal_paths(path_or_glob: str | Path) -> tuple[Path, ...]:
    path_text = str(path_or_glob)
    candidate_path = Path(path_text)
    if candidate_path.exists():
        if candidate_path.is_dir():
            return tuple(sorted(candidate_path.rglob("*.tsv")))
        return (candidate_path,)
    return tuple(sorted(Path(path) for path in glob.glob(path_text, recursive=True)))


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


def normalize_events_schema(events_table: pd.DataFrame) -> SchemaNormalizationResult:
    return _normalize_schema(
        table=events_table,
        aliases=EVENT_COLUMN_ALIASES,
        required={"dyad_id", "run", "participant_speaker", "fpp_onset"},
        table_label="events",
    )


def normalize_surprisal_schema(surprisal_table: pd.DataFrame) -> SchemaNormalizationResult:
    return _normalize_schema(
        table=surprisal_table,
        aliases=SURPRISAL_COLUMN_ALIASES,
        required={"dyad_id", "run", "speaker", "onset", "duration", "surprisal"},
        table_label="surprisal",
    )


def read_events_table(events_path: Path) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_csv(events_path)
    if table.empty:
        raise ValueError(f"Events table is empty: {events_path}")
    normalized = normalize_events_schema(table)
    return normalized.table, normalized.warnings


def read_surprisal_tables(
    surprisal_paths: tuple[Path, ...],
    *,
    unmatched_surprisal_strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not surprisal_paths:
        raise FileNotFoundError("No surprisal TSV files were found.")

    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    for path in surprisal_paths:
        table = pd.read_csv(path, sep="\t")
        normalized = normalize_surprisal_schema(table)
        warnings.extend(normalized.warnings)
        frames.append(normalized.table.assign(source_path=str(path)))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["onset"] = pd.to_numeric(combined["onset"], errors="coerce")
    combined["duration"] = pd.to_numeric(combined["duration"], errors="coerce")
    combined["offset"] = combined["onset"] + combined["duration"]
    if "surprisal" in combined.columns:
        combined["surprisal"] = pd.to_numeric(combined["surprisal"], errors="coerce")
    if "word_index" in combined.columns:
        combined["word_index"] = pd.to_numeric(combined["word_index"], errors="coerce")

    combined = combined.loc[
        combined["onset"].notna()
        & combined["duration"].notna()
        & (combined["duration"] >= 0.0)
        & combined["speaker"].astype(str).isin({"A", "B"})
    ].copy()

    if "alignment_status" in combined.columns:
        combined["alignment_status"] = combined["alignment_status"].astype(str)
        if unmatched_surprisal_strategy == "drop":
            combined = combined.loc[combined["alignment_status"] == "ok"].copy()
        elif unmatched_surprisal_strategy == "zero":
            unmatched_mask = combined["alignment_status"] != "ok"
            combined.loc[unmatched_mask & combined["surprisal"].isna(), "surprisal"] = 0.0

    if unmatched_surprisal_strategy != "keep_nan":
        combined = combined.loc[combined["surprisal"].notna() & combined["surprisal"].map(pd.notna)].copy()
    combined = combined.sort_values(
        [column for column in ["dyad_id", "run", "speaker", "onset", "offset", "word_index"] if column in combined],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined, warnings


def validate_participant_speaker_id(
    data: pd.DataFrame,
    *,
    dyad_col: str = "dyad_id",
    speaker_col: str = "speaker",
    output_col: str = "participant_speaker_id",
) -> dict[str, Any]:
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


def _normalize_anchor_type(anchor: str) -> str:
    normalized = str(anchor).strip().lower()
    if normalized not in EXPECTED_ANCHOR_TYPES:
        raise ValueError(f"Unexpected anchor_type {anchor!r}; expected one of: fpp, spp.")
    return normalized


def project_behavior_final_events(events: pd.DataFrame, *, anchor: str) -> pd.DataFrame:
    normalized_anchor = _normalize_anchor_type(anchor)
    working = events.copy()
    if "dyad_id" not in working.columns and "recording_id" in working.columns:
        working["dyad_id"] = working["recording_id"]
    if "speaker_fpp" not in working.columns and "participant_speaker" in working.columns:
        working["speaker_fpp"] = working["participant_speaker"]
    if "speaker_spp" not in working.columns and "partner_speaker" in working.columns:
        working["speaker_spp"] = working["partner_speaker"]
    required = {
        "dyad_id",
        "run",
        "speaker_fpp",
        "speaker_spp",
        "fpp_label",
        "spp_label",
        "fpp_onset",
        "fpp_offset",
        "spp_onset",
        "spp_offset",
    }
    missing = sorted(required - set(working.columns))
    if missing:
        raise ValueError(f"Behavior-final events table is missing required columns for paired FPP/SPP projection: {missing}")

    if normalized_anchor == "fpp":
        participant_col = "speaker_fpp"
        partner_col = "speaker_spp"
        event_label_col = "fpp_label"
        event_onset_col = "fpp_onset"
        event_offset_col = "fpp_offset"
        partner_onset_col = "spp_onset"
        partner_offset_col = "spp_offset"
        partner_label_col = "spp_label"
    else:
        participant_col = "speaker_spp"
        partner_col = "speaker_fpp"
        event_label_col = "spp_label"
        event_onset_col = "spp_onset"
        event_offset_col = "spp_offset"
        partner_onset_col = "fpp_onset"
        partner_offset_col = "fpp_offset"
        partner_label_col = "fpp_label"

    projected = pd.DataFrame(
        {
            "dyad_id": working["dyad_id"].astype(str),
            "run": working["run"].astype(str),
            "participant_speaker": working[participant_col],
            "partner_speaker": working[partner_col],
            "fpp_label": working[event_label_col],
            "fpp_onset": pd.to_numeric(working[event_onset_col], errors="coerce"),
            "fpp_offset": pd.to_numeric(working[event_offset_col], errors="coerce"),
            "spp_label": working[partner_label_col],
            "spp_onset": pd.to_numeric(working[partner_onset_col], errors="coerce"),
            "spp_offset": pd.to_numeric(working[partner_offset_col], errors="coerce"),
            "source_anchor_type": normalized_anchor,
            "source_event_id": np.arange(len(working), dtype=int),
        }
    )
    if "pair_id" in working.columns:
        projected["pair_id"] = working["pair_id"].astype(str)
    return projected


def normalize_final_riskset(table: pd.DataFrame, anchor: str) -> pd.DataFrame:
    out = table.copy()
    normalized_anchor = _normalize_anchor_type(anchor)
    out["anchor_type"] = normalized_anchor
    out["subject_id"] = out["participant_speaker"].astype(str)
    out["run_id"] = out["run"].astype(str)
    out = out.rename(
        columns={
            "event": "event_bin",
            "bin_start": "bin_start_s",
            "bin_end": "bin_end_s",
            "time_from_partner_onset": "time_from_partner_onset_s",
            "time_from_partner_offset": "time_from_partner_offset_s",
        }
    )
    if "prop_expected_cumulative_info" not in out.columns and "expected_cumulative_info" in out.columns:
        out["prop_expected_cumulative_info"] = out["expected_cumulative_info"]
    missing = [c for c in REQUIRED_FINAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required final riskset columns: {', '.join(missing)}")
    keep = list(dict.fromkeys(REQUIRED_FINAL_COLUMNS + [c for c in out.columns if c not in REQUIRED_FINAL_COLUMNS]))
    normalized = out.loc[:, keep].copy()
    normalized["event_bin"] = pd.to_numeric(normalized["event_bin"], errors="coerce").astype(int)
    normalized["anchor_type"] = normalized["anchor_type"].astype(str).str.strip().str.lower()
    return normalized


def infer_partner_speaker(participant_speaker: str) -> str:
    speaker = _normalize_speaker_label(participant_speaker)
    if speaker == "A":
        return "B"
    if speaker == "B":
        return "A"
    raise ValueError(f"Expected participant speaker A or B, found {participant_speaker!r}.")


def infer_participant_speaker(partner_speaker: str) -> str:
    return infer_partner_speaker(partner_speaker)


def extract_fpp_events(
    events_table: pd.DataFrame,
    config: BehaviourHazardConfig,
    *,
    token_speakers: set[str] | None = None,
) -> pd.DataFrame:
    working = events_table.copy()
    working["dyad_id"] = working["dyad_id"].map(_normalize_dyad_id)
    working["run"] = working["run"].map(_normalize_run_label)

    speaker_column = _resolve_fpp_speaker_column(working)
    working["fpp_speaker"] = working[speaker_column].map(
        lambda value: _normalize_speaker_label(
            value,
            event_speakers=set(str(item) for item in working[speaker_column].dropna().unique()),
            token_speakers=token_speakers,
            dyad_run_example=_first_dyad_run_example(working),
        )
    )
    working["fpp_onset"] = pd.to_numeric(working["fpp_onset"], errors="coerce")
    working["fpp_offset"] = pd.to_numeric(working.get("fpp_offset"), errors="coerce")
    label_column = "fpp_label" if "fpp_label" in working.columns else None
    if label_column is not None:
        working["fpp_label"] = working[label_column].astype(str)
        working = working.loc[working["fpp_label"].str.startswith(config.target_fpp_label_prefix)].copy()
    else:
        working["fpp_label"] = ""

    keep_columns = ["dyad_id", "run", "fpp_speaker", "fpp_onset", "fpp_offset", "fpp_label"]
    for optional in ("source_event_id", "pair_id", "source_anchor_type"):
        if optional in working.columns:
            keep_columns.append(optional)
    extracted = working.loc[working["fpp_onset"].notna(), keep_columns].copy()
    if "source_event_id" in extracted.columns:
        extracted["source_event_id"] = pd.to_numeric(extracted["source_event_id"], errors="coerce")
        missing_ids = extracted["source_event_id"].isna()
        if missing_ids.any():
            start_id = 0
            present_ids = extracted.loc[~missing_ids, "source_event_id"]
            if not present_ids.empty:
                start_id = int(present_ids.max()) + 1
            extracted.loc[missing_ids, "source_event_id"] = np.arange(start_id, start_id + int(missing_ids.sum()), dtype=int)
        extracted["source_event_id"] = extracted["source_event_id"].astype(int)
    else:
        extracted["source_event_id"] = np.arange(len(extracted), dtype=int)
    extracted = extracted.sort_values(["dyad_id", "run", "fpp_speaker", "fpp_onset"], kind="mergesort").reset_index(drop=True)
    return extracted


def build_partner_ipus_from_tokens(
    surprisal_table: pd.DataFrame,
    *,
    gap_threshold_s: float,
) -> pd.DataFrame:
    working = surprisal_table.copy()
    working["dyad_id"] = working["dyad_id"].map(_normalize_dyad_id)
    working["run"] = working["run"].map(_normalize_run_label)
    working["speaker"] = working["speaker"].map(_normalize_speaker_label)
    working["onset"] = pd.to_numeric(working["onset"], errors="coerce")
    working["offset"] = pd.to_numeric(working["offset"], errors="coerce")
    working = working.loc[
        working["speaker"].isin(SUPPORTED_SPEAKERS)
        & working["onset"].notna()
        & working["offset"].notna()
        & (working["offset"] > working["onset"])
    ].copy()

    rows: list[dict[str, object]] = []
    group_columns = ["dyad_id", "run", "speaker"]
    for group_key, speaker_tokens in working.groupby(group_columns, sort=False):
        sorted_tokens = speaker_tokens.sort_values(["onset", "offset"], kind="mergesort").reset_index(drop=True)
        rows.extend(
            _build_ipu_rows_from_tokens(
                sorted_tokens=sorted_tokens,
                group_key=(str(group_key[0]), str(group_key[1]), str(group_key[2])),
                gap_threshold_s=gap_threshold_s,
            )
        )

    ipu_table = pd.DataFrame(rows)
    if ipu_table.empty:
        return ipu_table
    ipu_table["partner_ipu_duration"] = ipu_table["partner_ipu_offset"] - ipu_table["partner_ipu_onset"]
    ipu_table["next_partner_ipu_onset"] = (
        ipu_table.sort_values(["dyad_id", "run", "speaker", "partner_ipu_onset"], kind="mergesort")
        .groupby(["dyad_id", "run", "speaker"], sort=False)["partner_ipu_onset"]
        .shift(-1)
    )
    ipu_table["anchor_source"] = "partner_ipu_tokens"
    return ipu_table.reset_index(drop=True)


def find_first_fpp_in_episode_window(
    fpp_events: pd.DataFrame,
    *,
    dyad_id: str,
    run: str,
    participant_speaker: str,
    episode_start: float,
    episode_end: float,
) -> pd.Series | None:
    matches = fpp_events.loc[
        (fpp_events["dyad_id"] == dyad_id)
        & (fpp_events["run"] == run)
        & (fpp_events["fpp_speaker"] == participant_speaker)
        & (fpp_events["fpp_onset"] >= episode_start)
        & (fpp_events["fpp_onset"] < episode_end)
    ].sort_values(["fpp_onset", "source_event_id"], kind="mergesort")
    if matches.empty:
        return None
    return matches.iloc[0]


def build_partner_ipu_anchored_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> EpisodeBuildResult:
    warnings: list[str] = []
    partner_ipu_table = build_partner_ipus_from_tokens(
        surprisal_table,
        gap_threshold_s=config.ipu_gap_threshold_s,
    )
    if partner_ipu_table.empty:
        raise ValueError("No partner IPUs could be constructed from the surprisal token table.")

    token_speakers = set(partner_ipu_table["speaker"].astype(str).unique())
    fpp_events = extract_fpp_events(events_table, config, token_speakers=token_speakers)
    episodes: list[dict[str, object]] = []
    assigned_event_ids: list[int] = []
    event_rows_debug: list[dict[str, object]] = []
    ipu_records = list(partner_ipu_table.to_dict("records"))
    for ipu_row in progress_iterable(
        ipu_records,
        total=len(ipu_records),
        description="Partner-IPU episodes",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        dyad_id = str(ipu_row["dyad_id"])
        run = str(ipu_row["run"])
        partner_speaker = str(ipu_row["speaker"])
        participant_speaker = infer_participant_speaker(partner_speaker)
        episode_start = float(ipu_row["partner_ipu_onset"])
        next_partner_ipu_onset = pd.to_numeric(pd.Series([ipu_row.get("next_partner_ipu_onset")]), errors="coerce").iloc[0]
        window_end, censor_reason = _compute_episode_window_end(
            episode_start=episode_start,
            next_partner_ipu_onset=next_partner_ipu_onset,
            run_end=None,
            config=config,
        )
        event_row = find_first_fpp_in_episode_window(
            fpp_events,
            dyad_id=dyad_id,
            run=run,
            participant_speaker=participant_speaker,
            episode_start=episode_start,
            episode_end=window_end,
        )
        has_event = event_row is not None
        own_fpp_onset = float(event_row["fpp_onset"]) if event_row is not None else np.nan
        own_fpp_offset = float(event_row["fpp_offset"]) if event_row is not None and pd.notna(event_row["fpp_offset"]) else np.nan
        own_fpp_label = str(event_row["fpp_label"]) if event_row is not None else ""
        censor_time = own_fpp_onset if has_event else float(window_end)
        event_latency_from_partner_onset = own_fpp_onset - episode_start if has_event else np.nan
        partner_ipu_offset = float(ipu_row["partner_ipu_offset"])
        event_latency_from_partner_offset = own_fpp_onset - partner_ipu_offset if has_event else np.nan
        if has_event:
            event_phase = "during_partner_ipu" if own_fpp_onset < partner_ipu_offset else "post_partner_ipu"
            assigned_event_ids.append(int(event_row["source_event_id"]))
        else:
            event_phase = "censored"
        episode = {
            "episode_id": f"{dyad_id}|run-{run}|{partner_speaker}|ipu-{str(ipu_row['partner_ipu_id']).split('|')[-1]}",
            "dyad_id": dyad_id,
            "run": run,
            "partner_speaker": partner_speaker,
            "participant_speaker": participant_speaker,
            "partner_ipu_id": str(ipu_row["partner_ipu_id"]),
            "partner_ipu_onset": episode_start,
            "partner_ipu_offset": partner_ipu_offset,
            "partner_ipu_duration": float(ipu_row["partner_ipu_duration"]),
            "next_partner_ipu_onset": float(next_partner_ipu_onset) if np.isfinite(next_partner_ipu_onset) else np.nan,
            "episode_start": episode_start,
            "episode_end": float(window_end),
            "censor_time": float(censor_time),
            "episode_has_event": bool(has_event),
            "source_event_id": int(event_row["source_event_id"]) if event_row is not None else np.nan,
            "own_fpp_onset": own_fpp_onset,
            "own_fpp_offset": own_fpp_offset,
            "own_fpp_label": own_fpp_label,
            "event_latency_from_partner_onset_s": event_latency_from_partner_onset,
            "event_latency_from_partner_offset_s": event_latency_from_partner_offset,
            "event_phase": event_phase,
            "episode_duration_s": float(censor_time - episode_start),
            "censor_reason": "event" if has_event else censor_reason,
            "anchor_source": str(ipu_row["anchor_source"]),
            "partner_ipu_class": str(ipu_row.get("partner_ipu_class", DEFAULT_PARTNER_IPU_CLASS)),
            "partner_role": "partner",
            "episode_kind": "event_positive" if has_event else "censored",
            "event_observed": int(has_event),
        }
        episodes.append(episode)
        event_rows_debug.append(
            {
                "partner_ipu_id": episode["partner_ipu_id"],
                "episode_id": episode["episode_id"],
                "dyad_id": dyad_id,
                "run": run,
                "partner_speaker": partner_speaker,
                "participant_speaker": participant_speaker,
                "episode_start": episode_start,
                "episode_end": float(window_end),
                "assigned_source_event_id": int(event_row["source_event_id"]) if event_row is not None else np.nan,
                "assigned_fpp_onset": own_fpp_onset,
                "assigned_fpp_label": own_fpp_label,
                "episode_has_event": bool(has_event),
                "censor_reason": "event" if has_event else censor_reason,
            }
        )

    episodes_table = pd.DataFrame(episodes).sort_values(["dyad_id", "run", "partner_ipu_onset"], kind="mergesort").reset_index(drop=True)
    if not config.include_censored:
        episodes_table = episodes_table.loc[episodes_table["episode_has_event"]].reset_index(drop=True)
    excluded = pd.DataFrame(columns=["episode_id", "invalid_reason"])
    event_rows_debug_table = pd.DataFrame(event_rows_debug)
    anchor_qc = compute_partner_ipu_anchor_qc(
        partner_ipu_table=partner_ipu_table,
        episodes_table=episodes_table,
        fpp_events=fpp_events,
        assigned_event_ids=assigned_event_ids,
        config=config,
    )
    validate_partner_ipu_episodes(
        episodes_table,
        fpp_events=fpp_events,
        assigned_event_ids=assigned_event_ids,
    )
    if anchor_qc["n_fpp_events_unassigned"] > 0:
        warnings.append(
            f"{anchor_qc['n_fpp_events_unassigned']} FPP events were not assigned to any partner-IPU episode."
        )
    return EpisodeBuildResult(
        episodes=episodes_table,
        candidate_episodes=episodes_table.copy(),
        excluded_episodes=excluded,
        validation_qc=anchor_qc,
        warnings=warnings,
        used_partner_anchor="partner_ipu_tokens",
        partner_ipu_table=partner_ipu_table,
        event_rows_debug=event_rows_debug_table,
    )


def build_event_positive_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> EpisodeBuildResult:
    return build_partner_ipu_anchored_episodes(
        events_table=events_table,
        surprisal_table=surprisal_table,
        config=config,
    )


def build_censored_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    positive_episodes: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    return pd.DataFrame()


def build_discrete_time_riskset(
    episodes_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> RiskSetResult:
    warnings: list[str] = []
    rows: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    episode_records = list(episodes_table.to_dict("records"))
    for episode_record in progress_iterable(
        episode_records,
        total=len(episode_records),
        description="Risk-set bins",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        episode = pd.Series(episode_record)
        episode_rows = _build_episode_bins(episode=episode, config=config)
        if episode_rows.empty:
            warnings.append(f"Skipped episode {episode['episode_id']} because no valid bins were constructed.")
            continue
        rows.extend(episode_rows.to_dict("records"))
        episode_summaries.append(
            {
                "episode_id": episode["episode_id"],
                "dyad_id": episode["dyad_id"],
                "run": episode["run"],
                "participant_speaker_id": episode.get("participant_speaker_id", f"{episode['dyad_id']}_{episode['participant_speaker']}"),
                "participant_speaker": episode["participant_speaker"],
                "partner_speaker": episode["partner_speaker"],
                "partner_ipu_class": str(episode.get("partner_ipu_class", "unknown")),
                "partner_role": str(episode.get("partner_role", "partner")),
                "partner_ipu_id": episode.get("partner_ipu_id", f"{episode['episode_id']}|anchor"),
                "partner_ipu_onset": episode["partner_ipu_onset"],
                "partner_ipu_offset": episode["partner_ipu_offset"],
                "partner_ipu_duration": episode.get(
                    "partner_ipu_duration",
                    float(episode["partner_ipu_offset"]) - float(episode["partner_ipu_onset"]),
                ),
                "episode_start": episode.get("episode_start", episode["partner_ipu_onset"]),
                "episode_end": episode.get("episode_end", episode["censor_time"]),
                "censor_time": episode["censor_time"],
                "episode_has_event": bool(
                    episode.get("episode_has_event", episode.get("event_observed", pd.notna(episode.get("own_fpp_onset"))))
                ),
                "own_fpp_onset": episode["own_fpp_onset"],
                "own_fpp_label": episode.get("own_fpp_label", ""),
                "event_phase": episode.get("event_phase", "censored"),
                "event_latency_from_partner_onset_s": episode.get("event_latency_from_partner_onset_s", np.nan),
                "event_latency_from_partner_offset_s": episode.get("event_latency_from_partner_offset_s", np.nan),
                "latency_from_partner_offset_s": episode.get(
                    "latency_from_partner_offset_s",
                    episode.get("event_latency_from_partner_offset_s", np.nan),
                ),
                "censor_reason": episode.get("censor_reason", ""),
                "anchor_source": episode.get("anchor_source", ""),
                "n_bins": int(len(episode_rows)),
                "n_event_rows": int(episode_rows["event"].sum()),
            }
        )

    riskset_table = pd.DataFrame(rows)
    if not riskset_table.empty:
        riskset_table = ensure_participant_speaker_id(
            riskset_table,
            dyad_col="dyad_id",
            speaker_col="participant_speaker",
            output_col="participant_speaker_id",
            overwrite=True,
        )
    event_qc = validate_riskset(riskset_table, episodes_table)
    return RiskSetResult(
        riskset_table=riskset_table,
        episode_summary=ensure_participant_speaker_id(
            pd.DataFrame(episode_summaries),
            dyad_col="dyad_id",
            speaker_col="participant_speaker",
            output_col="participant_speaker_id",
            overwrite=True,
        ),
        warnings=warnings,
        event_qc=event_qc,
    )


def validate_riskset(riskset_table: pd.DataFrame, episodes_table: pd.DataFrame) -> dict[str, object]:
    if riskset_table.empty:
        raise ValueError("Risk-set table is empty.")
    required_columns = {
        "episode_id",
        "bin_index",
        "bin_start",
        "bin_end",
        "time_from_partner_onset",
        "event",
        "episode_has_event",
    }
    missing = sorted(required_columns - set(riskset_table.columns))
    if missing:
        raise ValueError(f"Risk-set table is missing required columns: {missing}")
    event_values = pd.to_numeric(riskset_table["event"], errors="raise")
    if not event_values.isin([0, 1]).all():
        raise ValueError("Risk-set event column must contain only 0/1 values.")

    event_counts = riskset_table.groupby("episode_id")["event"].sum().astype(int)
    episode_flags = (
        episodes_table.assign(
            episode_has_event=episodes_table.get(
                "episode_has_event",
                episodes_table.get("event_observed", episodes_table["own_fpp_onset"].notna()),
            )
        )
        .set_index("episode_id")["episode_has_event"]
        .astype(bool)
    )
    expected_event_counts = episode_flags.map(lambda value: 1 if value else 0).astype(int)
    aligned = event_counts.reindex(expected_event_counts.index).fillna(0).astype(int)
    positive_failures = aligned.loc[(expected_event_counts == 1) & (aligned != 1)]
    censored_failures = aligned.loc[(expected_event_counts == 0) & (aligned != 0)]
    if not positive_failures.empty:
        raise ValueError(
            "Each event-positive episode must have exactly one event row. "
            f"Failed episode ids: {positive_failures.index.tolist()[:5]}"
        )
    if not censored_failures.empty:
        raise ValueError(
            "Each censored episode must have zero event rows. "
            f"Failed episode ids: {censored_failures.index.tolist()[:5]}"
        )
    return {
        "n_episodes_total": int(len(expected_event_counts)),
        "n_positive_episodes": int((expected_event_counts == 1).sum()),
        "n_censored_episodes": int((expected_event_counts == 0).sum()),
        "positive_episodes_have_exactly_one_event_row": bool(positive_failures.empty),
        "censored_episodes_have_zero_event_rows": bool(censored_failures.empty),
        "event_column_is_int_0_1": True,
        "identity_validation": validate_participant_speaker_id(
            riskset_table,
            dyad_col="dyad_id",
            speaker_col="participant_speaker",
            output_col="participant_speaker_id",
        ),
    }


def add_information_features_to_riskset(
    *,
    riskset_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    expected_total_info_by_group = compute_expected_total_information(
        surprisal_table=surprisal_table,
        episodes_table=episodes_table,
        config=config,
    )
    frames: list[pd.DataFrame] = []
    episodes_indexed = episodes_table.set_index("episode_id", drop=False)
    grouped_episodes = list(riskset_table.groupby("episode_id", sort=False))
    for episode_id, episode_rows in progress_iterable(
        grouped_episodes,
        total=len(grouped_episodes),
        description="Episode features",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        frames.append(
            compute_information_features_for_episode(
                episode_rows=episode_rows,
                episode=episodes_indexed.loc[episode_id],
                surprisal_table=surprisal_table,
                expected_total_info_by_group=expected_total_info_by_group,
                config=config,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False), expected_total_info_by_group


def compute_information_features_for_episode(
    *,
    episode_rows: pd.DataFrame,
    episode: pd.Series,
    surprisal_table: pd.DataFrame,
    expected_total_info_by_group: dict[str, float],
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    speaker_tokens = extract_partner_ipu_tokens(
        episode=episode,
        surprisal_table=surprisal_table,
        config=config,
    )

    actual_total_info = float(speaker_tokens["surprisal"].sum(min_count=1)) if not speaker_tokens.empty else np.nan
    n_tokens_total = int(len(speaker_tokens))
    alignment_ok_fraction = _compute_alignment_ok_fraction(speaker_tokens)
    group_value = _resolve_expected_info_group_value(episode, config.expected_info_group)
    expected_total_info = expected_total_info_by_group.get(group_value, expected_total_info_by_group.get("global", np.nan))

    rows = episode_rows.copy()
    cumulative_values: list[float] = []
    information_rate_values: list[float] = []
    observed_token_counts: list[int] = []
    for _, bin_row in rows.iterrows():
        bin_end = float(bin_row["bin_end"])
        observed_tokens = speaker_tokens.loc[speaker_tokens["availability_time"] < bin_end + EPSILON]
        causal_window_start = bin_end - config.information_rate_window_s
        window_tokens = speaker_tokens.loc[
            (speaker_tokens["availability_time"] >= causal_window_start - EPSILON)
            & (speaker_tokens["availability_time"] <= bin_end + EPSILON)
        ]
        cumulative_values.append(float(observed_tokens["surprisal"].sum(min_count=1)) if not observed_tokens.empty else 0.0)
        information_rate_values.append(
            (
                float(window_tokens["surprisal"].sum(min_count=1)) / config.information_rate_window_s
                if not window_tokens.empty
                else 0.0
            )
        )
        observed_token_counts.append(int(len(observed_tokens)))

    rows["cumulative_info"] = cumulative_values
    rows["information_rate"] = information_rate_values
    rows["actual_total_info"] = actual_total_info
    rows["expected_total_info"] = expected_total_info
    rows["n_tokens_observed"] = observed_token_counts
    rows["n_tokens_total"] = n_tokens_total
    rows["alignment_ok_fraction"] = alignment_ok_fraction
    rows["prop_expected_cumulative_info"] = _safe_ratio(rows["cumulative_info"], expected_total_info)
    if config.clip_proportions:
        low, high = config.clip_range
        rows["prop_expected_cumulative_info"] = rows["prop_expected_cumulative_info"].clip(low, high)
    rows["expected_info_group_value"] = group_value
    rows["feature_missing_actual_total"] = bool(not np.isfinite(actual_total_info) or actual_total_info <= 0.0)
    rows["feature_missing_expected_total"] = bool(not np.isfinite(expected_total_info) or expected_total_info <= 0.0)
    return rows


def extract_partner_ipu_tokens(
    *,
    episode: pd.Series,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    speaker_tokens = surprisal_table.loc[
        (surprisal_table["dyad_id"].astype(str) == str(episode["dyad_id"]))
        & (surprisal_table["run"].astype(str) == str(episode["run"]))
        & (surprisal_table["speaker"].astype(str) == str(episode["partner_speaker"]))
    ].copy()
    speaker_tokens = speaker_tokens.loc[
        (speaker_tokens["onset"] >= float(episode["partner_ipu_onset"]) - EPSILON)
        & (speaker_tokens["offset"] <= float(episode["partner_ipu_offset"]) + EPSILON)
    ].copy()

    if config.token_availability == "onset":
        speaker_tokens["availability_time"] = speaker_tokens["onset"]
    else:
        speaker_tokens["availability_time"] = speaker_tokens["offset"]
    speaker_tokens["availability_time_relative_to_ipu_onset"] = (
        pd.to_numeric(speaker_tokens["availability_time"], errors="coerce") - float(episode["partner_ipu_onset"])
    )
    return speaker_tokens


def compute_expected_total_information(
    *,
    surprisal_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> dict[str, float]:
    episodes_with_defaults = episodes_table.copy()
    if "partner_ipu_class" not in episodes_with_defaults.columns:
        episodes_with_defaults["partner_ipu_class"] = "unknown"
    if "partner_role" not in episodes_with_defaults.columns:
        episodes_with_defaults["partner_role"] = "partner"

    episode_ipus = episodes_with_defaults.loc[
        :,
        ["dyad_id", "run", "partner_speaker", "partner_ipu_onset", "partner_ipu_offset", "partner_ipu_class", "partner_role"],
    ].drop_duplicates()
    if episode_ipus.empty:
        return {"global": np.nan}

    ipu_totals: list[dict[str, object]] = []
    ipu_records = list(episode_ipus.to_dict("records"))
    for ipu_row in progress_iterable(
        ipu_records,
        total=len(ipu_records),
        description="Expected info",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        tokens = surprisal_table.loc[
            (surprisal_table["dyad_id"].astype(str) == str(ipu_row["dyad_id"]))
            & (surprisal_table["run"].astype(str) == str(ipu_row["run"]))
            & (surprisal_table["speaker"].astype(str) == str(ipu_row["partner_speaker"]))
            & (surprisal_table["onset"] >= float(ipu_row["partner_ipu_onset"]) - EPSILON)
            & (surprisal_table["offset"] <= float(ipu_row["partner_ipu_offset"]) + EPSILON)
        ]
        total_info = float(tokens["surprisal"].sum(min_count=1)) if not tokens.empty else np.nan
        ipu_totals.append(
            {
                "partner_ipu_class": str(ipu_row.get("partner_ipu_class", "unknown")),
                "partner_role": str(ipu_row.get("partner_role", "partner")),
                "global": "global",
                "actual_total_info": total_info,
            }
        )
    ipu_totals_table = pd.DataFrame(ipu_totals)
    result = {"global": float(ipu_totals_table["actual_total_info"].mean())}
    if config.expected_info_group == "global":
        return result

    group_column = config.expected_info_group
    if group_column not in episodes_with_defaults.columns:
        return result
    grouped = ipu_totals_table.groupby(group_column, dropna=False)["actual_total_info"].mean()
    for group_value, mean_value in grouped.items():
        result[str(group_value)] = float(mean_value)
    return result


def _build_episode_bins(
    *,
    episode: pd.Series,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    anchor = float(episode["partner_ipu_onset"])
    censor_time = float(episode["censor_time"])
    partner_ipu_offset = float(episode["partner_ipu_offset"])
    episode_has_event = bool(
        episode.get("episode_has_event", episode.get("event_observed", pd.notna(episode.get("own_fpp_onset"))))
    )
    partner_ipu_duration = float(episode.get("partner_ipu_duration", partner_ipu_offset - anchor))
    relative_stop = censor_time - anchor
    if relative_stop < 0.0:
        raise ValueError(f"Episode {episode['episode_id']} has censor_time before partner_ipu_onset.")

    last_bin_index = int(np.floor((relative_stop + FLOAT_TOLERANCE) / config.bin_size_s))
    if last_bin_index < 0:
        return pd.DataFrame()
    event_bin_index = None
    if episode_has_event:
        event_bin_index = int(np.floor(((float(episode["own_fpp_onset"]) - anchor) + FLOAT_TOLERANCE) / config.bin_size_s))

    rows: list[dict[str, object]] = []
    for bin_index in range(last_bin_index + 1):
        bin_start = round(anchor + bin_index * config.bin_size_s, 10)
        bin_end = round(bin_start + config.bin_size_s, 10)
        rows.append(
            {
                "dyad_id": str(episode["dyad_id"]),
                "run": str(episode["run"]),
                "participant_speaker_id": str(
                    episode.get("participant_speaker_id", f"{episode['dyad_id']}_{episode['participant_speaker']}")
                ),
                "participant_speaker": str(episode["participant_speaker"]),
                "partner_speaker": str(episode["partner_speaker"]),
                "partner_ipu_id": str(episode.get("partner_ipu_id", f"{episode['episode_id']}|anchor")),
                "episode_id": str(episode["episode_id"]),
                "episode_kind": str(episode.get("episode_kind", "event_positive" if episode_has_event else "censored")),
                "bin_index": int(bin_index),
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "time_from_partner_onset": float(bin_index * config.bin_size_s),
                "partner_ipu_onset": anchor,
                "partner_ipu_offset": partner_ipu_offset,
                "partner_ipu_duration": partner_ipu_duration,
                "partner_ipu_complete": bool(bin_end >= partner_ipu_offset),
                "time_from_partner_offset": float(bin_end - partner_ipu_offset),
                "time_since_partner_offset_positive": float(max(0.0, bin_end - partner_ipu_offset)),
                "phase": "during_partner_ipu" if bin_end < partner_ipu_offset else "post_partner_ipu",
                "event": int(event_bin_index is not None and bin_index == event_bin_index),
                "episode_has_event": int(episode_has_event),
                "source_event_id": int(episode["source_event_id"]) if pd.notna(episode.get("source_event_id")) else np.nan,
                "own_fpp_onset": float(episode["own_fpp_onset"]) if pd.notna(episode["own_fpp_onset"]) else np.nan,
                "own_fpp_label": str(episode.get("own_fpp_label", "")),
                "event_phase": str(episode.get("event_phase", "censored")),
                "censor_time": float(censor_time),
                "episode_start": float(episode.get("episode_start", anchor)),
                "episode_end": float(episode.get("episode_end", censor_time)),
                "next_partner_ipu_onset": (
                    float(episode.get("next_partner_ipu_onset"))
                    if pd.notna(episode.get("next_partner_ipu_onset"))
                    else np.nan
                ),
                "censor_reason": str(episode.get("censor_reason", "")),
                "anchor_source": str(episode.get("anchor_source", "")),
                "partner_ipu_class": str(episode.get("partner_ipu_class", "unknown")),
                "partner_role": str(episode.get("partner_role", "partner")),
            }
        )
    return pd.DataFrame(rows)


def compute_partner_ipu_anchor_qc(
    *,
    partner_ipu_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    fpp_events: pd.DataFrame,
    assigned_event_ids: list[int],
    config: BehaviourHazardConfig,
) -> dict[str, object]:
    event_positive = episodes_table.loc[episodes_table["episode_has_event"]].copy()
    latencies_onset = pd.to_numeric(event_positive["event_latency_from_partner_onset_s"], errors="coerce")
    latencies_offset = pd.to_numeric(event_positive["event_latency_from_partner_offset_s"], errors="coerce")
    assigned_unique = len(set(assigned_event_ids))
    events_by_pair = {
        f"{str(index[0])}|run-{str(index[1])}": int(value)
        for index, value in fpp_events.groupby(["dyad_id", "run"], sort=False).size().items()
    }
    tokens_by_pair = {
        f"{str(index[0])}|run-{str(index[1])}": int(value)
        for index, value in partner_ipu_table.groupby(["dyad_id", "run"], sort=False).size().items()
    }
    event_pairs = set(events_by_pair)
    token_pairs = set(tokens_by_pair)
    return {
        "n_partner_ipus": int(len(partner_ipu_table)),
        "n_episodes": int(len(episodes_table)),
        "n_event_positive_episodes": int(event_positive.shape[0]),
        "n_censored_episodes": int((~episodes_table["episode_has_event"]).sum()),
        "proportion_event_positive": _safe_proportion(int(event_positive.shape[0]), int(len(episodes_table))),
        "n_fpp_events_total": int(len(fpp_events)),
        "n_fpp_events_assigned_to_episode": int(assigned_unique),
        "n_fpp_events_unassigned": int(len(fpp_events) - assigned_unique),
        "proportion_fpp_events_assigned": _safe_proportion(int(assigned_unique), int(len(fpp_events))),
        "n_episodes_censored_by_next_partner_ipu": int((episodes_table["censor_reason"] == "next_partner_ipu").sum()),
        "n_episodes_censored_by_max_followup": int((episodes_table["censor_reason"] == "max_followup").sum()),
        "n_episodes_censored_by_run_end": int((episodes_table["censor_reason"] == "run_end").sum()),
        "median_partner_ipu_duration_s": _maybe_float(partner_ipu_table["partner_ipu_duration"].median()),
        "p95_partner_ipu_duration_s": _maybe_float(partner_ipu_table["partner_ipu_duration"].quantile(0.95)),
        "max_partner_ipu_duration_s": _maybe_float(partner_ipu_table["partner_ipu_duration"].max()),
        "median_event_latency_from_partner_onset_s": _maybe_float(latencies_onset.median()),
        "p95_event_latency_from_partner_onset_s": _maybe_float(latencies_onset.quantile(0.95)),
        "median_event_latency_from_partner_offset_s": _maybe_float(latencies_offset.median()),
        "proportion_events_during_partner_ipu": _safe_proportion(
            int((event_positive["event_phase"] == "during_partner_ipu").sum()),
            int(event_positive.shape[0]),
        ),
        "proportion_events_post_partner_ipu": _safe_proportion(
            int((event_positive["event_phase"] == "post_partner_ipu").sum()),
            int(event_positive.shape[0]),
        ),
        "max_followup_s": float(config.max_followup_s),
        "ipu_gap_threshold_s": float(config.ipu_gap_threshold_s),
        "bin_size_s": float(config.bin_size_s),
        "n_events_by_dyad_run": events_by_pair,
        "n_tokens_by_dyad_run": tokens_by_pair,
        "dyad_run_pairs_in_events_not_tokens": sorted(event_pairs - token_pairs),
        "dyad_run_pairs_in_tokens_not_events": sorted(token_pairs - event_pairs),
    }


def validate_partner_ipu_episodes(
    episodes_table: pd.DataFrame,
    *,
    fpp_events: pd.DataFrame,
    assigned_event_ids: list[int],
) -> None:
    if episodes_table.empty:
        raise ValueError("No partner-IPU episodes were constructed.")
    if episodes_table["partner_ipu_onset"].isna().any() or episodes_table["partner_ipu_offset"].isna().any():
        raise ValueError("Every partner-IPU-anchored episode must include partner_ipu_onset and partner_ipu_offset.")
    if not (episodes_table["partner_ipu_onset"] < episodes_table["partner_ipu_offset"]).all():
        raise ValueError("Each partner-IPU-anchored episode must satisfy partner_ipu_onset < partner_ipu_offset.")
    if not np.allclose(
        pd.to_numeric(episodes_table["episode_start"], errors="coerce"),
        pd.to_numeric(episodes_table["partner_ipu_onset"], errors="coerce"),
    ):
        raise ValueError("Each partner-IPU-anchored episode must satisfy episode_start == partner_ipu_onset.")
    if not (pd.to_numeric(episodes_table["episode_end"], errors="coerce") > pd.to_numeric(episodes_table["episode_start"], errors="coerce")).all():
        raise ValueError("Each partner-IPU-anchored episode must satisfy episode_end > episode_start.")

    event_positive = episodes_table.loc[episodes_table["episode_has_event"]]
    if not event_positive.empty:
        if not (
            pd.to_numeric(event_positive["own_fpp_onset"], errors="coerce")
            >= pd.to_numeric(event_positive["episode_start"], errors="coerce")
        ).all():
            raise ValueError("Event-positive partner-IPU episodes must satisfy own_fpp_onset >= episode_start.")
        if not (
            pd.to_numeric(event_positive["own_fpp_onset"], errors="coerce")
            < pd.to_numeric(event_positive["episode_end"], errors="coerce")
        ).all():
            raise ValueError("Event-positive partner-IPU episodes must satisfy own_fpp_onset < episode_end.")
    if len(assigned_event_ids) != len(set(assigned_event_ids)):
        raise ValueError("No FPP event may be assigned to more than one partner-IPU episode.")
    missing_assigned = set(assigned_event_ids) - set(pd.to_numeric(fpp_events["source_event_id"], errors="coerce").astype(int))
    if missing_assigned:
        raise ValueError(f"Assigned event ids were not found in extracted FPP events: {sorted(missing_assigned)}")


def _resolve_expected_info_group_value(episode: pd.Series, group_name: str) -> str:
    if group_name == "global":
        return "global"
    value = episode.get(group_name, "unknown")
    if pd.isna(value):
        return "unknown"
    return str(value)


def _compute_alignment_ok_fraction(tokens: pd.DataFrame) -> float:
    if tokens.empty or "alignment_status" not in tokens.columns:
        return float("nan")
    ok = tokens["alignment_status"].astype(str).eq("ok")
    return float(ok.mean()) if len(ok) else float("nan")


def _safe_ratio(values: pd.Series, denominator: float) -> pd.Series:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return pd.to_numeric(values, errors="coerce") / float(denominator)


def _safe_proportion(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _maybe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if np.isfinite(numeric) else float("nan")


def _normalize_dyad_id(value: object) -> str:
    return str(value).strip()


def _normalize_run_label(value: object) -> str:
    run = str(value).strip()
    return run.removeprefix("run-")


def _normalize_speaker_label(
    value: object,
    *,
    event_speakers: set[str] | None = None,
    token_speakers: set[str] | None = None,
    dyad_run_example: str | None = None,
) -> str:
    speaker = str(value).strip()
    if speaker in SUPPORTED_SPEAKERS:
        return speaker
    if speaker.endswith("_A"):
        return "A"
    if speaker.endswith("_B"):
        return "B"
    if speaker.upper() in SUPPORTED_SPEAKERS:
        return speaker.upper()
    context = f" for {dyad_run_example}" if dyad_run_example else ""
    raise ValueError(
        f"Unsupported speaker label {speaker!r}{context}. "
        f"Observed event speakers={sorted(event_speakers or [])}, token speakers={sorted(token_speakers or [])}."
    )


def _resolve_fpp_speaker_column(table: pd.DataFrame) -> str:
    for column_name in ("participant_speaker", "speaker_fpp", "fpp_speaker"):
        if column_name in table.columns:
            return column_name
    available = ", ".join(sorted(str(column) for column in table.columns))
    raise ValueError(f"Could not determine the FPP speaker column. Available columns: {available}")


def _first_dyad_run_example(table: pd.DataFrame) -> str | None:
    if table.empty or "dyad_id" not in table.columns or "run" not in table.columns:
        return None
    row = table.iloc[0]
    return f"{row['dyad_id']}/run-{row['run']}"


def _build_ipu_rows_from_tokens(
    *,
    sorted_tokens: pd.DataFrame,
    group_key: tuple[str, str, str],
    gap_threshold_s: float,
) -> list[dict[str, object]]:
    dyad_id, run, speaker = group_key
    if sorted_tokens.empty:
        return []
    rows: list[dict[str, object]] = []
    current_onset = float(sorted_tokens.iloc[0]["onset"])
    current_offset = float(sorted_tokens.iloc[0]["offset"])
    token_count = 1
    ipu_index = 0
    for _, token in sorted_tokens.iloc[1:].iterrows():
        onset = float(token["onset"])
        offset = float(token["offset"])
        gap = onset - current_offset
        if gap <= gap_threshold_s:
            current_offset = max(current_offset, offset)
            token_count += 1
            continue
        rows.append(
            {
                "dyad_id": dyad_id,
                "run": run,
                "speaker": speaker,
                "partner_ipu_id": f"{dyad_id}|run-{run}|{speaker}|ipu-{ipu_index:05d}",
                "partner_ipu_onset": current_onset,
                "partner_ipu_offset": current_offset,
                "n_tokens": token_count,
            }
        )
        ipu_index += 1
        current_onset = onset
        current_offset = offset
        token_count = 1
    rows.append(
        {
            "dyad_id": dyad_id,
            "run": run,
            "speaker": speaker,
            "partner_ipu_id": f"{dyad_id}|run-{run}|{speaker}|ipu-{ipu_index:05d}",
            "partner_ipu_onset": current_onset,
            "partner_ipu_offset": current_offset,
            "n_tokens": token_count,
        }
    )
    return rows


def _compute_episode_window_end(
    *,
    episode_start: float,
    next_partner_ipu_onset: float,
    run_end: float | None,
    config: BehaviourHazardConfig,
) -> tuple[float, str]:
    candidate_end = episode_start + float(config.max_followup_s)
    censor_reason = "max_followup"
    if np.isfinite(next_partner_ipu_onset) and float(next_partner_ipu_onset) < candidate_end:
        candidate_end = float(next_partner_ipu_onset)
        censor_reason = "next_partner_ipu"
    if run_end is not None and float(run_end) < candidate_end:
        candidate_end = float(run_end)
        censor_reason = "run_end"
    return candidate_end, censor_reason
