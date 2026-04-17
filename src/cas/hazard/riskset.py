"""Risk-set construction for partner-onset discrete-time hazard analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cas.hazard.config import HazardAnalysisConfig

LOGGER = logging.getLogger(__name__)
DYAD_NUMBER_PATTERN = re.compile(r"dyad-(?P<number>\d+)$")
ACTION_TIER_SPEAKER_PATTERN = re.compile(r"action\s+(?P<speaker>[AB])$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RiskSetBuildResult:
    """Outputs from partner-onset risk-set construction."""

    hazard_table: pd.DataFrame
    aligned_entropy_table: pd.DataFrame
    warnings: list[str]
    subject_mapping_assumptions: list[str]
    n_subjects: int
    n_episodes: int
    n_positive_episodes: int
    n_censored_episodes: int


def build_person_period_riskset(
    *,
    config: HazardAnalysisConfig,
    events_table: pd.DataFrame,
    pairing_issues_table: pd.DataFrame | None,
    entropy_by_run: dict[tuple[str, str], pd.DataFrame],
    dyad_table: pd.DataFrame | None,
) -> RiskSetBuildResult:
    """Build the pooled person-period table using partner onset as the risk clock."""

    subject_mapping_assumptions, dyad_mapping = _build_dyad_mapping(dyad_table)
    positive_episodes = _build_positive_episode_table(
        events_table=events_table,
        config=config,
        dyad_mapping=dyad_mapping,
        entropy_by_run=entropy_by_run,
    )
    censored_episodes = _build_censored_episode_table(
        pairing_issues_table=pairing_issues_table,
        config=config,
        dyad_mapping=dyad_mapping,
        entropy_by_run=entropy_by_run,
    )
    episode_frames = [frame for frame in (positive_episodes, censored_episodes) if not frame.empty]
    if not episode_frames:
        episode_table = pd.DataFrame()
    elif len(episode_frames) == 1:
        episode_table = episode_frames[0].copy()
    else:
        episode_table = pd.concat(episode_frames, ignore_index=True, sort=False)
    if episode_table.empty:
        raise ValueError("No valid partner-onset episodes remained after filtering.")

    episode_table = episode_table.sort_values(
        ["subject_id", "run_id", "partner_onset_seconds", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    warnings: list[str] = []
    hazard_rows: list[dict[str, Any]] = []
    for episode_row in episode_table.to_dict("records"):
        run_key = (str(episode_row["subject_id"]), str(episode_row["run_id"]))
        run_entropy_frame = entropy_by_run.get(run_key)
        if run_entropy_frame is None:
            warnings.append(
                f"Skipped episode {episode_row['event_id']} because no entropy timeline was available for {run_key}."
            )
            continue
        episode_rows, episode_warning = _build_episode_rows(
            episode_row=episode_row,
            run_entropy_frame=run_entropy_frame,
            time_axis_config=config.time_axis,
        )
        if episode_warning is not None:
            warnings.append(episode_warning)
        if episode_rows:
            hazard_rows.extend(episode_rows)

    if not hazard_rows:
        raise ValueError("No hazard rows were available after aligning entropy to partner-onset episodes.")

    hazard_table = pd.DataFrame(hazard_rows).sort_values(
        ["subject_id", "event_id", "bin_index"], kind="mergesort"
    ).reset_index(drop=True)
    _add_entropy_zscore_column(
        hazard_table=hazard_table,
        zscore_within_subject=config.entropy.zscore_within_subject,
    )
    _validate_hazard_table(hazard_table)

    subject_episode_counts = hazard_table.groupby("subject_id")["event_id"].nunique().sort_values()
    sparse_subjects = subject_episode_counts.loc[subject_episode_counts < 2]
    if not sparse_subjects.empty:
        warnings.append(
            "Some subjects contribute fewer than two usable episodes: "
            + ", ".join(f"{subject} ({count})" for subject, count in sparse_subjects.items())
        )

    aligned_entropy_table = hazard_table.loc[
        :,
        [
            "subject_id",
            "event_id",
            "run_id",
            "dyad_id",
            "partner_onset_seconds",
            "target_onset_seconds",
            "tau_seconds",
            "predictor_time_seconds",
            "entropy",
            "entropy_z",
            "event",
            "censored_episode",
        ],
    ].copy()

    n_positive_episodes = int(hazard_table.groupby("event_id")["event"].max().sum())
    n_episodes = int(hazard_table["event_id"].nunique())
    n_censored_episodes = int(
        hazard_table.groupby("event_id")["censored_episode"].first().astype(int).sum()
    )
    LOGGER.info(
        "Constructed %d person-period rows across %d episodes, %d positive episodes, and %d censored episodes",
        len(hazard_table),
        n_episodes,
        n_positive_episodes,
        n_censored_episodes,
    )

    return RiskSetBuildResult(
        hazard_table=hazard_table,
        aligned_entropy_table=aligned_entropy_table,
        warnings=warnings,
        subject_mapping_assumptions=subject_mapping_assumptions,
        n_subjects=int(hazard_table["subject_id"].nunique()),
        n_episodes=n_episodes,
        n_positive_episodes=n_positive_episodes,
        n_censored_episodes=n_censored_episodes,
    )


def _build_positive_episode_table(
    *,
    events_table: pd.DataFrame,
    config: HazardAnalysisConfig,
    dyad_mapping: dict[str, dict[str, str]],
    entropy_by_run: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Build episodes with an observed target response onset."""

    required_columns = {
        config.event_definition.recording_id_column,
        config.event_definition.run_column,
        config.event_definition.partner_speaker_column,
        config.event_definition.target_speaker_column,
        config.event_definition.partner_onset_column,
        config.event_definition.target_onset_column,
        config.event_definition.target_label_column,
        config.event_definition.partner_label_column,
        config.event_definition.event_id_column,
    }
    missing_columns = required_columns - set(events_table.columns)
    if missing_columns:
        raise ValueError(f"Events table is missing required columns: {sorted(missing_columns)}")

    working = events_table.copy()
    working["partner_onset_seconds"] = pd.to_numeric(
        working[config.event_definition.partner_onset_column], errors="coerce"
    )
    working = working.loc[
        working[config.event_definition.partner_label_column].astype(str).str.startswith(
            tuple(config.event_definition.fpp_label_prefixes)
        )
    ].copy()
    working["target_onset_seconds"] = pd.to_numeric(
        working[config.event_definition.target_onset_column], errors="coerce"
    )
    working = working.loc[
        working["partner_onset_seconds"].notna() & working["target_onset_seconds"].notna()
    ].copy()
    working = working.loc[
        working[config.event_definition.target_label_column].astype(str).str.len() > 0
    ].copy()
    working["tau_event_seconds"] = working["target_onset_seconds"] - working["partner_onset_seconds"]
    working = working.loc[
        (working["tau_event_seconds"] > 0.0)
        & (working["tau_event_seconds"] <= config.time_axis.observation_window_seconds)
    ].copy()
    working["subject_id"] = working.apply(
        lambda row: _resolve_subject_id_from_event(
            recording_id=str(row[config.event_definition.recording_id_column]),
            speaker_label=str(row[config.event_definition.target_speaker_column]),
            dyad_mapping=dyad_mapping,
        ),
        axis=1,
    )
    working["run_id"] = working[config.event_definition.run_column].astype(str)
    working["dyad_id"] = working[config.event_definition.recording_id_column].astype(str)
    working = working.loc[
        working["subject_id"].notna()
        & working.apply(lambda row: (str(row["subject_id"]), str(row["run_id"])) in entropy_by_run, axis=1)
    ].copy()
    working["event_id"] = working.apply(
        lambda row: f"{row['subject_id']}|run-{row['run_id']}|{row[config.event_definition.event_id_column]}",
        axis=1,
    )
    working["censored_episode"] = 0

    return working.loc[
        :,
        [
            "subject_id",
            "event_id",
            "run_id",
            "dyad_id",
            "partner_onset_seconds",
            "target_onset_seconds",
            "tau_event_seconds",
            "censored_episode",
        ],
    ].reset_index(drop=True)


def _build_censored_episode_table(
    *,
    pairing_issues_table: pd.DataFrame | None,
    config: HazardAnalysisConfig,
    dyad_mapping: dict[str, dict[str, str]],
    entropy_by_run: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Build censored episodes from unpaired initiating events."""

    if pairing_issues_table is None:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "event_id",
                "run_id",
                "dyad_id",
                "partner_onset_seconds",
                "target_onset_seconds",
                "tau_event_seconds",
                "censored_episode",
            ]
        )

    required_columns = {
        config.event_definition.recording_id_column,
        config.event_definition.run_column,
        config.event_definition.issue_partner_tier_column,
        config.event_definition.issue_partner_label_column,
        config.event_definition.issue_partner_onset_column,
        config.event_definition.issue_code_column,
    }
    missing_columns = required_columns - set(pairing_issues_table.columns)
    if missing_columns:
        raise ValueError(f"Pairing-issues table is missing required columns: {sorted(missing_columns)}")

    working = pairing_issues_table.copy()
    working = working.loc[
        working[config.event_definition.issue_code_column].astype(str).isin(
            config.event_definition.censoring_issue_codes
        )
    ].copy()
    working = working.loc[
        working[config.event_definition.issue_partner_label_column].astype(str).str.startswith(
            tuple(config.event_definition.fpp_label_prefixes)
        )
    ].copy()
    working["partner_onset_seconds"] = pd.to_numeric(
        working[config.event_definition.issue_partner_onset_column], errors="coerce"
    )
    working = working.loc[working["partner_onset_seconds"].notna()].copy()
    working["target_speaker"] = working[config.event_definition.issue_partner_tier_column].map(
        _infer_opposite_speaker_from_action_tier
    )
    working["subject_id"] = working.apply(
        lambda row: _resolve_subject_id_from_event(
            recording_id=str(row[config.event_definition.recording_id_column]),
            speaker_label=str(row["target_speaker"]),
            dyad_mapping=dyad_mapping,
        ),
        axis=1,
    )
    working["run_id"] = working[config.event_definition.run_column].astype(str)
    working["dyad_id"] = working[config.event_definition.recording_id_column].astype(str)
    working = working.loc[
        working["subject_id"].notna()
        & working.apply(lambda row: (str(row["subject_id"]), str(row["run_id"])) in entropy_by_run, axis=1)
    ].copy()
    working["event_id"] = working.apply(
        lambda row: (
            f"{row['subject_id']}|run-{row['run_id']}|censored|"
            f"{row[config.event_definition.issue_partner_tier_column]}|{row.name}"
        ),
        axis=1,
    )
    working["target_onset_seconds"] = np.nan
    working["tau_event_seconds"] = np.nan
    working["censored_episode"] = 1

    return working.loc[
        :,
        [
            "subject_id",
            "event_id",
            "run_id",
            "dyad_id",
            "partner_onset_seconds",
            "target_onset_seconds",
            "tau_event_seconds",
            "censored_episode",
        ],
    ].reset_index(drop=True)


def _build_episode_rows(
    *,
    episode_row: dict[str, Any],
    run_entropy_frame: pd.DataFrame,
    time_axis_config: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build one forward-running partner-onset episode."""

    entropy_lookup = _prepare_entropy_lookup(run_entropy_frame)
    bin_end_times = _build_forward_bin_end_times(
        observation_window_seconds=float(time_axis_config.observation_window_seconds),
        bin_size_seconds=float(time_axis_config.bin_size_seconds),
        exclude_initial_seconds=float(time_axis_config.exclude_initial_seconds),
    )
    target_onset_seconds = episode_row["target_onset_seconds"]
    tau_event_seconds = episode_row["tau_event_seconds"]
    is_censored = bool(episode_row["censored_episode"])

    if not is_censored and not np.isfinite(float(tau_event_seconds)):
        return [], f"Skipped episode {episode_row['event_id']} because the target onset was invalid."
    if not is_censored and float(tau_event_seconds) <= float(time_axis_config.exclude_initial_seconds):
        return [], (
            f"Skipped episode {episode_row['event_id']} because the target onset occurred before the "
            "eligible risk window started."
        )

    rows: list[dict[str, Any]] = []
    for bin_index, tau_seconds in enumerate(bin_end_times):
        predictor_time_seconds = (
            float(episode_row["partner_onset_seconds"])
            + float(tau_seconds)
            - float(time_axis_config.entropy_lag_seconds)
        )
        entropy_value = _lookup_entropy_at_or_before_time(
            time_seconds=predictor_time_seconds,
            entropy_lookup=entropy_lookup,
        )
        if entropy_value is None:
            if rows:
                continue
            return [], (
                f"Skipped episode {episode_row['event_id']} because lagged entropy was unavailable at episode start."
            )

        bin_start_seconds = float(time_axis_config.exclude_initial_seconds) if bin_index == 0 else float(
            bin_end_times[bin_index - 1]
        )
        event_indicator = 0
        if not is_censored and (float(tau_event_seconds) <= float(tau_seconds)) and (
            float(tau_event_seconds) > float(bin_start_seconds)
        ):
            event_indicator = 1

        rows.append(
            {
                "subject_id": str(episode_row["subject_id"]),
                "event_id": str(episode_row["event_id"]),
                "run_id": str(episode_row["run_id"]),
                "dyad_id": str(episode_row["dyad_id"]),
                "partner_onset_seconds": float(episode_row["partner_onset_seconds"]),
                "target_onset_seconds": (
                    np.nan if is_censored else float(target_onset_seconds)
                ),
                "tau_seconds": float(tau_seconds),
                "tau_seconds_sq": float(tau_seconds**2),
                "bin_index": int(bin_index),
                "predictor_time_seconds": float(predictor_time_seconds),
                "entropy": float(entropy_value),
                "event": int(event_indicator),
                "censored_episode": int(is_censored),
            }
        )
        if event_indicator == 1:
            break

    if not rows:
        return [], f"Skipped episode {episode_row['event_id']} because no valid rows remained after alignment."
    return rows, None


def _prepare_entropy_lookup(run_entropy_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sorted finite entropy times and values for causal lookup."""

    time_values = run_entropy_frame["time_s"].to_numpy(dtype=float)
    entropy_values = pd.to_numeric(run_entropy_frame["state_entropy"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(time_values) & np.isfinite(entropy_values)
    return time_values[finite_mask], entropy_values[finite_mask]


def _lookup_entropy_at_or_before_time(
    *,
    time_seconds: float,
    entropy_lookup: tuple[np.ndarray, np.ndarray],
) -> float | None:
    """Return the last finite entropy value available at or before a target time."""

    time_values, entropy_values = entropy_lookup
    if time_values.size == 0:
        return None
    insertion_index = int(np.searchsorted(time_values, time_seconds, side="right")) - 1
    if insertion_index < 0:
        return None
    return float(entropy_values[insertion_index])


def _build_forward_bin_end_times(
    *,
    observation_window_seconds: float,
    bin_size_seconds: float,
    exclude_initial_seconds: float,
) -> np.ndarray:
    """Build right-edge bin times running forward from partner onset."""

    first_bin_end = exclude_initial_seconds + bin_size_seconds
    bin_end_times = np.arange(first_bin_end, observation_window_seconds + bin_size_seconds, bin_size_seconds)
    return bin_end_times[bin_end_times <= observation_window_seconds + 1.0e-12]


def _add_entropy_zscore_column(*, hazard_table: pd.DataFrame, zscore_within_subject: bool) -> None:
    """Add the model input column ``entropy_z``."""

    if zscore_within_subject:
        hazard_table["entropy_z"] = (
            hazard_table.groupby("subject_id", sort=False)["entropy"]
            .transform(_zscore_with_zero_variance_protection)
            .astype(float)
        )
    else:
        hazard_table["entropy_z"] = hazard_table["entropy"].astype(float)


def _zscore_with_zero_variance_protection(values: pd.Series) -> pd.Series:
    """Z-score a series while protecting zero variance."""

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0))
    if std_value == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values.astype(float) - mean_value) / std_value


def _validate_hazard_table(hazard_table: pd.DataFrame) -> None:
    """Validate the person-period table."""

    if not np.isfinite(hazard_table["entropy"].to_numpy(dtype=float)).all():
        raise ValueError("Entropy contains non-finite values after alignment.")
    duplicated = hazard_table.duplicated(subset=["event_id", "bin_index"])
    if duplicated.any():
        raise ValueError("Duplicate event_id/bin_index rows were found in the hazard table.")

    event_group = hazard_table.groupby("event_id", sort=False)
    positive_counts = event_group["event"].sum()
    if (positive_counts > 1).any():
        raise ValueError("Each episode may contain at most one positive row.")
    for _, group in event_group:
        positive_positions = np.flatnonzero(group["event"].to_numpy(dtype=int) == 1)
        if positive_positions.size == 0:
            continue
        positive_position = int(positive_positions[0])
        if positive_position != len(group) - 1:
            raise ValueError("No rows may appear after the positive row within an episode.")


def _build_dyad_mapping(dyad_table: pd.DataFrame | None) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Build speaker-to-subject mapping by dyad."""

    assumptions: list[str] = []
    mapping: dict[str, dict[str, str]] = {}
    if dyad_table is not None and {"dyad_id", "subject_id"}.issubset(dyad_table.columns):
        for dyad_id, frame in dyad_table.groupby("dyad_id", sort=False):
            subject_ids = sorted(str(value) for value in frame["subject_id"].tolist())
            if len(subject_ids) != 2:
                continue
            mapping[str(dyad_id)] = {"A": subject_ids[0], "B": subject_ids[1]}
        assumptions.append(
            "Used the dyads CSV where available, ordering paired subject IDs lexicographically as speaker A then B."
        )
    assumptions.append(
        "Missing dyad mappings fall back to the repository convention odd subject number -> speaker A, even subject number -> speaker B."
    )
    assumptions.append(
        "Censored episodes are derived from pairing_issues.csv rows with issue_code in the configured censoring_issue_codes."
    )
    return assumptions, mapping


def _resolve_subject_id_from_event(
    *,
    recording_id: str,
    speaker_label: str,
    dyad_mapping: dict[str, dict[str, str]],
) -> str | None:
    """Resolve the subject ID for one event row."""

    speaker_clean = str(speaker_label).strip().upper()
    if speaker_clean not in {"A", "B"}:
        return None
    if recording_id in dyad_mapping and speaker_clean in dyad_mapping[recording_id]:
        return dyad_mapping[recording_id][speaker_clean]

    match = DYAD_NUMBER_PATTERN.match(recording_id)
    if match is None:
        return None
    dyad_number = int(match.group("number"))
    subject_a_number = (2 * dyad_number) - 1
    subject_b_number = 2 * dyad_number
    if speaker_clean == "A":
        return f"sub-{subject_a_number:03d}"
    return f"sub-{subject_b_number:03d}"


def _infer_opposite_speaker_from_action_tier(action_tier_label: str) -> str | None:
    """Infer the target speaker as the opposite of the initiating action tier."""

    match = ACTION_TIER_SPEAKER_PATTERN.search(str(action_tier_label).strip())
    if match is None:
        return None
    partner_speaker = str(match.group("speaker")).upper()
    return "B" if partner_speaker == "A" else "A"
