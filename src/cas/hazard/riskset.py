"""Risk-set construction for partner-onset discrete-time hazard analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cas.hazard.config import HazardAnalysisConfig, NeuralHazardConfig
from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_partner_ipus_from_tokens, infer_participant_speaker
from cas.hazard_behavior.riskset import RiskSetResult, build_discrete_time_riskset

LOGGER = logging.getLogger(__name__)
DYAD_NUMBER_PATTERN = re.compile(r"dyad-(?P<number>\d+)$")
ACTION_TIER_SPEAKER_PATTERN = re.compile(r"action\s+(?P<speaker>[AB])$", re.IGNORECASE)
EPSILON = 1.0e-9


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


@dataclass(frozen=True, slots=True)
class NeuralRiskSetBuildResult:
    """Partner-IPU-anchored low-level neural risk sets."""

    risksets_by_event: dict[str, pd.DataFrame]
    episode_summaries_by_event: dict[str, pd.DataFrame]
    event_qc_by_event: dict[str, dict[str, object]]
    warnings: list[str]


def build_neural_partner_ipu_risksets(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    neural_config: NeuralHazardConfig,
) -> NeuralRiskSetBuildResult:
    """Build partner-IPU anchored risk sets for FPP and SPP neural hazards.

    Usage example
    -------------
    >>> result = build_neural_partner_ipu_risksets(
    ...     events_table=events_table,
    ...     surprisal_table=surprisal_table,
    ...     neural_config=config.neural,
    ... )
    >>> sorted(result.risksets_by_event)
    ['fpp', 'spp']
    """

    warnings: list[str] = []
    ipu_table = build_partner_ipus_from_tokens(
        surprisal_table,
        gap_threshold_s=neural_config.episode.ipu_gap_threshold_s,
    )
    if ipu_table.empty:
        raise ValueError("No partner IPUs were available for neural hazard episode construction.")
    risksets_by_event: dict[str, pd.DataFrame] = {}
    summaries_by_event: dict[str, pd.DataFrame] = {}
    qc_by_event: dict[str, dict[str, object]] = {}

    for event_type in neural_config.event_types:
        episodes = _build_neural_episodes_for_event_type(
            events_table=events_table,
            ipu_table=ipu_table,
            event_type=str(event_type),
            neural_config=neural_config,
        )
        riskset_result = _build_neural_discrete_time_riskset(
            episodes,
            event_type=str(event_type),
            bin_size_s=neural_config.bin_size_s,
        )
        risksets_by_event[str(event_type)] = riskset_result.riskset_table
        summaries_by_event[str(event_type)] = riskset_result.episode_summary
        qc_by_event[str(event_type)] = riskset_result.event_qc
        warnings.extend(riskset_result.warnings)

    return NeuralRiskSetBuildResult(
        risksets_by_event=risksets_by_event,
        episode_summaries_by_event=summaries_by_event,
        event_qc_by_event=qc_by_event,
        warnings=warnings,
    )


def _build_neural_episodes_for_event_type(
    *,
    events_table: pd.DataFrame,
    ipu_table: pd.DataFrame,
    event_type: str,
    neural_config: NeuralHazardConfig,
) -> pd.DataFrame:
    target_events = _extract_target_events(events_table, event_type=event_type)
    episode_rows: list[dict[str, object]] = []
    for ipu_row in ipu_table.to_dict("records"):
        dyad_id = str(ipu_row["dyad_id"])
        run = str(ipu_row["run"])
        partner_speaker = str(ipu_row["speaker"])
        participant_speaker = infer_participant_speaker(partner_speaker)
        episode_start = float(ipu_row["partner_ipu_onset"])
        next_partner_ipu_onset = pd.to_numeric(
            pd.Series([ipu_row.get("next_partner_ipu_onset")]),
            errors="coerce",
        ).iloc[0]
        candidates = [(episode_start + neural_config.episode.max_followup_s, "max_followup")]
        if np.isfinite(next_partner_ipu_onset):
            candidates.append((float(next_partner_ipu_onset), "next_partner_ipu"))
        episode_end, censor_reason = min(candidates, key=lambda value: float(value[0]))
        if episode_end <= episode_start:
            raise ValueError(f"Episode end must be after start; found {episode_end} <= {episode_start}.")

        matches = target_events.loc[
            (target_events["dyad_id"] == dyad_id)
            & (target_events["run"] == run)
            & (target_events["event_speaker"] == participant_speaker)
            & (target_events["event_onset"] >= episode_start)
            & (target_events["event_onset"] < float(episode_end))
        ].sort_values(["event_onset", "source_event_id"], kind="mergesort")
        event_row = matches.iloc[0] if not matches.empty else None
        has_event = event_row is not None
        event_onset = float(event_row["event_onset"]) if event_row is not None else np.nan
        partner_ipu_offset = float(ipu_row["partner_ipu_offset"])
        latency_from_offset = event_onset - partner_ipu_offset if has_event else np.nan
        event_phase = "during_partner_ipu" if has_event and latency_from_offset < 0.0 else ("post_partner_ipu" if has_event else "censored")
        episode_id = (
            f"{dyad_id}|run-{run}|event-{event_type}|{partner_speaker}|"
            f"ipu-{str(ipu_row['partner_ipu_id']).split('|')[-1]}"
        )
        episode_rows.append(
            {
                "episode_id": episode_id,
                "dyad_id": dyad_id,
                "run": run,
                "partner_speaker": partner_speaker,
                "participant_speaker": participant_speaker,
                "partner_ipu_class": str(ipu_row.get("partner_ipu_class", "unknown")),
                "partner_role": "partner",
                "partner_ipu_id": str(ipu_row["partner_ipu_id"]),
                "partner_ipu_onset": episode_start,
                "partner_ipu_offset": partner_ipu_offset,
                "partner_ipu_duration": float(ipu_row["partner_ipu_duration"]),
                "next_partner_ipu_onset": float(next_partner_ipu_onset) if np.isfinite(next_partner_ipu_onset) else np.nan,
                "episode_start": episode_start,
                "episode_end": float(event_onset if has_event else episode_end),
                "episode_has_event": bool(has_event),
                "own_fpp_onset": float(event_onset) if has_event else np.nan,
                "own_fpp_offset": np.nan,
                "own_fpp_label": "" if event_row is None else str(event_row["event_label"]),
                "event_latency_from_partner_onset_s": float(event_onset - episode_start) if has_event else np.nan,
                "event_latency_from_partner_offset_s": float(latency_from_offset) if has_event else np.nan,
                "event_phase": event_phase,
                "censor_time": float(event_onset if has_event else episode_end),
                "episode_duration_s": float((event_onset if has_event else episode_end) - episode_start),
                "episode_kind": "event_positive" if has_event else "censored",
                "censor_reason": "event" if has_event else str(censor_reason),
                "anchor_source": str(ipu_row.get("anchor_source", "partner_ipu_tokens")),
                "target_event_type": event_type,
                "event_onset": float(event_onset) if has_event else np.nan,
            }
        )
    episodes = pd.DataFrame(episode_rows)
    if not neural_config.episode.include_censored:
        episodes = episodes.loc[episodes["episode_has_event"]].copy()
    episodes = episodes.sort_values(["dyad_id", "run", "partner_ipu_onset"], kind="mergesort").reset_index(drop=True)
    return episodes


def _build_neural_discrete_time_riskset(
    episodes_table: pd.DataFrame,
    *,
    event_type: str,
    bin_size_s: float,
) -> RiskSetResult:
    behaviour_like_config = BehaviourHazardConfig(
        events_path=Path("."),
        surprisal_paths=tuple(),
        out_dir=Path("."),
        bin_size_s=float(bin_size_s),
    )
    result = build_discrete_time_riskset(episodes_table, config=behaviour_like_config)
    riskset = result.riskset_table.copy()
    event_column = f"event_{event_type.lower()}"
    riskset[event_column] = pd.to_numeric(riskset["event"], errors="coerce").fillna(0).astype(int)
    riskset["event_type"] = str(event_type).lower()
    if "participant_id" not in riskset.columns:
        riskset["participant_id"] = riskset["dyad_id"].astype(str) + "_" + riskset["participant_speaker"].astype(str)
    _validate_neural_riskset_event_timing(riskset, event_column=event_column)
    return RiskSetResult(
        riskset_table=riskset,
        episode_summary=result.episode_summary,
        warnings=result.warnings,
        event_qc=result.event_qc,
    )


def _validate_neural_riskset_event_timing(riskset_table: pd.DataFrame, *, event_column: str) -> None:
    event_rows = riskset_table.loc[pd.to_numeric(riskset_table[event_column], errors="coerce") == 1].copy()
    if event_rows.empty:
        return
    onset_column = "event_onset" if "event_onset" in event_rows.columns else "own_fpp_onset"
    if not (
        pd.to_numeric(event_rows[onset_column], errors="coerce")
        < pd.to_numeric(event_rows["censor_time"], errors="coerce") + EPSILON
    ).all():
        raise ValueError("Event rows must occur at or before censoring.")
    if not (
        pd.to_numeric(event_rows[onset_column], errors="coerce")
        >= pd.to_numeric(event_rows["bin_start"], errors="coerce") - EPSILON
    ).all():
        raise ValueError("Event rows must fall within their event bins (left edge).")
    if not (
        pd.to_numeric(event_rows[onset_column], errors="coerce")
        < pd.to_numeric(event_rows["bin_end"], errors="coerce") + EPSILON
    ).all():
        raise ValueError("Event rows must fall within their event bins (right edge).")


def _extract_target_events(events_table: pd.DataFrame, *, event_type: str) -> pd.DataFrame:
    working = events_table.copy()
    working["dyad_id"] = working["dyad_id"].astype(str)
    working["run"] = working["run"].astype(str).str.replace(r"^run[-_ ]*", "", regex=True)
    normalized_event_type = str(event_type).lower()
    if normalized_event_type == "fpp":
        onset_column = "fpp_onset"
        label_column = "fpp_label" if "fpp_label" in working.columns else None
        speaker_column = _resolve_speaker_column(working, ("participant_speaker", "speaker_fpp", "fpp_speaker", "speaker"))
    elif normalized_event_type == "spp":
        onset_column = "spp_onset"
        label_column = "spp_label" if "spp_label" in working.columns else None
        speaker_column = _resolve_speaker_column(working, ("partner_speaker", "speaker_spp", "spp_speaker"))
    else:
        raise ValueError(f"Unsupported event type: {event_type}")
    working["event_onset"] = pd.to_numeric(working[onset_column], errors="coerce")
    working["event_speaker"] = working[speaker_column].astype(str).str.upper().str.strip()
    if label_column is not None:
        working["event_label"] = working[label_column].astype(str)
    else:
        working["event_label"] = ""
    if normalized_event_type == "fpp" and label_column is not None:
        working = working.loc[working["event_label"].str.startswith("FPP_")].copy()
    if normalized_event_type == "spp" and label_column is not None:
        working = working.loc[working["event_label"].str.startswith("SPP_")].copy()
    extracted = working.loc[
        working["event_onset"].notna() & working["event_speaker"].isin({"A", "B"}),
        ["dyad_id", "run", "event_speaker", "event_onset", "event_label"],
    ].copy()
    extracted["source_event_id"] = np.arange(len(extracted), dtype=int)
    return extracted.reset_index(drop=True)


def _resolve_speaker_column(events_table: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column_name in candidates:
        if column_name in events_table.columns:
            return column_name
    available = ", ".join(sorted(str(column) for column in events_table.columns))
    raise ValueError(f"Could not find speaker column among {candidates}; available: {available}")
