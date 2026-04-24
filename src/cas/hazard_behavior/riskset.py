"""Discrete-time risk-set construction for behavioural FPP hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.progress import progress_iterable

FLOAT_TOLERANCE = 1.0e-9
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskSetResult:
    """Risk-set outputs for the behavioural hazard pipeline."""

    riskset_table: pd.DataFrame
    episode_summary: pd.DataFrame
    warnings: list[str]


def build_discrete_time_riskset(
    episodes_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> RiskSetResult:
    """Build one row per at-risk 50 ms bin.

    Usage example
    -------------
        result = build_discrete_time_riskset(episodes, config=config)
        riskset = result.riskset_table
    """

    LOGGER.info("Building discrete-time risk set for %d episodes.", len(episodes_table))
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
                "participant_speaker": episode["participant_speaker"],
                "partner_speaker": episode["partner_speaker"],
                "partner_ipu_onset": episode["partner_ipu_onset"],
                "partner_ipu_offset": episode["partner_ipu_offset"],
                "own_fpp_onset": episode["own_fpp_onset"],
                "latency_from_partner_offset_s": episode.get("latency_from_partner_offset_s", np.nan),
                "partner_ipu_overlaps_fpp": bool(episode.get("partner_ipu_overlaps_fpp", False)),
                "partner_ipu_was_truncated": bool(episode.get("partner_ipu_was_truncated", False)),
                "episode_is_valid": bool(episode.get("episode_is_valid", True)),
                "invalid_reason": str(episode.get("invalid_reason", "")),
                "censor_time": episode["censor_time"],
                "episode_kind": episode["episode_kind"],
                "event_observed": episode["event_observed"],
                "n_bins": int(len(episode_rows)),
            }
        )

    riskset_table = pd.DataFrame(rows)
    LOGGER.info("Constructed %d risk-set rows across %d episodes.", len(riskset_table), len(episode_summaries))
    validate_riskset(riskset_table)
    return RiskSetResult(
        riskset_table=riskset_table,
        episode_summary=pd.DataFrame(episode_summaries),
        warnings=warnings,
    )


def assign_event_bins(
    episode_rows: pd.DataFrame,
    *,
    event_onset: float | None,
) -> pd.DataFrame:
    """Assign the unique event bin within an episode."""

    rows = episode_rows.copy()
    rows["event"] = 0
    if event_onset is None or not np.isfinite(event_onset):
        return rows
    event_index = _locate_event_index(rows, event_onset)
    if event_index is not None:
        rows.loc[event_index, "event"] = 1
        rows = rows.loc[:event_index].copy()
    return rows.reset_index(drop=True)


def validate_riskset(riskset_table: pd.DataFrame) -> None:
    """Validate core discrete-time hazard invariants."""

    if riskset_table.empty:
        raise ValueError("Risk-set table is empty.")
    required_columns = {
        "episode_id",
        "bin_index",
        "bin_start",
        "bin_end",
        "time_from_partner_onset",
        "event",
    }
    missing = sorted(required_columns - set(riskset_table.columns))
    if missing:
        raise ValueError(f"Risk-set table is missing required columns: {missing}")
    event_counts = riskset_table.groupby("episode_id")["event"].sum()
    if (event_counts > 1).any():
        raise ValueError("Each episode may contain at most one event bin.")


def _build_episode_bins(
    *,
    episode: pd.Series,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    anchor = float(episode["partner_ipu_onset"])
    censor_time = float(episode["censor_time"])
    duration = censor_time - anchor
    if duration < config.minimum_episode_duration_s - FLOAT_TOLERANCE:
        return pd.DataFrame()

    quotient = duration / config.bin_size_s
    if np.isclose(quotient, round(quotient)):
        n_bins = int(round(quotient)) + 1
    else:
        n_bins = int(np.floor(quotient + FLOAT_TOLERANCE)) + 1
    bin_starts = np.array(
        [round(anchor + bin_index * config.bin_size_s, 10) for bin_index in range(n_bins)],
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    for bin_index, bin_start in enumerate(bin_starts):
        bin_end = bin_start + config.bin_size_s
        rows.append(
            {
                "dyad_id": str(episode["dyad_id"]),
                "run": str(episode["run"]),
                "participant_speaker": str(episode["participant_speaker"]),
                "partner_speaker": str(episode["partner_speaker"]),
                "episode_id": str(episode["episode_id"]),
                "episode_kind": str(episode["episode_kind"]),
                "bin_index": int(bin_index),
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "time_from_partner_onset": float(bin_start - anchor),
                "partner_ipu_onset": float(episode["partner_ipu_onset"]),
                "partner_ipu_offset": float(episode["partner_ipu_offset"]),
                "own_fpp_onset": float(episode["own_fpp_onset"]) if pd.notna(episode["own_fpp_onset"]) else np.nan,
                "censor_time": float(censor_time),
                "partner_ipu_class": str(episode.get("partner_ipu_class", "unknown")),
                "partner_role": str(episode.get("partner_role", "partner")),
            }
        )
    episode_rows = pd.DataFrame(rows)
    event_onset = float(episode["own_fpp_onset"]) if pd.notna(episode["own_fpp_onset"]) else None
    return assign_event_bins(episode_rows, event_onset=event_onset)


def _locate_event_index(rows: pd.DataFrame, event_onset: float) -> int | None:
    for row_number, row in rows.iterrows():
        bin_start = float(row["bin_start"])
        bin_end = float(row["bin_end"])
        is_last_row = row_number == rows.index[-1]
        starts_here = event_onset > bin_start or np.isclose(event_onset, bin_start)
        ends_after = event_onset < bin_end and not np.isclose(event_onset, bin_start)
        final_edge_match = is_last_row and np.isclose(event_onset, bin_end)
        if (starts_here and ends_after) or np.isclose(event_onset, bin_start) or final_edge_match:
            return int(row_number)
    return None
