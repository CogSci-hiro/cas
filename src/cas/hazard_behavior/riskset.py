"""Discrete-time risk-set construction for behavioural FPP hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id
from cas.hazard_behavior.progress import progress_iterable

FLOAT_TOLERANCE = 1.0e-9
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskSetResult:
    """Risk-set outputs for the behavioural hazard pipeline."""

    riskset_table: pd.DataFrame
    episode_summary: pd.DataFrame
    warnings: list[str]
    event_qc: dict[str, object]


def build_discrete_time_riskset(
    episodes_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> RiskSetResult:
    """Build one row per at-risk discrete-time hazard bin."""

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
    LOGGER.info("Constructed %d risk-set rows across %d episodes.", len(riskset_table), len(episode_summaries))
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
