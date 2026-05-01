"""Episode construction for behavioural FPP hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.progress import progress_iterable

DEFAULT_PARTNER_IPU_CLASS = "unknown"
SUPPORTED_SPEAKERS = {"A", "B"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeBuildResult:
    """Episode tables and associated validation artifacts."""

    episodes: pd.DataFrame
    candidate_episodes: pd.DataFrame
    excluded_episodes: pd.DataFrame
    validation_qc: dict[str, object]
    warnings: list[str]
    used_partner_anchor: str
    partner_ipu_table: pd.DataFrame
    event_rows_debug: pd.DataFrame


def infer_partner_speaker(participant_speaker: str) -> str:
    """Infer the partner speaker in a two-speaker dyad."""

    speaker = _normalize_speaker_label(participant_speaker)
    if speaker == "A":
        return "B"
    if speaker == "B":
        return "A"
    raise ValueError(f"Expected participant speaker A or B, found {participant_speaker!r}.")


def infer_participant_speaker(partner_speaker: str) -> str:
    """Infer the non-partner speaker in a two-speaker dyad."""

    return infer_partner_speaker(partner_speaker)


def compute_ipu_token_table(
    surprisal_table: pd.DataFrame,
    *,
    gap_threshold_s: float,
) -> pd.DataFrame:
    """Group consecutive tokens into IPU-like intervals."""

    return build_partner_ipus_from_tokens(surprisal_table, gap_threshold_s=gap_threshold_s)


def extract_fpp_events(
    events_table: pd.DataFrame,
    config: BehaviourHazardConfig,
    *,
    token_speakers: set[str] | None = None,
) -> pd.DataFrame:
    """Extract one row per target FPP event from the events table."""

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
    """Infer partner IPUs from token timing gaps."""

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
        ipu_rows = _build_ipu_rows_from_tokens(
            sorted_tokens=sorted_tokens,
            group_key=(str(group_key[0]), str(group_key[1]), str(group_key[2])),
            gap_threshold_s=gap_threshold_s,
        )
        rows.extend(ipu_rows)

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


def compute_next_same_speaker_ipu_onset(ipu_table: pd.DataFrame) -> pd.Series:
    """Return the next onset for each IPU within speaker-specific dyad/run groups."""

    return (
        ipu_table.sort_values(["dyad_id", "run", "speaker", "partner_ipu_onset"], kind="mergesort")
        .groupby(["dyad_id", "run", "speaker"], sort=False)["partner_ipu_onset"]
        .shift(-1)
    )


def find_first_fpp_in_episode_window(
    fpp_events: pd.DataFrame,
    *,
    dyad_id: str,
    run: str,
    participant_speaker: str,
    episode_start: float,
    episode_end: float,
) -> pd.Series | None:
    """Find the first FPP onset within an episode window."""

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
    """Build partner-IPU-anchored behavioural hazard episodes."""

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


def compute_partner_ipu_anchor_qc(
    *,
    partner_ipu_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    fpp_events: pd.DataFrame,
    assigned_event_ids: list[int],
    config: BehaviourHazardConfig,
) -> dict[str, object]:
    """Compute anchor QC for partner-IPU episodes."""

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
    """Validate structural invariants for partner-IPU episodes."""

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
    duplicated_assignments = len(assigned_event_ids) != len(set(assigned_event_ids))
    if duplicated_assignments:
        raise ValueError("No FPP event may be assigned to more than one partner-IPU episode.")
    missing_assigned = set(assigned_event_ids) - set(pd.to_numeric(fpp_events["source_event_id"], errors="coerce").astype(int))
    if missing_assigned:
        raise ValueError(f"Assigned event ids were not found in extracted FPP events: {sorted(missing_assigned)}")


def build_event_positive_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> EpisodeBuildResult:
    """Build and validate event-positive episodes anchored at preceding partner speech."""

    if config.episode_anchor == "partner_ipu":
        return build_partner_ipu_anchored_episodes(
            events_table=events_table,
            surprisal_table=surprisal_table,
            config=config,
        )

    warnings: list[str] = []
    working = events_table.copy()
    working["dyad_id"] = working["dyad_id"].map(_normalize_dyad_id)
    working["run"] = working["run"].map(_normalize_run_label)
    working["participant_speaker"] = working["participant_speaker"].map(_normalize_speaker_label)
    working["fpp_onset"] = pd.to_numeric(working["fpp_onset"], errors="coerce")
    if "partner_speaker" not in working.columns:
        working["partner_speaker"] = working["participant_speaker"].map(infer_partner_speaker)
        warnings.append("Partner speaker was inferred from participant speaker labels.")
    else:
        working["partner_speaker"] = working["partner_speaker"].map(_normalize_speaker_label)

    used_partner_anchor = "events.spp_onset"
    if (
        "spp_onset" in working.columns
        and working["spp_onset"].notna().any()
        and "spp_offset" in working.columns
        and working["spp_offset"].notna().any()
    ):
        partner_assignment = _build_partner_assignment_from_event_columns(working, config=config)
    else:
        partner_assignment = infer_previous_partner_ipu_from_tokens(
            events_table=working,
            surprisal_table=surprisal_table,
            config=config,
        )
        used_partner_anchor = "tokens.ipu_fallback"
        warnings.append("Partner IPU was inferred from surprisal-token gaps because no usable SPP onset/offset pair was found.")

    candidate_episodes = working.merge(
        partner_assignment,
        on=["dyad_id", "run", "participant_speaker", "partner_speaker", "fpp_onset"],
        how="left",
    )
    candidate_episodes = _finalize_candidate_episodes(candidate_episodes, config=config)
    candidate_episodes["episode_id"] = [assign_episode_id(row, index) for index, row in candidate_episodes.iterrows()]

    valid_episodes = candidate_episodes.loc[candidate_episodes["episode_is_valid"]].copy().reset_index(drop=True)
    excluded_episodes = candidate_episodes.loc[~candidate_episodes["episode_is_valid"], ["episode_id", "invalid_reason"]].copy()
    validation_qc = compute_episode_validation_qc(candidate_episodes, valid_episodes, config=config)
    warnings.extend(build_episode_validation_warnings(validation_qc, config=config))
    return EpisodeBuildResult(
        episodes=valid_episodes,
        candidate_episodes=candidate_episodes.reset_index(drop=True),
        excluded_episodes=excluded_episodes.reset_index(drop=True),
        validation_qc=validation_qc,
        warnings=warnings,
        used_partner_anchor=used_partner_anchor,
        partner_ipu_table=pd.DataFrame(),
        event_rows_debug=pd.DataFrame(),
    )


def build_censored_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    positive_episodes: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Build optional censored episodes from partner IPUs not followed by own FPP."""

    if config.episode_anchor == "partner_ipu":
        return pd.DataFrame()
    if not config.include_censored:
        return pd.DataFrame()

    ipu_table = compute_ipu_token_table(surprisal_table, gap_threshold_s=config.ipu_gap_threshold_s)
    if ipu_table.empty:
        return pd.DataFrame()

    positive_pairs = positive_episodes.loc[:, ["dyad_id", "run", "partner_speaker", "partner_ipu_onset"]].drop_duplicates()
    candidate_ipus = ipu_table.rename(columns={"speaker": "partner_speaker"}).merge(
        positive_pairs.assign(_matched=1),
        on=["dyad_id", "run", "partner_speaker", "partner_ipu_onset"],
        how="left",
    )
    candidate_ipus = candidate_ipus.loc[candidate_ipus["_matched"].isna()].copy()
    if candidate_ipus.empty:
        return pd.DataFrame()

    fpp_rows = extract_fpp_events(events_table, config, token_speakers=set(ipu_table["speaker"].astype(str).unique()))
    episodes: list[dict[str, object]] = []
    for _, ipu_row in candidate_ipus.iterrows():
        partner_speaker = str(ipu_row["partner_speaker"])
        participant_speaker = infer_partner_speaker(partner_speaker)
        matching_events = fpp_rows.loc[
            (fpp_rows["dyad_id"] == str(ipu_row["dyad_id"]))
            & (fpp_rows["run"] == str(ipu_row["run"]))
            & (fpp_rows["fpp_speaker"] == participant_speaker)
            & (fpp_rows["fpp_onset"] > float(ipu_row["partner_ipu_onset"]))
            & (fpp_rows["fpp_onset"] <= float(ipu_row["partner_ipu_offset"]) + config.max_followup_s)
        ]
        if not matching_events.empty:
            continue
        censor_time = float(ipu_row["partner_ipu_offset"]) + config.max_followup_s
        episodes.append(
            {
                "dyad_id": str(ipu_row["dyad_id"]),
                "run": str(ipu_row["run"]),
                "participant_speaker": participant_speaker,
                "partner_speaker": partner_speaker,
                "partner_ipu_onset": float(ipu_row["partner_ipu_onset"]),
                "partner_ipu_offset": float(ipu_row["partner_ipu_offset"]),
                "partner_ipu_class": str(ipu_row.get("partner_ipu_class", DEFAULT_PARTNER_IPU_CLASS)),
                "own_fpp_onset": np.nan,
                "own_fpp_offset": np.nan,
                "censor_time": censor_time,
                "episode_duration": censor_time - float(ipu_row["partner_ipu_onset"]),
                "episode_kind": "censored",
                "event_observed": 0,
                "partner_role": "partner",
                "latency_from_partner_offset_s": np.nan,
                "partner_ipu_overlaps_fpp": False,
                "partner_ipu_was_truncated": False,
                "episode_is_valid": True,
                "invalid_reason": "",
            }
        )
    censored = pd.DataFrame(episodes)
    if censored.empty:
        return censored
    censored["episode_id"] = [assign_episode_id(row, index) for index, row in censored.iterrows()]
    return censored.reset_index(drop=True)


def infer_previous_partner_ipu_from_tokens(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Infer the latest partner IPU before each FPP onset using event-local token grouping."""

    rows: list[dict[str, object]] = []
    event_records = list(events_table.to_dict("records"))
    for event_row in progress_iterable(
        event_records,
        total=len(event_records),
        description="Inferring partner IPUs",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        dyad_id = _normalize_dyad_id(event_row["dyad_id"])
        run = _normalize_run_label(event_row["run"])
        participant_speaker = _normalize_speaker_label(event_row["participant_speaker"])
        partner_speaker = _normalize_speaker_label(event_row["partner_speaker"])
        fpp_onset = float(event_row["fpp_onset"])
        partner_tokens = surprisal_table.loc[
            (surprisal_table["dyad_id"].map(_normalize_dyad_id) == dyad_id)
            & (surprisal_table["run"].map(_normalize_run_label) == run)
            & (surprisal_table["speaker"].map(_normalize_speaker_label) == partner_speaker)
            & (pd.to_numeric(surprisal_table["onset"], errors="coerce") < fpp_onset)
        ].copy()
        partner_tokens["dyad_id"] = partner_tokens["dyad_id"].map(_normalize_dyad_id)
        partner_tokens["run"] = partner_tokens["run"].map(_normalize_run_label)
        partner_tokens["speaker"] = partner_tokens["speaker"].map(_normalize_speaker_label)
        partner_tokens = partner_tokens.sort_values(["onset", "offset"], kind="mergesort").reset_index(drop=True)
        local_ipus = _build_ipu_rows_from_tokens(
            sorted_tokens=partner_tokens,
            group_key=(dyad_id, run, partner_speaker),
            gap_threshold_s=config.ipu_gap_threshold_s,
        )
        rows.append(
            _select_partner_ipu_for_event(
                dyad_id=dyad_id,
                run=run,
                participant_speaker=participant_speaker,
                partner_speaker=partner_speaker,
                fpp_onset=fpp_onset,
                local_ipus=local_ipus,
                config=config,
            )
        )
    return pd.DataFrame(rows)


def assign_episode_id(row: pd.Series, index: int) -> str:
    """Assign a stable episode identifier."""

    onset = float(row["partner_ipu_onset"]) if pd.notna(row["partner_ipu_onset"]) else -1.0
    return (
        f"{row['dyad_id']}|run-{row['run']}|{row['participant_speaker']}|"
        f"{row['partner_speaker']}|{row['episode_kind']}|{onset:0.3f}|{index:04d}"
    )


def compute_episode_validation_qc(
    candidate_episodes: pd.DataFrame,
    valid_episodes: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> dict[str, object]:
    """Compute summary QC metrics for legacy episode validation."""

    candidate_latencies = pd.to_numeric(candidate_episodes["latency_from_partner_offset_s"], errors="coerce")
    valid_latencies = pd.to_numeric(valid_episodes["latency_from_partner_offset_s"], errors="coerce")
    negative_before = np.isfinite(candidate_latencies) & (
        candidate_latencies < -config.partner_offset_fpp_tolerance_s
    )
    negative_after = np.isfinite(valid_latencies) & (
        valid_latencies < -config.partner_offset_fpp_tolerance_s
    )
    n_candidates = int(len(candidate_episodes))
    n_valid = int(len(valid_episodes))
    n_excluded = int((~candidate_episodes["episode_is_valid"]).sum())
    return {
        "n_candidate_episodes": n_candidates,
        "n_valid_episodes": n_valid,
        "n_excluded_episodes": n_excluded,
        "n_overlapping_partner_ipu_episodes": int(candidate_episodes["partner_ipu_overlaps_fpp"].sum()),
        "proportion_overlapping_partner_ipu_episodes": _safe_proportion(
            int(candidate_episodes["partner_ipu_overlaps_fpp"].sum()),
            n_candidates,
        ),
        "n_truncated_episodes": int(candidate_episodes["partner_ipu_was_truncated"].sum()),
        "n_kept_overlapping_episodes": int(
            (candidate_episodes["partner_ipu_overlaps_fpp"] & candidate_episodes["episode_is_valid"]).sum()
        ),
        "median_latency_from_partner_offset_s": (
            float(valid_latencies.median()) if valid_latencies.notna().any() else None
        ),
        "min_latency_from_partner_offset_s": (
            float(valid_latencies.min()) if valid_latencies.notna().any() else None
        ),
        "max_latency_from_partner_offset_s": (
            float(valid_latencies.max()) if valid_latencies.notna().any() else None
        ),
        "proportion_negative_latency_before_exclusion": _safe_proportion(int(negative_before.sum()), int(np.isfinite(candidate_latencies).sum())),
        "proportion_negative_latency_after_exclusion": _safe_proportion(int(negative_after.sum()), int(np.isfinite(valid_latencies).sum())),
        "tolerance_used_s": float(config.partner_offset_fpp_tolerance_s),
        "overlapping_episode_strategy": config.overlapping_episode_strategy,
    }


def build_episode_validation_warnings(
    validation_qc: dict[str, object],
    *,
    config: BehaviourHazardConfig,
) -> list[str]:
    """Build warnings for problematic episode-validation patterns."""

    warnings: list[str] = []
    before = validation_qc.get("proportion_negative_latency_before_exclusion")
    after = validation_qc.get("proportion_negative_latency_after_exclusion")
    excluded = validation_qc.get("n_excluded_episodes")
    candidates = validation_qc.get("n_candidate_episodes")
    if isinstance(before, float) and before > 0.05:
        warnings.append(
            f"{before * 100:.1f}% of candidate episodes had partner_ipu_offset after own_fpp_onset. "
            f"These were handled under overlapping_episode_strategy='{config.overlapping_episode_strategy}'. "
            "This suggests token-based partner IPU grouping may be too broad or includes overlapping speech."
        )
    if (
        config.require_partner_offset_before_fpp
        and config.overlapping_episode_strategy != "keep"
        and isinstance(after, float)
        and after > 0.0
    ):
        warnings.append(
            "Negative partner-offset-to-FPP latencies remain after exclusion even though "
            "`require_partner_offset_before_fpp` is true."
        )
    if isinstance(excluded, int) and isinstance(candidates, int) and candidates > 0 and (excluded / candidates) > 0.2:
        warnings.append(
            f"{(excluded / candidates) * 100:.1f}% of candidate episodes were excluded because of invalid or "
            "overlapping partner intervals."
        )
    return warnings


def _build_partner_assignment_from_event_columns(
    events_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in events_table.iterrows():
        partner_ipu_onset = pd.to_numeric(pd.Series([row.get("spp_onset")]), errors="coerce").iloc[0]
        partner_ipu_offset = pd.to_numeric(pd.Series([row.get("spp_offset")]), errors="coerce").iloc[0]
        rows.append(
            _apply_overlap_strategy(
                dyad_id=str(row["dyad_id"]),
                run=str(row["run"]),
                participant_speaker=str(row["participant_speaker"]),
                partner_speaker=str(row["partner_speaker"]),
                fpp_onset=float(row["fpp_onset"]),
                partner_ipu_onset=partner_ipu_onset,
                partner_ipu_offset=partner_ipu_offset,
                partner_ipu_class=str(row.get("spp_label", DEFAULT_PARTNER_IPU_CLASS)),
                config=config,
            )
        )
    return pd.DataFrame(rows)


def _finalize_candidate_episodes(
    candidate_episodes: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    working = candidate_episodes.copy()
    working["own_fpp_onset"] = pd.to_numeric(working["fpp_onset"], errors="coerce")
    working["own_fpp_offset"] = pd.to_numeric(working.get("fpp_offset", np.nan), errors="coerce")
    working["partner_ipu_onset"] = pd.to_numeric(working["partner_ipu_onset"], errors="coerce")
    working["partner_ipu_offset"] = pd.to_numeric(working["partner_ipu_offset"], errors="coerce")
    if "partner_ipu_class" not in working.columns:
        working["partner_ipu_class"] = DEFAULT_PARTNER_IPU_CLASS

    working["latency_from_partner_offset_s"] = working["own_fpp_onset"] - working["partner_ipu_offset"]
    working["censor_time"] = working["own_fpp_onset"]
    working["episode_duration"] = working["censor_time"] - working["partner_ipu_onset"]
    working["episode_kind"] = "event_positive"
    working["event_observed"] = 1
    working["partner_role"] = "partner"
    if "partner_ipu_overlaps_fpp" not in working.columns:
        working["partner_ipu_overlaps_fpp"] = False
    if "partner_ipu_was_truncated" not in working.columns:
        working["partner_ipu_was_truncated"] = False
    if "episode_is_valid" not in working.columns:
        working["episode_is_valid"] = True
    if "invalid_reason" not in working.columns:
        working["invalid_reason"] = ""

    missing_anchor_mask = working["partner_ipu_onset"].isna()
    working.loc[missing_anchor_mask, "episode_is_valid"] = False
    working.loc[missing_anchor_mask & (working["invalid_reason"] == ""), "invalid_reason"] = "no_clean_preceding_partner_ipu"

    duration_mask = working["episode_duration"] < config.minimum_episode_duration_s
    working.loc[duration_mask, "episode_is_valid"] = False
    working.loc[duration_mask & (working["invalid_reason"] == ""), "invalid_reason"] = "episode_duration_below_minimum"

    order_mask = working["partner_ipu_onset"] >= working["censor_time"]
    working.loc[order_mask, "episode_is_valid"] = False
    working.loc[order_mask & (working["invalid_reason"] == ""), "invalid_reason"] = "partner_ipu_onset_not_before_fpp"
    return working.reset_index(drop=True)


def _select_partner_ipu_for_event(
    *,
    dyad_id: str,
    run: str,
    participant_speaker: str,
    partner_speaker: str,
    fpp_onset: float,
    local_ipus: list[dict[str, object]],
    config: BehaviourHazardConfig,
) -> dict[str, object]:
    tolerance = config.partner_offset_fpp_tolerance_s
    clean_candidates = [
        row
        for row in local_ipus
        if float(row["partner_ipu_onset"]) < fpp_onset
        and float(row["partner_ipu_offset"]) <= fpp_onset + tolerance
    ]
    if clean_candidates:
        chosen = clean_candidates[-1]
        return _apply_overlap_strategy(
            dyad_id=dyad_id,
            run=run,
            participant_speaker=participant_speaker,
            partner_speaker=partner_speaker,
            fpp_onset=fpp_onset,
            partner_ipu_onset=float(chosen["partner_ipu_onset"]),
            partner_ipu_offset=float(chosen["partner_ipu_offset"]),
            partner_ipu_class=str(chosen.get("partner_ipu_class", DEFAULT_PARTNER_IPU_CLASS)),
            config=config,
        )
    overlapping_candidates = [row for row in local_ipus if float(row["partner_ipu_onset"]) < fpp_onset]
    if not overlapping_candidates:
        return _invalid_partner_assignment(
            dyad_id=dyad_id,
            run=run,
            participant_speaker=participant_speaker,
            partner_speaker=partner_speaker,
            fpp_onset=fpp_onset,
            invalid_reason="no_clean_preceding_partner_ipu",
        )
    chosen = overlapping_candidates[-1]
    return _apply_overlap_strategy(
        dyad_id=dyad_id,
        run=run,
        participant_speaker=participant_speaker,
        partner_speaker=partner_speaker,
        fpp_onset=fpp_onset,
        partner_ipu_onset=float(chosen["partner_ipu_onset"]),
        partner_ipu_offset=float(chosen["partner_ipu_offset"]),
        partner_ipu_class=str(chosen.get("partner_ipu_class", DEFAULT_PARTNER_IPU_CLASS)),
        config=config,
    )


def _apply_overlap_strategy(
    *,
    dyad_id: str,
    run: str,
    participant_speaker: str,
    partner_speaker: str,
    fpp_onset: float,
    partner_ipu_onset: float | None,
    partner_ipu_offset: float | None,
    partner_ipu_class: str,
    config: BehaviourHazardConfig,
) -> dict[str, object]:
    tolerance = config.partner_offset_fpp_tolerance_s
    overlap_detected = (
        partner_ipu_offset is not None
        and np.isfinite(partner_ipu_offset)
        and partner_ipu_offset > fpp_onset + tolerance
    )
    if overlap_detected and config.require_partner_offset_before_fpp:
        if config.overlapping_episode_strategy == "exclude":
            return _invalid_partner_assignment(
                dyad_id=dyad_id,
                run=run,
                participant_speaker=participant_speaker,
                partner_speaker=partner_speaker,
                fpp_onset=fpp_onset,
                partner_ipu_onset=partner_ipu_onset,
                partner_ipu_offset=partner_ipu_offset,
                partner_ipu_class=partner_ipu_class,
                invalid_reason="partner_ipu_overlaps_fpp",
                partner_ipu_overlaps_fpp=True,
            )
        if config.overlapping_episode_strategy == "truncate":
            partner_ipu_offset = fpp_onset
            return {
                "dyad_id": dyad_id,
                "run": run,
                "participant_speaker": participant_speaker,
                "partner_speaker": partner_speaker,
                "fpp_onset": fpp_onset,
                "partner_ipu_onset": partner_ipu_onset,
                "partner_ipu_offset": partner_ipu_offset,
                "partner_ipu_class": partner_ipu_class,
                "partner_ipu_overlaps_fpp": True,
                "partner_ipu_was_truncated": True,
                "episode_is_valid": True,
                "invalid_reason": "",
            }
        if config.overlapping_episode_strategy == "keep":
            return {
                "dyad_id": dyad_id,
                "run": run,
                "participant_speaker": participant_speaker,
                "partner_speaker": partner_speaker,
                "fpp_onset": fpp_onset,
                "partner_ipu_onset": partner_ipu_onset,
                "partner_ipu_offset": partner_ipu_offset,
                "partner_ipu_class": partner_ipu_class,
                "partner_ipu_overlaps_fpp": True,
                "partner_ipu_was_truncated": False,
                "episode_is_valid": True,
                "invalid_reason": "",
            }
    return {
        "dyad_id": dyad_id,
        "run": run,
        "participant_speaker": participant_speaker,
        "partner_speaker": partner_speaker,
        "fpp_onset": fpp_onset,
        "partner_ipu_onset": partner_ipu_onset,
        "partner_ipu_offset": partner_ipu_offset,
        "partner_ipu_class": partner_ipu_class,
        "partner_ipu_overlaps_fpp": bool(overlap_detected),
        "partner_ipu_was_truncated": False,
        "episode_is_valid": True,
        "invalid_reason": "",
    }


def _invalid_partner_assignment(
    *,
    dyad_id: str,
    run: str,
    participant_speaker: str,
    partner_speaker: str,
    fpp_onset: float,
    invalid_reason: str,
    partner_ipu_onset: float | None = np.nan,
    partner_ipu_offset: float | None = np.nan,
    partner_ipu_class: str = DEFAULT_PARTNER_IPU_CLASS,
    partner_ipu_overlaps_fpp: bool = False,
) -> dict[str, object]:
    return {
        "dyad_id": dyad_id,
        "run": run,
        "participant_speaker": participant_speaker,
        "partner_speaker": partner_speaker,
        "fpp_onset": fpp_onset,
        "partner_ipu_onset": partner_ipu_onset,
        "partner_ipu_offset": partner_ipu_offset,
        "partner_ipu_class": partner_ipu_class,
        "partner_ipu_overlaps_fpp": partner_ipu_overlaps_fpp,
        "partner_ipu_was_truncated": False,
        "episode_is_valid": False,
        "invalid_reason": invalid_reason,
    }


def _build_ipu_rows_from_tokens(
    *,
    sorted_tokens: pd.DataFrame,
    group_key: tuple[str, str, str],
    gap_threshold_s: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if sorted_tokens.empty:
        return rows
    dyad_id, run, speaker = group_key
    ipu_index = -1
    current_start = 0.0
    current_end = 0.0
    token_indices: list[int] = []
    for row_index, token_row in sorted_tokens.iterrows():
        onset = float(token_row["onset"])
        offset = float(token_row["offset"])
        if ipu_index < 0 or onset - current_end > gap_threshold_s:
            if token_indices:
                rows.append(
                    _finish_ipu_row(
                        group_key=group_key,
                        ipu_index=ipu_index,
                        start=current_start,
                        end=current_end,
                        sorted_tokens=sorted_tokens,
                        token_indices=token_indices,
                    )
                )
            ipu_index += 1
            current_start = onset
            current_end = offset
            token_indices = [row_index]
        else:
            current_end = max(current_end, offset)
            token_indices.append(row_index)
    if token_indices:
        rows.append(
            _finish_ipu_row(
                group_key=group_key,
                ipu_index=ipu_index,
                start=current_start,
                end=current_end,
                sorted_tokens=sorted_tokens,
                token_indices=token_indices,
            )
        )
    return rows


def _finish_ipu_row(
    *,
    group_key: tuple[str, str, str],
    ipu_index: int,
    start: float,
    end: float,
    sorted_tokens: pd.DataFrame,
    token_indices: list[int],
) -> dict[str, object]:
    token_slice = sorted_tokens.iloc[token_indices]
    dyad_id, run, speaker = group_key
    first_row = token_slice.iloc[0]
    return {
        "dyad_id": dyad_id,
        "run": run,
        "speaker": speaker,
        "partner_ipu_id": f"{dyad_id}|run-{run}|{speaker}|ipu-{ipu_index:04d}",
        "partner_ipu_onset": float(start),
        "partner_ipu_offset": float(end),
        "partner_ipu_class": str(first_row.get("source_interval_id", DEFAULT_PARTNER_IPU_CLASS)),
        "n_tokens_total": int(len(token_slice)),
    }


def _compute_episode_window_end(
    *,
    episode_start: float,
    next_partner_ipu_onset: float | None,
    run_end: float | None,
    config: BehaviourHazardConfig,
) -> tuple[float, str]:
    candidates = [(episode_start + config.max_followup_s, "max_followup")]
    if next_partner_ipu_onset is not None and np.isfinite(next_partner_ipu_onset):
        candidates.append((float(next_partner_ipu_onset), "next_partner_ipu"))
    if run_end is not None and np.isfinite(run_end):
        candidates.append((float(run_end), "run_end"))
    window_end, reason = min(candidates, key=lambda item: item[0])
    if window_end <= episode_start:
        raise ValueError(f"Episode window end must be after episode start, found {window_end} <= {episode_start}.")
    return float(window_end), reason


def _resolve_fpp_speaker_column(events_table: pd.DataFrame) -> str:
    for candidate in ("participant_speaker", "fpp_speaker", "speaker"):
        if candidate in events_table.columns:
            return candidate
    available = ", ".join(sorted(str(column) for column in events_table.columns))
    raise ValueError(f"Could not resolve FPP speaker column from events table. Available columns: {available}")


def _normalize_speaker_label(
    value: object,
    *,
    event_speakers: set[str] | None = None,
    token_speakers: set[str] | None = None,
    dyad_run_example: str | None = None,
) -> str:
    speaker = str(value).strip().upper()
    if speaker in SUPPORTED_SPEAKERS:
        return speaker
    if event_speakers is not None and token_speakers is not None:
        raise ValueError(
            "speaker labels could not be mapped onto A/B speakers for behavioural hazard episodes. "
            f"Unique event speaker labels: {sorted(event_speakers)}. "
            f"Unique token speaker labels: {sorted(token_speakers)}. "
            f"Example dyad/run: {dyad_run_example or 'unknown'}. "
            "If events use subject ids rather than A/B labels, a speaker-mapping option is required."
        )
    raise ValueError(f"Expected speaker label A or B, found {value!r}.")


def _normalize_run_label(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:run[-_ ]*)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    return text


def _normalize_dyad_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:dyad[-_ ]*)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"dyad-{int(match.group(1)):03d}"
    return text


def _first_dyad_run_example(table: pd.DataFrame) -> str:
    if table.empty:
        return "unknown"
    row = table.iloc[0]
    return f"{row.get('dyad_id', 'unknown')}|run-{row.get('run', 'unknown')}"


def _maybe_float(value: object) -> float | None:
    return float(value) if value is not None and pd.notna(value) else None


def _safe_proportion(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)
