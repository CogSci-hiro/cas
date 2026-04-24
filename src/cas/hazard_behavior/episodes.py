"""Episode construction for behavioural FPP hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.progress import progress_iterable

DEFAULT_PARTNER_IPU_CLASS = "unknown"
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


def infer_partner_speaker(participant_speaker: str) -> str:
    """Infer the partner speaker in a two-speaker dyad."""

    speaker = str(participant_speaker).strip().upper()
    if speaker == "A":
        return "B"
    if speaker == "B":
        return "A"
    raise ValueError(f"Expected participant speaker A or B, found {participant_speaker!r}.")


def compute_ipu_token_table(
    surprisal_table: pd.DataFrame,
    *,
    gap_threshold_s: float,
) -> pd.DataFrame:
    """Group consecutive tokens into IPU-like intervals.

    Usage example
    -------------
        ipu_table = compute_ipu_token_table(tokens, gap_threshold_s=0.3)
    """

    rows: list[dict[str, object]] = []
    group_columns = ["dyad_id", "run", "speaker"]
    for group_key, speaker_tokens in surprisal_table.groupby(group_columns, sort=False):
        sorted_tokens = speaker_tokens.sort_values(["onset", "offset"], kind="mergesort").reset_index(drop=True)
        ipu_rows = _build_ipu_rows_from_tokens(
            sorted_tokens=sorted_tokens,
            group_key=(str(group_key[0]), str(group_key[1]), str(group_key[2])),
            gap_threshold_s=gap_threshold_s,
        )
        rows.extend(ipu_rows)
    return pd.DataFrame(rows)


def build_event_positive_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> EpisodeBuildResult:
    """Build and validate event-positive episodes anchored at preceding partner speech."""

    warnings: list[str] = []
    working = events_table.copy()
    working["dyad_id"] = working["dyad_id"].astype(str)
    working["run"] = working["run"].astype(str)
    working["participant_speaker"] = working["participant_speaker"].astype(str)
    working["fpp_onset"] = pd.to_numeric(working["fpp_onset"], errors="coerce")
    if "partner_speaker" not in working.columns:
        working["partner_speaker"] = working["participant_speaker"].map(infer_partner_speaker)
        warnings.append("Partner speaker was inferred from participant speaker labels.")
    else:
        working["partner_speaker"] = working["partner_speaker"].astype(str)

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
    )


def build_censored_episodes(
    *,
    events_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    positive_episodes: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Build optional censored episodes from partner IPUs not followed by own FPP."""

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

    fpp_rows = events_table.loc[:, ["dyad_id", "run", "participant_speaker", "fpp_onset"]].copy()
    fpp_rows["fpp_onset"] = pd.to_numeric(fpp_rows["fpp_onset"], errors="coerce")
    episodes: list[dict[str, object]] = []
    for _, ipu_row in candidate_ipus.iterrows():
        partner_speaker = str(ipu_row["partner_speaker"])
        participant_speaker = infer_partner_speaker(partner_speaker)
        matching_events = fpp_rows.loc[
            (fpp_rows["dyad_id"].astype(str) == str(ipu_row["dyad_id"]))
            & (fpp_rows["run"].astype(str) == str(ipu_row["run"]))
            & (fpp_rows["participant_speaker"].astype(str) == participant_speaker)
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
        dyad_id = str(event_row["dyad_id"])
        run = str(event_row["run"])
        participant_speaker = str(event_row["participant_speaker"])
        partner_speaker = str(event_row["partner_speaker"])
        fpp_onset = float(event_row["fpp_onset"])
        partner_tokens = surprisal_table.loc[
            (surprisal_table["dyad_id"].astype(str) == dyad_id)
            & (surprisal_table["run"].astype(str) == run)
            & (surprisal_table["speaker"].astype(str) == partner_speaker)
            & (pd.to_numeric(surprisal_table["onset"], errors="coerce") < fpp_onset)
        ].copy()
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
    """Compute summary QC metrics for episode validation."""

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


def _safe_proportion(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)
