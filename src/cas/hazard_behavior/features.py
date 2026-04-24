"""Information-feature construction for behavioural hazard analysis."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.progress import progress_iterable

EPSILON = 1.0e-12
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZScoreResult:
    """Z-scored table plus scaling metadata."""

    table: pd.DataFrame
    scaling: dict[str, dict[str, float]]


def compute_information_features_for_episode(
    *,
    episode_rows: pd.DataFrame,
    episode: pd.Series,
    surprisal_table: pd.DataFrame,
    expected_total_info_by_group: dict[str, float],
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Compute information features for a single episode."""

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
    rows["prop_actual_cumulative_info"] = _safe_ratio(rows["cumulative_info"], actual_total_info)
    rows["prop_expected_cumulative_info"] = _safe_ratio(rows["cumulative_info"], expected_total_info)
    if config.clip_proportions:
        low, high = config.clip_range
        rows["prop_actual_cumulative_info"] = rows["prop_actual_cumulative_info"].clip(low, high)
        rows["prop_expected_cumulative_info"] = rows["prop_expected_cumulative_info"].clip(low, high)
    rows["expected_info_group_value"] = group_value
    rows["feature_missing_actual_total"] = bool(not np.isfinite(actual_total_info) or actual_total_info <= 0.0)
    rows["feature_missing_expected_total"] = bool(not np.isfinite(expected_total_info) or expected_total_info <= 0.0)
    return rows


def compute_expected_total_information(
    *,
    surprisal_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> dict[str, float]:
    """Estimate expected total information by configurable group."""

    episode_ipus = episodes_table.loc[
        :,
        ["dyad_id", "run", "partner_speaker", "partner_ipu_onset", "partner_ipu_offset", "partner_ipu_class", "partner_role"],
    ].drop_duplicates()
    if episode_ipus.empty:
        return {"global": np.nan}

    LOGGER.info(
        "Estimating expected total information using grouping=%s across %d partner IPUs.",
        config.expected_info_group,
        len(episode_ipus),
    )
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
    if group_column not in episodes_table.columns:
        return result
    grouped = ipu_totals_table.groupby(group_column, dropna=False)["actual_total_info"].mean()
    for group_value, mean_value in grouped.items():
        result[str(group_value)] = float(mean_value)
    return result


def add_information_features_to_riskset(
    *,
    riskset_table: pd.DataFrame,
    episodes_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach information features to the full risk set."""

    LOGGER.info(
        "Computing information features for %d episodes and %d risk-set rows.",
        int(episodes_table["episode_id"].nunique()),
        len(riskset_table),
    )
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


def zscore_predictors(riskset_table: pd.DataFrame) -> ZScoreResult:
    """Z-score the main non-spline continuous predictors."""

    LOGGER.info("Z-scoring continuous predictors for %d risk-set rows.", len(riskset_table))
    predictors = [
        "information_rate",
        "cumulative_info",
        "prop_actual_cumulative_info",
        "prop_expected_cumulative_info",
    ]
    scaling: dict[str, dict[str, float]] = {}
    table = riskset_table.copy()
    for predictor in predictors:
        values = pd.to_numeric(table[predictor], errors="coerce")
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        scaling[predictor] = {"mean": mean_value, "std": std_value}
        if not np.isfinite(std_value) or std_value <= 0.0:
            table[f"z_{predictor}"] = 0.0
        else:
            table[f"z_{predictor}"] = (values - mean_value) / std_value
    return ZScoreResult(table=table, scaling=scaling)


def _resolve_expected_info_group_value(episode: pd.Series, expected_info_group: str) -> str:
    if expected_info_group == "global":
        return "global"
    if expected_info_group == "partner_role":
        return str(episode.get("partner_role", "partner"))
    return str(episode.get("partner_ipu_class", "unknown"))


def _safe_ratio(numerator: pd.Series, denominator: float) -> pd.Series:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return pd.Series(np.nan, index=numerator.index, dtype=float)
    return numerator.astype(float) / float(denominator)


def _compute_alignment_ok_fraction(tokens: pd.DataFrame) -> float:
    if tokens.empty or "alignment_status" not in tokens.columns:
        return 1.0
    return float((tokens["alignment_status"].astype(str) == "ok").mean())
