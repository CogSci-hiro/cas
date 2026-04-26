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
ACTIVE_LAGGED_INFORMATION_FEATURES = (
    "information_rate",
    "prop_expected_cumulative_info",
)
INFORMATION_TIMING_THRESHOLD_MAP = {
    0.25: "info_t25_s",
    0.50: "info_t50_s",
    0.75: "info_t75_s",
    0.90: "info_t90_s",
}


@dataclass(frozen=True, slots=True)
class ZScoreResult:
    """Z-scored table plus scaling metadata."""

    table: pd.DataFrame
    scaling: dict[str, dict[str, float]]
    warnings: list[str]


def compute_information_features_for_episode(
    *,
    episode_rows: pd.DataFrame,
    episode: pd.Series,
    surprisal_table: pd.DataFrame,
    expected_total_info_by_group: dict[str, float],
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Compute information features for a single episode."""

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
    """Return partner-IPU tokens with availability times for one episode."""

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
    """Estimate expected total information by configurable group."""

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
    if group_column not in episodes_with_defaults.columns:
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


def compute_lagged_feature_name(feature_name: str, lag_ms: int) -> str:
    """Return the canonical lagged feature name."""

    return f"{feature_name}_lag_{int(lag_ms)}ms"


def add_lagged_information_features(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add causal within-episode lagged information features."""

    LOGGER.info("Adding lagged information features for lags=%s.", list(config.lag_grid_ms))
    table = riskset_table.copy()
    warnings_list: list[str] = []
    feature_names_created = [
        compute_lagged_feature_name(feature_name, lag_ms)
        for feature_name in ACTIVE_LAGGED_INFORMATION_FEATURES
        for lag_ms in config.lag_grid_ms
    ]
    for feature_name in feature_names_created:
        table[feature_name] = float(config.lagged_feature_fill_value)

    grouped_episodes = list(table.groupby("episode_id", sort=False))
    for _, episode_rows in grouped_episodes:
        episode_rows = episode_rows.sort_values(["time_from_partner_onset", "bin_index"], kind="mergesort")
        episode_index = episode_rows.index
        times = pd.to_numeric(episode_rows["time_from_partner_onset"], errors="coerce").to_numpy(dtype=float)
        for base_feature in ACTIVE_LAGGED_INFORMATION_FEATURES:
            values = pd.to_numeric(episode_rows[base_feature], errors="coerce").to_numpy(dtype=float)
            for lag_ms in config.lag_grid_ms:
                lagged_values = _compute_episode_lagged_values(
                    times=times,
                    values=values,
                    lag_ms=int(lag_ms),
                    fill_value=float(config.lagged_feature_fill_value),
                )
                table.loc[episode_index, compute_lagged_feature_name(base_feature, lag_ms)] = lagged_values

    qc_payload = build_lagged_feature_qc(table, config=config, warnings_list=warnings_list)
    qc_payload["feature_names_created"] = feature_names_created
    return table, qc_payload


def compute_information_timing_summaries(
    *,
    episodes_table: pd.DataFrame,
    surprisal_table: pd.DataFrame,
    config: BehaviourHazardConfig,
) -> pd.DataFrame:
    """Compute episode-level information timing summaries for partner IPUs."""

    rows: list[dict[str, object]] = []
    episode_records = list(episodes_table.to_dict("records"))
    for episode_record in episode_records:
        episode = pd.Series(episode_record)
        tokens = extract_partner_ipu_tokens(
            episode=episode,
            surprisal_table=surprisal_table,
            config=config,
        )
        rows.append(_compute_information_timing_summary_row(episode=episode, tokens=tokens))
    return pd.DataFrame(rows)


def build_lagged_feature_qc(
    riskset_table: pd.DataFrame,
    *,
    config: BehaviourHazardConfig,
    warnings_list: list[str] | None = None,
) -> dict[str, object]:
    """Build QC metadata for lagged feature generation."""

    warnings_payload = list(warnings_list or [])
    feature_names_created = [
        compute_lagged_feature_name(feature_name, lag_ms)
        for feature_name in ACTIVE_LAGGED_INFORMATION_FEATURES
        for lag_ms in config.lag_grid_ms
        if compute_lagged_feature_name(feature_name, lag_ms) in riskset_table.columns
    ]
    max_abs_difference_lag0_vs_unlagged: dict[str, float | None] = {}
    for feature_name in ACTIVE_LAGGED_INFORMATION_FEATURES:
        lagged_feature = compute_lagged_feature_name(feature_name, 0)
        if lagged_feature not in riskset_table.columns:
            max_abs_difference_lag0_vs_unlagged[feature_name] = None
            continue
        difference = np.abs(
            pd.to_numeric(riskset_table[lagged_feature], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(riskset_table[feature_name], errors="coerce").to_numpy(dtype=float)
        )
        finite_difference = difference[np.isfinite(difference)]
        max_difference = float(finite_difference.max()) if finite_difference.size else None
        max_abs_difference_lag0_vs_unlagged[feature_name] = max_difference
        if max_difference is not None and max_difference > 1.0e-8:
            warnings_payload.append(
                f"Lag-0 feature {lagged_feature} did not match {feature_name} within tolerance; max abs difference={max_difference:.6g}."
            )

    n_missing_by_feature = {
        feature_name: int(pd.to_numeric(riskset_table[feature_name], errors="coerce").isna().sum())
        for feature_name in feature_names_created
    }
    constant_features = []
    for feature_name in feature_names_created:
        numeric = pd.to_numeric(riskset_table[feature_name], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            constant_features.append(feature_name)
            continue
        if np.isclose(float(finite.max()), float(finite.min())):
            constant_features.append(feature_name)

    return {
        "lag_grid_ms": [int(lag_ms) for lag_ms in config.lag_grid_ms],
        "n_rows": int(len(riskset_table)),
        "n_episodes": int(riskset_table["episode_id"].nunique()),
        "feature_names_created": feature_names_created,
        "n_missing_by_lagged_feature": n_missing_by_feature,
        "n_constant_lagged_features": int(len(constant_features)),
        "constant_lagged_features": constant_features,
        "fill_value_used": float(config.lagged_feature_fill_value),
        "max_abs_difference_lag0_vs_unlagged": max_abs_difference_lag0_vs_unlagged,
        "warnings": warnings_payload,
    }


def zscore_predictors(
    riskset_table: pd.DataFrame,
    *,
    predictors: list[str] | tuple[str, ...] | None = None,
) -> ZScoreResult:
    """Z-score the main non-spline continuous predictors."""

    LOGGER.info("Z-scoring continuous predictors for %d risk-set rows.", len(riskset_table))
    selected_predictors = list(predictors) if predictors is not None else list(_discover_predictors_for_zscoring(riskset_table))
    scaling: dict[str, dict[str, float]] = {}
    warnings_list: list[str] = []
    table = riskset_table.copy()
    for predictor in selected_predictors:
        values = pd.to_numeric(table[predictor], errors="coerce")
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        scaling[predictor] = {"mean": mean_value, "std": std_value}
        if not np.isfinite(std_value) or std_value <= 0.0:
            table[f"z_{predictor}"] = 0.0
            warning = f"Predictor {predictor} had zero or undefined variance during z-scoring; assigned z_{predictor}=0."
            LOGGER.warning(warning)
            warnings_list.append(warning)
        else:
            table[f"z_{predictor}"] = (values - mean_value) / std_value
    return ZScoreResult(table=table, scaling=scaling, warnings=warnings_list)


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


def _discover_predictors_for_zscoring(riskset_table: pd.DataFrame) -> tuple[str, ...]:
    predictors = [
        feature_name for feature_name in ACTIVE_LAGGED_INFORMATION_FEATURES if feature_name in riskset_table.columns
    ]
    predictors.extend(
        column_name
        for column_name in riskset_table.columns
        if any(
            column_name.startswith(f"{feature_name}_lag_")
            for feature_name in ACTIVE_LAGGED_INFORMATION_FEATURES
        )
    )
    return tuple(dict.fromkeys(predictors))


def _compute_episode_lagged_values(
    *,
    times: np.ndarray,
    values: np.ndarray,
    lag_ms: int,
    fill_value: float,
) -> np.ndarray:
    lag_s = float(lag_ms) / 1000.0
    query_times = times - lag_s
    lagged_values = np.full(times.shape[0], float(fill_value), dtype=float)
    for row_index, query_time in enumerate(query_times):
        if not np.isfinite(query_time) or query_time < 0.0:
            continue
        source_index = int(np.searchsorted(times, query_time + EPSILON, side="right") - 1)
        if source_index >= 0:
            lagged_values[row_index] = values[source_index]
    return lagged_values


def _compute_information_timing_summary_row(*, episode: pd.Series, tokens: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_id": str(episode["episode_id"]),
        "partner_ipu_id": str(episode.get("partner_ipu_id", f"{episode['episode_id']}|anchor")),
        "dyad_id": str(episode["dyad_id"]),
        "run": str(episode["run"]),
    }
    surprisal = pd.to_numeric(tokens.get("surprisal"), errors="coerce")
    availability = pd.to_numeric(tokens.get("availability_time_relative_to_ipu_onset"), errors="coerce")
    valid = tokens.loc[np.isfinite(surprisal) & np.isfinite(availability)].copy()
    if valid.empty:
        row["actual_total_info"] = np.nan
        row["info_centroid_s"] = np.nan
        for column_name in INFORMATION_TIMING_THRESHOLD_MAP.values():
            row[column_name] = np.nan
        row["info_prop_by_500ms"] = np.nan
        row["info_prop_by_1000ms"] = np.nan
        return row

    valid = valid.sort_values("availability_time_relative_to_ipu_onset", kind="mergesort").reset_index(drop=True)
    surprisal_values = pd.to_numeric(valid["surprisal"], errors="coerce").to_numpy(dtype=float)
    availability_values = pd.to_numeric(
        valid["availability_time_relative_to_ipu_onset"],
        errors="coerce",
    ).to_numpy(dtype=float)
    total_info = float(np.nansum(surprisal_values))
    row["actual_total_info"] = total_info
    if not np.isfinite(total_info) or total_info <= 0.0:
        row["info_centroid_s"] = np.nan
        for column_name in INFORMATION_TIMING_THRESHOLD_MAP.values():
            row[column_name] = np.nan
        row["info_prop_by_500ms"] = np.nan
        row["info_prop_by_1000ms"] = np.nan
        return row

    row["info_centroid_s"] = float(np.sum(surprisal_values * availability_values) / total_info)
    cumulative_info = np.cumsum(surprisal_values)
    cumulative_prop = cumulative_info / total_info
    for threshold, column_name in INFORMATION_TIMING_THRESHOLD_MAP.items():
        crossing_indices = np.flatnonzero(cumulative_prop >= threshold - EPSILON)
        row[column_name] = float(availability_values[crossing_indices[0]]) if crossing_indices.size else np.nan
    row["info_prop_by_500ms"] = _lookup_causal_cumulative_prop(
        availability_values=availability_values,
        cumulative_prop=cumulative_prop,
        query_time_s=0.5,
    )
    row["info_prop_by_1000ms"] = _lookup_causal_cumulative_prop(
        availability_values=availability_values,
        cumulative_prop=cumulative_prop,
        query_time_s=1.0,
    )
    return row


def _lookup_causal_cumulative_prop(
    *,
    availability_values: np.ndarray,
    cumulative_prop: np.ndarray,
    query_time_s: float,
) -> float:
    source_index = int(np.searchsorted(availability_values, query_time_s, side="right") - 1)
    if source_index < 0:
        return 0.0
    return float(cumulative_prop[source_index])
