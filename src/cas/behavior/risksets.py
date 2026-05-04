"""Risk-set construction for the behavioral hazard pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cas.behavior._legacy_support import (
    BehaviourHazardConfig,
    add_information_features_to_riskset,
    build_discrete_time_riskset,
    normalize_final_riskset,
)
from cas.behavior.config import BehaviorHazardConfig
from cas.behavior.episodes import build_anchor_episodes

LOGGER = logging.getLogger(__name__)


def _legacy_config(config: BehaviorHazardConfig) -> BehaviourHazardConfig:
    return BehaviourHazardConfig(
        events_path=Path(str(config.inputs["events_csv"])),
        surprisal_paths=(),
        out_dir=config.paths.hazard_root,
        bin_size_s=float(config.bin_size_ms) / 1000.0,
        include_censored=True,
        overwrite=True,
        save_riskset=False,
    )


def _numeric_run(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(r"(\d+)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(numeric.median())
    return pd.Series(np.arange(len(values), dtype=float), index=values.index)


def build_anchor_riskset(
    config: BehaviorHazardConfig,
    *,
    anchor: str,
    verbose: bool = False,
) -> pd.DataFrame:
    episodes, surprisal = build_anchor_episodes(config, anchor=anchor, verbose=verbose)
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s risk-set inputs: %d episodes, %d surprisal rows",
            anchor.upper(),
            len(episodes),
            len(surprisal),
        )
    legacy_config = _legacy_config(config)
    riskset_result = build_discrete_time_riskset(episodes, config=legacy_config)
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s discrete-time risk set: %d rows, %d events, %d warnings",
            anchor.upper(),
            len(riskset_result.riskset_table),
            int(pd.to_numeric(riskset_result.riskset_table.get("event", 0), errors="coerce").fillna(0).sum()),
            len(riskset_result.warnings),
        )
    riskset_with_info, _ = add_information_features_to_riskset(
        riskset_table=riskset_result.riskset_table,
        episodes_table=episodes,
        surprisal_table=surprisal,
        config=legacy_config,
    )
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s information-enriched risk set: %d rows",
            anchor.upper(),
            len(riskset_with_info),
        )
    normalized = normalize_final_riskset(riskset_with_info, anchor.lower())
    normalized["anchor_type"] = anchor.upper()
    normalized["event"] = normalized["event_bin"].astype(int)
    normalized["subject"] = normalized["participant_speaker_id"].astype(str)
    normalized["run"] = _numeric_run(normalized["run_id"])
    normalized["time_within_run"] = pd.to_numeric(normalized["bin_start_s"], errors="coerce")
    episode_duration = (
        normalized.groupby("episode_id", sort=False)["bin_end_s"].transform("max")
        - normalized.groupby("episode_id", sort=False)["bin_start_s"].transform("min")
        + (float(config.bin_size_ms) / 1000.0)
    )
    normalized["planned_response_duration"] = pd.to_numeric(episode_duration, errors="coerce")
    normalized["prop_expected_cum_info"] = pd.to_numeric(normalized["prop_expected_cumulative_info"], errors="coerce")
    normalized["planned_response_total_information"] = pd.to_numeric(
        normalized.get("expected_total_info", normalized["prop_expected_cum_info"]),
        errors="coerce",
    )
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s final risk set: %d rows, %d episodes, %d subjects",
            anchor.upper(),
            len(normalized),
            int(normalized["episode_id"].nunique()) if "episode_id" in normalized.columns else 0,
            int(normalized["subject"].nunique()) if "subject" in normalized.columns else 0,
        )
    return normalized
