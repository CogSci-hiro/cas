"""Episode helpers for the behavioral hazard pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cas.behavior._legacy_support import (
    BehaviourHazardConfig,
    build_censored_episodes,
    build_event_positive_episodes,
    project_behavior_final_events,
    read_events_table,
    read_surprisal_tables,
    resolve_surprisal_paths,
)
from cas.behavior.config import BehaviorHazardConfig

LOGGER = logging.getLogger(__name__)


def load_behavior_inputs(config: BehaviorHazardConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_path = Path(str(config.inputs["events_csv"]))
    surprisal_glob = str(config.inputs["surprisal_tsv"])
    surprisal_paths = tuple(resolve_surprisal_paths(surprisal_glob))
    events, _ = read_events_table(events_path)
    surprisal, _ = read_surprisal_tables(surprisal_paths, unmatched_surprisal_strategy="drop")
    return events, surprisal


def build_anchor_episodes(
    config: BehaviorHazardConfig,
    *,
    anchor: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, surprisal = load_behavior_inputs(config)
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s inputs: %d events rows, %d surprisal rows",
            anchor.upper(),
            len(events),
            len(surprisal),
        )
    projected_events = project_behavior_final_events(events, anchor=anchor.lower())
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s projected events: %d rows",
            anchor.upper(),
            len(projected_events),
        )
    legacy_config = BehaviourHazardConfig(
        events_path=Path(str(config.inputs["events_csv"])),
        surprisal_paths=tuple(resolve_surprisal_paths(str(config.inputs["surprisal_tsv"]))),
        out_dir=config.paths.hazard_root,
        bin_size_s=float(config.bin_size_ms) / 1000.0,
        include_censored=True,
        overwrite=True,
        save_riskset=False,
        target_fpp_label_prefix="FPP_" if anchor.upper() == "FPP" else "SPP_",
    )
    positive = build_event_positive_episodes(
        events_table=projected_events,
        surprisal_table=surprisal,
        config=legacy_config,
    )
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s positive episodes: %d episodes, %d warnings",
            anchor.upper(),
            len(positive.episodes),
            len(positive.warnings),
        )
    episodes = positive.episodes.copy()
    censored = build_censored_episodes(
        events_table=projected_events,
        surprisal_table=surprisal,
        positive_episodes=positive.episodes,
        config=legacy_config,
    )
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s censored episodes: %d episodes",
            anchor.upper(),
            len(censored),
        )
    if not censored.empty:
        episodes = pd.concat([episodes, censored], ignore_index=True, sort=False)
    if verbose:
        LOGGER.info(
            "[behavior hazard] %s combined episodes: %d episodes",
            anchor.upper(),
            len(episodes),
        )
    return episodes, surprisal
