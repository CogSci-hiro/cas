from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_event_positive_episodes
from cas.hazard_behavior.riskset import build_discrete_time_riskset


def test_clean_preceding_partner_ipu_is_valid(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens([("A", 0.0, 0.1), ("A", 0.3, 0.1), ("A", 0.7, 0.1)]),
        config=config,
    )

    episode = result.episodes.iloc[0]
    assert np.isclose(episode["partner_ipu_offset"], 0.8)
    assert np.isclose(episode["event_latency_from_partner_offset_s"], 0.4)
    assert episode["event_phase"] == "post_partner_ipu"
    assert bool(episode["event_observed"]) is True


def test_future_partner_tokens_do_not_extend_selected_previous_ipu(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens(
            [("A", 0.0, 0.1), ("A", 0.3, 0.1), ("A", 0.7, 0.1), ("A", 1.3, 0.1)]
        ),
        config=config,
    )

    episode = result.episodes.iloc[0]
    assert np.isclose(episode["partner_ipu_offset"], 0.8)
    assert np.isclose(episode["event_latency_from_partner_offset_s"], 0.4)
    assert episode["event_phase"] == "post_partner_ipu"
    assert bool(episode["event_observed"]) is True


def test_fpp_during_partner_ipu_is_valid_by_default(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens([("A", 1.0, 0.30)]),
        config=config,
    )

    episode = result.episodes.iloc[0]
    assert np.isclose(episode["partner_ipu_offset"], 1.3)
    assert episode["event_latency_from_partner_offset_s"] < 0.0
    assert episode["event_phase"] == "during_partner_ipu"
    assert bool(episode["event_observed"]) is True


def test_truncate_strategy_does_not_change_valid_during_ipu_event(tmp_path: Path) -> None:
    config = _build_config(tmp_path, overlapping_episode_strategy="truncate")

    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens([("A", 1.0, 0.30)]),
        config=config,
    )

    episode = result.episodes.iloc[0]
    assert np.isclose(episode["partner_ipu_offset"], 1.3)
    assert episode["event_latency_from_partner_offset_s"] < 0.0
    assert episode["event_phase"] == "during_partner_ipu"


def test_keep_strategy_allows_negative_latency(tmp_path: Path) -> None:
    config = _build_config(tmp_path, overlapping_episode_strategy="keep")

    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens([("A", 1.0, 0.30)]),
        config=config,
    )

    episode = result.episodes.iloc[0]
    assert episode["event_latency_from_partner_offset_s"] < 0.0
    assert episode["event_phase"] == "during_partner_ipu"
    assert bool(episode["event_observed"]) is True


def test_final_riskset_retains_negative_offset_latency_when_event_occurs_during_ipu(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    result = build_event_positive_episodes(
        events_table=_toy_events(),
        surprisal_table=_toy_tokens([("A", 1.0, 0.30)]),
        config=config,
    )

    riskset_result = build_discrete_time_riskset(result.episodes, config=config)
    episode_summary = riskset_result.episode_summary
    assert (episode_summary["latency_from_partner_offset_s"] < 0.0).any()


def _build_config(tmp_path: Path, *, overlapping_episode_strategy: str = "exclude") -> BehaviourHazardConfig:
    return BehaviourHazardConfig(
        events_path=tmp_path / "unused.csv",
        surprisal_paths=(tmp_path / "unused.tsv",),
        out_dir=tmp_path / "out",
        episode_anchor="partner_ipu",
        include_censored=False,
        overlapping_episode_strategy=overlapping_episode_strategy,
    )


def _toy_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "participant_speaker": ["B"],
            "fpp_onset": [1.2],
            "fpp_offset": [1.5],
            "fpp_label": ["FPP_RFC_TAG"],
        }
    )


def _toy_tokens(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    data = []
    for index, (speaker, onset, duration) in enumerate(rows):
        data.append(
            {
                "dyad_id": "dyad-001",
                "run": "1",
                "speaker": speaker,
                "onset": onset,
                "duration": duration,
                "offset": onset + duration,
                "surprisal": float(index + 1),
                "alignment_status": "ok",
            }
        )
    return pd.DataFrame(data)
