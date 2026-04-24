from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import (
    build_partner_ipu_anchored_episodes,
    build_partner_ipus_from_tokens,
)
from cas.hazard_behavior.plots import summarize_observed_event_rate_by_time_bin
from cas.hazard_behavior.riskset import build_discrete_time_riskset


def test_ipu_construction_from_tokens() -> None:
    tokens = _toy_tokens(
        [
            ("dyad-001", "1", "A", 0.0, 0.1, 1.0),
            ("dyad-001", "1", "A", 0.3, 0.4, 1.0),
            ("dyad-001", "1", "A", 0.7, 0.8, 1.0),
            ("dyad-001", "1", "A", 2.0, 2.1, 1.0),
        ]
    )

    ipus = build_partner_ipus_from_tokens(tokens, gap_threshold_s=0.3)
    a_ipus = ipus.loc[ipus["speaker"] == "A"].reset_index(drop=True)

    assert len(a_ipus) == 2
    assert np.isclose(a_ipus.loc[0, "partner_ipu_onset"], 0.0)
    assert np.isclose(a_ipus.loc[0, "partner_ipu_offset"], 0.8)
    assert np.isclose(a_ipus.loc[1, "partner_ipu_onset"], 2.0)


def test_partner_ipu_with_following_fpp() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([("dyad-001", "1", "B", 1.2, 1.4, "FPP_TEST")]),
        surprisal_table=_toy_tokens(
            [
                ("dyad-001", "1", "A", 0.0, 0.8, 1.0),
                ("dyad-001", "1", "B", 1.2, 1.3, 1.0),
            ]
        ),
        config=config,
    )
    episode = result.episodes.iloc[0]
    riskset = build_discrete_time_riskset(result.episodes, config=config).riskset_table

    assert episode["partner_speaker"] == "A"
    assert episode["participant_speaker"] == "B"
    assert np.isclose(episode["episode_start"], 0.0)
    assert np.isclose(episode["own_fpp_onset"], 1.2)
    assert bool(episode["episode_has_event"]) is True
    assert np.isclose(episode["event_latency_from_partner_onset_s"], 1.2)
    assert np.isclose(episode["event_latency_from_partner_offset_s"], 0.4)
    assert episode["event_phase"] == "post_partner_ipu"
    assert int(riskset["event"].sum()) == 1
    assert int(riskset.loc[riskset["event"] == 1, "bin_index"].iloc[0]) == 24


def test_fpp_during_partner_ipu_is_valid() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([("dyad-001", "1", "B", 1.2, 1.4, "FPP_TEST")]),
        surprisal_table=_toy_tokens([("dyad-001", "1", "A", 0.0, 2.0, 1.0)]),
        config=config,
    )
    episode = result.episodes.iloc[0]
    riskset = build_discrete_time_riskset(result.episodes, config=config).riskset_table

    assert bool(episode["episode_has_event"]) is True
    assert np.isclose(episode["event_latency_from_partner_offset_s"], -0.8)
    assert episode["event_phase"] == "during_partner_ipu"
    assert int(riskset["event"].sum()) == 1


def test_censored_by_next_same_speaker_ipu() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([]),
        surprisal_table=_toy_tokens(
            [
                ("dyad-001", "1", "A", 0.0, 0.8, 1.0),
                ("dyad-001", "1", "A", 2.0, 2.5, 1.0),
            ]
        ),
        config=config,
    )

    first = result.episodes.iloc[0]
    riskset = build_discrete_time_riskset(result.episodes, config=config).riskset_table

    assert bool(first["episode_has_event"]) is False
    assert first["censor_reason"] == "next_partner_ipu"
    assert np.isclose(first["episode_end"], 2.0)
    assert int(riskset.loc[riskset["episode_id"] == first["episode_id"], "event"].sum()) == 0


def test_fpp_assigned_to_most_recent_partner_ipu() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([("dyad-001", "1", "B", 2.8, 3.0, "FPP_TEST")]),
        surprisal_table=_toy_tokens(
            [
                ("dyad-001", "1", "A", 0.0, 0.8, 1.0),
                ("dyad-001", "1", "A", 2.0, 2.5, 1.0),
            ]
        ),
        config=config,
    )

    assert result.episodes.iloc[0]["censor_reason"] == "next_partner_ipu"
    assert bool(result.episodes.iloc[0]["episode_has_event"]) is False
    assert bool(result.episodes.iloc[1]["episode_has_event"]) is True
    assert np.isclose(result.episodes.iloc[1]["own_fpp_onset"], 2.8)


def test_max_followup_censoring() -> None:
    config = _config(max_followup_s=6.0)
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([]),
        surprisal_table=_toy_tokens([("dyad-001", "1", "A", 0.0, 0.8, 1.0)]),
        config=config,
    )
    episode = result.episodes.iloc[0]

    assert np.isclose(episode["episode_end"], 6.0)
    assert episode["censor_reason"] == "max_followup"


def test_partner_ipu_anchor_does_not_use_spp_columns() -> None:
    config = _config()
    events = _toy_events([("dyad-001", "1", "B", 1.0, 1.1, "FPP_TEST")]).assign(
        spp_onset=1.8,
        spp_offset=2.0,
    )
    result = build_partner_ipu_anchored_episodes(
        events_table=events,
        surprisal_table=_toy_tokens([("dyad-001", "1", "A", 0.0, 0.8, 1.0)]),
        config=config,
    )

    assert set(result.episodes["anchor_source"]) == {"partner_ipu_tokens"}


def test_riskset_event_validation() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([("dyad-001", "1", "B", 1.2, 1.4, "FPP_TEST")]),
        surprisal_table=_toy_tokens(
            [
                ("dyad-001", "1", "A", 0.0, 0.8, 1.0),
                ("dyad-001", "1", "A", 2.0, 2.5, 1.0),
            ]
        ),
        config=config,
    )
    riskset_result = build_discrete_time_riskset(result.episodes, config=config)

    assert riskset_result.event_qc["positive_episodes_have_exactly_one_event_row"] is True
    assert riskset_result.event_qc["censored_episodes_have_zero_event_rows"] is True
    assert set(riskset_result.riskset_table["event"].unique()) <= {0, 1}


def test_observed_event_rate_nonzero_on_toy_data() -> None:
    config = _config()
    result = build_partner_ipu_anchored_episodes(
        events_table=_toy_events([("dyad-001", "1", "B", 1.2, 1.4, "FPP_TEST")]),
        surprisal_table=_toy_tokens([("dyad-001", "1", "A", 0.0, 0.8, 1.0)]),
        config=config,
    )
    riskset = build_discrete_time_riskset(result.episodes, config=config).riskset_table
    summary, _ = summarize_observed_event_rate_by_time_bin(riskset)

    assert int(summary["n_events"].sum()) == 1


def test_speaker_mismatch_error() -> None:
    config = _config()
    events = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "speaker": ["sub-001"],
            "fpp_onset": [1.0],
            "fpp_offset": [1.1],
            "fpp_label": ["FPP_TEST"],
        }
    )
    tokens = _toy_tokens([("dyad-001", "1", "A", 0.0, 0.8, 1.0)])

    with pytest.raises(ValueError, match="speaker labels could not be mapped"):
        build_partner_ipu_anchored_episodes(
            events_table=events,
            surprisal_table=tokens,
            config=config,
        )


def _config(**overrides: object) -> BehaviourHazardConfig:
    defaults: dict[str, object] = {
        "events_path": Path("unused.csv"),
        "surprisal_paths": (Path("unused.tsv"),),
        "out_dir": Path("out"),
        "episode_anchor": "partner_ipu",
        "include_censored": True,
    }
    defaults.update(overrides)
    return BehaviourHazardConfig(**defaults)


def _toy_events(rows: list[tuple[str, str, str, float, float, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["dyad_id", "run", "speaker", "fpp_onset", "fpp_offset", "fpp_label"])
    return pd.DataFrame(rows, columns=["dyad_id", "run", "speaker", "fpp_onset", "fpp_offset", "fpp_label"])


def _toy_tokens(rows: list[tuple[str, str, str, float, float, float]]) -> pd.DataFrame:
    payload = []
    for dyad_id, run, speaker, onset, offset, surprisal in rows:
        payload.append(
            {
                "dyad_id": dyad_id,
                "run": run,
                "speaker": speaker,
                "onset": onset,
                "duration": offset - onset,
                "offset": offset,
                "word": "w",
                "surprisal": surprisal,
                "alignment_status": "ok",
            }
        )
    return pd.DataFrame(payload)
