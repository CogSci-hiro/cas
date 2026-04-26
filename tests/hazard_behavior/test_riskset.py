from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.riskset import build_discrete_time_riskset


def test_event_bin_assignment_and_time_axis() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        include_censored=False,
    )
    episodes = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "participant_speaker": ["B"],
            "partner_speaker": ["A"],
            "episode_id": ["ep-1"],
            "episode_kind": ["event_positive"],
            "partner_ipu_onset": [0.0],
            "partner_ipu_offset": [0.8],
            "partner_ipu_class": ["test"],
            "partner_role": ["partner"],
            "own_fpp_onset": [1.20],
            "censor_time": [1.20],
            "event_observed": [1],
        }
    )

    result = build_discrete_time_riskset(episodes, config=config)
    riskset = result.riskset_table

    assert int(riskset["event"].sum()) == 1
    assert riskset.iloc[-1]["bin_start"] == 1.20
    assert riskset.iloc[-1]["event"] == 1
    assert np.isclose(riskset.iloc[0]["time_from_partner_onset"], 0.0)
    assert np.isclose(riskset.iloc[5]["time_from_partner_onset"], 0.25)
    assert not (riskset["bin_start"] > 1.20).any()


def test_time_from_partner_offset_is_signed_and_row_count_is_unchanged() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        include_censored=False,
    )
    episodes = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "participant_speaker": ["B"],
            "partner_speaker": ["A"],
            "episode_id": ["ep-1"],
            "episode_kind": ["event_positive"],
            "partner_ipu_onset": [0.0],
            "partner_ipu_offset": [1.0],
            "partner_ipu_class": ["test"],
            "partner_role": ["partner"],
            "own_fpp_onset": [1.25],
            "censor_time": [1.25],
            "event_observed": [1],
        }
    )

    result = build_discrete_time_riskset(episodes, config=config)
    riskset = result.riskset_table

    assert len(riskset) == 26
    positive_row = riskset.loc[np.isclose(riskset["bin_end"], 1.25)].iloc[0]
    negative_row = riskset.loc[np.isclose(riskset["bin_end"], 0.75)].iloc[0]
    assert positive_row["time_from_partner_offset"] == pytest.approx(0.25)
    assert negative_row["time_from_partner_offset"] == pytest.approx(-0.25)
