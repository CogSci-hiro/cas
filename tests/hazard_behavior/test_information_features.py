from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.features import (
    add_information_features_to_riskset,
    compute_expected_total_information,
)
from cas.hazard_behavior.io import read_surprisal_tables
from cas.hazard_behavior.riskset import build_discrete_time_riskset


def test_information_features_match_toy_expectations(tmp_path: Path) -> None:
    surprisal_path = tmp_path / "toy.tsv"
    surprisal_path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\tonset\tduration\tword\tsurprisal\talignment_status",
                "dyad-001\t1\tA\t0.00\t0.10\toui\t1.0\tok",
                "dyad-001\t1\tA\t0.30\t0.10\talors\t2.0\tok",
                "dyad-001\t1\tA\t0.70\t0.10\trouge\t3.0\tok",
                "dyad-001\t1\tB\t1.20\t0.10\tok\t1.5\tok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    surprisal_table, _ = read_surprisal_tables((surprisal_path,), unmatched_surprisal_strategy="drop")
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(surprisal_path,),
        out_dir=tmp_path / "out",
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
            "partner_ipu_class": ["unknown"],
            "partner_role": ["partner"],
            "own_fpp_onset": [1.20],
            "censor_time": [1.20],
            "event_observed": [1],
        }
    )
    riskset = build_discrete_time_riskset(episodes, config=config).riskset_table
    riskset_with_features, expected = add_information_features_to_riskset(
        riskset_table=riskset,
        episodes_table=episodes,
        surprisal_table=surprisal_table,
        config=config,
    )

    row_035 = riskset_with_features.loc[np.isclose(riskset_with_features["bin_end"], 0.35)].iloc[0]
    row_075 = riskset_with_features.loc[np.isclose(riskset_with_features["bin_end"], 0.75)].iloc[0]

    assert np.isclose(row_035["cumulative_info"], 3.0)
    assert np.isclose(row_075["cumulative_info"], 6.0)
    assert np.isclose(row_075["actual_total_info"], 6.0)
    assert np.isclose(row_075["prop_actual_cumulative_info"], 1.0)
    assert np.isclose(row_075["information_rate"], 10.0)
    assert np.isclose(expected["global"], 6.0)
    assert np.isclose(row_075["prop_expected_cumulative_info"], 1.0)


def test_unmatched_surprisal_strategies_behave_as_expected(tmp_path: Path) -> None:
    path = tmp_path / "toy.tsv"
    path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\tonset\tduration\tsurprisal\talignment_status",
                "dyad-001\t1\tA\t0.00\t0.10\t1.0\tok",
                "dyad-001\t1\tA\t0.30\t0.10\t\tunmatched",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dropped, _ = read_surprisal_tables((path,), unmatched_surprisal_strategy="drop")
    zeroed, _ = read_surprisal_tables((path,), unmatched_surprisal_strategy="zero")
    kept, _ = read_surprisal_tables((path,), unmatched_surprisal_strategy="keep_nan")

    assert len(dropped) == 1
    assert len(zeroed) == 2
    assert float(zeroed.iloc[1]["surprisal"]) == 0.0
    assert len(kept) == 2
    assert kept["surprisal"].isna().sum() == 1
