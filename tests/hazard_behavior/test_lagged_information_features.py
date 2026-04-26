from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.features import (
    ACTIVE_LAGGED_INFORMATION_FEATURES,
    add_lagged_information_features,
    build_lagged_feature_qc,
    compute_information_timing_summaries,
)


def test_active_lag_0_equals_unlagged() -> None:
    config = _config()
    riskset = _toy_riskset_single_episode()

    lagged, _ = add_lagged_information_features(riskset, config=config)

    assert ACTIVE_LAGGED_INFORMATION_FEATURES == (
        "information_rate",
        "prop_expected_cumulative_info",
    )
    assert np.allclose(lagged["information_rate_lag_0ms"], lagged["information_rate"])
    assert np.allclose(
        lagged["prop_expected_cumulative_info_lag_0ms"],
        lagged["prop_expected_cumulative_info"],
    )
    assert "cumulative_info_lag_0ms" not in lagged.columns


def test_active_positive_lag_uses_past_values() -> None:
    config = _config(lag_grid_ms=(100,))
    riskset = _toy_riskset_single_episode()
    riskset["prop_expected_cumulative_info"] = [0.0, 0.1, 0.2, 0.3, 0.4]

    lagged, _ = add_lagged_information_features(riskset, config=config)

    assert np.allclose(
        lagged["prop_expected_cumulative_info_lag_100ms"],
        [0.0, 0.0, 0.0, 0.1, 0.2],
    )


def test_active_lag_is_computed_within_episode() -> None:
    config = _config(lag_grid_ms=(100,))
    riskset = pd.concat(
        [
            _toy_riskset_single_episode(episode_id="ep-1", offset=0.0),
            _toy_riskset_single_episode(episode_id="ep-2", offset=10.0),
        ],
        ignore_index=True,
    )

    lagged, _ = add_lagged_information_features(riskset, config=config)

    ep1 = lagged.loc[lagged["episode_id"] == "ep-1", "information_rate_lag_100ms"].to_list()
    ep2 = lagged.loc[lagged["episode_id"] == "ep-2", "information_rate_lag_100ms"].to_list()
    assert ep1 == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert ep2 == [0.0, 0.0, 10.0, 11.0, 12.0]


def test_lagged_feature_qc_lists_only_active_families() -> None:
    config = _config(lag_grid_ms=(0, 100))
    riskset = _toy_riskset_single_episode()
    lagged, _ = add_lagged_information_features(riskset, config=config)

    qc = build_lagged_feature_qc(lagged, config=config)

    assert set(qc["feature_names_created"]) == {
        "information_rate_lag_0ms",
        "information_rate_lag_100ms",
        "prop_expected_cumulative_info_lag_0ms",
        "prop_expected_cumulative_info_lag_100ms",
    }


def test_information_timing_summaries() -> None:
    config = _config()
    episodes = pd.DataFrame(
        {
            "episode_id": ["ep-1"],
            "partner_ipu_id": ["ipu-1"],
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "partner_speaker": ["A"],
            "partner_ipu_onset": [0.0],
            "partner_ipu_offset": [1.1],
        }
    )
    surprisal = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"] * 3,
            "run": ["1"] * 3,
            "speaker": ["A"] * 3,
            "onset": [0.0, 0.5, 1.0],
            "offset": [0.1, 0.6, 1.1],
            "surprisal": [1.0, 1.0, 2.0],
        }
    )

    summary = compute_information_timing_summaries(
        episodes_table=episodes,
        surprisal_table=surprisal,
        config=config,
    )
    row = summary.iloc[0]

    assert np.isclose(row["actual_total_info"], 4.0)
    assert np.isclose(row["info_centroid_s"], 0.625)
    assert np.isclose(row["info_t25_s"], 0.0)
    assert np.isclose(row["info_t50_s"], 0.5)
    assert np.isclose(row["info_t75_s"], 1.0)
    assert np.isclose(row["info_t90_s"], 1.0)
    assert np.isclose(row["info_prop_by_500ms"], 0.5)
    assert np.isclose(row["info_prop_by_1000ms"], 1.0)


def _config(**overrides: object) -> BehaviourHazardConfig:
    defaults: dict[str, object] = {
        "events_path": Path("unused.csv"),
        "surprisal_paths": (Path("unused.tsv"),),
        "out_dir": Path("out"),
    }
    defaults.update(overrides)
    return BehaviourHazardConfig(**defaults)


def _toy_riskset_single_episode(episode_id: str = "ep-1", offset: float = 0.0) -> pd.DataFrame:
    times = np.array([0.00, 0.05, 0.10, 0.15, 0.20])
    return pd.DataFrame(
        {
            "episode_id": [episode_id] * len(times),
            "bin_index": list(range(len(times))),
            "time_from_partner_onset": times,
            "information_rate": offset + np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "cumulative_info": offset + np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "prop_expected_cumulative_info": np.array([0.0, 0.20, 0.40, 0.60, 0.80]),
        }
    )
