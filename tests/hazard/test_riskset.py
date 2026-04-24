from __future__ import annotations

import numpy as np
import pandas as pd

from cas.hazard.config import (
    EntropyConfig,
    EventDefinitionConfig,
    HazardAnalysisConfig,
    InputConfig,
    MiscConfig,
    ModelConfig,
    OutputConfig,
    PlottingConfig,
    QcConfig,
    TimeAxisConfig,
)
from cas.hazard.riskset import build_person_period_riskset


def test_forward_running_episode_construction_and_censoring(tmp_path) -> None:
    config = _build_config(tmp_path)
    events_table = pd.DataFrame(
        {
            "recording_id": ["dyad-003"],
            "run": ["1"],
            "speaker_fpp": ["A"],
            "speaker_spp": ["B"],
            "fpp_label": ["FPP_RFC_DECL"],
            "fpp_onset": [1.0],
            "spp_onset": [1.17],
            "spp_label": ["SPP_CONF_SIMP"],
            "pair_id": ["pair_1"],
        }
    )
    pairing_issues_table = pd.DataFrame(
        {
            "recording_id": ["dyad-003"],
            "run": ["1"],
            "fpp_tier": ["action A"],
            "fpp_label": ["FPP_RFC_DECL"],
            "fpp_onset": [2.0],
            "fpp_offset": [2.2],
            "issue_code": ["no_opposite_spp"],
        }
    )
    entropy_by_run = {("sub-006", "1"): _build_entropy_frame()}

    result = build_person_period_riskset(
        config=config,
        events_table=events_table,
        pairing_issues_table=pairing_issues_table,
        entropy_by_run=entropy_by_run,
        dyad_table=None,
    )

    hazard_table = result.hazard_table
    assert result.n_episodes == 2
    assert result.n_positive_episodes == 1
    assert result.n_censored_episodes == 1
    positive_episode = hazard_table.loc[hazard_table["event_id"] == "sub-006|run-1|pair_1"]
    assert positive_episode["event"].sum() == 1
    assert positive_episode.iloc[-1]["event"] == 1
    censored_episode = hazard_table.loc[hazard_table["censored_episode"] == 1]
    assert censored_episode["event"].sum() == 0


def test_lagged_entropy_alignment_uses_time_since_partner_onset(tmp_path) -> None:
    config = _build_config(tmp_path)
    events_table = pd.DataFrame(
        {
            "recording_id": ["dyad-003"],
            "run": ["1"],
            "speaker_fpp": ["A"],
            "speaker_spp": ["B"],
            "fpp_label": ["FPP_RFC_DECL"],
            "fpp_onset": [1.0],
            "spp_onset": [1.20],
            "spp_label": ["SPP_CONF_SIMP"],
            "pair_id": ["pair_1"],
        }
    )
    entropy_by_run = {
        ("sub-006", "1"): pd.DataFrame(
            {
                "sample": [0, 1, 2, 3, 4],
                "time_s": [0.85, 0.90, 0.95, 1.00, 1.05],
                "state_entropy": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
    }

    result = build_person_period_riskset(
        config=config,
        events_table=events_table,
        pairing_issues_table=None,
        entropy_by_run=entropy_by_run,
        dyad_table=None,
    )

    first_row = result.hazard_table.iloc[0]
    assert np.isclose(first_row["tau_seconds"], 0.05)
    assert np.isclose(first_row["predictor_time_seconds"], 0.90)
    assert np.isclose(first_row["entropy"], 20.0)


def test_within_subject_zscoring_is_applied(tmp_path) -> None:
    config = _build_config(tmp_path)
    events_table = pd.DataFrame(
        {
            "recording_id": ["dyad-003", "dyad-003"],
            "run": ["1", "1"],
            "speaker_fpp": ["A", "A"],
            "speaker_spp": ["B", "B"],
            "fpp_label": ["FPP_RFC_DECL", "FPP_RFC_DECL"],
            "fpp_onset": [1.0, 2.0],
            "spp_onset": [1.20, 2.25],
            "spp_label": ["SPP_CONF_SIMP", "SPP_CONF_SIMP"],
            "pair_id": ["pair_1", "pair_2"],
        }
    )
    entropy_by_run = {("sub-006", "1"): _build_entropy_frame()}

    result = build_person_period_riskset(
        config=config,
        events_table=events_table,
        pairing_issues_table=None,
        entropy_by_run=entropy_by_run,
        dyad_table=None,
    )

    values = result.hazard_table["entropy_z"].to_numpy(dtype=float)
    assert np.isclose(values.mean(), 0.0, atol=1.0e-9)
    assert np.isclose(values.std(ddof=0), 1.0, atol=1.0e-9)


def _build_config(tmp_path) -> HazardAnalysisConfig:
    return HazardAnalysisConfig(
        input=InputConfig(
            tde_hmm_results_dir=tmp_path / "unused_hmm",
            events_table_path=tmp_path / "unused_events.csv",
            pairing_issues_table_path=tmp_path / "unused_pairing_issues.csv",
        ),
        output=OutputConfig(output_dir=tmp_path / "out"),
        event_definition=EventDefinitionConfig(
            partner_onset_column="fpp_onset",
            target_onset_column="spp_onset",
            target_label_column="spp_label",
            partner_label_column="fpp_label",
            fpp_label_prefixes=("FPP_",),
            event_id_column="pair_id",
            recording_id_column="recording_id",
            run_column="run",
            partner_speaker_column="speaker_fpp",
            target_speaker_column="speaker_spp",
        ),
        time_axis=TimeAxisConfig(
            observation_window_seconds=0.3,
            bin_size_seconds=0.05,
            entropy_lag_seconds=0.15,
            exclude_initial_seconds=0.0,
        ),
        entropy=EntropyConfig(
            normalize_by_log_k=True,
            epsilon=1.0e-12,
            zscore_within_subject=True,
        ),
        model=ModelConfig(),
        qc=QcConfig(),
        plotting=PlottingConfig(),
        misc=MiscConfig(overwrite=True),
    )


def _build_entropy_frame() -> pd.DataFrame:
    time_s = np.arange(0.0, 4.01, 0.05)
    state_entropy = 0.1 + time_s
    return pd.DataFrame(
        {
            "sample": np.arange(time_s.shape[0], dtype=int),
            "time_s": time_s,
            "state_entropy": state_entropy,
        }
    )
