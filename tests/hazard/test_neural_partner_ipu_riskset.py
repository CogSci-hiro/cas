from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard.config import (
    EntropyConfig,
    EventDefinitionConfig,
    HazardAnalysisConfig,
    InputConfig,
    MiscConfig,
    ModelConfig,
    NeuralHazardConfig,
    NeuralEpisodeConfig,
    NeuralInputConfig,
    NeuralModelConfig,
    NeuralPcaConfig,
    NeuralWindowConfig,
    OutputConfig,
    PlottingConfig,
    QcConfig,
    TimeAxisConfig,
)
from cas.hazard.pipeline import _add_neural_features_and_pcs
from cas.hazard.riskset import build_neural_partner_ipu_risksets


def test_neural_partner_ipu_riskset_keeps_during_partner_ipu_events() -> None:
    events_table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"],
            "run": ["1"],
            "participant_speaker": ["B"],
            "partner_speaker": ["B"],
            "fpp_onset": [0.35],
            "fpp_label": ["FPP_RFC_DECL"],
            "spp_onset": [0.55],
            "spp_label": ["SPP_CONF_SIMP"],
        }
    )
    surprisal_table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-001", "dyad-001"],
            "run": ["1", "1", "1", "1"],
            "speaker": ["A", "A", "A", "A"],
            "onset": [0.0, 0.2, 1.0, 1.2],
            "duration": [0.2, 0.2, 0.2, 0.2],
            "surprisal": [1.0, 1.1, 0.9, 1.0],
            "offset": [0.2, 0.4, 1.2, 1.4],
        }
    )
    config = NeuralHazardConfig(
        enabled=True,
        event_types=("fpp", "spp"),
        bin_size_s=0.05,
        episode=NeuralEpisodeConfig(max_followup_s=1.0),
    )

    result = build_neural_partner_ipu_risksets(
        events_table=events_table,
        surprisal_table=surprisal_table,
        neural_config=config,
    )
    fpp_table = result.risksets_by_event["fpp"]
    assert "event_fpp" in fpp_table.columns
    assert "participant_speaker_id" in fpp_table.columns
    assert int(fpp_table["event_fpp"].sum()) == 1
    fpp_event_row = fpp_table.loc[fpp_table["event_fpp"] == 1].iloc[0]
    assert fpp_event_row["event_phase"] == "during_partner_ipu"
    assert float(fpp_event_row["own_fpp_onset"]) < float(fpp_event_row["partner_ipu_offset"])

    spp_table = result.risksets_by_event["spp"]
    assert "event_spp" in spp_table.columns
    assert int(spp_table["event_spp"].sum()) == 1


def test_neural_feature_extraction_uses_participant_speaker_eeg() -> None:
    config = HazardAnalysisConfig(
        input=InputConfig(
            tde_hmm_results_dir=Path("."),
            events_table_path=Path("."),
        ),
        output=OutputConfig(output_dir=Path(".")),
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
        time_axis=TimeAxisConfig(observation_window_seconds=1.0, bin_size_seconds=0.05),
        entropy=EntropyConfig(),
        model=ModelConfig(),
        qc=QcConfig(),
        plotting=PlottingConfig(),
        misc=MiscConfig(overwrite=True),
        neural=NeuralHazardConfig(
            enabled=True,
            bin_size_s=0.05,
            events_path=Path("events.csv"),
            out_dir=Path("out"),
            input=NeuralInputConfig(surprisal_paths=(Path("s.tsv"),), lowlevel_neural_paths=(Path("n.tsv"),)),
            window=NeuralWindowConfig(start_lag_s=0.50, end_lag_s=0.10),
            pca=NeuralPcaConfig(mode="count", n_components=1),
            model=NeuralModelConfig(),
        ),
    )
    riskset = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001"],
            "run": ["1", "1"],
            "participant_speaker": ["B", "B"],
            "partner_speaker": ["A", "A"],
            "bin_start": [0.50, 0.55],
            "bin_end": [0.55, 0.60],
            "time_from_partner_onset": [0.50, 0.55],
            "time_from_partner_offset": [0.15, 0.20],
        }
    )
    lowlevel = pd.DataFrame(
        {
            "dyad_id": ["dyad-001"] * 12,
            "run": ["1"] * 12,
            "speaker": ["A"] * 6 + ["B"] * 6,
            "time": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] * 2,
            "alpha_Cz": [100.0] * 6 + [1.0] * 6,
            "beta_Cz": [100.0] * 6 + [2.0] * 6,
        }
    )

    enriched, qc = _add_neural_features_and_pcs(riskset, lowlevel_table=lowlevel, config=config)
    assert qc["participant_speaker_eeg_used"] is True
    assert enriched["participant_speaker_id"].tolist() == ["dyad-001_B", "dyad-001_B"]
    assert "z_alpha_pc1" in enriched.columns
    assert "z_beta_pc1" in enriched.columns
