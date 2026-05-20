from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cas.annotations.io import write_textgrid
from cas.annotations.models import Interval, TextGrid, Tier
from cas.trf import control
from cas.trf.control import (
    build_named_predictor_runs,
    summarize_model_delta_group,
    summarize_spp_onset_control_group,
)
from cas.trf.prepare import build_impulse_predictor
from cas.viz.lmeeeg import plot_joint_model_weights


def test_build_impulse_predictor_accumulates_valid_events() -> None:
    predictor = build_impulse_predictor(
        n_samples=10,
        sfreq_hz=2.0,
        event_times_s=np.asarray([0.0, 0.49, 0.51, 2.0, 20.0, np.nan], dtype=float),
    )

    assert predictor.shape == (10,)
    assert predictor.tolist() == [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_summarize_spp_onset_control_group_aggregates_subject_outputs(tmp_path) -> None:
    summary_paths = []
    coef_paths = []
    times_s = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    channel_names = np.asarray(["Cz", "Pz"], dtype=object)

    for index, delta in enumerate((0.02, 0.03), start=1):
        subject_id = f"sub-{index:03d}"
        summary_path = tmp_path / f"{subject_id}.summary.json"
        coef_path = tmp_path / f"{subject_id}.coefs.npz"

        payload = {
            "subject": subject_id,
            "models": {
                "full": {
                    "fold_scores": [
                        {"test_run": 1, "mean_score": 0.20 + delta, "channel_scores": [0.20 + delta, 0.21 + delta]},
                        {"test_run": 2, "mean_score": 0.18 + delta, "channel_scores": [0.18 + delta, 0.19 + delta]},
                    ]
                },
                "null": {
                    "fold_scores": [
                        {"test_run": 1, "mean_score": 0.20, "channel_scores": [0.20, 0.21]},
                        {"test_run": 2, "mean_score": 0.18, "channel_scores": [0.18, 0.19]},
                    ]
                },
            },
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

        full_coefficients = np.zeros((2, 3, 5, 2), dtype=float)
        full_coefficients[:, :, 3, :] = 1.0 + index
        np.savez(
            coef_path,
            times_s=times_s,
            channel_names=channel_names,
            full_predictors=np.asarray(
                ["envelope", "fpp_onset", "fpp_offset", "spp_onset", "spp_offset"],
                dtype=object,
            ),
            full_coefficients=full_coefficients,
            null_predictors=np.asarray(["envelope", "fpp_onset", "fpp_offset", "spp_offset"], dtype=object),
            null_coefficients=np.zeros((2, 3, 4, 2), dtype=float),
        )
        summary_paths.append(summary_path)
        coef_paths.append(coef_path)

    result = summarize_spp_onset_control_group(
        subject_summary_paths=summary_paths,
        subject_coefficient_paths=coef_paths,
    )

    assert list(result["subject_table"]["subject"]) == ["sub-001", "sub-002"]
    assert np.allclose(result["subject_table"]["delta_mean_r"].to_numpy(dtype=float), [0.02, 0.03])
    assert result["stats"]["n_subjects"] == 2
    assert result["stats"]["mean_delta_r"] > 0.0
    assert result["kernel"].shape == (2, 3)
    assert np.allclose(result["times_s"], times_s)


def test_build_named_predictor_runs_supports_ipu_and_weighted_impulses(tmp_path, monkeypatch) -> None:
    eeg_root = tmp_path / "eeg"
    features_root = tmp_path / "features"
    annotations_dir = tmp_path / "annotations"
    events_dir = tmp_path / "events"
    annotation_csv_root = annotations_dir / "csv"
    eeg_run_dir = eeg_root / "evoked" / "sub-001"
    other_feature_dir = features_root / "envelope" / "sub-002"
    eeg_run_dir.mkdir(parents=True)
    other_feature_dir.mkdir(parents=True)
    annotations_dir.mkdir()
    events_dir.mkdir()
    (annotation_csv_root / "ipu_v1").mkdir(parents=True)
    (annotation_csv_root / "syllable_v1").mkdir(parents=True)
    (annotation_csv_root / "palign_v1").mkdir(parents=True)
    (annotation_csv_root / "tokens_v1").mkdir(parents=True)

    np.save(eeg_run_dir / "run-1.npy", np.zeros((8, 2), dtype=float))
    np.save(other_feature_dir / "sub-002_task-conversation_run-1_envelope.npy", np.arange(8, dtype=float))

    events_table = pd.DataFrame(
        {
            "recording_id": ["dyad-001", "dyad-001", "dyad-001"],
            "run": ["1", "1", "1"],
            "speaker_fpp": ["A", "A", "A"],
            "speaker_spp": ["B", "B", "B"],
            "fpp_onset": [0.0, 0.0, 0.0],
            "fpp_offset": [0.0, 0.0, 0.0],
            "spp_onset": [0.125, 0.375, 0.625],
            "spp_offset": [0.25, 0.5, 0.75],
            "spp_label": ["SPP_CONF_SIMP", "SPP_DISC_CORR", "OTHER"],
        }
    )
    events_csv = events_dir / "events.csv"
    events_table.to_csv(events_csv, index=False)

    pd.DataFrame(
        {
            "tier": ["IPU", "IPU"],
            "start": [0.0, 0.5],
            "end": [0.25, 0.75],
            "duration": [0.25, 0.25],
            "n_syllables": [1, 1],
            "annotation": ["IPU", "IPU"],
            "rate": [4.0, 4.0],
        }
    ).to_csv(
        annotation_csv_root / "ipu_v1" / "sub-001_run-1_ipu.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "tier": ["IPU", "IPU"],
            "start": [0.125, 0.625],
            "end": [0.375, 0.875],
            "duration": [0.25, 0.25],
            "n_syllables": [1, 1],
            "annotation": ["IPU", "IPU"],
            "rate": [4.0, 4.0],
        }
    ).to_csv(
        annotation_csv_root / "ipu_v1" / "sub-002_run-1_ipu.csv",
        index=False,
    )
    pd.DataFrame(
        [
            ["SyllAlign", 0.125, 0.250, "sy1"],
            ["SyllAlign", 0.250, 0.500, None],
            ["SyllAlign", 0.625, 0.750, "sy2"],
        ]
    ).to_csv(
        annotation_csv_root / "syllable_v1" / "sub-002_run-1_syllable.csv",
        index=False,
        header=False,
    )
    pd.DataFrame(
        [
            ["PhonAlign", 0.125, 0.200, "p1"],
            ["TokensAlign", 0.125, 0.250, "tok"],
            ["PhonAlign", 0.200, 0.250, "#"],
            ["PhonAlign", 0.625, 0.700, "p2"],
            ["PronTokAlign", 0.625, 0.750, "pron"],
        ]
    ).to_csv(
        annotation_csv_root / "palign_v1" / "sub-002_run-1_palign.csv",
        index=False,
        header=False,
    )
    pd.DataFrame(
        {
            "run": [1, 1, 1, 1],
            "token": ["#", "ok", "#", "yes"],
            "speaker": ["A", "B", "B", "B"],
            "start": [0.0, 0.125, 0.250, 0.625],
            "end": [0.1, 0.250, 0.500, 0.750],
            "token_kind": ["silence", "lexical", "silence", "lexical"],
            "render_for_lm": [False, True, False, True],
            "rendered_text": [None, "ok", None, "yes"],
            "rendered_piece_count": [0, 1, 0, 1],
            "rendered_pieces_json": ["[]", "[\"ok\"]", "[]", "[\"yes\"]"],
        }
    ).to_csv(
        annotation_csv_root / "tokens_v1" / "dyad-001_tokens.csv",
        index=False,
    )

    monkeypatch.setattr(control, "_load_eeg_channel_names", lambda **_: ["Cz", "Pz"])

    trf_config = {
        "trf": {
            "annotation_csv": {"root": str(annotation_csv_root)},
            "predictor_definitions": {
                "envelope": {"kind": "continuous", "base_feature": "envelope", "role": "other"},
                "word_onset": {"kind": "annotation_csv_impulse", "source_kind": "word", "speaker_role": "other"},
                "syllable_onset": {"kind": "annotation_csv_impulse", "source_kind": "syllable", "speaker_role": "other"},
                "phoneme_onset": {"kind": "annotation_csv_impulse", "source_kind": "phoneme", "speaker_role": "other"},
                "self_speech_onset": {"kind": "annotation_csv_impulse", "source_kind": "ipu", "speaker_role": "self", "time_kind": "onset"},
                "other_speech_offset": {"kind": "annotation_csv_impulse", "source_kind": "ipu", "speaker_role": "other", "time_kind": "offset"},
                "spp_conf_vs_disconf": {
                    "kind": "weighted_impulse",
                    "speaker_column": "speaker_spp",
                    "speaker_role": "other",
                    "time_column": "spp_onset",
                    "label_column": "spp_label",
                    "label_prefix_weights": {"SPP_CONF": -1.0, "SPP_DISC": 1.0},
                },
            },
            "events": {"csv_path": str(events_csv)},
            "target": {"path": "evoked/{subject}/run-{run}.npy"},
            "timing": {"target_sfreq_hz": 8.0},
        }
    }
    (tmp_path / "paths.yaml").write_text(
        json.dumps(
            {
                "features_root": str(features_root),
                "eeg_array_root": str(eeg_root),
                "annotations_dir": str(annotations_dir),
            }
        ),
        encoding="utf-8",
    )

    eeg_runs, predictor_runs_by_name, channel_names = build_named_predictor_runs(
        trf_config=trf_config,
        subject_id="sub-001",
        runs=[1],
        project_root=tmp_path,
        config_root=tmp_path,
    )

    assert channel_names == ["Cz", "Pz"]
    assert eeg_runs[0].shape == (8, 2)
    assert predictor_runs_by_name["envelope"][0].reshape(-1).tolist() == list(np.arange(8, dtype=float))
    assert predictor_runs_by_name["word_onset"][0].reshape(-1).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert predictor_runs_by_name["syllable_onset"][0].reshape(-1).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert predictor_runs_by_name["phoneme_onset"][0].reshape(-1).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert predictor_runs_by_name["self_speech_onset"][0].reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert predictor_runs_by_name["other_speech_offset"][0].reshape(-1).tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert predictor_runs_by_name["spp_conf_vs_disconf"][0].reshape(-1).tolist() == [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_summarize_model_delta_group_aggregates_kernel_and_ttest(tmp_path) -> None:
    summary_paths = []
    coef_paths = []
    times_s = np.asarray([-0.1, 0.0, 0.1], dtype=float)
    channel_names = np.asarray(["Cz", "Pz"], dtype=object)

    for index, delta in enumerate((0.02, 0.03, 0.01), start=1):
        subject_id = f"sub-{index:03d}"
        summary_path = tmp_path / f"{subject_id}.summary.json"
        coef_path = tmp_path / f"{subject_id}.coefs.npz"
        payload = {
            "subject": subject_id,
            "models": {
                "M1": {
                    "fold_scores": [
                        {"test_run": 1, "mean_score": 0.20 + delta, "channel_scores": [0.20 + delta, 0.21 + delta]},
                        {"test_run": 2, "mean_score": 0.18 + delta, "channel_scores": [0.18 + delta, 0.19 + delta]},
                    ]
                },
                "M0": {
                    "fold_scores": [
                        {"test_run": 1, "mean_score": 0.20, "channel_scores": [0.20, 0.21]},
                        {"test_run": 2, "mean_score": 0.18, "channel_scores": [0.18, 0.19]},
                    ]
                },
            },
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

        m1_coefficients = np.zeros((2, 3, 2, 2), dtype=float)
        m1_coefficients[:, :, 1, :] = 0.5 + index
        np.savez(
            coef_path,
            times_s=times_s,
            channel_names=channel_names,
            M1_predictors=np.asarray(["envelope", "spp_conf_vs_disconf"], dtype=object),
            M1_coefficients=m1_coefficients,
            M0_predictors=np.asarray(["envelope"], dtype=object),
            M0_coefficients=np.zeros((2, 3, 1, 2), dtype=float),
        )
        summary_paths.append(summary_path)
        coef_paths.append(coef_path)

    result = summarize_model_delta_group(
        subject_summary_paths=summary_paths,
        subject_coefficient_paths=coef_paths,
        full_model_name="M1",
        null_model_name="M0",
        kernel_predictor="spp_conf_vs_disconf",
        score_test="ttest_1samp",
    )

    assert list(result["subject_table"]["subject"]) == ["sub-001", "sub-002", "sub-003"]
    assert np.allclose(result["subject_table"]["delta_mean_r"].to_numpy(dtype=float), [0.02, 0.03, 0.01])
    assert result["kernel"].shape == (2, 3)
    assert np.allclose(result["times_s"], times_s)
    assert result["stats"]["test"] == "ttest_1samp"
    assert result["stats"]["kernel_predictor"] == "spp_conf_vs_disconf"


def test_plot_joint_model_weights_falls_back_without_montage(tmp_path) -> None:
    times_s = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
    weights = np.asarray(
        [
            [0.1, 0.2, 0.1, -0.1],
            [0.0, 0.3, 0.2, -0.2],
            [-0.1, 0.1, 0.0, -0.1],
        ],
        dtype=float,
    )
    significance_mask = np.asarray(
        [
            [False, True, True, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    written = plot_joint_model_weights(
        weights,
        times=times_s,
        channel_names=["TRF000", "TRF001", "TRF002"],
        output_stem=tmp_path / "kernel_joint",
        title="SPP conf vs disconf",
        formats=("png",),
        significance_mask=significance_mask,
    )

    assert len(written) == 1
    assert written[0].exists()
