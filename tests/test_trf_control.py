from __future__ import annotations

import json

import numpy as np

from cas.trf.control import summarize_spp_onset_control_group
from cas.trf.prepare import build_impulse_predictor


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
