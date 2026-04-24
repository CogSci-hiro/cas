from __future__ import annotations

import numpy as np

from cas.features.voxatlas import extract_acoustic_features
from cas.trf.prepare import build_feature_path


def test_extract_acoustic_features_returns_envelope_and_f0() -> None:
    sampling_rate_hz = 16_000.0
    duration_s = 1.0
    time_s = np.arange(int(sampling_rate_hz * duration_s)) / sampling_rate_hz
    signal = (0.25 * np.sin(2.0 * np.pi * 120.0 * time_s)).astype(np.float32)
    config = {
        "acoustic": {
            "envelope": {
                "frame_length_s": 0.025,
                "frame_step_s": 0.010,
                "smoothing": 1,
            },
            "f0": {
                "fmin_hz": 75.0,
                "fmax_hz": 300.0,
                "frame_length_s": 0.040,
                "frame_step_s": 0.010,
            },
        }
    }

    bundle = extract_acoustic_features(signal, sampling_rate_hz, config)

    assert bundle.envelope.values.ndim == 1
    assert bundle.envelope.time_s.shape == bundle.envelope.values.shape
    assert bundle.envelope.values.size > 0
    assert np.isfinite(bundle.envelope.values).all()

    finite_f0 = bundle.f0.values[np.isfinite(bundle.f0.values)]
    assert bundle.f0.values.ndim == 1
    assert bundle.f0.time_s.shape == bundle.f0.values.shape
    assert finite_f0.size > 0
    assert np.isclose(np.median(finite_f0), 120.0, atol=10.0)


def test_build_feature_path_matches_subject_conversation_outputs() -> None:
    path = build_feature_path("f0", "sub-018", 7)
    assert path.as_posix() == (
        "derivatives/features/f0/sub-018/sub-018_task-conversation_run-7_f0.npy"
    )
