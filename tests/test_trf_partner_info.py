from __future__ import annotations

import numpy as np
import pandas as pd

from cas.annotations.io import write_textgrid
from cas.annotations.models import Interval, TextGrid, Tier
from cas.trf import partner_info


def _toy_token_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": [0.0, 0.5, 1.0],
            "offset": [0.5, 1.0, 1.5],
            "surprisal": [1.0, 2.0, 3.0],
            "partner_ipu_id": ["ipu-1", "ipu-1", "ipu-1"],
            "expected_total_info": [6.0, 6.0, 6.0],
        }
    )


def test_information_rate_divides_surprisal_by_duration_and_fills_interval() -> None:
    predictor = partner_info.build_information_rate_predictor(
        _toy_token_table(),
        n_samples=6,
        sfreq_hz=4.0,
        sigma_ms=0.0,
    )

    expected_raw = np.asarray([2.0, 2.0, 4.0, 4.0, 6.0, 6.0], dtype=float)
    expected = (expected_raw - expected_raw.mean()) / expected_raw.std(ddof=0)
    assert np.allclose(predictor, expected)


def test_prop_expected_cumulative_info_is_stepwise_from_word_onsets() -> None:
    predictor = partner_info.build_prop_expected_cumulative_info_predictor(
        _toy_token_table(),
        n_samples=6,
        sfreq_hz=4.0,
        sigma_ms=0.0,
    )

    expected_raw = np.asarray([1.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0, 1.0], dtype=float)
    expected = (expected_raw - expected_raw.mean()) / expected_raw.std(ddof=0)
    assert np.allclose(predictor, expected)


def test_unified_sigma_is_forwarded_to_both_information_predictors(monkeypatch) -> None:
    seen_sigma_values: list[float] = []

    def fake_smooth(values: np.ndarray, *, sfreq_hz: float, sigma_ms: float) -> np.ndarray:
        del sfreq_hz
        seen_sigma_values.append(float(sigma_ms))
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(partner_info, "_smooth_continuous", fake_smooth)

    token_table = _toy_token_table()
    partner_info.build_information_rate_predictor(token_table, n_samples=6, sfreq_hz=2.0, sigma_ms=200.0)
    partner_info.build_prop_expected_cumulative_info_predictor(token_table, n_samples=6, sfreq_hz=2.0, sigma_ms=200.0)

    assert seen_sigma_values == [200.0, 200.0]


def test_zscore_predictor_returns_finite_values_for_constant_signal() -> None:
    standardized = partner_info.zscore_predictor(np.asarray([5.0, 5.0, 5.0, 5.0], dtype=float))
    assert np.all(np.isfinite(standardized))
    assert np.allclose(standardized, 0.0)


def test_lag_window_samples_respects_sampling_frequency() -> None:
    lags = partner_info.lag_window_samples(start_ms=0.0, stop_ms=1500.0, sfreq_hz=50.0)
    assert lags[0] == 0
    assert lags[-1] == 75
    assert len(lags) == 76


def test_partner_info_model_specs_match_requested_sequence() -> None:
    controls = [
        "acoustic_envelope",
        "word_onset_impulse",
        "partner_onset_impulse",
    ]
    models = partner_info.build_partner_info_model_specs(controls)

    assert models["N0"] == controls
    assert models["N1"] == controls + ["information_rate"]
    assert models["N2"] == controls + ["prop_expected_cumulative_info"]
    assert models["N3"] == controls + ["information_rate", "prop_expected_cumulative_info"]


def test_partner_info_model_specs_can_be_read_from_config() -> None:
    controls = [
        "acoustic_envelope",
        "word_onset_impulse",
        "partner_onset_impulse",
    ]
    models = partner_info.build_partner_info_model_specs(
        controls,
        {
            "N0": {"predictors": controls},
            "N1": {"predictors": controls + ["information_rate"]},
            "N2": {"predictors": controls + ["prop_expected_cumulative_info"]},
            "N3": {"predictors": controls + ["information_rate", "prop_expected_cumulative_info"]},
        },
    )

    assert models["N0"] == controls
    assert models["N1"] == controls + ["information_rate"]
    assert models["N2"] == controls + ["prop_expected_cumulative_info"]
    assert models["N3"] == controls + ["information_rate", "prop_expected_cumulative_info"]


def test_annotation_ipu_loader_uses_canonical_ipu_tiers(tmp_path) -> None:
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    textgrid_path = annotations_dir / "dyad-009_run-2_combined.TextGrid"
    write_textgrid(
        TextGrid(
            xmin=0.0,
            xmax=2.0,
            tiers=[
                Tier(name="ipu-A", xmin=0.0, xmax=2.0, intervals=[Interval(0.0, 0.8, "ipu a")]),
                Tier(
                    name="ipu-B",
                    xmin=0.0,
                    xmax=2.0,
                    intervals=[
                        Interval(0.0, 0.5, "first"),
                        Interval(0.5, 1.2, ""),
                        Interval(1.2, 1.8, "second"),
                    ],
                ),
            ],
        ),
        textgrid_path,
    )

    ipu_table = partner_info._load_partner_ipus_from_annotations(
        subject_id="sub-018",
        run=2,
        dyad_id="dyad-009",
        partner_label="B",
        paths_config={"annotations_dir": str(annotations_dir)},
    )

    assert list(ipu_table["speaker"]) == ["B", "B"]
    assert np.allclose(
        ipu_table["partner_ipu_onset"].to_numpy(dtype=float),
        [0.0, 1.2],
    )
    assert np.allclose(
        ipu_table["partner_ipu_offset"].to_numpy(dtype=float),
        [0.5, 1.8],
    )
    assert ipu_table["anchor_source"].tolist() == ["annotation_ipu_tier", "annotation_ipu_tier"]


def test_partner_envelope_loader_does_not_apply_eeg_anchor_crop(tmp_path) -> None:
    features_root = tmp_path / "features"
    envelope_dir = features_root / "envelope" / "sub-002"
    envelope_dir.mkdir(parents=True)
    envelope_path = envelope_dir / "sub-002_task-conversation_run-1_envelope.npy"
    np.save(envelope_path, np.arange(8, dtype=float))
    envelope_path.with_name("sub-002_task-conversation_run-1_envelope.summary.json").write_text(
        '{"sampling_rate_hz": 4.0}',
        encoding="utf-8",
    )

    resampled, source_sfreq_hz = partner_info._load_partner_envelope_run(
        subject_id="sub-001",
        run=1,
        paths_config={"features_root": str(features_root)},
        target_n_samples=8,
    )

    assert source_sfreq_hz == 4.0
    assert np.array_equal(resampled, np.arange(8, dtype=float))
