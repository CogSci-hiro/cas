from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hmm.tde_hmm import (
    TdeHmmConfig,
    build_non_speaking_mask,
    compute_causal_lags,
    compute_state_entropy,
    concatenate_chunks_and_build_indices,
    fit_tde_hmm_pipeline,
    map_processed_feature_to_original_timeline,
    preprocess_data_for_glhmm,
    split_valid_samples_into_chunks,
)


def test_compute_causal_lags_128hz_100ms() -> None:
    lags = compute_causal_lags(128.0, 100.0)
    assert np.array_equal(lags, np.arange(-12, 1))
    assert lags[-1] == 0
    assert np.all(np.diff(lags) == 1)


def test_speech_mask_excludes_guard_bands_correctly() -> None:
    mask = build_non_speaking_mask(
        10,
        sampling_rate_hz=10.0,
        speech_intervals_s=np.array([[0.3, 0.5]]),
        guard_pre_s=0.1,
        guard_post_s=0.2,
    )
    expected = np.array([True, True, False, False, False, False, False, True, True, True])
    assert np.array_equal(mask, expected)


def test_split_valid_samples_into_chunks_simple_mask() -> None:
    keep_mask = np.array([False, True, True, False, True, True, True, False, True])
    chunks = split_valid_samples_into_chunks(keep_mask, minimum_chunk_samples=2)
    assert chunks == [(1, 3), (4, 7)]


def test_concatenation_and_indices_are_correct() -> None:
    data = np.arange(20, dtype=float).reshape(10, 2)
    chunks = [(1, 4), (6, 9)]
    concatenated, indices, mapping = concatenate_chunks_and_build_indices(data, chunks)

    assert concatenated.shape == (6, 2)
    assert np.array_equal(indices, np.array([[0, 3], [3, 6]]))
    assert mapping["original_sample_index"].tolist() == [1, 2, 3, 6, 7, 8]
    assert mapping["chunk_id"].tolist() == [0, 0, 0, 1, 1, 1]


def test_no_cross_chunk_leakage_after_embedding() -> None:
    data = np.arange(10, dtype=float).reshape(10, 1)
    chunks = [(0, 4), (6, 10)]
    concatenated, indices, mapping = concatenate_chunks_and_build_indices(data, chunks)
    config = TdeHmmConfig(sampling_rate_hz=10.0, causal_history_ms=200.0, verbose=False)
    processed, processed_indices, _, processed_mapping = preprocess_data_for_glhmm(
        concatenated,
        indices,
        config,
    )

    assert processed.shape[0] == 4
    assert np.array_equal(processed_indices, np.array([[0, 2], [2, 4]]))
    assert processed_mapping["concat_index"].tolist() == [2, 3, 6, 7]
    merged = processed_mapping.merge(mapping, on=["concat_index", "chunk_id"], how="left")
    assert merged["original_sample_index"].tolist() == [2, 3, 8, 9]


def test_entropy_computation_matches_simple_examples() -> None:
    gamma = np.array([[1.0, 0.0], [0.5, 0.5]])
    entropy = compute_state_entropy(gamma)
    assert np.isclose(entropy[0], 0.0, atol=1e-8)
    assert np.isclose(entropy[1], np.log(2.0), atol=1e-8)


def test_mapping_back_to_original_timeline_uses_nan_for_invalid_samples() -> None:
    entropy = np.array([0.1, 0.2, 0.3])
    processed_mapping = pd.DataFrame(
        {
            "processed_index": [0, 1, 2],
            "concat_index": [2, 3, 7],
            "chunk_id": [0, 0, 1],
        }
    )
    concat_mapping = pd.DataFrame(
        {
            "concat_index": [0, 1, 2, 3, 4, 5, 6, 7],
            "original_sample_index": [0, 1, 2, 3, 6, 7, 8, 9],
            "chunk_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    timeline = map_processed_feature_to_original_timeline(
        entropy,
        processed_mapping,
        concat_mapping,
        original_n_samples=10,
    )
    assert np.isnan(timeline[0])
    assert np.isclose(timeline[2], 0.1)
    assert np.isclose(timeline[3], 0.2)
    assert np.isclose(timeline[9], 0.3)
    assert np.isnan(timeline[4])


def test_pipeline_handles_tiny_excluded_segments_robustly(tmp_path: Path, monkeypatch) -> None:
    run_a = np.column_stack([np.arange(20, dtype=float), np.arange(20, dtype=float) * 2.0])
    run_b = np.column_stack([np.arange(24, dtype=float), np.arange(24, dtype=float) * 3.0])
    run_a_path = tmp_path / "run_a.npy"
    run_b_path = tmp_path / "run_b.npy"
    np.save(run_a_path, run_a)
    np.save(run_b_path, run_b)

    speech_a_path = tmp_path / "run_a_speech.csv"
    speech_b_path = tmp_path / "run_b_speech.csv"
    pd.DataFrame({"onset_s": [0.08], "offset_s": [0.09]}).to_csv(speech_a_path, index=False)
    pd.DataFrame({"onset_s": [0.12], "offset_s": [0.13]}).to_csv(speech_b_path, index=False)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "run": ["r1", "r2"],
            "source_path": [run_a_path.name, run_b_path.name],
            "speech_path": [speech_a_path.name, speech_b_path.name],
        }
    ).to_csv(manifest_path, index=False)

    def fake_evaluate_candidate_k_values(
        processed_data: np.ndarray,
        processed_indices: np.ndarray,
        config: TdeHmmConfig,
    ) -> tuple[pd.DataFrame, int, dict[int, tuple[dict[str, int], np.ndarray, None, float]]]:
        del processed_indices, config
        gamma = np.full((processed_data.shape[0], 4), 0.25, dtype=float)
        return (
            pd.DataFrame(
                {
                    "k": [4, 5],
                    "fit_success": [True, True],
                    "free_energy": [10.0, 12.0],
                    "mean_occupancy": [0.25, 0.20],
                    "minimum_occupancy": [0.20, 0.10],
                    "n_effectively_empty_states": [0, 0],
                    "is_degenerate": [False, False],
                }
            ),
            4,
            {
                4: ({"k": 4}, gamma, None, 10.0),
            },
        )

    def fake_fit_one_glhmm_model(
        processed_data: np.ndarray,
        processed_indices: np.ndarray,
        k: int,
        config: TdeHmmConfig,
    ) -> tuple[dict[str, int], np.ndarray, None, float]:
        del processed_indices, config
        gamma = np.full((processed_data.shape[0], k), 1.0 / k, dtype=float)
        return {"k": k}, gamma, None, 10.0

    monkeypatch.setattr(
        "cas.hmm.tde_hmm.evaluate_candidate_k_values",
        fake_evaluate_candidate_k_values,
    )
    monkeypatch.setattr(
        "cas.hmm.tde_hmm.fit_one_glhmm_model",
        fake_fit_one_glhmm_model,
    )

    config = TdeHmmConfig(
        sampling_rate_hz=100.0,
        causal_history_ms=20.0,
        speech_guard_pre_s=0.0,
        speech_guard_post_s=0.0,
        minimum_chunk_duration_s=0.03,
        candidate_k=(4, 5),
        verbose=False,
    )
    result = fit_tde_hmm_pipeline(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        config=config,
    )

    assert result.selected_k == 4
    assert result.model_path.exists()
    assert result.gamma_path.exists()
    assert result.chunk_table_path.exists()
    assert len(result.entropy_csv_paths) == 2

    gamma = np.load(result.gamma_path)
    assert gamma.ndim == 2
    assert gamma.shape[1] == 4

    entropy_processed = np.load(result.entropy_processed_path)
    assert np.allclose(entropy_processed, np.log(4.0))

    summary = json.loads(result.fit_summary_path.read_text(encoding="utf-8"))
    assert summary["selected_k"] == 4
    assert summary["number_of_chunks"] >= 2

    with result.model_path.open("rb") as handle:
        model = pickle.load(handle)
    assert model["k"] == 4
