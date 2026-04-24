from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hmm.qc import (
    compute_chunk_durations,
    compute_delta_free_energy,
    compute_dwell_times,
    compute_empirical_transition_matrix,
    compute_fractional_occupancy,
    compute_normalized_state_entropy,
    compute_subject_run_fractional_occupancy,
    load_qc_inputs,
    plot_pca_explained_variance,
    run_all_qc_plots,
)


def _write_minimal_qc_inputs(tmp_path: Path) -> Path:
    output_dir = tmp_path / "hmm"
    output_dir.mkdir()

    gamma = np.asarray(
        [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    np.save(output_dir / "gamma_k2.npy", gamma)
    np.save(
        output_dir / "state_entropy_processed_k2.npy",
        np.asarray([0.0, 0.56233514, 0.56233514, 0.0], dtype=float),
    )

    pd.DataFrame({"k": [2, 3], "free_energy": [10.0, 9.5]}).to_csv(
        output_dir / "model_selection.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "subject": ["s1"],
            "run": ["r1"],
            "chunk_id": [0],
            "original_start_sample": [0],
            "original_stop_sample": [4],
            "concat_start_sample": [0],
            "concat_stop_sample": [4],
        }
    ).to_csv(output_dir / "chunks.csv", index=False)
    pd.DataFrame(
        {
            "sample": [0, 1, 2, 3],
            "time_s": [0.0, 0.1, 0.2, 0.3],
            "state_entropy": [0.0, 0.56233514, 0.56233514, 0.0],
        }
    ).to_csv(output_dir / "subject-s1_run-r1_state_entropy.csv", index=False)

    fit_summary = {
        "selected_k": 2,
        "sampling_rate_hz": 10.0,
        "causal_history_ms": 100.0,
        "minimum_chunk_duration_s": 0.2,
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(fit_summary) + "\n", encoding="utf-8")
    return output_dir


def test_compute_fractional_occupancy_returns_state_means() -> None:
    gamma = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=float)
    occupancy = compute_fractional_occupancy(gamma)
    assert np.allclose(occupancy, np.asarray([0.5, 0.5], dtype=float))


def test_compute_normalized_state_entropy_matches_closed_form() -> None:
    gamma = np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=float)
    normalized_entropy = compute_normalized_state_entropy(gamma)
    assert np.isclose(normalized_entropy[0], 1.0, atol=1e-8)
    assert np.isclose(normalized_entropy[1], 0.0, atol=1e-8)


def test_compute_dwell_times_extracts_simple_runs() -> None:
    dwell_times = compute_dwell_times(np.asarray([0, 0, 1, 1, 1, 0], dtype=int), sampling_rate_hz=2.0)
    assert dwell_times["state"].tolist() == [0, 1, 0]
    assert dwell_times["dwell_samples"].tolist() == [2, 3, 1]
    assert np.allclose(dwell_times["dwell_seconds"].to_numpy(dtype=float), np.asarray([1.0, 1.5, 0.5]))


def test_compute_empirical_transition_matrix_counts_transitions_correctly() -> None:
    transitions = compute_empirical_transition_matrix(np.asarray([0, 0, 1, 1, 0], dtype=int), n_states=2)
    expected = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    assert np.allclose(transitions, expected)


def test_compute_subject_run_fractional_occupancy_aggregates_by_group() -> None:
    gamma = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    gamma_index = pd.DataFrame(
        {
            "processed_index": [0, 1, 2, 3],
            "subject": ["s1", "s1", "s2", "s2"],
            "run": ["r1", "r1", "r1", "r1"],
            "original_sample": [0, 1, 0, 1],
        }
    )
    occupancy = compute_subject_run_fractional_occupancy(gamma, gamma_index)
    s1 = occupancy.loc[occupancy["subject"] == "s1", "fractional_occupancy"].to_numpy(dtype=float)
    s2 = occupancy.loc[occupancy["subject"] == "s2", "fractional_occupancy"].to_numpy(dtype=float)
    assert np.allclose(s1, np.asarray([0.75, 0.25], dtype=float))
    assert np.allclose(s2, np.asarray([0.125, 0.875], dtype=float))


def test_load_qc_inputs_and_pca_plot_skip_gracefully_when_optional_files_missing(tmp_path: Path) -> None:
    output_dir = _write_minimal_qc_inputs(tmp_path)
    qc_inputs = load_qc_inputs(output_dir)
    written = plot_pca_explained_variance(qc_inputs, output_dir / "qc")
    assert written == []
    assert any("pca_explained_variance.csv" in warning for warning in qc_inputs.warnings)


def test_compute_chunk_durations_adds_samples_and_seconds() -> None:
    chunks = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "run": ["r1", "r1"],
            "chunk_id": [0, 1],
            "original_start_sample": [0, 10],
            "original_stop_sample": [5, 16],
        }
    )
    durations = compute_chunk_durations(chunks, sampling_rate_hz=10.0)
    assert durations["duration_samples"].tolist() == [5, 6]
    assert np.allclose(durations["duration_seconds"].to_numpy(dtype=float), np.asarray([0.5, 0.6]))


def test_compute_delta_free_energy_uses_previous_k() -> None:
    model_selection = pd.DataFrame({"k": [2, 4, 3], "free_energy": [10.0, 8.0, 9.0]})
    delta = compute_delta_free_energy(model_selection)
    assert delta["k"].tolist() == [2, 3, 4]
    assert np.isnan(delta["delta_free_energy"].iloc[0])
    assert np.isclose(delta["delta_free_energy"].iloc[1], -1.0)
    assert np.isclose(delta["delta_free_energy"].iloc[2], -1.0)


def test_run_all_qc_plots_writes_report_with_missing_optional_inputs(tmp_path: Path) -> None:
    output_dir = _write_minimal_qc_inputs(tmp_path)
    artifacts = run_all_qc_plots(output_dir)
    report_path = artifacts["report"]
    assert isinstance(report_path, Path)
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Missing optional input" in report_text
    assert (output_dir / "qc" / "qc_chunk_length_histogram.png").exists()
    assert (output_dir / "qc" / "qc_state_fractional_occupancy.png").exists()
