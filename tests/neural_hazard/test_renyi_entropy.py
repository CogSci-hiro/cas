from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.neural_hazard.renyi import compute_renyi_entropy, discover_state_probability_columns
from cas.neural_hazard.fpp_spp_renyi_alpha_pipeline import (
    NeuralHazardFppSppRenyiAlphaConfig,
    _compute_renyi_entropy_allow_missing_rows,
    _load_neural_features_with_state_probabilities,
    run_neural_hazard_fpp_spp_renyi_alpha_pipeline,
)


def test_uniform_distribution_is_log_k_for_all_alpha() -> None:
    k = 5
    probs = np.full((3, k), 1.0 / k)
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        h = compute_renyi_entropy(probs, alpha)
        assert np.allclose(h, np.log(k), atol=1e-10)


def test_deterministic_distribution_is_zero_like() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    for alpha in (0.25, 1.0, 2.0, 3.0):
        h = compute_renyi_entropy(probs, alpha)
        assert np.all(h < 1e-2)


def test_alpha_one_matches_shannon() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.5, 0.25, 0.25]], dtype=float)
    expected = -np.sum(probs * np.log(probs), axis=1)
    got = compute_renyi_entropy(probs, 1.0)
    assert np.allclose(got, expected, atol=1e-12)


def test_alpha_near_one_uses_shannon_path() -> None:
    probs = np.array([[0.7, 0.2, 0.1]], dtype=float)
    h1 = compute_renyi_entropy(probs, 1.0)
    hclose = compute_renyi_entropy(probs, 1.0 + 1e-8, alpha_one_tolerance=1e-6)
    assert np.allclose(h1, hclose, atol=1e-12)


def test_higher_alpha_downweights_tails() -> None:
    probs = np.array([[0.9, 0.09, 0.01]], dtype=float)
    h_low = compute_renyi_entropy(probs, 0.5)[0]
    h_high = compute_renyi_entropy(probs, 3.0)[0]
    assert h_high < h_low


def test_invalid_rows_raise() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compute_renyi_entropy(np.array([[0.5, np.nan]]), 1.0)
    with pytest.raises(ValueError, match="finite positive sum"):
        compute_renyi_entropy(np.array([[0.0, 0.0]]), 1.0)


def test_discover_state_probability_columns() -> None:
    df = pd.DataFrame(columns=["state_probability_1", "state_prob_2", "gamma_3", "entropy"])
    cols = discover_state_probability_columns(df)
    assert cols == ["gamma_3", "state_prob_2", "state_probability_1"]


def test_compute_renyi_entropy_allow_missing_rows_preserves_nan_rows() -> None:
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    out = _compute_renyi_entropy_allow_missing_rows(
        probs,
        alpha=1.0,
        epsilon=1e-12,
        alpha_one_tolerance=1e-6,
    )

    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert out[2] < 1e-2


def test_load_neural_features_with_state_probabilities_uses_existing_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = pd.DataFrame({"subject_id": ["sub-001"], "run_id": ["1"], "time_s": [0.0], "state_probability_1": [1.0]})
    monkeypatch.setattr(
        "cas.neural_hazard.fpp_spp_renyi_alpha_pipeline.base._load_table",
        lambda path: existing.copy(),
    )

    out = _load_neural_features_with_state_probabilities(
        neural_features_path=tmp_path / "features.parquet",
        raw_config={"paths": {}},
        project_root=tmp_path,
        derivatives_root=None,
        scratch_dir=tmp_path / "scratch",
    )

    assert out.equals(existing)


def test_load_neural_features_with_state_probabilities_rebuilds_from_glhmm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stale = pd.DataFrame({"subject_id": ["sub-001"], "run_id": ["1"], "time_s": [0.0], "entropy": [0.5]})
    enriched = pd.DataFrame(
        {
            "subject_id": ["sub-001"],
            "run_id": ["1"],
            "time_s": [0.0],
            "entropy": [0.5],
            "state_probability_1": [0.8],
            "state_probability_2": [0.2],
        }
    )
    glhmm_dir = tmp_path / "models" / "glhmm"
    glhmm_dir.mkdir(parents=True)

    def fake_load_table(path: Path) -> pd.DataFrame:
        if path == tmp_path / "features.parquet":
            return stale.copy()
        if path.parent == tmp_path / "scratch":
            return enriched.copy()
        raise AssertionError(f"Unexpected load path: {path}")

    captured: dict[str, Path] = {}

    def fake_build(glhmm_output_dir: Path, output_path: Path, *, instability_window_s: float) -> Path:
        captured["glhmm_output_dir"] = glhmm_output_dir
        captured["output_path"] = output_path
        captured["instability_window_s"] = instability_window_s
        output_path.write_bytes(b"stub")
        return output_path

    monkeypatch.setattr(
        "cas.neural_hazard.fpp_spp_renyi_alpha_pipeline.base._load_table",
        fake_load_table,
    )
    monkeypatch.setattr(
        "cas.neural_hazard.fpp_spp_renyi_alpha_pipeline.base.build_entropy_features_table_from_glhmm_output",
        fake_build,
    )

    out = _load_neural_features_with_state_probabilities(
        neural_features_path=tmp_path / "features.parquet",
        raw_config={"paths": {}},
        project_root=tmp_path,
        derivatives_root=None,
        scratch_dir=tmp_path / "scratch",
    )

    assert out.equals(enriched)
    assert captured["glhmm_output_dir"] == glhmm_dir
    assert captured["output_path"].parent == tmp_path / "scratch"
    assert captured["instability_window_s"] == pytest.approx(0.25)


def _make_renyi_test_neural_table() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows: list[dict[str, object]] = []
    subjects = ["sub-01", "sub-02", "sub-03"]
    runs = ["run-1", "run-2"]
    for subject_index, subject_id in enumerate(subjects):
        for run_index, run_id in enumerate(runs):
            times = np.round(np.arange(0.0, 2.61, 0.1), 3)
            for time_s in times:
                p1 = 0.55 + 0.25 * np.sin(2.4 * time_s + (0.3 * subject_index))
                p2 = 0.30 + 0.18 * np.cos(1.6 * time_s + (0.25 * run_index))
                p3 = max(1.0 - p1 - p2, 0.02)
                total = p1 + p2 + p3
                p1, p2, p3 = p1 / total, p2 / total, p3 / total
                entropy = float(compute_renyi_entropy(np.array([[p1, p2, p3]], dtype=float), 1.0)[0])
                latent = 0.9 * np.sin(2.0 * time_s + 0.2 * subject_index) + 0.4 * np.cos(1.1 * time_s + 0.15 * run_index)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "time_s": time_s,
                        "entropy": entropy,
                        "signal_variance": latent + float(rng.normal(0.0, 0.15)),
                        "broadband_power": 0.7 * latent + float(rng.normal(0.0, 0.15)),
                        "state_switching_rate": -0.4 * latent + float(rng.normal(0.0, 0.15)),
                        "model_badness": 0.5 * latent + float(rng.normal(0.0, 0.15)),
                        "state_probability_1": p1,
                        "state_probability_2": p2,
                        "state_probability_3": p3,
                    }
                )
    return pd.DataFrame(rows)


def _lookup_entropy_at(neural: pd.DataFrame, subject_id: str, run_id: str, lookup_time_s: float) -> float:
    subset = neural.loc[
        (neural["subject_id"].astype(str) == subject_id)
        & (neural["run_id"].astype(str) == run_id)
    ].copy()
    distances = np.abs(pd.to_numeric(subset["time_s"], errors="coerce").to_numpy(dtype=float) - float(lookup_time_s))
    row = subset.iloc[int(np.argmin(distances))]
    return float(row["entropy"])


def _make_renyi_test_riskset(anchor_type: str, neural: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(23 if anchor_type == "FPP" else 29)
    rows: list[dict[str, object]] = []
    anchor_beta = 1.05 if anchor_type == "FPP" else 0.55
    intercept = -1.9 if anchor_type == "FPP" else -2.0
    episode_index = 0
    for subject_id in ("sub-01", "sub-02", "sub-03"):
        for run_id in ("run-1", "run-2"):
            episode_id = f"{anchor_type.lower()}-ep-{episode_index:03d}"
            episode_index += 1
            for bin_index in range(20):
                bin_start_s = 0.30 + (0.1 * bin_index)
                bin_end_s = bin_start_s + 0.1
                bin_center_s = (bin_start_s + bin_end_s) / 2.0
                entropy_lag_50 = _lookup_entropy_at(neural, subject_id, run_id, bin_center_s - 0.05)
                entropy_lag_200 = _lookup_entropy_at(neural, subject_id, run_id, bin_center_s - 0.20)
                logit = (
                    intercept
                    + 0.35 * bin_start_s
                    + 0.20 * (bin_end_s - 0.55)
                    + anchor_beta * entropy_lag_50
                    + 0.25 * entropy_lag_200
                )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "dyad_id": f"dyad-{subject_id[-2:]}",
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "bin_start_s": round(bin_start_s, 3),
                        "bin_end_s": round(bin_end_s, 3),
                        "event_bin": int(rng.uniform() < (1.0 / (1.0 + np.exp(-logit)))),
                        "time_from_partner_onset_s": round(bin_start_s, 3),
                        "time_from_partner_offset_s": round(bin_end_s - 0.55, 3),
                        "time_within_run_s": round(bin_center_s, 3),
                    }
                )
    riskset = pd.DataFrame(rows)
    if int(riskset["event_bin"].sum()) == 0:
        riskset.loc[riskset.index[::7], "event_bin"] = 1
    if int((riskset["event_bin"] == 0).sum()) == 0:
        riskset.loc[riskset.index[::9], "event_bin"] = 0
    return riskset


def test_renyi_alpha_pipeline_writes_validation_outputs(tmp_path: Path) -> None:
    neural = _make_renyi_test_neural_table()
    fpp = _make_renyi_test_riskset("FPP", neural)
    spp = _make_renyi_test_riskset("SPP", neural)
    neural_path = tmp_path / "neural.parquet"
    fpp_path = tmp_path / "fpp.parquet"
    spp_path = tmp_path / "spp.parquet"
    neural.to_parquet(neural_path, index=False)
    fpp.to_parquet(fpp_path, index=False)
    spp.to_parquet(spp_path, index=False)

    raw_config = {
        "paths": {
            "fpp_risk_set": str(fpp_path),
            "spp_risk_set": str(spp_path),
            "neural_features": str(neural_path),
            "out_dir": str(tmp_path / "out"),
        },
        "analysis": {
            "bin_width_s": 0.1,
            "lag_grid_ms": [0, 50, 100, 150, 200],
            "avoid_zero_lag": True,
            "timing_zscore_scope": "global",
            "feature_zscore_scope": "subject_run",
        },
        "renyi": {
            "alpha_grid": [0.05, 0.25, 1.0, 2.0],
            "alpha_one_tolerance": 1.0e-6,
            "probability_epsilon": 1.0e-12,
            "edge_warning_enabled": True,
            "fixed_lag_sensitivity_ms": [50, 200],
        },
        "pca": {
            "input_columns": ["signal_variance", "broadband_power", "state_switching_rate", "model_badness"],
            "n_components": 1,
            "minimum_pc1_explained_variance": 0.40,
        },
        "circular_shift": {
            "enabled": True,
            "n_permutations": 5,
            "minimum_shift_s": 0.3,
            "random_seed": 12345,
            "run_for_best_alpha_only": True,
            "run_for_shannon": True,
            "shift_scope": "subject_run",
            "shift_entropy_only": True,
            "p_value_mode_interaction_beta": "two_sided",
        },
        "diagnostics": {"enabled": True},
        "motor_proximal_sensitivity": {
            "enabled": True,
            "minimum_lag_ms": 150,
            "run_circular_shift_for_best": False,
        },
        "plots": {"alpha_axis_scale": "linear"},
    }

    result = run_neural_hazard_fpp_spp_renyi_alpha_pipeline(
        NeuralHazardFppSppRenyiAlphaConfig(raw=raw_config, config_path=tmp_path / "config.yaml")
    )

    assert result.summary_json_path.exists()
    out_dir = tmp_path / "out"
    assert (out_dir / "tables" / "renyi_circular_shift_null.csv").exists()
    assert (out_dir / "tables" / "renyi_circular_shift_summary.csv").exists()
    assert (out_dir / "tables" / "renyi_entropy_descriptives.csv").exists()
    assert (out_dir / "tables" / "renyi_alpha_entropy_correlation_raw.csv").exists()
    assert (out_dir / "tables" / "renyi_same_lag_model_comparison.csv").exists()
    assert (out_dir / "tables" / "renyi_motor_exclusion_alpha_search_summary.csv").exists()
    assert (out_dir / "figures" / "renyi_alpha_entropy_correlation_heatmap.png").exists()
    assert (out_dir / "figures" / "renyi_same_lag_delta_loglik.png").exists()
    assert (out_dir / "figures" / "renyi_motor_exclusion_alpha_search_delta_loglik.png").exists()
    assert (out_dir / "figures" / "renyi_best_alpha_circular_shift_delta_loglik.png").exists()
    assert (out_dir / "figures" / "renyi_shannon_circular_shift_delta_loglik.png").exists()

    null_summary = pd.read_csv(out_dir / "tables" / "renyi_circular_shift_summary.csv")
    assert {"alpha", "alpha_label", "selected_lag_ms", "n_permutations_successful", "permutation_p_value_delta_loglik"}.issubset(
        null_summary.columns
    )
    assert int(null_summary["n_permutations_successful"].min()) >= 1
    assert null_summary["permutation_p_value_delta_loglik"].notna().all()

    null_rows = pd.read_csv(out_dir / "tables" / "renyi_circular_shift_null.csv")
    assert set(null_rows["alpha_label"].astype(str)) >= {"alpha_1p0"}

    shannon_validation = pd.read_csv(out_dir / "tables" / "renyi_shannon_validation.csv")
    assert bool(shannon_validation["existing_entropy_present"].iloc[0])
    assert int(shannon_validation["n_compared"].iloc[0]) > 0
    assert float(shannon_validation["max_abs_difference_alpha1_vs_existing_entropy"].iloc[0]) < 1e-10

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["best_alpha_circular_shift_p_value_delta_loglik"] is not None
    assert summary["shannon_circular_shift_p_value_delta_loglik"] is not None
    assert summary["best_alpha_circular_shift_n_successful"] >= 1
    assert summary["shannon_circular_shift_n_successful"] >= 1
    assert "motor_exclusion_best_alpha" in summary
    assert "same_lag_best_alpha_selected_lag_best_alpha" in summary
