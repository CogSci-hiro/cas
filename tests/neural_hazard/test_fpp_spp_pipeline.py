from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.neural_hazard import NeuralHazardFppSppConfig, run_neural_hazard_fpp_spp_pipeline
from cas.neural_hazard.fpp_spp_pipeline import build_entropy_features_table_from_glhmm_output


def _make_neural_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    subjects = ["sub-01", "sub-02", "sub-03"]
    runs = ["run-1", "run-2"]
    for subject_index, subject_id in enumerate(subjects):
        for run_index, run_id in enumerate(runs):
            times = np.round(np.arange(0.0, 2.61, 0.1), 3)
            for time_s in times:
                entropy = (
                    0.8 * np.sin(2.2 * time_s + 0.5 * subject_index)
                    + 0.4 * np.cos(1.3 * time_s + 0.2 * run_index)
                    + float(rng.normal(0.0, 0.15))
                )
                rows.append(
                    {
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "time_s": time_s,
                        "entropy": entropy,
                        "signal_variance": 0.7 * entropy + float(rng.normal(0.0, 0.20)),
                        "broadband_power": 0.5 * entropy + float(rng.normal(0.0, 0.20)),
                        "state_switching_rate": -0.4 * entropy + float(rng.normal(0.0, 0.20)),
                        "model_badness": 0.6 * entropy + float(rng.normal(0.0, 0.20)),
                    }
                )
    return pd.DataFrame(rows)


def _lookup_entropy(neural: pd.DataFrame, subject_id: str, run_id: str, lookup_time_s: float) -> float:
    subset = neural.loc[
        (neural["subject_id"].astype(str) == subject_id) & (neural["run_id"].astype(str) == run_id)
    ].copy()
    distances = np.abs(pd.to_numeric(subset["time_s"], errors="coerce").to_numpy(dtype=float) - float(lookup_time_s))
    row = subset.iloc[int(np.argmin(distances))]
    return float(row["entropy"])


def _make_riskset(anchor_type: str, neural: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11 if anchor_type == "FPP" else 13)
    rows: list[dict[str, object]] = []
    anchor_entropy_beta = 1.20 if anchor_type == "FPP" else 0.30
    anchor_intercept = -2.1 if anchor_type == "FPP" else -2.2
    episode_index = 0
    for subject_id in ("sub-01", "sub-02", "sub-03"):
        for run_id in ("run-1", "run-2"):
            episode_id = f"{anchor_type.lower()}-ep-{episode_index:03d}"
            episode_index += 1
            for bin_index in range(20):
                bin_start_s = 0.30 + (0.1 * bin_index)
                bin_end_s = bin_start_s + 0.1
                bin_center_s = (bin_start_s + bin_end_s) / 2.0
                entropy_lag_100 = _lookup_entropy(neural, subject_id, run_id, bin_center_s - 0.1)
                entropy_lag_0 = _lookup_entropy(neural, subject_id, run_id, bin_center_s)
                time_from_partner_onset_s = bin_start_s
                time_from_partner_offset_s = bin_end_s - 0.55
                time_within_run_s = bin_center_s
                logit = (
                    anchor_intercept
                    + 0.35 * time_from_partner_onset_s
                    + 0.25 * time_from_partner_offset_s
                    + 0.20 * entropy_lag_0
                    + anchor_entropy_beta * entropy_lag_100
                )
                event_probability = 1.0 / (1.0 + np.exp(-logit))
                event_bin = int(rng.uniform() < event_probability)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "dyad_id": f"dyad-{subject_id[-2:]}",
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "bin_start_s": round(bin_start_s, 3),
                        "bin_end_s": round(bin_end_s, 3),
                        "event_bin": event_bin,
                        "time_from_partner_onset_s": round(time_from_partner_onset_s, 3),
                        "time_from_partner_offset_s": round(time_from_partner_offset_s, 3),
                        "time_within_run_s": round(time_within_run_s, 3),
                    }
                )
    riskset = pd.DataFrame(rows)
    if int(riskset["event_bin"].sum()) == 0:
        riskset.loc[riskset.index[::9], "event_bin"] = 1
    if int((riskset["event_bin"] == 0).sum()) == 0:
        riskset.loc[riskset.index[::7], "event_bin"] = 0
    return riskset


def test_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    neural = _make_neural_table()
    fpp = _make_riskset("FPP", neural)
    spp = _make_riskset("SPP", neural)
    fpp_path = tmp_path / "fpp.parquet"
    spp_path = tmp_path / "spp.parquet"
    neural_path = tmp_path / "neural.parquet"
    fpp.to_parquet(fpp_path, index=False)
    spp.to_parquet(spp_path, index=False)
    neural.to_parquet(neural_path, index=False)

    config = NeuralHazardFppSppConfig(
        fpp_risk_set_path=fpp_path,
        spp_risk_set_path=spp_path,
        neural_features_path=neural_path,
        out_dir=tmp_path / "out",
        bin_width_s=0.1,
        lag_grid_ms=(0, 100, 200),
        pca_input_columns=("signal_variance", "broadband_power", "state_switching_rate", "model_badness"),
        nearest_merge_tolerance_s=0.051,
        n_circular_shift_permutations=6,
        minimum_circular_shift_duration_s=0.3,
    )

    result = run_neural_hazard_fpp_spp_pipeline(config)

    assert result.model_comparison_path.exists()
    assert result.coefficients_path.exists()
    assert result.circular_shift_summary_path.exists()
    assert result.summary_json_path.exists()
    assert (tmp_path / "out" / "tables" / "risk_set_summary.csv").exists()
    assert (tmp_path / "out" / "tables" / "fpp_lag_selection.csv").exists()
    assert (tmp_path / "out" / "tables" / "selected_entropy_lag.csv").exists()
    assert (tmp_path / "out" / "figures" / "predicted_hazard_by_entropy_anchor_type.png").exists()
    assert (tmp_path / "out" / "figures" / "circular_shift_null_delta_loglik.png").exists()

    selected_lag = pd.read_csv(tmp_path / "out" / "tables" / "selected_entropy_lag.csv")
    assert int(selected_lag["selected_lag_ms"].iloc[0]) in {100, 200}

    coefficients = pd.read_csv(result.coefficients_path)
    assert any("entropy_lag_" in term for term in coefficients["term"].astype(str))
    assert any("C(anchor_type)" in term and "entropy_lag_" in term for term in coefficients["term"].astype(str))

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["selected_lag_ms"] in {100, 200}
    assert summary["n_rows"] == len(fpp) + len(spp)
    assert summary["n_events_total"] >= 1


def test_pipeline_maps_ab_speakers_to_canonical_subject_ids(tmp_path: Path) -> None:
    neural = _make_neural_table()
    fpp = _make_riskset("FPP", neural)
    spp = _make_riskset("SPP", neural)
    speaker_mapping = {"sub-01": ("dyad-001", "A"), "sub-02": ("dyad-001", "B"), "sub-03": ("dyad-002", "A")}
    for table in (fpp, spp):
        original_subject = table["subject_id"].astype(str).copy()
        table["dyad_id"] = original_subject.map(lambda value: speaker_mapping[str(value)][0])
        table["subject_id"] = original_subject.map(lambda value: speaker_mapping[str(value)][1])
    fpp_path = tmp_path / "fpp_ab.parquet"
    spp_path = tmp_path / "spp_ab.parquet"
    neural_path = tmp_path / "neural.parquet"
    dyads_path = tmp_path / "dyads.csv"
    fpp.to_parquet(fpp_path, index=False)
    spp.to_parquet(spp_path, index=False)
    neural.to_parquet(neural_path, index=False)
    pd.DataFrame(
        [
            {"dyad_id": "dyad-001", "subject_id": "sub-01", "partner_id": "sub-02"},
            {"dyad_id": "dyad-001", "subject_id": "sub-02", "partner_id": "sub-01"},
            {"dyad_id": "dyad-002", "subject_id": "sub-03", "partner_id": "sub-04"},
            {"dyad_id": "dyad-002", "subject_id": "sub-04", "partner_id": "sub-03"},
        ]
    ).to_csv(dyads_path, index=False)

    config = NeuralHazardFppSppConfig(
        fpp_risk_set_path=fpp_path,
        spp_risk_set_path=spp_path,
        neural_features_path=neural_path,
        out_dir=tmp_path / "out_ab",
        bin_width_s=0.1,
        lag_grid_ms=(0, 100),
        pca_input_columns=("signal_variance", "broadband_power", "state_switching_rate", "model_badness"),
        nearest_merge_tolerance_s=0.051,
        n_circular_shift_permutations=2,
        minimum_circular_shift_duration_s=0.3,
        dyads_csv_path=dyads_path,
    )

    result = run_neural_hazard_fpp_spp_pipeline(config)

    assert result.model_comparison_path.exists()
    model_comparison = pd.read_csv(result.model_comparison_path)
    model_rows = model_comparison.loc[model_comparison["row_type"].astype(str) == "model"].copy()
    assert int(pd.to_numeric(model_rows["n_rows"], errors="coerce").min()) > 0


def test_pipeline_fails_if_requested_pca_inputs_are_missing(tmp_path: Path) -> None:
    neural = _make_neural_table().loc[:, ["subject_id", "run_id", "time_s", "entropy"]].copy()
    fpp = _make_riskset("FPP", _make_neural_table())
    spp = _make_riskset("SPP", _make_neural_table())
    fpp_path = tmp_path / "fpp.parquet"
    spp_path = tmp_path / "spp.parquet"
    neural_path = tmp_path / "neural_missing_pca.parquet"
    fpp.to_parquet(fpp_path, index=False)
    spp.to_parquet(spp_path, index=False)
    neural.to_parquet(neural_path, index=False)

    config = NeuralHazardFppSppConfig(
        fpp_risk_set_path=fpp_path,
        spp_risk_set_path=spp_path,
        neural_features_path=neural_path,
        out_dir=tmp_path / "out_missing_pca",
        bin_width_s=0.1,
        lag_grid_ms=(0, 100),
        pca_input_columns=("signal_variance", "broadband_power", "state_switching_rate", "model_badness"),
        nearest_merge_tolerance_s=0.051,
        n_circular_shift_permutations=2,
        minimum_circular_shift_duration_s=0.3,
    )

    with pytest.raises(ValueError, match="does not contain any of the configured instability PCA columns"):
        run_neural_hazard_fpp_spp_pipeline(config)


def test_build_entropy_features_table_adds_instability_columns(tmp_path: Path) -> None:
    glhmm_dir = tmp_path / "glhmm"
    glhmm_dir.mkdir()
    source_path = glhmm_dir / "source.npy"
    source = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.8, 0.1],
            [1.0, 0.6, 0.2],
            [0.8, 0.4, 0.4],
            [0.4, 0.2, 0.6],
            [0.1, 0.1, 0.8],
            [0.0, 0.2, 1.0],
            [0.2, 0.3, 0.7],
        ],
        dtype=float,
    )
    np.save(source_path, source)
    pd.DataFrame(
        [{"subject": "sub-001", "run": 1, "source_path": str(source_path), "speech_path": "unused"}]
    ).to_csv(glhmm_dir / "input_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject": "sub-001",
                "run": 1,
                "chunk_id": 0,
                "original_start_sample": 0,
                "original_stop_sample": 8,
                "concat_start_sample": 0,
                "concat_stop_sample": 8,
                "processed_start_sample": 0,
                "processed_stop_sample": 7,
            }
        ]
    ).to_csv(glhmm_dir / "chunks.csv", index=False)
    gamma = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.9, 0.1],
        ],
        dtype=float,
    )
    np.save(glhmm_dir / "gamma_k2.npy", gamma)
    (glhmm_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 2,
                "sampling_rate_hz": 4.0,
                "lags": [-1, 0],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "sample": np.arange(8, dtype=int),
            "time_s": np.arange(8, dtype=float) / 4.0,
            "state_entropy": [np.nan, 0.2, 0.3, 0.7, 0.6, 0.5, 0.4, 0.1],
        }
    ).to_csv(glhmm_dir / "subject-sub-001_run-1_state_entropy.csv", index=False)

    output_path = tmp_path / "features.parquet"
    written = build_entropy_features_table_from_glhmm_output(
        glhmm_dir,
        output_path,
        instability_window_s=0.5,
    )
    assert written.exists()
    table = pd.read_parquet(written)
    for column in (
        "entropy",
        "signal_variance",
        "broadband_power",
        "state_switching_rate",
        "model_badness",
    ):
        assert column in table.columns
        assert table[column].notna().sum() > 0
