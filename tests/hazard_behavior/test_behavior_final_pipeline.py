from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_event_positive_episodes
from cas.hazard_behavior.final_behavior import (
    _formula_sequence,
    _interaction_plot_grid,
    _interaction_formula,
    _normalize_final_riskset,
    _project_behavior_final_events,
    _prepare_final_riskset,
    build_fpp_vs_spp_report,
    build_information_effect_contrasts,
    lag_col_z,
    load_final_behavior_config,
    run_behavior_final_compare,
    run_behavior_final_fit,
    run_behavior_final_select_lag,
)
from cas.hazard_behavior.riskset import build_discrete_time_riskset


def _write_config_copy(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "behavior.yaml"
    cfg_path.write_text(Path("config/behavior.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return cfg_path


def _make_synthetic_riskset(anchor: str, *, seed: int = 0, n_episodes: int = 24, n_bins: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    anchor_shift = 0.35 if anchor == "fpp" else -0.05
    for episode_idx in range(n_episodes):
        dyad_id = f"dyad-{episode_idx % 6:02d}"
        subject_id = f"subj-{episode_idx % 8:02d}"
        run_id = str(1 + (episode_idx % 2))
        episode_id = f"{anchor}-ep-{episode_idx:03d}"
        for bin_idx in range(n_bins):
            bin_start_s = round(bin_idx * 0.05, 3)
            bin_end_s = round(bin_start_s + 0.05, 3)
            time_from_partner_onset_s = round(bin_idx * 0.05, 3)
            time_from_partner_offset_s = round(bin_end_s - 0.30, 3)
            info_rate = (
                -0.2
                + 0.12 * bin_idx
                + anchor_shift
                + 0.05 * np.sin((episode_idx + 1) * (bin_idx + 1))
                + float(rng.normal(0.0, 0.12))
            )
            prop_info = (
                0.1
                + 0.08 * bin_idx
                + 0.15 * (anchor == "fpp")
                + 0.04 * np.cos((episode_idx + 1) * (bin_idx + 2))
                + float(rng.normal(0.0, 0.08))
            )
            logit = (
                -2.0
                + 0.8 * time_from_partner_onset_s
                + 0.35 * time_from_partner_offset_s
                - 0.25 * (time_from_partner_offset_s ** 2)
                + (0.55 if anchor == "fpp" else 0.2) * info_rate
                + (0.25 if anchor == "fpp" else 0.05) * prop_info
            )
            probability = 1.0 / (1.0 + np.exp(-logit))
            event_bin = int(rng.uniform() < probability)
            rows.append(
                {
                    "episode_id": episode_id,
                    "dyad_id": dyad_id,
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "anchor_type": anchor,
                    "bin_start_s": bin_start_s,
                    "bin_end_s": bin_end_s,
                    "event_bin": event_bin,
                    "time_from_partner_onset_s": time_from_partner_onset_s,
                    "time_from_partner_offset_s": time_from_partner_offset_s,
                    "information_rate": info_rate,
                    "prop_expected_cumulative_info": prop_info,
                }
            )
    riskset = pd.DataFrame(rows)
    if int(riskset["event_bin"].sum()) == 0:
        riskset.loc[riskset.index[::7], "event_bin"] = 1
    if int((riskset["event_bin"] == 0).sum()) == 0:
        riskset.loc[riskset.index[::5], "event_bin"] = 0
    return riskset


def _make_synthetic_episodes(anchor: str, n_episodes: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": [f"{anchor}-ep-{episode_idx:03d}" for episode_idx in range(n_episodes)],
            "dyad_id": [f"dyad-{episode_idx % 6:02d}" for episode_idx in range(n_episodes)],
            "participant_speaker": ["A" if anchor == "fpp" else "B"] * n_episodes,
            "run": [str(1 + (episode_idx % 2)) for episode_idx in range(n_episodes)],
        }
    )


def test_config_loading() -> None:
    cfg = load_final_behavior_config(Path("config/behavior.yaml"))
    for key in ["analysis", "paths", "riskset", "anchors", "columns", "features", "lags", "models", "outputs", "figures"]:
        assert key in cfg.raw
    interaction_cfg = cfg.raw["models"]["timing_information_rate_interaction"]
    assert interaction_cfg["enabled"] is True
    assert interaction_cfg["compare_against"] == "full_information"
    assert interaction_cfg["fit_for_anchors"] == ["fpp", "spp"]


def test_riskset_schema_and_anchor_match() -> None:
    toy = pd.DataFrame(
        {
            "episode_id": ["e1"],
            "dyad_id": ["d1"],
            "participant_speaker": ["A"],
            "run": ["1"],
            "bin_start": [0.0],
            "bin_end": [0.05],
            "event": [0],
            "time_from_partner_onset": [0.0],
            "time_from_partner_offset": [0.0],
            "information_rate": [0.1],
            "prop_expected_cumulative_info": [0.2],
        }
    )
    fpp = _normalize_final_riskset(toy, "fpp")
    spp = _normalize_final_riskset(toy, "spp")
    required = {
        "episode_id",
        "dyad_id",
        "subject_id",
        "run_id",
        "anchor_type",
        "bin_start_s",
        "bin_end_s",
        "event_bin",
        "time_from_partner_onset_s",
        "time_from_partner_offset_s",
        "information_rate",
        "prop_expected_cumulative_info",
    }
    assert required.issubset(set(fpp.columns))
    assert set(fpp.columns) == set(spp.columns)


def test_primary_formulas_use_simple_timing_only() -> None:
    formulas = _formula_sequence(150)
    assert list(formulas) == [
        "timing_only",
        "information_rate",
        "full_information",
        "timing_information_rate_interaction",
    ]
    assert "b" + "s(" not in formulas["timing_only"]
    assert "z_time_from_partner_onset_s" in formulas["timing_only"]
    assert "z_time_from_partner_offset_s" in formulas["timing_only"]
    assert "z_time_from_partner_offset_s_squared" in formulas["timing_only"]
    assert "information_rate_lag_150ms_z" in formulas["information_rate"]
    assert "prop_expected_cumulative_info_lag_150ms_z" in formulas["full_information"]
    assert "z_time_from_partner_onset_s:information_rate_lag_150ms_z" in formulas["timing_information_rate_interaction"]
    assert "z_time_from_partner_offset_s:information_rate_lag_150ms_z" in formulas["timing_information_rate_interaction"]
    assert not any("SELECTED" in formula for formula in formulas.values())


def test_primary_pooled_formula_contains_anchor_interactions() -> None:
    formula = _interaction_formula(150)
    assert "b" + "s(" not in formula
    assert "z_time_from_partner_onset_s" in formula
    assert "z_time_from_partner_offset_s_squared" in formula
    assert "anchor_type * (" in formula
    assert "information_rate_lag_150ms_z" in formula
    assert "prop_expected_cumulative_info_lag_150ms_z" in formula


def test_interaction_model_compares_against_full_information() -> None:
    cfg = load_final_behavior_config(Path("config/behavior.yaml"))
    sequence = cfg.raw["models"]["sequence"]
    interaction_step = next(step for step in sequence if step["name"] == "timing_information_rate_interaction")
    assert interaction_step["compare_against"] == "full_information"


def test_offset_squared_column_matches_squared_offset_z() -> None:
    riskset, stats = _prepare_final_riskset(_make_synthetic_riskset("fpp", seed=1), [0, 50, 100])
    assert np.allclose(
        riskset["z_time_from_partner_offset_s_squared"].to_numpy(),
        riskset["z_time_from_partner_offset_s"].to_numpy() ** 2,
    )
    assert "z_time_from_partner_offset_s_squared" in stats["column"].astype(str).tolist()


def test_interaction_prediction_grid_uses_three_information_levels() -> None:
    riskset, _ = _prepare_final_riskset(_make_synthetic_riskset("fpp", seed=5), [0, 50, 100, 150])
    grid = _interaction_plot_grid(
        riskset,
        selected_lag_ms=150,
        varying_column="time_from_partner_onset_s",
        fixed_column="time_from_partner_offset_s",
    )
    assert set(grid["information_rate_z_level"].tolist()) == {-1.0, 0.0, 1.0}
    assert "information_rate_lag_150ms_z" in grid.columns
    assert "prop_expected_cumulative_info_lag_150ms_z" in grid.columns


def test_zero_variance_timing_raises_clear_error() -> None:
    toy = _make_synthetic_riskset("fpp", seed=2)
    toy["time_from_partner_offset_s"] = 1.0
    with pytest.raises(ValueError, match="zero or non-finite variance"):
        _prepare_final_riskset(toy, [0, 50, 100])


def test_lag_selection_uses_simple_timing_baseline(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config_copy(tmp_path)
    out_dir = tmp_path / "lag_selection"
    riskset = _make_synthetic_riskset("fpp", seed=11)
    episodes = _make_synthetic_episodes("fpp", 24)

    monkeypatch.setattr(
        "cas.hazard_behavior.final_behavior._build_anchor_riskset",
        lambda cfg, anchor, out_dir: (riskset.copy(), episodes.copy(), []),
    )

    class FakeFit:
        def __init__(self, bic: float, aic: float, formula: str):
            self.model_name = "full_information"
            self.formula = formula
            self.result = type("Result", (), {"params": pd.Series(dtype=float), "bse": pd.Series(dtype=float)})()
            self.n_rows = len(riskset)
            self.n_events = int(riskset["event_bin"].sum())
            self.converged = True
            self.log_likelihood = -10.0
            self.aic = aic
            self.bic = bic
            self.max_abs_coefficient = 0.0
            self.max_standard_error = 0.0
            self.any_nan_coefficients = False
            self.any_infinite_coefficients = False
            self.design_condition_number = 1.0
            self.overflow_warning = False
            self.stable = True
            self.fit_warnings = []
            self.safety_warnings = []

    def fake_fit(table: pd.DataFrame, model_name: str, formula: str):
        assert "b" + "s(" not in formula
        assert "z_time_from_partner_onset_s" in formula
        assert "z_time_from_partner_offset_s_squared" in formula
        if "lag_150ms_z" in formula:
            return FakeFit(90.0, 80.0, formula)
        if "lag_100ms_z" in formula:
            return FakeFit(95.0, 85.0, formula)
        if model_name == "timing_only":
            return FakeFit(120.0, 110.0, formula)
        return FakeFit(100.0, 90.0, formula)

    monkeypatch.setattr("cas.hazard_behavior.final_behavior._fit_final_model", fake_fit)

    selected_path = run_behavior_final_select_lag(cfg_path, out_dir)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    table = pd.read_csv(out_dir / "fpp_lag_selection_table.csv")
    assert selected["selected_lag_ms"] == 150
    assert selected["selection_anchor"] == "fpp"
    assert selected["spp_reselects_lag"] is False
    assert int(table.sort_values(["bic", "lag_ms"], kind="mergesort").iloc[0]["lag_ms"]) == 150


def test_selected_lag_reused_for_spp(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config_copy(tmp_path)
    riskset = _make_synthetic_riskset("spp", seed=21)
    episodes = _make_synthetic_episodes("spp", 24)
    selected_path = tmp_path / "selected_lag.json"
    selected_path.write_text(json.dumps({"selected_lag_ms": 150}), encoding="utf-8")

    monkeypatch.setattr(
        "cas.hazard_behavior.final_behavior._build_anchor_riskset",
        lambda cfg, anchor, out_dir: (riskset.copy(), episodes.copy(), []),
    )

    out_dir = tmp_path / "spp_control"
    run_behavior_final_fit(cfg_path, "spp", selected_path, out_dir)

    summary = pd.read_csv(out_dir / "models" / "model_summary.csv")
    diagnostics = json.loads((out_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert (out_dir / "riskset.parquet").exists()
    assert (out_dir / "episodes.csv").exists()
    assert (out_dir / "standardization_stats.csv").exists()
    assert diagnostics["lag_reselection_performed"] is False
    assert "SPP reuses the FPP-selected lag without reselection" in diagnostics["note"]
    assert any("information_rate_lag_150ms_z" in formula for formula in summary["formula"].astype(str))
    assert not any("information_rate_lag_100ms_z" in formula for formula in summary["formula"].astype(str))
    assert (out_dir / "models" / "timing_information_rate_interaction_summary.csv").exists()
    assert (out_dir / "models" / "timing_information_rate_interaction_coefficients.csv").exists()
    assert (out_dir / "models" / "timing_information_rate_interaction_comparison.csv").exists()
    assert (out_dir / "figures" / "timing_information_rate_interaction_onset.png").exists()
    assert (out_dir / "figures" / "timing_information_rate_interaction_offset.png").exists()


def test_contrast_table_computes_fpp_minus_spp_correctly() -> None:
    lag = 150
    rate = lag_col_z("information_rate", lag)
    prop = lag_col_z("prop_expected_cumulative_info", lag)
    summary = pd.DataFrame(
        {
            "term": [
                rate,
                f"anchor_type[T.fpp]:{rate}",
                prop,
                f"anchor_type[T.fpp]:{prop}",
            ],
            "estimate": [0.4, 0.3, 0.1, -0.2],
        }
    )
    covariance = pd.DataFrame(
        np.diag([0.04, 0.01, 0.09, 0.16]),
        index=summary["term"],
        columns=summary["term"],
    )
    contrasts = build_information_effect_contrasts(summary, covariance, info_rate_col=rate, prop_col=prop)
    rows = contrasts.set_index("contrast")
    assert np.isclose(rows.loc["SPP information_rate effect", "estimate"], 0.4)
    assert np.isclose(rows.loc["FPP information_rate effect", "estimate"], 0.7)
    assert np.isclose(rows.loc["FPP - SPP information_rate effect", "estimate"], 0.3)
    assert np.isclose(rows.loc["SPP prop_expected_cumulative_info effect", "estimate"], 0.1)
    assert np.isclose(rows.loc["FPP prop_expected_cumulative_info effect", "estimate"], -0.1)
    assert np.isclose(rows.loc["FPP - SPP prop_expected_cumulative_info effect", "estimate"], -0.2)


def test_report_warns_when_models_unstable() -> None:
    cfg = load_final_behavior_config(Path("config/behavior.yaml"))
    fpp_summary = pd.DataFrame(
        [
            {
                "model_name": "full_information",
                "n_rows": 100,
                "n_events": 20,
                "bic": 101.0,
                "converged": True,
                "stable": True,
                "safety_warnings": "",
            }
        ]
    )
    spp_summary = pd.DataFrame(
        [
            {
                "model_name": "full_information",
                "n_rows": 90,
                "n_events": 18,
                "bic": 103.0,
                "converged": True,
                "stable": False,
                "safety_warnings": "Maximum standard error exceeded or equaled 10.",
            }
        ]
    )
    pooled_summary = pd.DataFrame(
        [
            {
                "model_name": "fpp_vs_spp_interaction",
                "n_rows": 190,
                "n_events": 38,
                "converged": True,
                "stable": False,
                "safety_warnings": "Maximum standard error exceeded or equaled 10.",
            }
        ]
    )
    pooled_coefficients = pd.DataFrame([{"term": "x", "estimate": 0.1}])
    contrasts = pd.DataFrame(
        [
            {
                "contrast": "FPP - SPP information_rate effect",
                "estimate": 0.0,
                "standard_error": 99.0,
                "z": 0.0,
                "p_value": 1.0,
                "conf_low": -194.0,
                "conf_high": 194.0,
            }
        ]
    )
    markdown, payload = build_fpp_vs_spp_report(
        cfg=cfg,
        selected_lag_ms=150,
        fpp_summary=fpp_summary,
        spp_summary=spp_summary,
        pooled_summary=pooled_summary,
        pooled_coefficients=pooled_coefficients,
        contrasts=contrasts,
    )
    assert "Primary timing control: linear/quadratic parametric timing" in markdown
    assert "SPP models stable: False" in markdown
    assert payload["stability"]["spp_all_stable"] is False
    assert payload["stability"]["pooled_contrasts_interpretable"] is False


def test_compare_writes_expected_outputs(tmp_path: Path) -> None:
    cfg_path = _write_config_copy(tmp_path)
    lag = 150
    fpp, _ = _prepare_final_riskset(_make_synthetic_riskset("fpp", seed=31), [0, 50, 100, 150, 200, 250, 300, 400, 500])
    spp, _ = _prepare_final_riskset(_make_synthetic_riskset("spp", seed=32), [0, 50, 100, 150, 200, 250, 300, 400, 500])

    fpp_dir = tmp_path / "fpp"
    spp_dir = tmp_path / "spp_control"
    cmp_dir = tmp_path / "fpp_vs_spp"
    (fpp_dir / "models").mkdir(parents=True)
    (spp_dir / "models").mkdir(parents=True)
    fpp.to_parquet(fpp_dir / "riskset.parquet", index=False)
    spp.to_parquet(spp_dir / "riskset.parquet", index=False)

    fpp_summary = pd.DataFrame(
        [{"model_name": name, "n_rows": len(fpp), "n_events": int(fpp["event_bin"].sum()), "bic": 100.0 + idx, "converged": True, "stable": True, "safety_warnings": ""} for idx, name in enumerate(("timing_only", "information_rate", "full_information", "timing_information_rate_interaction"))]
    )
    spp_summary = pd.DataFrame(
        [{"model_name": name, "n_rows": len(spp), "n_events": int(spp["event_bin"].sum()), "bic": 101.0 + idx, "converged": True, "stable": True, "safety_warnings": ""} for idx, name in enumerate(("timing_only", "information_rate", "full_information", "timing_information_rate_interaction"))]
    )
    fpp_summary.to_csv(fpp_dir / "models" / "model_summary.csv", index=False)
    spp_summary.to_csv(spp_dir / "models" / "model_summary.csv", index=False)

    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"selected_lag_ms": lag}), encoding="utf-8")
    run_behavior_final_compare(cfg_path, selected_path, fpp_dir / "riskset.parquet", spp_dir / "riskset.parquet", cmp_dir)

    coef_table = pd.read_csv(cmp_dir / "interaction_coefficients.csv")
    report_json = json.loads((cmp_dir / "fpp_vs_spp_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((cmp_dir / "qc_plot_manifest.json").read_text(encoding="utf-8"))
    assert (cmp_dir / "combined_riskset.parquet").exists()
    assert (cmp_dir / "interaction_model_summary.csv").exists()
    assert (cmp_dir / "information_effect_contrasts.csv").exists()
    assert (cmp_dir / "timing_information_rate_anchor_interaction_summary.csv").exists()
    assert (cmp_dir / "timing_information_rate_anchor_interaction_coefficients.csv").exists()
    assert (cmp_dir / "timing_information_rate_anchor_interaction_contrasts.csv").exists()
    assert (cmp_dir / "fpp_vs_spp_report.md").exists()
    assert "anchor_type" in report_json["formulas"]["pooled_interaction"]
    assert any("anchor_type" in term and "information_rate_lag_150ms_z" in term for term in coef_table["term"].astype(str))
    assert any("anchor_type" in term and "prop_expected_cumulative_info_lag_150ms_z" in term for term in coef_table["term"].astype(str))
    assert "information_effect_contrasts_forest" in manifest


def test_anchor_specific_event_projection_assigns_distinct_fpp_and_spp_bins(tmp_path: Path) -> None:
    config = BehaviourHazardConfig(
        events_path=tmp_path / "events.csv",
        surprisal_paths=(tmp_path / "tokens.tsv",),
        out_dir=tmp_path / "out",
        include_censored=False,
        bin_size_s=0.05,
    )
    spp_config = BehaviourHazardConfig(
        events_path=tmp_path / "events.csv",
        surprisal_paths=(tmp_path / "tokens.tsv",),
        out_dir=tmp_path / "out_spp",
        include_censored=False,
        bin_size_s=0.05,
        target_fpp_label_prefix="SPP_",
    )
    paired_events = pd.DataFrame(
        [
            {
                "recording_id": "dyad-001",
                "run": "1",
                "speaker_fpp": "B",
                "speaker_spp": "A",
                "fpp_label": "FPP_TEST",
                "spp_label": "SPP_TEST",
                "fpp_onset": 1.00,
                "fpp_offset": 1.10,
                "spp_onset": 1.40,
                "spp_offset": 1.50,
                "pair_id": "pair-001",
            }
        ]
    )
    surprisal = pd.DataFrame(
        [
            {
                "dyad_id": "dyad-001",
                "run": "1",
                "speaker": "A",
                "onset": 0.80,
                "offset": 0.95,
                "duration": 0.15,
                "word": "a",
                "surprisal": 1.0,
                "alignment_status": "ok",
            },
            {
                "dyad_id": "dyad-001",
                "run": "1",
                "speaker": "B",
                "onset": 1.00,
                "offset": 1.10,
                "duration": 0.10,
                "word": "b",
                "surprisal": 1.0,
                "alignment_status": "ok",
            },
            {
                "dyad_id": "dyad-001",
                "run": "1",
                "speaker": "B",
                "onset": 1.20,
                "offset": 1.35,
                "duration": 0.15,
                "word": "c",
                "surprisal": 1.0,
                "alignment_status": "ok",
            },
            {
                "dyad_id": "dyad-001",
                "run": "1",
                "speaker": "A",
                "onset": 1.40,
                "offset": 1.50,
                "duration": 0.10,
                "word": "d",
                "surprisal": 1.0,
                "alignment_status": "ok",
            },
        ]
    )

    fpp_events = _project_behavior_final_events(paired_events, anchor="fpp")
    spp_events = _project_behavior_final_events(paired_events, anchor="spp")
    fpp_episodes = build_event_positive_episodes(events_table=fpp_events, surprisal_table=surprisal, config=config).episodes
    spp_episodes = build_event_positive_episodes(events_table=spp_events, surprisal_table=surprisal, config=spp_config).episodes
    fpp_riskset = build_discrete_time_riskset(fpp_episodes, config=config).riskset_table
    spp_riskset = build_discrete_time_riskset(spp_episodes, config=spp_config).riskset_table

    assert int(fpp_riskset["event"].sum()) == 1
    assert int(spp_riskset["event"].sum()) == 1

    fpp_event_row = fpp_riskset.loc[fpp_riskset["event"] == 1].iloc[0]
    spp_event_row = spp_riskset.loc[spp_riskset["event"] == 1].iloc[0]
    assert fpp_event_row["bin_start"] <= 1.00 < fpp_event_row["bin_end"]
    assert spp_event_row["bin_start"] <= 1.40 < spp_event_row["bin_end"]
    assert not (spp_event_row["bin_start"] <= 1.00 < spp_event_row["bin_end"])


def test_spp_zero_event_bins_raises_before_model_fit(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config_copy(tmp_path)
    selected_path = tmp_path / "selected_lag.json"
    selected_path.write_text(json.dumps({"selected_lag_ms": 150}), encoding="utf-8")
    bad_riskset = _make_synthetic_riskset("spp", seed=51)
    bad_riskset["event_bin"] = 0
    episodes = _make_synthetic_episodes("spp", 24)

    monkeypatch.setattr(
        "cas.hazard_behavior.final_behavior._build_anchor_riskset",
        lambda cfg, anchor, out_dir: (bad_riskset.copy(), episodes.copy(), []),
    )

    called = {"fit": 0}

    def fail_if_fit(*args, **kwargs):
        called["fit"] += 1
        raise AssertionError("model fitting should not run when SPP event bins are zero")

    monkeypatch.setattr("cas.hazard_behavior.final_behavior._fit_final_model", fail_if_fit)

    with pytest.raises(ValueError, match="zero event bins"):
        run_behavior_final_fit(cfg_path, "spp", selected_path, tmp_path / "spp_control")
    assert called["fit"] == 0


def test_report_uses_zero_spp_event_conclusion() -> None:
    cfg = load_final_behavior_config(Path("config/behavior.yaml"))
    fpp_summary = pd.DataFrame([{"model_name": "full_information", "n_rows": 10, "n_events": 3, "bic": 1.0, "converged": True, "stable": True, "safety_warnings": ""}])
    spp_summary = pd.DataFrame([{"model_name": "full_information", "n_rows": 10, "n_events": 0, "bic": 1.0, "converged": False, "stable": False, "safety_warnings": "zero events"}])
    pooled_summary = pd.DataFrame([{"model_name": "fpp_vs_spp_interaction", "n_rows": 20, "n_events": 3, "converged": False, "stable": False, "safety_warnings": "not fit"}])
    markdown, payload = build_fpp_vs_spp_report(
        cfg=cfg,
        selected_lag_ms=150,
        fpp_summary=fpp_summary,
        spp_summary=spp_summary,
        pooled_summary=pooled_summary,
        pooled_coefficients=pd.DataFrame([{"term": "x", "estimate": 0.0}]),
        contrasts=pd.DataFrame(columns=["contrast", "estimate", "standard_error", "z", "p_value", "conf_low", "conf_high"]),
    )
    expected = "SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins."
    assert expected in markdown
    assert payload["conclusion"] == expected


def test_compare_rejects_zero_spp_event_bins_before_pooled_fit(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config_copy(tmp_path)
    lag = 150
    fpp, _ = _prepare_final_riskset(_make_synthetic_riskset("fpp", seed=61), [0, 50, 100, 150])
    spp, _ = _prepare_final_riskset(_make_synthetic_riskset("spp", seed=62), [0, 50, 100, 150])
    spp["event_bin"] = 0

    fpp_dir = tmp_path / "fpp"
    spp_dir = tmp_path / "spp_control"
    cmp_dir = tmp_path / "fpp_vs_spp"
    (fpp_dir / "models").mkdir(parents=True)
    (spp_dir / "models").mkdir(parents=True)
    fpp.to_parquet(fpp_dir / "riskset.parquet", index=False)
    spp.to_parquet(spp_dir / "riskset.parquet", index=False)
    pd.DataFrame([{"model_name": "full_information", "n_rows": len(fpp), "n_events": int(fpp["event_bin"].sum()), "bic": 1.0, "converged": True, "stable": True, "safety_warnings": ""}]).to_csv(fpp_dir / "models" / "model_summary.csv", index=False)
    pd.DataFrame([{"model_name": "full_information", "n_rows": len(spp), "n_events": int(spp["event_bin"].sum()), "bic": 1.0, "converged": False, "stable": False, "safety_warnings": "zero events"}]).to_csv(spp_dir / "models" / "model_summary.csv", index=False)

    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"selected_lag_ms": lag}), encoding="utf-8")

    called = {"fit": 0}

    def fail_if_fit(*args, **kwargs):
        called["fit"] += 1
        raise AssertionError("pooled fit should not run when SPP event bins are zero")

    monkeypatch.setattr("cas.hazard_behavior.final_behavior._fit_final_model", fail_if_fit)

    with pytest.raises(ValueError, match="SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins"):
        run_behavior_final_compare(cfg_path, selected_path, fpp_dir / "riskset.parquet", spp_dir / "riskset.parquet", cmp_dir)
    assert called["fit"] == 0
