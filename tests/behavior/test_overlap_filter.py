from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cas.behavior.config import BehaviorHazardConfig, BehaviorHazardPaths, load_behavior_hazard_config
from cas.behavior.risksets import _legacy_config
from cas.behavior.pipeline import _apply_overlap_filter, add_predictors, select_lag


def _config(tmp_path: Path, *, only_overlap: bool) -> BehaviorHazardConfig:
    output_dir = tmp_path / "derivatives"
    behavior_root = output_dir / "behavior"
    return BehaviorHazardConfig(
        path=tmp_path / "hazard.yaml",
        raw={"inputs": {"events_csv": "dummy.csv"}, "behavior": {"only_overlap": only_overlap, "hazard": {"bin_size_ms": 50}}},
        paths_config_path=tmp_path / "paths.yaml",
        paths_config={"output_dir": str(output_dir)},
        paths=BehaviorHazardPaths(
            output_dir=output_dir,
            behavior_root=behavior_root,
            hazard_root=behavior_root / "hazard",
            figures_main_behavior=output_dir / "figures" / "main" / "behavior",
            figures_supp_behavior=output_dir / "figures" / "supp" / "behavior",
            figures_qc_behavior=output_dir / "figures" / "qc" / "behavior",
        ),
    )


def _pooled_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["fpp-neg", "fpp-neg", "fpp-pos", "spp-neg", "spp-pos"],
            "anchor_type": ["FPP", "FPP", "FPP", "SPP", "SPP"],
            "event_latency_from_partner_offset_s": [-0.2, -0.2, 0.1, -0.05, 0.4],
            "event": [0, 1, 1, 1, 1],
            "information_rate": [1.0, 1.2, 0.8, 0.9, 1.1],
            "prop_expected_cum_info": [0.1, 0.2, 0.3, 0.25, 0.35],
            "time_from_partner_onset_s": [-0.5, -0.45, -0.4, -0.35, -0.3],
            "time_from_partner_offset_s": [-0.2, -0.15, 0.1, -0.05, 0.4],
            "bin_start_s": [0.0, 0.05, 0.0, 0.0, 0.0],
            "bin_end_s": [0.05, 0.1, 0.05, 0.05, 0.05],
            "dyad_id": ["d1", "d1", "d1", "d2", "d2"],
            "subject": ["d1_A", "d1_A", "d1_A", "d2_B", "d2_A"],
            "run": [1, 1, 1, 1, 1],
        }
    )


def test_load_behavior_hazard_config_defaults_only_overlap_false(tmp_path: Path) -> None:
    paths_path = tmp_path / "paths.yaml"
    paths_path.write_text(f"output_dir: {tmp_path / 'derivatives'}\n", encoding="utf-8")
    config_path = tmp_path / "hazard.yaml"
    config_path.write_text(
        f"paths_config: {paths_path}\ninputs:\n  events_csv: dummy.csv\nbehavior:\n  hazard:\n    bin_size_ms: 50\n",
        encoding="utf-8",
    )

    config = load_behavior_hazard_config(config_path)

    assert config.only_overlap is False
    assert config.information_rate_window_ms == 500


def test_load_behavior_hazard_config_reads_information_rate_window_ms(tmp_path: Path) -> None:
    paths_path = tmp_path / "paths.yaml"
    paths_path.write_text(f"output_dir: {tmp_path / 'derivatives'}\n", encoding="utf-8")
    config_path = tmp_path / "hazard.yaml"
    config_path.write_text(
        (
            f"paths_config: {paths_path}\n"
            "inputs:\n"
            "  events_csv: dummy.csv\n"
            "behavior:\n"
            "  hazard:\n"
            "    bin_size_ms: 50\n"
            "    information_rate_window_ms: 750\n"
        ),
        encoding="utf-8",
    )

    config = load_behavior_hazard_config(config_path)

    assert config.information_rate_window_ms == 750


def test_legacy_config_uses_information_rate_window_from_config(tmp_path: Path) -> None:
    config = BehaviorHazardConfig(
        path=tmp_path / "hazard.yaml",
        raw={
            "inputs": {"events_csv": "dummy.csv"},
            "behavior": {"hazard": {"bin_size_ms": 50, "information_rate_window_ms": 750}},
        },
        paths_config_path=tmp_path / "paths.yaml",
        paths_config={"output_dir": str(tmp_path / "derivatives")},
        paths=BehaviorHazardPaths(
            output_dir=tmp_path / "derivatives",
            behavior_root=tmp_path / "derivatives" / "behavior",
            hazard_root=tmp_path / "derivatives" / "behavior" / "hazard",
            figures_main_behavior=tmp_path / "derivatives" / "figures" / "main" / "behavior",
            figures_supp_behavior=tmp_path / "derivatives" / "figures" / "supp" / "behavior",
            figures_qc_behavior=tmp_path / "derivatives" / "figures" / "qc" / "behavior",
        ),
    )

    legacy = _legacy_config(config)

    assert legacy.information_rate_window_s == pytest.approx(0.75)


def test_apply_overlap_filter_keeps_full_episode_rows_for_negative_latency() -> None:
    filtered, counts, by_anchor = _apply_overlap_filter(_pooled_table(), config=_config(Path("/tmp"), only_overlap=True))

    assert set(filtered["episode_id"]) == {"fpp-neg", "spp-neg"}
    assert list(filtered.loc[filtered["episode_id"] == "fpp-neg", "event"].tolist()) == [0, 1]
    assert counts["overlap_filter_column"] == "event_latency_from_partner_offset_s"
    assert counts["overlap_filter_definition"] == "event_latency_from_partner_offset_s < 0"
    assert counts["n_rows_after_overlap_filter"] <= counts["n_rows_before_overlap_filter"]
    assert counts["n_events_after_overlap_filter"] <= counts["n_events_before_overlap_filter"]
    assert set(by_anchor["anchor_type"]) == {"FPP", "SPP"}


def test_apply_overlap_filter_is_noop_when_disabled() -> None:
    filtered, counts, by_anchor = _apply_overlap_filter(_pooled_table(), config=_config(Path("/tmp"), only_overlap=False))

    assert len(filtered) == len(_pooled_table())
    assert counts["n_rows_before_overlap_filter"] == counts["n_rows_after_overlap_filter"]
    assert counts["n_events_before_overlap_filter"] == counts["n_events_after_overlap_filter"]
    assert bool(by_anchor["only_overlap"].iloc[0]) is False


def test_apply_overlap_filter_fails_when_latency_column_missing() -> None:
    table = _pooled_table().drop(columns=["event_latency_from_partner_offset_s"])
    with pytest.raises(ValueError, match="no overlap-latency column was found"):
        _apply_overlap_filter(table, config=_config(Path("/tmp"), only_overlap=True))


def test_apply_overlap_filter_fails_when_subset_empty() -> None:
    table = _pooled_table().assign(event_latency_from_partner_offset_s=0.2)
    with pytest.raises(ValueError, match="empty overlap-only subset|zero remaining events"):
        _apply_overlap_filter(table, config=_config(Path("/tmp"), only_overlap=True))


def test_add_predictors_writes_overlap_filter_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path, only_overlap=True)
    riskset_dir = config.paths.hazard_root / "risksets"
    riskset_dir.mkdir(parents=True, exist_ok=True)
    pooled = _pooled_table()
    pooled.to_parquet(riskset_dir / "pooled_fpp_spp.parquet", index=False)
    pooled.loc[pooled["anchor_type"] == "FPP"].to_parquet(riskset_dir / "fpp.parquet", index=False)
    pooled.loc[pooled["anchor_type"] == "SPP"].to_parquet(riskset_dir / "spp_control.parquet", index=False)

    monkeypatch.setattr("cas.behavior.pipeline.load_behavior_hazard_config", lambda _: config)

    outputs = add_predictors(config.path)

    assert outputs["fpp"].exists()
    summary = json.loads((config.paths.hazard_root / "diagnostics" / "overlap_filter_summary.json").read_text(encoding="utf-8"))
    assert summary["only_overlap"] is True
    assert summary["n_rows_after_overlap_filter"] < summary["n_rows_before_overlap_filter"]
    by_anchor = pd.read_csv(config.paths.hazard_root / "tables" / "overlap_filter_by_anchor.csv")
    assert {"anchor_type", "n_rows_before", "n_rows_after"} <= set(by_anchor.columns)


def test_select_lag_uses_filtered_predictor_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path, only_overlap=True)
    predictor_dir = config.paths.hazard_root / "predictors"
    predictor_dir.mkdir(parents=True, exist_ok=True)
    fpp = pd.DataFrame(
        {
            "episode_id": ["e1", "e1"],
            "event": [0, 1],
            "lag_feature": [1.0, 2.0],
            "overlap_filter_column": ["event_latency_from_partner_offset_s"] * 2,
        }
    )
    fpp.to_parquet(predictor_dir / "fpp_with_lags.parquet", index=False)
    pd.DataFrame({"episode_id": [], "event": []}).to_parquet(predictor_dir / "spp_control_with_lags.parquet", index=False)
    pd.DataFrame({"episode_id": [], "event": []}).to_parquet(predictor_dir / "pooled_with_lags.parquet", index=False)

    def _fake_run_r_lag_selection(**kwargs) -> None:
        assert len(kwargs["fpp_table"]) == 2
        assert set(kwargs["fpp_table"]["episode_id"]) == {"e1"}
        pd.DataFrame(
            {
                "lag_ms": [0, 50],
                "delta_BIC": [0.0, 1.0],
                "logLik": [-10.0, -11.0],
                "selected": [True, False],
                "lag_selection_criterion": ["bic", "bic"],
            }
        ).to_csv(kwargs["score_path"], index=False)
        Path(kwargs["selected_path"]).write_text(
            json.dumps({"selected_lag_ms": 0, "selector_model_id": "M_3", "anchor_subset": "fpp"}),
            encoding="utf-8",
        )
        pd.DataFrame().to_csv(kwargs["lag_sensitivity_path"], index=False)

    monkeypatch.setattr("cas.behavior.pipeline.load_behavior_hazard_config", lambda _: config)
    monkeypatch.setattr("cas.behavior.pipeline.run_r_lag_selection", _fake_run_r_lag_selection)

    select_lag(config.path)

    selected = json.loads((config.paths.hazard_root / "lag_selection" / "selected_lag.json").read_text(encoding="utf-8"))
    assert selected["only_overlap"] is True
    assert selected["n_rows"] == 2
    assert selected["n_events"] == 1
