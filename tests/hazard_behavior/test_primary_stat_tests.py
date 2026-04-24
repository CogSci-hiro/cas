from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.model import FittedBehaviourModel
from cas.hazard_behavior.reporting import (
    build_primary_publication_table,
    compute_primary_lrt_row,
    primary_supported,
)
from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline


def test_lrt_computation() -> None:
    row = compute_primary_lrt_row(
        parent_model="M0_time",
        child_model="M1_rate",
        parent_log_likelihood=-100.0,
        child_log_likelihood=-95.0,
        parent_df=5,
        child_df=6,
        parent_aic=210.0,
        child_aic=202.0,
        hypothesis="secondary",
        interpretation="secondary",
    )

    assert row["lrt_statistic"] == 10.0
    assert row["df_difference"] == 1
    assert row["delta_aic"] == -8.0


def test_delta_aic_direction() -> None:
    row = compute_primary_lrt_row(
        parent_model="M1_rate",
        child_model="M2_rate_prop_expected",
        parent_log_likelihood=-95.0,
        child_log_likelihood=-94.0,
        parent_df=6,
        child_df=7,
        parent_aic=202.0,
        child_aic=203.0,
        hypothesis="primary",
        interpretation="primary",
    )

    assert row["delta_aic"] == row["child_aic"] - row["parent_aic"]


def test_publication_table_extraction() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    fitted_models = {
        "M1_rate": _mock_fitted_model(
            "M1_rate",
            [
                {
                    "term": "z_information_rate_lag_0ms",
                    "estimate": 0.2,
                    "standard_error": 0.1,
                    "z_value": 2.0,
                    "p_value": 0.045,
                    "conf_low": 0.01,
                    "conf_high": 0.39,
                    "odds_ratio": 1.22,
                    "odds_ratio_conf_low": 1.01,
                    "odds_ratio_conf_high": 1.48,
                }
            ],
        ),
        "M2_rate_prop_expected": _mock_fitted_model(
            "M2_rate_prop_expected",
            [
                {
                    "term": "z_information_rate_lag_0ms",
                    "estimate": 0.1,
                    "standard_error": 0.1,
                    "z_value": 1.0,
                    "p_value": 0.30,
                    "conf_low": -0.1,
                    "conf_high": 0.3,
                    "odds_ratio": 1.11,
                    "odds_ratio_conf_low": 0.90,
                    "odds_ratio_conf_high": 1.35,
                },
                {
                    "term": "z_prop_expected_cumulative_info_lag_300ms",
                    "estimate": 0.3,
                    "standard_error": 0.12,
                    "z_value": 2.5,
                    "p_value": 0.012,
                    "conf_low": 0.07,
                    "conf_high": 0.53,
                    "odds_ratio": 1.35,
                    "odds_ratio_conf_low": 1.07,
                    "odds_ratio_conf_high": 1.70,
                },
            ],
        ),
    }

    table = build_primary_publication_table(fitted_models=fitted_models, config=config)

    assert set(table["term"]) == {
        "z_information_rate_lag_0ms",
        "z_prop_expected_cumulative_info_lag_300ms",
    }
    assert "odds_ratio" in table.columns


def test_primary_supported_logic() -> None:
    assert primary_supported(0.2, -5.0) is True
    assert primary_supported(-0.2, -5.0) is False
    assert primary_supported(0.2, 1.0) is False


def test_cli_smoke_writes_primary_stat_outputs(tmp_path: Path) -> None:
    events_path = tmp_path / "events.csv"
    surprisal_path = tmp_path / "toy_desc-lmSurprisal_features.tsv"
    out_dir = tmp_path / "results"

    events_path.write_text(
        "\n".join(
            [
                "dyad_id,run,speaker,fpp_onset,fpp_offset,fpp_label",
                "dyad-001,1,B,1.20,1.50,FPP_RFC_TAG",
                "dyad-001,1,A,3.20,3.40,FPP_RFC_TAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    surprisal_path.write_text(
        "\n".join(
            [
                "dyad_id\trun\tspeaker\tonset\tduration\tword\tsurprisal\talignment_status",
                "dyad-001\t1\tA\t0.00\t0.10\toui\t1.0\tok",
                "dyad-001\t1\tA\t0.30\t0.10\talors\t2.0\tok",
                "dyad-001\t1\tA\t0.70\t0.10\trouge\t3.0\tok",
                "dyad-001\t1\tB\t2.00\t0.10\toui\t1.1\tok",
                "dyad-001\t1\tB\t2.30\t0.10\talors\t1.3\tok",
                "dyad-001\t1\tB\t2.70\t0.10\trouge\t1.7\tok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = BehaviourHazardConfig(
        events_path=events_path,
        surprisal_paths=(surprisal_path,),
        out_dir=out_dir,
        overwrite=True,
    )
    run_behaviour_hazard_pipeline(config)

    assert (out_dir / "models" / "behaviour_primary_stat_tests.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_stat_tests.json").exists()
    assert (out_dir / "models" / "behaviour_primary_publication_table.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_interpretation.txt").exists()

    payload = json.loads((out_dir / "models" / "behaviour_primary_stat_tests.json").read_text(encoding="utf-8"))
    assert "primary_supported" in payload


def _mock_fitted_model(model_name: str, rows: list[dict[str, float | str]]) -> FittedBehaviourModel:
    summary_table = pd.DataFrame([{"model_name": model_name, **row} for row in rows])
    return FittedBehaviourModel(
        model_name=model_name,
        formula="event ~ x",
        result=SimpleNamespace(llf=-10.0, params=pd.Series([0.0])),
        summary_table=summary_table,
        fit_metrics={"aic": 20.0, "cluster_variable": "episode_id", "robust_covariance_used": True},
        robust_covariance_used=True,
        warnings=[],
    )
