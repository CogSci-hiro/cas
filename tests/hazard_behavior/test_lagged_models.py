from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.model import build_model_formulas, fit_binomial_glm
from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline


def test_lagged_model_formulas_include_expected_terms() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        lag_grid_ms=(300,),
    )

    formulas = build_model_formulas(config)

    assert "z_information_rate_lag_300ms" in formulas["M1_rate_lag_300"]
    assert "z_prop_actual_cumulative_info_lag_300ms" in formulas["M2a_prop_actual_lag_300"]


def test_lagged_model_fitting_does_not_crash() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        lag_grid_ms=(300,),
    )
    riskset = pd.DataFrame(
        {
            "episode_id": ["ep1", "ep1", "ep1", "ep2", "ep2", "ep2", "ep3", "ep3", "ep3"],
            "dyad_id": ["d1"] * 9,
            "time_from_partner_onset": [0.0, 0.05, 0.10] * 3,
            "event": [0, 0, 1, 0, 1, 0, 0, 0, 1],
            "z_information_rate": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
            "z_prop_actual_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
            "z_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
            "z_prop_expected_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
            "z_information_rate_lag_300ms": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
            "z_prop_actual_cumulative_info_lag_300ms": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0, -0.2, 0.1, 0.9],
        }
    )

    model = fit_binomial_glm(riskset, model_name="M1_rate_lag_300", config=config)

    assert model.model_name == "M1_rate_lag_300"
    assert "estimate" in model.summary_table.columns
    assert "standard_error" in model.summary_table.columns
    assert model.fit_metrics["aic"] is not None


def test_pipeline_writes_lagged_outputs(tmp_path: Path) -> None:
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
        lag_grid_ms=(0, 100, 300),
        save_lagged_feature_table=False,
    )

    run_behaviour_hazard_pipeline(config)

    assert (out_dir / "riskset" / "lagged_feature_qc.json").exists()
    assert (out_dir / "models" / "model_comparison_lagged_behaviour.csv").exists()
    assert (out_dir / "models" / "coefficient_by_lag.csv").exists()
    assert (out_dir / "models" / "model_fit_by_lag.csv").exists()
    assert (out_dir / "models" / "best_lag_by_aic.csv").exists()
    assert (out_dir / "figures" / "information_rate_coefficient_by_lag.png").exists()
    assert (out_dir / "figures" / "model_delta_aic_by_lag.png").exists()
