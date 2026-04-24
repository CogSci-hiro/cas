from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.model import compare_nested_models, fit_binomial_glm


def test_toy_riskset_fits_m0_and_m1() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    riskset = pd.DataFrame(
        {
            "episode_id": ["ep1", "ep1", "ep1", "ep2", "ep2", "ep2"],
            "dyad_id": ["d1"] * 6,
            "time_from_partner_onset": [0.0, 0.05, 0.10, 0.0, 0.05, 0.10],
            "event": [0, 0, 1, 0, 1, 0],
            "z_information_rate": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0],
            "z_prop_actual_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0],
            "z_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0],
            "z_prop_expected_cumulative_info": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0],
        }
    )

    m0 = fit_binomial_glm(riskset, model_name="M0", config=config)
    m1 = fit_binomial_glm(riskset, model_name="M1", config=config)

    assert {"estimate", "standard_error", "conf_low", "p_value", "odds_ratio"} <= set(m1.summary_table.columns)
    comparison = compare_nested_models({"M0": m0, "M1": m1, "M2a": m1, "M2b": m1, "M2c": m1})
    assert "M1 vs M0" in set(comparison["comparison"])


def test_fit_metrics_handle_single_class_outcomes() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    riskset = pd.DataFrame(
        {
            "episode_id": ["ep1", "ep1", "ep2", "ep2"],
            "dyad_id": ["d1"] * 4,
            "time_from_partner_onset": [0.0, 0.05, 0.0, 0.05],
            "event": [0, 0, 0, 0],
            "z_information_rate": [0.0, 0.0, 0.0, 0.0],
            "z_prop_actual_cumulative_info": [0.0, 0.0, 0.0, 0.0],
            "z_cumulative_info": [0.0, 0.0, 0.0, 0.0],
            "z_prop_expected_cumulative_info": [0.0, 0.0, 0.0, 0.0],
        }
    )

    model = fit_binomial_glm(riskset, model_name="M0", config=config)

    assert model.fit_metrics["log_loss_in_sample"] is not None
    assert model.fit_metrics["brier_score_in_sample"] is not None
    assert model.fit_metrics["auroc_in_sample"] is None
