from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.model import (
    build_primary_effects_payload,
    build_primary_model_formulas,
    build_timing_control_model_formulas,
    compare_primary_models,
    compare_timing_control_models,
    fit_primary_behaviour_models,
    select_best_timing_control_lag,
    select_timing_control_best_lags,
    fit_timing_control_behaviour_models,
    primary_unscaled_column_name,
    validate_timing_control_model_table,
)
from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline


def test_primary_formula_generation_uses_expected_predictors() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        primary_information_rate_lag_ms=0,
        primary_prop_expected_lag_ms=300,
    )

    formulas = build_primary_model_formulas(config)

    assert "z_information_rate_lag_0ms" in formulas["M1_rate"]
    assert "z_prop_expected_cumulative_info_lag_300ms" in formulas["M2_rate_prop_expected"]
    assert "z_prop_actual_cumulative_info" not in formulas["M2_rate_prop_expected"]
    assert "z_cumulative_info" not in formulas["M2_rate_prop_expected"]


def test_primary_models_fit_on_synthetic_data() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )

    fitted_models = fit_primary_behaviour_models(_synthetic_primary_riskset(), config=config)

    assert {"M0_time", "M1_rate", "M2_rate_prop_expected"} <= set(fitted_models)


def test_primary_model_comparison_uses_child_minus_parent_delta_aic() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )

    fitted_models = fit_primary_behaviour_models(_synthetic_primary_riskset(), config=config)
    comparison = compare_primary_models(fitted_models)
    row = comparison.loc[comparison["comparison"] == "M1_rate vs M0_time"].iloc[0]

    assert row["delta_aic"] == pytest.approx(row["child_aic"] - row["parent_aic"])


def test_primary_effects_payload_contains_main_decision_fields() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    riskset = _synthetic_primary_riskset()
    fitted_models = fit_primary_behaviour_models(riskset, config=config)
    comparison = compare_primary_models(fitted_models)

    payload = build_primary_effects_payload(
        riskset_table=riskset,
        fitted_models=fitted_models,
        comparison_table=comparison,
        config=config,
    )

    assert "delta_aic_m1_vs_m0" in payload
    assert "delta_aic_m2_vs_m1" in payload
    assert "beta_prop_expected" in payload
    assert "main_prediction_supported" in payload


def test_timing_control_formula_generation_uses_both_smooths_and_primary_predictors() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        primary_information_rate_lag_ms=0,
        primary_prop_expected_lag_ms=300,
    )

    formulas = build_timing_control_model_formulas(config)

    assert "bs(time_from_partner_onset" in formulas["M0_timing"]
    assert "bs(time_from_partner_offset" in formulas["M0_timing"]
    assert "bs(time_from_partner_onset" in formulas["M1_rate_best_timing"]
    assert "bs(time_from_partner_offset" in formulas["M1_rate_best_timing"]
    assert "bs(time_from_partner_onset" in formulas["M2_expected_best_timing"]
    assert "bs(time_from_partner_offset" in formulas["M2_expected_best_timing"]
    assert "z_information_rate_lag_0ms" in formulas["M1_rate_best_timing"]
    assert "z_information_rate_lag_0ms" in formulas["M2_expected_best_timing"]
    assert "z_prop_expected_cumulative_info_lag_300ms" in formulas["M2_expected_best_timing"]


def test_timing_control_validation_retains_negative_offset_times() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )

    validated = validate_timing_control_model_table(_synthetic_timing_control_riskset(), config=config)

    assert (validated["time_from_partner_offset"] < 0.0).any()


def test_timing_control_model_comparison_uses_child_minus_parent_delta_aic() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )

    fitted_models = fit_timing_control_behaviour_models(_synthetic_timing_control_riskset(), config=config)
    comparison = compare_timing_control_models(fitted_models)
    row = comparison.loc[comparison["comparison"] == "M1_rate_best_timing vs M0_timing"].iloc[0]

    assert row["delta_aic"] == pytest.approx(row["child_aic"] - row["parent_aic"])


def test_information_rate_lag_selection_uses_m0_timing_parent() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        lag_grid_ms=(0, 100, 300),
        select_lags_with_timing_controls=True,
    )

    selection_table, selected = select_timing_control_best_lags(_synthetic_timing_control_riskset(), config=config)

    info_rows = selection_table.loc[selection_table["predictor_family"] == "information_rate"]
    assert not info_rows.empty
    assert set(info_rows["parent_model"]) == {"M0_timing"}
    assert selected["lag_selection_parent_for_information_rate"] == "M0_timing"


def test_expected_info_lag_selection_uses_m1_rate_best_timing_parent() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
        lag_grid_ms=(0, 100, 300),
        select_lags_with_timing_controls=True,
    )

    selection_table, selected = select_timing_control_best_lags(_synthetic_timing_control_riskset(), config=config)

    expected_rows = selection_table.loc[
        selection_table["predictor_family"] == "prop_expected_cumulative_info"
    ]
    assert not expected_rows.empty
    assert set(expected_rows["parent_model"]) == {"M1_rate_best_timing"}
    assert selected["lag_selection_parent_for_expected_cumulative_info"] == "M1_rate_best_timing"


def test_timing_control_lag_selection_uses_most_negative_delta_aic() -> None:
    rows = pd.DataFrame(
        {
            "predictor_family": ["information_rate", "information_rate", "information_rate"],
            "lag_ms": [0, 100, 300],
            "delta_aic": [-10.0, -20.0, -15.0],
        }
    )

    best_row = select_best_timing_control_lag(rows, predictor_family="information_rate")

    assert int(best_row["lag_ms"]) == 100


def test_timing_control_model_comparison_uses_required_final_pairs() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )

    comparison = compare_timing_control_models(
        fit_timing_control_behaviour_models(_synthetic_timing_control_riskset(), config=config)
    )

    assert set(comparison["comparison"]) == {
        "M1_rate_best_timing vs M0_timing",
        "M2_expected_best_timing vs M1_rate_best_timing",
    }


def test_primary_predictor_validation_names_missing_column() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    riskset = _synthetic_primary_riskset().drop(columns=["prop_expected_cumulative_info_lag_300ms"])

    with pytest.raises(ValueError, match="prop_expected_cumulative_info_lag_300ms"):
        fit_primary_behaviour_models(riskset, config=config)


def test_primary_constant_predictor_fails_clearly() -> None:
    config = BehaviourHazardConfig(
        events_path=Path("unused.csv"),
        surprisal_paths=(Path("unused.tsv"),),
        out_dir=Path("out"),
    )
    riskset = _synthetic_primary_riskset()
    riskset["prop_expected_cumulative_info_lag_300ms"] = 1.0

    with pytest.raises(ValueError, match="prop_expected_cumulative_info_lag_300ms"):
        fit_primary_behaviour_models(riskset, config=config)


def test_cli_pipeline_writes_primary_outputs(tmp_path: Path) -> None:
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

    assert (out_dir / "models" / "behaviour_primary_model_summary.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_model_comparison.csv").exists()
    assert (out_dir / "models" / "behaviour_primary_effects.json").exists()
    assert (out_dir / "models" / "behaviour_primary_fit_metrics.json").exists()

    payload = json.loads((out_dir / "models" / "behaviour_primary_effects.json").read_text(encoding="utf-8"))
    assert "main_prediction_supported" in payload


def test_cli_pipeline_writes_timing_control_outputs_when_enabled(tmp_path: Path) -> None:
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
        fit_timing_control_models=True,
    )
    run_behaviour_hazard_pipeline(config)

    assert (out_dir / "models" / "behaviour_timing_control_model_summary.csv").exists()
    assert (out_dir / "models" / "behaviour_timing_control_model_comparison.csv").exists()
    assert (out_dir / "models" / "behaviour_timing_control_fit_metrics.json").exists()
    assert (out_dir / "models" / "behaviour_timing_control_selected_lags.json").exists()
    assert (out_dir / "riskset" / "hazard_behavior_riskset_with_timing_controls.tsv").exists()


def _synthetic_primary_riskset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_number in range(24):
        episode_id = f"ep-{episode_number:02d}"
        event_bin = 4 if episode_number % 3 != 0 else None
        for bin_index in range(6):
            time_value = round(bin_index * 0.1, 3)
            information_rate = float((episode_number % 5) - 2 + time_value * 2.0)
            prop_expected = float(-1.5 + (episode_number * 0.15) + (bin_index * 0.25))
            rows.append(
                {
                    "episode_id": episode_id,
                    "dyad_id": f"dyad-{episode_number % 4}",
                    "subject_id": f"subject-{episode_number % 6}",
                    "anchor_source": "partner_ipu",
                    "time_from_partner_onset": time_value,
                    "event": int(event_bin is not None and bin_index == event_bin),
                    "episode_has_event": int(event_bin is not None),
                    primary_unscaled_column_name("information_rate", 0): information_rate,
                    primary_unscaled_column_name("information_rate", 100): information_rate - 0.2,
                    primary_unscaled_column_name("information_rate", 300): information_rate - 0.5,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 300): prop_expected,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 0): prop_expected - 0.4,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 100): prop_expected - 0.2,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_timing_control_riskset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_number in range(24):
        episode_id = f"ep-{episode_number:02d}"
        event_bin = 4 if episode_number % 3 != 0 else None
        partner_ipu_duration = 0.18 + ((episode_number % 5) * 0.04)
        for bin_index in range(6):
            time_value = round(bin_index * 0.1, 3)
            information_rate = float((episode_number % 5) - 2 + time_value * 2.0)
            prop_expected = float(-1.5 + (episode_number * 0.15) + (bin_index * 0.25))
            rows.append(
                {
                    "episode_id": episode_id,
                    "dyad_id": f"dyad-{episode_number % 4}",
                    "subject_id": f"subject-{episode_number % 6}",
                    "anchor_source": "partner_ipu",
                    "time_from_partner_onset": time_value,
                    "time_from_partner_offset": (time_value + 0.1) - partner_ipu_duration,
                    "event": int(event_bin is not None and bin_index == event_bin),
                    "episode_has_event": int(event_bin is not None),
                    primary_unscaled_column_name("information_rate", 0): information_rate,
                    primary_unscaled_column_name("information_rate", 100): information_rate - 0.2,
                    primary_unscaled_column_name("information_rate", 300): information_rate - 0.5,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 300): prop_expected,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 0): prop_expected - 0.4,
                    primary_unscaled_column_name("prop_expected_cumulative_info", 100): prop_expected - 0.2,
                }
            )
    return pd.DataFrame(rows)
