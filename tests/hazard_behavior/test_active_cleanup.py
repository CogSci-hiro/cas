from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas.hazard_behavior.plot_r_results import (
    filter_active_fixed_effects,
    filter_active_model_comparisons,
    filter_active_r_glmm_lag_sweep_rows,
    plot_behaviour_hazard_results,
    select_best_r_glmm_lag,
)


PROJECT_ROOT = Path("/Users/hiro/Projects/active/cas")


def test_legacy_readme_exists() -> None:
    assert (PROJECT_ROOT / "legacy" / "behavior" / "README.md").exists()


def test_cleanup_inventory_exists() -> None:
    assert (PROJECT_ROOT / "legacy" / "behavior" / "CLEANUP_INVENTORY.md").exists()


def test_active_plotting_filters_obsolete_rows(tmp_path: Path) -> None:
    r_results_dir = tmp_path / "models"
    models_dir = r_results_dir / "lag_selection"
    predictions_dir = r_results_dir / "predictions"
    figures_dir = tmp_path / "figures"
    qc_dir = tmp_path / "qc_plots" / "lag_selection"
    r_results_dir.mkdir()
    models_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "predictor_family": [
                "information_rate",
                "prop_expected",
                "prop_actual",
                "cumulative_info",
            ],
            "lag_ms": [0, 300, 0, 300],
            "child_BIC": [100.0, 98.0, 120.0, 122.0],
            "delta_BIC": [-2.5, -3.5, -0.2, -0.1],
            "beta": [0.2, 0.3, 0.1, 0.1],
            "conf_low": [0.1, 0.2, -0.1, -0.1],
            "conf_high": [0.3, 0.4, 0.3, 0.3],
            "odds_ratio": [1.22, 1.35, 1.10, 1.10],
            "odds_ratio_conf_low": [1.10, 1.22, 0.90, 0.90],
            "odds_ratio_conf_high": [1.35, 1.49, 1.35, 1.35],
            "converged": [True, True, True, True],
        }
    ).to_csv(r_results_dir / "r_glmm_information_rate_lag_sweep.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["prop_expected", "M2a"],
            "lag_ms": [300, 300],
            "child_BIC": [97.0, 120.0],
            "delta_BIC": [-1.0, 0.0],
            "beta": [0.2, 0.0],
            "conf_low": [0.1, -0.1],
            "conf_high": [0.3, 0.1],
            "odds_ratio": [1.22, 1.0],
            "odds_ratio_conf_low": [1.10, 0.9],
            "odds_ratio_conf_high": [1.35, 1.1],
            "converged": [True, False],
        }
    ).to_csv(r_results_dir / "r_glmm_prop_expected_lag_sweep.csv", index=False)
    pd.DataFrame(
        {
            "child_model": [
                "M_final_glmm",
                "M_final_plus_expected_glmm",
                "M2a",
            ],
            "delta_BIC": [-2.0, -4.0, -1.0],
        }
    ).to_csv(r_results_dir / "r_glmm_final_behaviour_model_comparison.csv", index=False)
    (r_results_dir / "r_glmm_selected_behaviour_lags.json").write_text(
        '{"best_information_rate_lag_ms": 0, "best_prop_expected_lag_ms": 300}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "predictor": ["z_information_rate_lag_0ms"] * 3,
            "predictor_value": [-1, 0, 1],
            "predicted_probability": [0.1, 0.2, 0.3],
            "conf_low": [0.08, 0.17, 0.25],
            "conf_high": [0.12, 0.23, 0.35],
            "fixed_effect_only": [True, True, True],
        }
    ).to_csv(predictions_dir / "behaviour_r_glmm_final_predicted_hazard_information_rate.csv", index=False)
    pd.DataFrame(
        {
            "predictor_family": ["information_rate", "prop_expected_cumulative_info"],
            "lag_ms": [0, 300],
            "delta_bic": [-2.0, -3.0],
        }
    ).to_csv(models_dir / "behaviour_timing_control_lag_selection.csv", index=False)

    plot_behaviour_hazard_results(
        r_results_dir=r_results_dir,
        timing_control_models_dir=models_dir,
        output_dir=figures_dir,
        qc_output_dir=qc_dir,
    )

    expected = {
        "behaviour_r_glmm_delta_bic_by_lag.png",
        "behaviour_r_glmm_coefficient_by_lag.png",
        "behaviour_r_glmm_odds_ratio_by_lag.png",
        "behaviour_r_glmm_final_model_comparison.png",
        "behaviour_r_glmm_final_predicted_hazard_information_rate.png",
        "data",
    }
    assert {path.name for path in figures_dir.iterdir()} == expected
    assert {path.name for path in qc_dir.iterdir()} == {"behaviour_pooled_delta_bic_by_lag.png"}


def test_active_lag_plot_filters_obsolete_rows() -> None:
    selection = pd.DataFrame(
        {
            "predictor_family": [
                "information_rate",
                "prop_expected",
                "prop_actual",
                "cumulative_info",
            ],
            "lag_ms": [0, 300, 0, 300],
            "child_BIC": [10.0, 8.0, 20.0, 25.0],
            "converged": [True, True, True, True],
        }
    )

    filtered = filter_active_r_glmm_lag_sweep_rows(selection)

    assert filtered["predictor_family"].tolist() == [
        "information_rate",
        "prop_expected",
    ]


def test_active_model_comparison_filters_obsolete_rows() -> None:
    comparison = pd.DataFrame(
        {
            "child_model": [
                "M_final_glmm",
                "M_final_plus_expected_glmm",
                "M2a",
                "M2b",
                "M2c",
            ],
            "delta_BIC": [-2.0, -4.0, -1.0, 0.5, 1.0],
        }
    )

    filtered = filter_active_model_comparisons(comparison)

    assert filtered["child_model"].tolist() == [
        "M_final_glmm",
        "M_final_plus_expected_glmm",
    ]


def test_active_coefficient_plot_filters_obsolete_terms() -> None:
    fixed_effects = pd.DataFrame(
        {
            "term": [
                "Intercept",
                "bs(time_from_partner_onset, df = 6)[1]",
                "z_prop_actual_cumulative_info_lag_best",
                "z_cumulative_info_lag_best",
                "z_information_rate_lag_best",
                "z_prop_expected_cumulative_info_lag_best",
            ]
        }
    )

    filtered = filter_active_fixed_effects(fixed_effects)

    assert filtered["term"].tolist() == [
        "z_information_rate_lag_best",
        "z_prop_expected_cumulative_info_lag_best",
    ]


def test_best_r_glmm_lag_prefers_lowest_child_bic_among_converged_rows() -> None:
    lag_sweep = pd.DataFrame(
        {
            "predictor_family": ["information_rate", "information_rate", "information_rate"],
            "lag_ms": [50, 100, 150],
            "child_BIC": [110.0, 95.0, 80.0],
            "converged": [True, False, True],
        }
    )

    best_row = select_best_r_glmm_lag(lag_sweep, predictor_family="information_rate")

    assert int(best_row["lag_ms"]) == 150


def test_active_source_has_no_obsolete_model_names() -> None:
    disallowed = []
    for path in (PROJECT_ROOT / "src" / "cas" / "hazard_behavior").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in ("M2a", "M2b", "M2c")):
            disallowed.append(path)
    assert not disallowed


def test_active_source_has_no_obsolete_plotting_or_modelling_logic() -> None:
    disallowed = []
    for path in [
        PROJECT_ROOT / "src" / "cas" / "hazard_behavior" / "model.py",
        PROJECT_ROOT / "src" / "cas" / "hazard_behavior" / "pipeline.py",
        PROJECT_ROOT / "src" / "cas" / "hazard_behavior" / "plot_r_results.py",
        PROJECT_ROOT / "src" / "cas" / "hazard_behavior" / "plots.py",
        PROJECT_ROOT / "src" / "cas" / "cli" / "commands" / "hazard_behavior_fpp.py",
        PROJECT_ROOT / "workflow" / "rules" / "hazard_behavior.smk",
    ]:
        text = path.read_text(encoding="utf-8")
        if "prop_actual" in text:
            disallowed.append(path)
            continue
        if "cumulative_info" in text and "prop_expected_cumulative_info" not in text:
            disallowed.append(path)
    assert not disallowed


def test_current_model_names_are_retained_in_active_outputs() -> None:
    model_text = (PROJECT_ROOT / "src" / "cas" / "hazard_behavior" / "model.py").read_text(encoding="utf-8")
    assert "M0_timing" in model_text
    assert "M1_rate" in model_text
    assert "M2_expected" in model_text
