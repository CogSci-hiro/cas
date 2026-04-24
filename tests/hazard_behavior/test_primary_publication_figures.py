from __future__ import annotations

from pathlib import Path

import pandas as pd

from cas.hazard_behavior.plots import (
    plot_primary_coefficients,
    plot_primary_model_comparison,
    plot_primary_prediction_curve,
)


def test_coefficient_figure_writes(tmp_path: Path) -> None:
    publication_table = pd.DataFrame(
        {
            "model_name": ["M2_rate_prop_expected", "M2_rate_prop_expected"],
            "term": ["z_information_rate_lag_0ms", "z_prop_expected_cumulative_info_lag_300ms"],
            "estimate": [0.10, 0.25],
            "conf_low": [-0.05, 0.10],
            "conf_high": [0.25, 0.40],
            "odds_ratio": [1.11, 1.28],
        }
    )

    output_path = tmp_path / "behaviour_primary_coefficients.png"
    plot_primary_coefficients(publication_table, output_path)

    assert output_path.exists()


def test_model_comparison_figure_writes(tmp_path: Path) -> None:
    comparison_table = pd.DataFrame(
        {
            "comparison": ["M1_rate vs M0_time", "M2_rate_prop_expected vs M1_rate"],
            "delta_aic": [-3.0, -5.0],
        }
    )

    output_path = tmp_path / "behaviour_primary_model_comparison.png"
    plot_primary_model_comparison(comparison_table, output_path)

    assert output_path.exists()


def test_predicted_hazard_figure_writes(tmp_path: Path) -> None:
    prediction_table = pd.DataFrame(
        {
            "z_prop_expected_cumulative_info_lag_300ms": [-1.0, 0.0, 1.0],
            "predicted_hazard": [0.10, 0.15, 0.22],
            "conf_low": [0.08, 0.12, 0.18],
            "conf_high": [0.12, 0.18, 0.26],
        }
    )

    csv_path = tmp_path / "behaviour_primary_predicted_hazard_prop_expected.csv"
    prediction_table.to_csv(csv_path, index=False)
    output_path = tmp_path / "behaviour_primary_predicted_hazard_prop_expected.png"
    plot_primary_prediction_curve(
        prediction_table,
        x_column="z_prop_expected_cumulative_info_lag_300ms",
        x_label="Expected-relative cumulative information, 300 ms (z)",
        title="Predicted behavioural hazard by delayed expected-relative information",
        output_path=output_path,
    )

    assert output_path.exists()
    assert csv_path.exists()
