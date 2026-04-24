from __future__ import annotations

import numpy as np
import pandas as pd

from cas.hazard.plots import (
    compute_entropy_distribution_terminal_vs_nonterminal_summary,
    compute_observed_event_rate_by_entropy_quantile,
    compute_observed_hazard_by_time_and_entropy_group,
    compute_observed_hazard_by_time_bin,
    compute_smoothed_observed_hazard_by_time_bin,
)


def test_compute_observed_hazard_by_time_bin() -> None:
    hazard_table = pd.DataFrame(
        {
            "tau_seconds": [0.05, 0.05, 0.10, 0.10, 0.10],
            "event": [0, 1, 0, 0, 1],
        }
    )

    summary = compute_observed_hazard_by_time_bin(hazard_table)

    assert summary["n_at_risk"].tolist() == [2, 3]
    assert summary["n_events"].tolist() == [1, 1]
    assert np.allclose(summary["observed_hazard"], [0.5, 1.0 / 3.0])


def test_compute_smoothed_observed_hazard_by_time_bin() -> None:
    observed_hazard_table = pd.DataFrame(
        {
            "tau_seconds": [0.05, 0.10, 0.15],
            "observed_hazard": [0.0, 0.6, 0.0],
        }
    )

    smoothed = compute_smoothed_observed_hazard_by_time_bin(
        observed_hazard_table=observed_hazard_table,
        smoothing_window_bins=3,
    )

    assert np.isclose(smoothed.loc[1, "smoothed_observed_hazard"], 0.2)


def test_compute_observed_event_rate_by_entropy_quantile() -> None:
    hazard_table = pd.DataFrame(
        {
            "entropy_z": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "event": [0, 0, 1, 1, 1],
        }
    )

    summary = compute_observed_event_rate_by_entropy_quantile(
        hazard_table=hazard_table,
        entropy_quantile_count=5,
    )

    assert summary["n_rows"].sum() == 5
    assert summary["n_events"].sum() == 3
    assert summary["entropy_quantile_label"].tolist() == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_compute_observed_hazard_by_time_and_entropy_group() -> None:
    hazard_table = pd.DataFrame(
        {
            "tau_seconds": [0.05, 0.05, 0.10, 0.10, 0.15, 0.15],
            "entropy_z": [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
            "event": [0, 1, 0, 0, 1, 0],
        }
    )

    summary = compute_observed_hazard_by_time_and_entropy_group(
        hazard_table=hazard_table,
        entropy_group_count=3,
    )

    assert {"low entropy", "medium entropy", "high entropy"} <= set(summary["entropy_group"])
    assert (summary["n_at_risk"] >= summary["n_events"]).all()


def test_compute_entropy_distribution_terminal_vs_nonterminal_summary() -> None:
    hazard_table = pd.DataFrame(
        {
            "event": [0, 0, 1, 1],
            "entropy_z": [-1.0, 0.0, 1.0, 2.0],
        }
    )

    summary = compute_entropy_distribution_terminal_vs_nonterminal_summary(hazard_table)

    assert summary["group"].tolist() == ["event", "non_event"]
    assert summary.loc[summary["group"] == "event", "n_rows"].item() == 2
