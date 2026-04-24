from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.plots import (
    compute_prop_actual_saturation_qc,
    plot_information_feature_distributions,
    plot_prop_actual_by_time_from_partner_onset,
)


def test_information_feature_distribution_excludes_saturated_values_without_mutating_input(
    tmp_path: Path,
) -> None:
    riskset = _build_toy_riskset()
    original = riskset.copy(deep=True)

    metadata = plot_information_feature_distributions(
        riskset,
        tmp_path / "information_feature_distributions.png",
    )

    assert (tmp_path / "information_feature_distributions.png").exists()
    assert metadata["n_non_saturated_prop_actual_rows"] == 5
    pd.testing.assert_frame_equal(riskset, original)


def test_prop_actual_saturation_qc_has_expected_counts() -> None:
    riskset = _build_toy_riskset()

    qc = compute_prop_actual_saturation_qc(riskset)

    assert qc["n_rows_total"] == 10
    assert qc["n_rows_with_finite_prop_actual"] == 9
    assert qc["n_saturated_prop_actual_rows"] == 4
    assert qc["proportion_saturated_prop_actual_rows"] == 4 / 9
    assert qc["n_event_rows"] == 2
    assert qc["n_event_rows_with_finite_prop_actual"] == 2
    assert qc["n_saturated_event_rows"] == 1
    assert qc["proportion_saturated_event_rows"] == 0.5
    assert qc["phase_qc_available"] is True


def test_all_saturated_prop_actual_values_do_not_crash_distribution_plot(tmp_path: Path) -> None:
    riskset = _build_toy_riskset().assign(prop_actual_cumulative_info=1.0)

    metadata = plot_information_feature_distributions(
        riskset,
        tmp_path / "all_saturated.png",
    )

    assert (tmp_path / "all_saturated.png").exists()
    assert metadata["n_non_saturated_prop_actual_rows"] == 0


def test_phase_qc_unavailable_when_partner_offset_columns_absent() -> None:
    riskset = _build_toy_riskset().drop(columns=["partner_ipu_offset", "own_fpp_onset"])

    qc = compute_prop_actual_saturation_qc(riskset)

    assert qc["phase_qc_available"] is False


def test_prop_actual_by_time_plot_does_not_crash(tmp_path: Path) -> None:
    riskset = _build_toy_riskset()

    plot_prop_actual_by_time_from_partner_onset(
        riskset,
        tmp_path / "prop_actual_by_time_from_partner_onset.png",
    )

    assert (tmp_path / "prop_actual_by_time_from_partner_onset.png").exists()


def _build_toy_riskset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": [f"ep-{index}" for index in range(10)],
            "event": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "prop_actual_cumulative_info": [0.0, 0.2, 0.5, 0.99, 1.0, 1.0, 1.0, np.nan, 0.8, 1.0],
            "prop_expected_cumulative_info": [0.1, 0.2, 0.4, 0.6, 0.9, 1.0, 1.1, np.nan, 1.0, 0.7],
            "information_rate": np.linspace(0.0, 9.0, 10),
            "cumulative_info": np.linspace(0.0, 9.0, 10),
            "time_from_partner_onset": np.linspace(0.0, 0.9, 10),
            "bin_index": np.arange(10),
            "alignment_ok_fraction": np.ones(10),
            "partner_ipu_offset": np.repeat(0.4, 10),
            "own_fpp_onset": [np.nan, np.nan, np.nan, np.nan, 0.5, np.nan, np.nan, np.nan, 0.9, np.nan],
            "bin_end": np.linspace(0.05, 0.95, 10),
        }
    )
