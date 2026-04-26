from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.io import write_json, write_table
from cas.hazard_behavior.plots import plot_observed_event_rate_by_time_bin, summarize_observed_event_rate_by_time_bin


def test_observed_event_rate_handles_string_event_column(tmp_path: Path) -> None:
    riskset = _base_riskset(event_values=["0", "0", "1", "0"])

    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)
    _write_observed_event_rate_outputs(tmp_path, summary, qc)
    plot_observed_event_rate_by_time_bin(summary, qc, tmp_path / "observed_event_rate_by_time_bin.png")

    assert qc["n_events_input"] == 1
    assert qc["sum_binned_events"] == 1
    assert len(_read_nonzero_bins(tmp_path)) == 1
    assert (tmp_path / "observed_event_rate_by_time_bin.png").exists()


def test_observed_event_rate_handles_bool_event_column(tmp_path: Path) -> None:
    riskset = _base_riskset(event_values=[False, False, True, False])

    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)
    _write_observed_event_rate_outputs(tmp_path, summary, qc)

    assert qc["n_events_input"] == 1
    assert qc["sum_binned_events"] == 1
    assert len(_read_nonzero_bins(tmp_path)) == 1


def test_observed_event_rate_does_not_drop_events_for_nan_feature_columns() -> None:
    riskset = _base_riskset(event_values=[0, 0, 1, 0]).assign(
        information_rate=np.nan,
        prop_expected_cumulative_info=np.nan,
    )

    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)

    assert qc["n_events_after_required_column_filter"] == 1
    assert qc["sum_binned_events"] == 1
    assert int(summary["n_events"].sum()) == 1


def test_observed_event_rate_final_edge_binning_includes_max_event() -> None:
    riskset = pd.DataFrame(
        {
            "time_from_partner_onset": [0.0, 0.05, 0.10, 0.15],
            "event": [0, 0, 0, 1],
        }
    )

    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)

    assert qc["sum_binned_events"] == 1
    assert int(summary["n_events"].sum()) == 1


def test_observed_event_rate_zero_event_input_writes_warning_and_plot(tmp_path: Path) -> None:
    riskset = _base_riskset(event_values=[0, 0, 0, 0])

    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)
    _write_observed_event_rate_outputs(tmp_path, summary, qc)
    plot_observed_event_rate_by_time_bin(summary, qc, tmp_path / "observed_event_rate_by_time_bin.png")

    payload = json.loads((tmp_path / "observed_event_rate_plot_qc.json").read_text(encoding="utf-8"))
    assert (tmp_path / "observed_event_rate_by_time_bin.png").exists()
    assert "zero_event_rows_input" in payload["warning_flags"]


def test_observed_event_rate_plot_creates_missing_parent_directory(tmp_path: Path) -> None:
    riskset = _base_riskset(event_values=[0, 0, 1, 0])
    summary, qc = summarize_observed_event_rate_by_time_bin(riskset)

    output_path = tmp_path / "nested" / "figures" / "observed_event_rate_by_time_bin.png"
    plot_observed_event_rate_by_time_bin(summary, qc, output_path)

    assert output_path.exists()


def _base_riskset(*, event_values: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_from_partner_onset": [0.0, 0.05, 0.10, 0.15],
            "event": event_values,
            "bin_index": [0, 1, 2, 3],
        }
    )


def _write_observed_event_rate_outputs(tmp_path: Path, summary: pd.DataFrame, qc: dict[str, object]) -> None:
    write_table(summary, tmp_path / "observed_event_rate_by_time_bin.csv", sep=",")
    write_table(summary.loc[summary["n_events"] > 0].reset_index(drop=True), tmp_path / "observed_event_rate_nonzero_bins.csv", sep=",")
    write_json(qc, tmp_path / "observed_event_rate_plot_qc.json")


def _read_nonzero_bins(tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "observed_event_rate_nonzero_bins.csv")
