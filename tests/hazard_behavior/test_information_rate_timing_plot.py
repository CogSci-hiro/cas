from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cas.hazard_behavior.plots import compute_binned_median_iqr, plot_information_rate_by_partner_time


def _synthetic_riskset(n_rows: int = 200) -> pd.DataFrame:
    onset = pd.Series([index / 40.0 for index in range(n_rows)], dtype=float)
    offset = onset - 1.0
    information_rate = 0.5 + 0.2 * onset - 0.1 * offset
    return pd.DataFrame(
        {
            "time_from_partner_onset": onset,
            "time_from_partner_offset": offset,
            "information_rate_lag_150ms": information_rate,
            "partner_ipu_duration": [1.0] * n_rows,
            "event": [0] * (n_rows - 10) + [1] * 10,
        }
    )


def test_binned_summary_works() -> None:
    riskset = _synthetic_riskset()

    summary = compute_binned_median_iqr(
        riskset,
        "time_from_partner_onset",
        "information_rate_lag_150ms",
        20,
    )

    assert not summary.empty
    assert {
        "bin_center",
        "median_information_rate",
        "q25_information_rate",
        "q75_information_rate",
        "n_rows",
    } <= set(summary.columns)


def test_plot_writes_files(tmp_path: Path) -> None:
    riskset = _synthetic_riskset()

    output_path = plot_information_rate_by_partner_time(riskset, tmp_path)

    assert output_path == tmp_path / "information_rate_by_partner_time.png"
    assert (tmp_path / "information_rate_by_partner_time.png").exists()
    assert (tmp_path / "information_rate_by_partner_time.csv").exists()


def test_z_scored_fallback(tmp_path: Path) -> None:
    riskset = _synthetic_riskset().drop(columns=["information_rate_lag_150ms"])
    riskset["z_information_rate_lag_150ms"] = pd.Series(
        [(-1.5 + 3.0 * index / max(1, len(riskset) - 1)) for index in range(len(riskset))],
        dtype=float,
    )

    with pytest.warns(UserWarning, match="z-scored information rate"):
        output_path = plot_information_rate_by_partner_time(riskset, tmp_path)

    assert output_path == tmp_path / "information_rate_by_partner_time.png"
    assert (tmp_path / "information_rate_by_partner_time.png").exists()
    assert (tmp_path / "information_rate_by_partner_time.csv").exists()


def test_missing_columns_warns_and_returns_none(tmp_path: Path) -> None:
    riskset = _synthetic_riskset().drop(columns=["time_from_partner_offset"])

    with pytest.warns(UserWarning, match="requires columns"):
        output_path = plot_information_rate_by_partner_time(riskset, tmp_path)

    assert output_path is None
    assert not (tmp_path / "information_rate_by_partner_time.png").exists()
