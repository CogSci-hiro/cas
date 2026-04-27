from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.hazard_behavior.diagnose_spp_timing_debug import (
    compute_duplicate_check,
    compute_event_consistency_checks,
    compute_event_nonevent_overlap,
    compute_spline_design_diagnostic,
    plot_event_histograms,
    simple_fit_row,
)


def test_timing_consistency_check_passes_and_flags_inconsistency() -> None:
    valid = pd.DataFrame(
        {
            "event_spp": [0, 1],
            "episode_id": ["e1", "e1"],
            "dyad_id": ["d1", "d1"],
            "run": ["1", "1"],
            "bin_start": [0.00, 0.05],
            "bin_end": [0.05, 0.10],
            "partner_ipu_onset": [0.00, 0.00],
            "partner_ipu_offset": [0.02, 0.02],
            "time_from_partner_onset": [0.00, 0.05],
            "time_from_partner_offset": [0.03, 0.08],
            "own_fpp_onset": [np.nan, 0.08],
        }
    )
    checks = compute_event_consistency_checks(valid, event_column="event_spp", onset_column="own_fpp_onset")
    assert checks.loc[checks["check_name"] == "time_from_partner_onset_matches_bin_end", "status"].item() == "fail"
    assert checks.loc[checks["check_name"] == "time_from_partner_onset_matches_bin_start", "status"].item() == "pass"
    assert checks.loc[checks["check_name"] == "time_from_partner_offset_matches", "status"].item() == "pass"

    invalid = valid.copy()
    invalid.loc[1, "time_from_partner_offset"] = 99.0
    bad_checks = compute_event_consistency_checks(invalid, event_column="event_spp", onset_column="own_fpp_onset")
    assert bad_checks.loc[bad_checks["check_name"] == "time_from_partner_offset_matches", "status"].item() == "fail"


def test_event_alignment_check_passes_and_flags_outside_bin() -> None:
    valid = pd.DataFrame(
        {
            "event_spp": [1],
            "episode_id": ["e1"],
            "bin_start": [0.00],
            "bin_end": [0.05],
            "partner_ipu_onset": [0.00],
            "partner_ipu_offset": [0.02],
            "time_from_partner_onset": [0.05],
            "time_from_partner_offset": [0.03],
            "own_fpp_onset": [0.03],
        }
    )
    checks = compute_event_consistency_checks(valid, event_column="event_spp", onset_column="own_fpp_onset")
    assert checks.loc[checks["check_name"] == "event_rows_align_with_onset_bin", "status"].item() == "pass"

    invalid = valid.copy()
    invalid.loc[0, "own_fpp_onset"] = 0.10
    bad = compute_event_consistency_checks(invalid, event_column="event_spp", onset_column="own_fpp_onset")
    assert bad.loc[bad["check_name"] == "event_rows_align_with_onset_bin", "status"].item() == "fail"


def test_duplicate_check_detects_duplicate_keys() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["d1", "d1"],
            "run": ["1", "1"],
            "speaker": ["A", "A"],
            "episode_id": ["e1", "e1"],
            "bin_start": [0.0, 0.0],
            "bin_end": [0.05, 0.05],
            "event_spp": [0, 1],
        }
    )
    summary = compute_duplicate_check(table)
    assert int(summary.loc[0, "n_duplicate_keys"]) == 1
    assert int(summary.loc[0, "n_duplicate_event_rows"]) == 1


def test_event_nonevent_overlap_metric_separated_and_overlapping() -> None:
    separated = pd.DataFrame(
        {
            "event_spp": [1, 1, 0, 0],
            "time_from_partner_onset": [0.10, 0.11, 1.00, 1.10],
            "time_from_partner_offset": [0.05, 0.06, 0.90, 1.00],
        }
    )
    sep_summary, _ = compute_event_nonevent_overlap(separated, event_column="event_spp", bin_width_s=0.05)
    assert sep_summary["time_from_partner_onset"]["proportion_nonevents_inside_event_range"] == 0.0

    overlapping = pd.DataFrame(
        {
            "event_spp": [1, 1, 0, 0],
            "time_from_partner_onset": [0.10, 0.20, 0.15, 0.18],
            "time_from_partner_offset": [0.05, 0.10, 0.06, 0.09],
        }
    )
    overlap_summary, _ = compute_event_nonevent_overlap(overlapping, event_column="event_spp", bin_width_s=0.05)
    assert overlap_summary["time_from_partner_onset"]["proportion_nonevents_inside_event_range"] > 0.0


def test_density_histogram_plot_smoke(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "event_spp": [0, 1, 0, 1, 0, 1],
            "time_from_partner_onset": np.linspace(0.0, 0.25, 6),
            "time_from_partner_offset": np.linspace(-0.10, 0.15, 6),
        }
    )
    output_path = tmp_path / "hist.png"
    plot_event_histograms(table, event_column="event_spp", output_path=output_path)
    assert output_path.exists()


def test_spline_design_diagnostic_flags_rank_problem() -> None:
    matrix = pd.DataFrame(
        {
            "Intercept": [1.0] * 5,
            "basis_1": [0.0] * 5,
            "basis_2": [1.0] * 5,
        }
    )
    summary = compute_spline_design_diagnostic(matrix)
    assert summary["rank_deficiency"] > 0
    assert "basis_1" in summary["near_constant_columns"]


def test_simple_fit_failure_handling_continues() -> None:
    table = pd.DataFrame(
        {
            "event_spp": [0, 1, 0, 1],
            "time_from_partner_onset": [0.0, 0.1, 0.2, 0.3],
            "time_from_partner_offset": [0.0, 0.1, 0.2, 0.3],
        }
    )
    failed = simple_fit_row(
        table,
        event_column="event_spp",
        model_name="bad_formula",
        formula="event_spp ~ nonexistent_column",
        sample_type="fit_sample",
    )
    assert failed["status"] == "failed"
    assert failed["error_message"]
