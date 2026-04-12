"""Tests for TextGrid annotation validation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from cas.annotations.autocorrect import normalize_action_label, normalize_tier_name
from cas.annotations.io import load_textgrid, write_textgrid
from cas.annotations.models import Interval, TextGrid, Tier, ValidationConfig
from cas.annotations.report import write_csv_report
from cas.annotations.validation import validate_textgrid_file
from cas.cli.main import main


def test_normalize_tier_name_trivial_variant() -> None:
    """Tier name normalization should handle safe formatting variants."""

    result = normalize_tier_name("  action_A ")
    assert result.value == "action A"
    assert result.changed is True


def test_normalize_tier_name_plural_actions_variant() -> None:
    """Plural action tier labels should normalize to the canonical singular form."""

    result = normalize_tier_name("actions B")
    assert result.value == "action B"
    assert result.changed is True


def test_normalize_action_label_trivial_variant() -> None:
    """Action label normalization should canonicalize safe variants."""

    result = normalize_action_label(" spp-conf-simp ")
    assert result.value == "SPP_CONF_SIMP"
    assert result.changed is True


def test_invalid_action_label_detection(tmp_path: Path) -> None:
    """Invalid action labels should be reported without semantic guessing."""

    path = _write_grid(
        tmp_path / "invalid_label.TextGrid",
        action_a_label="not_a_real_label",
    )

    result = validate_textgrid_file(path, ValidationConfig())
    issue_codes = {issue.issue_code for issue in result.issues}
    assert "invalid_label" in issue_codes


def test_missing_tier_detection(tmp_path: Path) -> None:
    """Missing required tiers should be surfaced as errors."""

    path = _write_grid(tmp_path / "missing.TextGrid", include_action_b=False)

    result = validate_textgrid_file(path, ValidationConfig())
    assert any(issue.issue_code == "missing_tier" and issue.tier_name == "action B" for issue in result.issues)


def test_overlap_detection(tmp_path: Path) -> None:
    """Overlapping consecutive intervals should be reported."""

    palign_a = Tier(
        name="palign-A",
        xmin=0.0,
        xmax=1.0,
        intervals=[
            Interval(0.0, 0.6, "hello"),
            Interval(0.5, 0.9, "world"),
        ],
    )
    path = _write_grid(tmp_path / "overlap.TextGrid", palign_a=palign_a)

    result = validate_textgrid_file(path, ValidationConfig())
    assert any(issue.issue_code == "overlapping_intervals" for issue in result.issues)


def test_csv_report_generation(tmp_path: Path) -> None:
    """CSV reporting should include the expected columns and issue rows."""

    path = _write_grid(tmp_path / "csv.TextGrid", include_action_b=False)
    result = validate_textgrid_file(path, ValidationConfig())
    csv_path = tmp_path / "report.csv"
    write_csv_report(result.issues, csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0]["file_path"] == str(path)
    assert "issue_code" in rows[0]


def test_safe_autocorrection_behavior(tmp_path: Path) -> None:
    """Safe corrections should be applied to corrected TextGrid output only."""

    path = _write_grid(
        tmp_path / "corrected.TextGrid",
        tier_name_overrides={"action A": " action_A "},
        action_a_label=" spp-conf-simp ",
    )

    result = validate_textgrid_file(path, ValidationConfig())
    corrected_action_tier = next(tier for tier in result.corrected_textgrid.tiers if tier.name == "action A")
    assert corrected_action_tier.intervals[0].text == "SPP_CONF_SIMP"
    assert any(issue.issue_code == "tier_name_normalized" for issue in result.issues)
    assert any(issue.issue_code == "label_normalized" for issue in result.issues)


def test_cli_exit_code_behavior(tmp_path: Path, monkeypatch) -> None:
    """The CLI should return non-zero when validation errors exist."""

    invalid_dir = tmp_path / "invalid_input"
    invalid_dir.mkdir()
    _write_grid(invalid_dir / "cli_invalid.TextGrid", include_action_b=False)
    invalid_out_dir = tmp_path / "invalid_logs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "annotations",
            "validate",
            str(invalid_dir),
            str(invalid_out_dir),
        ],
    )
    assert main() == 1

    valid_dir = tmp_path / "valid_input"
    valid_dir.mkdir()
    _write_grid(valid_dir / "cli_valid.TextGrid")
    valid_out_dir = tmp_path / "valid_logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "annotations",
            "validate",
            str(valid_dir),
            str(valid_out_dir),
        ],
    )
    assert main() == 0
    assert (valid_out_dir / "validation_summary.csv").exists()
    assert (valid_out_dir / "cli_valid.TextGrid.csv").exists()


def test_write_corrected_outputs(tmp_path: Path, monkeypatch) -> None:
    """The CLI should write corrected TextGrids when requested."""

    input_dir = tmp_path / "corrected_input"
    input_dir.mkdir()
    input_path = _write_grid(
        input_dir / "write_corrected.TextGrid",
        tier_name_overrides={"action A": "action_A"},
        action_a_label="fpp_rfc_decl",
    )
    output_dir = tmp_path / "write_corrected_logs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "annotations",
            "validate",
            str(input_dir),
            str(output_dir),
            "--write-corrected",
        ],
    )
    assert main() == 0

    corrected_path = output_dir / "corrected" / input_path.name
    corrected_grid = load_textgrid(corrected_path)
    corrected_action_tier = next(tier for tier in corrected_grid.tiers if tier.name == "action A")
    assert corrected_action_tier.intervals[0].text == "FPP_RFC_DECL"


def test_cli_writes_per_file_csv_logs(tmp_path: Path, monkeypatch) -> None:
    """Directory validation should emit one CSV log per TextGrid plus a summary CSV."""

    input_dir = tmp_path / "batch_input"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    _write_grid(input_dir / "first.TextGrid")
    _write_grid(nested_dir / "second.TextGrid", include_action_b=False)
    output_dir = tmp_path / "batch_logs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "annotations",
            "validate",
            str(input_dir),
            str(output_dir),
            "--recursive",
        ],
    )
    assert main() == 1
    assert (output_dir / "validation_summary.csv").exists()
    assert (output_dir / "first.TextGrid.csv").exists()
    assert (output_dir / "nested" / "second.TextGrid.csv").exists()


def _write_grid(
    path: Path,
    *,
    include_action_b: bool = True,
    action_a_label: str = "FPP_RFC_DECL",
    palign_a: Tier | None = None,
    tier_name_overrides: dict[str, str] | None = None,
) -> Path:
    """Write a mostly valid TextGrid fixture to disk."""

    overrides = tier_name_overrides or {}
    tiers = [
        palign_a
        if palign_a is not None
        else Tier(
            name=overrides.get("palign-A", "palign-A"),
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 0.5, "hello"), Interval(0.5, 1.0, "world")],
        ),
        Tier(
            name=overrides.get("palign-B", "palign-B"),
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 0.5, "foo"), Interval(0.5, 1.0, "bar")],
        ),
        Tier(
            name=overrides.get("ipu-A", "ipu-A"),
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "utterance a")],
        ),
        Tier(
            name=overrides.get("ipu-B", "ipu-B"),
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "utterance b")],
        ),
        Tier(
            name=overrides.get("action A", "action A"),
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.1, 0.4, action_a_label)],
        ),
    ]
    if include_action_b:
        tiers.append(
            Tier(
                name=overrides.get("action B", "action B"),
                xmin=0.0,
                xmax=1.0,
                intervals=[Interval(0.6, 0.9, "SPP_CONF_SIMP")],
            )
        )

    textgrid = TextGrid(xmin=0.0, xmax=1.0, tiers=tiers)
    write_textgrid(textgrid, path)
    return path
