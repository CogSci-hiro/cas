"""Tests for deterministic FPP-SPP event extraction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from cas.annotations.action_labels import infer_speaker_from_action_tier, is_fpp_label, is_spp_label
from cas.annotations.io import load_textgrid, write_textgrid
from cas.annotations.models import Interval, TextGrid, Tier
from cas.cli.main import main
from cas.events import ExtractionConfig, extract_events_from_textgrid, extract_recording_metadata
from cas.events.io import write_events_csv
from cas.events.models import ActionInterval
from cas.events.pairing import build_pairing_candidates


def test_fpp_spp_label_classification() -> None:
    """FPP and SPP label detection should use normalized prefixes."""

    assert is_fpp_label(" fpp-rfc-decl ")
    assert is_spp_label(" spp_conf_simp ")
    assert not is_fpp_label("MISC_AMBIG")
    assert not is_spp_label("MISC_AMBIG")


def test_infer_speaker_from_action_tier() -> None:
    """Speaker identity should follow canonical action tier names."""

    assert infer_speaker_from_action_tier("action A") == "A"
    assert infer_speaker_from_action_tier(" action_B ") == "B"


def test_extracts_closest_spp_onset_within_margin(tmp_path: Path) -> None:
    """The closest eligible SPP onset to the FPP offset should be paired."""

    path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.30, "FPP_RFC_DECL")],
        action_b_intervals=[
            Interval(0.27, 0.29, "SPP_CONF_SIMP"),
            Interval(0.35, 0.40, "SPP_CONF_EXP"),
            Interval(0.50, 0.60, "SPP_DISC_CORR"),
        ],
    )

    result = extract_events_from_textgrid(path, ExtractionConfig(pairing_margin_s=0.10))

    assert len(result.events) == 1
    event = result.events[0]
    assert event.spp_label == "SPP_CONF_SIMP"
    assert event.latency == pytest.approx(-0.03)


def test_one_to_one_pairing_prevents_spp_reuse(tmp_path: Path) -> None:
    """A paired SPP should not be reused for a later FPP."""

    path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[
            Interval(0.10, 0.20, "FPP_RFC_DECL"),
            Interval(0.22, 0.24, "FPP_RFC_INT"),
        ],
        action_b_intervals=[Interval(0.30, 0.35, "SPP_CONF_SIMP")],
    )

    result = extract_events_from_textgrid(path, ExtractionConfig())

    assert len(result.events) == 1
    assert len(result.unpaired_fpp) == 1
    issue_codes = [issue.issue_code for issue in result.issues]
    assert "reused_spp_prevented" in issue_codes
    assert "no_opposite_spp" in issue_codes


def test_pairing_tie_break_is_deterministic() -> None:
    """Equal-distance candidates should resolve by onset, offset, and index."""

    fpp = ActionInterval(
        file_path=Path("/tmp/test.TextGrid"),
        tier_name="action A",
        interval_index=1,
        onset=0.0,
        offset=0.5,
        raw_label="FPP_RFC_DECL",
        normalized_label="FPP_RFC_DECL",
        speaker="A",
    )
    candidates, _ = build_pairing_candidates(
        fpp,
        [
            ActionInterval(
                file_path=Path("/tmp/test.TextGrid"),
                tier_name="action B",
                interval_index=3,
                onset=0.6,
                offset=0.9,
                raw_label="SPP_CONF_EXP",
                normalized_label="SPP_CONF_EXP",
                speaker="B",
            ),
            ActionInterval(
                file_path=Path("/tmp/test.TextGrid"),
                tier_name="action B",
                interval_index=2,
                onset=0.4,
                offset=0.7,
                raw_label="SPP_CONF_SIMP",
                normalized_label="SPP_CONF_SIMP",
                speaker="B",
            ),
        ],
        ExtractionConfig(pairing_margin_s=0.2),
        used_spp_keys=set(),
    )

    assert [candidate.spp.interval_index for candidate in candidates] == [2, 3]


def test_metadata_extraction_from_filename() -> None:
    """Recording metadata should be derived conservatively from the stem."""

    metadata = extract_recording_metadata(Path("/tmp/dyad-007_run-5_combined.TextGrid"), ExtractionConfig())
    assert metadata.recording_id == "dyad-007"
    assert metadata.run == "5"


def test_events_csv_writing(tmp_path: Path) -> None:
    """Events CSV output should include the canonical event columns."""

    path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.20, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.25, 0.35, "SPP_CONF_SIMP")],
    )
    result = extract_events_from_textgrid(path, ExtractionConfig())
    csv_path = tmp_path / "events.csv"
    write_events_csv(result.events, csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["recording_id"] == "dyad-001"
    assert rows[0]["run"] == "2"
    assert rows[0]["part"] == "SPP"
    assert rows[0]["response"] == "CONF"
    assert rows[0]["pair_id"] == "pair_A0001_B0001"


def test_handling_files_with_no_valid_pairs(tmp_path: Path) -> None:
    """Files with FPPs but no valid SPPs should yield no event rows."""

    path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.20, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.60, 0.68, "SPP_CONF_SIMP")],
    )

    result = extract_events_from_textgrid(path, ExtractionConfig(pairing_margin_s=0.1))

    assert result.events == []
    assert len(result.unpaired_fpp) == 1
    assert any(issue.issue_code == "no_opposite_spp" for issue in result.issues)


def test_negative_latency_is_valid_within_pairing_margin(tmp_path: Path) -> None:
    """Negative latency is valid when the SPP onset falls within the matching window."""

    path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.20, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.195, 0.30, "SPP_CONF_SIMP")],
    )

    result = extract_events_from_textgrid(path, ExtractionConfig(pairing_margin_s=0.01))

    assert len(result.events) == 1
    assert result.events[0].latency == pytest.approx(-0.005)
    assert not any(issue.issue_code == "negative_latency_tolerated" for issue in result.issues)


def test_load_textgrid_supports_utf16(tmp_path: Path) -> None:
    """TextGrid loading should support UTF-16 encoded Praat files."""

    path = _write_event_grid(
        tmp_path / "utf16.TextGrid",
        action_a_intervals=[Interval(0.10, 0.20, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.25, 0.35, "SPP_CONF_SIMP")],
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-16")

    textgrid = load_textgrid(path)

    assert [tier.name for tier in textgrid.tiers][-2:] == ["action A", "action B"]


def test_extract_events_cli(tmp_path: Path, monkeypatch) -> None:
    """The annotations CLI should write canonical events and optional issue reports."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_event_grid(
        input_dir / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.20, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.25, 0.35, "SPP_CONF_SIMP")],
    )
    events_csv = tmp_path / "events.csv"
    issues_csv = tmp_path / "pairing_issues.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cas",
            "annotations",
            "extract-events",
            str(input_dir),
            str(events_csv),
            "--output-pairing-issues-csv",
            str(issues_csv),
        ],
    )

    assert main() == 0
    assert events_csv.exists()
    assert issues_csv.exists()


def _write_event_grid(
    path: Path,
    *,
    action_a_intervals: list[Interval],
    action_b_intervals: list[Interval],
) -> Path:
    """Write a TextGrid fixture with the required tier inventory."""

    tiers = [
        Tier(
            name="palign-A",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "#")],
        ),
        Tier(
            name="palign-B",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "#")],
        ),
        Tier(
            name="ipu-A",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "ipu a")],
        ),
        Tier(
            name="ipu-B",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "ipu b")],
        ),
        Tier(name="action A", xmin=0.0, xmax=1.0, intervals=action_a_intervals),
        Tier(name="action B", xmin=0.0, xmax=1.0, intervals=action_b_intervals),
    ]
    write_textgrid(TextGrid(xmin=0.0, xmax=1.0, tiers=tiers), path)
    return path
