"""Regression tests for the standalone TextGrid FPP/SPP pairing script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "dev" / "textgrid_fpp_spp_pairs.py"


def _load_script_module():
    """Import the standalone script in a dataclass-safe way."""
    module_name = "textgrid_fpp_spp_pairs_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_textgrid_paths_recurses(tmp_path: Path) -> None:
    """Nested matching TextGrids should be discovered under the input root."""

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (tmp_path / "dyad-011_run-1_combined.TextGrid").write_text("", encoding="utf-8")
    (nested_dir / "dyad-012_run-2_combined.TextGrid").write_text("", encoding="utf-8")
    (nested_dir / "ignore_me.TextGrid").write_text("", encoding="utf-8")

    script = _load_script_module()
    paths = script.discover_textgrid_paths(tmp_path)

    assert paths == [
        tmp_path / "dyad-011_run-1_combined.TextGrid",
        nested_dir / "dyad-012_run-2_combined.TextGrid",
    ]


def test_parse_interval_tiers_normalizes_action_tier_variants() -> None:
    """Action tier spelling and case variants should normalize safely."""

    script = _load_script_module()
    tiers = script.parse_interval_tiers(
        _build_textgrid_text(action_a_name="action a", action_b_name="Action B")
    )

    assert "actions A" in tiers
    assert "actions B" in tiers
    assert "action a" not in tiers
    assert "Action B" not in tiers


def test_pair_events_for_file_supports_action_tier_variants(tmp_path: Path) -> None:
    """Files using singular action tiers should still produce matched rows."""

    path = tmp_path / "dyad-011_run-1_combined.TextGrid"
    path.write_text(
        _build_textgrid_text(action_a_name="action A", action_b_name="action B"),
        encoding="utf-16",
    )

    script = _load_script_module()
    rows = script.pair_events_for_file(path)

    assert len(rows) == 1
    assert rows[0]["dyad_id"] == "011"
    assert rows[0]["run"] == "1"
    assert rows[0]["fpp_speaker_id"] == "subject-021"
    assert rows[0]["spp_speaker_id"] == "subject-022"


def test_normalize_action_label_fixes_known_typos() -> None:
    """Known action-label typos should normalize to canonical components."""

    script = _load_script_module()

    assert script.normalize_action_label("SPP_C ONF_SIMP") == "SPP_CONF_SIMP"
    assert script.normalize_action_label("SPP_CONC_SIMP") == "SPP_CONF_SIMP"
    assert script.normalize_action_label("SPP_ONF_SIMP") == "SPP_CONF_SIMP"
    assert script.normalize_action_label("SPP_DSIC_SIMP") == "SPP_DISC_SIMP"
    assert script.normalize_action_label("SPP_DISCèTRBL") == "SPP_DISC_TRBL"
    assert script.normalize_action_label("FPP_RFC_?") == "FPP_RFC"


def test_infer_subject_ids_from_dyad_id() -> None:
    """Dyad IDs should map to odd/even subject IDs for A/B speakers."""

    script = _load_script_module()

    assert script.infer_subject_ids("001") == {
        "A": "subject-001",
        "B": "subject-002",
    }
    assert script.infer_subject_ids("011") == {
        "A": "subject-021",
        "B": "subject-022",
    }


def _build_textgrid_text(*, action_a_name: str, action_b_name: str) -> str:
    """Create a minimal ooTextFile TextGrid payload for testing."""

    return f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5
tiers? <exists>
size = 4
item []:
    item [1]:
        class = "IntervalTier"
        name = "palign-A"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "hello"
    item [2]:
        class = "IntervalTier"
        name = "palign-B"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 1.2
            xmax = 1.7
            text = "yes"
    item [3]:
        class = "IntervalTier"
        name = "{action_a_name}"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "FPP_RFC_DECL"
    item [4]:
        class = "IntervalTier"
        name = "{action_b_name}"
        xmin = 0
        xmax = 5
        intervals: size = 1
        intervals [1]:
            xmin = 1.2
            xmax = 1.7
            text = "SPP_CONF_SIMP"
"""
