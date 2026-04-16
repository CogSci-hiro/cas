"""Constants used by TextGrid annotation validation."""

from __future__ import annotations

from typing import Final

TEXTGRID_SUFFIXES: Final[tuple[str, ...]] = (".textgrid",)
DEFAULT_FLOAT_EPSILON: Final[float] = 1e-9
MILLISECONDS_PER_SECOND: Final[float] = 1000.0

SEVERITY_ERROR: Final[str] = "ERROR"
SEVERITY_WARNING: Final[str] = "WARNING"
SEVERITY_INFO: Final[str] = "INFO"

EXPECTED_TIERS: Final[tuple[str, ...]] = (
    "palign-A",
    "palign-B",
    "ipu-A",
    "ipu-B",
    "action A",
    "action B",
)

PALIGN_TIERS: Final[tuple[str, ...]] = ("palign-A", "palign-B")
IPU_TIERS: Final[tuple[str, ...]] = ("ipu-A", "ipu-B")
ACTION_TIERS: Final[tuple[str, ...]] = ("action A", "action B")

TIER_GROUP_TO_SPEAKER: Final[dict[str, str]] = {
    "palign-A": "A",
    "palign-B": "B",
    "ipu-A": "A",
    "ipu-B": "B",
    "action A": "A",
    "action B": "B",
}

ALLOWED_ACTION_LABELS: Final[frozenset[str]] = frozenset(
    {
        "FPP_RFC_DECL",
        "FPP_RFC_TAG",
        "FPP_RFC_INF",
        "FPP_RFC_INT",
        "FPP_RFRC",
        "SPP_CONF_SIMP",
        "SPP_CONF_ECHO",
        "SPP_CONF_EXP",
        "SPP_CONF_OVERLAP",
        "SPP_DISC_SIMP",
        "SPP_DISC_CORR",
        "SPP_DISC_TRBL",
        "MISC_AMBIG",
        "MISC_NO_UPT",
        "MISC_AUTO_D",
        "MISC_BACK_OVERLAP",
        "MISC_SEQ_CLOSE",
    }
)

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "file_path",
    "tier_name",
    "interval_index",
    "xmin",
    "xmax",
    "original_label",
    "normalized_label",
    "issue_code",
    "severity",
    "message",
    "auto_corrected",
    "correction_type",
)
