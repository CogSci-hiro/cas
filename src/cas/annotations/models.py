"""Datamodels for TextGrid annotation validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(slots=True)
class Interval:
    """A single interval annotation.

    Parameters
    ----------
    xmin
        Start time in seconds.
    xmax
        End time in seconds.
    text
        Interval label.
    """

    xmin: float
    xmax: float
    text: str


@dataclass(slots=True)
class Tier:
    """An interval tier in a TextGrid."""

    name: str
    xmin: float
    xmax: float
    intervals: list[Interval] = field(default_factory=list)
    class_name: str = "IntervalTier"


@dataclass(slots=True)
class TextGrid:
    """A TextGrid with interval tiers only."""

    xmin: float
    xmax: float
    tiers: list[Tier] = field(default_factory=list)


@dataclass(slots=True)
class ValidationIssue:
    """A single validation event for reporting."""

    file_path: str
    tier_name: str
    interval_index: int | None
    xmin: float | None
    xmax: float | None
    original_label: str
    normalized_label: str
    issue_code: str
    severity: str
    message: str
    auto_corrected: bool
    correction_type: str

    def to_csv_row(self) -> dict[str, str | int | float | bool | None]:
        """Return the issue as a CSV-compatible mapping."""

        return {
            "file_path": self.file_path,
            "tier_name": self.tier_name,
            "interval_index": self.interval_index,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "original_label": self.original_label,
            "normalized_label": self.normalized_label,
            "issue_code": self.issue_code,
            "severity": self.severity,
            "message": self.message,
            "auto_corrected": self.auto_corrected,
            "correction_type": self.correction_type,
        }


@dataclass(slots=True)
class ValidationConfig:
    """Configuration for TextGrid validation.

    Parameters
    ----------
    write_corrected
        Whether corrected TextGrids should be written.
    corrected_output_dir
        Destination directory for corrected TextGrids.
    fail_on_warning
        Whether warnings should produce a non-zero CLI exit code.
    snap_tolerance_ms
        Optional tolerance for snapping tiny overlap noise in milliseconds.
    allow_action_overlaps
        Whether action tiers may contain within-tier overlaps.
    """

    write_corrected: bool = False
    corrected_output_dir: Path | None = None
    fail_on_warning: bool = False
    snap_tolerance_ms: float = 0.0
    allow_action_overlaps: bool = False

    @property
    def snap_tolerance_s(self) -> float:
        """Return the snapping tolerance in seconds."""

        return self.snap_tolerance_ms / 1000.0


@dataclass(slots=True)
class FileValidationResult:
    """Validation result for a single TextGrid file."""

    file_path: Path
    original_textgrid: TextGrid
    corrected_textgrid: TextGrid
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count error-level issues."""

        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Count warning-level issues."""

        return sum(issue.severity == "WARNING" for issue in self.issues)

    @property
    def correction_count(self) -> int:
        """Count auto-correction info events."""

        return sum(issue.auto_corrected for issue in self.issues)


def clone_textgrid(textgrid: TextGrid) -> TextGrid:
    """Create a deep copy of a TextGrid."""

    return TextGrid(
        xmin=textgrid.xmin,
        xmax=textgrid.xmax,
        tiers=[
            Tier(
                name=tier.name,
                xmin=tier.xmin,
                xmax=tier.xmax,
                class_name=tier.class_name,
                intervals=[replace(interval) for interval in tier.intervals],
            )
            for tier in textgrid.tiers
        ],
    )
