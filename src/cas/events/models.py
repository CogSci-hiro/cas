"""Datamodels for deterministic FPP-SPP event extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ActionInterval:
    """A normalized action-tier interval used for pairing.

    Parameters
    ----------
    file_path
        Source TextGrid path.
    tier_name
        Canonical action tier name.
    interval_index
        One-based interval index within the tier.
    onset
        Interval onset in seconds.
    offset
        Interval offset in seconds.
    raw_label
        Raw interval label text.
    normalized_label
        Safely normalized interval label text.
    speaker
        Speaker inferred from the action tier name.
    """

    file_path: Path
    tier_name: str
    interval_index: int
    onset: float
    offset: float
    raw_label: str
    normalized_label: str
    speaker: str

    @property
    def duration(self) -> float:
        """Return interval duration in seconds."""

        return self.offset - self.onset


@dataclass(frozen=True, slots=True)
class PairingCandidate:
    """A candidate SPP interval for a specific FPP interval."""

    fpp: ActionInterval
    spp: ActionInterval
    latency: float
    offset_distance: float


@dataclass(frozen=True, slots=True)
class FppSppEvent:
    """A canonical extracted FPP-SPP pair event."""

    recording_id: str
    run: str
    file_path: str
    part: str
    response: str
    speaker_fpp: str
    speaker_spp: str
    fpp_label: str
    spp_label: str
    fpp_onset: float
    fpp_offset: float
    spp_onset: float
    spp_offset: float
    fpp_duration: float
    spp_duration: float
    latency: float
    pair_id: str

    def to_csv_row(self) -> dict[str, str | float]:
        """Return a CSV-compatible row mapping."""

        return {
            "recording_id": self.recording_id,
            "run": self.run,
            "file_path": self.file_path,
            "part": self.part,
            "response": self.response,
            "speaker_fpp": self.speaker_fpp,
            "speaker_spp": self.speaker_spp,
            "fpp_label": self.fpp_label,
            "spp_label": self.spp_label,
            "fpp_onset": self.fpp_onset,
            "fpp_offset": self.fpp_offset,
            "spp_onset": self.spp_onset,
            "spp_offset": self.spp_offset,
            "fpp_duration": self.fpp_duration,
            "spp_duration": self.spp_duration,
            "latency": self.latency,
            "pair_id": self.pair_id,
        }


@dataclass(frozen=True, slots=True)
class PairingIssue:
    """A deterministic extraction issue recorded during pairing."""

    file_path: str
    recording_id: str
    run: str
    fpp_tier: str
    fpp_index: int | None
    fpp_label: str
    fpp_onset: float | None
    fpp_offset: float | None
    issue_code: str
    message: str
    n_candidates: int

    def to_csv_row(self) -> dict[str, str | int | float | None]:
        """Return a CSV-compatible row mapping."""

        return {
            "file_path": self.file_path,
            "recording_id": self.recording_id,
            "run": self.run,
            "fpp_tier": self.fpp_tier,
            "fpp_index": self.fpp_index,
            "fpp_label": self.fpp_label,
            "fpp_onset": self.fpp_onset,
            "fpp_offset": self.fpp_offset,
            "issue_code": self.issue_code,
            "message": self.message,
            "n_candidates": self.n_candidates,
        }


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Configuration for deterministic event extraction.

    Parameters
    ----------
    allow_same_speaker_spp
        Whether same-speaker SPP intervals may be considered.
    pairing_margin_s
        Symmetric window in seconds around the FPP offset used to search for
        candidate SPP onsets.
    metadata_regexes
        Regex patterns used to derive recording metadata from file names.
    strict
        Whether any extraction issue should produce a non-zero CLI exit code.
    """

    allow_same_speaker_spp: bool = False
    pairing_margin_s: float = 1.0
    metadata_regexes: tuple[str, ...] = field(
        default_factory=lambda: (
            r"(?P<recording_id>.+?)_run[-_ ]?(?P<run>\d+)(?:_.+)?$",
            r"(?P<recording_id>.+?)[-_ ]run[-_ ]?(?P<run>\d+)(?:[-_ ].+)?$",
        )
    )
    strict: bool = False


@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    """Recording metadata derived conservatively from the file name."""

    recording_id: str
    run: str


@dataclass(slots=True)
class ExtractionResult:
    """Combined extraction result across one or more files."""

    events: list[FppSppEvent] = field(default_factory=list)
    issues: list[PairingIssue] = field(default_factory=list)
    unpaired_fpp: list[ActionInterval] = field(default_factory=list)
    unused_spp: list[ActionInterval] = field(default_factory=list)
    files_processed: int = 0
    fpp_count: int = 0
