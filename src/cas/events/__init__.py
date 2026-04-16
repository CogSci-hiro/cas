"""Canonical event extraction for annotated conversational TextGrids.

The current event table is a deterministic timing representation rather than a
full conversational theory. It captures only what is mechanically encoded in
action tiers: normalized FPP/SPP labels, speaker identity, and pair timing
under a documented offset-centered nearest-onset rule. Ambiguous or unsupported
cases are reported instead of being inferred.
"""

from cas.events.extract import extract_events_from_paths, extract_events_from_textgrid, extract_recording_metadata
from cas.events.io import write_events_csv, write_pairing_issues_csv
from cas.events.models import (
    ActionInterval,
    ExtractionConfig,
    ExtractionResult,
    FppSppEvent,
    PairingCandidate,
    PairingIssue,
    RecordingMetadata,
)

__all__ = [
    "ActionInterval",
    "ExtractionConfig",
    "ExtractionResult",
    "FppSppEvent",
    "PairingCandidate",
    "PairingIssue",
    "RecordingMetadata",
    "extract_events_from_paths",
    "extract_events_from_textgrid",
    "extract_recording_metadata",
    "write_events_csv",
    "write_pairing_issues_csv",
]
