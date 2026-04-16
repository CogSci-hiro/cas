"""Canonical FPP-SPP event extraction from validated TextGrid annotations.

This module defines a conservative operational representation of adjacency-like
pair timing for downstream EEG analyses. It only extracts pair timing that is
mechanically supported by action-tier annotations and deterministic pairing
rules. It intentionally does not infer richer dialogue structure beyond the
explicit FPP_/SPP_ labels and the configured offset-centered search rule.
"""

from __future__ import annotations

from pathlib import Path
import re

from cas.annotations.action_labels import (
    infer_speaker_from_action_tier,
    is_fpp_label,
    is_spp_label,
    normalize_action_label_for_classification,
)
from cas.annotations.autocorrect import normalize_tier_name
from cas.annotations.constants import ACTION_TIERS, ALLOWED_ACTION_LABELS
from cas.annotations.io import load_textgrid
from cas.events.models import (
    ActionInterval,
    ExtractionConfig,
    ExtractionResult,
    FppSppEvent,
    PairingIssue,
    RecordingMetadata,
)
from cas.events.pairing import build_pairing_candidates, select_best_candidate

_MISSING_VALUE = ""


def extract_events_from_paths(paths: list[Path], config: ExtractionConfig) -> ExtractionResult:
    """Extract canonical FPP-SPP events from multiple TextGrid files."""

    result = ExtractionResult(files_processed=len(paths))
    for path in paths:
        file_result = extract_events_from_textgrid(path, config)
        result.events.extend(file_result.events)
        result.issues.extend(file_result.issues)
        result.unpaired_fpp.extend(file_result.unpaired_fpp)
        result.unused_spp.extend(file_result.unused_spp)
        result.fpp_count += file_result.fpp_count
    return result


def extract_events_from_textgrid(path: Path, config: ExtractionConfig) -> ExtractionResult:
    """Extract canonical FPP-SPP events from one TextGrid file."""

    metadata = extract_recording_metadata(path, config)
    issues: list[PairingIssue] = []
    action_intervals = extract_action_intervals(path, config, metadata, issues)

    fpp_intervals = sorted(
        [interval for interval in action_intervals if is_fpp_label(interval.normalized_label)],
        key=lambda interval: (interval.onset, interval.offset, interval.tier_name, interval.interval_index),
    )
    spp_intervals = [
        interval for interval in action_intervals if is_spp_label(interval.normalized_label)
    ]

    events: list[FppSppEvent] = []
    used_spp_keys: set[tuple[str, int]] = set()
    unpaired_fpp: list[ActionInterval] = []

    for fpp in fpp_intervals:
        eligible_spp_intervals = [
            spp
            for spp in spp_intervals
            if config.allow_same_speaker_spp or spp.speaker != fpp.speaker
        ]
        candidates, reused_spp_prevented = build_pairing_candidates(
            fpp,
            eligible_spp_intervals,
            config,
            used_spp_keys=used_spp_keys,
        )

        if reused_spp_prevented:
            issues.append(
                _build_issue(
                    metadata=metadata,
                    file_path=path,
                    fpp=fpp,
                    issue_code="reused_spp_prevented",
                    message="One or more otherwise eligible SPP candidates were already paired.",
                    n_candidates=len(candidates),
                )
            )

        best_candidate = select_best_candidate(candidates)
        if best_candidate is None:
            issues.append(
                _build_issue(
                    metadata=metadata,
                    file_path=path,
                    fpp=fpp,
                    issue_code="no_opposite_spp",
                    message="No eligible SPP interval was found after the FPP offset.",
                    n_candidates=0,
                )
            )
            unpaired_fpp.append(fpp)
            continue

        spp = best_candidate.spp
        used_spp_keys.add((spp.tier_name, spp.interval_index))
        events.append(
            _build_event(
                metadata=metadata,
                file_path=path,
                fpp=fpp,
                spp=spp,
            )
        )

    unused_spp = [
        spp
        for spp in spp_intervals
        if (spp.tier_name, spp.interval_index) not in used_spp_keys
    ]

    return ExtractionResult(
        events=events,
        issues=issues,
        unpaired_fpp=unpaired_fpp,
        unused_spp=unused_spp,
        files_processed=1,
        fpp_count=len(fpp_intervals),
    )


def extract_recording_metadata(path: Path, config: ExtractionConfig) -> RecordingMetadata:
    """Extract recording metadata conservatively from a file path."""

    stem = path.stem
    for pattern in config.metadata_regexes:
        match = re.search(pattern, stem)
        if match is None:
            continue

        recording_id = match.groupdict().get("recording_id", "") or stem
        run = match.groupdict().get("run", "") or _MISSING_VALUE
        return RecordingMetadata(recording_id=recording_id, run=run)

    return RecordingMetadata(recording_id=stem, run=_MISSING_VALUE)


def extract_action_intervals(
    path: Path,
    config: ExtractionConfig,
    metadata: RecordingMetadata,
    issues: list[PairingIssue],
) -> list[ActionInterval]:
    """Extract normalized action intervals from a TextGrid file."""

    textgrid = load_textgrid(path)
    action_intervals: list[ActionInterval] = []
    present_action_tiers: set[str] = set()

    if metadata.run == _MISSING_VALUE:
        issues.append(
            PairingIssue(
                file_path=str(path),
                recording_id=metadata.recording_id,
                run=metadata.run,
                fpp_tier="",
                fpp_index=None,
                fpp_label="",
                fpp_onset=None,
                fpp_offset=None,
                issue_code="missing_run_metadata",
                message="Run metadata could not be inferred from the file name.",
                n_candidates=0,
            )
        )

    for tier in textgrid.tiers:
        normalized_tier_name = normalize_tier_name(tier.name).value
        if normalized_tier_name not in ACTION_TIERS:
            continue

        present_action_tiers.add(normalized_tier_name)
        speaker = infer_speaker_from_action_tier(normalized_tier_name)
        for interval_index, interval in enumerate(tier.intervals, start=1):
            normalized_label = normalize_action_label_for_classification(interval.text)
            if not normalized_label:
                continue

            if normalized_label not in ALLOWED_ACTION_LABELS:
                issues.append(
                    PairingIssue(
                        file_path=str(path),
                        recording_id=metadata.recording_id,
                        run=metadata.run,
                        fpp_tier=normalized_tier_name,
                        fpp_index=interval_index,
                        fpp_label=normalized_label,
                        fpp_onset=interval.xmin,
                        fpp_offset=interval.xmax,
                        issue_code="invalid_action_label",
                        message=f"Action label '{interval.text}' is not part of the canonical inventory.",
                        n_candidates=0,
                    )
                )
                continue

            if not (is_fpp_label(normalized_label) or is_spp_label(normalized_label)):
                continue

            action_interval = ActionInterval(
                file_path=path,
                tier_name=normalized_tier_name,
                interval_index=interval_index,
                onset=interval.xmin,
                offset=interval.xmax,
                raw_label=interval.text,
                normalized_label=normalized_label,
                speaker=speaker,
            )

            if action_interval.duration < 0.0:
                issues.append(
                    _build_issue(
                        metadata=metadata,
                        file_path=path,
                        fpp=action_interval,
                        issue_code="invalid_action_label",
                        message="Action interval has negative duration and was skipped.",
                        n_candidates=0,
                    )
                )
                continue

            action_intervals.append(action_interval)

    missing_action_tiers = sorted(set(ACTION_TIERS) - present_action_tiers)
    if missing_action_tiers:
        raise ValueError(f"Missing required action tiers in {path}: {', '.join(missing_action_tiers)}")

    return action_intervals


def _build_event(
    *,
    metadata: RecordingMetadata,
    file_path: Path,
    fpp: ActionInterval,
    spp: ActionInterval,
) -> FppSppEvent:
    """Build a canonical event row from paired intervals."""

    fpp_duration = fpp.duration
    spp_duration = spp.duration
    latency = spp.onset - fpp.offset

    return FppSppEvent(
        recording_id=metadata.recording_id,
        run=metadata.run,
        file_path=str(file_path),
        part=_extract_part(spp.normalized_label),
        response=_extract_response(spp.normalized_label),
        speaker_fpp=fpp.speaker,
        speaker_spp=spp.speaker,
        fpp_label=fpp.normalized_label,
        spp_label=spp.normalized_label,
        fpp_onset=fpp.onset,
        fpp_offset=fpp.offset,
        spp_onset=spp.onset,
        spp_offset=spp.offset,
        fpp_duration=fpp_duration,
        spp_duration=spp_duration,
        latency=latency,
        pair_id=f"pair_{fpp.speaker}{fpp.interval_index:04d}_{spp.speaker}{spp.interval_index:04d}",
    )


def _extract_part(label: str) -> str:
    """Extract the coarse label family from a normalized action label."""

    return label.split("_", maxsplit=1)[0] if label else ""


def _extract_response(label: str) -> str:
    """Extract the response subtype from a normalized SPP label."""

    parts = label.split("_")
    if len(parts) >= 2 and parts[0] == "SPP":
        return parts[1]
    return ""


def _build_issue(
    *,
    metadata: RecordingMetadata,
    file_path: Path,
    fpp: ActionInterval,
    issue_code: str,
    message: str,
    n_candidates: int,
) -> PairingIssue:
    """Build an extraction issue tied to an FPP interval."""

    return PairingIssue(
        file_path=str(file_path),
        recording_id=metadata.recording_id,
        run=metadata.run,
        fpp_tier=fpp.tier_name,
        fpp_index=fpp.interval_index,
        fpp_label=fpp.normalized_label,
        fpp_onset=fpp.onset,
        fpp_offset=fpp.offset,
        issue_code=issue_code,
        message=message,
        n_candidates=n_candidates,
    )
