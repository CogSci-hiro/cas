"""Mechanical validation for conversational TextGrid annotations.

This module validates:
- expected tier presence and naming
- interval timing mechanics
- action-label inventory membership
- same-speaker overlap relationships between action, IPU, and palign tiers

It intentionally does not validate higher-level interactional semantics beyond
what can be checked mechanically from the available annotations.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from cas.annotations.autocorrect import collapse_whitespace, normalize_action_label, normalize_tier_name
from cas.annotations.constants import (
    ACTION_TIERS,
    ALLOWED_ACTION_LABELS,
    DEFAULT_FLOAT_EPSILON,
    EXPECTED_TIERS,
    IPU_TIERS,
    PALIGN_TIERS,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)
from cas.annotations.io import load_textgrid
from cas.annotations.models import FileValidationResult, Interval, Tier, ValidationConfig, ValidationIssue, clone_textgrid


def validate_textgrid_file(path: Path, config: ValidationConfig) -> FileValidationResult:
    """Load and validate a TextGrid file."""

    textgrid = load_textgrid(path)
    return validate_textgrid(path=path, textgrid=textgrid, config=config)


def validate_textgrids(paths: list[Path], config: ValidationConfig) -> list[FileValidationResult]:
    """Validate multiple TextGrid files."""

    return [validate_textgrid_file(path, config) for path in paths]


def validate_textgrid(path: Path, textgrid, config: ValidationConfig) -> FileValidationResult:
    """Validate a parsed TextGrid and return all issues."""

    corrected_textgrid = clone_textgrid(textgrid)
    issues: list[ValidationIssue] = []

    matched_tiers: dict[str, list[Tier]] = {tier_name: [] for tier_name in EXPECTED_TIERS}

    for tier in corrected_textgrid.tiers:
        name_result = normalize_tier_name(tier.name)
        if name_result.changed:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=None,
                    interval=None,
                    original_label=tier.name,
                    normalized_label=name_result.value,
                    issue_code="tier_name_normalized",
                    severity=SEVERITY_INFO,
                    message=f"Normalized tier name to '{name_result.value}'.",
                    auto_corrected=True,
                    correction_type=name_result.correction_type or "tier_name_normalized",
                )
            )
            tier.name = name_result.value

        if tier.name in matched_tiers:
            matched_tiers[tier.name].append(tier)
        else:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=None,
                    interval=None,
                    original_label=tier.name,
                    normalized_label=tier.name,
                    issue_code="unexpected_tier",
                    severity=SEVERITY_ERROR,
                    message=f"Unexpected tier '{tier.name}'.",
                )
            )

    for tier_name, tiers in matched_tiers.items():
        if not tiers:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier_name,
                    interval_index=None,
                    interval=None,
                    original_label="",
                    normalized_label="",
                    issue_code="missing_tier",
                    severity=SEVERITY_ERROR,
                    message=f"Missing required tier '{tier_name}'.",
                )
            )
        if len(tiers) > 1:
            for duplicate_index, duplicate_tier in enumerate(tiers[1:], start=2):
                issues.append(
                    _make_issue(
                        path=path,
                        tier_name=duplicate_tier.name,
                        interval_index=None,
                        interval=None,
                        original_label=duplicate_tier.name,
                        normalized_label=duplicate_tier.name,
                        issue_code="duplicate_tier",
                        severity=SEVERITY_ERROR,
                        message=f"Duplicate tier '{tier_name}' encountered at occurrence {duplicate_index}.",
                    )
                )

    for tier in corrected_textgrid.tiers:
        if tier.name in PALIGN_TIERS:
            issues.extend(_validate_interval_tier(path, tier, config, tier_kind="palign"))
        elif tier.name in IPU_TIERS:
            issues.extend(_validate_interval_tier(path, tier, config, tier_kind="ipu"))
        elif tier.name in ACTION_TIERS:
            issues.extend(_validate_action_tier(path, tier, config))

    issues.extend(_validate_cross_tier_relationships(path, matched_tiers, config))
    return FileValidationResult(
        file_path=path,
        original_textgrid=textgrid,
        corrected_textgrid=corrected_textgrid,
        issues=issues,
    )


def _validate_interval_tier(
    path: Path,
    tier: Tier,
    config: ValidationConfig,
    *,
    tier_kind: str,
    normalize_labels: bool = True,
) -> list[ValidationIssue]:
    """Validate a palign or IPU interval tier."""

    issues: list[ValidationIssue] = []
    sorted_intervals = sorted(tier.intervals, key=lambda interval: (interval.xmin, interval.xmax))
    if sorted_intervals != tier.intervals:
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier.name,
                interval_index=None,
                interval=None,
                original_label="",
                normalized_label="",
                issue_code="invalid_interval_order",
                severity=SEVERITY_ERROR,
                message=f"Intervals in tier '{tier.name}' are out of order.",
            )
        )
        tier.intervals = sorted_intervals
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier.name,
                interval_index=None,
                interval=None,
                original_label="",
                normalized_label="",
                issue_code="intervals_sorted",
                severity=SEVERITY_INFO,
                message=f"Sorted intervals in tier '{tier.name}'.",
                auto_corrected=True,
                correction_type="interval_sort",
            )
        )

    previous_interval: Interval | None = None
    for interval_index, interval in enumerate(tier.intervals, start=1):
        issues.extend(_validate_interval_bounds(path, tier.name, interval_index, interval))

        normalized_label = collapse_whitespace(interval.text)
        if normalize_labels and normalized_label != interval.text:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=interval_index,
                    interval=interval,
                    original_label=interval.text,
                    normalized_label=normalized_label,
                    issue_code="label_normalized",
                    severity=SEVERITY_INFO,
                    message=f"Normalized whitespace in {tier_kind} label.",
                    auto_corrected=True,
                    correction_type="whitespace",
                )
            )
            interval.text = normalized_label

        if not interval.text:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=interval_index,
                    interval=interval,
                    original_label="",
                    normalized_label="",
                    issue_code="empty_label",
                    severity=SEVERITY_WARNING,
                    message=f"{tier_kind} interval has an empty label.",
                )
            )

        if previous_interval is not None:
            issues.extend(
                _check_consecutive_overlap(
                    path=path,
                    tier_name=tier.name,
                    previous_interval=previous_interval,
                    current_interval=interval,
                    current_index=interval_index,
                    config=config,
                )
            )
        previous_interval = interval

    return issues


def _validate_action_tier(path: Path, tier: Tier, config: ValidationConfig) -> list[ValidationIssue]:
    """Validate an action tier."""

    issues = _validate_interval_tier(path, tier, config, tier_kind="action", normalize_labels=False)
    for interval_index, interval in enumerate(tier.intervals, start=1):
        original_label = interval.text
        normalized_candidate = collapse_whitespace(original_label)
        label_result = normalize_action_label(interval.text)
        if label_result.changed:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=interval_index,
                    interval=interval,
                    original_label=original_label,
                    normalized_label=label_result.value,
                    issue_code="label_normalized",
                    severity=SEVERITY_INFO,
                    message=f"Normalized action label to '{label_result.value}'.",
                    auto_corrected=True,
                    correction_type=label_result.correction_type or "label_normalized",
                )
            )
            interval.text = label_result.value

        if not original_label.strip():
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=interval_index,
                    interval=interval,
                    original_label=original_label,
                    normalized_label=normalized_candidate,
                    issue_code="empty_label",
                    severity=SEVERITY_ERROR,
                    message="Action interval label is empty.",
                )
            )
            continue

        if interval.text not in ALLOWED_ACTION_LABELS:
            issues.append(
                _make_issue(
                    path=path,
                    tier_name=tier.name,
                    interval_index=interval_index,
                    interval=interval,
                    original_label=original_label,
                    normalized_label=normalized_candidate if not label_result.changed else label_result.value,
                    issue_code="invalid_label",
                    severity=SEVERITY_ERROR,
                    message=f"Invalid action label '{original_label}'.",
                )
            )

    if not config.allow_action_overlaps:
        for interval_index in range(1, len(tier.intervals)):
            previous_interval = tier.intervals[interval_index - 1]
            current_interval = tier.intervals[interval_index]
            if current_interval.xmin < previous_interval.xmax - DEFAULT_FLOAT_EPSILON:
                issues.append(
                    _make_issue(
                        path=path,
                        tier_name=tier.name,
                        interval_index=interval_index + 1,
                        interval=current_interval,
                        original_label=current_interval.text,
                        normalized_label=current_interval.text,
                        issue_code="overlapping_intervals",
                        severity=SEVERITY_ERROR,
                        message="Action intervals overlap within the same tier.",
                    )
                )

    return issues


def _validate_cross_tier_relationships(
    path: Path,
    matched_tiers: dict[str, list[Tier]],
    config: ValidationConfig,
) -> list[ValidationIssue]:
    """Validate same-speaker action/IPU/palign relationships."""

    del config
    issues: list[ValidationIssue] = []

    for speaker in ("A", "B"):
        action_tier_name = f"action {speaker}"
        ipu_tier_name = f"ipu-{speaker}"
        palign_tier_name = f"palign-{speaker}"

        action_tier = _first_unique_tier(matched_tiers[action_tier_name])
        ipu_tier = _first_unique_tier(matched_tiers[ipu_tier_name])
        palign_tier = _first_unique_tier(matched_tiers[palign_tier_name])
        if action_tier is None:
            continue

        for interval_index, action_interval in enumerate(action_tier.intervals, start=1):
            if ipu_tier is not None:
                overlapping_ipus = _collect_overlaps(action_interval, ipu_tier.intervals)
                if not overlapping_ipus:
                    issues.append(
                        _make_issue(
                            path=path,
                            tier_name=action_tier.name,
                            interval_index=interval_index,
                            interval=action_interval,
                            original_label=action_interval.text,
                            normalized_label=action_interval.text,
                            issue_code="action_without_ipu_overlap",
                            severity=SEVERITY_WARNING,
                            message=f"Action interval does not overlap any {ipu_tier_name} interval.",
                        )
                    )
                elif not any(_contains(interval, action_interval) for interval in overlapping_ipus):
                    issues.append(
                        _make_issue(
                            path=path,
                            tier_name=action_tier.name,
                            interval_index=interval_index,
                            interval=action_interval,
                            original_label=action_interval.text,
                            normalized_label=action_interval.text,
                            issue_code="action_not_contained_in_ipu",
                            severity=SEVERITY_WARNING,
                            message=f"Action interval is not contained within a single {ipu_tier_name} window.",
                        )
                    )

            if palign_tier is not None and not _collect_overlaps(action_interval, palign_tier.intervals):
                issues.append(
                    _make_issue(
                        path=path,
                        tier_name=action_tier.name,
                        interval_index=interval_index,
                        interval=action_interval,
                        original_label=action_interval.text,
                        normalized_label=action_interval.text,
                        issue_code="action_without_palign_overlap",
                        severity=SEVERITY_WARNING,
                        message=f"Action interval does not overlap any {palign_tier_name} interval.",
                    )
                )

    return issues


def _validate_interval_bounds(
    path: Path,
    tier_name: str,
    interval_index: int,
    interval: Interval,
) -> list[ValidationIssue]:
    """Validate bounds for a single interval."""

    issues: list[ValidationIssue] = []
    if not isfinite(interval.xmin) or not isfinite(interval.xmax):
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier_name,
                interval_index=interval_index,
                interval=interval,
                original_label=interval.text,
                normalized_label=interval.text,
                issue_code="invalid_interval_bounds",
                severity=SEVERITY_ERROR,
                message="Interval bounds must be finite numeric values.",
            )
        )
        return issues

    if interval.xmin < 0.0 or interval.xmax < 0.0:
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier_name,
                interval_index=interval_index,
                interval=interval,
                original_label=interval.text,
                normalized_label=interval.text,
                issue_code="negative_time",
                severity=SEVERITY_ERROR,
                message="Interval bounds must not be negative.",
            )
        )

    if interval.xmax < interval.xmin - DEFAULT_FLOAT_EPSILON:
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier_name,
                interval_index=interval_index,
                interval=interval,
                original_label=interval.text,
                normalized_label=interval.text,
                issue_code="invalid_interval_bounds",
                severity=SEVERITY_ERROR,
                message="Interval xmin exceeds xmax.",
            )
        )
    elif abs(interval.xmax - interval.xmin) <= DEFAULT_FLOAT_EPSILON:
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier_name,
                interval_index=interval_index,
                interval=interval,
                original_label=interval.text,
                normalized_label=interval.text,
                issue_code="zero_duration_interval",
                severity=SEVERITY_ERROR,
                message="Interval has zero duration.",
            )
        )

    return issues


def _check_consecutive_overlap(
    *,
    path: Path,
    tier_name: str,
    previous_interval: Interval,
    current_interval: Interval,
    current_index: int,
    config: ValidationConfig,
) -> list[ValidationIssue]:
    """Check consecutive intervals for overlaps and tiny snap repairs."""

    issues: list[ValidationIssue] = []
    overlap = previous_interval.xmax - current_interval.xmin
    if overlap <= DEFAULT_FLOAT_EPSILON:
        return issues

    if 0.0 < overlap <= config.snap_tolerance_s:
        original_xmin = current_interval.xmin
        current_interval.xmin = previous_interval.xmax
        issues.append(
            _make_issue(
                path=path,
                tier_name=tier_name,
                interval_index=current_index,
                interval=current_interval,
                original_label=current_interval.text,
                normalized_label=current_interval.text,
                issue_code="tiny_overlap_snapped",
                severity=SEVERITY_INFO,
                message=f"Snapped interval start from {original_xmin:.6f} to {current_interval.xmin:.6f}.",
                auto_corrected=True,
                correction_type="snap_tolerance",
            )
        )
        return issues

    issues.append(
        _make_issue(
            path=path,
            tier_name=tier_name,
            interval_index=current_index,
            interval=current_interval,
            original_label=current_interval.text,
            normalized_label=current_interval.text,
            issue_code="overlapping_intervals",
            severity=SEVERITY_ERROR,
            message="Consecutive intervals overlap.",
        )
    )
    return issues


def _collect_overlaps(target_interval: Interval, candidate_intervals: list[Interval]) -> list[Interval]:
    """Collect all intervals that overlap a target interval."""

    return [
        candidate
        for candidate in candidate_intervals
        if candidate.xmin < target_interval.xmax - DEFAULT_FLOAT_EPSILON
        and candidate.xmax > target_interval.xmin + DEFAULT_FLOAT_EPSILON
    ]


def _contains(container: Interval, target: Interval) -> bool:
    """Return whether one interval fully contains another."""

    return (
        container.xmin <= target.xmin + DEFAULT_FLOAT_EPSILON
        and container.xmax >= target.xmax - DEFAULT_FLOAT_EPSILON
    )


def _first_unique_tier(tiers: list[Tier]) -> Tier | None:
    """Return the unique tier if present exactly once."""

    if len(tiers) != 1:
        return None
    return tiers[0]


def _make_issue(
    *,
    path: Path,
    tier_name: str,
    interval_index: int | None,
    interval: Interval | None,
    original_label: str,
    normalized_label: str,
    issue_code: str,
    severity: str,
    message: str,
    auto_corrected: bool = False,
    correction_type: str = "",
) -> ValidationIssue:
    """Create a validation issue with consistent field population."""

    return ValidationIssue(
        file_path=str(path),
        tier_name=tier_name,
        interval_index=interval_index,
        xmin=None if interval is None else interval.xmin,
        xmax=None if interval is None else interval.xmax,
        original_label=original_label,
        normalized_label=normalized_label,
        issue_code=issue_code,
        severity=severity,
        message=message,
        auto_corrected=auto_corrected,
        correction_type=correction_type,
    )
