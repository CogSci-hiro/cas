"""Compatibility layer for safe auto-corrections.

The normalization helpers in this module are intentionally conservative. They
only rewrite formatting variants that map unambiguously to a known tier or
action label inventory item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cas.annotations.constants import ALLOWED_ACTION_LABELS, EXPECTED_TIERS

_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[-_\s]+")
_ACTIONS_WORD_RE = re.compile(r"\bactions\b")


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Result of a safe normalization attempt."""

    value: str
    changed: bool
    correction_type: str


def collapse_whitespace(value: str) -> str:
    """Trim and collapse repeated whitespace in a label."""

    return _WHITESPACE_RE.sub(" ", value.strip())


def normalize_tier_name(name: str) -> NormalizationResult:
    """Normalize a tier name when the mapping is trivial and unique."""

    compact_name = collapse_whitespace(name)
    if not compact_name:
        return NormalizationResult(value=name, changed=False, correction_type="")

    canonical_input = _canonicalize_tier_name_key(compact_name)
    matches = [
        tier_name
        for tier_name in EXPECTED_TIERS
        if _canonicalize_tier_name_key(tier_name) == canonical_input
    ]
    if len(matches) != 1:
        return NormalizationResult(value=name, changed=False, correction_type="")

    normalized_name = matches[0]
    return NormalizationResult(
        value=normalized_name,
        changed=normalized_name != name,
        correction_type="tier_name_normalized" if normalized_name != name else "",
    )


def _canonicalize_tier_name_key(name: str) -> str:
    """Return a conservative comparison key for tier names."""

    collapsed = _SEPARATOR_RE.sub(" ", name).strip().lower()
    return _ACTIONS_WORD_RE.sub("action", collapsed)


def normalize_action_label(label: str) -> NormalizationResult:
    """Normalize an action label when the mapping is trivial and unique."""

    compact_label = collapse_whitespace(label)
    if not compact_label:
        return NormalizationResult(value=compact_label, changed=False, correction_type="")

    canonical_input = _SEPARATOR_RE.sub("_", compact_label).strip("_").upper()
    matches = [
        action_label
        for action_label in ALLOWED_ACTION_LABELS
        if _SEPARATOR_RE.sub("_", action_label).strip("_").upper() == canonical_input
    ]
    if len(matches) != 1:
        return NormalizationResult(value=compact_label, changed=False, correction_type="")

    normalized_label = matches[0]
    return NormalizationResult(
        value=normalized_label,
        changed=normalized_label != label,
        correction_type="label_normalized" if normalized_label != label else "",
    )
