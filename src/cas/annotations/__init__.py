"""Validation utilities for conversational annotation TextGrid files.

The validators in this package only check mechanics that can be inferred
reliably from tier names, labels, interval timings, and same-speaker tier
relationships. They do not attempt to infer interactional semantics beyond the
explicit label inventory and timing-based overlap checks.
"""

from cas.annotations.autocorrect import normalize_action_label, normalize_tier_name
from cas.annotations.models import FileValidationResult, ValidationConfig, ValidationIssue
from cas.annotations.validation import validate_textgrid_file, validate_textgrids

__all__ = [
    "FileValidationResult",
    "ValidationConfig",
    "ValidationIssue",
    "normalize_action_label",
    "normalize_tier_name",
    "validate_textgrid_file",
    "validate_textgrids",
]
