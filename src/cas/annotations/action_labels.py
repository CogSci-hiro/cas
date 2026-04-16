"""Helpers for mechanically classifying normalized action labels.

The helpers in this module intentionally only reflect what is encoded in the
annotation inventory itself. They do not infer richer conversational structure
or reinterpret labels beyond safe normalization and prefix-based membership.
"""

from __future__ import annotations

from typing import Literal

from cas.annotations.autocorrect import normalize_action_label, normalize_tier_name
from cas.annotations.constants import ACTION_TIERS

_FPP_PREFIX = "FPP_"
_SPP_PREFIX = "SPP_"


def normalize_action_label_for_classification(label: str) -> str:
    """Return the normalized action label used for classification.

    Parameters
    ----------
    label
        Raw action label text from a TextGrid interval.

    Returns
    -------
    str
        Safely normalized action label text.
    """

    return normalize_action_label(label).value


def is_fpp_label(label: str) -> bool:
    """Return whether a normalized action label is an FPP label.

    Parameters
    ----------
    label
        Action label text. The label is normalized before classification.

    Returns
    -------
    bool
        True when the normalized label starts with ``FPP_``.
    """

    return normalize_action_label_for_classification(label).startswith(_FPP_PREFIX)


def is_spp_label(label: str) -> bool:
    """Return whether a normalized action label is an SPP label.

    Parameters
    ----------
    label
        Action label text. The label is normalized before classification.

    Returns
    -------
    bool
        True when the normalized label starts with ``SPP_``.
    """

    return normalize_action_label_for_classification(label).startswith(_SPP_PREFIX)


def infer_speaker_from_action_tier(tier_name: str) -> Literal["A", "B"]:
    """Infer the speaker identity from an action tier name.

    Parameters
    ----------
    tier_name
        Action tier name from the TextGrid.

    Returns
    -------
    Literal["A", "B"]
        Speaker corresponding to the action tier.

    Raises
    ------
    ValueError
        If the tier name is not one of the expected action tiers.
    """

    normalized_tier_name = normalize_tier_name(tier_name).value
    if normalized_tier_name not in ACTION_TIERS:
        raise ValueError(f"Unsupported action tier: {tier_name}")
    return "A" if normalized_tier_name == "action A" else "B"
