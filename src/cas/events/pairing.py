"""Deterministic FPP-SPP pairing strategies.

The current operational definition is intentionally simple: for each FPP, take
the FPP offset, search for eligible SPP onsets within a symmetric configured
window around that offset, and pair the SPP whose onset is closest to the FPP
offset. Remaining ties are resolved deterministically.
"""

from __future__ import annotations

from cas.events.models import ActionInterval, ExtractionConfig, PairingCandidate


def build_pairing_candidates(
    fpp: ActionInterval,
    spp_intervals: list[ActionInterval],
    config: ExtractionConfig,
    *,
    used_spp_keys: set[tuple[str, int]],
) -> tuple[list[PairingCandidate], bool]:
    """Build valid SPP candidates for an FPP interval.

    Parameters
    ----------
    fpp
        FPP interval to pair.
    spp_intervals
        Candidate SPP intervals from action tiers.
    config
        Extraction configuration controlling window size and speaker rules.
    used_spp_keys
        Already paired SPP intervals, represented by tier and interval index.

    Returns
    -------
    tuple[list[PairingCandidate], bool]
        Valid pairing candidates and whether any eligible candidate was rejected
        only because one-to-one reuse prevention was active.
    """

    candidates: list[PairingCandidate] = []
    reused_spp_prevented = False
    max_offset_distance = config.pairing_margin_s

    for spp in spp_intervals:
        if not config.allow_same_speaker_spp and spp.speaker == fpp.speaker:
            continue

        latency = spp.onset - fpp.offset
        offset_distance = abs(latency)
        if offset_distance > max_offset_distance:
            continue

        spp_key = (spp.tier_name, spp.interval_index)
        if spp_key in used_spp_keys:
            reused_spp_prevented = True
            continue

        candidates.append(
            PairingCandidate(
                fpp=fpp,
                spp=spp,
                latency=latency,
                offset_distance=offset_distance,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.offset_distance,
            candidate.spp.onset,
            candidate.spp.offset,
            candidate.spp.interval_index,
        )
    )
    return candidates, reused_spp_prevented


def select_best_candidate(candidates: list[PairingCandidate]) -> PairingCandidate | None:
    """Select the deterministically best pairing candidate."""

    if not candidates:
        return None
    return candidates[0]
