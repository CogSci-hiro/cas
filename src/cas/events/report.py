"""Summary helpers for event extraction outputs."""

from __future__ import annotations

from cas.events.models import ExtractionResult


def summarize_extraction(result: ExtractionResult) -> dict[str, int]:
    """Compute aggregate extraction summary counts."""

    return {
        "files_processed": result.files_processed,
        "fpp_count": result.fpp_count,
        "paired_events": len(result.events),
        "unpaired_fpp": len(result.unpaired_fpp),
        "unused_spp": len(result.unused_spp),
        "issues": len(result.issues),
    }
