from __future__ import annotations

import pandas as pd
import pytest

from cas.hazard_behavior.schema import normalize_events_schema, normalize_surprisal_schema


def test_events_schema_normalization_handles_aliases() -> None:
    events_table = pd.DataFrame(
        {
            "recording_id": ["dyad-001"],
            "run": ["1"],
            "speaker_fpp": ["B"],
            "fpp_onset": [1.2],
            "fpp_offset": [1.5],
            "fpp_type": ["FPP_RFC_TAG"],
            "speaker_spp": ["A"],
            "spp_onset": [0.0],
            "spp_offset": [0.8],
            "spp_type": ["SPP_CONF"],
        }
    )

    result = normalize_events_schema(events_table)

    assert "dyad_id" in result.table.columns
    assert "participant_speaker" in result.table.columns
    assert "partner_speaker" in result.table.columns
    assert result.table.loc[0, "dyad_id"] == "dyad-001"


def test_surprisal_schema_validation_catches_missing_required_columns() -> None:
    surprisal_table = pd.DataFrame({"dyad_id": ["dyad-001"], "run": ["1"], "speaker": ["A"]})

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_surprisal_schema(surprisal_table)
