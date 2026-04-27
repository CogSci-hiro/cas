from __future__ import annotations

import pandas as pd
import pytest

from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id


def test_canonical_id_creation() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "speaker": ["A", "B", "A", "B"],
        }
    )
    result = ensure_participant_speaker_id(table)
    assert result["participant_speaker_id"].tolist() == [
        "dyad-001_A",
        "dyad-001_B",
        "dyad-002_A",
        "dyad-002_B",
    ]
    assert result["participant_speaker_id"].nunique() == 4


def test_detect_a_b_collapse() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["A", "B", "A", "B"],
        }
    )
    with pytest.raises(ValueError, match="collapsed to plain A/B"):
        ensure_participant_speaker_id(table, overwrite=False)


def test_overwrite_invalid_id() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["A", "B", "A", "B"],
        }
    )
    result = ensure_participant_speaker_id(table, overwrite=True)
    assert result["participant_speaker_id"].tolist() == [
        "dyad-001_A",
        "dyad-001_B",
        "dyad-002_A",
        "dyad-002_B",
    ]


def test_preserve_valid_id() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["dyad-001_A", "dyad-001_B", "dyad-002_A", "dyad-002_B"],
        }
    )
    result = ensure_participant_speaker_id(table, overwrite=False)
    assert result["participant_speaker_id"].tolist() == table["participant_speaker_id"].tolist()


def test_qc_identity_validation_for_valid_data() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", "dyad-001", "dyad-002", "dyad-002"],
            "speaker": ["A", "B", "A", "B"],
            "participant_speaker_id": ["dyad-001_A", "dyad-001_B", "dyad-002_A", "dyad-002_B"],
        }
    )
    qc = validate_participant_speaker_id(table)
    assert qc["n_unique_dyad_id"] == 2
    assert qc["n_unique_speaker"] == 2
    assert qc["n_unique_dyad_speaker_pairs"] == 4
    assert qc["n_unique_participant_speaker_id"] == 4
    assert qc["participant_speaker_id_valid"] is True


def test_failure_on_missing_dyad_or_speaker() -> None:
    table = pd.DataFrame(
        {
            "dyad_id": ["dyad-001", None],
            "speaker": ["A", "B"],
        }
    )
    with pytest.raises(ValueError, match="non-missing values"):
        ensure_participant_speaker_id(table)
