"""Dyad-aware TRF predictor resolution utilities."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def load_dyad_table(dyads_csv: str | Path) -> pd.DataFrame:
    """Load a dyad mapping table from CSV.

    Parameters
    ----------
    dyads_csv
        Path to a CSV file with columns ``dyad_id``, ``subject_id``,
        and ``partner_id``.

    Returns
    -------
    pandas.DataFrame
        Loaded dyad table.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(dyads_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dyad table not found: {csv_path}")

    dyad_table = pd.read_csv(csv_path)
    required_columns = {"dyad_id", "subject_id", "partner_id"}
    missing_columns = required_columns.difference(dyad_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dyad table is missing required columns: {missing_str}")

    return dyad_table.loc[:, ["dyad_id", "subject_id", "partner_id"]].copy()


def canonical_partner_id(subject_id: str) -> str:
    """Infer the canonical partner subject id from the subject numbering scheme."""

    subject_number = int(str(subject_id).replace("sub-", "", 1))
    partner_number = subject_number + 1 if subject_number % 2 == 1 else subject_number - 1
    return f"sub-{partner_number:03d}"


def canonical_dyad_id(subject_id: str) -> str:
    """Infer the canonical dyad id from the subject numbering scheme."""

    subject_number = int(str(subject_id).replace("sub-", "", 1))
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def get_partner_id(subject_id: str, dyad_table: pd.DataFrame | None = None) -> str:
    """Get the partner subject id for a target subject.

    Parameters
    ----------
    subject_id
        Subject identifier, e.g. ``"sub-001"``.
    dyad_table
        Dyad mapping table returned by :func:`load_dyad_table`.

    Returns
    -------
    str
        Partner subject identifier.

    Raises
    ------
    ValueError
        If the subject is missing from the dyad table or maps ambiguously.
    """
    if dyad_table is None:
        return canonical_partner_id(subject_id)

    matches = dyad_table.loc[dyad_table["subject_id"] == subject_id, "partner_id"]
    if matches.empty:
        raise ValueError(f"Subject '{subject_id}' is missing from the dyad table.")
    unique_matches = matches.drop_duplicates()
    if len(unique_matches) != 1:
        raise ValueError(f"Subject '{subject_id}' maps to multiple partners in the dyad table.")
    return str(unique_matches.iloc[0])


def get_dyad_id(subject_id: str, dyad_table: pd.DataFrame | None = None) -> str:
    """Get the dyad identifier for a target subject."""

    if dyad_table is None:
        return canonical_dyad_id(subject_id)

    matches = dyad_table.loc[dyad_table["subject_id"] == subject_id, "dyad_id"]
    if matches.empty:
        raise ValueError(f"Subject '{subject_id}' is missing from the dyad table.")
    unique_matches = matches.drop_duplicates()
    if len(unique_matches) != 1:
        raise ValueError(f"Subject '{subject_id}' maps to multiple dyads in the dyad table.")
    return str(unique_matches.iloc[0])


def resolve_feature_subject_id(
    target_subject_id: str, role: str, dyad_table: pd.DataFrame | None = None
) -> str:
    """Resolve which subject provides a role-based predictor.

    Parameters
    ----------
    target_subject_id
        Subject being modeled.
    role
        Predictor role. Must be ``"self"`` or ``"other"``.
    dyad_table
        Dyad mapping table returned by :func:`load_dyad_table`.

    Returns
    -------
    str
        Subject id whose canonical feature file should be loaded.

    Raises
    ------
    ValueError
        If ``role`` is unsupported.
    """
    if role == "self":
        return target_subject_id
    if role == "other":
        return get_partner_id(target_subject_id, dyad_table)
    raise ValueError(f"Unsupported predictor role '{role}'. Expected one of: self, other.")


def subject_speaker_label(subject_id: str) -> str:
    """Resolve the canonical A/B speaker label for a subject id."""

    subject_number = int(str(subject_id).replace("sub-", "", 1))
    return "A" if subject_number % 2 == 1 else "B"


def other_speaker_label(subject_id: str) -> str:
    """Resolve the partner speaker label for a subject id."""

    return "B" if subject_speaker_label(subject_id) == "A" else "A"


def resolve_speaker_value(subject_id: str, speaker_role: str) -> str:
    """Resolve a speaker-role token into a canonical A/B label."""

    if speaker_role == "self":
        return subject_speaker_label(subject_id)
    if speaker_role == "other":
        return other_speaker_label(subject_id)
    raise ValueError(
        f"Unsupported speaker role '{speaker_role}'. Expected one of: self, other."
    )


def build_feature_path(base_feature: str, subject_id: str, run: int) -> Path:
    """Build the canonical path for a subject-level feature file.

    Parameters
    ----------
    base_feature
        Base feature name, e.g. ``"envelope"`` or ``"phoneme_onset"``.
    subject_id
        Subject identifier, e.g. ``"sub-001"``.
    run
        Run number.

    Returns
    -------
    pathlib.Path
        Canonical subject-level feature path.
    """
    run_token = f"{int(run)}"
    return (
        Path("derivatives")
        / "features"
        / base_feature
        / subject_id
        / f"{subject_id}_task-conversation_run-{run_token}_{base_feature}.npy"
    )


def load_events_table(events_csv: str | Path) -> pd.DataFrame:
    """Load the canonical events table used for event-locked TRF impulses."""

    csv_path = Path(events_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Events table not found: {csv_path}")

    table = pd.read_csv(csv_path)
    required_columns = {
        "recording_id",
        "run",
        "speaker_fpp",
        "speaker_spp",
        "fpp_onset",
        "fpp_offset",
        "spp_onset",
        "spp_offset",
    }
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Events table is missing required columns: {missing_str}")
    return table.copy()


def select_subject_run_events(
    *,
    events_table: pd.DataFrame,
    subject_id: str,
    run: int,
    dyad_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select one subject/run slice from the canonical events table."""

    dyad_id = get_dyad_id(subject_id, dyad_table)
    run_label = str(int(run))
    selected = events_table.loc[
        (events_table["recording_id"].astype(str) == dyad_id)
        & (events_table["run"].astype(str) == run_label)
    ].copy()
    return selected.reset_index(drop=True)


def build_impulse_predictor(
    *,
    n_samples: int,
    sfreq_hz: float,
    event_times_s: np.ndarray,
) -> np.ndarray:
    """Build a 1D impulse predictor from event times in seconds."""

    predictor = np.zeros(int(n_samples), dtype=float)
    if predictor.size == 0:
        return predictor

    finite_times = np.asarray(event_times_s, dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    if finite_times.size == 0:
        return predictor

    sample_indices = np.rint(finite_times * float(sfreq_hz)).astype(int)
    keep_mask = (sample_indices >= 0) & (sample_indices < predictor.shape[0])
    if not np.any(keep_mask):
        return predictor

    valid_indices = sample_indices[keep_mask]
    np.add.at(predictor, valid_indices, 1.0)
    return predictor


def _as_2d_array(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"{label} must be 1D or 2D, got shape {values.shape}.")
    return values


def _resample_run(array: np.ndarray, *, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(source_sfreq, target_sfreq):
        return np.asarray(array, dtype=float)
    source_hz = float(source_sfreq)
    target_hz = float(target_sfreq)
    scale = 1000
    up = int(round(target_hz * scale))
    down = int(round(source_hz * scale))
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor
    return np.asarray(
        resample_poly(np.asarray(array, dtype=float), up, down, axis=0),
        dtype=float,
    )


def prepare_trf_runs(
    *,
    eeg_runs: list[np.ndarray],
    predictor_runs: list[np.ndarray],
    eeg_sfreq: float,
    predictor_sfreq: float,
    target_sfreq: float,
    tmin_s: float,
    tmax_s: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Validate and optionally resample TRF inputs run by run."""

    del tmin_s, tmax_s

    if len(eeg_runs) != len(predictor_runs):
        raise ValueError(
            "Expected the same number of EEG and predictor runs, "
            f"got {len(eeg_runs)} and {len(predictor_runs)}."
        )

    prepared_X: list[np.ndarray] = []
    prepared_Y: list[np.ndarray] = []
    for run_index, (predictor_run, eeg_run) in enumerate(
        zip(predictor_runs, eeg_runs),
        start=1,
    ):
        X = _as_2d_array(predictor_run, label=f"predictor run {run_index}")
        Y = _as_2d_array(eeg_run, label=f"EEG run {run_index}")

        X = _resample_run(
            X,
            source_sfreq=float(predictor_sfreq),
            target_sfreq=float(target_sfreq),
        )
        Y = _resample_run(
            Y,
            source_sfreq=float(eeg_sfreq),
            target_sfreq=float(target_sfreq),
        )

        n_samples = min(X.shape[0], Y.shape[0])
        if n_samples <= 1:
            raise ValueError(f"Run {run_index} is too short after alignment: {n_samples} samples.")

        prepared_X.append(np.asarray(X[:n_samples], dtype=float))
        prepared_Y.append(np.asarray(Y[:n_samples], dtype=float))

    return prepared_X, prepared_Y


def resolve_predictor_paths(
    subject_id: str,
    run: int,
    predictors: list[dict],
    dyad_table: pd.DataFrame | None = None,
    *,
    feature_root: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve role-based predictor paths for one subject and run.

    Parameters
    ----------
    subject_id
        Subject being modeled.
    run
        Run number.
    predictors
        Predictor config dictionaries. Each must define ``name``,
        ``base_feature``, and ``role``.
    dyad_table
        Dyad mapping table returned by :func:`load_dyad_table`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from predictor name to resolved canonical feature path.

    Raises
    ------
    ValueError
        If predictor config is incomplete or invalid.
    FileNotFoundError
        If a resolved feature file does not exist.
    """
    resolved_paths: dict[str, Path] = {}

    for predictor in predictors:
        predictor_name = predictor.get("name")
        base_feature = predictor.get("base_feature")
        role = predictor.get("role")

        if not predictor_name:
            raise ValueError("Each predictor must define a non-empty 'name'.")
        if not base_feature:
            raise ValueError(f"Predictor '{predictor_name}' is missing 'base_feature'.")
        if not role:
            raise ValueError(f"Predictor '{predictor_name}' is missing 'role'.")

        feature_subject_id = resolve_feature_subject_id(subject_id, str(role), dyad_table)
        feature_path = build_feature_path(str(base_feature), feature_subject_id, run)
        candidate_paths = [feature_path]
        if feature_root is not None:
            candidate_paths.append(
                Path(feature_root)
                / str(base_feature)
                / feature_subject_id
                / feature_path.name
            )

        resolved_feature_path = next((path for path in candidate_paths if path.exists()), None)
        if resolved_feature_path is None:
            raise FileNotFoundError(
                f"Resolved feature file for predictor '{predictor_name}' does not exist: "
                f"{feature_path}"
            )

        resolved_paths[str(predictor_name)] = resolved_feature_path

    return resolved_paths
