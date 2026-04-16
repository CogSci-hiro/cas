"""Dyad-aware TRF predictor resolution utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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


def get_partner_id(subject_id: str, dyad_table: pd.DataFrame) -> str:
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
    matches = dyad_table.loc[dyad_table["subject_id"] == subject_id, "partner_id"]
    if matches.empty:
        raise ValueError(f"Subject '{subject_id}' is missing from the dyad table.")
    unique_matches = matches.drop_duplicates()
    if len(unique_matches) != 1:
        raise ValueError(f"Subject '{subject_id}' maps to multiple partners in the dyad table.")
    return str(unique_matches.iloc[0])


def resolve_feature_subject_id(
    target_subject_id: str, role: str, dyad_table: pd.DataFrame
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
    return Path("derivatives") / "features" / base_feature / subject_id / f"run-{run}.npy"


def resolve_predictor_paths(
    subject_id: str, run: int, predictors: list[dict], dyad_table: pd.DataFrame
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
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Resolved feature file for predictor '{predictor_name}' does not exist: "
                f"{feature_path}"
            )

        resolved_paths[str(predictor_name)] = feature_path

    return resolved_paths
