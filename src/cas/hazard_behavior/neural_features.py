"""Causal low-level neural feature aggregation for hazard risk sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class NeuralRisksetAugmentationResult:
    """Risk-set table with neural features plus QC metadata."""

    riskset_table: pd.DataFrame
    neural_feature_columns: tuple[str, ...]
    qc: dict[str, object]


def add_lowlevel_neural_features_to_riskset(
    riskset_table: pd.DataFrame,
    neural_table: pd.DataFrame,
    *,
    neural_feature_columns: list[str],
    neural_window_s: float,
    neural_guard_s: float,
) -> NeuralRisksetAugmentationResult:
    """Add causal low-level neural window means to the risk set."""

    required_riskset_columns = {"dyad_id", "run", "participant_speaker", "bin_end"}
    missing_riskset_columns = sorted(required_riskset_columns - set(riskset_table.columns))
    if missing_riskset_columns:
        raise ValueError(
            "Risk-set table is missing required columns for neural augmentation: "
            + ", ".join(missing_riskset_columns)
        )
    required_neural_columns = {"dyad_id", "run", "speaker", "time", *neural_feature_columns}
    missing_neural_columns = sorted(required_neural_columns - set(neural_table.columns))
    if missing_neural_columns:
        raise ValueError(
            "Neural feature table is missing required columns: "
            + ", ".join(missing_neural_columns)
        )

    table = riskset_table.copy()
    renamed_feature_columns = [f"neural_{column}" for column in neural_feature_columns]
    for renamed_column in renamed_feature_columns:
        table[renamed_column] = np.nan

    grouped_neural = {
        key: group.sort_values("time", kind="mergesort").reset_index(drop=True)
        for key, group in neural_table.groupby(["dyad_id", "run", "speaker"], sort=False)
    }
    for row_index, row in table.iterrows():
        key = (str(row["dyad_id"]), str(row["run"]), str(row["participant_speaker"]))
        speaker_neural = grouped_neural.get(key)
        if speaker_neural is None or speaker_neural.empty:
            continue
        aggregated = compute_neural_window_features(
            speaker_neural,
            bin_end=float(row["bin_end"]),
            neural_feature_columns=neural_feature_columns,
            neural_window_s=neural_window_s,
            neural_guard_s=neural_guard_s,
        )
        for original_column, renamed_column in zip(neural_feature_columns, renamed_feature_columns, strict=True):
            table.at[row_index, renamed_column] = aggregated.get(original_column, np.nan)

    qc = summarise_neural_feature_qc(
        riskset_with_neural=table,
        neural_feature_columns=tuple(renamed_feature_columns),
        neural_window_s=neural_window_s,
        neural_guard_s=neural_guard_s,
    )
    return NeuralRisksetAugmentationResult(
        riskset_table=table,
        neural_feature_columns=tuple(renamed_feature_columns),
        qc=qc,
    )


def compute_neural_window_features(
    neural_table: pd.DataFrame,
    *,
    bin_end: float,
    neural_feature_columns: list[str],
    neural_window_s: float,
    neural_guard_s: float,
) -> dict[str, float]:
    """Compute mean neural features in the causal guarded window for one hazard bin."""

    window_start = float(bin_end) - float(neural_window_s)
    window_end = float(bin_end) - float(neural_guard_s)
    window = neural_table.loc[
        (pd.to_numeric(neural_table["time"], errors="coerce") >= window_start)
        & (pd.to_numeric(neural_table["time"], errors="coerce") <= window_end)
    ]
    if window.empty:
        return {column: np.nan for column in neural_feature_columns}
    return {
        column: float(pd.to_numeric(window[column], errors="coerce").mean())
        for column in neural_feature_columns
    }


def summarise_neural_feature_qc(
    *,
    riskset_with_neural: pd.DataFrame,
    neural_feature_columns: tuple[str, ...],
    neural_window_s: float,
    neural_guard_s: float,
) -> dict[str, object]:
    """Summarize neural missingness and feature-family counts."""

    missingness_by_feature: dict[str, float] = {}
    family_counts = {"amplitude": 0, "alpha": 0, "beta": 0}
    for column in neural_feature_columns:
        missingness_by_feature[column] = float(riskset_with_neural[column].isna().mean())
        if column.startswith("neural_amp_"):
            family_counts["amplitude"] += 1
        elif column.startswith("neural_alpha_"):
            family_counts["alpha"] += 1
        elif column.startswith("neural_beta_"):
            family_counts["beta"] += 1
    any_neural_mask = riskset_with_neural.loc[:, list(neural_feature_columns)].notna().any(axis=1)
    return {
        "n_rows_total": int(len(riskset_with_neural)),
        "n_rows_with_any_neural": int(any_neural_mask.sum()),
        "n_rows_missing_all_neural": int((~any_neural_mask).sum()),
        "proportion_missing_all_neural": float((~any_neural_mask).mean()) if len(riskset_with_neural) else 0.0,
        "missingness_by_feature": missingness_by_feature,
        "n_neural_feature_columns": int(len(neural_feature_columns)),
        "n_amplitude_features": int(family_counts["amplitude"]),
        "n_alpha_features": int(family_counts["alpha"]),
        "n_beta_features": int(family_counts["beta"]),
        "neural_window_s": float(neural_window_s),
        "neural_guard_s": float(neural_guard_s),
    }
