"""Predictor engineering for the behavioral hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cas.behavior._legacy_support import progress_iterable
from cas.behavior.config import BehaviorHazardConfig


@dataclass(frozen=True, slots=True)
class ScalingSummary:
    table: pd.DataFrame


def _lag_steps(config: BehaviorHazardConfig, lag_ms: int) -> int:
    return int(round(int(lag_ms) / int(config.bin_size_ms)))


def add_candidate_lags(
    table: pd.DataFrame,
    *,
    config: BehaviorHazardConfig,
    verbose: bool = False,
) -> pd.DataFrame:
    out = table.copy().sort_values(["episode_id", "bin_start_s"], kind="mergesort")
    for lag_ms in progress_iterable(
        list(config.candidate_lags_ms),
        total=len(config.candidate_lags_ms),
        description="Candidate lags",
        enabled=verbose,
    ):
        steps = _lag_steps(config, lag_ms)
        out[f"information_rate_lag_{lag_ms}"] = (
            out.groupby("episode_id", sort=False)["information_rate"].shift(steps).fillna(0.0)
        )
        out[f"prop_expected_cum_info_lag_{lag_ms}"] = (
            out.groupby("episode_id", sort=False)["prop_expected_cum_info"].shift(steps).fillna(0.0)
        )
    return out


def _scaling_frame(table: pd.DataFrame, *, config: BehaviorHazardConfig) -> pd.DataFrame:
    scope = str((config.standardization or {}).get("scope", "pooled_fpp_spp"))
    if scope == "pooled_fpp_spp":
        return table
    if scope == "within_anchor":
        return table
    return table.loc[table["anchor_type"].astype(str) == "FPP"].copy()


def standardize_predictors(
    fpp: pd.DataFrame,
    spp: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    config: BehaviorHazardConfig,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fpp_with_lags = add_candidate_lags(fpp, config=config, verbose=verbose)
    spp_with_lags = add_candidate_lags(spp, config=config, verbose=verbose)
    pooled_with_lags = add_candidate_lags(pooled, config=config, verbose=verbose)
    scope_frame = _scaling_frame(pooled_with_lags, config=config)
    columns = [str(value) for value in list((config.standardization or {}).get("continuous_predictors") or [])]
    lag_columns = []
    for lag_ms in config.candidate_lags_ms:
        lag_columns.extend(
            [
                f"information_rate_lag_{lag_ms}",
                f"prop_expected_cum_info_lag_{lag_ms}",
            ]
        )
    all_columns = list(dict.fromkeys(columns + lag_columns))
    summary_rows: list[dict[str, float | str]] = []

    def _apply(table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        scope = str((config.standardization or {}).get("scope", "pooled_fpp_spp"))
        for column in progress_iterable(list(all_columns), total=len(all_columns), description="Standardizing", enabled=verbose):
            if scope == "within_anchor":
                out[f"z_{column}"] = np.nan
                for anchor_type, anchor_rows in out.groupby("anchor_type", sort=False):
                    anchor_mask = out["anchor_type"].astype(str) == str(anchor_type)
                    values = pd.to_numeric(anchor_rows[column], errors="coerce")
                    mean = float(values.mean())
                    sd = float(values.std(ddof=0))
                    if not np.isfinite(sd) or sd <= 0.0:
                        sd = 1.0
                    out.loc[anchor_mask, f"z_{column}"] = (pd.to_numeric(out.loc[anchor_mask, column], errors="coerce") - mean) / sd
            else:
                values = pd.to_numeric(scope_frame[column], errors="coerce")
                mean = float(values.mean())
                sd = float(values.std(ddof=0))
                if not np.isfinite(sd) or sd <= 0.0:
                    sd = 1.0
                out[f"z_{column}"] = (pd.to_numeric(out[column], errors="coerce") - mean) / sd
        if "z_time_from_partner_offset_s" in out.columns:
            out["z_time_from_partner_offset_s_squared"] = pd.to_numeric(out["z_time_from_partner_offset_s"], errors="coerce") ** 2
        return out

    for column in all_columns:
        scope = str((config.standardization or {}).get("scope", "pooled_fpp_spp"))
        if scope == "within_anchor":
            for anchor_type, anchor_rows in pooled_with_lags.groupby("anchor_type", sort=False):
                values = pd.to_numeric(anchor_rows[column], errors="coerce")
                mean = float(values.mean())
                sd = float(values.std(ddof=0))
                if not np.isfinite(sd) or sd <= 0.0:
                    sd = 1.0
                summary_rows.append({"column": column, "mean": mean, "sd": sd, "scope": f"within_anchor:{anchor_type}"})
        else:
            values = pd.to_numeric(scope_frame[column], errors="coerce")
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or sd <= 0.0:
                sd = 1.0
            summary_rows.append({"column": column, "mean": mean, "sd": sd, "scope": "pooled_fpp_spp"})

    return _apply(fpp_with_lags), _apply(spp_with_lags), _apply(pooled_with_lags), pd.DataFrame(summary_rows)
