"""Prediction helpers for the behavioral hazard pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cas.behavior.models import FittedBehaviorModel, prediction_summary_frame


def _median_row(table: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: float(pd.to_numeric(table[column], errors="coerce").median()) for column in columns}


def _conditional_median_curve(
    table: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_values: np.ndarray,
) -> np.ndarray:
    summary = (
        table.loc[:, [x_column, y_column]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .groupby(x_column, sort=True)[y_column]
        .median()
        .sort_index()
    )
    if summary.empty:
        return np.repeat(0.0, len(x_values))
    if len(summary) == 1:
        return np.repeat(float(summary.iloc[0]), len(x_values))
    x_support = summary.index.to_numpy(dtype=float)
    y_support = summary.to_numpy(dtype=float)
    return np.interp(x_values, x_support, y_support, left=y_support[0], right=y_support[-1])


def primary_effect_predictions(
    fpp_table: pd.DataFrame,
    pooled_table: pd.DataFrame,
    *,
    lag_ms: int,
    fitted_fpp: FittedBehaviorModel,
    fitted_pooled: FittedBehaviorModel,
) -> pd.DataFrame:
    base_columns = [
        "z_time_from_partner_onset_s",
        "z_time_from_partner_offset_s",
        "z_run",
        "z_time_within_run",
        "z_planned_response_duration",
        "z_planned_response_total_information",
    ]
    base = _median_row(fpp_table, base_columns)
    rate_raw = np.linspace(
        float(pd.to_numeric(fpp_table["information_rate"], errors="coerce").min()),
        float(pd.to_numeric(fpp_table["information_rate"], errors="coerce").max()),
        60,
    )
    rate_z = np.linspace(
        float(pd.to_numeric(fpp_table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").min()),
        float(pd.to_numeric(fpp_table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").max()),
        60,
    )
    prop_raw = np.linspace(
        float(pd.to_numeric(fpp_table["prop_expected_cum_info"], errors="coerce").min()),
        float(pd.to_numeric(fpp_table["prop_expected_cum_info"], errors="coerce").max()),
        60,
    )
    prop_z = np.linspace(
        float(pd.to_numeric(fpp_table[f"z_prop_expected_cum_info_lag_{lag_ms}"], errors="coerce").min()),
        float(pd.to_numeric(fpp_table[f"z_prop_expected_cum_info_lag_{lag_ms}"], errors="coerce").max()),
        60,
    )
    rows: list[pd.DataFrame] = []
    for raw, z_value in zip(rate_raw, rate_z):
        row = dict(base)
        row.update(
            {
                f"z_information_rate_lag_{lag_ms}": float(z_value),
                f"z_prop_expected_cum_info_lag_{lag_ms}": 0.0,
                "information_rate": float(raw),
                "prop_expected_cum_info": float(np.median(prop_raw)),
            }
        )
        frame = prediction_summary_frame(fitted_fpp, pd.DataFrame([row]))
        frame["figure"] = "fig02_primary_information_effects"
        frame["panel"] = "A"
        frame["anchor_type"] = "FPP"
        frame["predictor"] = "information_rate"
        frame["x_value_z"] = float(z_value)
        frame["x_value_original"] = float(raw)
        rows.append(frame)
    for raw, z_value in zip(prop_raw, prop_z):
        row = dict(base)
        row.update(
            {
                f"z_information_rate_lag_{lag_ms}": 0.0,
                f"z_prop_expected_cum_info_lag_{lag_ms}": float(z_value),
                "information_rate": float(np.median(rate_raw)),
                "prop_expected_cum_info": float(raw),
            }
        )
        frame = prediction_summary_frame(fitted_fpp, pd.DataFrame([row]))
        frame["figure"] = "fig02_primary_information_effects"
        frame["panel"] = "B"
        frame["anchor_type"] = "FPP"
        frame["predictor"] = "prop_expected_cum_info"
        frame["x_value_z"] = float(z_value)
        frame["x_value_original"] = float(raw)
        rows.append(frame)

    pooled_base = _median_row(
        pooled_table,
        base_columns + [f"z_prop_expected_cum_info_lag_{lag_ms}", f"z_information_rate_lag_{lag_ms}"],
    )
    for anchor in ["SPP", "FPP"]:
        for raw, z_value in zip(rate_raw, rate_z):
            row = dict(pooled_base)
            row.update(
                {
                    "anchor_type": anchor,
                    f"z_information_rate_lag_{lag_ms}": float(z_value),
                    "information_rate": float(raw),
                }
            )
            frame = prediction_summary_frame(fitted_pooled, pd.DataFrame([row]))
            frame["figure"] = "fig02_primary_information_effects"
            frame["panel"] = "C"
            frame["anchor_type"] = anchor
            frame["predictor"] = "information_rate"
            frame["x_value_z"] = float(z_value)
            frame["x_value_original"] = float(raw)
            rows.append(frame)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined.loc[:, ["figure", "panel", "anchor_type", "predictor", "x_value_z", "x_value_original", "predicted_hazard", "ci_low", "ci_high"]]


def timing_heatmap_predictions(
    table: pd.DataFrame,
    *,
    lag_ms: int,
    onset_model: FittedBehaviorModel,
    offset_model: FittedBehaviorModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate_raw = np.linspace(
        float(pd.to_numeric(table["information_rate"], errors="coerce").min()),
        float(pd.to_numeric(table["information_rate"], errors="coerce").max()),
        35,
    )
    rate_z = np.linspace(
        float(pd.to_numeric(table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").min()),
        float(pd.to_numeric(table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").max()),
        35,
    )
    onset_values = np.linspace(
        float(pd.to_numeric(table["time_from_partner_onset_s"], errors="coerce").min()),
        float(pd.to_numeric(table["time_from_partner_onset_s"], errors="coerce").max()),
        35,
    )
    offset_values = np.linspace(
        float(pd.to_numeric(table["time_from_partner_offset_s"], errors="coerce").min()),
        float(pd.to_numeric(table["time_from_partner_offset_s"], errors="coerce").max()),
        35,
    )
    onset_mean = float(pd.to_numeric(table["time_from_partner_onset_s"], errors="coerce").mean())
    onset_sd = float(pd.to_numeric(table["time_from_partner_onset_s"], errors="coerce").std(ddof=0)) or 1.0
    offset_mean = float(pd.to_numeric(table["time_from_partner_offset_s"], errors="coerce").mean())
    offset_sd = float(pd.to_numeric(table["time_from_partner_offset_s"], errors="coerce").std(ddof=0)) or 1.0
    conditioned_offsets = _conditional_median_curve(
        table,
        x_column="time_from_partner_onset_s",
        y_column="time_from_partner_offset_s",
        x_values=onset_values,
    )
    conditioned_onsets = _conditional_median_curve(
        table,
        x_column="time_from_partner_offset_s",
        y_column="time_from_partner_onset_s",
        x_values=offset_values,
    )
    medians = {
        "z_run": float(pd.to_numeric(table["z_run"], errors="coerce").median()),
        "z_time_within_run": float(pd.to_numeric(table["z_time_within_run"], errors="coerce").median()),
        "z_planned_response_duration": float(pd.to_numeric(table["z_planned_response_duration"], errors="coerce").median()),
        "z_planned_response_total_information": float(pd.to_numeric(table["z_planned_response_total_information"], errors="coerce").median()),
        f"z_prop_expected_cum_info_lag_{lag_ms}": float(pd.to_numeric(table[f"z_prop_expected_cum_info_lag_{lag_ms}"], errors="coerce").median()),
    }

    def _rows(x_values: np.ndarray, *, conditioned_values: np.ndarray, mode: str) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for x_value, conditioned_value in zip(x_values, conditioned_values):
            for raw, z_value in zip(rate_raw, rate_z):
                row = dict(medians)
                row.update(
                    {
                        "time_from_partner_onset_s": float(x_value if mode == "onset" else conditioned_value),
                        "time_from_partner_offset_s": float(x_value if mode == "offset" else conditioned_value),
                        f"z_information_rate_lag_{lag_ms}": float(z_value),
                        "information_rate": float(raw),
                    }
                )
                row["z_time_from_partner_onset_s"] = (row["time_from_partner_onset_s"] - onset_mean) / onset_sd
                row["z_time_from_partner_offset_s"] = (row["time_from_partner_offset_s"] - offset_mean) / offset_sd
                row["information_rate_z"] = float(z_value)
                row["information_rate_original"] = float(raw)
                rows.append(row)
        return pd.DataFrame(rows)

    onset_grid = prediction_summary_frame(onset_model, _rows(onset_values, conditioned_values=conditioned_offsets, mode="onset"))
    offset_grid = prediction_summary_frame(offset_model, _rows(offset_values, conditioned_values=conditioned_onsets, mode="offset"))
    onset_grid["panel"] = "A"
    onset_grid["timing_reference"] = "partner_onset"
    onset_grid["time_value_s"] = pd.to_numeric(onset_grid["time_from_partner_onset_s"], errors="coerce")
    offset_grid["panel"] = "B"
    offset_grid["timing_reference"] = "partner_offset"
    offset_grid["time_value_s"] = pd.to_numeric(offset_grid["time_from_partner_offset_s"], errors="coerce")
    onset_out = onset_grid.loc[:, ["panel", "timing_reference", "time_value_s", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high"]]
    offset_out = offset_grid.loc[:, ["panel", "timing_reference", "time_value_s", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high"]]
    return onset_out, offset_out


def three_way_predictions(
    pooled_table: pd.DataFrame,
    *,
    lag_ms: int,
    fitted_three_way: FittedBehaviorModel,
) -> pd.DataFrame:
    onset_values = np.linspace(
        float(pd.to_numeric(pooled_table["time_from_partner_onset_s"], errors="coerce").min()),
        float(pd.to_numeric(pooled_table["time_from_partner_onset_s"], errors="coerce").max()),
        30,
    )
    rate_raw = np.linspace(
        float(pd.to_numeric(pooled_table["information_rate"], errors="coerce").min()),
        float(pd.to_numeric(pooled_table["information_rate"], errors="coerce").max()),
        30,
    )
    rate_z = np.linspace(
        float(pd.to_numeric(pooled_table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").min()),
        float(pd.to_numeric(pooled_table[f"z_information_rate_lag_{lag_ms}"], errors="coerce").max()),
        30,
    )
    medians = {
        "z_time_from_partner_offset_s": float(pd.to_numeric(pooled_table["z_time_from_partner_offset_s"], errors="coerce").median()),
        "z_run": float(pd.to_numeric(pooled_table["z_run"], errors="coerce").median()),
        "z_time_within_run": float(pd.to_numeric(pooled_table["z_time_within_run"], errors="coerce").median()),
        "z_planned_response_duration": float(pd.to_numeric(pooled_table["z_planned_response_duration"], errors="coerce").median()),
        "z_planned_response_total_information": float(pd.to_numeric(pooled_table["z_planned_response_total_information"], errors="coerce").median()),
        f"z_prop_expected_cum_info_lag_{lag_ms}": float(pd.to_numeric(pooled_table[f"z_prop_expected_cum_info_lag_{lag_ms}"], errors="coerce").median()),
    }
    onset_mean = float(pd.to_numeric(pooled_table["time_from_partner_onset_s"], errors="coerce").mean())
    onset_sd = float(pd.to_numeric(pooled_table["time_from_partner_onset_s"], errors="coerce").std(ddof=0)) or 1.0
    rows: list[dict[str, object]] = []
    for anchor_type in ["FPP", "SPP"]:
        for time_value_s in onset_values:
            for info_raw, info_z in zip(rate_raw, rate_z):
                row = dict(medians)
                row.update(
                    {
                        "anchor_type": anchor_type,
                        "time_from_partner_onset_s": float(time_value_s),
                        "z_time_from_partner_onset_s": float((time_value_s - onset_mean) / onset_sd),
                        f"z_information_rate_lag_{lag_ms}": float(info_z),
                        "information_rate_z": float(info_z),
                        "information_rate_original": float(info_raw),
                    }
                )
                rows.append(row)
    predicted = prediction_summary_frame(fitted_three_way, pd.DataFrame(rows))
    predicted["timing_reference"] = "partner_onset"
    predicted["time_value_s"] = pd.to_numeric(predicted["time_from_partner_onset_s"], errors="coerce")
    return predicted.loc[:, ["anchor_type", "timing_reference", "time_value_s", "information_rate_z", "information_rate_original", "predicted_hazard", "ci_low", "ci_high"]]
