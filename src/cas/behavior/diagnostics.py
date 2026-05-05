"""Diagnostics for the behavioral hazard pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def event_rate_summary(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.groupby("anchor_type", sort=False)["event"]
    return grouped.agg(n_bins="size", n_events="sum", event_rate="mean").reset_index()


def bins_by_subject(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.groupby(["anchor_type", "subject"], sort=False)["event"]
    return grouped.agg(n_bins="size", n_events="sum", event_rate="mean").reset_index()


def collinearity_summary(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = table.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.empty:
        return pd.DataFrame(columns=["dataset", "term", "vif", "notes"])
    design = numeric.copy()
    design.insert(0, "intercept", 1.0)
    rows: list[dict[str, object]] = []
    for index, term in enumerate(design.columns):
        if term == "intercept":
            continue
        try:
            vif = float(variance_inflation_factor(design.to_numpy(dtype=float), index))
            note = ""
        except Exception as exc:
            vif = np.nan
            note = f"VIF unavailable: {exc}"
        rows.append({"dataset": "pooled_fpp_spp", "term": term, "vif": vif, "notes": note})
    return pd.DataFrame(rows)


def convergence_warnings(models: list[object], *, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fitted in models:
        if getattr(fitted, "notes", ()):
            rows.append({"dataset": dataset, "model_id": fitted.model_id, "warning": " | ".join(fitted.notes)})
        for message in fitted.warnings:
            rows.append({"dataset": dataset, "model_id": fitted.model_id, "warning": str(message)})
        if not fitted.converged:
            rows.append({"dataset": dataset, "model_id": fitted.model_id, "warning": "Model did not converge."})
    return pd.DataFrame(rows)


def lag_sensitivity_rows(table: pd.DataFrame, candidate_lags_ms: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    from cas.behavior.models import fit_registered_model

    for lag_ms in candidate_lags_ms:
        fitted = fit_registered_model(table, model_id="M_3", lag_ms=lag_ms)
        conf = fitted.result.conf_int() if hasattr(fitted.result, "conf_int") else None
        for term in [f"z_information_rate_lag_{lag_ms}", f"z_prop_expected_cum_info_lag_{lag_ms}"]:
            if term not in fitted.result.params.index:
                continue
            rows.append(
                {
                    "candidate_lag_ms": int(lag_ms),
                    "term": term,
                    "predictor": "information_rate" if "information_rate" in term else "prop_expected_cum_info",
                    "estimate": float(fitted.result.params[term]),
                    "ci_low": float(conf.loc[term, 0]) if conf is not None and term in conf.index else np.nan,
                    "ci_high": float(conf.loc[term, 1]) if conf is not None and term in conf.index else np.nan,
                    "backend": getattr(fitted, "backend", ""),
                }
            )
    return pd.DataFrame(rows)
