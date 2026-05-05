"""Lag-selection helpers for the behavioral hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cas.behavior._legacy_support import progress_iterable
from cas.behavior.formulas import render_formula
from cas.behavior.models import fit_formula_model


@dataclass(frozen=True, slots=True)
class SelectedLag:
    selected_lag_ms: int
    scores: pd.DataFrame
    payload: dict[str, object]


def select_behavior_lag(
    fpp_table: pd.DataFrame,
    *,
    candidate_lags_ms: list[int],
    model_backend: str = "glm",
    lag_selection_criterion: str = "log_likelihood",
    verbose: bool = False,
) -> SelectedLag:
    baseline = fit_formula_model(
        fpp_table,
        model_id="M_0",
        formula=render_formula("M_0", lag_ms=0, backend=model_backend),
    )
    rows: list[dict[str, object]] = []
    for lag_ms in progress_iterable(
        list(candidate_lags_ms),
        total=len(candidate_lags_ms),
        description="Lag selection",
        enabled=verbose,
    ):
        fitted = fit_formula_model(
            fpp_table,
            model_id=f"M_3_lag_{lag_ms}",
            formula=render_formula("M_3", lag_ms=lag_ms, backend=model_backend),
        )
        rows.append(
            {
                "candidate_lag_ms": int(lag_ms),
                "log_likelihood_baseline": baseline.log_likelihood,
                "log_likelihood_m3": fitted.log_likelihood,
                "delta_log_likelihood": fitted.log_likelihood - baseline.log_likelihood,
                "baseline_log_likelihood_finite": bool(np.isfinite(baseline.log_likelihood)),
                "m3_log_likelihood_finite": bool(np.isfinite(fitted.log_likelihood)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["candidate_lag_ms"], kind="mergesort").reset_index(drop=True)
    criterion = str(lag_selection_criterion).strip().lower()
    if criterion == "bic" and "bic" in scores:
        finite_mask = np.isfinite(pd.to_numeric(scores["bic"], errors="coerce"))
        scores["selection_score"] = pd.to_numeric(scores["bic"], errors="coerce")
        choose_index = lambda frame: int(frame["selection_score"].astype(float).idxmin())
        comparison_metric = "bic"
    else:
        finite_mask = np.isfinite(pd.to_numeric(scores["delta_log_likelihood"], errors="coerce"))
        scores["selection_score"] = pd.to_numeric(scores["delta_log_likelihood"], errors="coerce")
        choose_index = lambda frame: int(frame["selection_score"].astype(float).idxmax())
        comparison_metric = "delta_log_likelihood"
    scores["selection_fallback"] = ""
    if bool(finite_mask.any()):
        selected_idx = choose_index(scores.loc[finite_mask])
        selection_note = ""
    else:
        selected_idx = 0
        selection_note = "All candidate lag selection values were non-finite; selected the first candidate lag as a fallback."
        scores.loc[:, "selection_fallback"] = selection_note
    scores["selected"] = False
    scores.loc[selected_idx, "selected"] = True
    selected_lag_ms = int(scores.loc[selected_idx, "candidate_lag_ms"])
    payload = {
        "selected_lag_ms": selected_lag_ms,
        "candidate_lags_ms": [int(value) for value in candidate_lags_ms],
        "comparison_metric": comparison_metric,
        "baseline_model_id": "M_0",
        "selected_model_id": "M_3",
        "model_backend": str(model_backend),
    }
    if selection_note:
        payload["selection_fallback"] = selection_note
    return SelectedLag(
        selected_lag_ms=selected_lag_ms,
        scores=scores,
        payload=payload,
    )
