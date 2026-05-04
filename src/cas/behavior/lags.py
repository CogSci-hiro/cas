"""Lag-selection helpers for the behavioral hazard pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cas.behavior._legacy_support import progress_iterable
from cas.behavior.formulas import render_formula, strip_random_effects
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
    verbose: bool = False,
) -> SelectedLag:
    baseline = fit_formula_model(fpp_table, model_id="A0_timing", formula=strip_random_effects(render_formula("A0_timing", lag_ms=0)))
    rows: list[dict[str, object]] = []
    for lag_ms in progress_iterable(
        list(candidate_lags_ms),
        total=len(candidate_lags_ms),
        description="Lag selection",
        enabled=verbose,
    ):
        fitted = fit_formula_model(
            fpp_table,
            model_id=f"A3_joint_information_lag_{lag_ms}",
            formula=strip_random_effects(render_formula("A3_joint_information", lag_ms=lag_ms)),
        )
        rows.append(
            {
                "candidate_lag_ms": int(lag_ms),
                "log_likelihood_baseline": baseline.log_likelihood,
                "log_likelihood_joint_information": fitted.log_likelihood,
                "delta_log_likelihood": fitted.log_likelihood - baseline.log_likelihood,
                "baseline_log_likelihood_finite": bool(np.isfinite(baseline.log_likelihood)),
                "joint_log_likelihood_finite": bool(np.isfinite(fitted.log_likelihood)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["candidate_lag_ms"], kind="mergesort").reset_index(drop=True)
    finite_delta_mask = np.isfinite(pd.to_numeric(scores["delta_log_likelihood"], errors="coerce"))
    scores["selection_fallback"] = ""
    if bool(finite_delta_mask.any()):
        selected_idx = int(scores.loc[finite_delta_mask, "delta_log_likelihood"].astype(float).idxmax())
        selection_note = ""
    else:
        selected_idx = 0
        selection_note = "All candidate lag delta log-likelihood values were non-finite; selected the first candidate lag as a fallback."
        scores.loc[:, "selection_fallback"] = selection_note
    scores["selected"] = False
    scores.loc[selected_idx, "selected"] = True
    selected_lag_ms = int(scores.loc[selected_idx, "candidate_lag_ms"])
    payload = {
        "selected_lag_ms": selected_lag_ms,
        "candidate_lags_ms": [int(value) for value in candidate_lags_ms],
        "comparison_metric": "delta_log_likelihood",
        "baseline_model_id": "A0_timing",
        "selected_model_id": "A3_joint_information",
    }
    if selection_note:
        payload["selection_fallback"] = selection_note
    return SelectedLag(
        selected_lag_ms=selected_lag_ms,
        scores=scores,
        payload=payload,
    )
