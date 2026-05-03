from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as glm
import yaml

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.episodes import build_censored_episodes, build_event_positive_episodes
from cas.hazard_behavior.features import add_information_features_to_riskset
from cas.hazard_behavior.io import resolve_surprisal_paths
from cas.hazard_behavior.progress import progress_iterable
from cas.hazard_behavior.riskset import build_discrete_time_riskset

LOGGER = logging.getLogger(__name__)

glm.SET_USE_BIC_LLF(True)

TIMING_COLUMNS = (
    "z_time_from_partner_onset_s",
    "z_time_from_partner_offset_s",
    "z_time_from_partner_offset_s_squared",
)
REQUIRED_FINAL_COLUMNS = [
    "episode_id",
    "dyad_id",
    "subject_id",
    "run_id",
    "anchor_type",
    "bin_start_s",
    "bin_end_s",
    "event_bin",
    "time_from_partner_onset_s",
    "time_from_partner_offset_s",
    "information_rate",
    "prop_expected_cumulative_info",
]
EXPECTED_ANCHOR_TYPES = {"fpp", "spp"}
DEFAULT_FINAL_CONFIG_PATH = Path("config/behavior.yaml")


@dataclass(frozen=True, slots=True)
class FinalBehaviorConfig:
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Any]:
        return dict(self.raw["paths"])

    @property
    def lag_grid_ms(self) -> list[int]:
        return [int(v) for v in self.raw["lags"]["grid_ms"]]

    @property
    def bin_size_ms(self) -> int:
        return int(self.raw["riskset"]["bin_size_ms"])


@dataclass(frozen=True, slots=True)
class FittedFinalModel:
    model_name: str
    formula: str
    result: Any
    n_rows: int
    n_events: int
    converged: bool
    log_likelihood: float
    aic: float
    bic: float
    max_abs_coefficient: float
    max_standard_error: float
    any_nan_coefficients: bool
    any_infinite_coefficients: bool
    design_condition_number: float
    overflow_warning: bool
    stable: bool
    fit_warnings: list[str]
    safety_warnings: list[str]


def load_final_behavior_config(path: Path) -> FinalBehaviorConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ["analysis", "paths", "riskset", "anchors", "columns", "features", "lags", "models", "outputs", "figures"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    return FinalBehaviorConfig(raw=data)


def format_lag_ms(lag_ms: int) -> str:
    return f"{int(lag_ms):03d}ms"


def lag_col(base: str, lag_ms: int) -> str:
    return f"{base}_lag_{format_lag_ms(lag_ms)}"


def lag_col_z(base: str, lag_ms: int) -> str:
    return f"{lag_col(base, lag_ms)}_z"


def _expand_selected_lag_placeholders(text: str, selected_lag: int) -> str:
    expanded = str(text)
    expanded = expanded.replace("SELECTED_ms", format_lag_ms(selected_lag))
    expanded = expanded.replace("SELECTED", str(int(selected_lag)))
    if "SELECTED" in expanded:
        raise ValueError(f"Unresolved selected-lag placeholder remained in formula term: {text}")
    return expanded


def _strip_random_effect_terms(formula: str) -> str:
    stripped = re.sub(r"\s*\+\s*\(1\s*\|\s*[^)]+\)", "", formula)
    return " ".join(stripped.split())


def _config_sequence(cfg: FinalBehaviorConfig) -> list[dict[str, Any]]:
    sequence = list((cfg.raw.get("models") or {}).get("sequence") or [])
    if not sequence:
        raise ValueError("Behavior-final config is missing models.sequence.")
    return sequence


def _model_sequence_names(cfg: FinalBehaviorConfig) -> tuple[str, ...]:
    return tuple(str(step["name"]) for step in _config_sequence(cfg))


def _formula_from_terms(terms: list[str]) -> str:
    return f"event_bin ~ {' + '.join(terms)}"


def _sequence_spec(cfg: FinalBehaviorConfig, selected_lag: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for step in _config_sequence(cfg):
        name = str(step["name"])
        terms = [_expand_selected_lag_placeholders(str(term), selected_lag) for term in list(step.get("formula_terms") or [])]
        specs.append(
            {
                "name": name,
                "formula_terms": terms,
                "formula": _formula_from_terms(terms),
                "compare_against": step.get("compare_against"),
            }
        )
    return specs


def _timing_terms(cfg: FinalBehaviorConfig | None = None) -> str:
    active_cfg = cfg or load_final_behavior_config(DEFAULT_FINAL_CONFIG_PATH)
    for step in _config_sequence(active_cfg):
        if str(step["name"]) == "timing_only":
            return " + ".join(str(term) for term in list(step.get("formula_terms") or []))
    raise ValueError("Behavior-final config is missing the timing_only model.")


def _base_config(cfg: FinalBehaviorConfig, out_dir: Path, anchor: str) -> BehaviourHazardConfig:
    resolved_surprisal = resolve_surprisal_paths(str(cfg.paths["surprisal_tsv"]))
    if not resolved_surprisal:
        raise FileNotFoundError(f"No surprisal TSV files matched: {cfg.paths['surprisal_tsv']}")
    return BehaviourHazardConfig(
        events_path=Path(cfg.paths["events_csv"]),
        surprisal_paths=tuple(resolved_surprisal),
        out_dir=out_dir,
        bin_size_s=float(cfg.bin_size_ms) / 1000.0,
        minimum_episode_duration_s=float(cfg.raw["riskset"]["min_episode_duration_ms"]) / 1000.0,
        include_censored=bool(cfg.raw["riskset"]["include_censored_episodes"]),
        target_fpp_label_prefix="FPP_" if anchor == "fpp" else "SPP_",
        overwrite=True,
        save_riskset=False,
    )


def _normalize_anchor_type(anchor: str) -> str:
    normalized = str(anchor).strip().lower()
    if normalized not in EXPECTED_ANCHOR_TYPES:
        raise ValueError(f"Unexpected anchor_type {anchor!r}; expected one of: fpp, spp.")
    return normalized


def _project_behavior_final_events(events: pd.DataFrame, *, anchor: str) -> pd.DataFrame:
    normalized_anchor = _normalize_anchor_type(anchor)
    working = events.copy()
    if "dyad_id" not in working.columns and "recording_id" in working.columns:
        working["dyad_id"] = working["recording_id"]
    if "speaker_fpp" not in working.columns and "participant_speaker" in working.columns:
        working["speaker_fpp"] = working["participant_speaker"]
    if "speaker_spp" not in working.columns and "partner_speaker" in working.columns:
        working["speaker_spp"] = working["partner_speaker"]
    required = {
        "dyad_id",
        "run",
        "speaker_fpp",
        "speaker_spp",
        "fpp_label",
        "spp_label",
        "fpp_onset",
        "fpp_offset",
        "spp_onset",
        "spp_offset",
    }
    missing = sorted(required - set(working.columns))
    if missing:
        raise ValueError(f"Behavior-final events table is missing required columns for paired FPP/SPP projection: {missing}")

    if normalized_anchor == "fpp":
        participant_col = "speaker_fpp"
        partner_col = "speaker_spp"
        event_label_col = "fpp_label"
        event_onset_col = "fpp_onset"
        event_offset_col = "fpp_offset"
        partner_onset_col = "spp_onset"
        partner_offset_col = "spp_offset"
        partner_label_col = "spp_label"
    else:
        participant_col = "speaker_spp"
        partner_col = "speaker_fpp"
        event_label_col = "spp_label"
        event_onset_col = "spp_onset"
        event_offset_col = "spp_offset"
        partner_onset_col = "fpp_onset"
        partner_offset_col = "fpp_offset"
        partner_label_col = "fpp_label"

    projected = pd.DataFrame(
        {
            "dyad_id": working["dyad_id"].astype(str),
            "run": working["run"].astype(str),
            "participant_speaker": working[participant_col],
            "partner_speaker": working[partner_col],
            "fpp_label": working[event_label_col],
            "fpp_onset": pd.to_numeric(working[event_onset_col], errors="coerce"),
            "fpp_offset": pd.to_numeric(working[event_offset_col], errors="coerce"),
            "spp_label": working[partner_label_col],
            "spp_onset": pd.to_numeric(working[partner_onset_col], errors="coerce"),
            "spp_offset": pd.to_numeric(working[partner_offset_col], errors="coerce"),
            "source_anchor_type": normalized_anchor,
            "source_event_id": np.arange(len(working), dtype=int),
        }
    )
    if "pair_id" in working.columns:
        projected["pair_id"] = working["pair_id"].astype(str)
    return projected


def _detect_possible_millisecond_units(source_events: pd.DataFrame) -> bool:
    if source_events.empty or "fpp_onset" not in source_events.columns or "spp_onset" not in source_events.columns:
        return False
    relative = pd.to_numeric(source_events["fpp_onset"], errors="coerce") - pd.to_numeric(source_events["spp_onset"], errors="coerce")
    finite = relative[np.isfinite(relative)]
    if finite.empty:
        return False
    return bool(float(np.nanpercentile(np.abs(finite.to_numpy(dtype=float)), 95)) > 50.0)


def _validate_anchor_labels(table: pd.DataFrame, *, expected_anchor: str | None = None) -> pd.DataFrame:
    working = table.copy()
    if "anchor_type" not in working.columns:
        if expected_anchor is None:
            raise ValueError("Risk set is missing required `anchor_type` column.")
        working["anchor_type"] = _normalize_anchor_type(expected_anchor)
    working["anchor_type"] = working["anchor_type"].astype(str).str.strip().str.lower()
    labels = set(working["anchor_type"].dropna().unique())
    unexpected = sorted(labels - EXPECTED_ANCHOR_TYPES)
    if unexpected:
        raise ValueError(f"Unexpected anchor_type labels in risk set: {unexpected}. Expected lowercase fpp/spp.")
    if expected_anchor is not None:
        normalized_expected = _normalize_anchor_type(expected_anchor)
        wrong = sorted(labels - {normalized_expected})
        if wrong:
            raise ValueError(f"Risk set for anchor `{normalized_expected}` contains mismatched anchor_type labels: {wrong}")
    return working


def _summarize_anchor_riskset(
    *,
    anchor: str,
    riskset: pd.DataFrame,
    episodes: pd.DataFrame,
    source_events: pd.DataFrame,
) -> pd.DataFrame:
    event_values = pd.to_numeric(riskset["event_bin"], errors="coerce").fillna(0).astype(int)
    if riskset.empty:
        n_episodes_with_event = 0
    else:
        n_episodes_with_event = int(
            riskset.assign(_event_bin=event_values)
            .groupby("episode_id", sort=False)["_event_bin"]
            .max()
            .sum()
        )
    return pd.DataFrame(
        [
            {
                "anchor_type": _normalize_anchor_type(anchor),
                "n_rows": int(len(riskset)),
                "n_episodes": int(len(episodes)),
                "n_source_events": int(len(source_events)),
                "n_event_bins": int(event_values.sum()),
                "n_episodes_with_event": n_episodes_with_event,
            }
        ]
    )


def _build_event_alignment_diagnostics(
    *,
    anchor: str,
    source_events: pd.DataFrame,
    episodes: pd.DataFrame,
    riskset: pd.DataFrame,
) -> pd.DataFrame:
    normalized_anchor = _normalize_anchor_type(anchor)
    episode_lookup = episodes.copy()
    if "source_event_id" in episode_lookup.columns:
        episode_lookup["source_event_id"] = pd.to_numeric(episode_lookup["source_event_id"], errors="coerce")
    event_rows = riskset.loc[pd.to_numeric(riskset["event_bin"], errors="coerce") == 1].copy()
    if "source_event_id" in event_rows.columns:
        event_rows["source_event_id"] = pd.to_numeric(event_rows["source_event_id"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, source_row in source_events.iterrows():
        source_event_id = pd.to_numeric(pd.Series([source_row.get("source_event_id")]), errors="coerce").iloc[0]
        source_onset = pd.to_numeric(pd.Series([source_row.get("fpp_onset")]), errors="coerce").iloc[0]
        episode_match = episode_lookup.loc[episode_lookup["source_event_id"] == source_event_id] if "source_event_id" in episode_lookup.columns else pd.DataFrame()
        matched_bin_start = np.nan
        matched_bin_end = np.nan
        matched_event_bin = 0
        matched_success = False
        failure_reason = ""
        episode_id = np.nan

        if not np.isfinite(source_onset):
            failure_reason = "missing_source_event_onset"
        elif episode_match.empty:
            failure_reason = "no_episode_assigned"
        elif len(episode_match) > 1:
            failure_reason = "multiple_episodes_assigned"
            episode_id = str(episode_match.iloc[0]["episode_id"])
        else:
            episode_id = str(episode_match.iloc[0]["episode_id"])
            matched_event_rows = event_rows.loc[event_rows["episode_id"].astype(str) == episode_id]
            if matched_event_rows.empty:
                failure_reason = "no_event_bin_assigned"
            elif len(matched_event_rows) > 1:
                failure_reason = "multiple_event_bins_assigned"
                matched_bin_start = float(matched_event_rows.iloc[0]["bin_start_s"])
                matched_bin_end = float(matched_event_rows.iloc[0]["bin_end_s"])
                matched_event_bin = 1
            else:
                matched = matched_event_rows.iloc[0]
                matched_bin_start = float(matched["bin_start_s"])
                matched_bin_end = float(matched["bin_end_s"])
                matched_event_bin = int(matched["event_bin"])
                matched_success = bool(matched_bin_start <= float(source_onset) < matched_bin_end)
                if not matched_success:
                    failure_reason = "source_onset_outside_event_bin"

        rows.append(
            {
                "anchor_type": normalized_anchor,
                "source_event_id": int(source_event_id) if np.isfinite(source_event_id) else np.nan,
                "episode_id": episode_id,
                "source_event_onset_s": float(source_onset) if np.isfinite(source_onset) else np.nan,
                "matched_bin_start_s": matched_bin_start,
                "matched_bin_end_s": matched_bin_end,
                "matched_event_bin": matched_event_bin,
                "matched_success": bool(matched_success),
                "failure_reason": failure_reason,
            }
        )
    return pd.DataFrame(rows)


def _validate_behavior_final_anchor_riskset(
    *,
    anchor: str,
    riskset: pd.DataFrame,
    episodes: pd.DataFrame,
    source_events: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics_dir = out_dir / "riskset"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize_anchor_riskset(anchor=anchor, riskset=riskset, episodes=episodes, source_events=source_events)
    summary["possible_millisecond_units"] = _detect_possible_millisecond_units(source_events)
    alignment = _build_event_alignment_diagnostics(anchor=anchor, source_events=source_events, episodes=episodes, riskset=riskset)
    summary.to_csv(diagnostics_dir / "validation_summary.csv", index=False)
    alignment.to_csv(diagnostics_dir / "event_alignment_diagnostics.csv", index=False)
    n_source_events = int(summary.iloc[0]["n_source_events"])
    n_event_bins = int(summary.iloc[0]["n_event_bins"])
    if n_source_events > 0 and n_event_bins == 0:
        raise ValueError(
            f"Behavior-final {anchor} risk set is invalid: {n_source_events} source events were present but zero event bins were assigned."
    )
    return summary, alignment


def _raw_riskset_for_alignment_qc(riskset: pd.DataFrame, *, anchor: str) -> pd.DataFrame:
    working = riskset.copy()
    working["anchor_type"] = _normalize_anchor_type(anchor)
    rename_map = {
        "event": "event_bin",
        "bin_start": "bin_start_s",
        "bin_end": "bin_end_s",
        "time_from_partner_onset": "time_from_partner_onset_s",
        "time_from_partner_offset": "time_from_partner_offset_s",
    }
    for source_column, target_column in rename_map.items():
        if source_column in working.columns and target_column not in working.columns:
            working[target_column] = working[source_column]
    return working


def _guard_nonzero_event_bins_for_modeling(riskset: pd.DataFrame, *, anchor: str) -> None:
    n_rows = int(len(riskset))
    n_event_bins = int(pd.to_numeric(riskset["event_bin"], errors="coerce").fillna(0).sum())
    if n_rows > 0 and n_event_bins == 0:
        normalized_anchor = _normalize_anchor_type(anchor)
        if normalized_anchor == "spp":
            raise ValueError(
                "SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins."
            )
        raise ValueError(
            f"{normalized_anchor.upper()} risk set contained zero event bins; modeling was aborted."
        )


def _build_anchor_riskset(cfg: FinalBehaviorConfig, anchor: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    normalized_anchor = _normalize_anchor_type(anchor)
    LOGGER.info("Building %s anchor risk set.", normalized_anchor)
    bcfg = _base_config(cfg, out_dir=out_dir, anchor=normalized_anchor)
    from cas.hazard_behavior.io import read_events_table, read_surprisal_tables
    from cas.hazard_behavior.episodes import extract_fpp_events

    events, event_warnings = read_events_table(bcfg.events_path)
    surprisal, surprisal_warnings = read_surprisal_tables(
        bcfg.surprisal_paths,
        unmatched_surprisal_strategy=bcfg.unmatched_surprisal_strategy,
    )
    projected_events = _project_behavior_final_events(events, anchor=normalized_anchor)
    source_events = extract_fpp_events(projected_events, bcfg)

    positive = build_event_positive_episodes(events_table=projected_events, surprisal_table=surprisal, config=bcfg)
    episodes = positive.episodes
    warnings_list: list[str] = [*event_warnings, *surprisal_warnings, *positive.warnings]
    if bcfg.include_censored:
        censored = build_censored_episodes(
            events_table=projected_events,
            surprisal_table=surprisal,
            positive_episodes=positive.episodes,
            config=bcfg,
        )
        if not censored.empty:
            episodes = pd.concat([episodes, censored], ignore_index=True, sort=False)

    riskset_result = build_discrete_time_riskset(episodes, config=bcfg)
    qc_riskset = _raw_riskset_for_alignment_qc(riskset_result.riskset_table, anchor=normalized_anchor)
    summary, _alignment = _validate_behavior_final_anchor_riskset(
        anchor=normalized_anchor,
        riskset=qc_riskset,
        episodes=episodes,
        source_events=source_events,
        out_dir=out_dir,
    )
    riskset_with_info, _ = add_information_features_to_riskset(
        riskset_table=riskset_result.riskset_table,
        episodes_table=episodes,
        surprisal_table=surprisal,
        config=bcfg,
    )
    warnings_list.extend(riskset_result.warnings)
    normalized_riskset = _normalize_final_riskset(riskset_with_info, normalized_anchor)
    if bool(summary.iloc[0]["possible_millisecond_units"]):
        warnings_list.append(
            f"Behavior-final {normalized_anchor} source event latencies appear unusually large; check whether event onset columns are in milliseconds instead of seconds."
        )
    return normalized_riskset, episodes, warnings_list


def _normalize_final_riskset(table: pd.DataFrame, anchor: str) -> pd.DataFrame:
    out = table.copy()
    normalized_anchor = _normalize_anchor_type(anchor)
    out["anchor_type"] = normalized_anchor
    out["subject_id"] = out["participant_speaker"].astype(str)
    out["run_id"] = out["run"].astype(str)
    out = out.rename(
        columns={
            "event": "event_bin",
            "bin_start": "bin_start_s",
            "bin_end": "bin_end_s",
            "time_from_partner_onset": "time_from_partner_onset_s",
            "time_from_partner_offset": "time_from_partner_offset_s",
        }
    )
    if "prop_expected_cumulative_info" not in out.columns and "expected_cumulative_info" in out.columns:
        out["prop_expected_cumulative_info"] = out["expected_cumulative_info"]
    missing = [c for c in REQUIRED_FINAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required final riskset columns: {', '.join(missing)}")
    keep = list(dict.fromkeys(REQUIRED_FINAL_COLUMNS + [c for c in out.columns if c not in REQUIRED_FINAL_COLUMNS]))
    normalized = out.loc[:, keep].copy()
    normalized["event_bin"] = pd.to_numeric(normalized["event_bin"], errors="coerce").astype(int)
    normalized = _validate_anchor_labels(normalized, expected_anchor=normalized_anchor)
    return normalized


def _zscore_strict(values: pd.Series, *, source_column: str, output_column: str) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"Timing feature `{source_column}` has zero or non-finite variance within anchor; cannot z-score.")
    z_values = (numeric - mean) / sd
    return z_values, {
        "anchor_scope": "within_anchor",
        "feature_family": "timing",
        "source_column": source_column,
        "column": output_column,
        "mean": mean,
        "sd": sd,
    }


def _add_primary_timing_columns(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = table.copy()
    onset_z, onset_stats = _zscore_strict(
        out["time_from_partner_onset_s"],
        source_column="time_from_partner_onset_s",
        output_column="z_time_from_partner_onset_s",
    )
    offset_z, offset_stats = _zscore_strict(
        out["time_from_partner_offset_s"],
        source_column="time_from_partner_offset_s",
        output_column="z_time_from_partner_offset_s",
    )
    out["z_time_from_partner_onset_s"] = onset_z
    out["z_time_from_partner_offset_s"] = offset_z
    out["z_time_from_partner_offset_s_squared"] = out["z_time_from_partner_offset_s"] ** 2
    stats_rows = [
        onset_stats,
        offset_stats,
        {
            "anchor_scope": "within_anchor",
            "feature_family": "timing",
            "source_column": "z_time_from_partner_offset_s",
            "column": "z_time_from_partner_offset_s_squared",
            "mean": float(out["z_time_from_partner_offset_s_squared"].mean()),
            "sd": float(out["z_time_from_partner_offset_s_squared"].std(ddof=0)),
            "note": "Derived deterministically as z_time_from_partner_offset_s ** 2",
        },
    ]
    return out, pd.DataFrame(stats_rows)


def _add_lagged_z_columns(table: pd.DataFrame, lag_grid_ms: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = table.copy().sort_values(["episode_id", "bin_start_s"], kind="mergesort")
    for lag in progress_iterable(
        lag_grid_ms,
        total=len(lag_grid_ms),
        description="Lagged columns",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        steps = int(round(lag / 50.0))
        for base in ("information_rate", "prop_expected_cumulative_info"):
            column_name = lag_col(base, lag)
            out[column_name] = out.groupby("episode_id", sort=False)[base].shift(steps).fillna(0.0)

    stats_rows: list[dict[str, Any]] = []
    for base in ("information_rate", "prop_expected_cumulative_info"):
        for lag in lag_grid_ms:
            column_name = lag_col(base, lag)
            z_column_name = lag_col_z(base, lag)
            values = pd.to_numeric(out[column_name], errors="coerce")
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or sd <= 0.0:
                sd = 1.0
            out[z_column_name] = (values - mean) / sd
            stats_rows.append(
                {
                    "anchor_scope": "within_anchor",
                    "feature_family": "information",
                    "base_column": base,
                    "lag_ms": int(lag),
                    "source_column": column_name,
                    "column": z_column_name,
                    "mean": mean,
                    "sd": sd,
                }
            )
    return out, pd.DataFrame(stats_rows)


def _prepare_final_riskset(table: pd.DataFrame, lag_grid_ms: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_timing, timing_stats = _add_primary_timing_columns(table)
    with_lags, lag_stats = _add_lagged_z_columns(with_timing, lag_grid_ms)
    standardization = pd.concat([timing_stats, lag_stats], ignore_index=True, sort=False)
    return with_lags, standardization


def _fit_formula(table: pd.DataFrame, formula: str) -> tuple[Any, list[str]]:
    fit_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.GLM.from_formula(formula=formula, data=table, family=sm.families.Binomial())
        result = model.fit()
    for warning in caught:
        message = str(warning.message)
        if warning.category is RuntimeWarning or "overflow" in message.lower():
            fit_warnings.append(f"{warning.category.__name__}: {message}")
        elif message:
            fit_warnings.append(f"{warning.category.__name__}: {message}")
    return result, fit_warnings


def _coef(result: Any, name: str) -> float | None:
    return float(result.params[name]) if name in result.params.index else None


def _condition_number(result: Any) -> float:
    exog = np.asarray(result.model.exog, dtype=float)
    if exog.size == 0:
        return float("nan")
    try:
        return float(np.linalg.cond(exog))
    except np.linalg.LinAlgError:
        return float("inf")


def _build_safety_warnings(
    *,
    converged: bool,
    fit_warnings: list[str],
    max_abs_coefficient: float,
    max_standard_error: float,
    any_nan_coefficients: bool,
    any_infinite_coefficients: bool,
    design_condition_number: float,
) -> tuple[bool, bool, list[str]]:
    warnings_list: list[str] = []
    overflow_warning = any("overflow" in message.lower() for message in fit_warnings)
    if not converged:
        warnings_list.append("Model did not converge.")
    if overflow_warning:
        warnings_list.append("Overflow warning occurred during fit.")
    if np.isfinite(max_abs_coefficient) and max_abs_coefficient >= 20.0:
        warnings_list.append("Maximum absolute coefficient exceeded or equaled 20.")
    if np.isfinite(max_standard_error) and max_standard_error >= 10.0:
        warnings_list.append("Maximum standard error exceeded or equaled 10.")
    if any_nan_coefficients:
        warnings_list.append("NaN coefficient detected.")
    if any_infinite_coefficients:
        warnings_list.append("Infinite coefficient detected.")
    if np.isfinite(design_condition_number) and design_condition_number >= 1000.0:
        warnings_list.append("Design condition number exceeded or equaled 1000.")
    stable = (
        converged
        and not overflow_warning
        and bool(np.isfinite(max_abs_coefficient) and max_abs_coefficient < 20.0)
        and bool(np.isfinite(max_standard_error) and max_standard_error < 10.0)
        and not any_nan_coefficients
        and not any_infinite_coefficients
        and bool(np.isfinite(design_condition_number) and design_condition_number < 1000.0)
    )
    return overflow_warning, stable, warnings_list


def _summarize_fit(
    *,
    result: Any,
    model_name: str,
    formula: str,
    n_rows: int,
    n_events: int,
    fit_warnings: list[str],
) -> FittedFinalModel:
    params = pd.to_numeric(pd.Series(result.params), errors="coerce")
    bse = pd.to_numeric(pd.Series(result.bse), errors="coerce")
    converged = bool(getattr(result, "converged", True))
    max_abs_coefficient = float(np.nanmax(np.abs(params.to_numpy(dtype=float)))) if len(params) else float("nan")
    max_standard_error = float(np.nanmax(np.abs(bse.to_numpy(dtype=float)))) if len(bse) else float("nan")
    any_nan_coefficients = bool(np.isnan(params.to_numpy(dtype=float)).any())
    any_infinite_coefficients = bool(np.isinf(params.to_numpy(dtype=float)).any())
    design_condition_number = _condition_number(result)
    overflow_warning, stable, safety_warnings = _build_safety_warnings(
        converged=converged,
        fit_warnings=fit_warnings,
        max_abs_coefficient=max_abs_coefficient,
        max_standard_error=max_standard_error,
        any_nan_coefficients=any_nan_coefficients,
        any_infinite_coefficients=any_infinite_coefficients,
        design_condition_number=design_condition_number,
    )
    return FittedFinalModel(
        model_name=model_name,
        formula=formula,
        result=result,
        n_rows=n_rows,
        n_events=n_events,
        converged=converged,
        log_likelihood=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        max_abs_coefficient=max_abs_coefficient,
        max_standard_error=max_standard_error,
        any_nan_coefficients=any_nan_coefficients,
        any_infinite_coefficients=any_infinite_coefficients,
        design_condition_number=design_condition_number,
        overflow_warning=overflow_warning,
        stable=stable,
        fit_warnings=fit_warnings,
        safety_warnings=safety_warnings,
    )


def _fit_final_model(table: pd.DataFrame, model_name: str, formula: str) -> FittedFinalModel:
    result, fit_warnings = _fit_formula(table, formula)
    return _summarize_fit(
        result=result,
        model_name=model_name,
        formula=formula,
        n_rows=int(len(table)),
        n_events=int(pd.to_numeric(table["event_bin"], errors="coerce").sum()),
        fit_warnings=fit_warnings,
    )


def _summary_row(fitted: FittedFinalModel, *, anchor_type: str) -> dict[str, Any]:
    return {
        "anchor_type": anchor_type,
        "model_name": fitted.model_name,
        "formula": fitted.formula,
        "n_rows": fitted.n_rows,
        "n_events": fitted.n_events,
        "converged": fitted.converged,
        "fit_warnings": " | ".join(fitted.fit_warnings),
        "log_likelihood": fitted.log_likelihood,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "max_abs_coefficient": fitted.max_abs_coefficient,
        "max_standard_error": fitted.max_standard_error,
        "any_nan_coefficients": fitted.any_nan_coefficients,
        "any_infinite_coefficients": fitted.any_infinite_coefficients,
        "design_condition_number": fitted.design_condition_number,
        "overflow_warning": fitted.overflow_warning,
        "stable": fitted.stable,
        "safety_warnings": " | ".join(fitted.safety_warnings),
    }


def _coefficient_rows(fitted: FittedFinalModel, *, anchor_type: str) -> list[dict[str, Any]]:
    conf_int = fitted.result.conf_int()
    rows: list[dict[str, Any]] = []
    for term in fitted.result.params.index:
        estimate = float(fitted.result.params[term])
        standard_error = float(fitted.result.bse[term])
        z_value = estimate / standard_error if np.isfinite(standard_error) and standard_error != 0.0 else np.nan
        p_value = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        rows.append(
            {
                "anchor_type": anchor_type,
                "model_name": fitted.model_name,
                "term": str(term),
                "estimate": estimate,
                "standard_error": standard_error,
                "z": z_value,
                "p_value": p_value,
                "conf_low": float(conf_int.loc[term, 0]),
                "conf_high": float(conf_int.loc[term, 1]),
            }
        )
    return rows


def _lr_test(parent: FittedFinalModel, child: FittedFinalModel) -> tuple[float | None, float | None, float | None]:
    try:
        parent_df = float(getattr(parent.result, "df_model", np.nan))
        child_df = float(getattr(child.result, "df_model", np.nan))
        lr_df = child_df - parent_df
        lr_statistic = 2.0 * (child.log_likelihood - parent.log_likelihood)
        if not np.isfinite(lr_df) or lr_df <= 0.0 or not np.isfinite(lr_statistic):
            return None, None, None
        lr_p_value = float(stats.chi2.sf(lr_statistic, lr_df))
        return float(lr_statistic), float(lr_df), lr_p_value
    except Exception:
        return None, None, None


def _comparison_row(
    parent: FittedFinalModel,
    child: FittedFinalModel,
    *,
    anchor_type: str,
    selected_lag_ms: int,
) -> dict[str, Any]:
    lr_statistic, lr_df, lr_p_value = _lr_test(parent, child)
    return {
        "anchor_type": anchor_type,
        "selected_lag_ms": int(selected_lag_ms),
        "base_model": parent.model_name,
        "interaction_model": child.model_name,
        "parent_model": parent.model_name,
        "child_model": child.model_name,
        "base_aic": parent.aic,
        "interaction_aic": child.aic,
        "parent_aic": parent.aic,
        "child_aic": child.aic,
        "delta_aic": child.aic - parent.aic,
        "base_bic": parent.bic,
        "interaction_bic": child.bic,
        "parent_bic": parent.bic,
        "child_bic": child.bic,
        "delta_bic": child.bic - parent.bic,
        "base_log_likelihood": parent.log_likelihood,
        "interaction_log_likelihood": child.log_likelihood,
        "parent_log_likelihood": parent.log_likelihood,
        "child_log_likelihood": child.log_likelihood,
        "lr_statistic": lr_statistic,
        "lr_df": lr_df,
        "lr_p_value": lr_p_value,
        "both_converged": parent.converged and child.converged,
        "both_stable": parent.stable and child.stable,
        "warnings": " | ".join(parent.safety_warnings + child.safety_warnings),
    }


def _comparison_specs(cfg: FinalBehaviorConfig) -> list[tuple[str, str]]:
    specs = _sequence_spec(cfg, selected_lag=0)
    names = [str(spec["name"]) for spec in specs]
    rows: list[tuple[str, str]] = []
    for idx, spec in enumerate(specs):
        child_name = str(spec["name"])
        if idx == 0:
            continue
        parent_name = str(spec.get("compare_against") or names[idx - 1])
        rows.append((parent_name, child_name))
    return rows


def _compare_rows(
    fitted_models: dict[str, FittedFinalModel],
    *,
    anchor_type: str,
    selected_lag_ms: int,
    cfg: FinalBehaviorConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent_name, child_name in _comparison_specs(cfg):
        rows.append(
            _comparison_row(
                fitted_models[parent_name],
                fitted_models[child_name],
                anchor_type=anchor_type,
                selected_lag_ms=selected_lag_ms,
            )
        )
    return rows


def _lag_selection_row(fitted: FittedFinalModel, *, lag_ms: int, baseline: FittedFinalModel, r_col: str, p_col: str) -> dict[str, Any]:
    return {
        "lag_ms": int(lag_ms),
        "formula": fitted.formula,
        "n_rows": fitted.n_rows,
        "n_events": fitted.n_events,
        "converged": fitted.converged,
        "stable": fitted.stable,
        "log_likelihood": fitted.log_likelihood,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "delta_aic_vs_timing": fitted.aic - baseline.aic,
        "delta_bic_vs_timing": fitted.bic - baseline.bic,
        "information_rate_coef": _coef(fitted.result, r_col),
        "prop_expected_cumulative_info_coef": _coef(fitted.result, p_col),
        "fit_warnings": " | ".join(fitted.fit_warnings),
        "safety_warnings": " | ".join(fitted.safety_warnings),
    }


def _selected_lag_payload(selected_lag_ms: int) -> dict[str, Any]:
    return {
        "selected_lag_ms": int(selected_lag_ms),
        "selection_anchor": "fpp",
        "selection_model": "full_information",
        "selection_criterion": "bic",
        "shared_lag_across_information_predictors": True,
        "shared_lag_across_anchors": True,
        "spp_reselects_lag": False,
        "reuse_policy": "SPP and pooled FPP-vs-SPP reuse the FPP-selected lag without reselection.",
    }


def _formula_sequence(selected_lag: int, cfg: FinalBehaviorConfig | None = None) -> dict[str, str]:
    active_cfg = cfg or load_final_behavior_config(DEFAULT_FINAL_CONFIG_PATH)
    return {spec["name"]: spec["formula"] for spec in _sequence_spec(active_cfg, selected_lag)}


def _interaction_formula(selected_lag: int, cfg: FinalBehaviorConfig | None = None) -> str:
    active_cfg = cfg or load_final_behavior_config(DEFAULT_FINAL_CONFIG_PATH)
    formula = str(((active_cfg.raw.get("models") or {}).get("pooled_comparison") or {}).get("formula") or "").strip()
    if not formula:
        raise ValueError("Behavior-final config is missing models.pooled_comparison.formula.")
    formula = " ".join(formula.split())
    return _strip_random_effect_terms(_expand_selected_lag_placeholders(formula, selected_lag))


def _pooled_timing_information_rate_interaction_formula(selected_lag: int, cfg: FinalBehaviorConfig | None = None) -> str:
    active_cfg = cfg or load_final_behavior_config(DEFAULT_FINAL_CONFIG_PATH)
    interaction_cfg = ((active_cfg.raw.get("models") or {}).get("timing_information_rate_interaction") or {})
    if not bool(((interaction_cfg.get("pooled_anchor_comparison") or {}).get("enabled"))):
        raise ValueError("Pooled timing-by-information-rate interaction comparison is disabled in config.")
    full_formula = _formula_sequence(selected_lag, active_cfg)["timing_information_rate_interaction"]
    rhs = full_formula.split("~", maxsplit=1)[1].strip()
    return f"event_bin ~ anchor_type * ({rhs})"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _infer_anchor_delta_terms(terms: list[str], predictor: str) -> tuple[str, str]:
    predictor_terms = [term for term in terms if predictor in term]
    main_term = next((term for term in predictor_terms if "anchor_type" not in term), predictor)
    interaction_term = next((term for term in predictor_terms if "anchor_type" in term), "")
    return main_term, interaction_term


def _linear_contrast(summary: pd.DataFrame, covariance: pd.DataFrame, weights: dict[str, float], label: str) -> dict[str, Any]:
    params = summary.set_index("term")["estimate"]
    aligned_weights = pd.Series(0.0, index=params.index, dtype=float)
    for term, weight in weights.items():
        if term in aligned_weights.index:
            aligned_weights.loc[term] = float(weight)
    estimate = float((aligned_weights * params).sum())
    cov_aligned = covariance.reindex(index=aligned_weights.index, columns=aligned_weights.index, fill_value=0.0)
    variance = float(aligned_weights.to_numpy() @ cov_aligned.to_numpy() @ aligned_weights.to_numpy())
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))
    z_value = estimate / standard_error if standard_error > 0.0 else np.nan
    p_value = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    conf_low = estimate - 1.96 * standard_error if np.isfinite(standard_error) else np.nan
    conf_high = estimate + 1.96 * standard_error if np.isfinite(standard_error) else np.nan
    return {
        "contrast": label,
        "estimate": estimate,
        "standard_error": standard_error,
        "z": z_value,
        "p_value": p_value,
        "conf_low": conf_low,
        "conf_high": conf_high,
    }


def build_information_effect_contrasts(summary: pd.DataFrame, covariance: pd.DataFrame, *, info_rate_col: str, prop_col: str) -> pd.DataFrame:
    terms = summary["term"].astype(str).tolist()
    rate_main, rate_interaction = _infer_anchor_delta_terms(terms, info_rate_col)
    prop_main, prop_interaction = _infer_anchor_delta_terms(terms, prop_col)
    rows = [
        _linear_contrast(summary, covariance, {rate_main: 1.0, rate_interaction: 1.0}, "FPP information_rate effect"),
        _linear_contrast(summary, covariance, {rate_main: 1.0}, "SPP information_rate effect"),
        _linear_contrast(summary, covariance, {rate_interaction: 1.0}, "FPP - SPP information_rate effect"),
        _linear_contrast(summary, covariance, {prop_main: 1.0, prop_interaction: 1.0}, "FPP prop_expected_cumulative_info effect"),
        _linear_contrast(summary, covariance, {prop_main: 1.0}, "SPP prop_expected_cumulative_info effect"),
        _linear_contrast(summary, covariance, {prop_interaction: 1.0}, "FPP - SPP prop_expected_cumulative_info effect"),
    ]
    return pd.DataFrame(rows)


def build_timing_information_anchor_contrasts(coefficients: pd.DataFrame, *, info_rate_col: str) -> pd.DataFrame:
    mask = (
        coefficients["term"].astype(str).str.contains("anchor_type", regex=False)
        & coefficients["term"].astype(str).str.contains(info_rate_col, regex=False)
        & (
            coefficients["term"].astype(str).str.contains("z_time_from_partner_onset_s", regex=False)
            | coefficients["term"].astype(str).str.contains("z_time_from_partner_offset_s", regex=False)
        )
    )
    contrasts = coefficients.loc[mask].copy()
    if contrasts.empty:
        return pd.DataFrame(
            columns=[
                "contrast",
                "term",
                "estimate",
                "standard_error",
                "z",
                "p_value",
                "conf_low",
                "conf_high",
            ]
        )
    contrasts.insert(0, "contrast", "Anchor difference in timing x information-rate interaction")
    return contrasts.loc[:, ["contrast", "term", "estimate", "standard_error", "z", "p_value", "conf_low", "conf_high"]]


def _conclusion_text(contrasts: pd.DataFrame) -> str:
    if "contrast" not in contrasts.columns:
        return "SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins."
    inferential = contrasts.loc[contrasts["contrast"].astype(str).str.startswith("FPP - SPP"), :].copy()
    if inferential.empty:
        return "Information predictors were not selectively associated with FPP initiation relative to SPP response timing."
    significant = bool((pd.to_numeric(inferential["p_value"], errors="coerce") < 0.05).fillna(False).any())
    if significant:
        return "Information predictors were selectively associated with FPP initiation relative to SPP response timing."
    return "Information predictors were not selectively associated with FPP initiation relative to SPP response timing."


def _is_interpretable(contrasts: pd.DataFrame, pooled_summary: pd.DataFrame, spp_summary: pd.DataFrame) -> bool:
    pooled_stable = bool(pd.Series(pooled_summary["stable"]).fillna(False).all())
    spp_stable = bool(pd.Series(spp_summary["stable"]).fillna(False).all())
    finite_rows = pd.to_numeric(contrasts["standard_error"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all()
    return pooled_stable and spp_stable and bool(finite_rows)


def _save_line_plot(path: Path, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str, vline: float | None = None) -> Path:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    if vline is not None:
        plt.axvline(vline, color="tab:red", linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _save_lag_selection_plots(table: pd.DataFrame, out_dir: Path, selected_lag_ms: int) -> dict[str, str]:
    files: dict[str, str] = {}
    files["lag_selection_plot"] = str(_save_line_plot(
        out_dir / "lag_selection_plot.png",
        table["lag_ms"].to_numpy(),
        table["bic"].to_numpy(),
        xlabel="Lag (ms)",
        ylabel="BIC",
        title="FPP Lag Selection BIC",
        vline=float(selected_lag_ms),
    ))
    files["fpp_lag_selection_bic_curve"] = str(_save_line_plot(
        out_dir / "fpp_lag_selection_bic_curve.png",
        table["lag_ms"].to_numpy(),
        table["bic"].to_numpy(),
        xlabel="Lag (ms)",
        ylabel="BIC",
        title="FPP Lag Selection BIC",
        vline=float(selected_lag_ms),
    ))
    files["fpp_lag_selection_delta_bic_curve"] = str(_save_line_plot(
        out_dir / "fpp_lag_selection_delta_bic_curve.png",
        table["lag_ms"].to_numpy(),
        table["delta_bic_vs_timing"].to_numpy(),
        xlabel="Lag (ms)",
        ylabel="Delta BIC vs timing-only",
        title="FPP Lag Selection Delta BIC",
        vline=float(selected_lag_ms),
    ))
    return files


def _predict_probabilities(table: pd.DataFrame, fitted: FittedFinalModel) -> pd.Series:
    return pd.Series(fitted.result.predict(table), index=table.index, dtype=float)


def _calibration_bins(observed: pd.Series, predicted: pd.Series, *, n_bins: int = 10) -> pd.DataFrame:
    working = pd.DataFrame({"observed": observed.astype(float), "predicted": predicted.astype(float)})
    working["bin"] = pd.qcut(working["predicted"], q=min(n_bins, max(2, working["predicted"].nunique())), duplicates="drop")
    return (
        working.groupby("bin", observed=True)
        .agg(mean_predicted=("predicted", "mean"), mean_observed=("observed", "mean"))
        .reset_index(drop=True)
    )


def _fit_anchor_models_for_qc(riskset: pd.DataFrame, selected_lag_ms: int) -> dict[str, FittedFinalModel]:
    formulas = _formula_sequence(selected_lag_ms)
    return {name: _fit_final_model(riskset, name, formula) for name, formula in formulas.items()}


def _interaction_plot_grid(
    riskset: pd.DataFrame,
    *,
    selected_lag_ms: int,
    varying_column: str,
    fixed_column: str,
    info_levels: tuple[float, ...] = (-1.0, 0.0, 1.0),
    n_points: int = 120,
) -> pd.DataFrame:
    info_col = lag_col_z("information_rate", selected_lag_ms)
    prop_col = lag_col_z("prop_expected_cumulative_info", selected_lag_ms)
    varying_values = pd.to_numeric(riskset[varying_column], errors="coerce")
    fixed_values = pd.to_numeric(riskset[fixed_column], errors="coerce")
    x_grid = np.linspace(float(varying_values.min()), float(varying_values.max()), n_points)
    fixed_value = float(fixed_values.median())
    rows: list[dict[str, float]] = []
    for info_level in info_levels:
        for x_value in x_grid:
            row = {
                "time_from_partner_onset_s": float(fixed_value if varying_column != "time_from_partner_onset_s" else x_value),
                "time_from_partner_offset_s": float(fixed_value if varying_column != "time_from_partner_offset_s" else x_value),
                info_col: float(info_level),
                prop_col: 0.0,
                "information_rate_z_level": float(info_level),
            }
            if varying_column == "time_from_partner_onset_s":
                row["time_from_partner_offset_s"] = float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").median())
            if varying_column == "time_from_partner_offset_s":
                row["time_from_partner_onset_s"] = float(pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce").median())
            rows.append(row)
    grid = pd.DataFrame(rows)
    onset_mean = float(pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce").mean())
    onset_sd = float(pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce").std(ddof=0))
    offset_mean = float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").mean())
    offset_sd = float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").std(ddof=0))
    grid["z_time_from_partner_onset_s"] = (pd.to_numeric(grid["time_from_partner_onset_s"], errors="coerce") - onset_mean) / onset_sd
    grid["z_time_from_partner_offset_s"] = (pd.to_numeric(grid["time_from_partner_offset_s"], errors="coerce") - offset_mean) / offset_sd
    grid["z_time_from_partner_offset_s_squared"] = grid["z_time_from_partner_offset_s"] ** 2
    return grid


def _save_interaction_line_figure(
    *,
    path: Path,
    grid: pd.DataFrame,
    x_column: str,
    fitted: FittedFinalModel,
    title: str,
    xlabel: str,
) -> Path:
    plotted = grid.copy()
    plotted["predicted_hazard"] = pd.Series(fitted.result.predict(plotted), index=plotted.index, dtype=float)
    plt.figure(figsize=(6.5, 4.5))
    for info_level, label in ((-1.0, "information_rate_z = -1"), (0.0, "information_rate_z = 0"), (1.0, "information_rate_z = +1")):
        subset = plotted.loc[np.isclose(plotted["information_rate_z_level"], info_level)]
        plt.plot(subset[x_column], subset["predicted_hazard"], label=label, linewidth=2.0)
    plt.xlabel(xlabel)
    plt.ylabel("Predicted hazard")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _save_interaction_surface_figure(
    *,
    path: Path,
    riskset: pd.DataFrame,
    selected_lag_ms: int,
    fitted: FittedFinalModel,
) -> Path:
    info_col = lag_col_z("information_rate", selected_lag_ms)
    prop_col = lag_col_z("prop_expected_cumulative_info", selected_lag_ms)
    onset_values = pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce")
    info_values = pd.to_numeric(riskset[info_col], errors="coerce")
    onset_grid = np.linspace(float(onset_values.min()), float(onset_values.max()), 80)
    info_grid = np.linspace(-1.5, 1.5, 80) if info_values.isna().all() else np.linspace(float(info_values.min()), float(info_values.max()), 80)
    onset_mesh, info_mesh = np.meshgrid(onset_grid, info_grid)
    pred_grid = pd.DataFrame(
        {
            "time_from_partner_onset_s": onset_mesh.ravel(),
            "time_from_partner_offset_s": float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").median()),
            info_col: info_mesh.ravel(),
            prop_col: 0.0,
        }
    )
    onset_mean = float(pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce").mean())
    onset_sd = float(pd.to_numeric(riskset["time_from_partner_onset_s"], errors="coerce").std(ddof=0))
    offset_mean = float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").mean())
    offset_sd = float(pd.to_numeric(riskset["time_from_partner_offset_s"], errors="coerce").std(ddof=0))
    pred_grid["z_time_from_partner_onset_s"] = (pd.to_numeric(pred_grid["time_from_partner_onset_s"], errors="coerce") - onset_mean) / onset_sd
    pred_grid["z_time_from_partner_offset_s"] = (pd.to_numeric(pred_grid["time_from_partner_offset_s"], errors="coerce") - offset_mean) / offset_sd
    pred_grid["z_time_from_partner_offset_s_squared"] = pred_grid["z_time_from_partner_offset_s"] ** 2
    predicted = np.asarray(fitted.result.predict(pred_grid), dtype=float).reshape(onset_mesh.shape)
    plt.figure(figsize=(6.5, 4.75))
    contour = plt.contourf(onset_mesh, info_mesh, predicted, levels=24, cmap="viridis")
    plt.colorbar(contour, label="Predicted hazard")
    plt.xlabel("Time from partner onset (s)")
    plt.ylabel("Information rate z")
    plt.title("Timing x Information-Rate Interaction Surface")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _write_interaction_figures(
    *,
    riskset: pd.DataFrame,
    selected_lag_ms: int,
    fitted: FittedFinalModel,
    figures_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    try:
        onset_grid = _interaction_plot_grid(
            riskset,
            selected_lag_ms=selected_lag_ms,
            varying_column="time_from_partner_onset_s",
            fixed_column="time_from_partner_offset_s",
        )
        files["timing_information_rate_interaction_onset"] = str(
            _save_interaction_line_figure(
                path=figures_dir / "timing_information_rate_interaction_onset.png",
                grid=onset_grid,
                x_column="time_from_partner_onset_s",
                fitted=fitted,
                title="Predicted hazard by onset timing and information rate",
                xlabel="Time from partner onset (s)",
            )
        )
    except Exception as exc:
        LOGGER.warning("Could not produce onset interaction plot: %s", exc)
    try:
        offset_grid = _interaction_plot_grid(
            riskset,
            selected_lag_ms=selected_lag_ms,
            varying_column="time_from_partner_offset_s",
            fixed_column="time_from_partner_onset_s",
        )
        files["timing_information_rate_interaction_offset"] = str(
            _save_interaction_line_figure(
                path=figures_dir / "timing_information_rate_interaction_offset.png",
                grid=offset_grid,
                x_column="time_from_partner_offset_s",
                fitted=fitted,
                title="Predicted hazard by offset timing and information rate",
                xlabel="Time from partner offset (s)",
            )
        )
    except Exception as exc:
        LOGGER.warning("Could not produce offset interaction plot: %s", exc)
    try:
        files["timing_information_rate_interaction_surface"] = str(
            _save_interaction_surface_figure(
                path=figures_dir / "timing_information_rate_interaction_surface.png",
                riskset=riskset,
                selected_lag_ms=selected_lag_ms,
                fitted=fitted,
            )
        )
    except Exception as exc:
        LOGGER.warning("Could not produce timing-information-rate interaction surface plot: %s", exc)
    return files


def _pooled_lag_sweep(combined: pd.DataFrame, lag_grid_ms: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag in lag_grid_ms:
        formula = _interaction_formula(lag)
        fitted = _fit_final_model(combined, f"pooled_lag_{lag}", formula)
        rows.append(
            {
                "lag_ms": int(lag),
                "bic": fitted.bic,
                "aic": fitted.aic,
                "stable": fitted.stable,
                "information_rate_interaction_coef": _coef(fitted.result, f"anchor_type[T.fpp]:{lag_col_z('information_rate', lag)}"),
                "prop_expected_interaction_coef": _coef(fitted.result, f"anchor_type[T.fpp]:{lag_col_z('prop_expected_cumulative_info', lag)}"),
            }
        )
    return pd.DataFrame(rows).sort_values(["lag_ms"], kind="mergesort").reset_index(drop=True)


def _save_compare_qc_plots(
    *,
    combined: pd.DataFrame,
    fpp: pd.DataFrame,
    spp: pd.DataFrame,
    selected_lag_ms: int,
    lag_grid_ms: list[int],
    contrasts: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    files: dict[str, str] = {}
    pooled_lag_table = _pooled_lag_sweep(combined, lag_grid_ms)
    pooled_lag_table.to_csv(out_dir / "pooled_lag_sweep.csv", index=False)
    selected_bic = float(pooled_lag_table.loc[pooled_lag_table["lag_ms"] == selected_lag_ms, "bic"].iloc[0])
    files["fpp_vs_spp_delta_bic_by_lag"] = str(_save_line_plot(
        out_dir / "fpp_vs_spp_delta_bic_by_lag.png",
        pooled_lag_table["lag_ms"].to_numpy(),
        (pooled_lag_table["bic"] - selected_bic).to_numpy(),
        xlabel="Lag (ms)",
        ylabel="Delta BIC vs selected lag",
        title="Pooled FPP-vs-SPP Delta BIC by Lag",
        vline=float(selected_lag_ms),
    ))

    plt.figure(figsize=(6, 4))
    plt.plot(pooled_lag_table["lag_ms"], pooled_lag_table["information_rate_interaction_coef"], marker="o", label="Information rate")
    plt.plot(pooled_lag_table["lag_ms"], pooled_lag_table["prop_expected_interaction_coef"], marker="o", label="Prop expected cumulative info")
    plt.axvline(selected_lag_ms, color="tab:red", linestyle="--", linewidth=1.0)
    plt.xlabel("Lag (ms)")
    plt.ylabel("Interaction coefficient")
    plt.title("FPP-vs-SPP Information Coefficients by Lag")
    plt.legend(frameon=False)
    plt.tight_layout()
    coeff_path = out_dir / "fpp_vs_spp_information_coefficients_by_lag.png"
    plt.savefig(coeff_path, dpi=180)
    plt.close()
    files["fpp_vs_spp_information_coefficients_by_lag"] = str(coeff_path)

    selected_info = lag_col_z("information_rate", selected_lag_ms)
    obs = combined.copy()
    obs["quantile"] = pd.qcut(obs[selected_info], q=5, duplicates="drop")
    event_rates = (
        obs.groupby(["anchor_type", "quantile"], observed=True)["event_bin"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(7, 4))
    for anchor_name in ("fpp", "spp"):
        subset = event_rates.loc[event_rates["anchor_type"].astype(str) == anchor_name]
        plt.plot(range(len(subset)), subset["event_bin"], marker="o", label=anchor_name.upper())
    plt.xlabel("Information-rate quantile")
    plt.ylabel("Observed event rate")
    plt.title("Observed Event Rate by Information Quantile")
    plt.legend(frameon=False)
    plt.tight_layout()
    obs_path = out_dir / "observed_event_rate_by_information_quantile.png"
    plt.savefig(obs_path, dpi=180)
    plt.close()
    files["observed_event_rate_by_information_quantile"] = str(obs_path)

    plt.figure(figsize=(7, 4))
    y_positions = np.arange(len(contrasts))[::-1]
    plt.errorbar(
        contrasts["estimate"],
        y_positions,
        xerr=1.96 * contrasts["standard_error"],
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        capsize=3,
    )
    plt.axvline(0.0, color="black", linewidth=1.0)
    plt.yticks(y_positions, contrasts["contrast"])
    plt.xlabel("Estimate")
    plt.title("Information Effect Contrasts")
    plt.tight_layout()
    forest_path = out_dir / "information_effect_contrasts_forest.png"
    plt.savefig(forest_path, dpi=180)
    plt.close()
    files["information_effect_contrasts_forest"] = str(forest_path)

    fpp_models = _fit_anchor_models_for_qc(fpp, selected_lag_ms)
    spp_models = _fit_anchor_models_for_qc(spp, selected_lag_ms)
    plt.figure(figsize=(6, 5))
    for anchor_name, table, models in (
        ("FPP", fpp, fpp_models),
        ("SPP", spp, spp_models),
    ):
        timing_bins = _calibration_bins(table["event_bin"], _predict_probabilities(table, models["timing_only"]))
        full_bins = _calibration_bins(table["event_bin"], _predict_probabilities(table, models["full_information"]))
        plt.plot(timing_bins["mean_predicted"], timing_bins["mean_observed"], marker="o", label=f"{anchor_name} timing")
        plt.plot(full_bins["mean_predicted"], full_bins["mean_observed"], marker="s", label=f"{anchor_name} full")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Mean predicted hazard")
    plt.ylabel("Observed event rate")
    plt.title("Calibration: Timing vs Full")
    plt.legend(frameon=False)
    plt.tight_layout()
    calib_path = out_dir / "calibration_timing_vs_full_fpp_spp.png"
    plt.savefig(calib_path, dpi=180)
    plt.close()
    files["calibration_timing_vs_full_fpp_spp"] = str(calib_path)

    prop_selected = lag_col_z("prop_expected_cumulative_info", selected_lag_ms)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(
        float(min(combined[selected_info].min(), combined[prop_selected].min())),
        float(max(combined[selected_info].max(), combined[prop_selected].max())),
        30,
    )
    for anchor_name, color in (("fpp", "tab:blue"), ("spp", "tab:orange")):
        subset = combined.loc[combined["anchor_type"].astype(str) == anchor_name]
        plt.hist(subset[selected_info], bins=bins, alpha=0.35, label=f"{anchor_name.upper()} information_rate", color=color)
    plt.xlabel("Selected-lag z-scored information predictor")
    plt.ylabel("Count")
    plt.title("Information Predictor Distributions")
    plt.legend(frameon=False)
    plt.tight_layout()
    dist_path = out_dir / "information_predictor_distributions_fpp_spp.png"
    plt.savefig(dist_path, dpi=180)
    plt.close()
    files["information_predictor_distributions_fpp_spp"] = str(dist_path)

    plt.figure(figsize=(7, 4))
    event_values = combined.loc[combined["event_bin"] == 1, selected_info]
    nonevent_values = combined.loc[combined["event_bin"] == 0, selected_info]
    bins2 = np.linspace(float(combined[selected_info].min()), float(combined[selected_info].max()), 30)
    plt.hist(nonevent_values, bins=bins2, alpha=0.5, label="Non-event", color="tab:gray")
    plt.hist(event_values, bins=bins2, alpha=0.5, label="Event", color="tab:red")
    plt.xlabel("Selected-lag information_rate_z")
    plt.ylabel("Count")
    plt.title("Event vs Non-event Information Distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    ev_path = out_dir / "event_vs_nonevent_information_distribution.png"
    plt.savefig(ev_path, dpi=180)
    plt.close()
    files["event_vs_nonevent_information_distribution"] = str(ev_path)
    return files


def build_fpp_vs_spp_report(
    *,
    cfg: FinalBehaviorConfig,
    selected_lag_ms: int,
    fpp_summary: pd.DataFrame,
    spp_summary: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    pooled_coefficients: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    fpp_rows = int(pd.to_numeric(fpp_summary["n_rows"], errors="coerce").max())
    fpp_events = int(pd.to_numeric(fpp_summary["n_events"], errors="coerce").max())
    spp_rows = int(pd.to_numeric(spp_summary["n_rows"], errors="coerce").max())
    spp_events = int(pd.to_numeric(spp_summary["n_events"], errors="coerce").max())
    pooled_row = pooled_summary.iloc[0].to_dict()
    warnings_list: list[str] = []
    for frame in (fpp_summary, spp_summary, pooled_summary):
        for warning_text in frame.get("safety_warnings", pd.Series(dtype=str)).fillna("").astype(str):
            if warning_text:
                warnings_list.append(warning_text)
    warnings_list = sorted(dict.fromkeys(warnings_list))
    spp_stable = bool(pd.Series(spp_summary["stable"]).fillna(False).all())
    interpretable = _is_interpretable(contrasts, pooled_summary, spp_summary)
    conclusion = (
        "SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins."
        if spp_events == 0
        else _conclusion_text(contrasts)
    )
    if spp_events == 0:
        interpretable = False

    md_lines = [
        "# Behavioral Final Hazard Report",
        "",
        f"- Selected lag: {selected_lag_ms} ms",
        f"- Bin size: {cfg.bin_size_ms} ms",
        f"- FPP rows/events: {fpp_rows}/{fpp_events}",
        f"- SPP rows/events: {spp_rows}/{spp_events}",
        "- Primary timing control: linear/quadratic parametric timing using z-scored onset, z-scored offset, and squared z-scored offset.",
        "- FPP and SPP use the same primary timing controls.",
        "- Lag policy: FPP selected the shared lag by minimum BIC; SPP did not reselect lag.",
        f"- SPP models stable: {spp_stable}",
        f"- Pooled FPP-vs-SPP contrasts interpretable: {interpretable}",
        "",
        "## Formulas",
        "",
        f"- Timing-only / per-anchor baseline: `{_timing_terms(cfg)}`",
        f"- FPP and SPP final models: `{_formula_sequence(selected_lag_ms, cfg)['full_information']}`",
        f"- Pooled interaction model: `{_interaction_formula(selected_lag_ms, cfg)}`",
        "",
        "## Diagnostics",
        "",
        f"- Pooled model converged: {bool(pooled_row['converged'])}",
        f"- Pooled model stable: {bool(pooled_row['stable'])}",
        f"- Model warnings: {' | '.join(warnings_list) if warnings_list else 'none'}",
        "",
        "## Per-Anchor Model Comparison",
        "",
        f"- FPP best BIC model: {str(fpp_summary.sort_values(['bic', 'model_name'], kind='mergesort').iloc[0]['model_name'])}",
        f"- SPP best BIC model: {str(spp_summary.sort_values(['bic', 'model_name'], kind='mergesort').iloc[0]['model_name'])}",
        "",
        "## Pooled Interaction Results",
        "",
    ]
    for _, row in contrasts.iterrows():
        md_lines.append(
            f"- {row['contrast']}: estimate={row['estimate']:.4f}, SE={row['standard_error']:.4f}, "
            f"z={row['z']:.4f}, p={row['p_value']:.4g}, 95% CI [{row['conf_low']:.4f}, {row['conf_high']:.4f}]"
        )
    md_lines.extend(["", "## Conclusion", "", conclusion])

    report_json = {
        "selected_lag_ms": int(selected_lag_ms),
        "bin_size_ms": int(cfg.bin_size_ms),
        "lag_selection_policy": "FPP selected one shared lag by minimum BIC; SPP did not reselect lag.",
        "primary_timing_rationale": "Primary behavioural-final models use a shared linear/quadratic timing baseline for FPP and SPP.",
        "formulas": {
            "timing_terms": _timing_terms(cfg),
            "full_information": _formula_sequence(selected_lag_ms, cfg)["full_information"],
            "pooled_interaction": _interaction_formula(selected_lag_ms, cfg),
        },
        "counts": {
            "fpp_rows": fpp_rows,
            "fpp_events": fpp_events,
            "spp_rows": spp_rows,
            "spp_events": spp_events,
        },
        "convergence": {
            "fpp_all_converged": bool(pd.Series(fpp_summary["converged"]).all()),
            "spp_all_converged": bool(pd.Series(spp_summary["converged"]).all()),
            "pooled_converged": bool(pooled_row["converged"]),
        },
        "stability": {
            "fpp_all_stable": bool(pd.Series(fpp_summary["stable"]).fillna(False).all()),
            "spp_all_stable": spp_stable,
            "pooled_stable": bool(pooled_row["stable"]),
            "pooled_contrasts_interpretable": interpretable,
        },
        "model_warnings": warnings_list,
        "per_anchor_model_summary": {
            "fpp": fpp_summary.to_dict(orient="records"),
            "spp": spp_summary.to_dict(orient="records"),
        },
        "pooled_model_summary": pooled_summary.to_dict(orient="records"),
        "pooled_coefficients": pooled_coefficients.to_dict(orient="records"),
        "information_effect_contrasts": contrasts.to_dict(orient="records"),
        "conclusion": conclusion,
    }
    return "\n".join(md_lines), report_json


def run_behavior_final_select_lag(config_path: Path, out_dir: Path, *, verbose: bool = False) -> Path:
    _configure_logging(verbose=verbose)
    cfg = load_final_behavior_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running final lag selection with config: %s", config_path)

    fpp, _episodes, warnings_list = _build_anchor_riskset(cfg, "fpp", out_dir)
    fpp, _std = _prepare_final_riskset(fpp, cfg.lag_grid_ms)

    selection_model = str(cfg.raw["lags"]["selection_model"])
    selection_against = str(cfg.raw["lags"]["selection_against"])
    baseline_formula = _formula_sequence(0, cfg)[selection_against]
    baseline = _fit_final_model(fpp, selection_against, baseline_formula)
    rows: list[dict[str, Any]] = []
    for lag in progress_iterable(
        cfg.lag_grid_ms,
        total=len(cfg.lag_grid_ms),
        description="Lag sweep (FPP)",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        r_col = lag_col_z("information_rate", lag)
        p_col = lag_col_z("prop_expected_cumulative_info", lag)
        formula = _formula_sequence(lag, cfg)[selection_model]
        fitted = _fit_final_model(fpp, selection_model, formula)
        rows.append(_lag_selection_row(fitted, lag_ms=lag, baseline=baseline, r_col=r_col, p_col=p_col))

    table = pd.DataFrame(rows).sort_values(["lag_ms"], kind="mergesort").reset_index(drop=True)
    selected_row = table.sort_values(["bic", "lag_ms"], kind="mergesort").iloc[0]
    selected_lag_ms = int(selected_row["lag_ms"])
    table.to_csv(out_dir / "fpp_lag_selection_table.csv", index=False)
    _write_json(out_dir / "selected_lag.json", _selected_lag_payload(selected_lag_ms))
    plot_files = _save_lag_selection_plots(table, out_dir, selected_lag_ms)
    _write_json(out_dir / "qc_plot_manifest.json", plot_files)
    _write_json(
        out_dir.parent / "warnings.json",
        {
            "warnings": warnings_list,
            "lag_selection_warnings": baseline.safety_warnings,
        },
    )
    return out_dir / "selected_lag.json"


def run_behavior_final_fit(
    config_path: Path,
    anchor_type: str,
    selected_lag_json: Path,
    out_dir: Path,
    *,
    verbose: bool = False,
) -> Path:
    _configure_logging(verbose=verbose)
    cfg = load_final_behavior_config(config_path)
    selected_lag = int(json.loads(selected_lag_json.read_text(encoding="utf-8"))["selected_lag_ms"])
    LOGGER.info("Fitting final model sequence for anchor=%s at selected lag=%d ms", anchor_type, selected_lag)

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    riskset, episodes, warnings_list = _build_anchor_riskset(cfg, anchor_type, out_dir)
    riskset, standardization = _prepare_final_riskset(riskset, cfg.lag_grid_ms)
    _guard_nonzero_event_bins_for_modeling(riskset, anchor=anchor_type)
    riskset.to_parquet(out_dir / "riskset.parquet", index=False)
    episodes.to_csv(out_dir / "episodes.csv", index=False)
    standardization.to_csv(out_dir / "standardization_stats.csv", index=False)

    sequence_specs = _sequence_spec(cfg, selected_lag)
    formulas = {spec["name"]: spec["formula"] for spec in sequence_specs}
    fitted_models: dict[str, FittedFinalModel] = {}
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    model_names = tuple(spec["name"] for spec in sequence_specs)
    for model_name in progress_iterable(
        model_names,
        total=len(model_names),
        description=f"Fit models ({anchor_type})",
        enabled=LOGGER.isEnabledFor(logging.INFO),
    ):
        fitted = _fit_final_model(riskset, model_name, formulas[model_name])
        fitted_models[model_name] = fitted
        summary_rows.append(_summary_row(fitted, anchor_type=anchor_type))
        coefficient_rows.extend(_coefficient_rows(fitted, anchor_type=anchor_type))

    summary_table = pd.DataFrame(summary_rows)
    coefficients_table = pd.DataFrame(coefficient_rows)
    comparison_table = pd.DataFrame(_compare_rows(fitted_models, anchor_type=anchor_type, selected_lag_ms=selected_lag, cfg=cfg))
    summary_table.to_csv(models_dir / "model_summary.csv", index=False)
    coefficients_table.to_csv(models_dir / "coefficients.csv", index=False)
    comparison_table.to_csv(models_dir / "model_comparison.csv", index=False)
    for model_name, fitted in fitted_models.items():
        summary_table.loc[summary_table["model_name"].astype(str) == model_name].to_csv(models_dir / f"{model_name}_summary.csv", index=False)
        coefficients_table.loc[coefficients_table["model_name"].astype(str) == model_name].to_csv(
            models_dir / f"{model_name}_coefficients.csv",
            index=False,
        )
    interaction_name = "timing_information_rate_interaction"
    interaction_base = str(
        next(
            (spec.get("compare_against") or "full_information")
            for spec in sequence_specs
            if str(spec["name"]) == interaction_name
        )
    )
    interaction_comparison = comparison_table.loc[
        (comparison_table["child_model"].astype(str) == interaction_name)
        & (comparison_table["parent_model"].astype(str) == interaction_base)
    ].copy()
    if not interaction_comparison.empty:
        interaction_comparison.to_csv(models_dir / "timing_information_rate_interaction_comparison.csv", index=False)
    figure_manifest: dict[str, str] = {}
    if interaction_name in fitted_models:
        figure_manifest = _write_interaction_figures(
            riskset=riskset,
            selected_lag_ms=selected_lag,
            fitted=fitted_models[interaction_name],
            figures_dir=figures_dir,
        )
        _write_json(figures_dir / "timing_information_rate_interaction_manifest.json", figure_manifest)

    config_snapshot = out_dir.parent / "config_snapshot.yaml"
    if not config_snapshot.exists():
        config_snapshot.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(
        out_dir / "diagnostics.json",
        {
            "anchor_type": anchor_type,
            "selected_lag_ms": selected_lag,
            "lag_reselection_performed": False,
            "note": "SPP reuses the FPP-selected lag without reselection." if anchor_type == "spp" else "FPP final fitting uses the frozen shared lag selected upstream.",
            "warnings": warnings_list,
            "model_safety_warnings": {fitted.model_name: fitted.safety_warnings for fitted in fitted_models.values()},
            "interaction_figures": figure_manifest,
        },
    )
    _write_json(
        out_dir.parent / "warnings.json",
        {
            "warnings": warnings_list,
            "anchor_type": anchor_type,
            "model_safety_warnings": {fitted.model_name: fitted.safety_warnings for fitted in fitted_models.values()},
        },
    )
    return models_dir / "model_summary.csv"


def run_behavior_final_compare(
    config_path: Path,
    selected_lag_json: Path,
    fpp_riskset_path: Path,
    spp_riskset_path: Path,
    out_dir: Path,
    *,
    verbose: bool = False,
) -> Path:
    _configure_logging(verbose=verbose)
    cfg = load_final_behavior_config(config_path)
    selected_lag = int(json.loads(selected_lag_json.read_text(encoding="utf-8"))["selected_lag_ms"])
    LOGGER.info("Fitting pooled FPP-vs-SPP model at selected lag=%d ms", selected_lag)

    fpp = pd.read_parquet(fpp_riskset_path)
    spp = pd.read_parquet(spp_riskset_path)
    fpp = _validate_anchor_labels(fpp, expected_anchor="fpp")
    spp = _validate_anchor_labels(spp, expected_anchor="spp")
    for column_name in REQUIRED_FINAL_COLUMNS + list(TIMING_COLUMNS):
        if column_name not in fpp.columns or column_name not in spp.columns:
            raise ValueError(f"Missing required column in pooled compare inputs: {column_name}")
    _guard_nonzero_event_bins_for_modeling(fpp, anchor="fpp")
    _guard_nonzero_event_bins_for_modeling(spp, anchor="spp")
    spp_events = int(pd.to_numeric(spp["event_bin"], errors="coerce").sum())
    if int(len(spp)) > 0 and spp_events == 0:
        raise ValueError(
            "SPP negative-control comparison could not be evaluated because the SPP risk set contained zero event bins."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([fpp, spp], ignore_index=True, sort=False)
    combined["anchor_type"] = pd.Categorical(combined["anchor_type"].astype(str), categories=["spp", "fpp"])
    combined.to_parquet(out_dir / "combined_riskset.parquet", index=False)

    formula = _interaction_formula(selected_lag, cfg)
    pooled = _fit_final_model(combined, "fpp_vs_spp_interaction", formula)
    pooled_summary = pd.DataFrame([_summary_row(pooled, anchor_type="pooled_fpp_spp")])
    pooled_summary.to_csv(out_dir / "interaction_model_summary.csv", index=False)

    pooled_coefficients = pd.DataFrame(_coefficient_rows(pooled, anchor_type="pooled_fpp_spp"))
    pooled_coefficients.to_csv(out_dir / "interaction_coefficients.csv", index=False)

    covariance = pd.DataFrame(
        pooled.result.cov_params(),
        index=pooled.result.params.index,
        columns=pooled.result.params.index,
    )
    r_col = lag_col_z("information_rate", selected_lag)
    p_col = lag_col_z("prop_expected_cumulative_info", selected_lag)
    contrasts = build_information_effect_contrasts(
        pooled_coefficients,
        covariance,
        info_rate_col=r_col,
        prop_col=p_col,
    )
    contrasts.to_csv(out_dir / "information_effect_contrasts.csv", index=False)

    interaction_cfg = ((cfg.raw.get("models") or {}).get("timing_information_rate_interaction") or {})
    pooled_interaction_files: dict[str, str] = {}
    if bool(((interaction_cfg.get("pooled_anchor_comparison") or {}).get("enabled"))):
        pooled_interaction_formula = _pooled_timing_information_rate_interaction_formula(selected_lag, cfg)
        pooled_interaction = _fit_final_model(combined, "timing_information_rate_anchor_interaction", pooled_interaction_formula)
        pooled_interaction_summary = pd.DataFrame([_summary_row(pooled_interaction, anchor_type="pooled_fpp_spp")])
        pooled_interaction_summary.to_csv(out_dir / "timing_information_rate_anchor_interaction_summary.csv", index=False)
        pooled_interaction_coefficients = pd.DataFrame(_coefficient_rows(pooled_interaction, anchor_type="pooled_fpp_spp"))
        pooled_interaction_coefficients.to_csv(out_dir / "timing_information_rate_anchor_interaction_coefficients.csv", index=False)
        pooled_interaction_contrasts = build_timing_information_anchor_contrasts(
            pooled_interaction_coefficients,
            info_rate_col=r_col,
        )
        pooled_interaction_contrasts.to_csv(out_dir / "timing_information_rate_anchor_interaction_contrasts.csv", index=False)
        pooled_interaction_files = {
            "summary": str(out_dir / "timing_information_rate_anchor_interaction_summary.csv"),
            "coefficients": str(out_dir / "timing_information_rate_anchor_interaction_coefficients.csv"),
            "contrasts": str(out_dir / "timing_information_rate_anchor_interaction_contrasts.csv"),
        }

    fpp_summary = pd.read_csv(fpp_riskset_path.parent / "models" / "model_summary.csv")
    spp_summary = pd.read_csv(spp_riskset_path.parent / "models" / "model_summary.csv")
    report_md, report_json = build_fpp_vs_spp_report(
        cfg=cfg,
        selected_lag_ms=selected_lag,
        fpp_summary=fpp_summary,
        spp_summary=spp_summary,
        pooled_summary=pooled_summary,
        pooled_coefficients=pooled_coefficients,
        contrasts=contrasts,
    )
    _write_markdown(out_dir / "fpp_vs_spp_report.md", report_md)
    _write_json(out_dir / "fpp_vs_spp_report.json", report_json)

    qc_files = _save_compare_qc_plots(
        combined=combined,
        fpp=fpp,
        spp=spp,
        selected_lag_ms=selected_lag,
        lag_grid_ms=cfg.lag_grid_ms,
        contrasts=contrasts,
        out_dir=out_dir,
    )
    if pooled_interaction_files:
        qc_files.update({f"pooled_timing_information_rate_{key}": value for key, value in pooled_interaction_files.items()})
    _write_json(out_dir / "qc_plot_manifest.json", qc_files)
    return out_dir / "interaction_model_summary.csv"


def _configure_logging(*, verbose: bool) -> None:
    if not verbose:
        return
    if LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
