"""Final diagnostics pass for SPP timing-only hazard failures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import textwrap
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from cas.hazard_behavior.diagnose_spp_neural_failure import sample_diagnostic_fit_rows
from cas.hazard_behavior.identity import ensure_participant_speaker_id, validate_participant_speaker_id
from cas.hazard_behavior.io import write_table

DEFAULT_OUTPUT_DIR = Path("results/hazard_behavior/neural_lowlevel/diagnostics/spp_timing_debug")
DEFAULT_RISKSET_PATH = Path("/Users/hiro/Datasets/working/cas/reports/hazard_neural_fpp/riskset/neural_spp_hazard_table.parquet")
DEFAULT_FPP_RISKSET_PATH = Path("/Users/hiro/Datasets/working/cas/reports/hazard_neural_fpp/riskset/neural_fpp_hazard_table.parquet")

OUTPUT_FILENAMES = {
    "report": "spp_timing_debug_report.md",
    "variable_summary": "spp_timing_variable_summary.csv",
    "event_consistency": "spp_timing_event_consistency_checks.csv",
    "event_vs_nonevent_bins": "spp_timing_event_vs_nonevent_bins.csv",
    "duplicate_check": "spp_timing_duplicate_check.csv",
    "spline_design": "spp_timing_spline_design_check.csv",
    "sampling_check": "spp_timing_sampling_check.csv",
    "simple_fit_check": "spp_timing_simple_fit_check.csv",
    "ridge_fit_check": "spp_timing_ridge_fit_check.csv",
    "event_histograms": "spp_timing_event_histograms.png",
    "event_vs_nonevent_density": "spp_timing_event_vs_nonevent_time_density.png",
    "event_rate_by_time_bin": "spp_timing_event_rate_by_time_bin.png",
    "spline_basis_by_time": "spp_timing_spline_basis_by_time.png",
    "sampled_vs_full_nonevent_time": "spp_timing_sampled_vs_full_nonevent_time.png",
    "fpp_vs_spp_time_histograms": "spp_timing_fpp_vs_spp_time_histograms.png",
}

EVENT_TIME_VARIABLES = ("time_from_partner_onset", "time_from_partner_offset")
SUMMARY_TIME_VARIABLES = ("time_from_partner_onset", "time_from_partner_offset", "bin_end", "partner_ipu_duration")
DUPLICATE_KEY_COLUMNS = ("dyad_id", "run", "speaker", "episode_id", "bin_start", "bin_end")
TIME_TOLERANCE = 1.0e-6


@dataclass(frozen=True, slots=True)
class SppTimingDebugResult:
    output_dir: Path
    report_path: Path


def diagnose_spp_timing_failure(
    *,
    riskset_path: Path = DEFAULT_RISKSET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fpp_riskset_path: Path | None = DEFAULT_FPP_RISKSET_PATH,
    max_fit_non_event_rows: int = 100_000,
    bin_width_s: float = 0.05,
) -> SppTimingDebugResult:
    """Run a final diagnostic pass on the SPP timing-only hazard failure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / OUTPUT_FILENAMES["report"]
    notes: list[str] = []

    riskset = _load_riskset(riskset_path, event_column="event_spp")
    if riskset is None:
        notes.append(f"SPP riskset was unavailable: {riskset_path}")
        _write_empty_outputs(output_dir)
        report_path.write_text(_build_missing_input_report(notes), encoding="utf-8")
        return SppTimingDebugResult(output_dir=output_dir, report_path=report_path)

    event_column = "event_spp"
    riskset = _prepare_identity_columns(riskset)
    onset_column = _resolve_event_onset_column(riskset)
    consistency = compute_event_consistency_checks(riskset, event_column=event_column, onset_column=onset_column)
    variable_summary = summarize_timing_variables(riskset, event_column=event_column)
    overlap_summary, binned_overlap = compute_event_nonevent_overlap(
        riskset,
        event_column=event_column,
        bin_width_s=float(bin_width_s),
    )
    duplicate_check = compute_duplicate_check(riskset)
    fit_sample = sample_diagnostic_fit_rows(
        riskset,
        event_column=event_column,
        max_non_event_rows=int(max_fit_non_event_rows),
    )
    sampling_check = compute_sampling_check(
        riskset,
        fit_sample=fit_sample,
        event_column=event_column,
        bin_width_s=float(bin_width_s),
    )
    design_check = build_spline_design_check(
        riskset,
        fit_sample=fit_sample,
        event_column=event_column,
    )
    simple_fit_check = run_simple_timing_fit_checks(
        riskset,
        fit_sample=fit_sample,
        event_column=event_column,
    )
    ridge_fit_check = run_ridge_timing_diagnostics(
        fit_sample,
        event_column=event_column,
    )

    fpp_riskset = _load_riskset(fpp_riskset_path, event_column="event_fpp") if fpp_riskset_path else None
    fpp_summary = summarize_event_time_distribution(fpp_riskset, event_column="event_fpp") if fpp_riskset is not None else pd.DataFrame()
    spp_summary = summarize_event_time_distribution(riskset, event_column=event_column)

    write_table(variable_summary, output_dir / OUTPUT_FILENAMES["variable_summary"], sep=",")
    write_table(consistency, output_dir / OUTPUT_FILENAMES["event_consistency"], sep=",")
    write_table(binned_overlap, output_dir / OUTPUT_FILENAMES["event_vs_nonevent_bins"], sep=",")
    write_table(duplicate_check, output_dir / OUTPUT_FILENAMES["duplicate_check"], sep=",")
    write_table(design_check, output_dir / OUTPUT_FILENAMES["spline_design"], sep=",")
    write_table(sampling_check, output_dir / OUTPUT_FILENAMES["sampling_check"], sep=",")
    write_table(simple_fit_check, output_dir / OUTPUT_FILENAMES["simple_fit_check"], sep=",")
    write_table(ridge_fit_check, output_dir / OUTPUT_FILENAMES["ridge_fit_check"], sep=",")

    plot_event_histograms(riskset, event_column=event_column, output_path=output_dir / OUTPUT_FILENAMES["event_histograms"])
    plot_event_vs_nonevent_density(
        riskset,
        event_column=event_column,
        output_path=output_dir / OUTPUT_FILENAMES["event_vs_nonevent_density"],
    )
    plot_event_rate_by_time_bin(
        binned_overlap,
        output_path=output_dir / OUTPUT_FILENAMES["event_rate_by_time_bin"],
    )
    plot_spline_basis_by_time(
        riskset,
        fit_sample=fit_sample,
        output_path=output_dir / OUTPUT_FILENAMES["spline_basis_by_time"],
    )
    plot_sampled_vs_full_nonevent_time(
        riskset,
        fit_sample=fit_sample,
        event_column=event_column,
        output_path=output_dir / OUTPUT_FILENAMES["sampled_vs_full_nonevent_time"],
    )
    plot_fpp_vs_spp_time_histograms(
        fpp_riskset=fpp_riskset,
        spp_riskset=riskset,
        output_path=output_dir / OUTPUT_FILENAMES["fpp_vs_spp_time_histograms"],
    )

    report_payload = {
        "riskset_path": str(riskset_path),
        "fpp_riskset_path": None if fpp_riskset_path is None else str(fpp_riskset_path),
        "identity_validation": validate_participant_speaker_id(
            riskset,
            dyad_col="dyad_id",
            speaker_col="speaker",
            output_col="participant_speaker_id",
        ),
        "event_column": event_column,
        "event_onset_column": onset_column,
        "n_rows": int(len(riskset)),
        "n_events": int(pd.to_numeric(riskset[event_column], errors="coerce").fillna(0).sum()),
        "n_nonevents": int((pd.to_numeric(riskset[event_column], errors="coerce").fillna(0) == 0).sum()),
        "n_episodes": int(riskset["episode_id"].nunique()) if "episode_id" in riskset.columns else 0,
        "fit_sample_rows": int(len(fit_sample)),
        "fit_sample_events": int(pd.to_numeric(fit_sample[event_column], errors="coerce").fillna(0).sum()),
        "consistency_summary": _checks_to_dict(consistency),
        "overlap_summary": overlap_summary,
        "duplicate_summary": _duplicate_summary(duplicate_check),
        "design_summary": _design_summary(design_check),
        "simple_fit_summary": simple_fit_check.to_dict(orient="records"),
        "ridge_fit_summary": ridge_fit_check.to_dict(orient="records"),
        "fpp_event_summary": fpp_summary.to_dict(orient="records"),
        "spp_event_summary": spp_summary.to_dict(orient="records"),
    }
    report_path.write_text(build_spp_timing_debug_report(report_payload), encoding="utf-8")
    return SppTimingDebugResult(output_dir=output_dir, report_path=report_path)


def _load_riskset(path: Path | None, *, event_column: str) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    if Path(path).suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python")
    if event_column not in table.columns:
        return None
    return table


def _prepare_identity_columns(table: pd.DataFrame) -> pd.DataFrame:
    working = table.copy()
    if "speaker" not in working.columns and "participant_speaker" in working.columns:
        working["speaker"] = working["participant_speaker"].astype(str)
    working = ensure_participant_speaker_id(
        working,
        dyad_col="dyad_id",
        speaker_col="speaker",
        output_col="participant_speaker_id",
        overwrite="participant_speaker_id" not in working.columns,
    )
    return working


def _resolve_event_onset_column(table: pd.DataFrame) -> str | None:
    for candidate in ("event_onset", "own_fpp_onset", "spp_onset"):
        if candidate in table.columns and pd.to_numeric(table[candidate], errors="coerce").notna().any():
            return candidate
    return None


def compute_event_consistency_checks(
    table: pd.DataFrame,
    *,
    event_column: str,
    onset_column: str | None,
) -> pd.DataFrame:
    """Check event coding, timing identities, and onset alignment."""

    rows: list[dict[str, Any]] = []
    event_values = pd.to_numeric(table.get(event_column), errors="coerce")
    valid_binary = event_values.dropna().isin([0, 1]).all()
    rows.append(
        {
            "check_name": "event_column_validity",
            "status": "pass" if valid_binary else "fail",
            "n_rows_checked": int(len(table)),
            "n_violations": int((~event_values.dropna().isin([0, 1])).sum()) if not valid_binary else 0,
            "details": json.dumps(
                {
                    "unique_values": sorted(pd.Series(event_values.dropna().unique()).astype(float).tolist()),
                    "n_missing": int(event_values.isna().sum()),
                    "n_events": int(event_values.fillna(0).sum()),
                    "n_nonevents": int((event_values.fillna(0) == 0).sum()),
                    "event_rate": float(event_values.fillna(0).mean()) if len(event_values) > 0 else None,
                },
                sort_keys=True,
            ),
        }
    )

    if "episode_id" in table.columns:
        episode_counts = table.groupby("episode_id", sort=False)[event_column].sum(min_count=1).fillna(0)
        rows.append(
            {
                "check_name": "one_event_per_episode",
                "status": "pass" if int((episode_counts > 1).sum()) == 0 else "fail",
                "n_rows_checked": int(len(episode_counts)),
                "n_violations": int((episode_counts > 1).sum()),
                "details": json.dumps(
                    {
                        "n_episodes": int(len(episode_counts)),
                        "n_episodes_with_0_events": int((episode_counts == 0).sum()),
                        "n_episodes_with_1_event": int((episode_counts == 1).sum()),
                        "n_episodes_with_more_than_1_event": int((episode_counts > 1).sum()),
                        "max_events_per_episode": int(episode_counts.max()) if len(episode_counts) > 0 else 0,
                    },
                    sort_keys=True,
                ),
            }
        )

    if onset_column is not None:
        event_rows = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1].copy()
        onset = pd.to_numeric(event_rows[onset_column], errors="coerce")
        start = pd.to_numeric(event_rows["bin_start"], errors="coerce")
        end = pd.to_numeric(event_rows["bin_end"], errors="coerce")
        aligned = (start - TIME_TOLERANCE <= onset) & (onset <= end + TIME_TOLERANCE)
        violation_columns = [
            column_name
            for column_name in ("episode_id", "dyad_id", "run", "bin_start", "bin_end", onset_column)
            if column_name in event_rows.columns
        ]
        violations = event_rows.loc[~aligned, violation_columns].head(10)
        rows.append(
            {
                "check_name": "event_rows_align_with_onset_bin",
                "status": "pass" if bool(aligned.all()) else "fail",
                "n_rows_checked": int(len(event_rows)),
                "n_violations": int((~aligned).sum()),
                "details": violations.to_json(orient="records"),
            }
        )

    time_checks = {
        "bin_end_ge_bin_start": pd.to_numeric(table["bin_end"], errors="coerce") >= pd.to_numeric(table["bin_start"], errors="coerce") - TIME_TOLERANCE,
        "partner_ipu_offset_ge_onset": pd.to_numeric(table["partner_ipu_offset"], errors="coerce") >= pd.to_numeric(table["partner_ipu_onset"], errors="coerce") - TIME_TOLERANCE,
        "time_from_partner_onset_matches_bin_start": np.isclose(
            pd.to_numeric(table["time_from_partner_onset"], errors="coerce"),
            pd.to_numeric(table["bin_start"], errors="coerce") - pd.to_numeric(table["partner_ipu_onset"], errors="coerce"),
            atol=TIME_TOLERANCE,
            rtol=0.0,
            equal_nan=False,
        ),
        "time_from_partner_onset_matches_bin_end": np.isclose(
            pd.to_numeric(table["time_from_partner_onset"], errors="coerce"),
            pd.to_numeric(table["bin_end"], errors="coerce") - pd.to_numeric(table["partner_ipu_onset"], errors="coerce"),
            atol=TIME_TOLERANCE,
            rtol=0.0,
            equal_nan=False,
        ),
        "time_from_partner_offset_matches": np.isclose(
            pd.to_numeric(table["time_from_partner_offset"], errors="coerce"),
            pd.to_numeric(table["bin_end"], errors="coerce") - pd.to_numeric(table["partner_ipu_offset"], errors="coerce"),
            atol=TIME_TOLERANCE,
            rtol=0.0,
            equal_nan=False,
        ),
    }
    for check_name, mask in time_checks.items():
        rows.append(
            {
                "check_name": check_name,
                "status": "pass" if bool(np.asarray(mask).all()) else "fail",
                "n_rows_checked": int(len(table)),
                "n_violations": int((~np.asarray(mask)).sum()),
                "details": "",
            }
        )
    return pd.DataFrame(rows)


def summarize_timing_variables(table: pd.DataFrame, *, event_column: str) -> pd.DataFrame:
    """Summarize key timing variables by event status."""

    subsets = {
        "all_rows": table,
        "event_rows": table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1].copy(),
        "nonevent_rows": table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 0].copy(),
    }
    rows: list[dict[str, Any]] = []
    for subset_name, frame in subsets.items():
        for variable in SUMMARY_TIME_VARIABLES:
            if variable == "partner_ipu_duration":
                values = pd.to_numeric(frame["partner_ipu_offset"], errors="coerce") - pd.to_numeric(frame["partner_ipu_onset"], errors="coerce")
            else:
                if variable not in frame.columns:
                    continue
                values = pd.to_numeric(frame[variable], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "subset": subset_name,
                    "variable": variable,
                    **_series_summary(finite),
                    "n_missing": int(values.isna().sum()),
                    "n_infinite": int((~np.isfinite(values.fillna(np.nan))).sum() - values.isna().sum()),
                    "is_constant": bool(finite.nunique(dropna=True) <= 1) if len(finite) > 0 else False,
                }
            )
    return pd.DataFrame(rows)


def compute_event_nonevent_overlap(
    table: pd.DataFrame,
    *,
    event_column: str,
    bin_width_s: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Quantify event/non-event timing overlap and binned event rates."""

    event_mask = pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1
    event_rows = table.loc[event_mask].copy()
    nonevent_rows = table.loc[~event_mask].copy()
    summary: dict[str, Any] = {}
    binned_rows: list[dict[str, Any]] = []
    for variable in EVENT_TIME_VARIABLES:
        event_values = pd.to_numeric(event_rows[variable], errors="coerce").dropna()
        nonevent_values = pd.to_numeric(nonevent_rows[variable], errors="coerce").dropna()
        if event_values.empty or nonevent_values.empty:
            summary[variable] = {"status": "insufficient_data"}
            continue
        event_range = (float(event_values.min()), float(event_values.max()))
        nonevent_range = (float(nonevent_values.min()), float(nonevent_values.max()))
        inside_nonevent = ((event_values >= nonevent_range[0]) & (event_values <= nonevent_range[1])).mean()
        inside_event = ((nonevent_values >= event_range[0]) & (nonevent_values <= event_range[1])).mean()
        q1, q99 = np.quantile(event_values, [0.01, 0.99])
        q5, q95 = np.quantile(event_values, [0.05, 0.95])
        summary[variable] = {
            "event_min": event_range[0],
            "event_max": event_range[1],
            "nonevent_min": nonevent_range[0],
            "nonevent_max": nonevent_range[1],
            "fully_separated": bool(event_range[1] < nonevent_range[0] or nonevent_range[1] < event_range[0]),
            "proportion_events_inside_nonevent_range": float(inside_nonevent),
            "proportion_nonevents_inside_event_range": float(inside_event),
            "proportion_nonevents_within_event_1_99_percentile_range": float(((nonevent_values >= q1) & (nonevent_values <= q99)).mean()),
            "proportion_nonevents_within_event_5_95_percentile_range": float(((nonevent_values >= q5) & (nonevent_values <= q95)).mean()),
        }
        binned = bin_event_rate_by_variable(
            table,
            variable=variable,
            event_column=event_column,
            bin_width_s=float(bin_width_s),
        )
        binned_rows.extend(binned.to_dict(orient="records"))
    return summary, pd.DataFrame(binned_rows)


def bin_event_rate_by_variable(
    table: pd.DataFrame,
    *,
    variable: str,
    event_column: str,
    bin_width_s: float,
) -> pd.DataFrame:
    """Bin one timing variable and compute event rates."""

    working = table.loc[:, [variable, event_column]].copy()
    working[variable] = pd.to_numeric(working[variable], errors="coerce")
    working[event_column] = pd.to_numeric(working[event_column], errors="coerce").fillna(0).astype(int)
    working = working.loc[np.isfinite(working[variable])].copy()
    if working.empty:
        return pd.DataFrame(columns=["variable", "bin_left", "bin_right", "n_rows", "n_events", "event_rate"])
    min_value = float(working[variable].min())
    max_value = float(working[variable].max())
    start = np.floor(min_value / bin_width_s) * bin_width_s
    stop = np.ceil(max_value / bin_width_s) * bin_width_s + bin_width_s
    edges = np.arange(start, stop + 0.5 * bin_width_s, bin_width_s)
    if len(edges) < 2:
        edges = np.array([min_value, min_value + bin_width_s], dtype=float)
    working["bin_index"] = np.clip(np.digitize(working[variable], edges, right=False) - 1, 0, len(edges) - 2)
    grouped = working.groupby("bin_index", sort=True)
    rows = []
    for bin_index, frame in grouped:
        rows.append(
            {
                "variable": variable,
                "bin_left": float(edges[int(bin_index)]),
                "bin_right": float(edges[int(bin_index) + 1]),
                "n_rows": int(len(frame)),
                "n_events": int(frame[event_column].sum()),
                "event_rate": float(frame[event_column].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_duplicate_check(table: pd.DataFrame) -> pd.DataFrame:
    """Detect duplicate riskset rows under the likely unique key."""

    available_keys = [column for column in DUPLICATE_KEY_COLUMNS if column in table.columns]
    grouped = table.groupby(available_keys, dropna=False, sort=False)
    counts = grouped.size().reset_index(name="n_duplicate_rows")
    duplicate_keys = counts.loc[counts["n_duplicate_rows"] > 1].copy()
    if duplicate_keys.empty:
        return pd.DataFrame(
            [
                {
                    "row_type": "summary",
                    "n_rows": int(len(table)),
                    "n_duplicate_keys": 0,
                    "n_duplicate_event_rows": 0,
                    "duplicate_examples": "[]",
                }
            ]
        )
    merged = table.merge(duplicate_keys, on=available_keys, how="inner")
    merged["event_value"] = pd.to_numeric(merged.get("event_spp", merged.get("event")), errors="coerce").fillna(0).astype(int)
    examples = merged.loc[:, [*available_keys, "n_duplicate_rows", "event_value"]].head(10).to_dict(orient="records")
    return pd.DataFrame(
        [
            {
                "row_type": "summary",
                "n_rows": int(len(table)),
                "n_duplicate_keys": int(len(duplicate_keys)),
                "n_duplicate_event_rows": int(merged["event_value"].sum()),
                "duplicate_examples": json.dumps(examples, sort_keys=True),
            }
        ]
    )


def compute_sampling_check(
    table: pd.DataFrame,
    *,
    fit_sample: pd.DataFrame,
    event_column: str,
    bin_width_s: float,
) -> pd.DataFrame:
    """Compare the full and sampled nonevent timing distributions."""

    rows: list[dict[str, Any]] = []
    full_nonevents = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 0].copy()
    sampled_nonevents = fit_sample.loc[pd.to_numeric(fit_sample[event_column], errors="coerce").fillna(0) == 0].copy()
    events = fit_sample.loc[pd.to_numeric(fit_sample[event_column], errors="coerce").fillna(0) == 1].copy()
    for sample_name, frame in (
        ("full_nonevents", full_nonevents),
        ("sampled_nonevents", sampled_nonevents),
        ("events", events),
    ):
        for variable in EVENT_TIME_VARIABLES:
            values = pd.to_numeric(frame[variable], errors="coerce").dropna()
            rows.append(
                {
                    "sample_group": sample_name,
                    "variable": variable,
                    **_series_summary(values),
                }
            )
    overlap_summary, overlap_bins = compute_event_nonevent_overlap(
        pd.concat([sampled_nonevents, events], ignore_index=True, sort=False),
        event_column=event_column,
        bin_width_s=float(bin_width_s),
    )
    for variable, payload in overlap_summary.items():
        rows.append(
            {
                "sample_group": "sampled_vs_events_overlap",
                "variable": variable,
                "n": int(len(overlap_bins.loc[overlap_bins["variable"] == variable])),
                "mean": payload.get("proportion_nonevents_inside_event_range"),
                "sd": payload.get("proportion_nonevents_within_event_5_95_percentile_range"),
                "min": payload.get("nonevent_min"),
                "q001": None,
                "q01": None,
                "q05": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q95": None,
                "q99": None,
                "q999": None,
                "max": payload.get("nonevent_max"),
            }
        )
    return pd.DataFrame(rows)


def build_spline_design_check(
    table: pd.DataFrame,
    *,
    fit_sample: pd.DataFrame,
    event_column: str,
) -> pd.DataFrame:
    """Build spline design diagnostics for full and sampled timing-only fits."""

    formulas = {
        "timing_df6": "bs(time_from_partner_onset, df=6, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=6, degree=3, include_intercept=False)",
        "timing_df3": "bs(time_from_partner_onset, df=3, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=3, degree=3, include_intercept=False)",
    }
    rows: list[dict[str, Any]] = []
    for sample_type, frame in (("full", table), ("fit_sample", fit_sample)):
        for model_name, rhs in formulas.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matrix = dmatrix(rhs, data=frame, return_type="dataframe")
            rows.append(
                {
                    "sample_type": sample_type,
                    "model_name": model_name,
                    **compute_spline_design_diagnostic(matrix),
                }
            )
    return pd.DataFrame(rows)


def compute_spline_design_diagnostic(matrix: pd.DataFrame) -> dict[str, Any]:
    """Compute rank and stability diagnostics for one spline design matrix."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Design matrix must be 2D.")
    column_sd = values.std(axis=0, ddof=0)
    constant_columns = [str(column) for column, sd in zip(matrix.columns, column_sd, strict=True) if np.isclose(sd, 0.0)]
    near_constant_columns = [str(column) for column, sd in zip(matrix.columns, column_sd, strict=True) if sd < 1.0e-8]
    singular_values = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(values))
    max_sv = float(np.max(singular_values)) if singular_values.size else 0.0
    min_sv = float(np.min(singular_values)) if singular_values.size else 0.0
    condition_number = None if min_sv <= 0.0 else float(max_sv / min_sv)
    return {
        "n_rows": int(values.shape[0]),
        "n_columns": int(values.shape[1]),
        "rank": rank,
        "rank_deficiency": int(values.shape[1] - rank),
        "condition_number": condition_number,
        "constant_columns": json.dumps(constant_columns),
        "near_constant_columns": json.dumps(near_constant_columns),
        "min": float(np.nanmin(values)) if values.size else None,
        "max": float(np.nanmax(values)) if values.size else None,
        "n_nan": int(np.isnan(values).sum()),
        "n_inf": int(np.isinf(values).sum()),
    }


def run_simple_timing_fit_checks(
    table: pd.DataFrame,
    *,
    fit_sample: pd.DataFrame,
    event_column: str,
) -> pd.DataFrame:
    """Fit a sequence of increasingly flexible timing-only models."""

    formulas = [
        ("intercept_only", f"{event_column} ~ 1"),
        ("linear_onset_only", f"{event_column} ~ time_from_partner_onset"),
        ("linear_offset_only", f"{event_column} ~ time_from_partner_offset"),
        ("linear_onset_offset", f"{event_column} ~ time_from_partner_onset + time_from_partner_offset"),
        ("spline_df3_onset_only", f"{event_column} ~ bs(time_from_partner_onset, df=3, degree=3, include_intercept=False)"),
        ("spline_df3_offset_only", f"{event_column} ~ bs(time_from_partner_offset, df=3, degree=3, include_intercept=False)"),
        ("spline_df3_onset_offset", f"{event_column} ~ bs(time_from_partner_onset, df=3, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=3, degree=3, include_intercept=False)"),
        ("spline_df6_onset_offset", f"{event_column} ~ bs(time_from_partner_onset, df=6, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=6, degree=3, include_intercept=False)"),
    ]
    rows: list[dict[str, Any]] = []
    for sample_type, frame in (("full", table), ("fit_sample", fit_sample)):
        for model_name, formula in formulas:
            rows.append(simple_fit_row(frame, event_column=event_column, model_name=model_name, formula=formula, sample_type=sample_type))
    return pd.DataFrame(rows)


def simple_fit_row(
    table: pd.DataFrame,
    *,
    event_column: str,
    model_name: str,
    formula: str,
    sample_type: str,
) -> dict[str, Any]:
    """Fit one diagnostic GLM and return a robust summary row."""

    subset = table.loc[:, list({event_column, *[column for column in ("time_from_partner_onset", "time_from_partner_offset") if column in table.columns]})].copy()
    subset[event_column] = pd.to_numeric(subset[event_column], errors="coerce")
    if "time_from_partner_onset" in subset.columns:
        subset["time_from_partner_onset"] = pd.to_numeric(subset["time_from_partner_onset"], errors="coerce")
    if "time_from_partner_offset" in subset.columns:
        subset["time_from_partner_offset"] = pd.to_numeric(subset["time_from_partner_offset"], errors="coerce")
    subset = subset.dropna()
    n_events = int(subset[event_column].sum()) if not subset.empty else 0
    if subset.empty or n_events <= 0:
        return _simple_fit_failure_row(model_name, formula, sample_type, len(subset), n_events, "no_data", "No usable rows or events.")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = sm.GLM.from_formula(formula=formula, data=subset, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        fit_warnings = "; ".join(str(item.message) for item in caught)
        params = np.asarray(getattr(fitted, "params", []), dtype=float)
        bse = np.asarray(getattr(fitted, "bse", []), dtype=float)
        pathological = "overflow encountered in exp" in fit_warnings.lower() or "perfect separation" in fit_warnings.lower()
        converged = bool(getattr(fitted, "converged", True) and not pathological)
        status = "converged" if converged else "failed"
        return {
            "model_name": model_name,
            "sample_type": sample_type,
            "formula": formula,
            "n_rows": int(len(subset)),
            "n_events": n_events,
            "n_predictors": int(len(params)),
            "converged": converged,
            "status": status,
            "warnings": fit_warnings,
            "error_message": None if converged else "Model failed convergence checks.",
            "aic": _safe_float(getattr(fitted, "aic", None)),
            "bic": _safe_float(getattr(fitted, "bic", None)),
            "log_likelihood": _safe_float(getattr(fitted, "llf", None)),
            "max_abs_coefficient": float(np.max(np.abs(params))) if params.size else None,
            "max_standard_error": float(np.max(np.abs(bse))) if bse.size else None,
            "any_nan_coefficients": bool(np.isnan(params).any()) if params.size else False,
            "any_infinite_coefficients": bool(np.isinf(params).any()) if params.size else False,
        }
    except Exception as error:
        return _simple_fit_failure_row(model_name, formula, sample_type, len(subset), n_events, "failed", str(error))


def _simple_fit_failure_row(
    model_name: str,
    formula: str,
    sample_type: str,
    n_rows: int,
    n_events: int,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "sample_type": sample_type,
        "formula": formula,
        "n_rows": int(n_rows),
        "n_events": int(n_events),
        "n_predictors": 0,
        "converged": False,
        "status": status,
        "warnings": "",
        "error_message": error_message,
        "aic": None,
        "bic": None,
        "log_likelihood": None,
        "max_abs_coefficient": None,
        "max_standard_error": None,
        "any_nan_coefficients": None,
        "any_infinite_coefficients": None,
    }


def run_ridge_timing_diagnostics(
    table: pd.DataFrame,
    *,
    event_column: str,
) -> pd.DataFrame:
    """Fit ridge-logistic timing-only diagnostics on the sampled dataset."""

    specs = [
        ("ridge_linear_onset_offset", "time_from_partner_onset + time_from_partner_offset"),
        ("ridge_spline_df3_onset_offset", "bs(time_from_partner_onset, df=3, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=3, degree=3, include_intercept=False)"),
        ("ridge_spline_df6_onset_offset", "bs(time_from_partner_onset, df=6, degree=3, include_intercept=False) + bs(time_from_partner_offset, df=6, degree=3, include_intercept=False)"),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, rhs in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x = dmatrix(f"1 + {rhs}", data=table, return_type="dataframe")
            except Exception as error:
                rows.append({"model_name": model_name, "n_rows": 0, "n_events": 0, "n_predictors": 0, "penalty": "ridge", "C_or_lambda": 1.0, "converged": False, "train_log_loss": None, "train_brier": None, "train_auroc": None, "max_abs_coefficient": None, "error_message": str(error)})
                continue
        y_values = pd.to_numeric(table[event_column], errors="coerce").fillna(0).astype(int).to_numpy()
        x_values = np.asarray(x, dtype=float)
        if np.unique(y_values).size < 2:
            rows.append({"model_name": model_name, "n_rows": int(len(table)), "n_events": int(y_values.sum()), "n_predictors": int(x_values.shape[1]), "penalty": "ridge", "C_or_lambda": 1.0, "converged": False, "train_log_loss": None, "train_brier": None, "train_auroc": None, "max_abs_coefficient": None, "error_message": "Only one class present."})
            continue
        model = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000))])
        try:
            model.fit(x_values, y_values)
            prob = model.predict_proba(x_values)[:, 1]
            coef = np.asarray(model.named_steps["logit"].coef_, dtype=float)
            rows.append(
                {
                    "model_name": model_name,
                    "n_rows": int(len(table)),
                    "n_events": int(y_values.sum()),
                    "n_predictors": int(x_values.shape[1]),
                    "penalty": "ridge",
                    "C_or_lambda": 1.0,
                    "converged": True,
                    "train_log_loss": float(log_loss(y_values, prob, labels=[0, 1])),
                    "train_brier": float(brier_score_loss(y_values, prob)),
                    "train_auroc": float(roc_auc_score(y_values, prob)),
                    "max_abs_coefficient": float(np.max(np.abs(coef))) if coef.size else None,
                    "error_message": None,
                }
            )
        except Exception as error:
            rows.append({"model_name": model_name, "n_rows": int(len(table)), "n_events": int(y_values.sum()), "n_predictors": int(x_values.shape[1]), "penalty": "ridge", "C_or_lambda": 1.0, "converged": False, "train_log_loss": None, "train_brier": None, "train_auroc": None, "max_abs_coefficient": None, "error_message": str(error)})
    return pd.DataFrame(rows)


def summarize_event_time_distribution(table: pd.DataFrame | None, *, event_column: str) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()
    event_rows = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1].copy()
    rows = []
    for variable in EVENT_TIME_VARIABLES:
        values = pd.to_numeric(event_rows[variable], errors="coerce").dropna()
        rows.append({"variable": variable, **_series_summary(values)})
    return pd.DataFrame(rows)


def plot_event_histograms(table: pd.DataFrame, *, event_column: str, output_path: Path) -> None:
    event_rows = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, variable in zip(axes, EVENT_TIME_VARIABLES, strict=True):
        values = pd.to_numeric(event_rows[variable], errors="coerce").dropna()
        if values.empty:
            axis.text(0.5, 0.5, "No event rows available.", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.hist(values, bins=min(50, max(10, int(np.sqrt(len(values))))), color="#3b7ea1", alpha=0.85)
        axis.set_title(variable)
        axis.set_xlabel("seconds")
        axis.set_ylabel("n_events")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_event_vs_nonevent_density(table: pd.DataFrame, *, event_column: str, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    event_mask = pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 1
    for axis, variable in zip(axes, EVENT_TIME_VARIABLES, strict=True):
        event_values = pd.to_numeric(table.loc[event_mask, variable], errors="coerce").dropna()
        nonevent_values = pd.to_numeric(table.loc[~event_mask, variable], errors="coerce").dropna()
        if event_values.empty or nonevent_values.empty:
            axis.text(0.5, 0.5, "Insufficient data.", ha="center", va="center")
            axis.set_axis_off()
            continue
        bins = min(80, max(20, int(np.sqrt(len(event_values)))))
        axis.hist(nonevent_values, bins=bins, density=True, color="#b7c7d6", alpha=0.75, label="nonevent")
        axis.hist(event_values, bins=bins, density=True, color="#d17c4b", alpha=0.65, label="event")
        axis.set_title(variable)
        axis.set_xlabel("seconds")
        axis.set_ylabel("density")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_event_rate_by_time_bin(binned: pd.DataFrame, *, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, variable in zip(axes, EVENT_TIME_VARIABLES, strict=True):
        subset = binned.loc[binned["variable"] == variable].copy()
        if subset.empty:
            axis.text(0.5, 0.5, "No binned data.", ha="center", va="center")
            axis.set_axis_off()
            continue
        centers = (pd.to_numeric(subset["bin_left"], errors="coerce") + pd.to_numeric(subset["bin_right"], errors="coerce")) / 2.0
        axis.plot(centers, pd.to_numeric(subset["event_rate"], errors="coerce"), color="#3b7ea1")
        axis.set_title(variable)
        axis.set_xlabel("time bin center (s)")
        axis.set_ylabel("event_rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_spline_basis_by_time(table: pd.DataFrame, *, fit_sample: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, (sample_name, frame) in enumerate((("full", table), ("fit_sample", fit_sample))):
        for column_index, variable in enumerate(EVENT_TIME_VARIABLES):
            axis = axes[row_index, column_index]
            ordered = frame.sort_values(variable, kind="mergesort").head(5000).copy()
            if ordered.empty:
                axis.text(0.5, 0.5, "No data.", ha="center", va="center")
                axis.set_axis_off()
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                basis = dmatrix(f"bs({variable}, df=6, degree=3, include_intercept=False) - 1", data=ordered, return_type="dataframe")
            x_values = pd.to_numeric(ordered[variable], errors="coerce").to_numpy(dtype=float)
            for basis_column in basis.columns[: min(6, len(basis.columns))]:
                axis.plot(x_values, pd.to_numeric(basis[basis_column], errors="coerce").to_numpy(dtype=float), alpha=0.7)
            axis.set_title(f"{sample_name}: {variable}")
            axis.set_xlabel("seconds")
            axis.set_ylabel("basis value")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_sampled_vs_full_nonevent_time(
    table: pd.DataFrame,
    *,
    fit_sample: pd.DataFrame,
    event_column: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    full_nonevent = table.loc[pd.to_numeric(table[event_column], errors="coerce").fillna(0) == 0].copy()
    sampled_nonevent = fit_sample.loc[pd.to_numeric(fit_sample[event_column], errors="coerce").fillna(0) == 0].copy()
    event_rows = fit_sample.loc[pd.to_numeric(fit_sample[event_column], errors="coerce").fillna(0) == 1].copy()
    for axis, variable in zip(axes, EVENT_TIME_VARIABLES, strict=True):
        full_values = pd.to_numeric(full_nonevent[variable], errors="coerce").dropna()
        sampled_values = pd.to_numeric(sampled_nonevent[variable], errors="coerce").dropna()
        event_values = pd.to_numeric(event_rows[variable], errors="coerce").dropna()
        axis.hist(full_values, bins=50, density=True, color="#d9dee2", alpha=0.65, label="full nonevent")
        axis.hist(sampled_values, bins=50, density=True, color="#7fa650", alpha=0.65, label="sampled nonevent")
        axis.hist(event_values, bins=50, density=True, color="#d17c4b", alpha=0.55, label="events")
        axis.set_title(variable)
        axis.set_xlabel("seconds")
        axis.set_ylabel("density")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_fpp_vs_spp_time_histograms(
    *,
    fpp_riskset: pd.DataFrame | None,
    spp_riskset: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, variable in zip(axes, EVENT_TIME_VARIABLES, strict=True):
        spp_values = pd.to_numeric(
            spp_riskset.loc[pd.to_numeric(spp_riskset["event_spp"], errors="coerce").fillna(0) == 1, variable],
            errors="coerce",
        ).dropna()
        if fpp_riskset is not None and "event_fpp" in fpp_riskset.columns:
            fpp_values = pd.to_numeric(
                fpp_riskset.loc[pd.to_numeric(fpp_riskset["event_fpp"], errors="coerce").fillna(0) == 1, variable],
                errors="coerce",
            ).dropna()
        else:
            fpp_values = pd.Series(dtype=float)
        if not fpp_values.empty:
            axis.hist(fpp_values, bins=50, alpha=0.6, color="#3b7ea1", label="FPP")
        axis.hist(spp_values, bins=50, alpha=0.6, color="#d17c4b", label="SPP")
        axis.set_title(variable)
        axis.set_xlabel("seconds")
        axis.set_ylabel("n_events")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def build_spp_timing_debug_report(payload: dict[str, Any]) -> str:
    """Build the final markdown report."""

    event_checks = payload["consistency_summary"]
    overlap = payload["overlap_summary"]
    design_rows = payload["design_summary"]
    simple_fit_rows = payload["simple_fit_summary"]
    ridge_rows = payload["ridge_fit_summary"]

    def _status(name: str) -> str:
        item = event_checks.get(name, {})
        return f"{item.get('status', 'unknown')} (violations={item.get('n_violations', 'NA')})"

    simple_failures = [row for row in simple_fit_rows if row.get("status") != "converged"]
    intercept_failed = any(row["model_name"] == "intercept_only" and row["status"] != "converged" for row in simple_fit_rows)
    linear_ok = any(row["model_name"] == "linear_onset_offset" and row["status"] == "converged" for row in simple_fit_rows)
    offset_spline_failed = any(row["model_name"] == "spline_df3_offset_only" and row["status"] != "converged" for row in simple_fit_rows)
    spline_df6_failed = any(row["model_name"] == "spline_df6_onset_offset" and row["status"] != "converged" for row in simple_fit_rows)
    sampled_overlap = overlap.get("time_from_partner_offset", {}).get("proportion_nonevents_within_event_5_95_percentile_range")
    duplicate_summary = payload["duplicate_summary"]
    onset_start_ok = event_checks.get("time_from_partner_onset_matches_bin_start", {}).get("status") == "pass"
    onset_end_ok = event_checks.get("time_from_partner_onset_matches_bin_end", {}).get("status") == "pass"

    if intercept_failed:
        conclusion = "The failure pattern still looks compatible with a coding or fitting bug, because even the intercept-only model does not converge."
    elif offset_spline_failed and linear_ok:
        conclusion = "The strongest evidence points to a genuine timing-baseline instability concentrated in the offset spline: basic event coding is valid, linear timing terms converge, but flexible offset-based spline terms become numerically unstable."
    elif linear_ok and spline_df6_failed:
        conclusion = "The strongest evidence points to a genuine timing-baseline instability: simpler timing models converge, but the flexible df=6 onset+offset spline becomes numerically unstable."
    elif simple_failures:
        conclusion = "The failure is mixed: some simple timing models already struggle, so the issue may combine strong timing separation with limited numerical robustness."
    else:
        conclusion = "The timing-only baseline does not fail under these diagnostics, which suggests the original failure may have depended on a different sample or fitting path."

    return textwrap.dedent(
        f"""\
        # SPP Timing Debug Report

        ## 1. Summary

        This report checks whether the SPP timing-only failure is caused by invalid event coding, bad timing variables, duplicate rows, pathological subsampling, or genuine near-separation by timing.

        Riskset path: `{payload['riskset_path']}`
        FPP comparison riskset path: `{payload['fpp_riskset_path']}`

        Rows: {payload['n_rows']}, events: {payload['n_events']}, nonevents: {payload['n_nonevents']}, episodes: {payload['n_episodes']}.
        Fit-sample rows: {payload['fit_sample_rows']}, fit-sample events: {payload['fit_sample_events']}.

        ## 2. Event Coding Checks

        A. Are `event_spp` labels valid?
        Result: {_status('event_column_validity')}

        B. Are `time_from_partner_onset` and `time_from_partner_offset` computed correctly?
        Results:
        - `time_from_partner_onset` vs `bin_start - partner_ipu_onset`: {_status('time_from_partner_onset_matches_bin_start')}
        - `time_from_partner_onset` vs `bin_end - partner_ipu_onset`: {_status('time_from_partner_onset_matches_bin_end')}
        - `time_from_partner_offset`: {_status('time_from_partner_offset_matches')}
        - `bin_end >= bin_start`: {_status('bin_end_ge_bin_start')}
        - `partner_ipu_offset >= partner_ipu_onset`: {_status('partner_ipu_offset_ge_onset')}

        Interpretation: {"The onset clock is internally consistent with a left-edge/bin-start convention." if onset_start_ok and not onset_end_ok else "The onset clock follows the expected convention." if onset_end_ok else "The onset clock is not internally consistent with either simple bin-start or bin-end convention."}

        C. Are event rows aligned with SPP onset bins?
        Result: {_status('event_rows_align_with_onset_bin')}
        Onset column used: `{payload['event_onset_column']}`

        ## 3. Timing Variable Checks

        Identity validation:
        ```json
        {json.dumps(payload['identity_validation'], indent=2, sort_keys=True)}
        ```

        ## 4. Event vs Non-Event Timing Overlap

        ```json
        {json.dumps(overlap, indent=2, sort_keys=True)}
        ```

        ## 5. FPP vs SPP Timing Comparison

        FPP event timing spread:
        ```json
        {json.dumps(payload['fpp_event_summary'], indent=2)}
        ```

        SPP event timing spread:
        ```json
        {json.dumps(payload['spp_event_summary'], indent=2)}
        ```

        ## 6. Duplicate Checks

        D. Are duplicate rows or merge artifacts present?
        ```json
        {json.dumps(duplicate_summary, indent=2, sort_keys=True)}
        ```

        ## 7. Subsampling Checks

        F. Is subsampling causing artificial separation?
        Evidence: sampled nonevent overlap with the SPP event 5-95% timing range is {sampled_overlap}.
        Compare this against the full-nonevent summaries in `spp_timing_sampling_check.csv` and the figure `spp_timing_sampled_vs_full_nonevent_time.png`.

        ## 8. Spline Design Checks

        E. Is the spline design numerically sane?
        ```json
        {json.dumps(design_rows, indent=2, sort_keys=True)}
        ```

        ## 9. Simple Fit Checks

        ```json
        {json.dumps(simple_fit_rows, indent=2)}
        ```

        ## 10. Ridge Diagnostic

        ```json
        {json.dumps(ridge_rows, indent=2)}
        ```

        ## 11. Conclusion: Genuine Timing Separation Or Bug?

        G. Does SPP timing genuinely create near-separation?
        Provisional answer: {conclusion}

        H. Is the current SPP unpenalized hazard model appropriate?
        Provisional answer: {"No; the diagnostics indicate that unpenalized offset-spline timing terms are too unstable for this SPP setup, even though simpler timing baselines converge." if spline_df6_failed or offset_spline_failed else "Possibly, but only if rerun on the same sample and fit path reproduces stable simple models."}

        ## 12. Recommended Next Steps

        - If any event/timing consistency checks failed, fix riskset construction or event coding before interpreting SPP at all.
        - If simple linear models converge but df=6 splines fail, prefer a lower-df timing baseline or penalized timing diagnostics before claiming an SPP null.
        - If sampled nonevents poorly overlap the event timing range, use time-matched or stratified nonevent sampling for diagnostic refits.
        - If ridge timing models are stable while unpenalized spline models fail, treat the failure as quasi-separation/numerical instability rather than substantive absence of signal.
        """
    )


def _checks_to_dict(checks: pd.DataFrame) -> dict[str, Any]:
    if checks.empty:
        return {}
    return {
        str(row["check_name"]): {
            "status": row.get("status"),
            "n_rows_checked": row.get("n_rows_checked"),
            "n_violations": row.get("n_violations"),
            "details": row.get("details"),
        }
        for row in checks.to_dict(orient="records")
    }


def _duplicate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _design_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _series_summary(values: pd.Series) -> dict[str, Any]:
    if values.empty:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "q001": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "q999": None,
            "max": None,
        }
    quantiles = values.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q001": float(quantiles.loc[0.001]),
        "q01": float(quantiles.loc[0.01]),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
        "q99": float(quantiles.loc[0.99]),
        "q999": float(quantiles.loc[0.999]),
        "max": float(values.max()),
    }


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _write_empty_outputs(output_dir: Path) -> None:
    for key, filename in OUTPUT_FILENAMES.items():
        path = output_dir / filename
        if path.suffix.lower() == ".csv":
            write_table(pd.DataFrame(), path, sep=",")
        elif path.suffix.lower() == ".png":
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.text(0.5, 0.5, "Input riskset unavailable.", ha="center", va="center")
            axis.set_axis_off()
            figure.tight_layout()
            figure.savefig(path, dpi=300)
            plt.close(figure)


def _build_missing_input_report(notes: list[str]) -> str:
    return "# SPP Timing Debug Report\n\nUnable to run the timing debug pass because the source SPP riskset was unavailable.\n\n" + "\n".join(f"- {note}" for note in notes) + "\n"
