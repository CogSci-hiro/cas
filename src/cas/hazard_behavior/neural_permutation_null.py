"""Circular-shift event-label permutation null for FPP neural hazard analysis."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cas.hazard.config import NeuralHazardConfig
from cas.hazard_behavior.diagnose_spp_neural_failure import (
    ensure_participant_id_column,
    infer_lag_metadata_from_columns,
    inspect_riskset_columns,
    load_riskset_table,
)
from cas.hazard_behavior.io import write_json, write_table
from cas.hazard_behavior.neural_lowlevel import (
    FittedFormulaModel,
    _append_terms,
    _discover_band_pc_terms,
    _fit_formula_model,
    _maybe_float,
)
from cas.hazard_behavior.plot_neural_permutation_null import plot_family_comparison, plot_permutation_null_outputs
NEURAL_FAMILY_ORDER = ("alpha", "beta", "alpha_beta")
DEFAULT_OUTPUT_DIR = Path("results/hazard_behavior/neural_lowlevel/permutation_null")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FppNeuralPermutationNullResult:
    """Output locations for a completed permutation-null run."""

    output_dir: Path
    family_output_dirs: dict[str, Path]
    combined_summary_path: Path | None


def circular_shift_events_within_episode(
    data: pd.DataFrame,
    event_column: str,
    episode_column: str,
    rng: np.random.Generator,
    require_nonzero_shift: bool = True,
) -> pd.Series:
    """Circularly shift event labels within each episode while preserving row order."""

    if event_column not in data.columns:
        raise ValueError(f"Missing event column `{event_column}`.")
    if episode_column not in data.columns:
        raise ValueError(f"Missing episode column `{episode_column}`.")
    shifted = pd.Series(index=data.index, dtype="int64")
    working_events = pd.to_numeric(data[event_column], errors="coerce").fillna(0).astype(int)
    for _, episode_rows in data.assign(_event=working_events).groupby(episode_column, sort=False, dropna=False):
        values = episode_rows["_event"].to_numpy(dtype=int, copy=True)
        episode_length = int(values.size)
        if episode_length <= 1 or int(values.sum()) == 0:
            shifted.loc[episode_rows.index] = values
            continue
        if require_nonzero_shift:
            offset = int(rng.integers(1, episode_length))
        else:
            offset = int(rng.integers(0, episode_length))
        shifted.loc[episode_rows.index] = np.roll(values, offset)
    return shifted.astype(int)


def compute_empirical_p_value(real_value: float, null_values: list[float] | np.ndarray) -> float | None:
    """Lower-tail empirical p-value with a +1 correction."""

    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    count = int(np.count_nonzero(finite <= float(real_value)))
    return float((1 + count) / (1 + finite.size))


def run_fpp_neural_permutation_null(
    *,
    riskset_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    neural_family: str = "beta",
    n_permutations: int = 1000,
    seed: int = 12345,
    event_column: str = "event_fpp",
    episode_column: str = "episode_id",
    participant_column: str = "participant_speaker_id",
    run_column: str = "run",
    delta_criterion: str = "bic",
    n_jobs: int = 1,
    max_permutations_for_smoke_test: int | None = None,
    max_fit_rows: int | None = None,
    fit_model_fn: Callable[..., FittedFormulaModel] | None = None,
    verbose: bool = False,
) -> FppNeuralPermutationNullResult:
    """Run the opt-in FPP neural permutation null on a model-ready riskset."""

    _configure_logging(verbose=verbose)
    if str(delta_criterion).lower() not in {"bic", "aic"}:
        raise ValueError("`delta_criterion` must be one of bic or aic.")
    requested_families = _resolve_neural_families(neural_family)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_fn = fit_model_fn or _fit_formula_model
    LOGGER.info("Starting FPP neural permutation null.")
    LOGGER.info("Riskset path: %s", riskset_path)
    LOGGER.info(
        "Requested families=%s n_permutations=%d seed=%d n_jobs=%d delta_criterion=%s",
        ",".join(requested_families),
        int(n_permutations),
        int(seed),
        int(n_jobs),
        str(delta_criterion).lower(),
    )

    available_columns = inspect_riskset_columns(riskset_path)
    event_column = _resolve_existing_column(available_columns, preferred=event_column, fallback="event")
    lag_metadata = infer_lag_metadata_from_columns(available_columns, models_dir=None)
    info_column = lag_metadata["information_rate_column"]
    prop_column = lag_metadata["prop_expected_column"]

    required_columns = sorted(
        {
            event_column,
            episode_column,
            participant_column,
            run_column,
            "dyad_id",
            "speaker",
            "time_from_partner_onset",
            "time_from_partner_offset",
            info_column,
            prop_column,
            *[name for name in available_columns if name.startswith("z_alpha_pc") or name.startswith("z_beta_pc")],
        }
    )
    table = load_riskset_table(riskset_path, columns=[name for name in required_columns if name in available_columns])
    table = ensure_participant_id_column(table)
    participant_column = _resolve_identity_column(
        table.columns,
        preferred=participant_column,
        fallbacks=("participant_speaker_id", "participant_id", "participant_speaker"),
    )
    if max_fit_rows is not None:
        table = table.head(int(max_fit_rows)).copy()
        LOGGER.info("Applied max_fit_rows=%d; truncated riskset to %d rows.", int(max_fit_rows), int(len(table)))
    LOGGER.info(
        "Loaded riskset with %d rows and %d columns. Using event_column=%s participant_column=%s.",
        int(len(table)),
        int(len(table.columns)),
        event_column,
        participant_column,
    )
    family_output_dirs: dict[str, Path] = {}
    combined_rows: list[dict[str, object]] = []
    effective_n_permutations = int(n_permutations)
    if max_permutations_for_smoke_test is not None:
        effective_n_permutations = min(effective_n_permutations, int(max_permutations_for_smoke_test))
        LOGGER.info(
            "Applied max_permutations_for_smoke_test=%d; effective permutations=%d.",
            int(max_permutations_for_smoke_test),
            int(effective_n_permutations),
        )

    for family in requested_families:
        family_dir = output_dir / family
        family_output_dirs[family] = family_dir
        family_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = family_dir / "figures"
        LOGGER.info("Preparing family=%s output_dir=%s", family, family_dir)

        prepared = prepare_family_complete_case_data(
            riskset_table=table,
            event_column=event_column,
            episode_column=episode_column,
            participant_column=participant_column,
            run_column=run_column,
            neural_family=family,
            information_rate_column=info_column,
            prop_expected_column=prop_column,
        )
        LOGGER.info(
            "Family=%s complete-case rows=%d events=%d episodes=%d multi_event_episodes=%d",
            family,
            int(len(prepared.data)),
            int(prepared.data[prepared.event_column].sum()),
            int(prepared.qc_counts["n_episodes"]),
            int(prepared.qc_counts["n_multi_event_episodes"]),
        )
        real_row = fit_real_family_comparison(
            family_data=prepared,
            fit_model_fn=fit_fn,
        )
        LOGGER.info(
            "Family=%s real fit delta_bic=%s delta_aic=%s comparison_valid=%s",
            family,
            real_row.get("delta_bic"),
            real_row.get("delta_aic"),
            real_row.get("comparison_valid"),
        )
        real_comparison = pd.DataFrame([real_row])
        permutation_distribution = run_family_permutations(
            family_data=prepared,
            n_permutations=effective_n_permutations,
            seed=seed,
            n_jobs=n_jobs,
            fit_model_fn=fit_fn,
            verbose=verbose,
        )
        summary = summarize_family_permutations(
            family_data=prepared,
            real_comparison=real_comparison,
            permutation_distribution=permutation_distribution,
            seed=seed,
            n_permutations_requested=effective_n_permutations,
            delta_criterion=str(delta_criterion).lower(),
        )
        qc_payload = build_family_qc_payload(
            family_data=prepared,
            real_comparison=real_comparison,
            permutation_distribution=permutation_distribution,
        )
        report_text = build_family_report(
            summary=summary,
            qc_payload=qc_payload,
            real_comparison=real_comparison,
            permutation_distribution=permutation_distribution,
        )

        write_table(real_comparison, family_dir / "fpp_neural_permutation_real_comparison.csv", sep=",")
        write_table(permutation_distribution, family_dir / "fpp_neural_permutation_null_distribution.csv", sep=",")
        write_json(summary, family_dir / "fpp_neural_permutation_summary.json")
        write_json(qc_payload, family_dir / "fpp_neural_permutation_qc.json")
        (family_dir / "fpp_neural_permutation_report.md").write_text(report_text, encoding="utf-8")
        plot_permutation_null_outputs(
            real_comparison=real_comparison,
            null_distribution=permutation_distribution,
            summary=summary,
            figures_dir=figures_dir,
        )
        LOGGER.info(
            "Family=%s finished: successful_permutations=%s failed_permutations=%s empirical_p_delta_bic=%s",
            family,
            summary.get("n_permutations_successful"),
            summary.get("n_permutations_failed"),
            summary.get("empirical_p_delta_bic"),
        )
        combined_rows.append(
            {
                "neural_family": family,
                "real_delta_bic": summary.get("real_delta_bic"),
                "null_median_delta_bic": summary.get("null_delta_bic_median"),
                "null_q025_delta_bic": summary.get("null_delta_bic_q025"),
                "null_q975_delta_bic": summary.get("null_delta_bic_q975"),
                "empirical_p_delta_bic": summary.get("empirical_p_delta_bic"),
                "n_permutations_successful": summary.get("n_permutations_successful"),
                "n_permutations_failed": summary.get("n_permutations_failed"),
            }
        )

    combined_summary_path = None
    if len(combined_rows) > 1:
        combined_summary = pd.DataFrame(combined_rows)
        combined_summary_path = output_dir / "fpp_neural_permutation_combined_summary.csv"
        write_table(combined_summary, combined_summary_path, sep=",")
        plot_family_comparison(combined_summary, output_path=output_dir / "figures" / "fpp_neural_permutation_family_comparison.png")
        LOGGER.info("Wrote combined family summary to %s", combined_summary_path)

    LOGGER.info("FPP neural permutation null completed. Output root: %s", output_dir)

    return FppNeuralPermutationNullResult(
        output_dir=output_dir,
        family_output_dirs=family_output_dirs,
        combined_summary_path=combined_summary_path,
    )


@dataclass(frozen=True, slots=True)
class PreparedFamilyData:
    neural_family: str
    data: pd.DataFrame
    event_column: str
    episode_column: str
    participant_column: str
    run_column: str
    baseline_formula: str
    child_formula: str
    parent_model_name: str
    child_model_name: str
    complete_case_columns: list[str]
    qc_counts: dict[str, object]
    row_ids: pd.Series


def prepare_family_complete_case_data(
    *,
    riskset_table: pd.DataFrame,
    event_column: str,
    episode_column: str,
    participant_column: str,
    run_column: str,
    neural_family: str,
    information_rate_column: str,
    prop_expected_column: str,
) -> PreparedFamilyData:
    """Prepare one family-specific complete-case riskset for same-row fitting."""

    _validate_required_columns(
        riskset_table.columns,
        required=[
            event_column,
            episode_column,
            participant_column,
            run_column,
            "dyad_id",
            "time_from_partner_onset",
            "time_from_partner_offset",
            information_rate_column,
            prop_expected_column,
        ],
    )
    family_terms = _family_terms_from_table(riskset_table, neural_family=neural_family)
    template = NeuralHazardConfig(enabled=True)
    baseline_formula = _build_baseline_formula(
        event_column=event_column,
        information_rate_column=information_rate_column,
        prop_expected_column=prop_expected_column,
        neural_config=template,
    )
    child_formula = _append_terms(baseline_formula, family_terms)
    complete_case_columns = [
        event_column,
        episode_column,
        participant_column,
        run_column,
        "dyad_id",
        "time_from_partner_onset",
        "time_from_partner_offset",
        information_rate_column,
        prop_expected_column,
        *family_terms,
    ]
    subset = _subset_family_complete_cases(
        riskset_table,
        numeric_columns=[
            event_column,
            "time_from_partner_onset",
            "time_from_partner_offset",
            information_rate_column,
            prop_expected_column,
            *family_terms,
        ],
        id_columns=[episode_column],
        carry_columns=complete_case_columns,
    )
    row_ids = _build_row_ids(subset, episode_column=episode_column, participant_column=participant_column, run_column=run_column)
    suffix = "FPP"
    child_model_name = f"M_{neural_family}_{suffix}" if neural_family != "alpha_beta" else f"M_alpha_beta_{suffix}"
    parent_model_name = f"M_behaviour_{suffix}_{neural_family}_sample"
    qc_counts = compute_episode_qc(subset, event_column=event_column, episode_column=episode_column)
    return PreparedFamilyData(
        neural_family=neural_family,
        data=subset,
        event_column=event_column,
        episode_column=episode_column,
        participant_column=participant_column,
        run_column=run_column,
        baseline_formula=baseline_formula,
        child_formula=child_formula,
        parent_model_name=parent_model_name,
        child_model_name=child_model_name,
        complete_case_columns=complete_case_columns,
        qc_counts=qc_counts,
        row_ids=row_ids,
    )


def fit_real_family_comparison(
    *,
    family_data: PreparedFamilyData,
    fit_model_fn: Callable[..., FittedFormulaModel],
) -> dict[str, object]:
    """Fit the real parent and child models on the fixed complete-case rows."""

    comparison = _fit_parent_child_models(
        data=family_data.data,
        event_column=family_data.event_column,
        baseline_formula=family_data.baseline_formula,
        child_formula=family_data.child_formula,
        parent_model_name=family_data.parent_model_name,
        child_model_name=family_data.child_model_name,
        fit_model_fn=fit_model_fn,
    )
    return {
        "event_type": "fpp",
        "neural_family": family_data.neural_family,
        "parent_model": family_data.parent_model_name,
        "child_model": family_data.child_model_name,
        "n_rows": int(len(family_data.data)),
        "n_events": int(family_data.data[family_data.event_column].sum()),
        "parent_aic": comparison["parent_aic"],
        "child_aic": comparison["child_aic"],
        "delta_aic": comparison["delta_aic"],
        "parent_bic": comparison["parent_bic"],
        "child_bic": comparison["child_bic"],
        "delta_bic": comparison["delta_bic"],
        "parent_log_likelihood": comparison["parent_log_likelihood"],
        "child_log_likelihood": comparison["child_log_likelihood"],
        "lrt_statistic": comparison["lrt_statistic"],
        "df_difference": comparison["df_difference"],
        "p_value": comparison["p_value"],
        "n_predictors_parent": comparison["n_predictors_parent"],
        "n_predictors_child": comparison["n_predictors_child"],
        "comparison_valid": comparison["comparison_valid"],
        "same_rows": True,
    }


def run_family_permutations(
    *,
    family_data: PreparedFamilyData,
    n_permutations: int,
    seed: int,
    n_jobs: int,
    fit_model_fn: Callable[..., FittedFormulaModel],
    verbose: bool,
) -> pd.DataFrame:
    """Run the permutation loop for one family."""

    child_seeds = [int(value) for value in np.random.SeedSequence(int(seed)).generate_state(int(n_permutations), dtype=np.uint32)]
    payloads = [
        {
            "permutation_id": permutation_id + 1,
            "seed": permutation_seed,
            "data": family_data.data,
            "event_column": family_data.event_column,
            "episode_column": family_data.episode_column,
            "baseline_formula": family_data.baseline_formula,
            "child_formula": family_data.child_formula,
            "parent_model_name": family_data.parent_model_name,
            "child_model_name": family_data.child_model_name,
        }
        for permutation_id, permutation_seed in enumerate(child_seeds)
    ]
    LOGGER.info(
        "Running %d permutations for family=%s with n_jobs=%d.",
        int(n_permutations),
        family_data.neural_family,
        int(n_jobs),
    )
    if int(n_jobs) > 1 and fit_model_fn is _fit_formula_model:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            rows = list(_permutation_progress(
                executor.map(_run_single_permutation_worker, payloads),
                total=len(payloads),
                description=f"{family_data.neural_family} permutations",
                enabled=bool(verbose),
            )
            )
    else:
        rows = [
            _run_single_permutation(
                permutation_id=int(payload["permutation_id"]),
                permutation_seed=int(payload["seed"]),
                data=payload["data"],  # type: ignore[arg-type]
                event_column=str(payload["event_column"]),
                episode_column=str(payload["episode_column"]),
                baseline_formula=str(payload["baseline_formula"]),
                child_formula=str(payload["child_formula"]),
                parent_model_name=str(payload["parent_model_name"]),
                child_model_name=str(payload["child_model_name"]),
                fit_model_fn=fit_model_fn,
            )
            for payload in _permutation_progress(
                payloads,
                total=len(payloads),
                description=f"{family_data.neural_family} permutations",
                enabled=bool(verbose),
            )
        ]
    result = pd.DataFrame(rows).sort_values("permutation_id").reset_index(drop=True)
    LOGGER.info(
        "Family=%s permutation loop complete: successful=%d failed=%d",
        family_data.neural_family,
        int((result["status"].astype(str) == "ok").sum()),
        int((result["status"].astype(str) != "ok").sum()),
    )
    return result


def summarize_family_permutations(
    *,
    family_data: PreparedFamilyData,
    real_comparison: pd.DataFrame,
    permutation_distribution: pd.DataFrame,
    seed: int,
    n_permutations_requested: int,
    delta_criterion: str,
) -> dict[str, object]:
    """Summarize one family's real-vs-null comparison."""

    successful = permutation_distribution.loc[permutation_distribution["status"].astype(str) == "ok"].copy()
    delta_bic = pd.to_numeric(successful.get("delta_bic"), errors="coerce")
    delta_bic = delta_bic[np.isfinite(delta_bic)]
    delta_aic = pd.to_numeric(successful.get("delta_aic"), errors="coerce")
    delta_aic = delta_aic[np.isfinite(delta_aic)]
    real_row = real_comparison.iloc[0].to_dict()
    summary = {
        "event_type": "fpp",
        "neural_family": family_data.neural_family,
        "n_permutations_requested": int(n_permutations_requested),
        "n_permutations_successful": int((permutation_distribution["status"].astype(str) == "ok").sum()),
        "n_permutations_failed": int((permutation_distribution["status"].astype(str) != "ok").sum()),
        "seed": int(seed),
        "statistic_primary": f"delta_{delta_criterion}",
        "delta_convention": "child - parent; negative favours child",
        "real_delta_bic": real_row.get("delta_bic"),
        "null_delta_bic_mean": _summary_stat(delta_bic, np.mean),
        "null_delta_bic_sd": _summary_stat(delta_bic, np.std),
        "null_delta_bic_median": _summary_quantile(delta_bic, 0.50),
        "null_delta_bic_q01": _summary_quantile(delta_bic, 0.01),
        "null_delta_bic_q025": _summary_quantile(delta_bic, 0.025),
        "null_delta_bic_q05": _summary_quantile(delta_bic, 0.05),
        "null_delta_bic_q95": _summary_quantile(delta_bic, 0.95),
        "null_delta_bic_q975": _summary_quantile(delta_bic, 0.975),
        "null_delta_bic_q99": _summary_quantile(delta_bic, 0.99),
        "empirical_p_delta_bic": compute_empirical_p_value(float(real_row["delta_bic"]), delta_bic.to_numpy()) if pd.notna(real_row.get("delta_bic")) else None,
        "real_delta_aic": real_row.get("delta_aic"),
        "empirical_p_delta_aic": compute_empirical_p_value(float(real_row["delta_aic"]), delta_aic.to_numpy()) if pd.notna(real_row.get("delta_aic")) else None,
        "n_rows": int(real_row["n_rows"]),
        "n_events": int(real_row["n_events"]),
        "n_episodes": family_data.qc_counts["n_episodes"],
        "n_event_episodes": family_data.qc_counts["n_event_episodes"],
        "n_censored_episodes": family_data.qc_counts["n_censored_episodes"],
        "n_length_one_episodes": family_data.qc_counts["n_length_one_episodes"],
        "n_multi_event_episodes": family_data.qc_counts["n_multi_event_episodes"],
        "complete_case_columns": family_data.complete_case_columns,
        "model_formulas": {
            "parent": family_data.baseline_formula,
            "child": family_data.child_formula,
        },
        "convergence_summary": {
            "real_comparison_valid": bool(real_row["comparison_valid"]),
            "failed_permutation_ids": permutation_distribution.loc[permutation_distribution["status"].astype(str) != "ok", "permutation_id"].astype(int).tolist(),
        },
    }
    return summary


def build_family_qc_payload(
    *,
    family_data: PreparedFamilyData,
    real_comparison: pd.DataFrame,
    permutation_distribution: pd.DataFrame,
) -> dict[str, object]:
    """Build the QC payload for one family."""

    successful = permutation_distribution.loc[permutation_distribution["status"].astype(str) == "ok"].copy()
    event_counts_after = pd.to_numeric(successful.get("n_events_after_shift"), errors="coerce")
    changed_rows = pd.to_numeric(successful.get("proportion_event_rows_changed"), errors="coerce")
    changed_episodes = pd.to_numeric(successful.get("proportion_event_episodes_changed"), errors="coerce")
    warnings: list[str] = []
    if int(family_data.qc_counts["n_multi_event_episodes"]) > 0:
        warnings.append("Multiple-event episodes were present; counts were preserved but this deviates from the usual FPP setup.")
    if real_comparison.loc[0, "comparison_valid"] is False:
        warnings.append("The real parent/child comparison was not fully valid.")
    if successful.empty:
        warnings.append("No successful permutations were available for null-distribution summaries.")
    return {
        "event_preservation_checks": {
            "total_events_before": family_data.qc_counts["total_events_before"],
            "total_events_after_unique": sorted({int(value) for value in event_counts_after.dropna().astype(int).tolist()}),
            "events_preserved_all_successful_permutations": bool(
                not successful.empty and (event_counts_after.dropna().astype(int) == int(family_data.qc_counts["total_events_before"])).all()
            ) if not successful.empty else False,
        },
        "row_count_checks": {
            "real_n_rows": int(real_comparison.loc[0, "n_rows"]),
            "permutation_n_rows_unique": sorted({int(value) for value in pd.to_numeric(permutation_distribution["n_rows"], errors="coerce").dropna().astype(int).tolist()}),
        },
        "same_row_checks": {
            "same_rows_real_comparison": bool(real_comparison.loc[0, "same_rows"]),
            "same_complete_case_row_ids_used_for_parent_and_child": True,
        },
        "missingness_checks": {
            "complete_case_columns": family_data.complete_case_columns,
            "n_rows_complete_case": int(len(family_data.data)),
            "n_rows_input": int(family_data.qc_counts["n_rows_input"]),
        },
        "episode_shift_checks": {
            **family_data.qc_counts,
            "proportion_event_rows_changed_range": _finite_range(changed_rows.to_numpy(dtype=float, copy=False)),
            "proportion_event_episodes_changed_range": _finite_range(changed_episodes.to_numpy(dtype=float, copy=False)),
        },
        "warnings": warnings,
    }


def build_family_report(
    *,
    summary: dict[str, object],
    qc_payload: dict[str, object],
    real_comparison: pd.DataFrame,
    permutation_distribution: pd.DataFrame,
) -> str:
    """Build the markdown report for one family."""

    real_row = real_comparison.iloc[0]
    null_values = pd.to_numeric(
        permutation_distribution.loc[permutation_distribution["status"].astype(str) == "ok", "delta_bic"],
        errors="coerce",
    )
    null_values = null_values[np.isfinite(null_values)]
    interpretation = (
        "The FPP neural improvement depends on true temporal alignment between neural features and FPP onset."
        if null_values.size > 0 and pd.notna(real_row["delta_bic"]) and float(real_row["delta_bic"]) <= float(np.nanpercentile(null_values, 1.0))
        else "The apparent neural improvement may be explainable by generic episode/timing structure or chance alignment."
    )
    failed_count = int((permutation_distribution["status"].astype(str) != "ok").sum())
    return (
        "# Purpose\n\n"
        "This analysis tests whether the low-level neural improvement for FPP depends on the true temporal alignment between neural features and FPP onset.\n\n"
        "# Model comparison\n\n"
        f"Real comparison: `{real_row['child_model']}` vs `{real_row['parent_model']}` on `{int(real_row['n_rows'])}` complete-case rows with `{int(real_row['n_events'])}` events.\n\n"
        "# Permutation null design\n\n"
        "Within each FPP hazard episode, the FPP event label is circularly shifted across risk bins while preserving row order, episode length, predictor trajectories, and event counts.\n\n"
        "# Real neural improvement\n\n"
        f"Real delta BIC = `{real_row['delta_bic']}` and real delta AIC = `{real_row['delta_aic']}`. Negative values favour the neural child model.\n\n"
        "# Null distribution\n\n"
        f"Successful permutations: `{summary['n_permutations_successful']}` of `{summary['n_permutations_requested']}` requested.\n\n"
        "# Empirical p-value\n\n"
        f"Primary statistic: `{summary['statistic_primary']}`. Empirical p(delta BIC) = `{summary['empirical_p_delta_bic']}` and empirical p(delta AIC) = `{summary['empirical_p_delta_aic']}`.\n\n"
        "# Fit failures / warnings\n\n"
        f"Failed permutations: `{failed_count}`. QC warnings: `{'; '.join(qc_payload['warnings']) if qc_payload['warnings'] else 'none'}`.\n\n"
        "# Interpretation\n\n"
        f"{interpretation}\n\n"
        "This does not prove FPP specificity against SPP. It tests temporal alignment for FPP.\n\n"
        "# Limitations\n\n"
        "The null preserves episode structure and predictor trajectories, so it isolates alignment but does not address every alternative explanation. SPP instability due to timing quasi-separation is not treated as a null result.\n"
    )


def compute_episode_qc(data: pd.DataFrame, *, event_column: str, episode_column: str) -> dict[str, object]:
    """Compute episode-level QC counts on the complete-case fitting set."""

    grouped = data.groupby(episode_column, sort=False, dropna=False)
    event_counts = grouped[event_column].sum()
    lengths = grouped.size()
    return {
        "n_rows_input": int(len(data)),
        "n_episodes": int(len(event_counts)),
        "n_event_episodes": int((event_counts > 0).sum()),
        "n_censored_episodes": int((event_counts == 0).sum()),
        "n_length_one_episodes": int((lengths == 1).sum()),
        "n_multi_event_episodes": int((event_counts > 1).sum()),
        "total_events_before": int(data[event_column].sum()),
    }


def summarize_shift(
    original: pd.Series,
    shifted: pd.Series,
    *,
    episode_ids: pd.Series,
) -> dict[str, object]:
    """Summarize one permutation's event-shift effects."""

    original_values = pd.to_numeric(original, errors="coerce").fillna(0).astype(int)
    shifted_values = pd.to_numeric(shifted, errors="coerce").fillna(0).astype(int)
    event_episode_changed = 0
    n_event_episodes = 0
    for _, group in pd.DataFrame({"original": original_values, "shifted": shifted_values, "episode_id": episode_ids}).groupby("episode_id", sort=False, dropna=False):
        if int(group["original"].sum()) > 0:
            n_event_episodes += 1
            if not group["original"].equals(group["shifted"]):
                event_episode_changed += 1
    return {
        "n_events_after_shift": int(shifted_values.sum()),
        "proportion_event_rows_changed": float((original_values != shifted_values).mean()) if len(original_values) else 0.0,
        "proportion_event_episodes_changed": (
            float(event_episode_changed / n_event_episodes) if n_event_episodes > 0 else 0.0
        ),
    }


def _run_single_permutation(
    *,
    permutation_id: int,
    permutation_seed: int,
    data: pd.DataFrame,
    event_column: str,
    episode_column: str,
    baseline_formula: str,
    child_formula: str,
    parent_model_name: str,
    child_model_name: str,
    fit_model_fn: Callable[..., FittedFormulaModel],
) -> dict[str, object]:
    rng = np.random.default_rng(int(permutation_seed))
    working = data.copy()
    permuted_event_column = f"{event_column}_permuted"
    shifted = circular_shift_events_within_episode(
        working,
        event_column=event_column,
        episode_column=episode_column,
        rng=rng,
        require_nonzero_shift=True,
    )
    working[permuted_event_column] = shifted
    shift_summary = summarize_shift(working[event_column], shifted, episode_ids=working[episode_column])
    comparison = _fit_parent_child_models(
        data=working,
        event_column=permuted_event_column,
        baseline_formula=_replace_formula_lhs(baseline_formula, permuted_event_column),
        child_formula=_replace_formula_lhs(child_formula, permuted_event_column),
        parent_model_name=parent_model_name,
        child_model_name=child_model_name,
        fit_model_fn=fit_model_fn,
    )
    status = "ok" if comparison["comparison_valid"] else "failed"
    return {
        "permutation_id": int(permutation_id),
        "seed": int(permutation_seed),
        "neural_family": _family_from_model_name(child_model_name),
        "status": status,
        "n_rows": int(len(working)),
        "n_events": int(pd.to_numeric(working[event_column], errors="coerce").fillna(0).sum()),
        "n_events_after_shift": int(shift_summary["n_events_after_shift"]),
        "proportion_event_rows_changed": shift_summary["proportion_event_rows_changed"],
        "proportion_event_episodes_changed": shift_summary["proportion_event_episodes_changed"],
        "parent_aic": comparison["parent_aic"],
        "child_aic": comparison["child_aic"],
        "delta_aic": comparison["delta_aic"],
        "parent_bic": comparison["parent_bic"],
        "child_bic": comparison["child_bic"],
        "delta_bic": comparison["delta_bic"],
        "parent_log_likelihood": comparison["parent_log_likelihood"],
        "child_log_likelihood": comparison["child_log_likelihood"],
        "lrt_statistic": comparison["lrt_statistic"],
        "df_difference": comparison["df_difference"],
        "error_message": comparison["error_message"],
        "fit_warnings": comparison["fit_warnings"],
    }


def _run_single_permutation_worker(payload: dict[str, object]) -> dict[str, object]:
    return _run_single_permutation(
        permutation_id=int(payload["permutation_id"]),
        permutation_seed=int(payload["seed"]),
        data=payload["data"],  # type: ignore[arg-type]
        event_column=str(payload["event_column"]),
        episode_column=str(payload["episode_column"]),
        baseline_formula=str(payload["baseline_formula"]),
        child_formula=str(payload["child_formula"]),
        parent_model_name=str(payload["parent_model_name"]),
        child_model_name=str(payload["child_model_name"]),
        fit_model_fn=_fit_formula_model,
    )


def _fit_parent_child_models(
    *,
    data: pd.DataFrame,
    event_column: str,
    baseline_formula: str,
    child_formula: str,
    parent_model_name: str,
    child_model_name: str,
    fit_model_fn: Callable[..., FittedFormulaModel],
) -> dict[str, object]:
    parent_fit = fit_model_fn(
        riskset_table=data,
        model_name=parent_model_name,
        formula=baseline_formula,
        event_column=event_column,
    )
    child_fit = fit_model_fn(
        riskset_table=data,
        model_name=child_model_name,
        formula=child_formula,
        event_column=event_column,
    )
    same_rows = True
    parent_aic = _maybe_float(getattr(parent_fit.result, "aic", None)) if parent_fit.success else None
    child_aic = _maybe_float(getattr(child_fit.result, "aic", None)) if child_fit.success else None
    parent_bic = _maybe_float(getattr(parent_fit.result, "bic", None)) if parent_fit.success else None
    child_bic = _maybe_float(getattr(child_fit.result, "bic", None)) if child_fit.success else None
    parent_ll = _maybe_float(getattr(parent_fit.result, "llf", None)) if parent_fit.success else None
    child_ll = _maybe_float(getattr(child_fit.result, "llf", None)) if child_fit.success else None
    df_difference = int(child_fit.n_predictors - parent_fit.n_predictors) if parent_fit.success and child_fit.success else None
    comparison_valid = bool(
        same_rows
        and parent_fit.success
        and child_fit.success
        and parent_fit.converged
        and child_fit.converged
        and parent_aic is not None
        and child_aic is not None
        and parent_bic is not None
        and child_bic is not None
        and parent_ll is not None
        and child_ll is not None
    )
    lrt_statistic = None
    p_value = None
    if comparison_valid and df_difference is not None and df_difference > 0:
        from scipy import stats

        lrt_statistic = float(2.0 * (child_ll - parent_ll))
        p_value = float(stats.chi2.sf(lrt_statistic, df=df_difference))
    fit_warnings = "; ".join(value for value in [*parent_fit.fit_warnings, *child_fit.fit_warnings] if value)
    error_message = child_fit.error_message or parent_fit.error_message
    return {
        "comparison_valid": comparison_valid,
        "same_rows": same_rows,
        "parent_aic": parent_aic,
        "child_aic": child_aic,
        "delta_aic": (child_aic - parent_aic) if comparison_valid else None,
        "parent_bic": parent_bic,
        "child_bic": child_bic,
        "delta_bic": (child_bic - parent_bic) if comparison_valid else None,
        "parent_log_likelihood": parent_ll,
        "child_log_likelihood": child_ll,
        "lrt_statistic": lrt_statistic if comparison_valid else None,
        "df_difference": df_difference,
        "p_value": p_value if comparison_valid else None,
        "n_predictors_parent": int(parent_fit.n_predictors),
        "n_predictors_child": int(child_fit.n_predictors),
        "error_message": error_message,
        "fit_warnings": fit_warnings,
    }


def _subset_family_complete_cases(
    table: pd.DataFrame,
    *,
    numeric_columns: list[str],
    id_columns: list[str],
    carry_columns: list[str],
) -> pd.DataFrame:
    working = table.loc[:, carry_columns].copy()
    for column_name in numeric_columns:
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
    mask = np.ones(len(working), dtype=bool)
    for column_name in numeric_columns:
        mask &= np.isfinite(working[column_name].to_numpy(dtype=float))
    for column_name in id_columns:
        mask &= working[column_name].notna().to_numpy()
    subset = working.loc[mask].copy()
    if event_column := next((name for name in numeric_columns if str(name).startswith("event")), None):
        subset[event_column] = subset[event_column].astype(int)
    return subset


def _build_baseline_formula(
    *,
    event_column: str,
    information_rate_column: str,
    prop_expected_column: str,
    neural_config: NeuralHazardConfig,
) -> str:
    del neural_config
    return (
        f"{event_column} ~ time_from_partner_onset + time_from_partner_offset"
        f" + I(time_from_partner_offset ** 2) + {information_rate_column} + {prop_expected_column}"
    )


def _replace_formula_lhs(formula: str, event_column: str) -> str:
    _lhs, rhs = formula.split("~", 1)
    return f"{event_column} ~ {rhs.strip()}"


def _family_terms_from_table(table: pd.DataFrame, *, neural_family: str) -> list[str]:
    if neural_family == "alpha":
        return _discover_band_pc_terms(table, band_name="alpha")
    if neural_family == "beta":
        return _discover_band_pc_terms(table, band_name="beta")
    if neural_family == "alpha_beta":
        return [*_discover_band_pc_terms(table, band_name="alpha"), *_discover_band_pc_terms(table, band_name="beta")]
    raise ValueError("`neural_family` must be one of alpha, beta, alpha_beta.")


def _resolve_existing_column(columns: list[str], *, preferred: str, fallback: str) -> str:
    if preferred in columns:
        return preferred
    if fallback in columns:
        return fallback
    raise ValueError(f"Riskset is missing `{preferred}` and `{fallback}`.")


def _resolve_identity_column(columns: Any, *, preferred: str, fallbacks: tuple[str, ...]) -> str:
    names = set(str(name) for name in columns)
    if preferred in names:
        return preferred
    for candidate in fallbacks:
        if candidate in names:
            return candidate
    raise ValueError("Riskset is missing required identity columns: " + ", ".join((preferred, *fallbacks)))


def _validate_required_columns(columns: Any, *, required: list[str]) -> None:
    missing = [name for name in required if name not in set(columns)]
    if missing:
        raise ValueError("Riskset is missing required columns: " + ", ".join(sorted(missing)))


def _resolve_neural_families(neural_family: str) -> list[str]:
    if neural_family == "all":
        return list(NEURAL_FAMILY_ORDER)
    if neural_family not in NEURAL_FAMILY_ORDER:
        raise ValueError("`neural_family` must be one of alpha, beta, alpha_beta, or all.")
    return [neural_family]


def _build_row_ids(data: pd.DataFrame, *, episode_column: str, participant_column: str, run_column: str) -> pd.Series:
    pieces = [
        data[episode_column].astype(str),
        data.get("dyad_id", pd.Series([""] * len(data), index=data.index)).astype(str),
        data[participant_column].astype(str),
        data[run_column].astype(str),
    ]
    row_ids = pieces[0].copy()
    for piece in pieces[1:]:
        row_ids = row_ids + "|" + piece
    return row_ids.astype("string")


def _summary_stat(series: pd.Series, reducer: Callable[[np.ndarray], float]) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return float(reducer(values.to_numpy(dtype=float)))


def _summary_quantile(series: pd.Series, quantile: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return float(np.nanquantile(values.to_numpy(dtype=float), quantile))


def _finite_range(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def _family_from_model_name(model_name: str) -> str:
    if "alpha_beta" in model_name:
        return "alpha_beta"
    if "_alpha_" in model_name:
        return "alpha"
    return "beta"


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(level)
    LOGGER.setLevel(level)


def _permutation_progress(
    iterable: Any,
    *,
    total: int | None,
    description: str,
    enabled: bool,
) -> Any:
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=description,
        leave=True,
        dynamic_ncols=True,
    )
