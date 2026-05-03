"""Top-level pipeline runner for source-level pooled FPP/SPP DICS power."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from cas.source_dics.config import SourceDicsConfig, load_source_dics_config
from cas.source_dics.dics import (
    apply_common_filters_to_epochs,
    compute_common_dics_filters,
    resolve_forward_model,
    save_filters,
)
from cas.source_dics.events import prepare_epoch_metadata, split_anchor_metadata
from cas.source_dics.export import export_long_table, export_metadata, export_trial_power, summarize_mean_power
from cas.source_dics.io import (
    EpochRecord,
    configure_mne_runtime,
    discover_epoch_records,
    discover_preprocessed_records,
    load_epochs,
    load_events_table,
)
from cas.source_dics.qc import (
    write_band_subject_counts,
    write_config_snapshot,
    write_event_counts,
    write_filter_windows,
    write_mean_power,
    write_missing_inputs,
    write_run_summary,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceDicsPipelineResult:
    out_dir: Path
    summary_path: Path
    summary: dict[str, Any]


def _setup_logging(config: SourceDicsConfig, *, verbose_override: bool | None = None) -> None:
    level = logging.INFO if (config.logging.verbose if verbose_override is None else verbose_override) else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s", force=True)


def _format_config_summary(config: SourceDicsConfig) -> str:
    band_summary = ", ".join(
        f"{name}={band.fmin:.1f}-{band.fmax:.1f}Hz" for name, band in config.dics.bands.items()
    )
    tested_effect = (
        config.lmeeeg.contrast_of_interest
        or ", ".join(config.lmeeeg.test_predictors)
        or "unspecified"
    )
    return (
        f"config={config.config_path}\n"
        f"epochs_dir={config.paths.epochs_dir}\n"
        f"events_csv={config.paths.events_csv}\n"
        f"source_dir={config.paths.source_dir}\n"
        f"derivatives_dir={config.paths.derivatives_dir}\n"
        f"filter_window={config.dics.filter_tmin:.3f}..{config.dics.filter_tmax:.3f}s\n"
        f"analysis_window={config.dics.analysis_tmin:.3f}..{config.dics.analysis_tmax:.3f}s\n"
        f"bands={band_summary}\n"
        f"lmeeeg_tested_effect={tested_effect}"
    )


def _collect_missing_inputs(config: SourceDicsConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, path in (("events_csv", config.paths.events_csv),):
        if not path.exists():
            rows.append({"input": label, "path": str(path), "reason": "missing"})
    has_epochs_root = config.paths.epochs_dir.exists()
    has_raw_fallback = (
        config.paths.preprocessed_eeg_root is not None
        and config.paths.preprocessed_eeg_root.exists()
        and config.paths.bids_root is not None
        and config.paths.bids_root.exists()
    )
    if not has_epochs_root and not has_raw_fallback:
        rows.append(
            {
                "input": "epochs_or_raw_inputs",
                "path": (
                    f"epochs_dir={config.paths.epochs_dir}; "
                    f"preprocessed_eeg_root={config.paths.preprocessed_eeg_root}; "
                    f"bids_root={config.paths.bids_root}"
                ),
                "reason": "need either readable epochs_dir or both preprocessed_eeg_root and bids_root",
            }
        )
    if config.source_space.forward_template is not None and not config.source_space.forward_template.exists():
        rows.append(
            {
                "input": "source_space.forward_template",
                "path": str(config.source_space.forward_template),
                "reason": "missing",
            }
        )
    if config.source_space.forward_template is None and not config.source_space.subjects_dir.exists():
        rows.append(
            {
                "input": "source_space.subjects_dir",
                "path": str(config.source_space.subjects_dir),
                "reason": "missing and needed for fsaverage template-forward fallback",
            }
        )
    return rows


def _select_records(records: list[EpochRecord], *, subject_filter: set[str] | None) -> list[EpochRecord]:
    if subject_filter is None:
        return records
    return [record for record in records if record.subject_id in subject_filter]


def _load_subject_to_dyad_map() -> dict[str, str]:
    import re

    mapping: dict[str, str] = {}
    dyads_csv_path = Path(__file__).resolve().parents[3] / "config" / "dyads.csv"
    if dyads_csv_path.exists():
        table = pd.read_csv(dyads_csv_path)
        if {"subject_id", "dyad_id"}.issubset(table.columns):
            for row in table.itertuples(index=False):
                mapping[str(row.subject_id)] = str(row.dyad_id)

    log_roots = [
        Path("/Users/hiro/Datasets/Miscellaneous/speech-rate-testing/logs/evoked_trf"),
        Path("/Users/hiro/Datasets/Miscellaneous/speech-rate-testing/logs/features_emg"),
    ]
    pattern = re.compile(r"(?P<dyad>dyad-\d+)_sub-(?P<subject>\d+)\.log$")
    for log_root in log_roots:
        if not log_root.exists():
            continue
        for path in log_root.glob("*.log"):
            match = pattern.match(path.name)
            if match is None:
                continue
            mapping[f"sub-{match.group('subject')}"] = match.group("dyad")
    return mapping


def _find_conversation_anchor_time_s(raw_source) -> float:
    import mne

    anchor_kwargs = {"shortest_event": 1, "verbose": "ERROR"}
    if "Status" in raw_source.ch_names:
        anchor_kwargs["stim_channel"] = "Status"
    anchor_events = mne.find_events(raw_source, **anchor_kwargs)
    if anchor_events.size == 0:
        raise ValueError("No conversation-start trigger found in the raw BIDS EEG file.")
    anchor_sample = int(anchor_events[0, 0])
    return float((anchor_sample - raw_source.first_samp) / raw_source.info["sfreq"])


def _build_onset_locked_epochs(record: EpochRecord, *, events_table: pd.DataFrame, config: SourceDicsConfig, subject_to_dyad: dict[str, str]):
    import mne
    import numpy as np

    if record.preprocessed_eeg_path is None or record.raw_eeg_path is None:
        raise ValueError("Preprocessed/raw EEG paths are required for on-the-fly epoch construction.")

    raw = mne.io.read_raw_fif(record.preprocessed_eeg_path, preload=True, verbose="ERROR")
    raw_source = mne.io.read_raw(record.raw_eeg_path, preload=False, verbose="ERROR")
    anchor_time_s = _find_conversation_anchor_time_s(raw_source)
    recording_id = subject_to_dyad.get(record.subject_id)
    if recording_id is None:
        raise ValueError(
            f"No dyad/recording mapping found for {record.subject_id}. "
            "Provide a full mapping in config/dyads.csv or a discoverable log-based fallback."
        )

    recording_events = events_table.loc[
        (events_table["recording_id"].astype(str) == recording_id)
        & (events_table["run"].astype(str) == str(int(record.run_id)))
    ].copy()
    metadata_rows: list[dict[str, Any]] = []
    for row in recording_events.to_dict(orient="records"):
        for anchor_type, onset_column, duration_column, label_column in (
            ("FPP", "fpp_onset", "fpp_duration", "fpp_label"),
            ("SPP", "spp_onset", "spp_duration", "spp_label"),
        ):
            onset_s = pd.to_numeric(pd.Series([row.get(onset_column)]), errors="coerce").iloc[0]
            if pd.isna(onset_s):
                continue
            metadata_rows.append(
                {
                    "subject": record.subject_id,
                    "subject_id": record.subject_id,
                    "dyad": recording_id,
                    "dyad_id": recording_id,
                    "run": int(record.run_id),
                    "anchor_type": anchor_type,
                    "onset": float(onset_s),
                    "duration": pd.to_numeric(pd.Series([row.get(duration_column)]), errors="coerce").iloc[0],
                    "latency": pd.to_numeric(pd.Series([row.get("latency")]), errors="coerce").iloc[0],
                    "label": row.get(label_column),
                    "pair_id": row.get("pair_id"),
                    "recording_id": row.get("recording_id"),
                    "part": row.get("part"),
                    "response": row.get("response"),
                    "speaker_fpp": row.get("speaker_fpp"),
                    "speaker_spp": row.get("speaker_spp"),
                    "fpp_label": row.get("fpp_label"),
                    "spp_label": row.get("spp_label"),
                    "fpp_duration": row.get("fpp_duration"),
                    "spp_duration": row.get("spp_duration"),
                    "fpp_onset": row.get("fpp_onset"),
                    "spp_onset": row.get("spp_onset"),
                    "time_within_run": float(onset_s),
                    "event_id": f"{row.get('pair_id')}_{anchor_type.lower()}",
                    "event_onset_s": float(onset_s) + anchor_time_s,
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    if metadata.empty:
        raise ValueError(f"No canonical FPP/SPP events found for {record.subject_id} run={record.run_id} ({recording_id}).")
    metadata = metadata.sort_values(["event_onset_s", "anchor_type", "pair_id"], kind="mergesort").reset_index(drop=True)
    onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
    metadata = metadata.loc[
        (metadata["event_onset_s"] >= 0.0)
        & (onset_samples >= raw.first_samp)
        & (onset_samples <= raw.last_samp)
    ].reset_index(drop=True)
    onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
    sample_events = np.empty((len(metadata), 3), dtype=int)
    sample_events[:, 0] = onset_samples
    sample_events[:, 1] = 0
    event_id = {"FPP": 1, "SPP": 2}
    sample_events[:, 2] = metadata["anchor_type"].map(event_id).to_numpy(dtype=int)
    baseline = config.epoching.baseline
    epochs = mne.Epochs(
        raw,
        sample_events,
        event_id=event_id,
        tmin=config.epoching.tmin,
        tmax=config.epoching.tmax,
        baseline=baseline,
        metadata=metadata,
        preload=True,
        reject_by_annotation=bool(config.epoching.reject_by_annotation),
        event_repeated="drop",
        verbose="ERROR",
    )
    return epochs


def run_source_dics_pipeline(
    config: SourceDicsConfig,
    *,
    subjects: list[str] | None = None,
    bands: list[str] | None = None,
    overwrite: bool | None = None,
    verbose: bool | None = None,
) -> SourceDicsPipelineResult:
    """Run the pooled FPP/SPP source-level DICS power pipeline.

    Usage example
    -------------
    >>> cfg = load_source_dics_config("config/source_dics_fpp_spp_alpha_beta.yaml")  # doctest: +SKIP
    >>> run_source_dics_pipeline(cfg)  # doctest: +SKIP
    """

    configure_mne_runtime()
    _setup_logging(config, verbose_override=verbose)
    LOGGER.info("Loading source DICS config and writing QC snapshot.")
    LOGGER.info("Resolved config summary:\n%s", _format_config_summary(config))
    write_config_snapshot(config.to_dict(), qc_dir=config.paths.qc_dir)
    write_filter_windows(
        qc_dir=config.paths.qc_dir,
        filter_tmin=config.dics.filter_tmin,
        filter_tmax=config.dics.filter_tmax,
        analysis_tmin=config.dics.analysis_tmin,
        analysis_tmax=config.dics.analysis_tmax,
    )

    selected_bands = list(config.dics.bands.keys()) if bands is None else bands
    missing_inputs = _collect_missing_inputs(config)
    if missing_inputs:
        write_missing_inputs(missing_inputs, qc_dir=config.paths.qc_dir)
        raise FileNotFoundError(
            "Source DICS input validation failed. See results/source_dics_fpp_spp_alpha_beta/qc/missing_inputs.csv."
        )

    LOGGER.info("Discovering subjects/runs for source DICS processing.")
    subject_filter = None if subjects is None else set(subjects)
    if config.paths.preprocessed_eeg_root is not None and config.paths.bids_root is not None:
        records = _select_records(
            discover_preprocessed_records(
                config.paths.preprocessed_eeg_root,
                bids_root=config.paths.bids_root,
                subjects=subject_filter,
            ),
            subject_filter=subject_filter,
        )
        LOGGER.info(
            "Using on-the-fly epoch construction from preprocessed EEG in %s.",
            config.paths.preprocessed_eeg_root,
        )
    else:
        records = _select_records(
            discover_epoch_records(config.paths.epochs_dir, subjects=subject_filter),
            subject_filter=subject_filter,
        )
    if not records:
        raise FileNotFoundError("No usable source DICS inputs were discovered.")

    LOGGER.info("Loading canonical FPP/SPP events from %s.", config.paths.events_csv)
    events_table = load_events_table(config.paths.events_csv)
    subject_to_dyad = _load_subject_to_dyad_map()

    all_metadata_rows: list[pd.DataFrame] = []
    band_subject_rows: list[dict[str, Any]] = []
    mean_power_rows: list[pd.DataFrame] = []
    subject_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    forward = None

    subject_progress = tqdm(
        records,
        desc="Source DICS subjects/runs",
        disable=not config.logging.progress,
    )
    for record in subject_progress:
        try:
            LOGGER.info("Preparing epochs for %s run=%s.", record.subject_id, record.run_id)
            if record.preprocessed_eeg_path is not None:
                epochs = _build_onset_locked_epochs(
                    record,
                    events_table=events_table,
                    config=config,
                    subject_to_dyad=subject_to_dyad,
                )
            elif record.epochs_path is not None:
                epochs = load_epochs(record.epochs_path)
            else:
                raise ValueError(f"No valid EEG source paths were available for {record.subject_id} run={record.run_id}.")
            if epochs.metadata is None:
                raise ValueError(
                    "The prepared epochs do not contain metadata required to distinguish FPP vs SPP epochs."
                )
            if float(epochs.times[0]) > config.epoching.tmin or float(epochs.times[-1]) < config.epoching.tmax:
                raise ValueError(
                    "The prepared epochs do not span the requested full epoch window "
                    f"{config.epoching.tmin}..{config.epoching.tmax} s."
                )
            if forward is None:
                LOGGER.info("Resolving forward model for %s run=%s.", record.subject_id, record.run_id)
                forward = resolve_forward_model(config, info=epochs.info)

            normalized_metadata = prepare_epoch_metadata(
                epochs.metadata,
                events_table=events_table,
                events_config=config.events,
                record=record,
            )
            if normalized_metadata.empty:
                LOGGER.warning("Skipping %s run=%s because no FPP/SPP events survived metadata normalization.", record.subject_id, record.run_id)
                continue
            all_metadata_rows.append(normalized_metadata)
            anchor_selections = split_anchor_metadata(normalized_metadata)
            pooled_indices = normalized_metadata["_epoch_index"].to_numpy(dtype=int)
            pooled_epochs = epochs[pooled_indices]

            record_summary: dict[str, Any] = {
                "subject": record.subject_id,
                "run": record.run_id,
                "epochs_path": str(record.epochs_path) if record.epochs_path is not None else "",
                "preprocessed_eeg_path": str(record.preprocessed_eeg_path) if record.preprocessed_eeg_path is not None else "",
                "n_epochs_total": int(len(epochs)),
                "n_epochs_retained": int(len(pooled_epochs)),
                "anchors": {},
                "bands": {},
            }
            band_progress = tqdm(
                selected_bands,
                desc=f"{record.subject_id} run {record.run_id} bands",
                leave=False,
                disable=not config.logging.progress,
            )
            for band_name in band_progress:
                if band_name not in config.dics.bands:
                    raise ValueError(f"Unknown band {band_name!r}; configured bands are {sorted(config.dics.bands)}.")
                band = config.dics.bands[band_name]
                LOGGER.info(
                    "Computing CSD and common DICS filters for %s run=%s band=%s.",
                    record.subject_id,
                    record.run_id,
                    band_name,
                )
                filters = compute_common_dics_filters(
                    pooled_epochs,
                    band_name=band_name,
                    band=band,
                    forward=forward,
                    config=config,
                )
                if config.output.save_filters:
                    filter_path = config.paths.filters_dir / f"{record.subject_id}_run-{record.run_id}_band-{band_name}-dics.h5"
                    save_filters(filters, filter_path, overwrite=bool(overwrite if overwrite is not None else config.output.overwrite))
                for anchor_type, selection in anchor_selections.items():
                    anchor_epochs = epochs[selection.epoch_indices]
                    LOGGER.info(
                        "Applying pooled band-specific filters to %s run=%s anchor=%s (%d epochs).",
                        record.subject_id,
                        record.run_id,
                        anchor_type,
                        len(anchor_epochs),
                    )
                    result = apply_common_filters_to_epochs(
                        anchor_epochs,
                        filters=filters,
                        forward=forward,
                        anchor_type=anchor_type,
                        band_name=band_name,
                        band=band,
                        config=config,
                    )
                    if config.output.save_trial_power:
                        export_trial_power(result, selection.metadata, record=record, config=config)
                    export_metadata(selection.metadata, record=record, result=result, config=config)
                    long_table_paths: tuple[Path, ...] = ()
                    if config.output.save_long_table:
                        long_table_paths = export_long_table(result, selection.metadata, record=record, config=config)
                    mean_power_rows.append(summarize_mean_power(result, selection.metadata))
                    band_subject_rows.append(
                        {
                            "band": band_name,
                            "subject": record.subject_id,
                            "run": record.run_id,
                            "anchor_type": anchor_type,
                            "n_events": int(len(selection.metadata)),
                            "n_sources": int(len(result.source_ids)),
                            "n_times": int(len(result.times)),
                            "n_long_table_parts": int(len(long_table_paths)),
                        }
                    )
                    record_summary["anchors"][anchor_type] = int(len(selection.metadata))
                    record_summary["bands"].setdefault(band_name, {})[anchor_type] = {
                        "n_events": int(len(selection.metadata)),
                        "n_sources": int(len(result.source_ids)),
                        "n_times": int(len(result.times)),
                    }
            subject_summaries.append(record_summary)
        except Exception as exc:  # noqa: BLE001
            if "No canonical FPP/SPP events found" in str(exc):
                LOGGER.warning(
                    "Skipping %s run=%s because the canonical events table has no matching FPP/SPP rows: %s",
                    record.subject_id,
                    record.run_id,
                    exc,
                )
                continue
            LOGGER.exception("Source DICS processing failed for %s run=%s.", record.subject_id, record.run_id)
            failures.append(
                {
                    "subject": record.subject_id,
                    "run": record.run_id,
                    "epochs_path": str(record.epochs_path) if record.epochs_path is not None else "",
                    "error": str(exc),
                }
            )

    if all_metadata_rows:
        combined_metadata = pd.concat(all_metadata_rows, ignore_index=True)
        write_event_counts(combined_metadata, qc_dir=config.paths.qc_dir)
    write_band_subject_counts(band_subject_rows, qc_dir=config.paths.qc_dir)
    write_mean_power(mean_power_rows, qc_dir=config.paths.qc_dir)

    summary = {
        "status": "failed" if failures else "ok",
        "config_path": str(config.config_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_subject_runs_discovered": int(len(records)),
        "n_subject_runs_processed": int(len(subject_summaries)),
        "bands": selected_bands,
        "anchor_types": list(config.events.anchor_types),
        "common_filter_guardrail": "Common DICS filters were estimated from pooled FPP+SPP data only.",
        "post_onset_guardrail": "Samples with t > 0 were excluded from filter estimation and final exported statistics.",
        "lmeeeg_model_spec": {
            "enabled": bool(config.lmeeeg.enabled),
            "dependent_variable": config.lmeeeg.dependent_variable,
            "formula": config.lmeeeg.formula,
            "predictors": list(config.lmeeeg.predictors),
            "test_predictors": list(config.lmeeeg.test_predictors),
            "contrast_of_interest": config.lmeeeg.contrast_of_interest,
        },
        "forward_template": None if config.source_space.forward_template is None else str(config.source_space.forward_template),
        "subject_runs": subject_summaries,
        "failures": failures,
    }
    summary_path = write_run_summary(summary, qc_dir=config.paths.qc_dir)
    if failures:
        raise RuntimeError(
            "Source DICS pipeline finished with failures. Inspect "
            f"{summary_path} for details."
        )
    return SourceDicsPipelineResult(
        out_dir=config.paths.derivatives_dir,
        summary_path=summary_path,
        summary=summary,
    )


def load_source_dics_pipeline_config(config_path: str | Path) -> SourceDicsConfig:
    return load_source_dics_config(config_path)
