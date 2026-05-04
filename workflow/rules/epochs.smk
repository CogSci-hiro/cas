EPOCHS_SETTINGS = EPOCHS_CONFIG["epochs"]
EPOCHS_OUTPUT_ROOT = os.path.join(OUT_DIR, EPOCHS_SETTINGS["output"]["root"])
EPOCHS_OUTPUT_PATTERN = (
    f"{EPOCHS_OUTPUT_ROOT}/sub-{{subject}}/eeg/"
    f"sub-{{subject}}_task-{{task}}_run-{{run}}_desc-tasklocked_epo.fif"
)
INDUCED_SOURCE_EPOCHS_SETTINGS = dict(EPOCHS_CONFIG.get("induced_source_epochs") or {})
INDUCED_SOURCE_EPOCHS_OUTPUT_ROOT = os.path.join(
    OUT_DIR,
    str((INDUCED_SOURCE_EPOCHS_SETTINGS.get("output") or {}).get("root", "induced_source_epochs")),
)
INDUCED_SOURCE_EPOCHS_OUTPUT_PATTERN = (
    f"{INDUCED_SOURCE_EPOCHS_OUTPUT_ROOT}/sub-{{subject}}/eeg/"
    f"sub-{{subject}}_task-{{task}}_run-{{run}}_desc-tasklocked_epo.fif"
)


def _normalize_run_label(run: str) -> str:
    return str(int(run))


def _subject_speaker_label(subject: str) -> str:
    return "A" if int(subject) % 2 == 1 else "B"


def _other_speaker_label(subject: str) -> str:
    return "B" if _subject_speaker_label(subject) == "A" else "A"


def _subject_dyad_id(subject: str) -> str:
    subject_id = f"sub-{subject}"
    dyad_id = SUBJECT_TO_DYAD_MAP.get(subject_id)
    if dyad_id:
        return dyad_id

    subject_number = int(subject)
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _epochs_recording_id(subject: str) -> str:
    return _subject_dyad_id(subject)


def _epoch_annotation_path(subject: str, run: str) -> str | None:
    annotation_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{_subject_dyad_id(subject)}_run-{run}_combined.TextGrid",
    )
    if os.path.exists(annotation_path):
        return annotation_path
    return None


def epoch_annotation_input(wildcards):
    annotation_path = _epoch_annotation_path(wildcards.subject, wildcards.run)
    if annotation_path is None:
        subject_id = f"sub-{wildcards.subject}"
        raise FileNotFoundError(
            f"No combined annotation file found for {subject_id}, task {wildcards.task}, run {wildcards.run}."
        )
    return annotation_path


def _discover_epoch_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for record in PREPROCESSED_EEG_RECORDS:
        if _epoch_annotation_path(record["subject"], record["run"]) is None:
            continue
        records.append(record)
    return records


EPOCH_RECORDS = _discover_epoch_records()
EPOCH_OUTPUTS = expand(
    EPOCHS_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in EPOCH_RECORDS],
    task=[record["task"] for record in EPOCH_RECORDS],
    run=[record["run"] for record in EPOCH_RECORDS],
)
INDUCED_SOURCE_EPOCH_OUTPUTS = expand(
    INDUCED_SOURCE_EPOCHS_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in EPOCH_RECORDS],
    task=[record["task"] for record in EPOCH_RECORDS],
    run=[record["run"] for record in EPOCH_RECORDS],
)
INDUCED_EPOCH_BANDS = [
    str(value)
    for value in dict(EPOCHS_CONFIG.get("induced_epochs") or {}).get("bands", ["theta"])
]


def _induced_epoch_summary_outputs() -> list[str]:
    subjects = sorted({record["subject"] for record in EPOCH_RECORDS})
    return [
        f"{OUT_DIR}/induced_epochs/sub-{subject}/summary.json"
        for subject in subjects
    ]


def _induced_epoch_band_summary_outputs(config_path: str | None = None) -> list[str]:
    subjects = sorted({record["subject"] for record in EPOCH_RECORDS})
    resolved_config_path = config_path or LMEEEG_CONFIG_PATH
    requested_bands = [
        str(value)
        for value in dict(_load_lmeeeg_workflow_config(resolved_config_path).get("induced_epochs") or {}).get(
            "bands",
            INDUCED_EPOCH_BANDS,
        )
    ]
    return [
        f"{OUT_DIR}/induced_epochs/{band}/sub-{subject}/epoching_summary-time_s.json"
        for subject in subjects
        for band in requested_bands
    ]


def _epoching_settings(section_name: str) -> dict[str, object]:
    if section_name == "induced_source_epochs":
        settings = dict(INDUCED_SOURCE_EPOCHS_SETTINGS)
        if "event_types" not in settings:
            settings["event_types"] = list(EPOCHS_SETTINGS.get("event_types", ()))
        if "reject_by_annotation" not in settings:
            settings["reject_by_annotation"] = bool(EPOCHS_SETTINGS.get("reject_by_annotation", True))
        if "timing" not in settings:
            settings["timing"] = dict(EPOCHS_SETTINGS.get("timing", {}))
        return settings
    return dict(EPOCHS_SETTINGS)


def _normalize_baseline(baseline_s, *, tmin_s: float, tmax_s: float):
    if baseline_s is None:
        return None

    start, end = (
        None if value is None else float(value)
        for value in baseline_s
    )
    if start is not None and end is not None and start > end:
        start, end = end, start

    if start is not None:
        start = max(start, tmin_s)
    if end is not None:
        end = min(end, tmax_s)

    if start is not None and end is not None and start >= end:
        return None
    return (start, end)


def _write_tasklocked_epochs(*, input_eeg, input_raw, events_csv, output_epochs, subject, task, run, settings):
    import numpy as np
    import pandas as pd
    import mne

    raw = mne.io.read_raw_fif(input_eeg, preload=True, verbose="ERROR")
    raw_source = mne.io.read_raw(input_raw, preload=False, verbose="ERROR")
    events_df = pd.read_csv(events_csv)

    status_channel = "Status" if "Status" in raw_source.ch_names else None
    anchor_kwargs = {"shortest_event": 1, "verbose": "ERROR"}
    if status_channel is not None:
        anchor_kwargs["stim_channel"] = status_channel
    anchor_events = mne.find_events(raw_source, **anchor_kwargs)
    if anchor_events.size == 0:
        raise ValueError(
            f"No conversation-start trigger found for sub-{subject}, task {task}, run {run}."
        )

    anchor_sample = int(anchor_events[0, 0])
    anchor_time_s = float((anchor_sample - raw_source.first_samp) / raw_source.info["sfreq"])

    recording_id = _epochs_recording_id(subject)
    run_label = _normalize_run_label(run)
    self_speaker_label = _subject_speaker_label(subject)
    other_speaker_label = _other_speaker_label(subject)
    event_types = tuple(settings.get("event_types", ("fpp", "spp")))

    recording_events = events_df.loc[
        (events_df["recording_id"] == recording_id)
        & (events_df["run"].astype(str) == run_label)
    ].copy()

    metadata_frames = []
    for event_type in event_types:
        if event_type == "self_fpp_onset":
            speaker_column = "speaker_fpp"
            onset_column = "fpp_onset"
            offset_column = "fpp_offset"
            speaker_value = self_speaker_label
            event_role = "self"
            event_family = "fpp"
            event_lock = "onset"
        elif event_type == "self_fpp_offset":
            speaker_column = "speaker_fpp"
            onset_column = "fpp_onset"
            offset_column = "fpp_offset"
            speaker_value = self_speaker_label
            event_role = "self"
            event_family = "fpp"
            event_lock = "offset"
        elif event_type == "other_fpp_onset":
            speaker_column = "speaker_fpp"
            onset_column = "fpp_onset"
            offset_column = "fpp_offset"
            speaker_value = other_speaker_label
            event_role = "other"
            event_family = "fpp"
            event_lock = "onset"
        elif event_type == "other_fpp_offset":
            speaker_column = "speaker_fpp"
            onset_column = "fpp_onset"
            offset_column = "fpp_offset"
            speaker_value = other_speaker_label
            event_role = "other"
            event_family = "fpp"
            event_lock = "offset"
        elif event_type == "self_spp_onset":
            speaker_column = "speaker_spp"
            onset_column = "spp_onset"
            offset_column = "spp_offset"
            speaker_value = self_speaker_label
            event_role = "self"
            event_family = "spp"
            event_lock = "onset"
        elif event_type == "self_spp_offset":
            speaker_column = "speaker_spp"
            onset_column = "spp_onset"
            offset_column = "spp_offset"
            speaker_value = self_speaker_label
            event_role = "self"
            event_family = "spp"
            event_lock = "offset"
        elif event_type == "other_spp_onset":
            speaker_column = "speaker_spp"
            onset_column = "spp_onset"
            offset_column = "spp_offset"
            speaker_value = other_speaker_label
            event_role = "other"
            event_family = "spp"
            event_lock = "onset"
        elif event_type == "other_spp_offset":
            speaker_column = "speaker_spp"
            onset_column = "spp_onset"
            offset_column = "spp_offset"
            speaker_value = other_speaker_label
            event_role = "other"
            event_family = "spp"
            event_lock = "offset"
        else:
            raise ValueError(f"Unsupported epoch event type: {event_type}")

        selected = recording_events.loc[
            recording_events[speaker_column].astype(str) == speaker_value
        ].copy()
        if selected.empty:
            continue

        selected["event_type"] = event_type
        selected["event_role"] = event_role
        selected["event_family"] = event_family
        selected["event_lock"] = event_lock
        selected["event_speaker_label"] = speaker_value
        selected["event_source_column"] = onset_column if event_lock == "onset" else offset_column
        selected["event_source_offset_column"] = offset_column
        selected["event_onset_conversation_s"] = pd.to_numeric(selected[onset_column], errors="coerce")
        selected["event_offset_conversation_s"] = pd.to_numeric(selected[offset_column], errors="coerce")
        selected["conversation_anchor_sample"] = anchor_sample
        selected["conversation_anchor_time_s"] = anchor_time_s
        selected["event_latency_conversation_s"] = np.where(
            selected["event_lock"] == "onset",
            selected["event_onset_conversation_s"],
            selected["event_offset_conversation_s"],
        )
        selected["event_onset_s"] = selected["event_latency_conversation_s"] + anchor_time_s
        selected["event_offset_s"] = selected["event_offset_conversation_s"] + anchor_time_s
        selected["subject_id"] = f"sub-{subject}"
        selected["task"] = task
        selected["run"] = run_label
        selected["recording_id"] = recording_id
        selected["speaker_current"] = self_speaker_label
        selected["speaker_role"] = event_role
        metadata_frames.append(selected)

    if metadata_frames:
        metadata = pd.concat(metadata_frames, ignore_index=True)
        metadata = metadata.loc[metadata["event_onset_s"].notna()].copy()
        metadata = metadata.sort_values(["event_onset_s", "pair_id", "event_type"]).reset_index(drop=True)
    else:
        metadata = pd.DataFrame(columns=["event_type"])

    event_id = {event_type: index + 1 for index, event_type in enumerate(event_types)}
    if len(metadata):
        onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
        metadata = metadata.loc[
            (metadata["event_onset_s"] >= 0.0)
            & (onset_samples >= raw.first_samp)
            & (onset_samples <= raw.last_samp)
        ].reset_index(drop=True)
        sample_events = np.empty((len(metadata), 3), dtype=int)
        onset_samples = raw.first_samp + np.rint(metadata["event_onset_s"].to_numpy(dtype=float) * raw.info["sfreq"]).astype(int)
        sample_events[:, 0] = onset_samples
        sample_events[:, 1] = 0
        sample_events[:, 2] = metadata["event_type"].map(event_id).to_numpy(dtype=int)
    else:
        sample_events = np.empty((0, 3), dtype=int)

    tmin_s = float(settings["timing"]["tmin_s"])
    tmax_s = float(settings["timing"]["tmax_s"])
    baseline_s = settings.get("timing", {}).get("baseline_s")
    baseline = _normalize_baseline(baseline_s, tmin_s=tmin_s, tmax_s=tmax_s)

    if len(metadata):
        observed_event_types = list(dict.fromkeys(metadata["event_type"].tolist()))
        observed_event_id = {event_type: event_id[event_type] for event_type in observed_event_types}
        epochs = mne.Epochs(
            raw,
            sample_events,
            event_id=observed_event_id,
            tmin=tmin_s,
            tmax=tmax_s,
            baseline=baseline,
            metadata=metadata,
            preload=True,
            reject_by_annotation=bool(settings.get("reject_by_annotation", True)),
            event_repeated="drop",
            verbose="ERROR",
        )
    else:
        dummy_sample = max(0, int(np.ceil(abs(tmin_s) * raw.info["sfreq"])))
        dummy_events = np.array([[dummy_sample, 0, 1]], dtype=int)
        dummy_metadata = pd.DataFrame([{"event_type": "__empty__"}])
        epochs = mne.Epochs(
            raw,
            dummy_events,
            event_id={"__empty__": 1},
            tmin=tmin_s,
            tmax=tmax_s,
            baseline=baseline,
            metadata=dummy_metadata,
            preload=True,
            reject_by_annotation=bool(settings.get("reject_by_annotation", True)),
            event_repeated="drop",
            verbose="ERROR",
        )
        epochs.drop([0], reason="no_matching_events")

    os.makedirs(os.path.dirname(output_epochs), exist_ok=True)
    epochs.save(output_epochs, overwrite=True)


LMEEEG_CONFIG_PATH = f"{CONFIG_DIR}/induced/alpha_beta_lmeeeg.yaml"
FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH = f"{CONFIG_DIR}/induced/alpha_beta_cycle_position.yaml"
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG_PATH = (
    f"{CONFIG_DIR}/induced/alpha_beta_conf_disc.yaml"
)
INFO_RATE_INDUCED_LMEEEG_CONFIG_PATH = f"{CONFIG_DIR}/induced/info_rate_induced_lmeeg.yaml"
LMEEEG_OUTPUT_DIR = f"{OUT_DIR}/lmeeeg"
LMEEEG_SUMMARY_OUTPUT = f"{LMEEEG_OUTPUT_DIR}/lmeeeg_analysis_summary.json"
LMEEEG_INDUCED_SUMMARY_OUTPUT = f"{LMEEEG_OUTPUT_DIR}/induced_lmeeeg_analysis_summary.json"


def _load_lmeeeg_workflow_config(config_path: str = LMEEEG_CONFIG_PATH) -> dict[str, object]:
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    section = payload.get("lmeeeg", payload)
    if not isinstance(section, dict):
        raise ValueError(f"`lmeeeg` in {config_path} must be a mapping.")
    loaded = dict(section)
    if "induced_epochs" in payload:
        loaded["induced_epochs"] = payload["induced_epochs"]
    return loaded


def _induced_lmeeeg_model_names(config_path: str = LMEEEG_CONFIG_PATH) -> list[str]:
    models_cfg = dict(_load_lmeeeg_workflow_config(config_path).get("models") or {})
    selected = [
        str(model_name)
        for model_name, model_cfg in models_cfg.items()
        if str((model_cfg or {}).get("modality", "evoked")).strip().lower() == "induced"
    ]
    return sorted(selected)


def _evoked_lmeeeg_model_names(config_path: str = LMEEEG_CONFIG_PATH) -> list[str]:
    models_cfg = dict(_load_lmeeeg_workflow_config(config_path).get("models") or {})
    selected = [
        str(model_name)
        for model_name, model_cfg in models_cfg.items()
        if str((model_cfg or {}).get("modality", "evoked")).strip().lower() != "induced"
    ]
    return sorted(selected)


def _lmeeeg_analysis_root_from_config(config_path: str) -> str:
    config_payload = _load_lmeeeg_workflow_config(config_path)
    analysis_name = str(config_payload.get("analysis_name", "")).strip()
    return f"{OUT_DIR}/lmeeeg/{analysis_name}" if analysis_name else f"{OUT_DIR}/lmeeeg"


def _resolve_erp_output_dir(path_value: str | None, default_subdir: str) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return f"{OUT_DIR}/{default_subdir}"
    if os.path.isabs(path_value):
        return path_value
    return f"{OUT_DIR}/{path_value.lstrip('/')}"


def _resolve_erp_input_path(path_value: str | None, default_relative: str) -> str:
    path_text = default_relative if not isinstance(path_value, str) or not path_value.strip() else path_value
    if os.path.isabs(path_text):
        return path_text
    project_candidate = os.path.join(PROJECT_ROOT, path_text.lstrip("/"))
    if os.path.exists(project_candidate):
        return project_candidate
    return os.path.join(OUT_DIR, path_text.lstrip("/"))


LMEEEG_EVOKED_MODEL_NAMES = _evoked_lmeeeg_model_names()
LMEEEG_EVOKED_MODEL_SUMMARY_OUTPUTS = expand(
    f"{LMEEEG_OUTPUT_DIR}/{{model}}/summary.json",
    model=LMEEEG_EVOKED_MODEL_NAMES,
)
LMEEEG_EVOKED_DURATION_QC_OUTPUTS = [
    f"{LMEEEG_OUTPUT_DIR}/{model_name}/{artifact_name}"
    for model_name in LMEEEG_EVOKED_MODEL_NAMES
    for artifact_name in (
        "duration_summary_by_class.csv",
        "trial_counts_by_class.csv",
        "model_formula.txt",
        "design_columns.csv",
        "term_tests.json",
        "inference_summary.json",
    )
]


FPP_SPP_CYCLE_POSITION_LMEEEG_OUTPUT_DIR = _lmeeeg_analysis_root_from_config(
    FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH
)
FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT = (
    f"{FPP_SPP_CYCLE_POSITION_LMEEEG_OUTPUT_DIR}/lmeeeg_analysis_summary.json"
)
FPP_SPP_CYCLE_POSITION_LMEEEG_INDUCED_BANDS = _load_lmeeeg_workflow_config(
    FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH
).get("induced_epochs", {}).get("bands", ["theta"])
FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS = expand(
    (
        f"{FPP_SPP_CYCLE_POSITION_LMEEEG_OUTPUT_DIR}/induced/{{band}}/"
        "fpp_vs_spp_cycle_position/pair_positionFPP_beta.npy"
    ),
    band=[str(band) for band in FPP_SPP_CYCLE_POSITION_LMEEEG_INDUCED_BANDS],
)

FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_OUTPUT_DIR = _lmeeeg_analysis_root_from_config(
    FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG_PATH
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_SUMMARY_OUTPUT = (
    f"{FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_OUTPUT_DIR}/lmeeeg_analysis_summary.json"
)
_FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG = _load_lmeeeg_workflow_config(
    FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG_PATH
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INPUT = dict(
    _FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG.get("input") or {}
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_SUBDIR = str(
    FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INPUT.get(
        "induced_epochs_subdir",
        "induced_epochs_fpp_spp_conf_disc_alpha_beta_lmeeeg",
    )
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_BANDS = _FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG.get(
    "induced_epochs",
    {},
).get("bands", ["alpha", "beta"])
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODELS = sorted(
    [str(name) for name in dict(_FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG.get("models") or {}).keys()]
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODEL_SUMMARY_OUTPUTS = expand(
    (
        f"{FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_OUTPUT_DIR}/induced/{{band}}/"
        "{model}/summary.json"
    ),
    band=[str(band) for band in FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_BANDS],
    model=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODELS,
)
FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONTRAST_OUTPUTS = [
    (
        f"{FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_OUTPUT_DIR}/induced/{band}/"
        f"{model}/{effect}_beta.npy"
    )
    for band in [str(value) for value in FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_BANDS]
    for model in ("class_3", "class_3_plus_duration")
    for effect in ("class_3SPP_CONF", "class_3SPP_DISC")
]

with open(INFO_RATE_INDUCED_LMEEEG_CONFIG_PATH, encoding="utf-8") as _info_rate_handle:
    _INFO_RATE_INDUCED_LMEEEG_CONFIG = yaml.safe_load(_info_rate_handle) or {}
_INFO_RATE_INDUCED_LMEEEG_INPUT = dict(_INFO_RATE_INDUCED_LMEEEG_CONFIG.get("input") or {})
_INFO_RATE_INDUCED_LMEEEG_OUTPUT = dict(_INFO_RATE_INDUCED_LMEEEG_CONFIG.get("output") or {})
INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR = _resolve_erp_output_dir(
    _INFO_RATE_INDUCED_LMEEEG_OUTPUT.get("out_dir"),
    "results/info_rate_induced_lmeeg",
)
INFO_RATE_INDUCED_LMEEEG_BEHAVIOUR_RISKSET_INPUT = _resolve_erp_input_path(
    _INFO_RATE_INDUCED_LMEEEG_INPUT.get("behaviour_riskset_path"),
    "behavior/hazard/risksets/pooled_fpp_spp.parquet",
)
INFO_RATE_INDUCED_LMEEEG_OUTPUTS = [
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/induced_power_trials.parquet",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/model_input_binned.parquet",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/model_results_long.csv",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/model_diagnostics.csv",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/config_used.yaml",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/alpha_anchor_x_information_rate_tmap.png",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/beta_anchor_x_information_rate_tmap.png",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/alpha_anchor_x_prop_expected_cum_info_tmap.png",
    f"{INFO_RATE_INDUCED_LMEEEG_OUTPUT_DIR}/beta_anchor_x_prop_expected_cum_info_tmap.png",
]


rule make_epochs:
    input:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        raw=preprocess_raw_input,
        annotation=epoch_annotation_input,
        events_csv=EVENTS_CSV_OUTPUT,
    output:
        epochs=EPOCHS_OUTPUT_PATTERN,
    run:
        _write_tasklocked_epochs(
            input_eeg=input.eeg,
            input_raw=input.raw,
            events_csv=input.events_csv,
            output_epochs=output.epochs,
            subject=wildcards.subject,
            task=wildcards.task,
            run=wildcards.run,
            settings=_epoching_settings("epochs"),
        )


rule make_induced_source_epochs:
    input:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        raw=preprocess_raw_input,
        annotation=epoch_annotation_input,
        events_csv=EVENTS_CSV_OUTPUT,
    output:
        epochs=INDUCED_SOURCE_EPOCHS_OUTPUT_PATTERN,
    run:
        _write_tasklocked_epochs(
            input_eeg=input.eeg,
            input_raw=input.raw,
            events_csv=input.events_csv,
            output_epochs=output.epochs,
            subject=wildcards.subject,
            task=wildcards.task,
            run=wildcards.run,
            settings=_epoching_settings("induced_source_epochs"),
        )


rule run_lmeeeg:
    input:
        epochs=EPOCH_OUTPUTS,
        config=LMEEEG_CONFIG_PATH,
    output:
        summary=LMEEEG_SUMMARY_OUTPUT,
        model_summaries=LMEEEG_EVOKED_MODEL_SUMMARY_OUTPUTS,
        qc=LMEEEG_EVOKED_DURATION_QC_OUTPUTS,
    run:
        import os
        import tempfile

        from cas.stats.lmeeeg_pipeline import run_pooled_lmeeeg_analysis

        config_payload = _load_lmeeeg_workflow_config()
        evoked_models = {
            model_name: model_cfg
            for model_name, model_cfg in dict(config_payload.get("models") or {}).items()
            if str((model_cfg or {}).get("modality", "evoked")).strip().lower() != "induced"
        }
        if not evoked_models:
            raise ValueError(
                "No evoked lmeEEG models are configured in config/induced/alpha_beta_lmeeeg.yaml."
            )

        evoked_config_payload = {
            "lmeeeg": {
                key: value
                for key, value in config_payload.items()
                if key not in {"models", "induced_epochs"}
            }
        }
        evoked_config_payload["lmeeeg"]["models"] = evoked_models

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
            yaml.safe_dump(evoked_config_payload, handle, sort_keys=False)
            temp_config_path = handle.name

        try:
            run_pooled_lmeeeg_analysis(
                epochs_paths=list(input.epochs),
                config_path=temp_config_path,
                output_dir=os.path.dirname(output.summary),
            )
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)


rule run_induced_lmeeeg:
    input:
        epochs=EPOCH_OUTPUTS,
        induced_subjects=_induced_epoch_summary_outputs(),
        induced_bands=_induced_epoch_band_summary_outputs(),
        config=LMEEEG_CONFIG_PATH,
    output:
        summary=LMEEEG_INDUCED_SUMMARY_OUTPUT,
    run:
        import json
        import os
        import tempfile
        from pathlib import Path

        from cas.stats.lmeeeg_pipeline import run_pooled_lmeeeg_analysis

        config_payload = _load_lmeeeg_workflow_config()
        induced_models = {
            model_name: model_cfg
            for model_name, model_cfg in dict(config_payload.get("models") or {}).items()
            if str((model_cfg or {}).get("modality", "evoked")).strip().lower() == "induced"
        }
        if not induced_models:
            raise ValueError(
                "No induced lmeEEG models are configured in config/induced/alpha_beta_lmeeeg.yaml."
            )

        induced_config_payload = {
            "lmeeeg": {
                key: value
                for key, value in config_payload.items()
                if key not in {"models", "induced_epochs"}
            }
        }
        induced_config_payload["lmeeeg"]["models"] = induced_models
        if "induced_epochs" in config_payload:
            induced_config_payload["induced_epochs"] = config_payload["induced_epochs"]

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
            yaml.safe_dump(induced_config_payload, handle, sort_keys=False)
            temp_config_path = handle.name

        try:
            summary = run_pooled_lmeeeg_analysis(
                epochs_paths=list(input.epochs),
                config_path=temp_config_path,
                output_dir=LMEEEG_OUTPUT_DIR,
            )
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

        Path(output.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(output.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


rule run_fpp_spp_cycle_position_lmeeeg:
    input:
        epochs=EPOCH_OUTPUTS,
        induced=_induced_epoch_summary_outputs(),
        config=FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH,
    output:
        summary=FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        contrast=FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS,
    run:
        from cas.stats.lmeeeg_pipeline import run_pooled_lmeeeg_analysis

        # First-pass active-sensing cycle-position contrast:
        # FPP = initiation phase, SPP = response/update phase.
        # This is not by itself a pure motor-free belief-state test.
        run_pooled_lmeeeg_analysis(
            epochs_paths=list(input.epochs),
            config_path=input.config,
            output_dir=os.path.dirname(output.summary),
        )


rule run_info_rate_induced_lmeeg:
    input:
        epochs=INDUCED_SOURCE_EPOCH_OUTPUTS,
        config=INFO_RATE_INDUCED_LMEEEG_CONFIG_PATH,
        behaviour_riskset=INFO_RATE_INDUCED_LMEEEG_BEHAVIOUR_RISKSET_INPUT,
    output:
        INFO_RATE_INDUCED_LMEEEG_OUTPUTS,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main info-rate-induced-lmeeg \
          --config "{input.config}"
        """
