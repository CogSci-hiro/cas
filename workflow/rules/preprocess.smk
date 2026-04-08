EEG_RAW_EDF_PATTERN = (
    f"{BIDS_DIR}/sub-{{subject}}/eeg/sub-{{subject}}_task-{{task}}_run-{{run}}_eeg.edf"
)
EEG_RAW_FIF_PATTERN = (
    f"{BIDS_DIR}/sub-{{subject}}/eeg/sub-{{subject}}_task-{{task}}_run-{{run}}_eeg.fif"
)
EEG_CHANNELS_TSV_PATTERN = (
    f"{BIDS_DIR}/sub-{{subject}}/eeg/sub-{{subject}}_task-{{task}}_run-{{run}}_channels.tsv"
)
PREPROCESSING_RUN_DIR_PATTERN = (
    f"{PREPROCESSED_EEG_ROOT}/sub-{{subject}}/task-{{task}}/run-{{run}}"
)
PREPROCESSED_EEG_OUTPUT_PATTERN = f"{PREPROCESSING_RUN_DIR_PATTERN}/preprocessed_eeg.fif"
PREPROCESSED_EMG_OUTPUT_PATTERN = f"{PREPROCESSING_RUN_DIR_PATTERN}/emg.npz"
PREPROCESSED_EVENTS_OUTPUT_PATTERN = f"{PREPROCESSING_RUN_DIR_PATTERN}/events.tsv"
PREPROCESSED_SUMMARY_OUTPUT_PATTERN = f"{PREPROCESSING_RUN_DIR_PATTERN}/summary.json"

PREPROCESSING_SETTINGS = PREPROCESSING_CONFIG["preprocessing"]
APPLY_PRECOMPUTED_ICA = bool(PREPROCESSING_SETTINGS.get("apply_ica", False))


def _is_included_eeg_record(subject: str, task: str, run: str) -> bool:
    subject_id = f"sub-{subject}"
    if TASKS and task not in TASKS:
        return False
    if RUNS and run not in RUNS:
        return False
    if subject_id in EXCLUDED_SUBJECTS:
        return False
    if (subject_id, run) in EXCLUDED_SUBJECT_RUNS:
        return False
    return True


def _discover_preprocessed_eeg_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_record_keys: set[tuple[str, str, str]] = set()

    for eeg_pattern in (EEG_RAW_EDF_PATTERN, EEG_RAW_FIF_PATTERN):
        wildcard_values = glob_wildcards(eeg_pattern)
        for subject, task, run in zip(
            wildcard_values.subject,
            wildcard_values.task,
            wildcard_values.run,
        ):
            record_key = (subject, task, run)
            if record_key in seen_record_keys:
                continue
            if not _is_included_eeg_record(subject, task, run):
                continue
            records.append({"subject": subject, "task": task, "run": run})
            seen_record_keys.add(record_key)

    return sorted(records, key=lambda record: (record["subject"], record["task"], record["run"]))


def preprocess_raw_input(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    candidate_paths = [
        f"{BIDS_DIR}/{subject_id}/eeg/{subject_id}_task-{wildcards.task}_run-{wildcards.run}_eeg.edf",
        f"{BIDS_DIR}/{subject_id}/eeg/{subject_id}_task-{wildcards.task}_run-{wildcards.run}_eeg.fif",
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(
        f"No raw EEG file found for {subject_id}, task {wildcards.task}, run {wildcards.run}."
    )


def preprocess_channels_tsv_input(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    channels_tsv_path = (
        f"{BIDS_DIR}/{subject_id}/eeg/"
        f"{subject_id}_task-{wildcards.task}_run-{wildcards.run}_channels.tsv"
    )
    if not os.path.exists(channels_tsv_path):
        raise FileNotFoundError(
            f"No channels.tsv file found for {subject_id}, task {wildcards.task}, run {wildcards.run}."
        )
    return channels_tsv_path


def preprocess_annotation_input(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    dyad_id = SUBJECT_TO_DYAD_MAP.get(subject_id)
    if not dyad_id:
        subject_number = int(wildcards.subject)
        dyad_number = (subject_number + 1) // 2
        dyad_id = f"dyad-{dyad_number:03d}"

    annotation_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{dyad_id}_run-{wildcards.run}_combined.TextGrid",
    )
    if os.path.exists(annotation_path):
        return [annotation_path]
    return []


def preprocess_ica_input(wildcards):
    if not APPLY_PRECOMPUTED_ICA:
        return []

    ica_filename_template = PREPROCESSING_SETTINGS["ica_filename_template"]
    ica_path = os.path.join(
        ICA_DIR,
        ica_filename_template.format(
            subject=wildcards.subject,
            task=wildcards.task,
            run=wildcards.run,
        ),
    )
    # Keep ICA optional at the workflow layer to match CAS/rate preprocessing,
    # which applies a precomputed ICA only when the file is present.
    if os.path.exists(ica_path):
        return [ica_path]
    return []


PREPROCESSED_EEG_RECORDS = _discover_preprocessed_eeg_records()
PREPROCESSED_EEG_OUTPUTS = expand(
    PREPROCESSED_EEG_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in PREPROCESSED_EEG_RECORDS],
    task=[record["task"] for record in PREPROCESSED_EEG_RECORDS],
    run=[record["run"] for record in PREPROCESSED_EEG_RECORDS],
)


rule preprocess_eeg:
    input:
        raw=preprocess_raw_input,
        channels_tsv=preprocess_channels_tsv_input,
        annotation=preprocess_annotation_input,
        ica=preprocess_ica_input,
    output:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        emg=PREPROCESSED_EMG_OUTPUT_PATTERN,
        events=PREPROCESSED_EVENTS_OUTPUT_PATTERN,
        summary=PREPROCESSED_SUMMARY_OUTPUT_PATTERN,
    run:
        command_parts = [
            f'PYTHONPATH="{SRC_DIR}" python -m cas.cli.main preprocess-raw',
            f'--input "{input.raw}"',
            f'--output "{output.eeg}"',
            f'--channels-tsv "{input.channels_tsv}"',
            f'--emg-output "{output.emg}"',
            f'--events-output "{output.events}"',
            f'--summary-json "{output.summary}"',
        ]

        annotation_inputs = list(input.annotation)
        if annotation_inputs:
            command_parts.append(f'--annotations-path "{annotation_inputs[0]}"')

        target_sampling_rate_hz = PREPROCESSING_SETTINGS.get("target_sampling_rate_hz")
        if target_sampling_rate_hz is not None:
            command_parts.append(f"--target-sfreq-hz {float(target_sampling_rate_hz)}")

        low_cut_hz = PREPROCESSING_SETTINGS.get("low_cut_hz")
        if low_cut_hz is not None:
            command_parts.append(f"--low-cut-hz {float(low_cut_hz)}")

        high_cut_hz = PREPROCESSING_SETTINGS.get("high_cut_hz")
        if high_cut_hz is not None:
            command_parts.append(f"--high-cut-hz {float(high_cut_hz)}")

        montage_name = PREPROCESSING_SETTINGS.get("montage")
        if montage_name is not None:
            command_parts.append(f'--montage "{montage_name}"')

        annotation_pairing_margin_s = PREPROCESSING_SETTINGS.get("annotation_pairing_margin_s")
        if annotation_pairing_margin_s is not None:
            command_parts.append(f"--annotation-pairing-margin-s {float(annotation_pairing_margin_s)}")

        for channel_name in PREPROCESSING_SETTINGS.get("eeg_channel_names", []):
            command_parts.append(f'--eeg-channel-name "{channel_name}"')
        for channel_pattern in PREPROCESSING_SETTINGS.get("eeg_channel_patterns", []):
            command_parts.append(f'--eeg-channel-pattern "{channel_pattern}"')
        for channel_name in PREPROCESSING_SETTINGS.get("emg_channel_names", []):
            command_parts.append(f'--emg-channel-name "{channel_name}"')
        for channel_pattern in PREPROCESSING_SETTINGS.get("emg_channel_patterns", []):
            command_parts.append(f'--emg-channel-pattern "{channel_pattern}"')

        if not bool(PREPROCESSING_SETTINGS.get("interpolate_bad_channels", True)):
            command_parts.append("--skip-interpolate-bads")

        if not bool(PREPROCESSING_SETTINGS.get("apply_rereference", True)):
            command_parts.append("--skip-rereference")

        if not bool(PREPROCESSING_SETTINGS.get("keep_emg", True)):
            command_parts.append("--skip-emg")

        ica_inputs = list(input.ica)
        if ica_inputs:
            command_parts.append(f'--ica-path "{ica_inputs[0]}"')

        shell(" ".join(command_parts))
