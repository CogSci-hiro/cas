from pathlib import Path

from cas.preprocessing.config import (
    build_preprocessing_run_paths,
    resolve_preprocessing_output_layout,
)
from cas.preprocessing.qc import write_preprocessing_aggregates

PREPROCESSING_LAYOUT = resolve_preprocessing_output_layout(
    PATHS_CONFIG,
    PREPROCESSING_CONFIG,
    base_dir=Path(PROJECT_ROOT),
)
EEG_RAW_EDF_PATTERN = (
    f"{BIDS_DIR}/sub-{{subject}}/eeg/sub-{{subject}}_task-{{task}}_run-{{run}}_eeg.edf"
)
EEG_RAW_FIF_PATTERN = (
    f"{BIDS_DIR}/sub-{{subject}}/eeg/sub-{{subject}}_task-{{task}}_run-{{run}}_eeg.fif"
)
PREPROCESSING_TABLES_DIR = str(PREPROCESSING_LAYOUT.tables_dir)
PREPROCESSING_QC_DIR = str(PREPROCESSING_LAYOUT.qc_dir)
PREPROCESSING_FINAL_DIR = str(PREPROCESSING_LAYOUT.final_dir)
PREPROCESSING_INTERMEDIATE_DIR = str(PREPROCESSING_LAYOUT.intermediate_dir)
PREPROCESSING_SUMMARY_TABLE = f"{PREPROCESSING_TABLES_DIR}/preprocessing_summary.tsv"
PREPROCESSING_BAD_CHANNELS_TABLE = f"{PREPROCESSING_TABLES_DIR}/bad_channels.tsv"
PREPROCESSING_REJECTED_SEGMENTS_TABLE = f"{PREPROCESSING_TABLES_DIR}/rejected_segments.tsv"
PREPROCESSING_QC_SUMMARY = f"{PREPROCESSING_QC_DIR}/preprocessing_qc_summary.json"

PREPROCESSING_SETTINGS = PREPROCESSING_CONFIG["preprocessing"]
APPLY_PRECOMPUTED_ICA = bool(PREPROCESSING_SETTINGS.get("apply_ica", False))


def _subject_dyad_id(subject: str) -> str:
    subject_id = f"sub-{subject}"
    dyad_id = SUBJECT_TO_DYAD_MAP.get(subject_id)
    if dyad_id:
        return dyad_id

    subject_number = int(subject)
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _expected_ica_path(subject: str, task: str, run: str) -> str:
    ica_filename_template = PREPROCESSING_SETTINGS["ica_filename_template"]
    return os.path.join(
        ICA_DIR,
        ica_filename_template.format(
            subject=subject,
            task=task,
            run=run,
        ),
    )


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
            if APPLY_PRECOMPUTED_ICA and not os.path.exists(_expected_ica_path(subject, task, run)):
                continue
            records.append({"subject": subject, "task": task, "run": run})
            seen_record_keys.add(record_key)

    return sorted(records, key=lambda record: (record["subject"], record["task"], record["run"]))


def _preprocessing_run_paths(subject: str, task: str, run: str):
    return build_preprocessing_run_paths(
        layout=PREPROCESSING_LAYOUT,
        subject=subject,
        task=task,
        run=run,
        dyad_id=_subject_dyad_id(subject),
    )


PREPROCESSED_EEG_OUTPUT_PATTERN = os.path.join(
    PREPROCESSING_FINAL_DIR,
    "sub-{subject}",
    "task-{task}",
    "run-{run}",
    "preprocessed_eeg.fif",
)
PREPROCESSED_EMG_OUTPUT_PATTERN = os.path.join(
    PREPROCESSING_FINAL_DIR,
    "sub-{subject}",
    "task-{task}",
    "run-{run}",
    "emg.npz",
)
PREPROCESSED_EVENTS_OUTPUT_PATTERN = os.path.join(
    PREPROCESSING_TABLES_DIR,
    "by_run",
    "sub-{subject}_task-{task}_run-{run}_events.tsv",
)
PREPROCESSED_SUMMARY_OUTPUT_PATTERN = os.path.join(
    PREPROCESSING_QC_DIR,
    "by_run",
    "sub-{subject}_task-{task}_run-{run}_summary.json",
)


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
    annotation_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{_subject_dyad_id(wildcards.subject)}_run-{wildcards.run}_combined.TextGrid",
    )
    if os.path.exists(annotation_path):
        return [annotation_path]
    return []


def preprocess_ica_input(wildcards):
    if not APPLY_PRECOMPUTED_ICA:
        return []

    ica_path = _expected_ica_path(wildcards.subject, wildcards.task, wildcards.run)
    if not os.path.exists(ica_path):
        raise FileNotFoundError(
            f"ICA file unexpectedly missing for included record "
            f"sub-{wildcards.subject} task={wildcards.task} run={wildcards.run}: {ica_path}"
        )
    return [ica_path]


PREPROCESSED_EEG_RECORDS = _discover_preprocessed_eeg_records()
PREPROCESSED_EEG_OUTPUTS = expand(
    PREPROCESSED_EEG_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in PREPROCESSED_EEG_RECORDS],
    task=[record["task"] for record in PREPROCESSED_EEG_RECORDS],
    run=[record["run"] for record in PREPROCESSED_EEG_RECORDS],
)
PREPROCESSED_SUMMARY_OUTPUTS = expand(
    PREPROCESSED_SUMMARY_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in PREPROCESSED_EEG_RECORDS],
    task=[record["task"] for record in PREPROCESSED_EEG_RECORDS],
    run=[record["run"] for record in PREPROCESSED_EEG_RECORDS],
)


rule preprocess_eeg_run:
    input:
        raw=preprocess_raw_input,
        channels_tsv=preprocess_channels_tsv_input,
        annotation=preprocess_annotation_input,
        ica=preprocess_ica_input,
        config=PREPROCESSING_CONFIG_PATH,
        paths=PATHS_CONFIG_PATH,
    output:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
        emg=PREPROCESSED_EMG_OUTPUT_PATTERN,
        events=PREPROCESSED_EVENTS_OUTPUT_PATTERN,
        summary=PREPROCESSED_SUMMARY_OUTPUT_PATTERN,
    params:
        dyad_id=lambda wildcards: _subject_dyad_id(wildcards.subject),
        intermediates_dir=lambda wildcards: (
            str(_preprocessing_run_paths(wildcards.subject, wildcards.task, wildcards.run).intermediates_dir)
            if PREPROCESSING_LAYOUT.save_intermediates
            else ""
        ),
    run:
        command_parts = [
            f'PYTHONPATH="{SRC_DIR}" python -m cas.cli.main preprocess-raw',
            f'--input "{input.raw}"',
            f'--output "{output.eeg}"',
            f'--channels-tsv "{input.channels_tsv}"',
            f'--emg-output "{output.emg}"',
            f'--events-output "{output.events}"',
            f'--summary-json "{output.summary}"',
            f'--subject-id "sub-{wildcards.subject}"',
            f'--task "{wildcards.task}"',
            f'--run-id "{wildcards.run}"',
            f'--dyad-id "{params.dyad_id}"',
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

        if params.intermediates_dir:
            command_parts.append(f'--intermediates-dir "{params.intermediates_dir}"')

        ica_inputs = list(input.ica)
        if ica_inputs:
            command_parts.append(f'--ica-path "{ica_inputs[0]}"')

        shell(" ".join(command_parts))


rule aggregate_preprocessing_qc:
    input:
        summaries=PREPROCESSED_SUMMARY_OUTPUTS,
        config=PREPROCESSING_CONFIG_PATH,
        paths=PATHS_CONFIG_PATH,
    output:
        summary_table=PREPROCESSING_SUMMARY_TABLE,
        bad_channels=PREPROCESSING_BAD_CHANNELS_TABLE,
        rejected_segments=PREPROCESSING_REJECTED_SEGMENTS_TABLE,
        qc_summary=PREPROCESSING_QC_SUMMARY,
    run:
        write_preprocessing_aggregates(
            summary_paths=list(input.summaries),
            preprocessing_summary_tsv=output.summary_table,
            bad_channels_tsv=output.bad_channels,
            rejected_segments_tsv=output.rejected_segments,
            qc_summary_json=output.qc_summary,
        )


rule preprocess_eeg:
    input:
        PREPROCESSED_EEG_OUTPUTS
