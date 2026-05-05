def eeg_raw_input(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    candidates = [
        f"{BIDS_DIR}/{subject_id}/eeg/{subject_id}_task-conversation_run-{wildcards.run}_eeg.edf",
        f"{BIDS_DIR}/{subject_id}/eeg/{subject_id}_task-conversation_run-{wildcards.run}_eeg.fif",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No raw EEG file found for {subject_id} run {wildcards.run}.")


rule eeg_array:
    input:
        eeg_raw_input
    output:
        f"{EEG_ARRAY_ROOT}/evoked/sub-{{subject}}/run-{{run}}.npy"
    params:
        target_sfreq_hz=TRF_CONFIG["trf"]["timing"]["target_sfreq_hz"]
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main eeg-array \
            --input "{input}" \
            --output "{output}" \
            --target-sfreq-hz {params.target_sfreq_hz}
        """


def _trf_target_run_path(subject_id, run):
    return os.path.join(EEG_ARRAY_ROOT, TRF_TARGET_PATH_TEMPLATE.format(subject=subject_id, run=run))


def trf_input_files(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    partner_id = PARTNER_MAP[subject_id]
    partner_subject = partner_id.replace("sub-", "", 1)

    envelope_inputs = expand(
        ENVELOPE_OUTPUT_PATTERN,
        zip,
        subject=[wildcards.subject] * len(RUNS),
        task=["conversation"] * len(RUNS),
        run=RUNS,
    )
    partner_envelope_inputs = expand(
        ENVELOPE_OUTPUT_PATTERN,
        zip,
        subject=[partner_subject] * len(RUNS),
        task=["conversation"] * len(RUNS),
        run=RUNS,
    )
    eeg_inputs = [_trf_target_run_path(subject_id, run) for run in RUNS]

    return [TRF_CONFIG_PATH] + envelope_inputs + partner_envelope_inputs + eeg_inputs


rule fit_trf_subject:
    input:
        trf_input_files,
    output:
        score=TRF_SCORE_PATTERN,
        coef=TRF_COEF_PATTERN
    params:
        src_dir=SRC_DIR,
        config_path=TRF_CONFIG_PATH,
        data_root=DATA_ROOT
    shell:
        """
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main trf-config \
            --config "{params.config_path}" \
            --subject "sub-{wildcards.subject}" \
            --project-root "{params.data_root}"
        """


rule fit_trf:
    input:
        TRF_SCORE_OUTPUTS,
        TRF_COEF_OUTPUTS


TRF_SPP_ONSET_CONTROL_CONFIG_PATH = f"{CONFIG_DIR}/trf/spp_onset_control.yaml"
with open(TRF_SPP_ONSET_CONTROL_CONFIG_PATH, encoding="utf-8") as f:
    TRF_SPP_ONSET_CONTROL_CONFIG = yaml.safe_load(f) or {}
TRF_SPP_ONSET_CONTROL_DATASET_CONFIG_PATH = f"{CONFIG_DIR}/dataset.yaml"
with open(TRF_SPP_ONSET_CONTROL_DATASET_CONFIG_PATH, encoding="utf-8") as f:
    TRF_SPP_ONSET_CONTROL_DATASET_CONFIG = yaml.safe_load(f) or {}


def _trf_spp_onset_control_dyad_id(subject: str) -> str:
    subject_number = int(str(subject))
    dyad_number = (subject_number + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _trf_spp_onset_control_subjects():
    dataset_section = dict(TRF_SPP_ONSET_CONTROL_DATASET_CONFIG.get("dataset") or {})
    exclude_section = dict(dataset_section.get("exclude") or {})

    excluded_subjects = {
        str(item.get("subject_id"))
        for item in list(exclude_section.get("subjects") or [])
        if item.get("subject_id")
    }
    excluded_subject_runs = {
        (str(item.get("subject_id")), str(int(item.get("run"))))
        for item in list(exclude_section.get("subject_runs") or [])
        if item.get("subject_id") and item.get("run") is not None
    }
    excluded_dyads = {
        str(item.get("dyad_id"))
        for item in list(exclude_section.get("dyads") or [])
        if item.get("dyad_id")
    }
    excluded_dyad_runs = {
        (str(item.get("dyad_id")), str(int(item.get("run"))))
        for item in list(exclude_section.get("dyad_runs") or [])
        if item.get("dyad_id") and item.get("run") is not None
    }

    subject_to_runs = {}
    for record in AUDIO_RECORDS:
        subject = str(record["subject"])
        subject_id = f"sub-{subject}"
        run = str(record["run"])
        dyad_id = _trf_spp_onset_control_dyad_id(subject)
        if subject_id in excluded_subjects:
            continue
        if (subject_id, run) in excluded_subject_runs:
            continue
        if dyad_id in excluded_dyads:
            continue
        if (dyad_id, run) in excluded_dyad_runs:
            continue
        subject_to_runs.setdefault(subject, set()).add(run)

    eligible_subjects = []
    for subject, subject_runs in subject_to_runs.items():
        if subject_runs != set(RUNS):
            continue
        subject_number = int(subject)
        partner_number = subject_number + 1 if subject_number % 2 == 1 else subject_number - 1
        partner_subject = f"{partner_number:03d}"
        if subject_to_runs.get(partner_subject) != set(RUNS):
            continue
        eligible_subjects.append(subject)

    return sorted(eligible_subjects)

TRF_SPP_ONSET_CONTROL_ANALYSIS_ID = TRF_SPP_ONSET_CONTROL_CONFIG["trf"]["analysis_id"]
TRF_SPP_ONSET_CONTROL_SUBJECTS = _trf_spp_onset_control_subjects()
TRF_SPP_ONSET_CONTROL_OUTPUT_ROOT = os.path.join(
    TRF_ROOT,
    TRF_SPP_ONSET_CONTROL_CONFIG["trf"]["output"]["root"].format(
        analysis_id=TRF_SPP_ONSET_CONTROL_ANALYSIS_ID
    ),
)
TRF_SPP_ONSET_CONTROL_SUBJECT_SUMMARY_PATTERN = (
    f"{TRF_SPP_ONSET_CONTROL_OUTPUT_ROOT}/sub-{{subject}}/control.summary.json"
)
TRF_SPP_ONSET_CONTROL_SUBJECT_COEF_PATTERN = (
    f"{TRF_SPP_ONSET_CONTROL_OUTPUT_ROOT}/sub-{{subject}}/control.coefs.npz"
)
TRF_SPP_ONSET_CONTROL_SUBJECT_SUMMARIES = expand(
    TRF_SPP_ONSET_CONTROL_SUBJECT_SUMMARY_PATTERN,
    subject=TRF_SPP_ONSET_CONTROL_SUBJECTS,
)
TRF_SPP_ONSET_CONTROL_SUBJECT_COEFS = expand(
    TRF_SPP_ONSET_CONTROL_SUBJECT_COEF_PATTERN,
    subject=TRF_SPP_ONSET_CONTROL_SUBJECTS,
)
TRF_SPP_ONSET_CONTROL_GROUP_DIR = f"{TRF_SPP_ONSET_CONTROL_OUTPUT_ROOT}/group"
TRF_SPP_ONSET_CONTROL_GROUP_SUMMARY_JSON = (
    f"{TRF_SPP_ONSET_CONTROL_GROUP_DIR}/trf_spp_onset_control_summary.json"
)
TRF_SPP_ONSET_CONTROL_GROUP_SUBJECT_CSV = (
    f"{TRF_SPP_ONSET_CONTROL_GROUP_DIR}/trf_spp_onset_control_subject_scores.csv"
)
TRF_SPP_ONSET_CONTROL_GROUP_FOLD_CSV = (
    f"{TRF_SPP_ONSET_CONTROL_GROUP_DIR}/trf_spp_onset_control_fold_scores.csv"
)
TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PNG = (
    f"{TRF_SPP_ONSET_CONTROL_GROUP_DIR}/trf_spp_onset_kernel_joint.png"
)
TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PDF = (
    f"{TRF_SPP_ONSET_CONTROL_GROUP_DIR}/trf_spp_onset_kernel_joint.pdf"
)


def trf_spp_onset_control_input_files(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    subject_number = int(wildcards.subject)
    partner_number = subject_number + 1 if subject_number % 2 == 1 else subject_number - 1
    partner_subject = f"{partner_number:03d}"
    partner_envelopes = expand(
        ENVELOPE_OUTPUT_PATTERN,
        zip,
        subject=[partner_subject] * len(RUNS),
        task=["conversation"] * len(RUNS),
        run=RUNS,
    )
    eeg_inputs = [_trf_target_run_path(subject_id, run) for run in RUNS]
    return [
        TRF_SPP_ONSET_CONTROL_CONFIG_PATH,
        EVENTS_CSV_OUTPUT,
    ] + partner_envelopes + eeg_inputs


rule fit_trf_spp_onset_control_subject:
    input:
        trf_spp_onset_control_input_files,
    output:
        summary=TRF_SPP_ONSET_CONTROL_SUBJECT_SUMMARY_PATTERN,
        coef=TRF_SPP_ONSET_CONTROL_SUBJECT_COEF_PATTERN,
    params:
        src_dir=SRC_DIR,
        config_path=TRF_SPP_ONSET_CONTROL_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    shell:
        """
        NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl \
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main trf-spp-onset-control-fit \
            --config "{params.config_path}" \
            --subject "sub-{wildcards.subject}" \
            --project-root "{params.project_root}" \
            --output-json "{output.summary}" \
            --output-npz "{output.coef}"
        """


rule aggregate_trf_spp_onset_control:
    input:
        summaries=TRF_SPP_ONSET_CONTROL_SUBJECT_SUMMARIES,
        coefs=TRF_SPP_ONSET_CONTROL_SUBJECT_COEFS,
    output:
        summary=TRF_SPP_ONSET_CONTROL_GROUP_SUMMARY_JSON,
        subject_csv=TRF_SPP_ONSET_CONTROL_GROUP_SUBJECT_CSV,
        fold_csv=TRF_SPP_ONSET_CONTROL_GROUP_FOLD_CSV,
        kernel_png=TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PNG,
        kernel_pdf=TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PDF,
    params:
        src_dir=SRC_DIR,
    shell:
        """
        NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl \
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main trf-spp-onset-control-group \
            --subject-jsons {input.summaries} \
            --subject-npzs {input.coefs} \
            --summary-json "{output.summary}" \
            --subject-csv "{output.subject_csv}" \
            --fold-csv "{output.fold_csv}" \
            --kernel-png "{output.kernel_png}" \
            --kernel-pdf "{output.kernel_pdf}"
        """


rule trf_all:
    input:
        TRF_SCORE_OUTPUTS,
        TRF_COEF_OUTPUTS


rule trf_spp_onset_control_all:
    input:
        TRF_SPP_ONSET_CONTROL_GROUP_SUMMARY_JSON,
        TRF_SPP_ONSET_CONTROL_GROUP_SUBJECT_CSV,
        TRF_SPP_ONSET_CONTROL_GROUP_FOLD_CSV,
        TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PNG,
        TRF_SPP_ONSET_CONTROL_GROUP_KERNEL_PDF
