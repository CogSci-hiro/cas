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
    shell:
        """
        PYTHONPATH="{SRC_DIR}" python -m cas.cli.main eeg-array \
            --input "{input}" \
            --output "{output}" \
            --target-sfreq-hz {TRF_CONFIG["trf"]["timing"]["target_sfreq_hz"]}
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

    return [TRF_CONFIG_PATH, DYADS_CSV_PATH] + envelope_inputs + partner_envelope_inputs + eeg_inputs


rule fit_trf:
    input:
        trf_input_files,
    output:
        score="/Users/hiro/Datasets/working/cas/derivatives/trf/env_self_other/sub-{subject}/trf.scores.json",
        coef="/Users/hiro/Datasets/working/cas/derivatives/trf/env_self_other/sub-{subject}/trf.coefs.npz"
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
