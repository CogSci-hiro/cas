PARTNER_INFO_TRF_CONFIG_PATH = f"{CONFIG_DIR}/trf/partner_info.yaml"
with open(PARTNER_INFO_TRF_CONFIG_PATH, encoding="utf-8") as f:
    PARTNER_INFO_TRF_CONFIG = yaml.safe_load(f) or {}
PARTNER_INFO_TRF_DATASET_CONFIG_PATH = f"{CONFIG_DIR}/dataset.yaml"
with open(PARTNER_INFO_TRF_DATASET_CONFIG_PATH, encoding="utf-8") as f:
    PARTNER_INFO_TRF_DATASET_CONFIG = yaml.safe_load(f) or {}


def _partner_info_trf_dyad_id(subject: str) -> str:
    subject_number = int(str(subject))
    return f"dyad-{(subject_number + 1) // 2:03d}"


def _partner_info_trf_subjects():
    dataset_section = dict(PARTNER_INFO_TRF_DATASET_CONFIG.get("dataset") or {})
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
        dyad_id = _partner_info_trf_dyad_id(subject)
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


PARTNER_INFO_TRF_ANALYSIS_ID = PARTNER_INFO_TRF_CONFIG["trf"]["analysis_id"]
PARTNER_INFO_TRF_SUBJECTS = _partner_info_trf_subjects()
PARTNER_INFO_TRF_OUTPUT_ROOT = os.path.join(
    TRF_ROOT,
    PARTNER_INFO_TRF_CONFIG["trf"]["output"]["root"].format(
        analysis_id=PARTNER_INFO_TRF_ANALYSIS_ID
    ),
)
PARTNER_INFO_TRF_SUBJECT_SUMMARY_PATTERN = (
    f"{PARTNER_INFO_TRF_OUTPUT_ROOT}/sub-{{subject}}/partner_info.summary.json"
)
PARTNER_INFO_TRF_SUBJECT_COEF_PATTERN = (
    f"{PARTNER_INFO_TRF_OUTPUT_ROOT}/sub-{{subject}}/partner_info.coefs.npz"
)
PARTNER_INFO_TRF_SUBJECT_SUMMARIES = expand(
    PARTNER_INFO_TRF_SUBJECT_SUMMARY_PATTERN,
    subject=PARTNER_INFO_TRF_SUBJECTS,
)
PARTNER_INFO_TRF_SUBJECT_COEFS = expand(
    PARTNER_INFO_TRF_SUBJECT_COEF_PATTERN,
    subject=PARTNER_INFO_TRF_SUBJECTS,
)
PARTNER_INFO_TRF_GROUP_DIR = f"{PARTNER_INFO_TRF_OUTPUT_ROOT}/group"
PARTNER_INFO_TRF_GROUP_MAIN_DIR = f"{PARTNER_INFO_TRF_GROUP_DIR}/main"
PARTNER_INFO_TRF_GROUP_QC_DIR = f"{PARTNER_INFO_TRF_GROUP_DIR}/qc"
PARTNER_INFO_TRF_GROUP_KERNEL_DIR = f"{PARTNER_INFO_TRF_GROUP_MAIN_DIR}/kernels"
PARTNER_INFO_TRF_GROUP_SUMMARY_JSON = f"{PARTNER_INFO_TRF_GROUP_DIR}/partner_info_trf_summary.json"
PARTNER_INFO_TRF_GROUP_SUBJECT_CSV = f"{PARTNER_INFO_TRF_GROUP_DIR}/partner_info_trf_subject_scores.csv"
PARTNER_INFO_TRF_GROUP_FOLD_CSV = f"{PARTNER_INFO_TRF_GROUP_DIR}/partner_info_trf_fold_scores.csv"
PARTNER_INFO_TRF_GROUP_COMPARISON_CSV = f"{PARTNER_INFO_TRF_GROUP_DIR}/partner_info_trf_model_comparisons.csv"
PARTNER_INFO_TRF_GROUP_DIAGNOSTICS_CSV = f"{PARTNER_INFO_TRF_GROUP_DIR}/partner_info_trf_predictor_diagnostics.csv"
PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PNG = f"{PARTNER_INFO_TRF_GROUP_MAIN_DIR}/partner_info_trf_model_comparison.png"
PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PDF = f"{PARTNER_INFO_TRF_GROUP_MAIN_DIR}/partner_info_trf_model_comparison.pdf"
PARTNER_INFO_TRF_GROUP_SIGMA_PNG = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_selected_sigma.png"
PARTNER_INFO_TRF_GROUP_SIGMA_PDF = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_selected_sigma.pdf"
PARTNER_INFO_TRF_GROUP_ALPHA_PNG = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_selected_alpha.png"
PARTNER_INFO_TRF_GROUP_ALPHA_PDF = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_selected_alpha.pdf"
PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PNG = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_fold_scores.png"
PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PDF = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_fold_scores.pdf"
PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PNG = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_predictor_corr.png"
PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PDF = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_predictor_corr.pdf"
PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PNG = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_predictor_variance.png"
PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PDF = f"{PARTNER_INFO_TRF_GROUP_QC_DIR}/partner_info_trf_predictor_variance.pdf"
PARTNER_INFO_TRF_GROUP_KERNEL_OUTPUTS = expand(
    f"{PARTNER_INFO_TRF_GROUP_KERNEL_DIR}/{{target}}_{{predictor}}_kernel_joint.{{ext}}",
    target=["alpha", "beta", "raw"],
    predictor=["r", "p"],
    ext=["png", "pdf"],
)


def partner_info_trf_input_files(wildcards):
    subject_id = f"sub-{wildcards.subject}"
    subject_number = int(wildcards.subject)
    partner_number = subject_number + 1 if subject_number % 2 == 1 else subject_number - 1
    partner_subject = f"sub-{partner_number:03d}"
    partner_envelopes = expand(
        ENVELOPE_OUTPUT_PATTERN,
        zip,
        subject=[partner_subject.replace("sub-", "", 1)] * len(RUNS),
        task=["conversation"] * len(RUNS),
        run=RUNS,
    )
    preprocessed_eeg = expand(
        PREPROCESSED_EEG_OUTPUT_PATTERN,
        zip,
        subject=[wildcards.subject] * len(RUNS),
        task=["conversation"] * len(RUNS),
        run=RUNS,
    )
    return [PARTNER_INFO_TRF_CONFIG_PATH] + partner_envelopes + preprocessed_eeg


rule fit_trf_partner_info_subject:
    input:
        partner_info_trf_input_files,
    output:
        summary=PARTNER_INFO_TRF_SUBJECT_SUMMARY_PATTERN,
        coef=PARTNER_INFO_TRF_SUBJECT_COEF_PATTERN,
    params:
        src_dir=SRC_DIR,
        config_path=PARTNER_INFO_TRF_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    shell:
        """
        NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl \
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main trf-partner-info-fit \
            --config "{params.config_path}" \
            --subject "sub-{wildcards.subject}" \
            --project-root "{params.project_root}" \
            --output-json "{output.summary}" \
            --output-npz "{output.coef}"
        """


rule aggregate_trf_partner_info:
    input:
        summaries=PARTNER_INFO_TRF_SUBJECT_SUMMARIES,
        coefs=PARTNER_INFO_TRF_SUBJECT_COEFS,
    output:
        summary=PARTNER_INFO_TRF_GROUP_SUMMARY_JSON,
        subject_csv=PARTNER_INFO_TRF_GROUP_SUBJECT_CSV,
        fold_csv=PARTNER_INFO_TRF_GROUP_FOLD_CSV,
        comparison_csv=PARTNER_INFO_TRF_GROUP_COMPARISON_CSV,
        diagnostics_csv=PARTNER_INFO_TRF_GROUP_DIAGNOSTICS_CSV,
        model_comparison_png=PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PNG,
        model_comparison_pdf=PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PDF,
        sigma_png=PARTNER_INFO_TRF_GROUP_SIGMA_PNG,
        sigma_pdf=PARTNER_INFO_TRF_GROUP_SIGMA_PDF,
        alpha_png=PARTNER_INFO_TRF_GROUP_ALPHA_PNG,
        alpha_pdf=PARTNER_INFO_TRF_GROUP_ALPHA_PDF,
        fold_scores_png=PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PNG,
        fold_scores_pdf=PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PDF,
        predictor_corr_png=PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PNG,
        predictor_corr_pdf=PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PDF,
        predictor_variance_png=PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PNG,
        predictor_variance_pdf=PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PDF,
        kernels=PARTNER_INFO_TRF_GROUP_KERNEL_OUTPUTS,
    params:
        src_dir=SRC_DIR,
        kernel_dir=PARTNER_INFO_TRF_GROUP_KERNEL_DIR,
    shell:
        """
        NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl \
        PYTHONPATH="{params.src_dir}" python -m cas.cli.main trf-partner-info-group \
            --subject-jsons {input.summaries} \
            --subject-npzs {input.coefs} \
            --summary-json "{output.summary}" \
            --subject-csv "{output.subject_csv}" \
            --fold-csv "{output.fold_csv}" \
            --comparison-csv "{output.comparison_csv}" \
            --diagnostics-csv "{output.diagnostics_csv}" \
            --model-comparison-png "{output.model_comparison_png}" \
            --model-comparison-pdf "{output.model_comparison_pdf}" \
            --kernel-dir "{params.kernel_dir}" \
            --sigma-png "{output.sigma_png}" \
            --sigma-pdf "{output.sigma_pdf}" \
            --alpha-png "{output.alpha_png}" \
            --alpha-pdf "{output.alpha_pdf}" \
            --fold-scores-png "{output.fold_scores_png}" \
            --fold-scores-pdf "{output.fold_scores_pdf}" \
            --predictor-corr-png "{output.predictor_corr_png}" \
            --predictor-corr-pdf "{output.predictor_corr_pdf}" \
            --predictor-variance-png "{output.predictor_variance_png}" \
            --predictor-variance-pdf "{output.predictor_variance_pdf}"
        """


rule trf_partner_info_all:
    input:
        PARTNER_INFO_TRF_GROUP_SUMMARY_JSON,
        PARTNER_INFO_TRF_GROUP_SUBJECT_CSV,
        PARTNER_INFO_TRF_GROUP_FOLD_CSV,
        PARTNER_INFO_TRF_GROUP_COMPARISON_CSV,
        PARTNER_INFO_TRF_GROUP_DIAGNOSTICS_CSV,
        PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PNG,
        PARTNER_INFO_TRF_GROUP_MODEL_COMPARISON_PDF,
        PARTNER_INFO_TRF_GROUP_SIGMA_PNG,
        PARTNER_INFO_TRF_GROUP_SIGMA_PDF,
        PARTNER_INFO_TRF_GROUP_ALPHA_PNG,
        PARTNER_INFO_TRF_GROUP_ALPHA_PDF,
        PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PNG,
        PARTNER_INFO_TRF_GROUP_FOLD_SCORES_PDF,
        PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PNG,
        PARTNER_INFO_TRF_GROUP_PREDICTOR_CORR_PDF,
        PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PNG,
        PARTNER_INFO_TRF_GROUP_PREDICTOR_VARIANCE_PDF,
        PARTNER_INFO_TRF_GROUP_KERNEL_OUTPUTS
