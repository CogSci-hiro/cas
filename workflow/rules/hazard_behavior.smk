import glob
import os

HAZARD_BEHAVIOR_WORKFLOW_CONFIG = HAZARD_BEHAVIOR_CONFIG if isinstance(HAZARD_BEHAVIOR_CONFIG, dict) else {}
HAZARD_BEHAVIOR_OUTPUT_CONFIG = dict(HAZARD_BEHAVIOR_WORKFLOW_CONFIG.get("output") or {})
HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG = dict(HAZARD_BEHAVIOR_WORKFLOW_CONFIG.get("timing_control") or {})


def _resolve_hazard_behavior_output_dir(config_value: str | None, default_subdir: str) -> str:
    if not isinstance(config_value, str) or not config_value.strip():
        return f"{OUT_DIR}/{default_subdir}"
    if os.path.isabs(config_value):
        return config_value
    return f"{OUT_DIR}/{config_value.lstrip('/')}"


LM_FEATURE_DIR = PATHS_CONFIG.get("lm_feature_dir")
NEURAL_FEATURE_DIR = PATHS_CONFIG.get("neural_feature_dir", f"{OUT_DIR}/features/neural_lowlevel")
HAZARD_BEHAVIOR_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    HAZARD_BEHAVIOR_OUTPUT_CONFIG.get("behaviour_output_dir"),
    "reports/hazard_behavior_fpp",
)
HAZARD_NEURAL_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    HAZARD_BEHAVIOR_OUTPUT_CONFIG.get("neural_output_dir"),
    "reports/hazard_neural_fpp",
)
HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS = bool(HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("fit_models", False))
LOWLEVEL_NEURAL_FEATURE_OUTPUT_PATTERN = (
    f"{NEURAL_FEATURE_DIR}/sub-{{subject}}/task-{{task}}/run-{{run}}/"
    f"sub-{{subject}}_task-{{task}}_run-{{run}}_desc-lowlevelNeural_features.tsv"
)
HAZARD_BEHAVIOR_RISKSET_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/riskset/hazard_behavior_riskset.tsv"
)
HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/riskset/hazard_behavior_riskset_with_timing_controls.tsv"
)
HAZARD_BEHAVIOR_MODEL_COMPARISON_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/model_comparison_behaviour.csv"
)
HAZARD_BEHAVIOR_FIT_METRICS_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/model_fit_metrics.json"
)
HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS = [
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_stat_tests.csv",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_stat_tests.json",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_publication_table.csv",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_interpretation.txt",
]
HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS = [
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_model_summary.csv",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_model_comparison.csv",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_effects.json",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_primary_fit_metrics.json",
]
HAZARD_BEHAVIOR_MAIN_FIGURES = [
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/predicted_hazard_by_time.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/predicted_hazard_by_information_rate.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/observed_event_rate_by_time_bin.png",
]
HAZARD_BEHAVIOR_PRIMARY_FIGURES = [
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_coefficients.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_model_comparison.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_predicted_hazard_prop_expected.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_predicted_hazard_information_rate.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_observed_event_rate.png",
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_primary_lag_sensitivity.png",
]
HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_model_summary.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_model_comparison.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_fit_metrics.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_selected_lags.json",
        HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT,
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_timing_control_model_comparison.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_timing_control_coefficients.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_exports/behaviour_glmm_data.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_exports/behaviour_glmm_export_qc.json",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_model_summary.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_model_comparison.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_fit_metrics.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_fixed_effects.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_random_effects.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_predictions_expected_info.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models/behaviour_glmm_predictions_information_rate.csv",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_FIGURES = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_glmm_coefficients.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_glmm_model_comparison.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_glmm_predicted_hazard_expected_info.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_glmm_predicted_hazard_information_rate.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_RULE_OUTPUTS = [
    HAZARD_BEHAVIOR_RISKSET_OUTPUT,
    HAZARD_BEHAVIOR_MODEL_COMPARISON_OUTPUT,
    HAZARD_BEHAVIOR_FIT_METRICS_OUTPUT,
    *HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS,
    *HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS,
    *HAZARD_BEHAVIOR_MAIN_FIGURES,
    *HAZARD_BEHAVIOR_PRIMARY_FIGURES,
    *HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS,
]
HAZARD_BEHAVIOR_NEURAL_OUTPUTS = [
    f"{HAZARD_NEURAL_OUTPUT_DIR}/riskset/neural_feature_qc.json",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_model_summary.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_model_comparison.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_family_model_comparison.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_fit_metrics.json",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_summary_amplitude.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_summary_alpha.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_summary_beta.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_loadings_amplitude.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_loadings_alpha.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_pca_loadings_beta.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_lowlevel_effects.json",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_lowlevel_pca_variance.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_lowlevel_model_comparison.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_lowlevel_coefficients.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_lowlevel_predicted_hazard_pc1.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_lowlevel_feature_missingness.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/logs/neural_lowlevel_warnings.txt",
]
LOWLEVEL_NEURAL_FEATURE_OUTPUTS = expand(
    LOWLEVEL_NEURAL_FEATURE_OUTPUT_PATTERN,
    zip,
    subject=[record["subject"] for record in PREPROCESSED_EEG_RECORDS],
    task=[record["task"] for record in PREPROCESSED_EEG_RECORDS],
    run=[record["run"] for record in PREPROCESSED_EEG_RECORDS],
)


def _hazard_behavior_surprisal_inputs(wildcards) -> list[str]:
    del wildcards
    if not isinstance(LM_FEATURE_DIR, str) or not LM_FEATURE_DIR:
        raise ValueError("`lm_feature_dir` is missing from config/paths.yaml.")
    pattern = os.path.join(LM_FEATURE_DIR, "**", "*desc-lmSurprisal_features.tsv")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise ValueError(
            "No LM surprisal TSV files matched the configured pattern under "
            f"{LM_FEATURE_DIR!r}."
        )
    return matches


def _subject_speaker_label(subject: str) -> str:
    return "A" if int(subject) % 2 == 1 else "B"


def _subject_dyad_id(subject: str) -> str:
    subject_id = f"sub-{subject}"
    dyad_id = SUBJECT_TO_DYAD_MAP.get(subject_id)
    if dyad_id:
        return dyad_id
    dyad_number = (int(subject) + 1) // 2
    return f"dyad-{dyad_number:03d}"


def _hazard_behavior_timing_control_flag() -> str:
    return "--fit-timing-control-models" if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS else ""


rule hazard_behavior_fpp:
    input:
        events_csv=EVENTS_CSV_OUTPUT,
        surprisal_tsvs=_hazard_behavior_surprisal_inputs,
    output:
        HAZARD_BEHAVIOR_RULE_OUTPUTS
    params:
        out_dir=HAZARD_BEHAVIOR_OUTPUT_DIR,
        surprisal_glob=lambda wildcards: os.path.join(
            LM_FEATURE_DIR,
            "**",
            "*desc-lmSurprisal_features.tsv",
        ),
        timing_control_flag=lambda wildcards: _hazard_behavior_timing_control_flag(),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main hazard-behavior-fpp \
          --events "{input.events_csv}" \
          --surprisal "{params.surprisal_glob}" \
          --out-dir "{params.out_dir}" \
          {params.timing_control_flag} \
          --overwrite
        """


if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS:
    rule export_behaviour_glmm_data:
        input:
            riskset_tsv=HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT,
            selected_lags_json=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_selected_lags.json",
        output:
            export_csv=HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS[0],
            export_qc_json=HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS[1],
        shell:
            r"""
            set -euo pipefail
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main export-behaviour-glmm-data \
              --input-riskset "{input.riskset_tsv}" \
              --selected-lags-json "{input.selected_lags_json}" \
              --output-csv "{output.export_csv}" \
              --output-qc-json "{output.export_qc_json}"
            """


    rule fit_behaviour_hazard_glmm:
        input:
            export_csv=HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS[0],
        output:
            model_summary=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[0],
            model_comparison=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[1],
            fit_metrics=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[2],
            fixed_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[3],
            random_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[4],
            predictions_expected=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[5],
            predictions_information=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[6],
        params:
            output_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models",
        shell:
            r"""
            set -euo pipefail
            Rscript "{PROJECT_ROOT}/scripts/r/fit_behaviour_hazard_glmm.R" \
              --input-csv "{input.export_csv}" \
              --output-dir "{params.output_dir}"
            """


    rule plot_behaviour_glmm_results:
        input:
            model_summary=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[0],
            model_comparison=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[1],
            fit_metrics=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[2],
            fixed_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[3],
            random_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[4],
            predictions_expected=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[5],
            predictions_information=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[6],
        output:
            coefficients_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[0],
            comparison_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[1],
            expected_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[2],
            information_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[3],
        params:
            r_results_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/r_models",
            output_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures",
        shell:
            r"""
            set -euo pipefail
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main plot-behaviour-glmm-results \
              --r-results-dir "{params.r_results_dir}" \
              --output-dir "{params.output_dir}"
            """


rule extract_lowlevel_neural_features:
    input:
        eeg=PREPROCESSED_EEG_OUTPUT_PATTERN,
    output:
        features_tsv=LOWLEVEL_NEURAL_FEATURE_OUTPUT_PATTERN,
    params:
        dyad_id=lambda wildcards: _subject_dyad_id(wildcards.subject),
        speaker=lambda wildcards: _subject_speaker_label(wildcards.subject),
    run:
        from cas.neural.lowlevel import export_lowlevel_neural_feature_table

        print(
            f"[snakemake] extract_lowlevel_neural_features subject={wildcards.subject} "
            f"task={wildcards.task} run={wildcards.run}",
            flush=True,
        )
        export_lowlevel_neural_feature_table(
            raw_path=input.eeg,
            output_path=output.features_tsv,
            dyad_id=params.dyad_id,
            run=wildcards.run,
            speaker=params.speaker,
        )


rule hazard_behavior_fpp_neural_lowlevel:
    input:
        events_csv=EVENTS_CSV_OUTPUT,
        surprisal_tsvs=_hazard_behavior_surprisal_inputs,
        neural_tsvs=LOWLEVEL_NEURAL_FEATURE_OUTPUTS,
    output:
        neural_feature_qc=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[0],
        neural_model_summary=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[1],
        neural_model_comparison=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[2],
        neural_family_model_comparison=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[3],
        neural_fit_metrics=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[4],
        neural_pca_summary_amplitude=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[5],
        neural_pca_summary_alpha=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[6],
        neural_pca_summary_beta=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[7],
        neural_pca_loadings_amplitude=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[8],
        neural_pca_loadings_alpha=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[9],
        neural_pca_loadings_beta=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[10],
        neural_effects=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[11],
        neural_pca_variance_figure=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[12],
        neural_model_comparison_figure=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[13],
        neural_coefficients_figure=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[14],
        neural_predicted_pc1_figure=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[15],
        neural_feature_missingness_figure=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[16],
        neural_warnings=HAZARD_BEHAVIOR_NEURAL_OUTPUTS[17],
    params:
        out_dir=HAZARD_NEURAL_OUTPUT_DIR,
        surprisal_glob=lambda wildcards: os.path.join(
            LM_FEATURE_DIR,
            "**",
            "*desc-lmSurprisal_features.tsv",
        ),
        neural_feature_dir=NEURAL_FEATURE_DIR,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main hazard-behavior-fpp \
          --events "{input.events_csv}" \
          --surprisal "{params.surprisal_glob}" \
          --neural-features "{params.neural_feature_dir}" \
          --fit-neural-lowlevel-models \
          --no-run-behaviour-model-suite \
          --no-fit-primary-behaviour-models \
          --no-fit-primary-stat-tests \
          --no-make-primary-publication-figures \
          --out-dir "{params.out_dir}" \
          --overwrite
        """
