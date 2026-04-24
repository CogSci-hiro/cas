import glob

LM_FEATURE_DIR = PATHS_CONFIG.get("lm_feature_dir")
HAZARD_BEHAVIOR_OUTPUT_DIR = f"{OUT_DIR}/reports/hazard_behavior_fpp"
HAZARD_BEHAVIOR_RISKSET_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/riskset/hazard_behavior_riskset.tsv"
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


rule hazard_behavior_fpp:
    input:
        events_csv=EVENTS_CSV_OUTPUT,
        surprisal_tsvs=_hazard_behavior_surprisal_inputs,
    output:
        riskset=HAZARD_BEHAVIOR_RISKSET_OUTPUT,
        comparison=HAZARD_BEHAVIOR_MODEL_COMPARISON_OUTPUT,
        fit_metrics=HAZARD_BEHAVIOR_FIT_METRICS_OUTPUT,
        primary_model_summary=HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS[0],
        primary_model_comparison=HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS[1],
        primary_effects=HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS[2],
        primary_fit_metrics=HAZARD_BEHAVIOR_PRIMARY_MODEL_OUTPUTS[3],
        primary_stat_tests=HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS[0],
        primary_stat_tests_json=HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS[1],
        primary_publication_table=HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS[2],
        primary_interpretation=HAZARD_BEHAVIOR_PRIMARY_STATS_OUTPUTS[3],
        predicted_hazard_by_time=HAZARD_BEHAVIOR_MAIN_FIGURES[0],
        predicted_hazard_by_information_rate=HAZARD_BEHAVIOR_MAIN_FIGURES[1],
        observed_event_rate_by_time_bin=HAZARD_BEHAVIOR_MAIN_FIGURES[2],
        primary_coefficients_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[0],
        primary_model_comparison_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[1],
        primary_prop_expected_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[2],
        primary_information_rate_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[3],
        primary_observed_event_rate_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[4],
        primary_lag_sensitivity_figure=HAZARD_BEHAVIOR_PRIMARY_FIGURES[5],
    params:
        out_dir=HAZARD_BEHAVIOR_OUTPUT_DIR,
        surprisal_glob=lambda wildcards: os.path.join(
            LM_FEATURE_DIR,
            "**",
            "*desc-lmSurprisal_features.tsv",
        ),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main hazard-behavior-fpp \
          --events "{input.events_csv}" \
          --surprisal "{params.surprisal_glob}" \
          --out-dir "{params.out_dir}" \
          --overwrite
        """
