import glob
import os

HAZARD_BEHAVIOR_WORKFLOW_CONFIG = HAZARD_BEHAVIOR_CONFIG if isinstance(HAZARD_BEHAVIOR_CONFIG, dict) else {}
HAZARD_BEHAVIOR_OUTPUT_CONFIG = dict(HAZARD_BEHAVIOR_WORKFLOW_CONFIG.get("output") or {})
HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG = dict(HAZARD_BEHAVIOR_WORKFLOW_CONFIG.get("timing_control") or {})
HAZARD_NEURAL_PERMUTATION_WORKFLOW_CONFIG = (
    HAZARD_FPP_NEURAL_PERMUTATION_NULL_CONFIG if isinstance(HAZARD_FPP_NEURAL_PERMUTATION_NULL_CONFIG, dict) else {}
)
HAZARD_NEURAL_PERMUTATION_INPUT_CONFIG = dict(HAZARD_NEURAL_PERMUTATION_WORKFLOW_CONFIG.get("input") or {})
HAZARD_NEURAL_PERMUTATION_OUTPUT_CONFIG = dict(HAZARD_NEURAL_PERMUTATION_WORKFLOW_CONFIG.get("output") or {})
HAZARD_NEURAL_PERMUTATION_RUN_CONFIG = dict(HAZARD_NEURAL_PERMUTATION_WORKFLOW_CONFIG.get("permutation") or {})
NEURAL_HAZARD_FPP_SPP_WORKFLOW_CONFIG = (
    NEURAL_HAZARD_FPP_SPP_CONFIG if isinstance(NEURAL_HAZARD_FPP_SPP_CONFIG, dict) else {}
)
NEURAL_HAZARD_FPP_SPP_INPUT_CONFIG = dict(NEURAL_HAZARD_FPP_SPP_WORKFLOW_CONFIG.get("input") or {})
NEURAL_HAZARD_FPP_SPP_OUTPUT_CONFIG = dict(NEURAL_HAZARD_FPP_SPP_WORKFLOW_CONFIG.get("output") or {})


def _resolve_hazard_behavior_output_dir(config_value: str | None, default_subdir: str) -> str:
    if not isinstance(config_value, str) or not config_value.strip():
        return f"{OUT_DIR}/{default_subdir}"
    if os.path.isabs(config_value):
        return config_value
    return f"{OUT_DIR}/{config_value.lstrip('/')}"


def _resolve_project_input_path(path_value: str | None) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value.lstrip("/"))


def _resolve_project_output_dir(path_value: str | None, default_subdir: str) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return os.path.join(PROJECT_ROOT, default_subdir)
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value.lstrip("/"))


LM_FEATURE_DIR = PATHS_CONFIG.get("lm_feature_dir")
HAZARD_BEHAVIOR_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    HAZARD_BEHAVIOR_OUTPUT_CONFIG.get("behaviour_output_dir"),
    "reports/hazard_behavior_fpp",
)
HAZARD_NEURAL_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    HAZARD_BEHAVIOR_OUTPUT_CONFIG.get("neural_output_dir"),
    "reports/hazard_neural_fpp",
)
HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR = os.path.join(HAZARD_NEURAL_OUTPUT_DIR, "lag_selection")
HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS = bool(HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("fit_models", False))
HAZARD_BEHAVIOR_RUN_R_GLMM_LAG_SWEEP = bool(HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("run_r_glmm_lag_sweep", True))
HAZARD_BEHAVIOR_R_GLMM_LAG_GRID = ",".join(
    str(int(lag_ms))
    for lag_ms in (HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_lag_grid_ms") or [0, 50, 100, 150, 200, 300, 500, 700, 1000])
)
HAZARD_BEHAVIOR_R_GLMM_INCLUDE_QUADRATIC_OFFSET_TIMING = bool(
    HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_include_quadratic_offset_timing", True)
)
HAZARD_BEHAVIOR_R_GLMM_BACKEND = str(HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_backend", "glmmTMB"))
HAZARD_BEHAVIOR_R_GLMM_INCLUDE_RUN_RANDOM_EFFECT = bool(
    HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_include_run_random_effect", False)
)
HAZARD_BEHAVIOR_R_GLMM_INCLUDE_RUN_RANDOM_EFFECT_TEXT = str(
    HAZARD_BEHAVIOR_R_GLMM_INCLUDE_RUN_RANDOM_EFFECT
).lower()
HAZARD_BEHAVIOR_R_GLMM_PROP_EXPECTED_MODE = str(
    HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_prop_expected_mode", "after_best_rate")
)
HAZARD_BEHAVIOR_R_GLMM_INCLUDE_PROP_EXPECTED_IN_FINAL = bool(
    HAZARD_BEHAVIOR_TIMING_CONTROL_CONFIG.get("r_glmm_include_prop_expected_in_final", False)
)
HAZARD_BEHAVIOR_R_GLMM_INCLUDE_PROP_EXPECTED_IN_FINAL_TEXT = str(
    HAZARD_BEHAVIOR_R_GLMM_INCLUDE_PROP_EXPECTED_IN_FINAL
).lower()
HAZARD_BEHAVIOR_RISKSET_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/riskset/hazard_behavior_riskset.tsv"
)
HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/riskset/hazard_behavior_riskset_with_timing_controls.tsv"
)
HAZARD_BEHAVIOR_MODEL_COMPARISON_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_model_comparison.csv"
)
HAZARD_BEHAVIOR_FIT_METRICS_OUTPUT = (
    f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_fit_metrics.json"
)
HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/lag_selection/behaviour_timing_control_lag_selection.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_model_summary.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_model_comparison.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_fit_metrics.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_selected_lags.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_lag_screening_note.txt",
        HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT,
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/qc_plots/lag_selection/behaviour_pooled_delta_bic_by_lag.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/exports/r_behaviour_glmm_lag_sweep_input.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/exports/r_behaviour_glmm_lag_sweep_export_qc.json",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_information_rate_lag_sweep.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_prop_expected_lag_sweep.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_selected_behaviour_lags.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_final_behaviour_model_summary.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_final_behaviour_model_comparison.csv",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/r_glmm_final_behaviour_effects.json",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/predictions/behaviour_r_glmm_final_predicted_hazard_information_rate.csv",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_GLMM_FIGURES = (
    [
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_r_glmm_delta_bic_by_lag.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_r_glmm_coefficient_by_lag.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_r_glmm_odds_ratio_by_lag.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_r_glmm_final_model_comparison.png",
        f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures/behaviour_r_glmm_final_predicted_hazard_information_rate.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_RULE_OUTPUTS = [
    HAZARD_BEHAVIOR_RISKSET_OUTPUT,
    *HAZARD_BEHAVIOR_TIMING_CONTROL_OUTPUTS,
]
HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR = f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/latency_regime"
HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/behaviour_latency_regime_data.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/behaviour_latency_regime_export_qc.json",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS = (
    [
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_a_one_student_t_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_s_skew_unimodal_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_b_two_student_t_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_c_mixture_of_experts_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_r1_student_t_location_regression_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_r2_student_t_location_scale_regression_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_r3_shifted_lognormal_location_regression_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_model_r4_shifted_lognormal_location_scale_regression_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_loo_comparison.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_fit_metrics.json",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_component_parameters.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_gating_coefficients.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_event_probabilities.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_posterior_predictive.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_regression_coefficients.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_regression_predictions.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models/behaviour_latency_regime_shifted_lognormal_diagnostics.csv",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES = (
    [
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_components.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_probability_by_expected_info.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_probability_by_information_rate.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_gating_coefficients.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_skew_vs_mixture.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_regression_vs_mixture.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_ppc.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_loo_comparison.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_latency_by_expected_info_regression.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_latency_by_information_rate_regression.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES = (
    [
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_conditional_density_fixed_predictors.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_ppc_by_expected_info_quantile.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_ppc_by_information_rate_quantile.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_r_model_residual_density.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_pointwise_elpd_c_minus_r_by_latency.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_pointwise_elpd_c_minus_r_by_expected_info.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_pointwise_elpd_c_minus_r_by_information_rate.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_pointwise_elpd_c_minus_r_by_p_late.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_p_late_vs_r_mu.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_latency_vs_r_mu_coloured_by_p_late.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_latency_vs_information_rate_coloured_by_p_late.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_latency_vs_expected_cum_info_coloured_by_p_late.png",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures/behaviour_latency_regime_counterfactual_predictor_distribution.png",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)
HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS = (
    [
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_conditional_density_fixed_predictors.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_ppc_by_predictor_quantile.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_r_model_residuals.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_r_model_residual_bimodality_summary.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_pointwise_elpd_differences.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_p_late_vs_r_predictions.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_p_late_r_prediction_correlations.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_counterfactual_predictor_distribution.csv",
        f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics/latency_regime_conditional_vs_marginal_bimodality_report.md",
    ]
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS
    else []
)

HAZARD_NEURAL_RISKSET_OUTPUTS = [
    f"{HAZARD_NEURAL_OUTPUT_DIR}/riskset/neural_fpp_hazard_table.parquet",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/riskset/neural_spp_hazard_table.parquet",
]
HAZARD_NEURAL_MODEL_OUTPUTS = [
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_model_comparison.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_coefficients.csv",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/models/neural_fit_metrics.json",
]
HAZARD_NEURAL_FIGURES = [
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_delta_bic_fpp_vs_spp.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_delta_aic_fpp_vs_spp.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_coefficients_fpp_vs_spp.png",
    f"{HAZARD_NEURAL_OUTPUT_DIR}/figures/neural_power_by_partner_time.png",
]
HAZARD_NEURAL_ALL_OUTPUTS = [
    *HAZARD_NEURAL_RISKSET_OUTPUTS,
    *HAZARD_NEURAL_MODEL_OUTPUTS,
    *HAZARD_NEURAL_FIGURES,
]
# Dedicated low-level neural lag-selection outputs.
# Kept in a separate subdirectory so the legacy fixed-window neural outputs remain reproducible.
HAZARD_NEURAL_LAG_SELECTION_OUTPUTS = [
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_lag_selection.csv",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_selected_lags.json",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_model_comparison.csv",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_selected_model_comparison.csv",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_lag_null_summary.csv",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/models/neural_lowlevel_fit_metrics.json",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/figures/neural_lowlevel_delta_bic_by_lag.png",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/figures/neural_lowlevel_lag_null_comparison.png",
    f"{HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR}/figures/neural_lowlevel_selected_model_comparison.png",
]
HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    HAZARD_NEURAL_PERMUTATION_OUTPUT_CONFIG.get("output_dir"),
    "reports/hazard_neural_fpp/permutation_null",
)
NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    NEURAL_HAZARD_FPP_SPP_OUTPUT_CONFIG.get("out_dir"),
    "reports/neural_hazard/fpp_spp",
)
NEURAL_HAZARD_FPP_SPP_OUTPUTS = [
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/tables/model_comparison.csv",
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/tables/M2_entropy_coefficients.csv",
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/tables/circular_shift_summary.csv",
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/figures/predicted_hazard_by_entropy_anchor_type.png",
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/figures/circular_shift_null_delta_loglik.png",
    f"{NEURAL_HAZARD_FPP_SPP_OUTPUT_DIR}/summary.json",
]
HAZARD_NEURAL_PERMUTATION_NULL_INPUT_RISKSET = (
    HAZARD_NEURAL_PERMUTATION_INPUT_CONFIG.get("riskset_path")
    if isinstance(HAZARD_NEURAL_PERMUTATION_INPUT_CONFIG.get("riskset_path"), str)
    else f"{HAZARD_NEURAL_OUTPUT_DIR}/riskset/neural_fpp_hazard_table.parquet"
)
HAZARD_NEURAL_PERMUTATION_NULL_NEURAL_FAMILY = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("neural_family", "all")
)
HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES = (
    ["alpha", "beta", "alpha_beta"]
    if HAZARD_NEURAL_PERMUTATION_NULL_NEURAL_FAMILY == "all"
    else [HAZARD_NEURAL_PERMUTATION_NULL_NEURAL_FAMILY]
)
HAZARD_NEURAL_PERMUTATION_NULL_N_PERMUTATIONS = int(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("n_permutations", 1000)
)
HAZARD_NEURAL_PERMUTATION_NULL_SEED = int(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("seed", 12345)
)
HAZARD_NEURAL_PERMUTATION_NULL_N_JOBS = int(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("n_jobs", 1)
)
HAZARD_NEURAL_PERMUTATION_NULL_EVENT_COLUMN = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("event_column", "event_fpp")
)
HAZARD_NEURAL_PERMUTATION_NULL_EPISODE_COLUMN = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("episode_column", "episode_id")
)
HAZARD_NEURAL_PERMUTATION_NULL_PARTICIPANT_COLUMN = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("participant_column", "participant_speaker_id")
)
HAZARD_NEURAL_PERMUTATION_NULL_RUN_COLUMN = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("run_column", "run")
)
HAZARD_NEURAL_PERMUTATION_NULL_DELTA_CRITERION = str(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("delta_criterion", "bic")
)
HAZARD_NEURAL_PERMUTATION_NULL_VERBOSE = bool(
    HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get("verbose", True)
)
HAZARD_NEURAL_PERMUTATION_NULL_MAX_SMOKE = HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get(
    "max_permutations_for_smoke_test"
)
HAZARD_NEURAL_PERMUTATION_NULL_MAX_FIT_ROWS = HAZARD_NEURAL_PERMUTATION_RUN_CONFIG.get(
    "max_fit_rows"
)
HAZARD_NEURAL_PERMUTATION_NULL_OUTPUTS = [
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/fpp_neural_permutation_real_comparison.csv"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/fpp_neural_permutation_null_distribution.csv"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/fpp_neural_permutation_summary.json"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/fpp_neural_permutation_qc.json"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/fpp_neural_permutation_report.md"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/figures/fpp_neural_permutation_delta_bic_null.png"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/figures/fpp_neural_permutation_delta_bic_ecdf.png"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/figures/fpp_neural_permutation_real_vs_null_summary.png"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    *[
        f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/{family}/figures/fpp_neural_permutation_shift_qc.png"
        for family in HAZARD_NEURAL_PERMUTATION_NULL_FAMILIES
    ],
    f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/fpp_neural_permutation_combined_summary.csv",
    f"{HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR}/figures/fpp_neural_permutation_family_comparison.png",
]


NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR = _resolve_hazard_behavior_output_dir(
    "reports/neural_hazard/fpp_spp_renyi_alpha",
    "reports/neural_hazard/fpp_spp_renyi_alpha",
)
NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS = [
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_alpha_search_summary.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_best_alpha.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_circular_shift_summary.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_entropy_descriptives.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_alpha_entropy_correlation_raw.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_same_lag_model_comparison.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/tables/renyi_motor_exclusion_alpha_search_summary.csv",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/figures/renyi_alpha_search_delta_loglik.png",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/figures/renyi_alpha_entropy_correlation_heatmap.png",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/figures/renyi_same_lag_delta_loglik.png",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/figures/renyi_motor_exclusion_alpha_search_delta_loglik.png",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/figures/renyi_best_alpha_predicted_hazard_by_anchor_type.png",
    f"{NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUT_DIR}/summary.json",
]

def _hazard_neural_permutation_optional_flags() -> str:
    flags = []
    if HAZARD_NEURAL_PERMUTATION_NULL_VERBOSE:
        flags.append("--verbose")
    if HAZARD_NEURAL_PERMUTATION_NULL_MAX_SMOKE is not None:
        flags.extend(["--max-permutations-for-smoke-test", str(int(HAZARD_NEURAL_PERMUTATION_NULL_MAX_SMOKE))])
    if HAZARD_NEURAL_PERMUTATION_NULL_MAX_FIT_ROWS is not None:
        flags.extend(["--max-fit-rows", str(int(HAZARD_NEURAL_PERMUTATION_NULL_MAX_FIT_ROWS))])
    return " ".join(flags)


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


def _hazard_behavior_timing_control_flag() -> str:
    flags = []
    if HAZARD_BEHAVIOR_FIT_TIMING_CONTROL_MODELS:
        flags.extend(["--fit-timing-control-models", "--select-lags-with-timing-controls"])
    if HAZARD_BEHAVIOR_RUN_R_GLMM_LAG_SWEEP:
        flags.append("--run-r-glmm-lag-sweep")
    return " ".join(flags)


def _hazard_neural_lowlevel_inputs(wildcards) -> list[str]:
    del wildcards
    if not isinstance(LM_FEATURE_DIR, str) or not LM_FEATURE_DIR:
        raise ValueError("`lm_feature_dir` is missing from config/paths.yaml.")
    surprisal_pattern = os.path.join(LM_FEATURE_DIR, "**", "*desc-lmSurprisal_features.tsv")
    surprisal_matches = sorted(glob.glob(surprisal_pattern, recursive=True))
    if not surprisal_matches:
        raise ValueError(
            "No LM surprisal TSV files matched the neural hazard pattern under "
            f"{LM_FEATURE_DIR!r}."
        )
    lowlevel_pattern = os.path.join(OUT_DIR, "features", "neural_lowlevel", "**", "*desc-lowlevelNeural_features.tsv")
    lowlevel_matches = sorted(glob.glob(lowlevel_pattern, recursive=True))
    if not lowlevel_matches:
        raise ValueError(
            "No low-level neural TSV files matched the neural hazard pattern under "
            f"{os.path.join(OUT_DIR, 'features', 'neural_lowlevel')!r}."
        )
    return [EVENTS_CSV_OUTPUT, *surprisal_matches, *lowlevel_matches]


rule hazard_neural_lowlevel:
    input:
        _hazard_neural_lowlevel_inputs,
    output:
        HAZARD_NEURAL_ALL_OUTPUTS
    params:
        config_path=f"{PROJECT_ROOT}/config/hazard_fpp_tde_hmm.yaml",
        neural_out_dir=HAZARD_NEURAL_OUTPUT_DIR,
        surprisal_glob=lambda wildcards: os.path.join(
            LM_FEATURE_DIR,
            "**",
            "*desc-lmSurprisal_features.tsv",
        ),
        lowlevel_glob=lambda wildcards: os.path.join(
            OUT_DIR,
            "features",
            "neural_lowlevel",
            "**",
            "*desc-lowlevelNeural_features.tsv",
        ),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main hazard-fpp-tde-hmm \
          --config "{params.config_path}" \
          --neural-out-dir "{params.neural_out_dir}" \
          --neural-surprisal "{params.surprisal_glob}" \
          --neural-lowlevel "{params.lowlevel_glob}"
        """


rule hazard_neural_lowlevel_lag_selection:
    # Explicit Snakemake entrypoint for guarded causal neural lag selection.
    # This keeps lag-sweep artifacts separate from the default fixed-window neural hazard run.
    input:
        _hazard_neural_lowlevel_inputs,
    output:
        HAZARD_NEURAL_LAG_SELECTION_OUTPUTS
    params:
        config_path=f"{PROJECT_ROOT}/config/hazard_fpp_tde_hmm.yaml",
        neural_out_dir=HAZARD_NEURAL_LAG_SELECTION_OUTPUT_DIR,
        surprisal_glob=lambda wildcards: os.path.join(
            LM_FEATURE_DIR,
            "**",
            "*desc-lmSurprisal_features.tsv",
        ),
        lowlevel_glob=lambda wildcards: os.path.join(
            OUT_DIR,
            "features",
            "neural_lowlevel",
            "**",
            "*desc-lowlevelNeural_features.tsv",
        ),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        echo "[hazard_neural_lowlevel_lag_selection] Starting guarded causal neural lag selection."
        echo "[hazard_neural_lowlevel_lag_selection] Output directory: {params.neural_out_dir}"
        echo "[hazard_neural_lowlevel_lag_selection] Neural inputs: surprisal={params.surprisal_glob} lowlevel={params.lowlevel_glob}"
        echo "[hazard_neural_lowlevel_lag_selection] Enabling lag sweep, delta-BIC selection, and skip-spp-on-failure."
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main hazard-fpp-tde-hmm \
          --config "{params.config_path}" \
          --neural-out-dir "{params.neural_out_dir}" \
          --neural-surprisal "{params.surprisal_glob}" \
          --neural-lowlevel "{params.lowlevel_glob}" \
          --select-neural-lags \
          --neural-lag-selection-criterion bic \
          --skip-spp-on-failure
        echo "[hazard_neural_lowlevel_lag_selection] Finished neural lag selection."
        """


rule hazard_neural_permutation_null:
    input:
        riskset=HAZARD_NEURAL_PERMUTATION_NULL_INPUT_RISKSET,
    output:
        HAZARD_NEURAL_PERMUTATION_NULL_OUTPUTS
    params:
        output_dir=HAZARD_NEURAL_PERMUTATION_NULL_OUTPUT_DIR,
        neural_family=HAZARD_NEURAL_PERMUTATION_NULL_NEURAL_FAMILY,
        n_permutations=HAZARD_NEURAL_PERMUTATION_NULL_N_PERMUTATIONS,
        seed=HAZARD_NEURAL_PERMUTATION_NULL_SEED,
        n_jobs=HAZARD_NEURAL_PERMUTATION_NULL_N_JOBS,
        event_column=HAZARD_NEURAL_PERMUTATION_NULL_EVENT_COLUMN,
        episode_column=HAZARD_NEURAL_PERMUTATION_NULL_EPISODE_COLUMN,
        participant_column=HAZARD_NEURAL_PERMUTATION_NULL_PARTICIPANT_COLUMN,
        run_column=HAZARD_NEURAL_PERMUTATION_NULL_RUN_COLUMN,
        delta_criterion=HAZARD_NEURAL_PERMUTATION_NULL_DELTA_CRITERION,
        optional_flags=_hazard_neural_permutation_optional_flags(),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main run-fpp-neural-permutation-null \
          --riskset-path "{input.riskset}" \
          --output-dir "{params.output_dir}" \
          --neural-family "{params.neural_family}" \
          --n-permutations "{params.n_permutations}" \
          --seed "{params.seed}" \
          --event-column "{params.event_column}" \
          --episode-column "{params.episode_column}" \
          --participant-column "{params.participant_column}" \
          --run-column "{params.run_column}" \
          --delta-criterion "{params.delta_criterion}" \
          --n-jobs "{params.n_jobs}" \
          {params.optional_flags}
        """


rule run_neural_hazard_fpp_spp:
    input:
        config_path=f"{PROJECT_ROOT}/config/neural_hazard_fpp_spp.yaml",
        fpp_riskset=lambda wildcards: f"{BEHAVIOR_FINAL_ROOT}/fpp/riskset.parquet",
        spp_riskset=lambda wildcards: f"{BEHAVIOR_FINAL_ROOT}/spp_control/riskset.parquet",
        neural_features=GLHMM_ENTROPY_FEATURES_OUTPUT,
    output:
        model_comparison=NEURAL_HAZARD_FPP_SPP_OUTPUTS[0],
        coefficients=NEURAL_HAZARD_FPP_SPP_OUTPUTS[1],
        circular_shift_summary=NEURAL_HAZARD_FPP_SPP_OUTPUTS[2],
        predicted_hazard=NEURAL_HAZARD_FPP_SPP_OUTPUTS[3],
        circular_shift_histogram=NEURAL_HAZARD_FPP_SPP_OUTPUTS[4],
        summary_json=NEURAL_HAZARD_FPP_SPP_OUTPUTS[5],
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main neural-hazard-fpp-spp \
          --config "{input.config_path}"
        """


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
              --lag-grid-ms "{HAZARD_BEHAVIOR_R_GLMM_LAG_GRID}" \
              --output-csv "{output.export_csv}" \
              --output-qc-json "{output.export_qc_json}"
            """


    rule fit_behaviour_hazard_glmm:
        input:
            export_csv=HAZARD_BEHAVIOR_GLMM_EXPORT_OUTPUTS[0],
        output:
            information_rate_sweep=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[0],
            prop_expected_sweep=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[1],
            selected_lags=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[2],
            final_model_summary=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[3],
            final_model_comparison=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[4],
            final_model_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[5],
            prediction_information=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[6],
        params:
            output_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models",
            quadratic_timing_flag=(
                "--r-glmm-include-quadratic-offset-timing"
                if HAZARD_BEHAVIOR_R_GLMM_INCLUDE_QUADRATIC_OFFSET_TIMING
                else "--no-r-glmm-include-quadratic-offset-timing"
            ),
        shell:
            r"""
            set -euo pipefail
            Rscript "{PROJECT_ROOT}/scripts/r/fit_behaviour_glmm_lag_sweep.R" \
              --input-csv "{input.export_csv}" \
              --output-dir "{params.output_dir}" \
              --lag-grid-ms "{HAZARD_BEHAVIOR_R_GLMM_LAG_GRID}" \
              {params.quadratic_timing_flag} \
              --backend "{HAZARD_BEHAVIOR_R_GLMM_BACKEND}" \
              --include-run-random-effect "{HAZARD_BEHAVIOR_R_GLMM_INCLUDE_RUN_RANDOM_EFFECT_TEXT}" \
              --prop-expected-mode "{HAZARD_BEHAVIOR_R_GLMM_PROP_EXPECTED_MODE}" \
              --include-prop-expected-in-final "{HAZARD_BEHAVIOR_R_GLMM_INCLUDE_PROP_EXPECTED_IN_FINAL_TEXT}"
            """


    rule plot_behaviour_glmm_results:
        input:
            information_rate_sweep=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[0],
            prop_expected_sweep=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[1],
            selected_lags=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[2],
            final_model_summary=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[3],
            final_model_comparison=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[4],
            final_model_effects=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[5],
            prediction_information=HAZARD_BEHAVIOR_GLMM_MODEL_OUTPUTS[6],
        output:
            lag_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[0],
            coefficient_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[1],
            odds_ratio_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[2],
            final_comparison_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[3],
            final_prediction_figure=HAZARD_BEHAVIOR_GLMM_FIGURES[4],
        params:
            r_results_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models",
            output_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/figures",
            qc_output_dir=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/qc_plots/lag_selection",
        shell:
            r"""
            set -euo pipefail
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main plot-behaviour-hazard-results \
              --r-results-dir "{params.r_results_dir}" \
              --timing-control-models-dir "{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/lag_selection" \
              --qc-output-dir "{params.qc_output_dir}" \
              --output-dir "{params.output_dir}"
            """


    rule export_behaviour_latency_regime_data:
        input:
            riskset_tsv=HAZARD_BEHAVIOR_TIMING_CONTROL_RISKSET_OUTPUT,
            selected_lags_json=f"{HAZARD_BEHAVIOR_OUTPUT_DIR}/models/behaviour_timing_control_selected_lags.json",
        output:
            export_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS[0],
            export_qc_json=HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS[1],
        shell:
            r"""
            set -euo pipefail
            echo "[latency-regime] Exporting event-only Stan input CSV"
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main export-behaviour-latency-regime-data \
              --input-riskset "{input.riskset_tsv}" \
              --selected-lags-json "{input.selected_lags_json}" \
              --output-csv "{output.export_csv}" \
              --output-qc-json "{output.export_qc_json}" \
              --verbose
            """


    rule fit_behaviour_latency_regime_stan:
        input:
            export_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS[0],
        output:
            model_a_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[0],
            model_s_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[1],
            model_b_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[2],
            model_c_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[3],
            model_r1_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[4],
            model_r2_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[5],
            model_r3_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[6],
            model_r4_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[7],
            loo_comparison=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[8],
            fit_metrics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[9],
            component_parameters=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[10],
            gating_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[11],
            event_probabilities=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[12],
            posterior_predictive=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[13],
            regression_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[14],
            regression_predictions=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[15],
            shifted_lognormal_diagnostics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[16],
        params:
            output_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models",
            stan_dir=f"{PROJECT_ROOT}/scripts/stan",
        shell:
            r"""
            set -euo pipefail
            echo "[latency-regime] Fitting latency-regime models A/S/B/C/R1-R4"
            Rscript "{PROJECT_ROOT}/scripts/r/fit_behaviour_latency_regime_stan.R" \
              --input-csv "{input.export_csv}" \
              --output-dir "{params.output_dir}" \
              --stan-dir "{params.stan_dir}" \
              --verbose
            """


    rule plot_behaviour_latency_regime_results:
        input:
            event_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS[0],
            model_a_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[0],
            model_s_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[1],
            model_b_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[2],
            model_c_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[3],
            model_r1_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[4],
            model_r2_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[5],
            model_r3_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[6],
            model_r4_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[7],
            loo_comparison=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[8],
            fit_metrics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[9],
            component_parameters=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[10],
            gating_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[11],
            event_probabilities=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[12],
            posterior_predictive=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[13],
            regression_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[14],
            regression_predictions=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[15],
            shifted_lognormal_diagnostics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[16],
        output:
            components=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[0],
            probability_by_expected=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[1],
            probability_by_rate=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[2],
            gating_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[3],
            skew_vs_mixture=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[4],
            regression_vs_mixture=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[5],
            ppc=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[6],
            loo=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[7],
            latency_by_expected_regression=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[8],
            latency_by_rate_regression=HAZARD_BEHAVIOR_LATENCY_REGIME_FIGURES[9],
        params:
            stan_results_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models",
            output_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures",
        shell:
            r"""
            set -euo pipefail
            echo "[latency-regime] Plotting exploratory latency-regime figures"
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main plot-behaviour-latency-regime-results \
              --stan-results-dir "{params.stan_results_dir}" \
              --event-data-csv "{input.event_csv}" \
              --output-dir "{params.output_dir}" \
              --verbose
            """


    rule diagnose_behaviour_latency_regime_bimodality:
        input:
            event_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_EXPORT_OUTPUTS[0],
            model_a_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[0],
            model_s_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[1],
            model_b_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[2],
            model_c_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[3],
            model_r1_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[4],
            model_r2_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[5],
            model_r3_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[6],
            model_r4_summary=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[7],
            loo_comparison=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[8],
            fit_metrics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[9],
            component_parameters=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[10],
            gating_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[11],
            event_probabilities=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[12],
            posterior_predictive=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[13],
            regression_coefficients=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[14],
            regression_predictions=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[15],
            shifted_lognormal_diagnostics=HAZARD_BEHAVIOR_LATENCY_REGIME_MODEL_OUTPUTS[16],
        output:
            conditional_density_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[0],
            ppc_expected_info_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[1],
            ppc_information_rate_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[2],
            residual_density_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[3],
            elpd_latency_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[4],
            elpd_expected_info_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[5],
            elpd_information_rate_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[6],
            elpd_p_late_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[7],
            p_late_vs_r_mu_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[8],
            latency_vs_r_mu_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[9],
            latency_vs_information_rate_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[10],
            latency_vs_expected_cum_info_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[11],
            counterfactual_figure=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_FIGURES[12],
            conditional_density_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[0],
            ppc_by_quantile_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[1],
            residuals_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[2],
            residual_bimodality_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[3],
            pointwise_elpd_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[4],
            p_late_vs_r_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[5],
            p_late_r_correlations_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[6],
            counterfactual_csv=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[7],
            report_md=HAZARD_BEHAVIOR_LATENCY_REGIME_BIMODALITY_DIAGNOSTICS[8],
        params:
            stan_results_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/stan_models",
            figures_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/figures",
            diagnostics_dir=f"{HAZARD_BEHAVIOR_LATENCY_REGIME_OUTPUT_DIR}/diagnostics",
        shell:
            r"""
            set -euo pipefail
            echo "[latency-regime] Running conditional-vs-marginal bimodality diagnostics"
            PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main diagnose-behaviour-latency-regime-bimodality \
              --stan-results-dir "{params.stan_results_dir}" \
              --event-data-csv "{input.event_csv}" \
              --figures-dir "{params.figures_dir}" \
              --diagnostics-dir "{params.diagnostics_dir}" \
              --verbose
            """

BEHAVIOR_FINAL_CONFIG_PATH = "config/behavior.yaml"
with open(BEHAVIOR_FINAL_CONFIG_PATH, encoding="utf-8") as _f:
    _BEHAVIOR_FINAL_CONFIG = yaml.safe_load(_f) or {}
BEHAVIOR_FINAL_ROOT = _resolve_hazard_behavior_output_dir(
    dict(_BEHAVIOR_FINAL_CONFIG.get("paths") or {}).get("out_dir"),
    "reports/hazard_behavior_final",
)

rule behavior_final_lag_selection:
    input:
        config=BEHAVIOR_FINAL_CONFIG_PATH
    output:
        selected_lag=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/selected_lag.json",
        table=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/fpp_lag_selection_table.csv",
        qc_manifest=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/qc_plot_manifest.json"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior-final-select-lag \
          --config {input.config} \
          --out-dir {BEHAVIOR_FINAL_ROOT}/lag_selection \
          --verbose
        """


rule behavior_final_fpp:
    input:
        config=BEHAVIOR_FINAL_CONFIG_PATH,
        selected_lag=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/selected_lag.json"
    output:
        riskset=f"{BEHAVIOR_FINAL_ROOT}/fpp/riskset.parquet",
        summary=f"{BEHAVIOR_FINAL_ROOT}/fpp/models/model_summary.csv",
        interaction_summary=f"{BEHAVIOR_FINAL_ROOT}/fpp/models/timing_information_rate_interaction_summary.csv",
        interaction_coefficients=f"{BEHAVIOR_FINAL_ROOT}/fpp/models/timing_information_rate_interaction_coefficients.csv",
        interaction_comparison=f"{BEHAVIOR_FINAL_ROOT}/fpp/models/timing_information_rate_interaction_comparison.csv",
        interaction_onset_figure=f"{BEHAVIOR_FINAL_ROOT}/fpp/figures/timing_information_rate_interaction_onset.png",
        interaction_offset_figure=f"{BEHAVIOR_FINAL_ROOT}/fpp/figures/timing_information_rate_interaction_offset.png"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior-final-fit \
          --config {input.config} \
          --anchor-type fpp \
          --selected-lag {input.selected_lag} \
          --out-dir {BEHAVIOR_FINAL_ROOT}/fpp \
          --verbose
        """


rule behavior_final_spp:
    input:
        config=BEHAVIOR_FINAL_CONFIG_PATH,
        selected_lag=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/selected_lag.json"
    output:
        riskset=f"{BEHAVIOR_FINAL_ROOT}/spp_control/riskset.parquet",
        summary=f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/model_summary.csv",
        interaction_summary=f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/timing_information_rate_interaction_summary.csv",
        interaction_coefficients=f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/timing_information_rate_interaction_coefficients.csv",
        interaction_comparison=f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/timing_information_rate_interaction_comparison.csv",
        interaction_onset_figure=f"{BEHAVIOR_FINAL_ROOT}/spp_control/figures/timing_information_rate_interaction_onset.png",
        interaction_offset_figure=f"{BEHAVIOR_FINAL_ROOT}/spp_control/figures/timing_information_rate_interaction_offset.png"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior-final-fit \
          --config {input.config} \
          --anchor-type spp \
          --selected-lag {input.selected_lag} \
          --out-dir {BEHAVIOR_FINAL_ROOT}/spp_control \
          --verbose
        """


rule behavior_final_fpp_vs_spp:
    input:
        config=BEHAVIOR_FINAL_CONFIG_PATH,
        selected_lag=f"{BEHAVIOR_FINAL_ROOT}/lag_selection/selected_lag.json",
        fpp=f"{BEHAVIOR_FINAL_ROOT}/fpp/riskset.parquet",
        spp=f"{BEHAVIOR_FINAL_ROOT}/spp_control/riskset.parquet"
    output:
        summary=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/interaction_model_summary.csv",
        contrasts=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/information_effect_contrasts.csv",
        pooled_interaction_summary=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/timing_information_rate_anchor_interaction_summary.csv",
        pooled_interaction_coefficients=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/timing_information_rate_anchor_interaction_coefficients.csv",
        pooled_interaction_contrasts=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/timing_information_rate_anchor_interaction_contrasts.csv",
        report=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/fpp_vs_spp_report.md",
        qc_manifest=f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/qc_plot_manifest.json"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior-final-compare \
          --config {input.config} \
          --selected-lag {input.selected_lag} \
          --fpp-riskset {input.fpp} \
          --spp-riskset {input.spp} \
          --out-dir {BEHAVIOR_FINAL_ROOT}/fpp_vs_spp \
          --verbose
        """


rule behavior_final_all:
    input:
        f"{BEHAVIOR_FINAL_ROOT}/lag_selection/selected_lag.json",
        f"{BEHAVIOR_FINAL_ROOT}/fpp/models/model_summary.csv",
        f"{BEHAVIOR_FINAL_ROOT}/fpp/models/timing_information_rate_interaction_summary.csv",
        f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/model_summary.csv",
        f"{BEHAVIOR_FINAL_ROOT}/spp_control/models/timing_information_rate_interaction_summary.csv",
        f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/information_effect_contrasts.csv",
        f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/timing_information_rate_anchor_interaction_summary.csv",
        f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/fpp_vs_spp_report.md",
        f"{BEHAVIOR_FINAL_ROOT}/fpp_vs_spp/qc_plot_manifest.json"


rule run_neural_hazard_fpp_spp_renyi_alpha:
    input:
        config_path=f"{PROJECT_ROOT}/config/neural_hazard_fpp_spp_renyi_alpha.yaml",
        fpp_riskset=lambda wildcards: f"{BEHAVIOR_FINAL_ROOT}/fpp/riskset.parquet",
        spp_riskset=lambda wildcards: f"{BEHAVIOR_FINAL_ROOT}/spp_control/riskset.parquet",
        neural_features=GLHMM_ENTROPY_FEATURES_OUTPUT,
    output:
        alpha_summary=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[0],
        best_alpha=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[1],
        null_summary=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[2],
        entropy_descriptives=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[3],
        alpha_correlation=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[4],
        same_lag=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[5],
        motor_summary=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[6],
        alpha_curve=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[7],
        corr_heatmap=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[8],
        same_lag_plot=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[9],
        motor_plot=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[10],
        best_hazard=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[11],
        summary_json=NEURAL_HAZARD_FPP_SPP_RENYI_ALPHA_OUTPUTS[12],
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main neural-hazard-fpp-spp-renyi-alpha \
          --config "{input.config_path}"
        """
