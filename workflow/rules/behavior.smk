import os

with open(BEHAVIOR_HAZARD_CONFIG_PATH := f"{PROJECT_ROOT}/config/behavior/hazard.yaml", encoding="utf-8") as f:
    BEHAVIOR_HAZARD_RULE_CONFIG = yaml.safe_load(f) or {}

BEHAVIOR_HAZARD_SECTION = dict(((BEHAVIOR_HAZARD_RULE_CONFIG.get("behavior") or {}).get("hazard") or {}))
BEHAVIOR_HAZARD_FIGURE_CONFIG = dict(BEHAVIOR_HAZARD_SECTION.get("figures") or {})
BEHAVIOR_HAZARD_SUPP_CONFIG = dict(BEHAVIOR_HAZARD_FIGURE_CONFIG.get("supplementary") or {})
BEHAVIOR_HAZARD_MAIN_CONFIG = dict(BEHAVIOR_HAZARD_FIGURE_CONFIG.get("main") or {})
BEHAVIOR_HAZARD_DIAGNOSTIC_CONFIG = dict(BEHAVIOR_HAZARD_SECTION.get("diagnostics") or {})

BEHAVIOR_OUTPUT_DIR = PATHS_CONFIG.get("output_dir", OUT_DIR)
BEHAVIOR_HAZARD_ROOT = f"{BEHAVIOR_OUTPUT_DIR}/behavior/hazard"
BEHAVIOR_FIGURES_MAIN = f"{BEHAVIOR_OUTPUT_DIR}/figures/main/behavior"
BEHAVIOR_FIGURES_SUPP = f"{BEHAVIOR_OUTPUT_DIR}/figures/supp/behavior"
BEHAVIOR_FIGURES_QC = f"{BEHAVIOR_OUTPUT_DIR}/figures/qc/behavior"

BEHAVIOR_HAZARD_RISKSET_OUTPUTS = [
    f"{BEHAVIOR_HAZARD_ROOT}/risksets/fpp.parquet",
    f"{BEHAVIOR_HAZARD_ROOT}/risksets/spp_control.parquet",
    f"{BEHAVIOR_HAZARD_ROOT}/risksets/pooled_fpp_spp.parquet",
]

BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS = [
    f"{BEHAVIOR_HAZARD_ROOT}/predictors/fpp_with_lags.parquet",
    f"{BEHAVIOR_HAZARD_ROOT}/predictors/spp_control_with_lags.parquet",
    f"{BEHAVIOR_HAZARD_ROOT}/predictors/pooled_with_lags.parquet",
    f"{BEHAVIOR_HAZARD_ROOT}/predictors/standardization_summary.csv",
]

BEHAVIOR_HAZARD_LAG_OUTPUTS = [
    f"{BEHAVIOR_HAZARD_ROOT}/lag_selection/candidate_lag_scores.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/lag_selection/selected_lag.json",
]

BEHAVIOR_HAZARD_TABLE_OUTPUTS = [
    f"{BEHAVIOR_HAZARD_ROOT}/tables/model_comparisons.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/model_summary.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/coefficient_summary.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/odds_ratios.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/event_rate_summary.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/bins_by_subject.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/collinearity_summary.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/figure_predictions.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/timing_heatmap_predictions.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/tables/three_way_heatmap_predictions.csv",
]

BEHAVIOR_HAZARD_DIAGNOSTIC_OUTPUTS = [
    f"{BEHAVIOR_HAZARD_ROOT}/diagnostics/convergence_warnings.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/diagnostics/lag_sensitivity.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/diagnostics/loo_subject_summary.csv",
    f"{BEHAVIOR_HAZARD_ROOT}/logs/summary_manifest.json",
]

BEHAVIOR_HAZARD_FIGURE_OUTPUTS = [
    f"{BEHAVIOR_FIGURES_MAIN}/fig01_lag_selection.png",
    f"{BEHAVIOR_FIGURES_MAIN}/fig02_primary_information_effects.png",
    f"{BEHAVIOR_FIGURES_MAIN}/fig03_timing_information_interaction.png",
]
if bool(BEHAVIOR_HAZARD_MAIN_CONFIG.get("three_way_interaction", False)):
    BEHAVIOR_HAZARD_FIGURE_OUTPUTS.append(f"{BEHAVIOR_FIGURES_MAIN}/fig04_three_way_interaction.png")

BEHAVIOR_HAZARD_SUPP_OUTPUTS = [
    f"{BEHAVIOR_FIGURES_SUPP}/figS01_lag_sensitivity.png",
    f"{BEHAVIOR_FIGURES_SUPP}/figS02_extra_timing_maps.png",
]
if bool(BEHAVIOR_HAZARD_SUPP_CONFIG.get("three_way_interaction", False)):
    BEHAVIOR_HAZARD_SUPP_OUTPUTS.append(f"{BEHAVIOR_FIGURES_SUPP}/figS03_three_way_interaction.png")
if bool(BEHAVIOR_HAZARD_SUPP_CONFIG.get("leave_one_subject_out", False)):
    BEHAVIOR_HAZARD_SUPP_OUTPUTS.append(f"{BEHAVIOR_FIGURES_SUPP}/figS04_leave_one_subject_out.png")

BEHAVIOR_HAZARD_QC_OUTPUTS = [
    f"{BEHAVIOR_FIGURES_QC}/qc_event_rates.png",
    f"{BEHAVIOR_FIGURES_QC}/qc_bins_by_subject.png",
    f"{BEHAVIOR_FIGURES_QC}/qc_collinearity.png",
    f"{BEHAVIOR_FIGURES_QC}/qc_model_convergence.png",
]


rule behavior_hazard_risksets:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH
    output:
        *BEHAVIOR_HAZARD_RISKSET_OUTPUTS
    shell:
        r"""
        set -euo pipefail
        printf '%s\n' "[behavior_hazard_risksets] config: {BEHAVIOR_HAZARD_CONFIG_PATH}"
        printf '%s\n' "[behavior_hazard_risksets] outputs: {output}"
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard build-risksets --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_predictors:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_RISKSET_OUTPUTS
    output:
        *BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard add-predictors --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_lag_selection:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS
    output:
        *BEHAVIOR_HAZARD_LAG_OUTPUTS
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard select-lag --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_models:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS,
        *BEHAVIOR_HAZARD_LAG_OUTPUTS
    output:
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_0.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_1.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_2.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_3.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_4.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_0.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_1.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_2.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_3.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_4.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/pooled_anchor/M_pooled_main.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/pooled_anchor/M_pooled_anchor_interaction.json",
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard fit-models --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_tables:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS,
        *BEHAVIOR_HAZARD_LAG_OUTPUTS,
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_0.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_1.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_2.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_3.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/fpp/M_4.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_0.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_1.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_2.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_3.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/spp_control/M_4.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/pooled_anchor/M_pooled_main.json",
        f"{BEHAVIOR_HAZARD_ROOT}/models/pooled_anchor/M_pooled_anchor_interaction.json",
    output:
        *BEHAVIOR_HAZARD_TABLE_OUTPUTS,
        *BEHAVIOR_HAZARD_DIAGNOSTIC_OUTPUTS
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard tables --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_figures:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_TABLE_OUTPUTS,
        *BEHAVIOR_HAZARD_LAG_OUTPUTS
    output:
        *BEHAVIOR_HAZARD_FIGURE_OUTPUTS,
        *BEHAVIOR_HAZARD_SUPP_OUTPUTS,
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard figures --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_qc:
    input:
        BEHAVIOR_HAZARD_CONFIG_PATH,
        *BEHAVIOR_HAZARD_TABLE_OUTPUTS,
        *BEHAVIOR_HAZARD_DIAGNOSTIC_OUTPUTS
    output:
        *BEHAVIOR_HAZARD_QC_OUTPUTS
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main behavior hazard qc --config {BEHAVIOR_HAZARD_CONFIG_PATH} --verbose
        """


rule behavior_hazard_all:
    input:
        *BEHAVIOR_HAZARD_RISKSET_OUTPUTS,
        *BEHAVIOR_HAZARD_PREDICTOR_OUTPUTS,
        *BEHAVIOR_HAZARD_LAG_OUTPUTS,
        *BEHAVIOR_HAZARD_TABLE_OUTPUTS,
        *BEHAVIOR_HAZARD_DIAGNOSTIC_OUTPUTS,
        *BEHAVIOR_HAZARD_FIGURE_OUTPUTS,
        *BEHAVIOR_HAZARD_SUPP_OUTPUTS,
        *BEHAVIOR_HAZARD_QC_OUTPUTS
