rule trf:
    input:
        rules.trf_all.input


rule spp_trf:
    input:
        rules.spp_trf_all.input


rule evoked_sensor_lmeeeg:
    input:
        LMEEEG_SUMMARY_OUTPUT,
        EVOKED_SENSOR_QC_MANIFEST,
        EVOKED_SENSOR_FIGURE_MANIFEST


rule confirmatory_core:
    input:
        f"{OUT_DIR}/dataset/index.parquet",
        f"{OUT_DIR}/events/events.parquet",
        f"{OUT_DIR}/features/entropy/entropy.parquet",
        f"{OUT_DIR}/neural/beta/beta.parquet",
        f"{OUT_DIR}/neural/alpha/alpha.parquet",
        f"{OUT_DIR}/models/lagged_logistic/results.parquet",
        f"{OUT_DIR}/models/hazard/results.parquet",
        f"{OUT_DIR}/models/mediation/results.parquet",
        f"{OUT_DIR}/stats/n400/results.parquet",
        f"{OUT_DIR}/_meta/run_complete.txt"


rule exploratory:
    input:
        rules.behavior_hazard_all.input


rule all:
    input:
        rules.confirmatory_core.input
