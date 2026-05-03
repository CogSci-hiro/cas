SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH = (
    f"{CONFIG_DIR}/source_dics_fpp_spp_alpha_beta.yaml"
)
SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT = (
    f"{PROJECT_ROOT}/results/source_dics_fpp_spp_alpha_beta/qc/run_summary.json"
)


rule run_source_dics_fpp_spp_alpha_beta:
    input:
        config=SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH,
    output:
        summary=SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT,
    shell:
        (
            "PYTHONPATH={SRC_DIR} "
            "{PYTHON_BIN} -m cas.cli.main source-dics-fpp-spp "
            "--config {input.config}"
        )
