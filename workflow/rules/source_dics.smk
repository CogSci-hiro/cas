from pathlib import Path

SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH = (
    f"{CONFIG_DIR}/source_dics_fpp_spp_alpha_beta.yaml"
)

with open(SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG_PATH, encoding="utf-8") as _source_dics_handle:
    _SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG = yaml.safe_load(_source_dics_handle) or {}

SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT = (
    str(
        Path(_SOURCE_DICS_FPP_SPP_ALPHA_BETA_CONFIG["paths"]["derivatives_dir"])
        / "qc"
        / "run_summary.json"
    )
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
