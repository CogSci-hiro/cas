VIZ_CONFIG_PATH = f"{CONFIG_DIR}/viz.yaml"
LMEEEG_FIGURE_MANIFEST = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_INFERENCE_FIGURE_MANIFEST = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"


rule figures_lmeeeg:
    input:
        summary=LMEEEG_SUMMARY_OUTPUT,
        config=LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=LMEEEG_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main figures-lmeeeg \
          --config-root "{CONFIG_DIR}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule figures_lmeeeg_inference:
    input:
        summary=LMEEEG_SUMMARY_OUTPUT,
        config=LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=LMEEEG_INFERENCE_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main figures-lmeeeg-inference \
          --config-root "{CONFIG_DIR}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """
