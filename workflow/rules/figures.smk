VIZ_CONFIG_PATH = f"{CONFIG_DIR}/viz.yaml"
LMEEEG_FIGURE_MANIFEST = f"{OUT_DIR}/figures/lmeeeg/figure_manifest.json"
LMEEEG_INFERENCE_FIGURE_MANIFEST = f"{OUT_DIR}/figures/lmeeeg_inference/figure_manifest.json"
FPP_SPP_CYCLE_POSITION_LMEEEG_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/fpp_spp_cycle_position/figure_manifest.json"
)
FPP_SPP_CYCLE_POSITION_LMEEEG_INFERENCE_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/fpp_spp_cycle_position_inference/figure_manifest.json"
)


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
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
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
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main figures-lmeeeg-inference \
          --config-root "{CONFIG_DIR}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule figures_fpp_spp_cycle_position_lmeeeg:
    input:
        summary=FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=FPP_SPP_CYCLE_POSITION_LMEEEG_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main figures-lmeeeg \
          --config-root "{CONFIG_DIR}" \
          --lmeeeg-config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule figures_fpp_spp_cycle_position_lmeeeg_inference:
    input:
        summary=FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=FPP_SPP_CYCLE_POSITION_LMEEEG_INFERENCE_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main figures-lmeeeg-inference \
          --config-root "{CONFIG_DIR}" \
          --lmeeeg-config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """
