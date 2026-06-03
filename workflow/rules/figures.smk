VIZ_CONFIG_PATH = f"{CONFIG_DIR}/viz.yaml"

EVOKED_FIGURES_OUTPUT_DIR = _resolve_erp_output_dir(None, "figures")
EVOKED_SENSOR_QC_MANIFEST = (
    f"{EVOKED_FIGURES_OUTPUT_DIR}/qc/evoked/sensor_lmeeeg/figure_manifest.json"
)
EVOKED_SENSOR_FIGURE_MANIFEST = (
    f"{EVOKED_FIGURES_OUTPUT_DIR}/main/evoked/sensor_lmeeeg/figure_manifest.json"
)

INDUCED_SENSOR_PRIMARY_QC_MANIFEST = (
    f"{OUT_DIR}/figures/qc/induced/sensor_lmeeeg/figure_manifest.json"
)
INDUCED_SENSOR_PRIMARY_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/main/induced/sensor_lmeeeg/figure_manifest.json"
)
INDUCED_SENSOR_CYCLE_POSITION_QC_MANIFEST = (
    f"{OUT_DIR}/figures/qc/induced/sensor_lmeeeg/cycle_position/figure_manifest.json"
)
INDUCED_SENSOR_CYCLE_POSITION_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/supp/induced/sensor_lmeeeg/cycle_position/figure_manifest.json"
)
INDUCED_SENSOR_CONF_DISC_FPP_LOCKED_QC_MANIFEST = (
    f"{OUT_DIR}/figures/qc/induced/sensor_lmeeeg/conf_disc/fpp_locked/figure_manifest.json"
)
INDUCED_SENSOR_CONF_DISC_FPP_LOCKED_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/supp/induced/sensor_lmeeeg/conf_disc/fpp_locked/figure_manifest.json"
)
INDUCED_SENSOR_CONF_DISC_SPP_LOCKED_QC_MANIFEST = (
    f"{OUT_DIR}/figures/qc/induced/sensor_lmeeeg/conf_disc/spp_locked/figure_manifest.json"
)
INDUCED_SENSOR_CONF_DISC_SPP_LOCKED_FIGURE_MANIFEST = (
    f"{OUT_DIR}/figures/supp/induced/sensor_lmeeeg/conf_disc/spp_locked/figure_manifest.json"
)


rule induced_sensor_qc_primary:
    input:
        summary=LMEEEG_INDUCED_SUMMARY_OUTPUT,
        config=LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_PRIMARY_QC_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced qc \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
        --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule evoked_sensor_qc_primary:
    input:
        summary=LMEEEG_SUMMARY_OUTPUT,
        config=EVOKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=EVOKED_SENSOR_QC_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced qc \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule evoked_sensor_figures_primary:
    input:
        summary=LMEEEG_SUMMARY_OUTPUT,
        config=EVOKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=EVOKED_SENSOR_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced figures \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_figures_primary:
    input:
        summary=LMEEEG_INDUCED_SUMMARY_OUTPUT,
        config=LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_PRIMARY_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced figures \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_qc_cycle_position:
    input:
        summary=FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CYCLE_POSITION_QC_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced qc \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_figures_cycle_position:
    input:
        summary=FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CYCLE_POSITION_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CYCLE_POSITION_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced figures \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_qc_conf_disc_fpp_locked:
    input:
        summary=FPP_SPP_CONF_DISC_ALPHA_BETA_FPP_LOCKED_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_FPP_LOCKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CONF_DISC_FPP_LOCKED_QC_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced qc \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_figures_conf_disc_fpp_locked:
    input:
        summary=FPP_SPP_CONF_DISC_ALPHA_BETA_FPP_LOCKED_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_FPP_LOCKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CONF_DISC_FPP_LOCKED_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced figures \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_qc_conf_disc_spp_locked:
    input:
        summary=FPP_SPP_CONF_DISC_ALPHA_BETA_SPP_LOCKED_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_SPP_LOCKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CONF_DISC_SPP_LOCKED_QC_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced qc \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """


rule induced_sensor_figures_conf_disc_spp_locked:
    input:
        summary=FPP_SPP_CONF_DISC_ALPHA_BETA_SPP_LOCKED_LMEEEG_SUMMARY_OUTPUT,
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_SPP_LOCKED_LMEEEG_CONFIG_PATH,
        viz=VIZ_CONFIG_PATH,
    output:
        manifest=INDUCED_SENSOR_CONF_DISC_SPP_LOCKED_FIGURE_MANIFEST,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{resources.tmpdir}/mpl" "{resources.tmpdir}/cache"
        MPLCONFIGDIR="{resources.tmpdir}/mpl" XDG_CACHE_HOME="{resources.tmpdir}/cache" \
        PYTHONPATH="{SRC_DIR}:{PROJECT_ROOT}" "{PYTHON_BIN}" -m cas.cli.main induced figures \
          --config-root "{CONFIG_DIR}" \
          --config "{input.config}" \
          --viz-config "{input.viz}" \
          --output "{output.manifest}"
        """
