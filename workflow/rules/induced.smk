FPP_SPP_CONF_DISC_ALPHA_BETA_DOWNSAMPLED_SUMMARY_OUTPUTS = expand(
    (
        f"{OUT_DIR}/{FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_SUBDIR}/"
        "sub-{subject}/summary.json"
    ),
    subject=INDUCED_EPOCH_SUBJECTS,
)


rule downsample_fpp_spp_conf_disc_induced_epochs_subject:
    input:
        summary=f"{OUT_DIR}/induced_epochs/sub-{{subject}}/summary.json",
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG_PATH,
    output:
        summary=(
            f"{OUT_DIR}/{FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_INDUCED_SUBDIR}/"
            "sub-{subject}/summary.json"
        ),
    run:
        from cas.eeg.induced.downsample import downsample_subject_induced_epochs

        downsample_subject_induced_epochs(
            subject=wildcards.subject,
            summary_path=input.summary,
            config_path=input.config,
            output_root=OUT_DIR,
        )


rule build_induced_sensor_conf_disc:
    input:
        epochs=EPOCH_OUTPUTS,
        induced=FPP_SPP_CONF_DISC_ALPHA_BETA_DOWNSAMPLED_SUMMARY_OUTPUTS,
        config=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONFIG_PATH,
    output:
        summary=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_SUMMARY_OUTPUT,
        model_summaries=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODEL_SUMMARY_OUTPUTS,
        contrast=FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONTRAST_OUTPUTS,
    run:
        import os

        from cas.stats.lmeeeg_pipeline import run_pooled_lmeeeg_analysis

        run_pooled_lmeeeg_analysis(
            epochs_paths=list(input.epochs),
            config_path=input.config,
            output_dir=os.path.dirname(output.summary),
        )


INDUCED_POWER_OUTPUTS = [
    *INDUCED_EPOCH_SUMMARY_OUTPUTS,
    *_induced_epoch_band_summary_outputs(),
    *FPP_SPP_CONF_DISC_ALPHA_BETA_DOWNSAMPLED_SUMMARY_OUTPUTS,
    *INDUCED_SOURCE_EPOCH_OUTPUTS,
]

INDUCED_SENSOR_LMEEEG_OUTPUTS = [
    LMEEEG_INDUCED_SUMMARY_OUTPUT,
    FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
    *FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS,
    FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_SUMMARY_OUTPUT,
    *FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODEL_SUMMARY_OUTPUTS,
    *FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONTRAST_OUTPUTS,
    *INFO_RATE_INDUCED_LMEEEG_OUTPUTS,
]

INDUCED_SOURCE_LMEEEG_OUTPUTS = [
    SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT,
]

INDUCED_TABLE_OUTPUTS = [
    LMEEEG_INDUCED_SUMMARY_OUTPUT,
    FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT,
    *FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS,
    FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_SUMMARY_OUTPUT,
    *FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_MODEL_SUMMARY_OUTPUTS,
    *FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONTRAST_OUTPUTS,
    *INFO_RATE_INDUCED_LMEEEG_OUTPUTS,
]

INDUCED_FIGURE_OUTPUTS = [
    INDUCED_SENSOR_PRIMARY_FIGURE_MANIFEST,
    INDUCED_SENSOR_CYCLE_POSITION_FIGURE_MANIFEST,
    INDUCED_SENSOR_CONF_DISC_FIGURE_MANIFEST,
    SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX,
]

INDUCED_QC_OUTPUTS = [
    INDUCED_SENSOR_PRIMARY_QC_MANIFEST,
    INDUCED_SENSOR_CYCLE_POSITION_QC_MANIFEST,
    INDUCED_SENSOR_CONF_DISC_QC_MANIFEST,
    *INDUCED_EPOCH_SUMMARY_OUTPUTS,
    SOURCE_DICS_FPP_SPP_ALPHA_BETA_SUMMARY_OUTPUT,
]


rule induced_power:
    input:
        INDUCED_POWER_OUTPUTS


rule induced_sensor_lmeeeg:
    input:
        INDUCED_SENSOR_LMEEEG_OUTPUTS


rule induced_source_lmeeeg:
    input:
        INDUCED_SOURCE_LMEEEG_OUTPUTS


rule induced_tables:
    input:
        INDUCED_TABLE_OUTPUTS


rule induced_figures:
    input:
        INDUCED_FIGURE_OUTPUTS


rule induced_qc:
    input:
        INDUCED_QC_OUTPUTS


rule induced_all:
    input:
        rules.induced_power.input,
        rules.induced_sensor_lmeeeg.input,
        rules.induced_source_lmeeeg.input,
        rules.induced_tables.input,
        rules.induced_figures.input,
        rules.induced_qc.input
