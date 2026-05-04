from __future__ import annotations

from pathlib import Path


def test_fpp_spp_cycle_position_workflow_target_is_wired() -> None:
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")
    figures_text = Path("workflow/rules/figures.smk").read_text(encoding="utf-8")

    assert "rule run_fpp_spp_cycle_position_lmeeeg:" in erp_text
    assert "FPP_SPP_CYCLE_POSITION_LMEEEG_CONTRAST_OUTPUTS" in erp_text
    assert "rule lme_eeg_fpp_spp_cycle_position:" in targets_text
    assert "FPP_SPP_CYCLE_POSITION_LMEEEG_SUMMARY_OUTPUT" in targets_text
    assert "rule figures_fpp_spp_cycle_position_lmeeeg:" in figures_text
    assert "rule figures_fpp_spp_cycle_position_lmeeeg_inference:" in figures_text
    assert "rule figures_lme_eeg_fpp_spp_cycle_position:" in targets_text


def test_fpp_spp_cycle_position_config_contains_requested_contrast() -> None:
    config_text = Path("config/lmeeeg_fpp_spp_cycle_position.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "fpp_spp_cycle_position"' in config_text
    assert 'formula: "power ~ pair_position + z_event_duration + z_latency + run + z_time_within_run + (1 | subject)"' in config_text
    assert 'contrast_of_interest: "pair_positionFPP"' in config_text
    assert 'pair_position: "SPP"' in config_text


def test_info_rate_induced_lmeeg_workflow_target_is_wired() -> None:
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule run_info_rate_induced_lmeeg:" in erp_text
    assert "INFO_RATE_INDUCED_LMEEEG_OUTPUTS" in erp_text
    assert "rule info_rate_induced_lmeeg_all:" in targets_text
    assert "INFO_RATE_INDUCED_LMEEEG_OUTPUTS" in targets_text


def test_info_rate_induced_lmeeg_config_contains_core_model_terms() -> None:
    config_text = Path("config/info_rate_induced_lmeeg.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "info_rate_induced_lmeeg"' in config_text
    assert "neural_bin_width_s: 0.050" in config_text
    assert "info_bin_width_s: 0.050" in config_text
    assert "min_causal_lag_s: 0.050" in config_text
    assert "max_causal_lag_s: 1.000" in config_text


def test_preprocess_workflow_tracks_preprocessing_config() -> None:
    preprocess_text = Path("workflow/rules/preprocessing.smk").read_text(encoding="utf-8")
    wrapper_text = Path("workflow/rules/preprocess.smk").read_text(encoding="utf-8")

    assert "rule preprocess_eeg:" in preprocess_text
    assert "config=PREPROCESSING_CONFIG_PATH" in preprocess_text
    assert "rule aggregate_preprocessing_qc:" in preprocess_text
    assert 'include: "preprocessing.smk"' in wrapper_text


def test_lmeeeg_workflow_tracks_evoked_duration_qc_outputs() -> None:
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "LMEEEG_EVOKED_DURATION_QC_OUTPUTS" in erp_text
    assert "model_formula.txt" in erp_text
    assert "duration_summary_by_class.csv" in erp_text
    assert "LMEEEG_EVOKED_DURATION_QC_OUTPUTS" in targets_text


def test_lmeeeg_config_contains_duration_controls_for_spp_evoked_model() -> None:
    config_text = Path("config/lmeeeg.yaml").read_text(encoding="utf-8")

    assert 'formula: "~ spp_class_1 + latency + run"' in config_text
    assert "duration_controls:" in config_text
    assert 'output_column: "z_log_spp_duration"' in config_text
    assert 'output_prefix: "spline_log_spp_duration"' in config_text
    assert 'output_column: "spp_duration_bin"' in config_text
    assert "term_tests:" in config_text
    assert "duration_spline:" in config_text


def test_induced_epochs_config_covers_lmeeeg_requested_bands() -> None:
    epochs_text = Path("config/epochs.yaml").read_text(encoding="utf-8")
    lmeeeg_text = Path("config/lmeeeg.yaml").read_text(encoding="utf-8")

    assert 'bands: ["theta", "alpha", "beta"]' in epochs_text
    assert 'bands: ["theta", "alpha", "beta"]' in lmeeeg_text


def test_lmeeeg_workflow_tracks_band_level_induced_epoch_outputs() -> None:
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "_induced_epoch_band_summary_outputs" in erp_text
    assert "epoching_summary-time_s.json" in erp_text
    assert "band_summaries=expand(" in targets_text
    assert "config=EPOCHS_CONFIG_PATH" in targets_text


def test_run_lmeeeg_is_evoked_only() -> None:
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")

    assert "rule run_lmeeeg:" in erp_text
    assert "evoked_models = {" in erp_text
    assert '!= "induced"' in erp_text
    run_lmeeeg_section = erp_text.split("rule run_lmeeeg:", 1)[1].split("rule run_induced_lmeeeg:", 1)[0]
    assert "induced_subjects" not in run_lmeeeg_section
    assert "induced_bands" not in run_lmeeeg_section


def test_fpp_spp_conf_disc_workflow_target_is_wired() -> None:
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")
    figures_text = Path("workflow/rules/figures.smk").read_text(encoding="utf-8")
    erp_text = Path("workflow/rules/erp.smk").read_text(encoding="utf-8")

    assert "rule downsample_fpp_spp_conf_disc_induced_epochs_subject:" in targets_text
    assert "rule run_fpp_spp_conf_disc_alpha_beta_lmeeeg:" in targets_text
    assert "rule lme_eeg_fpp_spp_conf_disc_alpha_beta:" in targets_text
    assert "FPP_SPP_CONF_DISC_ALPHA_BETA_LMEEEG_CONTRAST_OUTPUTS" in erp_text
    assert "rule figures_fpp_spp_conf_disc_alpha_beta_lmeeeg:" in figures_text
    assert "rule figures_fpp_spp_conf_disc_alpha_beta_lmeeeg_inference:" in figures_text
    assert "rule figures_lme_eeg_fpp_spp_conf_disc_alpha_beta:" in targets_text


def test_fpp_spp_conf_disc_config_contains_requested_models() -> None:
    config_text = Path("config/lmeeeg_fpp_spp_conf_disc_alpha_beta.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "fpp_spp_conf_disc_alpha_beta"' in config_text
    assert 'induced_epochs_subdir: "induced_epochs_fpp_spp_conf_disc_alpha_beta_lmeeeg"' in config_text
    assert 'formula: "power ~ class_3 + (1 | subject)"' in config_text
    assert 'formula: "power ~ class_3 + log_duration_within_class + (1 | subject)"' in config_text
    assert 'formula: "power ~ log_duration + (1 | subject)"' in config_text
    assert 'target_sfreq: 20' in config_text
