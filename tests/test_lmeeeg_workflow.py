from __future__ import annotations

from pathlib import Path


def test_induced_public_target_family_is_wired() -> None:
    induced_text = Path("workflow/rules/induced.smk").read_text(encoding="utf-8")
    snakefile_text = Path("workflow/Snakefile").read_text(encoding="utf-8")

    assert 'include: "rules/induced.smk"' in snakefile_text
    for rule_name in (
        "induced_all",
        "induced_power",
        "induced_sensor_lmeeeg",
        "induced_source_lmeeeg",
        "induced_tables",
        "induced_figures",
        "induced_qc",
    ):
        assert f"rule {rule_name}:" in induced_text


def test_induced_public_target_family_collects_expected_outputs() -> None:
    induced_text = Path("workflow/rules/induced.smk").read_text(encoding="utf-8")

    assert "INDUCED_POWER_OUTPUTS" in induced_text
    assert "INDUCED_SENSOR_LMEEEG_OUTPUTS" in induced_text
    assert "INDUCED_SOURCE_LMEEEG_OUTPUTS" in induced_text
    assert "INDUCED_TABLE_OUTPUTS" in induced_text
    assert "INDUCED_FIGURE_OUTPUTS" in induced_text
    assert "INDUCED_QC_OUTPUTS" in induced_text
    assert "SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX" in induced_text
    assert "INFO_RATE_INDUCED_LMEEEG_OUTPUTS" in induced_text


def test_induced_sensor_workflow_uses_generic_internal_rules() -> None:
    epochs_text = Path("workflow/rules/epochs.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "rule build_induced_sensor_lmeeeg:" in epochs_text
    assert "rule build_induced_sensor_cycle_position:" in epochs_text
    assert "rule build_induced_sensor_conf_disc:" in targets_text
    assert "rule build_induced_binned_info_rate:" in epochs_text


def test_legacy_public_lmeeeg_target_names_are_removed() -> None:
    combined_text = "\n".join(
        [
            Path("workflow/rules/epochs.smk").read_text(encoding="utf-8"),
            Path("workflow/rules/figures.smk").read_text(encoding="utf-8"),
            Path("workflow/rules/targets.smk").read_text(encoding="utf-8"),
            Path("workflow/rules/induced.smk").read_text(encoding="utf-8"),
        ]
    )

    for legacy_rule_name in (
        "rule lmeeeg_all:",
        "rule induced_lmeeeg_all:",
        "rule lme_eeg_fpp_spp_cycle_position:",
        "rule info_rate_induced_lmeeg_all:",
        "rule figures_lmeeeg_all:",
        "rule figures_lme_eeg_fpp_spp_cycle_position:",
        "rule figures_lme_eeg_fpp_spp_conf_disc_alpha_beta:",
        "rule run_lmeeeg:",
        "rule run_induced_lmeeeg:",
        "rule run_fpp_spp_cycle_position_lmeeeg:",
        "rule run_info_rate_induced_lmeeg:",
        "rule figures_lmeeeg:",
        "rule figures_lmeeeg_inference:",
        "rule figures_fpp_spp_cycle_position_lmeeeg:",
        "rule figures_fpp_spp_cycle_position_lmeeeg_inference:",
        "rule figures_fpp_spp_conf_disc_alpha_beta_lmeeeg:",
        "rule figures_fpp_spp_conf_disc_alpha_beta_lmeeeg_inference:",
    ):
        assert legacy_rule_name not in combined_text


def test_induced_sensor_figures_are_centralized() -> None:
    figures_text = Path("workflow/rules/figures.smk").read_text(encoding="utf-8")

    assert "figures/main/induced/sensor_lmeeeg/figure_manifest.json" in figures_text
    assert "figures/supp/induced/sensor_lmeeeg/cycle_position/figure_manifest.json" in figures_text
    assert "figures/supp/induced/sensor_lmeeeg/conf_disc/figure_manifest.json" in figures_text
    assert "figures/qc/induced/sensor_lmeeeg/figure_manifest.json" in figures_text


def test_fpp_spp_cycle_position_config_contains_requested_contrast() -> None:
    config_text = Path("config/induced/alpha_beta_cycle_position.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "fpp_spp_cycle_position"' in config_text
    assert 'formula: "power ~ pair_position + z_event_duration + z_latency + run + z_time_within_run + (1 | subject)"' in config_text
    assert 'contrast_of_interest: "pair_positionFPP"' in config_text
    assert 'pair_position: "SPP"' in config_text


def test_info_rate_induced_lmeeg_config_contains_core_model_terms() -> None:
    config_text = Path("config/induced/info_rate_induced_lmeeg.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "info_rate_induced_lmeeg"' in config_text
    assert "neural_bin_width_s: 0.050" in config_text
    assert "info_bin_width_s: 0.050" in config_text
    assert "min_causal_lag_s: 0.050" in config_text
    assert "max_causal_lag_s: 1.000" in config_text


def test_preprocess_workflow_tracks_preprocessing_config() -> None:
    preprocess_text = Path("workflow/rules/preprocessing.smk").read_text(encoding="utf-8")

    assert "rule preprocess_eeg:" in preprocess_text
    assert "config=PREPROCESSING_CONFIG_PATH" in preprocess_text
    assert "rule aggregate_preprocessing_qc:" in preprocess_text


def test_lmeeeg_config_contains_duration_controls_for_spp_evoked_model() -> None:
    config_text = Path("config/induced/alpha_beta_lmeeeg.yaml").read_text(encoding="utf-8")

    assert 'formula: "~ spp_class_1 + latency + run"' in config_text
    assert "duration_controls:" in config_text
    assert 'output_column: "z_log_spp_duration"' in config_text
    assert 'output_prefix: "spline_log_spp_duration"' in config_text
    assert 'output_column: "spp_duration_bin"' in config_text
    assert "term_tests:" in config_text
    assert "duration_spline:" in config_text


def test_induced_epochs_config_covers_lmeeeg_requested_bands() -> None:
    epochs_text = Path("config/epochs/erp.yaml").read_text(encoding="utf-8")
    lmeeeg_text = Path("config/induced/alpha_beta_lmeeeg.yaml").read_text(encoding="utf-8")

    assert 'bands: ["theta", "alpha", "beta"]' in epochs_text
    assert 'bands: ["theta", "alpha", "beta"]' in lmeeeg_text


def test_lmeeeg_workflow_tracks_band_level_induced_epoch_outputs() -> None:
    epochs_text = Path("workflow/rules/epochs.smk").read_text(encoding="utf-8")
    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")

    assert "_induced_epoch_band_summary_outputs" in epochs_text
    assert "epoching_summary-time_s.json" in epochs_text
    assert "band_summaries=expand(" in targets_text
    assert "config=EPOCHS_CONFIG_PATH" in targets_text


def test_fpp_spp_conf_disc_config_contains_requested_models() -> None:
    config_text = Path("config/induced/alpha_beta_conf_disc.yaml").read_text(encoding="utf-8")

    assert 'analysis_name: "fpp_spp_conf_disc_alpha_beta"' in config_text
    assert 'induced_epochs_subdir: "induced_epochs_fpp_spp_conf_disc_alpha_beta_lmeeeg"' in config_text
    assert 'formula: "power ~ class_3 + (1 | subject)"' in config_text
    assert 'formula: "power ~ class_3 + log_duration_within_class + (1 | subject)"' in config_text
    assert 'formula: "power ~ log_duration + (1 | subject)"' in config_text
    assert 'target_sfreq: 20' in config_text
