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
