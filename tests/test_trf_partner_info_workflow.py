from pathlib import Path

import yaml


def test_partner_info_trf_workflow_is_registered() -> None:
    snakefile_text = Path("workflow/Snakefile").read_text(encoding="utf-8")
    workflow_text = Path("workflow/rules/trf_partner_info.smk").read_text(encoding="utf-8")

    assert 'include: "rules/trf_partner_info.smk"' in snakefile_text
    for rule_name in (
        "fit_trf_partner_info_subject",
        "aggregate_trf_partner_info",
        "trf_partner_info_all",
    ):
        assert f"rule {rule_name}:" in workflow_text
    assert "PARTNER_INFO_TRF_TARGETS" in workflow_text
    assert "/figures/main/trf/" in workflow_text
    assert "/figures/qc/trf/" in workflow_text
    assert "/trf/" in workflow_text


def test_partner_info_trf_config_contains_core_analysis_terms() -> None:
    config_text = Path("config/trf/partner_info.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert 'name: "partner_info_trf"' in config_text
    assert 'description: "partner-turn information-tracking TRF"' in config_text
    assert '"information_rate"' in config_text
    assert '"prop_expected_cumulative_info"' in config_text
    assert "ipu_gap_threshold_s" not in config_text
    assert float(config["targets"]["modelling_sampling_rate_hz"]) < 64.0
    assert isinstance(config["targets"]["include"], list)
    assert len(config["targets"]["include"]) >= 1
    assert float(config["inputs"]["conversation_duration_s"]) == 240.0
