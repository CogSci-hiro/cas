from pathlib import Path

import yaml


def test_spp_trf_workflow_is_registered() -> None:
    workflow_text = Path("workflow/rules/trf.smk").read_text(encoding="utf-8")
    snakefile_text = Path("workflow/Snakefile").read_text(encoding="utf-8")

    for snippet in (
        'SPP_TRF_CONFIG_PATH = f"{CONFIG_DIR}/trf/spp_trf.yaml"',
        "rule fit_spp_trf_subject:",
        "rule fit_spp_trf:",
        "rule aggregate_spp_trf:",
        "rule spp_trf_all:",
    ):
        assert snippet in workflow_text

    assert "def _raw_eeg_exists(subject_id: str, run: str) -> bool:" in snakefile_text
    assert "def _trf_run_has_preprocessable_eeg(subject_id: str, run: str) -> bool:" in snakefile_text
    assert "and _subject_has_complete_trf_preprocessed_eeg(subject)" in snakefile_text
    assert "def eeg_preprocessed_input(wildcards):" in workflow_text
    assert "def eeg_raw_input(wildcards):" in workflow_text
    assert "def eeg_envelope_summary_input(wildcards):" in workflow_text
    assert "PREPROCESSED_EEG_OUTPUT_PATTERN.format(" in workflow_text
    assert '--anchor-raw "{input.raw}"' in workflow_text
    assert '--conversation-summary-json "{input.envelope_summary}"' in workflow_text
    assert "--low-cut-hz {params.low_cut_hz}" in workflow_text
    assert "--high-cut-hz {params.high_cut_hz}" in workflow_text

    targets_text = Path("workflow/rules/targets.smk").read_text(encoding="utf-8")
    assert "rule spp_trf:" in targets_text
    assert "rules.spp_trf_all.input" in targets_text


def test_spp_trf_config_is_registered_and_parallel_to_trf() -> None:
    config_registry = Path("config/config.yaml").read_text(encoding="utf-8")
    assert 'spp_trf: "config/trf/spp_trf.yaml"' in config_registry

    spp_trf_config_path = Path("config/trf/spp_trf.yaml")
    spp_trf_config = yaml.safe_load(spp_trf_config_path.read_text(encoding="utf-8"))

    assert spp_trf_config["trf"]["analysis_id"] == "spp_trf"
    assert spp_trf_config["trf"]["target"]["path"] == "evoked/{subject}/run-{run}.npy"
    assert spp_trf_config["trf"]["timing"]["target_sfreq_hz"] == 32
    assert set(spp_trf_config["trf"]["models"]) == {"M0", "M1"}
    assert spp_trf_config["trf"]["models"]["M1"]["predictors"][-1] == "spp_conf_vs_disconf"
    assert spp_trf_config["trf"]["predictor_definitions"]["spp_conf_vs_disconf"]["label_prefix_weights"] == {
        "SPP_CONF": -1.0,
        "SPP_DISC": 1.0,
    }
