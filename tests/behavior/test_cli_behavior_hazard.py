from __future__ import annotations

from pathlib import Path
import sys

import pytest

from cas.behavior.config import load_behavior_hazard_config
from cas.cli.main import _build_parser
from cas.cli.main import main
from cas.cli.commands.behavior_hazard import BEHAVIOR_HAZARD_STAGES


def test_behavior_hazard_nested_cli_dispatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "hazard.yaml"
    config_path.write_text(
        "paths_config: config/paths.yaml\ninputs:\n  events_csv: x\n  surprisal_tsv: y\nbehavior:\n  hazard:\n    bin_size_ms: 50\n",
        encoding="utf-8",
    )
    called: dict[str, object] = {}

    def _fake_run(stage: str, *, config_path: str, verbose: bool = False) -> dict[str, Path]:
        called["stage"] = stage
        called["config_path"] = config_path
        called["verbose"] = verbose
        return {}

    monkeypatch.setattr("cas.cli.commands.behavior_hazard.run_behavior_hazard_stage", _fake_run)
    monkeypatch.setattr(sys, "argv", ["cas", "behavior", "hazard", "all", "--config", str(config_path), "--verbose"])
    assert main() == 0
    assert called["stage"] == "all"
    assert called["config_path"] == str(config_path)
    assert called["verbose"] is True


def test_behavior_hazard_cli_stage_list_includes_tables_and_qc() -> None:
    assert {"build-risksets", "add-predictors", "select-lag", "fit-models", "tables", "figures", "qc", "all"} <= set(BEHAVIOR_HAZARD_STAGES)


def test_behavior_hazard_config_resolves_centralized_figure_paths() -> None:
    cfg = load_behavior_hazard_config("config/behavior/hazard.yaml")
    assert str(cfg.paths.figures_main_behavior).endswith("/figures/main/behavior")
    assert str(cfg.paths.figures_supp_behavior).endswith("/figures/supp/behavior")
    assert str(cfg.paths.figures_qc_behavior).endswith("/figures/qc/behavior")
    assert "/behavior/hazard/figures" not in str(cfg.paths.figures_main_behavior)


@pytest.mark.parametrize(
    "legacy_command",
    [
        "behavior-final-fit",
        "hazard-behavior-fpp",
        "export-behaviour-glmm-data",
        "plot-behaviour-hazard-results",
        "neural-hazard-fpp-spp",
        "fit-tde-hmm",
    ],
)
def test_legacy_behavior_commands_are_not_registered(legacy_command: str) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([legacy_command])
