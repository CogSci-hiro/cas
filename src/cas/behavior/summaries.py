"""Output-path helpers for the behavioral hazard pipeline."""

from __future__ import annotations

from pathlib import Path

from cas.behavior.config import BehaviorHazardConfig


def ensure_behavior_directories(config: BehaviorHazardConfig) -> dict[str, Path]:
    paths = {
        "hazard_root": config.paths.hazard_root,
        "risksets": config.paths.hazard_root / "risksets",
        "predictors": config.paths.hazard_root / "predictors",
        "lag_selection": config.paths.hazard_root / "lag_selection",
        "lag_selection_models": config.paths.hazard_root / "lag_selection" / "models",
        "models": config.paths.hazard_root / "models",
        "models_fpp": config.paths.hazard_root / "models" / "fpp",
        "models_spp": config.paths.hazard_root / "models" / "spp_control",
        "models_pooled": config.paths.hazard_root / "models" / "pooled_anchor",
        "tables": config.paths.hazard_root / "tables",
        "diagnostics": config.paths.hazard_root / "diagnostics",
        "logs": config.paths.hazard_root / "logs",
        "figures_main": config.paths.figures_main_behavior,
        "figures_supp": config.paths.figures_supp_behavior,
        "figures_qc": config.paths.figures_qc_behavior,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
