from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cas.behavior.config import BehaviorHazardPaths, BehaviorHazardConfig
from cas.behavior.predictors import standardize_predictors


def _config(tmp_path: Path) -> BehaviorHazardConfig:
    output_dir = tmp_path / "derivatives"
    behavior_root = output_dir / "behavior"
    return BehaviorHazardConfig(
        path=tmp_path / "hazard.yaml",
        raw={
            "inputs": {"events_csv": "dummy.csv"},
            "behavior": {
                "hazard": {
                    "bin_size_ms": 50,
                    "candidate_lags_ms": [0, 50],
                    "standardization": {
                        "scope": "within_anchor",
                        "continuous_predictors": [
                            "information_rate",
                            "prop_expected_cum_info",
                            "time_from_partner_onset_s",
                            "time_from_partner_offset_s",
                        ],
                    },
                }
            },
        },
        paths_config_path=tmp_path / "paths.yaml",
        paths_config={"output_dir": str(output_dir)},
        paths=BehaviorHazardPaths(
            output_dir=output_dir,
            behavior_root=behavior_root,
            hazard_root=behavior_root / "hazard",
            figures_main_behavior=output_dir / "figures" / "main" / "behavior",
            figures_supp_behavior=output_dir / "figures" / "supp" / "behavior",
            figures_qc_behavior=output_dir / "figures" / "qc" / "behavior",
        ),
    )


def test_standardize_predictors_adds_onset_squared_term(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base = pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2", "e2"],
            "anchor_type": ["FPP", "FPP", "SPP", "SPP"],
            "bin_start_s": [0.0, 0.05, 0.0, 0.05],
            "event": [0, 1, 0, 1],
            "information_rate": [1.0, 2.0, 1.5, 2.5],
            "prop_expected_cum_info": [0.1, 0.2, 0.3, 0.4],
            "time_from_partner_onset_s": [-1.0, 1.0, -2.0, 2.0],
            "time_from_partner_offset_s": [-0.5, 0.5, -1.0, 1.0],
        }
    )
    fpp = base.loc[base["anchor_type"] == "FPP"].copy()
    spp = base.loc[base["anchor_type"] == "SPP"].copy()

    fpp_out, spp_out, pooled_out, _ = standardize_predictors(fpp, spp, base, config=config, verbose=False)

    for table in (fpp_out, spp_out, pooled_out):
        assert "z_time_from_partner_onset_s_squared" in table.columns
        assert np.allclose(
            pd.to_numeric(table["z_time_from_partner_onset_s_squared"], errors="coerce"),
            pd.to_numeric(table["z_time_from_partner_onset_s"], errors="coerce") ** 2,
        )
