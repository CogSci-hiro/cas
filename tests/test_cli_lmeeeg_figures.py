from __future__ import annotations

from pathlib import Path

import numpy as np

from cas.cli import main as cli_main


def test_load_significance_masks_falls_back_to_analysis_summary(tmp_path: Path) -> None:
    model_payloads = {
        "demo_model": {
            "column_names": ["Intercept", "spp_class_1[T.DISC]"],
            "betas": np.zeros((2, 3, 4), dtype=float),
        }
    }
    corrected_p_values = np.asarray(
        [
            [0.20, 0.01, 0.40, 0.30],
            [0.80, 0.02, 0.60, 0.70],
            [0.90, 0.03, 0.80, 0.90],
        ],
        dtype=float,
    )
    corrected_p_path = tmp_path / "demo_corrected_p.npy"
    np.save(corrected_p_path, corrected_p_values)
    analysis_summary = {
        "models": [
            {
                "model_name": "demo_model",
                "inference": [
                    {
                        "effect": "spp_class_1[T.DISC]",
                        "corrected_p_values": str(corrected_p_path),
                    }
                ],
            }
        ]
    }

    masks = cli_main._load_significance_masks(
        stats_root=corrected_p_path.parent / "missing_stats_root",
        model_payloads=model_payloads,
        analysis_summary=analysis_summary,
    )

    assert "demo_model" in masks
    assert "spp_class_1[T.DISC]" in masks["demo_model"]
    assert np.array_equal(
        masks["demo_model"]["spp_class_1[T.DISC]"],
        corrected_p_values < 0.05,
    )


def test_load_significance_masks_keeps_induced_bands_separate(tmp_path: Path) -> None:
    model_payloads = {
        "induced_model__theta": {
            "column_names": ["spp_class_1[T.DISC]"],
            "betas": np.zeros((1, 2, 3), dtype=float),
        },
        "induced_model__alpha": {
            "column_names": ["spp_class_1[T.DISC]"],
            "betas": np.zeros((1, 2, 3), dtype=float),
        },
    }
    theta_path = tmp_path / "theta_corrected_p.npy"
    alpha_path = tmp_path / "alpha_corrected_p.npy"
    np.save(theta_path, np.asarray([[0.01, 0.20, 0.30], [0.40, 0.02, 0.50]], dtype=float))
    np.save(alpha_path, np.asarray([[0.60, 0.70, 0.03], [0.80, 0.90, 0.04]], dtype=float))
    analysis_summary = {
        "models": [
            {
                "model_name": "induced_model",
                "band_name": "theta",
                "inference": [
                    {
                        "effect": "spp_class_1[T.DISC]",
                        "corrected_p_values": str(theta_path),
                    }
                ],
            },
            {
                "model_name": "induced_model",
                "band_name": "alpha",
                "inference": [
                    {
                        "effect": "spp_class_1[T.DISC]",
                        "corrected_p_values": str(alpha_path),
                    }
                ],
            },
        ]
    }

    masks = cli_main._load_significance_masks(
        stats_root=tmp_path / "missing_stats_root",
        model_payloads=model_payloads,
        analysis_summary=analysis_summary,
    )

    assert np.array_equal(
        masks["induced_model__theta"]["spp_class_1[T.DISC]"],
        np.asarray([[True, False, False], [False, True, False]], dtype=bool),
    )
    assert np.array_equal(
        masks["induced_model__alpha"]["spp_class_1[T.DISC]"],
        np.asarray([[False, False, True], [False, False, True]], dtype=bool),
    )
