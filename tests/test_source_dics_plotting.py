from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cas.source_dics.broad_rois import map_aparc_to_broad_roi
from cas.source_dics.config import load_source_dics_config
from cas.source_dics.plotting import (
    TimeSummarySpec,
    _compute_roi_summary,
    apply_significance_rule,
    run_source_dics_plotting,
    summarize_time_window,
)


def _write_plotting_config(tmp_path: Path) -> Path:
    config_payload = {
        "paths": {
            "events_csv": str((tmp_path / "events.csv").relative_to(tmp_path)),
            "epochs_dir": str((tmp_path / "epochs").relative_to(tmp_path)),
            "source_dir": str((tmp_path / "source").relative_to(tmp_path)),
            "derivatives_dir": str((tmp_path / "results").relative_to(tmp_path)),
            "lmeeeg_dir": str((tmp_path / "stats").relative_to(tmp_path)),
        },
        "events": {
            "anchor_types": ["FPP", "SPP"],
            "subject_column": "subject",
            "dyad_column": "dyad",
            "run_column": "run",
            "anchor_type_column": "anchor_type",
            "onset_column": "onset",
            "duration_column": "duration",
            "latency_column": "latency",
            "label_column": "label",
        },
        "epoching": {
            "tmin": -2.0,
            "tmax": 0.5,
            "baseline": None,
            "reject_by_annotation": True,
        },
        "dics": {
            "method": "dics",
            "common_filter": True,
            "filter_tmin": -1.5,
            "filter_tmax": -0.1,
            "analysis_tmin": -1.5,
            "analysis_tmax": -0.1,
            "bands": {
                "alpha": {"fmin": 8.0, "fmax": 12.0},
                "beta": {"fmin": 13.0, "fmax": 30.0},
            },
            "csd_method": "multitaper",
            "mt_bandwidth": None,
            "regularization": 0.05,
            "pick_ori": "max-power",
            "weight_norm": "unit-noise-gain",
            "reduce_rank": True,
            "real_filter": True,
            "n_jobs": 1,
        },
        "source_space": {
            "kind": "surface",
            "mode": "fsaverage_surface",
            "spacing": "oct5",
            "subjects_dir": str((tmp_path / "subjects").relative_to(tmp_path)),
            "subject": "fsaverage",
            "trans": "fsaverage",
            "bem": None,
            "forward_template": str((tmp_path / "forward-template.fif").relative_to(tmp_path)),
            "parcellation": "aparc",
            "aggregate_to_labels": False,
            "aggregation": "mean",
        },
        "output": {
            "save_filters": True,
            "save_trial_power": True,
            "save_long_table": True,
            "save_qc": True,
            "overwrite": True,
            "long_table_chunk_rows": 200,
        },
        "lmeeeg": {
            "enabled": True,
            "formula": "power ~ anchor_type + duration + (1 | subject)",
            "dependent_variable": "power",
            "predictors": ["anchor_type", "duration"],
            "test_predictors": ["anchor_type"],
        },
        "logging": {"verbose": True, "progress": False},
        "plots": {
            "enabled": True,
            "output_dir": str((tmp_path / "results" / "figures").relative_to(tmp_path)),
            "statistics_dir": str((tmp_path / "stats").relative_to(tmp_path)),
            "source_space": "fsaverage",
            "surface": "inflated",
            "views": ["lateral"],
            "hemispheres": ["lh", "rh"],
            "bands": ["alpha"],
            "predictors": ["anchor_type"],
            "significance": {
                "alpha": 0.05,
                "use_cluster_mask_if_available": True,
                "fallback_p_column": "p_value",
                "fallback_corrected_p_column": "p_value_corrected",
                "mask_non_significant": True,
            },
            "time_summary": {
                "mode": "cluster_peak_and_mean",
                "windows": {
                    "pre_event_full": {"tmin": -1.5, "tmax": -0.1},
                },
            },
            "surface_plot": {
                "cmap": "RdBu_r",
                "symmetric_clim": True,
                "background": "white",
                "transparent_background": False,
                "dpi": 100,
                "show_colorbar": True,
                "non_significant_style": "transparent",
            },
            "roi_plot": {
                "enabled": True,
                "atlas": "aparc",
                "broad_roi_scheme": "source_dics_broad",
                "aggregation": "mean",
                "significance_weighted": False,
                "include_n_vertices": True,
                "sort_by": ["hemisphere", "broad_roi"],
                "dpi": 100,
            },
            "report": {
                "write_markdown_index": True,
                "write_csv_index": True,
            },
        },
    }

    (tmp_path / "epochs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stats").mkdir(parents=True, exist_ok=True)
    (tmp_path / "subjects").mkdir(parents=True, exist_ok=True)
    (tmp_path / "events.csv").write_text("pair_id\npair-001\n", encoding="utf-8")
    (tmp_path / "forward-template.fif").write_text("placeholder", encoding="utf-8")

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config" / "source_dics_plotting.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    return config_path


def test_significance_mask_priority_cluster_over_corrected_and_raw() -> None:
    frame = pd.DataFrame(
        {
            "p_value": [0.9, 0.9],
            "p_value_corrected": [0.9, 0.01],
            "cluster_significant": [True, False],
            "cluster_p_value": [0.9, 0.9],
        }
    )

    out, rule = apply_significance_rule(frame, alpha=0.05, use_cluster_mask_if_available=True)

    assert rule == "cluster"
    assert out["significant"].tolist() == [True, False]


def test_time_window_mean_summary_uses_mean_and_any_significance() -> None:
    table = pd.DataFrame(
        {
            "band": ["alpha", "alpha", "alpha"],
            "predictor": ["duration", "duration", "duration"],
            "hemi": ["lh", "lh", "lh"],
            "source_id": ["10", "10", "10"],
            "time": [-1.0, -0.7, -0.3],
            "t_value": [1.0, 3.0, -2.0],
            "significant": [False, True, False],
        }
    )

    out = summarize_time_window(
        table,
        spec=TimeSummarySpec(name="pre_event_full_mean", tmin=-1.5, tmax=-0.1, mode="mean"),
    )

    assert out.shape[0] == 1
    assert np.isclose(float(out.loc[0, "t_value"]), (1.0 + 3.0 - 2.0) / 3.0)
    assert bool(out.loc[0, "significant"]) is True


def test_peak_abs_summary_preserves_signed_t_value() -> None:
    table = pd.DataFrame(
        {
            "band": ["alpha", "alpha"],
            "predictor": ["duration", "duration"],
            "hemi": ["lh", "lh"],
            "source_id": ["10", "10"],
            "time": [-1.0, -0.5],
            "t_value": [1.9, -2.5],
            "significant": [True, True],
        }
    )

    out = summarize_time_window(
        table,
        spec=TimeSummarySpec(name="pre_event_full_peak_abs", tmin=-1.5, tmax=-0.1, mode="peak_abs"),
    )

    assert out.shape[0] == 1
    assert float(out.loc[0, "t_value"]) == -2.5


def test_broad_roi_mapping_accepts_common_label_variants() -> None:
    labels = [
        "lh.superiorfrontal",
        "superiorfrontal-lh",
        "ctx-lh-superiorfrontal",
        "superiorfrontal",
    ]

    mapped = [map_aparc_to_broad_roi(label) for label in labels]

    assert mapped == ["Frontal", "Frontal", "Frontal", "Frontal"]


def test_roi_summary_output_schema_has_required_columns() -> None:
    summary = pd.DataFrame(
        {
            "band": ["alpha", "alpha"],
            "predictor": ["anchor_type", "anchor_type"],
            "time_summary": ["pre_event_full_mean", "pre_event_full_mean"],
            "hemi": ["lh", "lh"],
            "source_id": ["1", "2"],
            "t_value": [1.0, -2.0],
            "significant": [True, False],
            "label": ["lh.precentral", "lh.postcentral"],
        }
    )

    out = _compute_roi_summary(summary)

    required = {
        "hemisphere",
        "broad_roi",
        "mean_t_value",
        "mean_abs_t_value",
        "max_abs_t_value",
        "n_sources",
        "n_significant_sources",
        "proportion_significant_sources",
        "mean_t_value_significant_only",
        "mean_abs_t_value_significant_only",
    }
    assert required.issubset(set(out.columns))


def test_plot_source_dics_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cas.cli.main", "plot-source-dics-fpp-spp", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "plot-source-dics-fpp-spp" in result.stdout
    assert "--time-summaries" in result.stdout


def test_missing_surface_backend_does_not_break_roi_plotting(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_plotting_config(tmp_path)
    stats_table = pd.DataFrame(
        {
            "band": ["alpha", "alpha", "alpha", "alpha"],
            "predictor": ["anchor_type", "anchor_type", "anchor_type", "anchor_type"],
            "source_id": ["lh:1", "lh:2", "rh:3", "rh:4"],
            "hemi": ["lh", "lh", "rh", "rh"],
            "time": [-1.0, -0.8, -1.0, -0.8],
            "t_value": [2.0, -1.0, 1.5, -0.7],
            "p_value": [0.01, 0.9, 0.02, 0.9],
            "label": ["lh.precentral", "lh.postcentral", "rh.precentral", "rh.postcentral"],
        }
    )
    stats_table.to_csv(tmp_path / "stats" / "synthetic_source_stats.csv", index=False)

    config = load_source_dics_config(config_path)

    monkeypatch.setattr(
        "cas.source_dics.plotting._render_surface_images",
        lambda *args, **kwargs: ("vertex-level", "pyvista backend unavailable"),
    )

    index_path = run_source_dics_plotting(config, overwrite=True)

    assert index_path.exists()
    note_path = config.plots.output_dir / "index" / "missing_surface_backend.txt"
    assert note_path.exists()

    roi_csvs = sorted((config.plots.output_dir / "roi" / "alpha").glob("*_broad_roi_tvalues.csv"))
    assert roi_csvs


def test_source_dics_plotting_targets_exist() -> None:
    source_dics_rules = (
        Path(__file__).resolve().parents[1] / "workflow" / "rules" / "source_localisation.smk"
    ).read_text(encoding="utf-8")
    targets_rules = (
        Path(__file__).resolve().parents[1] / "workflow" / "rules" / "targets.smk"
    ).read_text(encoding="utf-8")

    assert "rule plot_source_dics_fpp_spp_alpha_beta:" in source_dics_rules
    assert "rule source_dics_fpp_spp_alpha_beta_figures:" in source_dics_rules
    assert "SOURCE_DICS_FPP_SPP_ALPHA_BETA_FIGURES_INDEX" in targets_rules
