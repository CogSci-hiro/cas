from __future__ import annotations

import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from cas.viz import lmeeeg as lmeeeg_viz


def test_build_lmeeeg_qc_manifest_from_model_payloads_uses_speech_rate_layout(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    captured_calls: list[dict[str, object]] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, significance_mask=None, **kwargs) -> list[Path]:
        captured_calls.append(
            {
                "output_stem": output_stem,
                "significance_mask": None if significance_mask is None else np.asarray(significance_mask, dtype=bool).copy(),
            }
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest_from_model_payloads(
        out_dir=out_dir,
        model_payloads={
            "demo_model": {
                "model_name": "demo_model",
                "band_name": None,
                "model_label": "demo_model",
                "betas": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=float),
                "t_values": np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
                "times": np.asarray([-0.1, 0.0], dtype=float),
                "channel_names": ["Fz", "Cz"],
                "column_names": ["latency"],
            }
        },
        manifest_path=out_dir / "figures" / "lmeeeg" / "figure_manifest.json",
        significance_masks={"demo_model": {"latency": np.asarray([[True, False], [False, True]], dtype=bool)}},
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 2
    assert [plot["measure"] for plot in manifest["plots"]] == ["beta", "t_value"]
    assert all(plot["has_significance_overlay"] is True for plot in manifest["plots"])
    assert [call["output_stem"] for call in captured_calls] == [
        out_dir / "figures" / "lmeeeg" / "betas" / "demo_model" / "latency",
        out_dir / "figures" / "lmeeeg" / "t_values" / "demo_model" / "latency",
    ]
    assert all(np.array_equal(call["significance_mask"], np.asarray([[True, False], [False, True]], dtype=bool)) for call in captured_calls)


def test_build_lmeeeg_qc_manifest_from_model_payloads_keeps_band_metadata(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"

    def fake_plot_joint_model_weights(*args, output_stem: Path, significance_mask=None, **kwargs) -> list[Path]:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest_from_model_payloads(
        out_dir=out_dir,
        model_payloads={
            "induced_model__theta": {
                "model_name": "induced_model",
                "band_name": "theta",
                "model_label": "induced_model [theta]",
                "betas": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=float),
                "t_values": np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
                "times": np.asarray([-0.1, 0.0], dtype=float),
                "channel_names": ["Fz", "Cz"],
                "column_names": ["latency"],
            }
        },
        manifest_path=out_dir / "figures" / "lmeeeg" / "figure_manifest.json",
        formats=("png",),
        dpi=72,
    )

    assert manifest["plots"][0]["model_key"] == "induced_model__theta"
    assert manifest["plots"][0]["model_name"] == "induced_model"
    assert manifest["plots"][0]["band_name"] == "theta"
    assert manifest["plots"][0]["files"][0].endswith("/figures/lmeeeg/betas/induced_model__theta/latency.png")


def test_build_lmeeeg_qc_manifest_from_model_payloads_supports_custom_figure_subdir(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    captured_stems: list[Path] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, significance_mask=None, **kwargs) -> list[Path]:
        captured_stems.append(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest_from_model_payloads(
        out_dir=out_dir,
        model_payloads={
            "fpp_vs_spp_cycle_position__theta": {
                "model_name": "fpp_vs_spp_cycle_position",
                "band_name": "theta",
                "model_label": "fpp_vs_spp_cycle_position [theta]",
                "betas": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=float),
                "t_values": np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=float),
                "times": np.asarray([-0.1, 0.0], dtype=float),
                "channel_names": ["Fz", "Cz"],
                "column_names": ["pair_positionFPP"],
            }
        },
        manifest_path=out_dir / "figures" / "fpp_spp_cycle_position" / "figure_manifest.json",
        figure_subdir="fpp_spp_cycle_position",
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 2
    assert captured_stems[0] == (
        out_dir / "figures" / "fpp_spp_cycle_position" / "betas" / "fpp_vs_spp_cycle_position__theta" / "pair_positionFPP"
    )


def test_build_lmeeeg_qc_manifest_keeps_distinct_categorical_terms_in_output_stems(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    captured_stems: list[Path] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, significance_mask=None, **kwargs) -> list[Path]:
        captured_stems.append(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest_from_model_payloads(
        out_dir=out_dir,
        model_payloads={
            "class_3__alpha": {
                "model_name": "class_3",
                "band_name": "alpha",
                "model_label": "class_3 [alpha]",
                "betas": np.asarray(
                    [
                        [[0.1, 0.2], [0.3, 0.4]],
                        [[0.5, 0.6], [0.7, 0.8]],
                    ],
                    dtype=float,
                ),
                "t_values": np.asarray(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ],
                    dtype=float,
                ),
                "times": np.asarray([-0.1, 0.0], dtype=float),
                "channel_names": ["Fz", "Cz"],
                "column_names": ["class_3[T.SPP_CONF]", "class_3[T.SPP_DISC]"],
            }
        },
        manifest_path=out_dir / "figures" / "lmeeeg" / "figure_manifest.json",
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 4
    assert captured_stems == [
        out_dir / "figures" / "lmeeeg" / "betas" / "class_3__alpha" / "class_3_T_SPP_CONF",
        out_dir / "figures" / "lmeeeg" / "betas" / "class_3__alpha" / "class_3_T_SPP_DISC",
        out_dir / "figures" / "lmeeeg" / "t_values" / "class_3__alpha" / "class_3_T_SPP_CONF",
        out_dir / "figures" / "lmeeeg" / "t_values" / "class_3__alpha" / "class_3_T_SPP_DISC",
    ]
    assert manifest["plots"][0]["kernel"] == "class_3[T.SPP_CONF]"
    assert manifest["plots"][1]["kernel"] == "class_3[T.SPP_DISC]"


def test_build_lmeeeg_qc_manifest_from_stats_writes_statistics_plot_with_overlay(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    stats_effect_dir = out_dir / "stats" / "lmeeeg" / "demo_model" / "latency"
    stats_effect_dir.mkdir(parents=True, exist_ok=True)
    observed_path = stats_effect_dir / "observed_statistic.npy"
    corrected_p_path = stats_effect_dir / "corrected_p_values.npy"
    np.save(observed_path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    np.save(corrected_p_path, np.asarray([[0.04, 0.2], [0.03, 0.5]], dtype=float))
    (stats_effect_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_name": "demo_model",
                "effect": "latency",
                "observed_statistic": str(observed_path),
                "corrected_p_values": str(corrected_p_path),
                "min_corrected_p": 0.03,
                "n_significant_p_lt_0_05": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured_calls: list[dict[str, object]] = []

    def fake_plot_joint_model_weights(*args, output_stem: Path, significance_mask=None, **kwargs) -> list[Path]:
        captured_calls.append(
            {
                "output_stem": output_stem,
                "significance_mask": None if significance_mask is None else np.asarray(significance_mask, dtype=bool).copy(),
            }
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_path = output_stem.with_suffix(".png")
        written_path.write_text("stub\n", encoding="utf-8")
        return [written_path]

    monkeypatch.setattr(lmeeeg_viz, "plot_joint_model_weights", fake_plot_joint_model_weights)

    manifest = lmeeeg_viz.build_lmeeeg_qc_manifest_from_stats(
        out_dir=out_dir,
        stats_root=out_dir / "stats" / "lmeeeg",
        manifest_path=out_dir / "figures" / "lmeeeg" / "figure_manifest.json",
        model_axes={"demo_model": {"channel_names": ["Fz", "Cz"], "times": np.asarray([-0.1, 0.0], dtype=float)}},
        formats=("png",),
        dpi=72,
    )

    assert manifest["plot_count"] == 1
    assert manifest["plots"][0]["source"] == "stats_fallback"
    assert manifest["plots"][0]["has_significant_samples"] is True
    assert captured_calls[0]["output_stem"] == out_dir / "figures" / "lmeeeg" / "statistics" / "demo_model" / "latency"
    assert np.array_equal(
        captured_calls[0]["significance_mask"],
        np.asarray([[True, False], [True, False]], dtype=bool),
    )


def test_find_joint_timeseries_axis_prefers_axis_covering_time_window() -> None:
    figure = Figure()
    topomap_axis = figure.add_subplot(121)
    topomap_axis.plot([-0.1, 0.1], [0.0, 1.0])
    topomap_axis.set_xlim(-0.12, 0.12)

    butterfly_axis = figure.add_subplot(122)
    butterfly_axis.plot([-0.1, 0.3], [0.0, 1.0])
    butterfly_axis.plot([-0.1, 0.3], [1.0, 0.0])
    butterfly_axis.set_xlim(-0.1, 0.3)

    selected = lmeeeg_viz._find_joint_timeseries_axis(figure, time_array=np.asarray([-0.1, 0.0, 0.3], dtype=float))

    assert selected is butterfly_axis


def test_plot_joint_model_weights_writes_pdf_with_significance_overlay(tmp_path) -> None:
    output_stem = tmp_path / "joint"
    data = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [0.0, 0.1, 0.4, 0.1, 0.0],
            [0.2, 0.3, 0.5, 0.3, 0.2],
            [0.1, 0.2, 0.4, 0.2, 0.1],
        ],
        dtype=float,
    )
    significance_mask = np.asarray(
        [
            [False, True, True, False, False],
            [False, True, True, False, False],
            [False, True, True, False, False],
            [False, True, True, False, False],
        ],
        dtype=bool,
    )

    written = lmeeeg_viz.plot_joint_model_weights(
        data,
        times=np.linspace(-0.1, 0.3, 5),
        channel_names=["Fz", "Cz", "Pz", "Oz"],
        output_stem=output_stem,
        title="demo",
        formats=("pdf",),
        dpi=72,
        significance_mask=significance_mask,
    )

    assert written == [output_stem.with_suffix(".pdf")]
    assert written[0].exists()
