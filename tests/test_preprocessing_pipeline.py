from __future__ import annotations

from pathlib import Path

import numpy as np
import mne
import pytest

from cas.annotations.io import write_textgrid
from cas.annotations.models import Interval, TextGrid, Tier
import cas.preprocessing.pipeline as preprocessing_pipeline
from cas.preprocessing.pipeline import preprocess_raw, preprocess_run
from cas.cli.main import _build_parser


def test_preprocess_raw_interpolates_bads_without_digitization() -> None:
    ch_names = ["Fp1", "Fp2", "C3", "C4", "Pz"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")

    rng = np.random.default_rng(7)
    data = rng.standard_normal((len(ch_names), 500))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020")

    with raw.info._unlock():
        raw.info["dig"] = None

    raw.info["bads"] = ["C3"]

    processed_raw = preprocess_raw(
        raw,
        interpolate_bad_channels=True,
        apply_rereference=False,
    )

    assert processed_raw.info["bads"] == []
    assert processed_raw.info["dig"] is not None
    assert np.isfinite(processed_raw.get_data()).all()


def test_preprocess_run_falls_back_to_mne_events_and_splits_channels(tmp_path: Path) -> None:
    ch_names = ["Fp1", "Fp2", "C3", "C4", "EMG1", "STI 014", "MISC"]
    ch_types = ["eeg", "eeg", "eeg", "eeg", "emg", "stim", "misc"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=ch_types)

    rng = np.random.default_rng(11)
    data = rng.standard_normal((len(ch_names), 400))
    data[ch_names.index("STI 014"), :] = 0.0
    data[ch_names.index("STI 014"), 50:60] = 1.0
    data[ch_names.index("STI 014"), 200:210] = 2.0

    raw = mne.io.RawArray(data, info, verbose="ERROR")

    channels_tsv_path = tmp_path / "channels.tsv"
    channels_tsv_path.write_text("name\tstatus\nC3\tbad\n", encoding="utf-8")

    result = preprocess_run(
        raw,
        channels_tsv_path=channels_tsv_path,
        target_sampling_rate_hz=200.0,
        low_cut_hz=0.1,
        high_cut_hz=40.0,
        emg_channel_patterns=[r"^EMG1$"],
        interpolate_bad_channels=True,
        apply_rereference=False,
    )

    assert result.summary["event_source"] == "mne"
    assert result.summary["event_count"] == 2
    assert result.summary["sampling_rate_before_hz"] == 250.0
    assert result.summary["sampling_rate_after_hz"] == 200.0
    assert result.summary["kept_eeg_channels"] == ["Fp1", "Fp2", "C3", "C4"]
    assert result.summary["kept_emg_channels"] == ["EMG1"]
    assert result.summary["dropped_channels"] == ["STI 014", "MISC"]
    assert result.emg_channel_names == ["EMG1"]
    assert result.emg_data.shape == (1, 400)
    assert result.events_rows[0]["source"] == "mne"
    assert {"event_index", "source", "sample", "onset_s", "event_id", "label", "pair_id"} <= set(
        result.events_rows[0]
    )
    assert result.eeg_raw.info["bads"] == []


def test_preprocess_run_uses_annotation_events_before_mne_fallback(tmp_path: Path) -> None:
    raw = _make_eeg_emg_raw()

    annotation_path = _write_event_grid(
        tmp_path / "dyad-001_run-2.TextGrid",
        action_a_intervals=[Interval(0.10, 0.30, "FPP_RFC_DECL")],
        action_b_intervals=[Interval(0.35, 0.40, "SPP_CONF_EXP")],
    )

    result = preprocess_run(
        raw,
        annotation_path=annotation_path,
        emg_channel_patterns=[r"^EMG1$"],
        apply_rereference=False,
        interpolate_bad_channels=False,
    )

    assert result.summary["event_source"] == "annotations"
    assert result.summary["event_count"] == 1
    assert result.events_rows[0]["source"] == "annotations"
    assert result.events_rows[0]["pair_id"]


def test_preprocess_run_delegates_processing_steps_to_local_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_eeg_emg_raw()
    calls: list[tuple[object, ...]] = []

    def fake_downsample(raw_arg, config) -> None:
        calls.append(("downsample", config["preprocessing"]["downsample_hz"]))

    def fake_filter(raw_arg, config) -> None:
        filter_cfg = dict(config["preprocessing"]["filter"])
        calls.append(("filter", filter_cfg["l_freq_hz"], filter_cfg["h_freq_hz"]))

    def fake_rereference(raw_arg) -> None:
        calls.append(("rereference", tuple(raw_arg.ch_names)))

    monkeypatch.setattr(preprocessing_pipeline, "downsample_raw", fake_downsample)
    monkeypatch.setattr(preprocessing_pipeline, "apply_bandpass_filter", fake_filter)
    monkeypatch.setattr(preprocessing_pipeline, "apply_average_reference", fake_rereference)

    result = preprocess_run(
        raw,
        target_sampling_rate_hz=200.0,
        low_cut_hz=0.1,
        high_cut_hz=40.0,
        emg_channel_patterns=[r"^EMG1$"],
        interpolate_bad_channels=False,
    )

    assert calls == [
        ("downsample", 200.0),
        ("filter", 0.1, 40.0),
        ("rereference", ("Fp1", "Fp2", "C3", "C4")),
    ]
    assert result.summary["kept_eeg_channels"] == ["Fp1", "Fp2", "C3", "C4"]


def test_preprocess_run_rejects_missing_channels_tsv(tmp_path: Path) -> None:
    raw = _make_eeg_emg_raw()

    with pytest.raises(FileNotFoundError, match="channels.tsv"):
        preprocess_run(
            raw,
            channels_tsv_path=tmp_path / "missing_channels.tsv",
            apply_rereference=False,
            interpolate_bad_channels=False,
        )


def test_preprocess_run_fails_for_invalid_ica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _make_eeg_emg_raw()
    ica_path = tmp_path / "invalid_ica.fif"
    ica_path.write_text("not an ICA file", encoding="utf-8")

    def fake_apply_precomputed_ica(raw_arg, path_arg) -> None:
        raise ValueError("Could not find ICA data")

    monkeypatch.setattr(preprocessing_pipeline, "apply_precomputed_ica", fake_apply_precomputed_ica)

    with pytest.raises(RuntimeError, match="Failed to apply precomputed ICA"):
        preprocess_run(
            raw,
            ica_path=ica_path,
            emg_channel_patterns=[r"^EMG1$"],
            apply_rereference=False,
            interpolate_bad_channels=False,
        )


def test_preprocess_run_fails_for_missing_ica(tmp_path: Path) -> None:
    raw = _make_eeg_emg_raw()

    with pytest.raises(FileNotFoundError, match="ICA file not found"):
        preprocess_run(
            raw,
            ica_path=tmp_path / "missing_ica.fif",
            emg_channel_patterns=[r"^EMG1$"],
            apply_rereference=False,
            interpolate_bad_channels=False,
        )


def test_preprocess_run_saves_only_requested_intermediate_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _make_eeg_emg_raw()
    channels_tsv_path = tmp_path / "channels.tsv"
    channels_tsv_path.write_text("name\tstatus\nC3\tbad\n", encoding="utf-8")

    monkeypatch.setattr(preprocessing_pipeline, "downsample_raw", lambda raw_arg, config: None)
    monkeypatch.setattr(preprocessing_pipeline, "apply_bandpass_filter", lambda raw_arg, config: None)
    monkeypatch.setattr(preprocessing_pipeline, "interpolate_bad_eeg_channels", lambda raw_arg: None)
    monkeypatch.setattr(preprocessing_pipeline, "apply_average_reference", lambda raw_arg: None)

    result = preprocess_run(
        raw,
        channels_tsv_path=channels_tsv_path,
        target_sampling_rate_hz=200.0,
        low_cut_hz=0.1,
        high_cut_hz=40.0,
        emg_channel_patterns=[r"^EMG1$"],
        interpolate_bad_channels=True,
        apply_rereference=True,
        intermediates_dir=tmp_path / "intermediates",
    )

    intermediate_steps = [item["step"] for item in result.summary["intermediate_files"]]
    assert intermediate_steps == [
        "downsample",
        "filter",
        "apply_ica",
        "interpolate_bad_channels",
        "rereference",
    ]
    for item in result.summary["intermediate_files"]:
        assert Path(item["path"]).exists()


def test_cli_exposes_grouped_preprocess_eeg_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "preprocess",
            "eeg",
            "--input",
            "input.edf",
            "--output",
            "output.fif",
            "--config",
            "config/preprocessing.yaml",
        ]
    )

    assert args.command == "preprocess"
    assert args.preprocess_command == "eeg"
    assert args.config == "config/preprocessing.yaml"


def _make_eeg_emg_raw() -> mne.io.BaseRaw:
    ch_names = ["Fp1", "Fp2", "C3", "C4", "EMG1"]
    ch_types = ["eeg", "eeg", "eeg", "eeg", "emg"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=ch_types)
    data = np.random.default_rng(17).standard_normal((len(ch_names), 300))
    return mne.io.RawArray(data, info, verbose="ERROR")


def _write_event_grid(
    path: Path,
    *,
    action_a_intervals: list[Interval],
    action_b_intervals: list[Interval],
) -> Path:
    tiers = [
        Tier(
            name="palign-A",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "#")],
        ),
        Tier(
            name="palign-B",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "#")],
        ),
        Tier(
            name="ipu-A",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "ipu a")],
        ),
        Tier(
            name="ipu-B",
            xmin=0.0,
            xmax=1.0,
            intervals=[Interval(0.0, 1.0, "ipu b")],
        ),
        Tier(name="action A", xmin=0.0, xmax=1.0, intervals=action_a_intervals),
        Tier(name="action B", xmin=0.0, xmax=1.0, intervals=action_b_intervals),
    ]
    write_textgrid(TextGrid(xmin=0.0, xmax=1.0, tiers=tiers), path)
    return path
