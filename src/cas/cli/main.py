"""CLI entry point for CAS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy.io import wavfile

from cas.cli.commands.annotations import add_annotations_parser, run_annotations_command

if TYPE_CHECKING:
    import mne


def _load_runs(paths: list[str], *, label: str) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    for p in paths:
        arr = np.load(p)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{label} file is not an ndarray: {p}")
        runs.append(arr)
    return runs


def _load_array(path: str, *, label: str) -> np.ndarray:
    array = np.load(path)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} file is not an ndarray: {path}")
    return array


def _as_2d_array(array: np.ndarray, *, label: str) -> np.ndarray:
    x = np.asarray(array, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{label} must be 1D or 2D.")
    if not np.isfinite(x).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return x


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def _load_paths_config(config_root: Path) -> dict:
    paths_path = config_root / "paths.yaml"
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths config not found: {paths_path}")
    return _load_yaml(paths_path)


def _resolve_path(path_str: str, *, project_root: Path, fallback_root: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_candidate = project_root / path
    if project_candidate.exists() or fallback_root is None:
        return project_candidate
    return fallback_root / path


def _load_signal(path: str) -> tuple[np.ndarray, float | None]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".npy":
        return _load_array(path, label="input"), None

    if input_path.suffix.lower() == ".wav":
        sampling_rate_hz, signal = wavfile.read(input_path)
        signal_array = np.asarray(signal, dtype=float)
        if signal_array.ndim == 2:
            # Average channels to mono so the envelope extractor gets a 1D signal.
            signal_array = signal_array.mean(axis=1)
        return signal_array, float(sampling_rate_hz)

    raise ValueError(f"Unsupported input format for envelope extraction: {input_path.suffix}")


def _load_raw_eeg(path: str) -> "mne.io.BaseRaw":
    import mne

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".fif":
        return mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(input_path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG input format: {input_path.suffix}")


def _save_raw_fif(raw: "mne.io.BaseRaw", output_path_str: str) -> Path:
    output_path = Path(output_path_str)
    if output_path.suffix.lower() != ".fif":
        raise ValueError("Output raw path must end with .fif.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=True)
    return output_path


def _save_npz(output_path_str: str, **arrays: object) -> Path:
    output_path = Path(output_path_str)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output NPZ path must end with .npz.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    return output_path


def _save_json(data: dict[str, object], output_path_str: str) -> Path:
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cas", description="CAS command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_annotations_parser(subparsers)

    trf_parser = subparsers.add_parser("trf", help="Run TRF nested CV from run-wise arrays.")
    trf_parser.add_argument("--eeg-runs", nargs="+", required=True, help="Run-wise EEG .npy paths.")
    trf_parser.add_argument(
        "--predictor-runs", nargs="+", required=True, help="Run-wise predictor .npy paths."
    )
    trf_parser.add_argument("--eeg-sfreq", type=float, required=True)
    trf_parser.add_argument("--predictor-sfreq", type=float, required=True)
    trf_parser.add_argument("--target-sfreq", type=float, required=True)
    trf_parser.add_argument("--tmin-s", type=float, required=True)
    trf_parser.add_argument("--tmax-s", type=float, required=True)
    trf_parser.add_argument("--alphas", nargs="+", type=float, required=True)
    trf_parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix for scores (.json) and coefficients (.npz).",
    )

    trf_config_parser = subparsers.add_parser(
        "trf-config",
        help="Run TRF nested CV from config-driven self/other predictor definitions.",
    )
    trf_config_parser.add_argument("--config", required=True, help="Path to config/trf.yaml.")
    trf_config_parser.add_argument("--subject", required=True, help="Target subject id, e.g. sub-001.")
    trf_config_parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve relative config paths. Defaults to cwd.",
    )
    trf_config_parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=None,
        help="Optional run list. Defaults to 1..n_runs from the TRF config.",
    )

    eeg_array_parser = subparsers.add_parser(
        "eeg-array",
        help="Convert raw EEG (.edf or .fif) into a channel-wise NumPy array for TRF input.",
    )
    eeg_array_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    eeg_array_parser.add_argument("--output", required=True, help="Output .npy path.")
    eeg_array_parser.add_argument(
        "--target-sfreq-hz",
        type=float,
        default=None,
        help="Optional resampling rate in Hz.",
    )
    eeg_array_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON sidecar path with channel names and sampling rate.",
    )

    envelope_parser = subparsers.add_parser(
        "envelope",
        help="Extract a Hilbert speech envelope from a .wav or 1D .npy signal.",
    )
    envelope_parser.add_argument("--input", required=True, help="Input .wav or 1D signal .npy path.")
    envelope_parser.add_argument("--output", required=True, help="Output envelope .npy path.")
    envelope_parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=None,
        help="Required for .npy inputs; inferred automatically for .wav inputs.",
    )
    envelope_parser.add_argument("--lowpass-hz", type=float, default=10.0)
    envelope_parser.add_argument("--filter-order", type=int, default=4)
    envelope_parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON sidecar path for envelope metadata.",
    )

    preprocess_raw_parser = subparsers.add_parser(
        "preprocess-raw",
        help="Run the EEG preprocessing pipeline and save the result as .fif.",
    )
    preprocess_raw_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    preprocess_raw_parser.add_argument("--output", required=True, help="Output preprocessed .fif path.")
    preprocess_raw_parser.add_argument(
        "--annotations-path",
        default=None,
        help="Optional annotation TextGrid path used to derive events before MNE fallback.",
    )
    preprocess_raw_parser.add_argument(
        "--channels-tsv",
        default=None,
        help="Optional BIDS channels.tsv path used to load bad channels.",
    )
    preprocess_raw_parser.add_argument(
        "--ica-path",
        default=None,
        help="Optional precomputed ICA .fif path.",
    )
    preprocess_raw_parser.add_argument(
        "--target-sfreq-hz",
        type=float,
        default=None,
        help="Optional target sampling rate in Hz.",
    )
    preprocess_raw_parser.add_argument(
        "--low-cut-hz",
        type=float,
        default=None,
        help="Optional high-pass cutoff in Hz.",
    )
    preprocess_raw_parser.add_argument(
        "--high-cut-hz",
        type=float,
        default=None,
        help="Optional low-pass cutoff in Hz.",
    )
    preprocess_raw_parser.add_argument(
        "--montage",
        default="standard_1020",
        help="Montage name applied to EEG channels. Use an empty string to skip.",
    )
    preprocess_raw_parser.add_argument(
        "--annotation-pairing-margin-s",
        type=float,
        default=1.0,
        help="Pairing margin used when deriving events from annotations.",
    )
    preprocess_raw_parser.add_argument(
        "--eeg-channel-name",
        action="append",
        default=[],
        help="Explicit channel name to force into the EEG set. Repeat as needed.",
    )
    preprocess_raw_parser.add_argument(
        "--eeg-channel-pattern",
        action="append",
        default=[],
        help="Regex pattern used to identify EEG channels. Repeat as needed.",
    )
    preprocess_raw_parser.add_argument(
        "--emg-channel-name",
        action="append",
        default=[],
        help="Explicit channel name to force into the EMG set. Repeat as needed.",
    )
    preprocess_raw_parser.add_argument(
        "--emg-channel-pattern",
        action="append",
        default=[],
        help="Regex pattern used to identify EMG channels. Repeat as needed.",
    )
    preprocess_raw_parser.add_argument(
        "--emg-output",
        default=None,
        help="Optional EMG artifact .npz output path.",
    )
    preprocess_raw_parser.add_argument(
        "--events-output",
        default=None,
        help="Optional preprocessing events .tsv output path.",
    )
    preprocess_raw_parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional preprocessing summary .json output path.",
    )
    preprocess_raw_parser.add_argument(
        "--skip-interpolate-bads",
        action="store_true",
        help="Skip bad channel interpolation.",
    )
    preprocess_raw_parser.add_argument(
        "--skip-rereference",
        action="store_true",
        help="Skip average rereferencing.",
    )
    preprocess_raw_parser.add_argument(
        "--skip-emg",
        action="store_true",
        help="Do not preserve EMG channels as a separate NPZ artifact.",
    )

    downsample_raw_parser = subparsers.add_parser(
        "downsample-raw",
        help="Downsample raw EEG and save the result as .fif.",
    )
    downsample_raw_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    downsample_raw_parser.add_argument("--output", required=True, help="Output downsampled .fif path.")
    downsample_raw_parser.add_argument(
        "--target-sfreq-hz",
        type=float,
        required=True,
        help="Target sampling rate in Hz.",
    )

    filter_raw_parser = subparsers.add_parser(
        "filter-raw",
        help="Apply broad EEG filtering and save the result as .fif.",
    )
    filter_raw_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    filter_raw_parser.add_argument("--output", required=True, help="Output filtered .fif path.")
    filter_raw_parser.add_argument(
        "--low-cut-hz",
        type=float,
        default=None,
        help="Optional high-pass cutoff in Hz.",
    )
    filter_raw_parser.add_argument(
        "--high-cut-hz",
        type=float,
        default=None,
        help="Optional low-pass cutoff in Hz.",
    )

    set_bad_channels_parser = subparsers.add_parser(
        "set-bad-channels",
        help="Load bad channels from BIDS channels.tsv, set them on raw, and save as .fif.",
    )
    set_bad_channels_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    set_bad_channels_parser.add_argument("--output", required=True, help="Output .fif path.")
    set_bad_channels_parser.add_argument(
        "--channels-tsv",
        required=True,
        help="BIDS channels.tsv path used to load bad channels.",
    )

    average_reference_parser = subparsers.add_parser(
        "average-reference",
        help="Apply average EEG reference and save the result as .fif.",
    )
    average_reference_parser.add_argument(
        "--input",
        required=True,
        help="Input EEG .edf or .fif path.",
    )
    average_reference_parser.add_argument(
        "--output",
        required=True,
        help="Output rereferenced .fif path.",
    )
    average_reference_parser.add_argument(
        "--projection",
        action="store_true",
        help="Add the average reference as a projection instead of applying it directly.",
    )

    apply_ica_parser = subparsers.add_parser(
        "apply-ica",
        help="Apply precomputed ICA and save the result as .fif.",
    )
    apply_ica_parser.add_argument("--input", required=True, help="Input EEG .edf or .fif path.")
    apply_ica_parser.add_argument("--output", required=True, help="Output ICA-cleaned .fif path.")
    apply_ica_parser.add_argument(
        "--ica-path",
        required=True,
        help="Precomputed ICA .fif path.",
    )

    lmeeeg_parser = subparsers.add_parser(
        "lmeeeg",
        help="Run config-driven lmeEEG analysis from an existing epochs file.",
    )
    lmeeeg_parser.add_argument("--epochs", required=True, help="Input epochs .fif path.")
    lmeeeg_parser.add_argument(
        "--config",
        required=True,
        help="Standalone lmeEEG YAML config path.",
    )
    lmeeeg_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where lmeEEG outputs will be written.",
    )
    lmeeeg_parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV path when epochs.metadata is missing or should be overridden.",
    )
    return parser


def _run_trf(args: argparse.Namespace) -> int:
    from trf.nested_cv import loro_nested_cv
    from trf.prepare import prepare_trf_runs

    eeg_runs = _load_runs(args.eeg_runs, label="eeg")
    predictor_runs = _load_runs(args.predictor_runs, label="predictor")

    X_runs, Y_runs = prepare_trf_runs(
        eeg_runs=eeg_runs,
        predictor_runs=predictor_runs,
        eeg_sfreq=args.eeg_sfreq,
        predictor_sfreq=args.predictor_sfreq,
        target_sfreq=args.target_sfreq,
        tmin_s=args.tmin_s,
        tmax_s=args.tmax_s,
    )
    fold_scores, fold_coefficients = loro_nested_cv(
        X_runs=X_runs, Y_runs=Y_runs, alphas=args.alphas
    )

    print(json.dumps(fold_scores, indent=2))

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        score_path = prefix.with_suffix(".scores.json")
        coef_path = prefix.with_suffix(".coefs.npz")
        score_path.write_text(json.dumps(fold_scores, indent=2) + "\n", encoding="utf-8")
        np.savez(
            coef_path,
            fold_coefficients=np.array(fold_coefficients, dtype=object),
        )
        print(f"Saved scores to {score_path}")
        print(f"Saved coefficients to {coef_path}")

    return 0


def _run_envelope(args: argparse.Namespace) -> int:
    from cas.features.envelope import extract_hilbert_envelope

    signal, inferred_sampling_rate_hz = _load_signal(args.input)
    sampling_rate_hz = args.sampling_rate_hz
    if sampling_rate_hz is None:
        sampling_rate_hz = inferred_sampling_rate_hz
    if sampling_rate_hz is None:
        raise ValueError("`--sampling-rate-hz` is required when the input is not a .wav file.")

    envelope = extract_hilbert_envelope(
        signal=signal,
        sampling_rate_hz=sampling_rate_hz,
        lowpass_hz=args.lowpass_hz,
        filter_order=args.filter_order,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, envelope)
    print(f"Saved envelope to {output_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "input": str(Path(args.input)),
            "output": str(output_path),
            "sampling_rate_hz": sampling_rate_hz,
            "lowpass_hz": args.lowpass_hz,
            "filter_order": args.filter_order,
            "n_samples": int(envelope.shape[0]),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary to {summary_path}")

    return 0


def _run_eeg_array(args: argparse.Namespace) -> int:
    raw = _load_raw_eeg(args.input)
    raw.pick("eeg")
    if args.target_sfreq_hz is not None:
        raw.resample(float(args.target_sfreq_hz))

    data = raw.get_data().T
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(data, dtype=float))
    print(f"Saved EEG array to {output_path}")

    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input": str(Path(args.input)),
            "output": str(output_path),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_samples": int(data.shape[0]),
            "n_channels": int(data.shape[1]),
            "channel_names": list(raw.ch_names),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved EEG metadata to {metadata_path}")

    return 0


def _run_preprocess_raw(args: argparse.Namespace) -> int:
    from cas.preprocessing import preprocess_run, write_events_tsv

    raw = _load_raw_eeg(args.input)
    result = preprocess_run(
        raw,
        annotation_path=args.annotations_path,
        channels_tsv_path=args.channels_tsv,
        ica_path=args.ica_path,
        target_sampling_rate_hz=args.target_sfreq_hz,
        low_cut_hz=args.low_cut_hz,
        high_cut_hz=args.high_cut_hz,
        montage_name=args.montage or None,
        annotation_pairing_margin_s=float(args.annotation_pairing_margin_s),
        eeg_channel_names=list(args.eeg_channel_name),
        eeg_channel_patterns=list(args.eeg_channel_pattern),
        emg_channel_names=list(args.emg_channel_name),
        emg_channel_patterns=list(args.emg_channel_pattern),
        interpolate_bad_channels=not args.skip_interpolate_bads,
        apply_rereference=not args.skip_rereference,
        keep_emg=not args.skip_emg,
    )
    output_path = _save_raw_fif(result.eeg_raw, args.output)
    print(f"Saved preprocessed raw to {output_path}")

    if args.emg_output:
        emg_output_path = _save_npz(
            args.emg_output,
            data=np.asarray(result.emg_data, dtype=float),
            channel_names=np.asarray(result.emg_channel_names, dtype=object),
        )
        print(f"Saved EMG artifact to {emg_output_path}")

    if args.events_output:
        events_output_path = write_events_tsv(result.events_rows, args.events_output)
        print(f"Saved preprocessing events to {events_output_path}")

    if args.summary_json:
        summary_output_path = _save_json(result.summary, args.summary_json)
        print(f"Saved preprocessing summary to {summary_output_path}")

    return 0


def _run_downsample_raw(args: argparse.Namespace) -> int:
    from cas.preprocessing import downsample_raw

    raw = _load_raw_eeg(args.input)
    downsample_raw(raw, sampling_rate_hz=float(args.target_sfreq_hz))
    output_path = _save_raw_fif(raw, args.output)
    print(f"Saved downsampled raw to {output_path}")
    return 0


def _run_filter_raw(args: argparse.Namespace) -> int:
    from cas.preprocessing import bandpass_filter_raw

    raw = _load_raw_eeg(args.input)
    bandpass_filter_raw(
        raw,
        low_cut_hz=args.low_cut_hz,
        high_cut_hz=args.high_cut_hz,
    )
    output_path = _save_raw_fif(raw, args.output)
    print(f"Saved filtered raw to {output_path}")
    return 0


def _run_set_bad_channels(args: argparse.Namespace) -> int:
    from cas.preprocessing import read_bad_channels_from_bids_tsv, set_bad_channels

    raw = _load_raw_eeg(args.input)
    bad_channel_names = read_bad_channels_from_bids_tsv(args.channels_tsv)
    set_bad_channels(raw, bad_channel_names)
    output_path = _save_raw_fif(raw, args.output)
    print(f"Saved raw with bad channels to {output_path}")
    return 0


def _run_average_reference(args: argparse.Namespace) -> int:
    from cas.preprocessing import apply_average_reference

    raw = _load_raw_eeg(args.input)
    apply_average_reference(raw, projection=bool(args.projection))
    output_path = _save_raw_fif(raw, args.output)
    print(f"Saved average-referenced raw to {output_path}")
    return 0


def _run_apply_ica(args: argparse.Namespace) -> int:
    from cas.preprocessing import apply_precomputed_ica

    raw = _load_raw_eeg(args.input)
    apply_precomputed_ica(raw, args.ica_path)
    output_path = _save_raw_fif(raw, args.output)
    print(f"Saved ICA-cleaned raw to {output_path}")
    return 0


def _run_lmeeeg(args: argparse.Namespace) -> int:
    from cas.stats.lmeeeg_pipeline import run_lmeeeg_analysis

    summary = run_lmeeeg_analysis(
        epochs_path=args.epochs,
        config_path=args.config,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _build_config_driven_trf_inputs(
    *,
    trf_config: dict,
    subject_id: str,
    runs: list[int],
    project_root: Path,
    config_root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    from cas.trf.prepare import load_dyad_table, resolve_predictor_paths

    trf_section = trf_config.get("trf", {})
    paths_config = _load_paths_config(config_root)
    predictors = trf_section.get("predictors", [])
    target_config = trf_section.get("target", {})
    pairing_config = trf_section.get("pairing", {})

    dyads_csv = pairing_config.get("dyads_csv")
    if not isinstance(dyads_csv, str) or not dyads_csv:
        raise ValueError("TRF config must define trf.pairing.dyads_csv.")
    dyad_table = load_dyad_table(
        _resolve_path(dyads_csv, project_root=project_root, fallback_root=config_root)
    )

    target_path_template = target_config.get("path")
    if not isinstance(target_path_template, str) or not target_path_template:
        raise ValueError("TRF config must define trf.target.path.")
    eeg_array_root = Path(paths_config["eeg_array_root"])

    predictor_runs: list[np.ndarray] = []
    eeg_runs: list[np.ndarray] = []
    predictor_names = [str(predictor["name"]) for predictor in predictors]

    original_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        for run in runs:
            predictor_paths = resolve_predictor_paths(subject_id, run, predictors, dyad_table)
            predictor_arrays = [
                _as_2d_array(
                    _load_array(str(_resolve_path(str(path), project_root=project_root)), label=name),
                    label=name,
                )
                for name, path in predictor_paths.items()
            ]
            predictor_runs.append(np.concatenate(predictor_arrays, axis=1))

            target_path = _resolve_path(
                str(eeg_array_root / target_path_template.format(subject=subject_id, run=run)),
                project_root=project_root,
            )
            eeg_runs.append(
                _as_2d_array(_load_array(str(target_path), label="target"), label="target")
            )
    finally:
        os.chdir(original_cwd)

    return eeg_runs, predictor_runs, predictor_names


def _default_trf_output_prefix(*, trf_config: dict, subject_id: str, config_root: Path) -> Path:
    trf_section = trf_config.get("trf", {})
    analysis_id = trf_section.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError("TRF config must define trf.analysis_id.")

    output_root_template = trf_section.get("output", {}).get("root")
    if not isinstance(output_root_template, str) or not output_root_template:
        raise ValueError("TRF config must define trf.output.root.")

    paths_config = _load_paths_config(config_root)
    trf_root = Path(paths_config["trf_root"])
    output_root = trf_root / output_root_template.format(analysis_id=analysis_id)
    return output_root / subject_id / "trf"


def _run_trf_config(args: argparse.Namespace) -> int:
    from trf.nested_cv import loro_nested_cv
    from trf.prepare import prepare_trf_runs

    project_root = Path(args.project_root).resolve()
    config_path = _resolve_path(args.config, project_root=project_root).resolve()
    trf_config = _load_yaml(config_path)
    config_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    trf_section = trf_config.get("trf", {})
    timing_config = trf_section.get("timing", {})
    model_config = trf_section.get("model", {})
    output_config = trf_section.get("output", {})
    cv_config = trf_section.get("cv", {})

    n_runs = int(cv_config.get("n_runs", 0))
    runs = list(args.runs) if args.runs is not None else list(range(1, n_runs + 1))
    if not runs:
        raise ValueError("No runs requested. Provide --runs or set trf.cv.n_runs > 0.")

    eeg_runs, predictor_runs, predictor_names = _build_config_driven_trf_inputs(
        trf_config=trf_config,
        subject_id=args.subject,
        runs=runs,
        project_root=project_root,
        config_root=config_root,
    )
    target_sfreq_hz = float(timing_config["target_sfreq_hz"])
    X_runs, Y_runs = prepare_trf_runs(
        eeg_runs=eeg_runs,
        predictor_runs=predictor_runs,
        eeg_sfreq=target_sfreq_hz,
        predictor_sfreq=target_sfreq_hz,
        target_sfreq=target_sfreq_hz,
        tmin_s=float(timing_config["tmin_s"]),
        tmax_s=float(timing_config["tmax_s"]),
    )
    fold_scores, fold_coefficients = loro_nested_cv(
        X_runs=X_runs,
        Y_runs=Y_runs,
        alphas=[float(alpha) for alpha in model_config["alphas"]],
    )

    result = {
        "analysis_id": trf_section.get("analysis_id"),
        "subject": args.subject,
        "runs": runs,
        "predictors": predictor_names,
        "fold_scores": fold_scores,
    }
    print(json.dumps(result, indent=2))

    output_prefix = _default_trf_output_prefix(
        trf_config=trf_config,
        subject_id=args.subject,
        config_root=config_root,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if bool(output_config.get("save_scores", True)):
        score_path = output_prefix.with_suffix(".scores.json")
        score_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved scores to {score_path}")

    if bool(output_config.get("save_betas", True)):
        coef_path = output_prefix.with_suffix(".coefs.npz")
        np.savez(
            coef_path,
            fold_coefficients=np.array(fold_coefficients, dtype=object),
            predictor_names=np.array(predictor_names, dtype=object),
        )
        print(f"Saved coefficients to {coef_path}")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "annotations":
        return run_annotations_command(args)
    if args.command == "trf":
        return _run_trf(args)
    if args.command == "trf-config":
        return _run_trf_config(args)
    if args.command == "eeg-array":
        return _run_eeg_array(args)
    if args.command == "envelope":
        return _run_envelope(args)
    if args.command == "preprocess-raw":
        return _run_preprocess_raw(args)
    if args.command == "downsample-raw":
        return _run_downsample_raw(args)
    if args.command == "filter-raw":
        return _run_filter_raw(args)
    if args.command == "set-bad-channels":
        return _run_set_bad_channels(args)
    if args.command == "average-reference":
        return _run_average_reference(args)
    if args.command == "apply-ica":
        return _run_apply_ica(args)
    if args.command == "lmeeeg":
        return _run_lmeeeg(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
