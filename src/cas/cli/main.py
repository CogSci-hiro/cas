"""CLI entry point for CAS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from cas.features.envelope import extract_hilbert_envelope
from trf.nested_cv import loro_nested_cv
from trf.prepare import prepare_trf_runs


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cas", description="CAS command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    return parser


def _run_trf(args: argparse.Namespace) -> int:
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "trf":
        return _run_trf(args)
    if args.command == "envelope":
        return _run_envelope(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
