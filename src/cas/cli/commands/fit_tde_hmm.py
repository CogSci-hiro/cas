"""CLI command for fitting a causal TDE-HMM on source-level EEG."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hmm import TdeHmmConfig, fit_tde_hmm_pipeline


def add_fit_tde_hmm_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``fit-tde-hmm`` command."""

    parser = subparsers.add_parser(
        "fit-tde-hmm",
        help="Fit a group-level causal TDE-HMM on non-speaking source EEG and export state entropy.",
    )
    parser.add_argument("--input-manifest", required=True, help="Manifest CSV with one row per run.")
    parser.add_argument("--output-dir", required=True, help="Directory where HMM outputs will be written.")
    parser.add_argument("--source-array-column", default="source_path")
    parser.add_argument("--speech-intervals-column", default="speech_path")
    parser.add_argument("--subject-column", default="subject")
    parser.add_argument("--run-column", default="run")
    parser.add_argument(
        "--input-sampling-rate-column",
        default="input_sampling_rate_hz",
        help="Optional manifest column containing the pre-downsampling sampling rate.",
    )
    parser.add_argument("--sampling-rate-hz", type=float, default=128.0)
    parser.add_argument("--causal-history-ms", type=float, default=100.0)
    parser.add_argument("--speech-guard-pre-s", type=float, default=0.2)
    parser.add_argument("--speech-guard-post-s", type=float, default=0.2)
    parser.add_argument("--minimum-chunk-duration-s", type=float, default=0.5)
    parser.add_argument("--candidate-k", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--n-initializations", type=int, default=5)
    parser.add_argument("--pca-n-components", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--covariance-type", default="full")
    parser.add_argument(
        "--no-standardize-per-run",
        action="store_true",
        help="Disable per-run feature standardization before masking and chunking.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )


def run_fit_tde_hmm_command(args: argparse.Namespace) -> int:
    """Run the causal TDE-HMM fitting pipeline."""

    config = TdeHmmConfig(
        sampling_rate_hz=float(args.sampling_rate_hz),
        causal_history_ms=float(args.causal_history_ms),
        speech_guard_pre_s=float(args.speech_guard_pre_s),
        speech_guard_post_s=float(args.speech_guard_post_s),
        minimum_chunk_duration_s=float(args.minimum_chunk_duration_s),
        candidate_k=tuple(int(k) for k in args.candidate_k),
        n_initializations=int(args.n_initializations),
        standardize_per_run=not bool(args.no_standardize_per_run),
        pca_n_components=None if args.pca_n_components is None else int(args.pca_n_components),
        random_seed=int(args.random_seed),
        covariance_type=str(args.covariance_type),
        verbose=not bool(args.quiet),
    )
    result = fit_tde_hmm_pipeline(
        manifest_path=Path(args.input_manifest),
        output_dir=Path(args.output_dir),
        config=config,
        source_array_column=str(args.source_array_column),
        speech_intervals_column=str(args.speech_intervals_column),
        subject_column=str(args.subject_column),
        run_column=str(args.run_column),
        input_sampling_rate_column=(
            None if args.input_sampling_rate_column == "" else str(args.input_sampling_rate_column)
        ),
    )
    print(result.fit_summary_path)
    return 0
