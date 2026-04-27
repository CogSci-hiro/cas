"""CLI command for the FPP neural permutation null."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.neural_permutation_null import run_fpp_neural_permutation_null


def add_run_fpp_neural_permutation_null_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``run-fpp-neural-permutation-null`` command."""

    parser = subparsers.add_parser(
        "run-fpp-neural-permutation-null",
        help="Run the circular-shift FPP neural permutation null on a model-ready riskset.",
    )
    parser.add_argument("--riskset-path", required=True, help="Path to the model-ready FPP neural riskset.")
    parser.add_argument(
        "--output-dir",
        default="results/hazard_behavior/neural_lowlevel/permutation_null",
        help="Directory where permutation-null outputs should be written.",
    )
    parser.add_argument("--neural-family", choices=["alpha", "beta", "alpha_beta", "all"], default="beta")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--event-column", default="event_fpp")
    parser.add_argument("--episode-column", default="episode_id")
    parser.add_argument("--participant-column", default="participant_speaker_id")
    parser.add_argument("--run-column", default="run")
    parser.add_argument("--delta-criterion", choices=["bic", "aic"], default="bic")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-permutations-for-smoke-test", type=int, default=None)
    parser.add_argument("--max-fit-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print progress and show a permutation progress bar.")


def run_fpp_neural_permutation_null_command(args: argparse.Namespace) -> int:
    """Run the FPP neural permutation-null CLI command."""

    result = run_fpp_neural_permutation_null(
        riskset_path=Path(args.riskset_path),
        output_dir=Path(args.output_dir),
        neural_family=str(args.neural_family),
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
        event_column=str(args.event_column),
        episode_column=str(args.episode_column),
        participant_column=str(args.participant_column),
        run_column=str(args.run_column),
        delta_criterion=str(args.delta_criterion),
        n_jobs=int(args.n_jobs),
        max_permutations_for_smoke_test=(
            None if args.max_permutations_for_smoke_test is None else int(args.max_permutations_for_smoke_test)
        ),
        max_fit_rows=None if args.max_fit_rows is None else int(args.max_fit_rows),
        verbose=bool(args.verbose),
    )
    print(result.output_dir)
    return 0
