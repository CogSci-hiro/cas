"""CLI command for the first-pass FPP hazard analysis."""

from __future__ import annotations

import argparse

from cas.hazard import run_hazard_analysis_from_config


def add_hazard_fpp_tde_hmm_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``hazard-fpp-tde-hmm`` command."""

    parser = subparsers.add_parser(
        "hazard-fpp-tde-hmm",
        help="Run a first-pass pooled discrete-time hazard analysis anchored at partner onset using existing TDE-HMM outputs.",
    )
    parser.add_argument(
        "--config",
        default="config/hazard_fpp_tde_hmm.yaml",
        help="Path to the dedicated hazard-analysis config YAML.",
    )


def run_hazard_fpp_tde_hmm_command(args: argparse.Namespace) -> int:
    """Run the first-pass FPP hazard analysis."""

    result = run_hazard_analysis_from_config(args.config)
    print(result.output_dir)
    return 0
