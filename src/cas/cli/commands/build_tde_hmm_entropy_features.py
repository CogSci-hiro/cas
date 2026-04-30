"""CLI command for aggregating per-run TDE-HMM entropy CSVs into one features table."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.neural_hazard.fpp_spp_pipeline import build_entropy_features_table_from_glhmm_output


def add_build_tde_hmm_entropy_features_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``build-tde-hmm-entropy-features`` command."""

    parser = subparsers.add_parser(
        "build-tde-hmm-entropy-features",
        help="Aggregate per-run TDE-HMM state-entropy CSVs into one features table.",
    )
    parser.add_argument("--glhmm-output-dir", required=True)
    parser.add_argument("--output-path", required=True)


def run_build_tde_hmm_entropy_features_command(args: argparse.Namespace) -> int:
    """Run the entropy-table aggregation command."""

    output_path = build_entropy_features_table_from_glhmm_output(
        Path(args.glhmm_output_dir),
        Path(args.output_path),
    )
    print(output_path)
    return 0
