"""CLI command for time-resolved induced-power info-rate lmeEEG bridge analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.info_rate_induced_lmeeg import (
    load_info_rate_induced_lmeeg_config,
    run_info_rate_induced_lmeeg_pipeline,
)


def add_info_rate_induced_lmeeg_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``info-rate-induced-lmeeg`` command."""

    parser = subparsers.add_parser(
        "info-rate-induced-lmeeg",
        help="Run the lagged info-rate -> induced alpha/beta lmeEEG bridge analysis.",
    )
    parser.add_argument(
        "--config",
        default="config/info_rate_induced_lmeeg.yaml",
        help="Path to the info-rate induced lmeEEG YAML config.",
    )


def run_info_rate_induced_lmeeg_command(args: argparse.Namespace) -> int:
    """Run the lagged info-rate -> induced alpha/beta lmeEEG bridge analysis."""

    config = load_info_rate_induced_lmeeg_config(Path(args.config))
    result = run_info_rate_induced_lmeeg_pipeline(config)
    print(result.out_dir)
    return 0
