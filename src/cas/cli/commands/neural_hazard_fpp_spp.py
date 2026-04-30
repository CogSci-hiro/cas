"""CLI command for the final TDE-HMM entropy FPP-vs-SPP hazard interaction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.neural_hazard import load_neural_hazard_fpp_spp_config, run_neural_hazard_fpp_spp_pipeline


def add_neural_hazard_fpp_spp_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``neural-hazard-fpp-spp`` command."""

    parser = subparsers.add_parser(
        "neural-hazard-fpp-spp",
        help="Run the pooled FPP-vs-SPP entropy hazard interaction pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config/neural_hazard_fpp_spp.yaml",
        help="Path to the neural hazard FPP-vs-SPP YAML config.",
    )


def run_neural_hazard_fpp_spp_command(args: argparse.Namespace) -> int:
    """Run the pooled FPP-vs-SPP entropy hazard interaction pipeline."""

    config = load_neural_hazard_fpp_spp_config(Path(args.config))
    result = run_neural_hazard_fpp_spp_pipeline(config)
    print(result.out_dir)
    return 0
