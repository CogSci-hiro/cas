from __future__ import annotations

import argparse
from pathlib import Path

from cas.neural_hazard.fpp_spp_renyi_alpha_pipeline import (
    load_neural_hazard_fpp_spp_renyi_alpha_config,
    run_neural_hazard_fpp_spp_renyi_alpha_pipeline,
)


def add_neural_hazard_fpp_spp_renyi_alpha_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "neural-hazard-fpp-spp-renyi-alpha",
        help="Run the pooled FPP-vs-SPP Renyi-alpha hazard pipeline.",
    )
    parser.add_argument("--config", default="config/neural_hazard_fpp_spp_renyi_alpha.yaml")


def run_neural_hazard_fpp_spp_renyi_alpha_command(args: argparse.Namespace) -> int:
    cfg = load_neural_hazard_fpp_spp_renyi_alpha_config(Path(args.config))
    out = run_neural_hazard_fpp_spp_renyi_alpha_pipeline(cfg)
    print(out.out_dir)
    return 0
