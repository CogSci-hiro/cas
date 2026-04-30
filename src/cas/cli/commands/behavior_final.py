from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.final_behavior import (
    run_behavior_final_compare,
    run_behavior_final_fit,
    run_behavior_final_select_lag,
)


def add_behavior_final_select_lag_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("behavior-final-select-lag", help="Select one shared final FPP lag for behavioral hazard models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs and progress bars.")


def run_behavior_final_select_lag_command(args: argparse.Namespace) -> int:
    path = run_behavior_final_select_lag(Path(args.config), Path(args.out_dir), verbose=bool(args.verbose))
    print(path)
    return 0


def add_behavior_final_fit_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("behavior-final-fit", help="Fit final behavioral hazard model sequence for one anchor type.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-type", required=True, choices=["fpp", "spp"])
    parser.add_argument("--selected-lag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs and progress bars.")


def run_behavior_final_fit_command(args: argparse.Namespace) -> int:
    path = run_behavior_final_fit(
        Path(args.config),
        str(args.anchor_type),
        Path(args.selected_lag),
        Path(args.out_dir),
        verbose=bool(args.verbose),
    )
    print(path)
    return 0


def add_behavior_final_compare_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("behavior-final-compare", help="Fit pooled FPP-vs-SPP interaction model at selected lag.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected-lag", required=True)
    parser.add_argument("--fpp-riskset", required=True)
    parser.add_argument("--spp-riskset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs and progress bars.")


def run_behavior_final_compare_command(args: argparse.Namespace) -> int:
    path = run_behavior_final_compare(
        Path(args.config),
        Path(args.selected_lag),
        Path(args.fpp_riskset),
        Path(args.spp_riskset),
        Path(args.out_dir),
        verbose=bool(args.verbose),
    )
    print(path)
    return 0
