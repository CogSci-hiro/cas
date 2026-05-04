"""Nested CLI for the behavioral hazard pipeline."""

from __future__ import annotations

import argparse

from cas.behavior.pipeline import run_behavior_hazard_stage


BEHAVIOR_HAZARD_STAGES = (
    "build-risksets",
    "add-predictors",
    "select-lag",
    "fit-models",
    "tables",
    "summarize",
    "figures",
    "qc",
    "all",
)


def add_behavior_hazard_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    behavior_parser = subparsers.add_parser("behavior", help="Behavioral analysis commands.")
    behavior_subparsers = behavior_parser.add_subparsers(dest="behavior_command", required=True)
    hazard_parser = behavior_subparsers.add_parser("hazard", help="Behavioral hazard commands.")
    hazard_subparsers = hazard_parser.add_subparsers(dest="behavior_hazard_command", required=True)
    for stage in BEHAVIOR_HAZARD_STAGES:
        stage_parser = hazard_subparsers.add_parser(stage, help=f"Run behavioral hazard stage: {stage}.")
        stage_parser.add_argument("--config", default="config/behavior/hazard.yaml", help="Behavioral hazard config path.")
        stage_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable progress output and verbose stage logging.",
        )


def run_behavior_hazard_command(args: argparse.Namespace) -> int:
    run_behavior_hazard_stage(
        str(args.behavior_hazard_command),
        config_path=str(args.config),
        verbose=bool(getattr(args, "verbose", False)),
    )
    return 0
