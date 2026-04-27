"""CLI command for diagnosing behavioural latency-regime Stan results."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.diagnose_latency_regime import diagnose_behaviour_latency_regime


def add_diagnose_behaviour_latency_regime_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``diagnose-behaviour-latency-regime`` command."""

    parser = subparsers.add_parser(
        "diagnose-behaviour-latency-regime",
        help="Generate diagnostics for an existing behavioural latency-regime Stan analysis.",
    )
    parser.add_argument(
        "--event-data-csv",
        default="results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv",
        help="Event-only behavioural latency-regime CSV.",
    )
    parser.add_argument(
        "--stan-results-dir",
        default="results/hazard_behavior/latency_regime/stan_models",
        help="Directory containing behavioural latency-regime Stan outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hazard_behavior/latency_regime/diagnostics",
        help="Directory where diagnostics should be written.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic progress.")


def run_diagnose_behaviour_latency_regime_command(args: argparse.Namespace) -> int:
    """Run the behavioural latency-regime diagnostics command."""

    result = diagnose_behaviour_latency_regime(
        event_data_csv=Path(args.event_data_csv),
        stan_results_dir=Path(args.stan_results_dir),
        output_dir=Path(args.output_dir),
        verbose=bool(args.verbose),
    )
    print(result.report_path)
    return 0
