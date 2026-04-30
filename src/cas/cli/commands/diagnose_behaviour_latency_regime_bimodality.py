"""CLI command for conditional-vs-marginal latency-regime bimodality diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.conditional_bimodality_diagnostics import (
    run_latency_regime_conditional_bimodality_diagnostics,
)


def add_diagnose_behaviour_latency_regime_bimodality_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``diagnose-behaviour-latency-regime-bimodality`` command."""

    parser = subparsers.add_parser(
        "diagnose-behaviour-latency-regime-bimodality",
        help="Generate conditional-vs-marginal bimodality diagnostics for the event-only latency-regime analysis.",
    )
    parser.add_argument(
        "--event-data-csv",
        default="results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv",
        help="Event-only behavioural latency-regime CSV.",
    )
    parser.add_argument(
        "--stan-results-dir",
        default="results/hazard_behavior/latency_regime/stan_models",
        help="Directory containing behavioural latency-regime model outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/hazard_behavior/latency_regime/figures",
        help="Directory where figures should be written.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="results/hazard_behavior/latency_regime/diagnostics",
        help="Directory where diagnostic tables and reports should be written.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic progress.")


def run_diagnose_behaviour_latency_regime_bimodality_command(args: argparse.Namespace) -> int:
    """Run the conditional-vs-marginal bimodality diagnostics command."""

    result = run_latency_regime_conditional_bimodality_diagnostics(
        event_data_csv=Path(args.event_data_csv),
        stan_results_dir=Path(args.stan_results_dir),
        figures_dir=Path(args.figures_dir),
        diagnostics_dir=Path(args.diagnostics_dir),
        verbose=bool(args.verbose),
    )
    print(result.report_path)
    return 0
