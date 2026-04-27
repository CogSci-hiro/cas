"""CLI command for plotting behavioural latency-regime Stan results."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.plot_latency_regime import plot_behaviour_latency_regime_results


def add_plot_behaviour_latency_regime_results_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``plot-behaviour-latency-regime-results`` command."""

    parser = subparsers.add_parser(
        "plot-behaviour-latency-regime-results",
        help="Plot the exploratory behavioural latency-regime Stan results.",
    )
    parser.add_argument(
        "--stan-results-dir",
        default="results/hazard_behavior/latency_regime/stan_models",
        help="Directory containing behavioural latency-regime Stan outputs.",
    )
    parser.add_argument(
        "--event-data-csv",
        default="results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv",
        help="Event-only behavioural latency-regime CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hazard_behavior/latency_regime/figures",
        help="Directory where behavioural latency-regime figures should be written.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print latency-regime plotting progress.")


def run_plot_behaviour_latency_regime_results_command(args: argparse.Namespace) -> int:
    """Run the behavioural latency-regime plotting command."""

    plot_behaviour_latency_regime_results(
        stan_results_dir=Path(args.stan_results_dir),
        event_data_csv=Path(args.event_data_csv),
        output_dir=Path(args.output_dir),
        verbose=bool(args.verbose),
    )
    print(Path(args.output_dir))
    return 0
