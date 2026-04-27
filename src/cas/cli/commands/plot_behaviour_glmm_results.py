"""CLI command for plotting behavioural GLMM results produced in R."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.plot_r_results import plot_behaviour_hazard_results


def add_plot_behaviour_glmm_results_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``plot-behaviour-glmm-results`` command."""

    for command_name in ("plot-behaviour-glmm-results", "plot-behaviour-hazard-results"):
        parser = subparsers.add_parser(
            command_name,
            help="Plot the active behavioural hazard result suite from timing-control and R GLMM outputs.",
        )
        parser.add_argument("--r-results-dir", required=True, help="Directory containing behavioural GLMM R outputs.")
        parser.add_argument("--output-dir", required=True, help="Directory where figures should be written.")
        parser.add_argument(
            "--timing-control-models-dir",
            default=None,
            help="Optional directory containing behaviour_timing_control_lag_selection.csv.",
        )
        parser.add_argument(
            "--qc-output-dir",
            default=None,
            help="Optional directory where behavioural hazard QC figures should be written.",
        )


def run_plot_behaviour_glmm_results_command(args: argparse.Namespace) -> int:
    """Run the behavioural GLMM plotting command."""

    plot_behaviour_hazard_results(
        r_results_dir=Path(args.r_results_dir),
        output_dir=Path(args.output_dir),
        timing_control_models_dir=Path(args.timing_control_models_dir) if args.timing_control_models_dir else None,
        qc_output_dir=Path(args.qc_output_dir) if args.qc_output_dir else None,
    )
    print(Path(args.output_dir))
    return 0
