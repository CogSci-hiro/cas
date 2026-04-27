"""CLI command for diagnosing SPP neural hazard convergence failures."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.diagnose_spp_neural_failure import diagnose_spp_neural_hazard_failure


def add_diagnose_spp_neural_hazard_failure_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``diagnose-spp-neural-hazard-failure`` command."""

    parser = subparsers.add_parser(
        "diagnose-spp-neural-hazard-failure",
        help="Generate diagnostics for failed SPP low-level neural hazard fits.",
    )
    parser.add_argument("--riskset-path", required=True, help="Path to the SPP neural riskset table.")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Optional neural low-level models directory used to recover lag metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hazard_behavior/neural_lowlevel/diagnostics",
        help="Directory where diagnostic artifacts should be written.",
    )
    parser.add_argument(
        "--skip-incremental-fits",
        action="store_true",
        help="Skip the incremental GLM diagnostics and only write the descriptive diagnostics.",
    )
    parser.add_argument("--run-ridge-diagnostic", action="store_true", help="Run ridge-logistic fallback diagnostics.")
    parser.add_argument(
        "--max-fit-non-event-rows",
        type=int,
        default=100000,
        help="Maximum number of non-event rows to keep for bounded incremental and ridge fits.",
    )
    parser.add_argument(
        "--max-design-rows",
        type=int,
        default=50000,
        help="Maximum number of rows to use when building sampled design matrices for diagnostics.",
    )
    parser.add_argument(
        "--max-heatmap-rows",
        type=int,
        default=50000,
        help="Maximum number of rows to use for the neural correlation heatmap.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic progress.")


def run_diagnose_spp_neural_hazard_failure_command(args: argparse.Namespace) -> int:
    """Run the SPP neural hazard failure diagnostics command."""

    result = diagnose_spp_neural_hazard_failure(
        riskset_path=Path(args.riskset_path),
        models_dir=Path(args.models_dir) if args.models_dir else None,
        output_dir=Path(args.output_dir),
        run_ridge_diagnostic=bool(args.run_ridge_diagnostic),
        skip_incremental_fits=bool(args.skip_incremental_fits),
        max_fit_non_event_rows=int(args.max_fit_non_event_rows),
        max_design_rows=int(args.max_design_rows),
        max_heatmap_rows=int(args.max_heatmap_rows),
        verbose=bool(args.verbose),
    )
    print(result.report_path)
    return 0
