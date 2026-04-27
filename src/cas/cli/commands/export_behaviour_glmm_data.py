"""CLI command for exporting behavioural GLMM-ready CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.r_export import export_behaviour_glmm_data_from_path


def add_export_behaviour_glmm_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``export-behaviour-glmm-data`` command."""

    parser = subparsers.add_parser(
        "export-behaviour-glmm-data",
        help="Export behavioural hazard risk-set data for downstream R GLMM fitting.",
    )
    parser.add_argument("--input-riskset", required=True, help="Path to the behavioural hazard risk-set table.")
    parser.add_argument(
        "--selected-lags-json",
        default=None,
        help="Path to behaviour_timing_control_selected_lags.json.",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path for the model-ready export.")
    parser.add_argument("--output-qc-json", required=True, help="Output JSON path for export QC metadata.")
    parser.add_argument("--information-rate-lag-ms", type=int, default=None)
    parser.add_argument("--expected-cumulative-info-lag-ms", type=int, default=None)
    parser.add_argument(
        "--lag-grid-ms",
        default="0,50,100,150,200,300,500,700,1000",
        help="Comma-separated lag grid in milliseconds for the R GLMM lag-sweep export.",
    )


def run_export_behaviour_glmm_data_command(args: argparse.Namespace) -> int:
    """Run the behavioural GLMM export command."""

    result = export_behaviour_glmm_data_from_path(
        input_riskset=Path(args.input_riskset),
        output_csv=Path(args.output_csv),
        output_qc_json=Path(args.output_qc_json),
        selected_lags_json=Path(args.selected_lags_json) if args.selected_lags_json else None,
        information_rate_lag_ms=args.information_rate_lag_ms,
        expected_cumulative_info_lag_ms=args.expected_cumulative_info_lag_ms,
        lag_grid_ms=_parse_lag_grid_ms(args.lag_grid_ms),
    )
    print(result.output_csv)
    return 0


def _parse_lag_grid_ms(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",")]
    lags = [int(part) for part in parts if part]
    if not lags:
        raise ValueError("`--lag-grid-ms` must contain at least one integer.")
    return tuple(lags)
