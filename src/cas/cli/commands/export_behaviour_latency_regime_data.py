"""CLI command for exporting event-only behavioural latency-regime data."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.latency_regime_export import export_behaviour_latency_regime_data_from_path


def add_export_behaviour_latency_regime_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``export-behaviour-latency-regime-data`` command."""

    parser = subparsers.add_parser(
        "export-behaviour-latency-regime-data",
        help="Export event-only behavioural latency data for the exploratory Stan latency-regime analysis.",
    )
    parser.add_argument("--input-riskset", required=True, help="Path to the behavioural hazard risk-set table.")
    parser.add_argument(
        "--selected-lags-json",
        default="results/hazard_behavior/models/behaviour_timing_control_selected_lags.json",
        help="Path to behaviour_timing_control_selected_lags.json.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/hazard_behavior/latency_regime/behaviour_latency_regime_data.csv",
        help="Output CSV path for the event-only latency-regime export.",
    )
    parser.add_argument(
        "--output-qc-json",
        default="results/hazard_behavior/latency_regime/behaviour_latency_regime_export_qc.json",
        help="Output JSON path for latency-regime export QC metadata.",
    )
    parser.add_argument("--information-rate-lag-ms", type=int, default=None)
    parser.add_argument("--expected-cumulative-info-lag-ms", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print latency-regime export progress.")


def run_export_behaviour_latency_regime_data_command(args: argparse.Namespace) -> int:
    """Run the behavioural latency-regime export command."""

    result = export_behaviour_latency_regime_data_from_path(
        input_riskset=Path(args.input_riskset),
        output_csv=Path(args.output_csv),
        output_qc_json=Path(args.output_qc_json),
        selected_lags_json=Path(args.selected_lags_json) if args.selected_lags_json else None,
        information_rate_lag_ms=args.information_rate_lag_ms,
        expected_cumulative_info_lag_ms=args.expected_cumulative_info_lag_ms,
        verbose=bool(args.verbose),
    )
    print(result.output_csv)
    return 0
