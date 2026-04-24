"""CLI command for behavioural FPP hazard analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.io import resolve_surprisal_paths
from cas.hazard_behavior.pipeline import run_behaviour_hazard_pipeline


def add_hazard_behavior_fpp_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``hazard-behavior-fpp`` command."""

    parser = subparsers.add_parser(
        "hazard-behavior-fpp",
        help="Run the first-pass behavioural discrete-time hazard analysis for FPP onset.",
    )
    parser.add_argument("--events", required=True, help="Path to the events CSV.")
    parser.add_argument("--surprisal", required=True, help="Path, directory, or glob for surprisal TSV files.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--bin-size-s", type=float, default=0.050)
    parser.add_argument("--information-rate-window-s", type=float, default=0.500)
    parser.add_argument("--baseline-spline-df", type=int, default=6)
    parser.add_argument("--baseline-spline-degree", type=int, default=3)
    parser.add_argument("--ipu-gap-threshold-s", type=float, default=0.300)
    parser.add_argument("--max-followup-s", type=float, default=5.0)
    parser.add_argument("--include-censored", dest="include_censored", action="store_true")
    parser.add_argument("--event-positive-only", dest="include_censored", action="store_false")
    parser.set_defaults(include_censored=True)
    parser.add_argument(
        "--unmatched-surprisal-strategy",
        choices=["drop", "zero", "keep_nan"],
        default="drop",
    )
    parser.add_argument("--token-availability", choices=["onset", "offset"], default="onset")
    parser.add_argument(
        "--expected-info-group",
        choices=["partner_ipu_class", "partner_role", "global"],
        default="partner_ipu_class",
    )
    parser.add_argument(
        "--require-partner-offset-before-fpp",
        dest="require_partner_offset_before_fpp",
        action="store_true",
    )
    parser.add_argument(
        "--allow-partner-offset-after-fpp",
        dest="require_partner_offset_before_fpp",
        action="store_false",
    )
    parser.set_defaults(require_partner_offset_before_fpp=True)
    parser.add_argument("--partner-offset-fpp-tolerance-s", type=float, default=0.020)
    parser.add_argument(
        "--overlapping-episode-strategy",
        choices=["exclude", "truncate", "keep"],
        default="exclude",
    )
    parser.add_argument("--cluster-column", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-riskset", dest="save_riskset", action="store_true")
    parser.add_argument("--no-save-riskset", dest="save_riskset", action="store_false")
    parser.set_defaults(save_riskset=True)


def run_hazard_behavior_fpp_command(args: argparse.Namespace) -> int:
    """Run the behavioural hazard command."""

    surprisal_paths = resolve_surprisal_paths(args.surprisal)
    config = BehaviourHazardConfig(
        events_path=Path(args.events),
        surprisal_paths=tuple(surprisal_paths),
        out_dir=Path(args.out_dir),
        bin_size_s=float(args.bin_size_s),
        information_rate_window_s=float(args.information_rate_window_s),
        baseline_spline_df=int(args.baseline_spline_df),
        baseline_spline_degree=int(args.baseline_spline_degree),
        ipu_gap_threshold_s=float(args.ipu_gap_threshold_s),
        max_followup_s=float(args.max_followup_s),
        include_censored=bool(args.include_censored),
        unmatched_surprisal_strategy=str(args.unmatched_surprisal_strategy),
        token_availability=str(args.token_availability),
        expected_info_group=str(args.expected_info_group),
        require_partner_offset_before_fpp=bool(args.require_partner_offset_before_fpp),
        partner_offset_fpp_tolerance_s=float(args.partner_offset_fpp_tolerance_s),
        overlapping_episode_strategy=str(args.overlapping_episode_strategy),
        cluster_column=args.cluster_column,
        overwrite=bool(args.overwrite),
        save_riskset=bool(args.save_riskset),
    )
    result = run_behaviour_hazard_pipeline(config)
    print(result.out_dir)
    return 0
