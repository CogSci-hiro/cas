"""CLI command for behavioural FPP hazard analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.hazard_behavior.config import BehaviourHazardConfig
from cas.hazard_behavior.io import resolve_surprisal_paths
from cas.hazard_behavior.neural_io import resolve_neural_feature_paths
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
    parser.add_argument("--max-followup-s", type=float, default=6.0)
    parser.add_argument(
        "--episode-anchor",
        choices=["partner_ipu", "legacy_fpp_previous_partner"],
        default="partner_ipu",
    )
    parser.add_argument("--include-censored", dest="include_censored", action="store_true")
    parser.add_argument("--include-censored-episodes", dest="include_censored", action="store_true")
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
    parser.add_argument(
        "--lag-grid-ms",
        default="0,100,200,300,500,700,1000",
        help="Comma-separated lag grid in milliseconds.",
    )
    parser.add_argument("--fit-lagged-models", dest="fit_lagged_models", action="store_true")
    parser.add_argument("--no-fit-lagged-models", dest="fit_lagged_models", action="store_false")
    parser.set_defaults(fit_lagged_models=True)
    parser.add_argument("--fit-primary-stat-tests", dest="fit_primary_stat_tests", action="store_true")
    parser.add_argument("--no-fit-primary-stat-tests", dest="fit_primary_stat_tests", action="store_false")
    parser.set_defaults(fit_primary_stat_tests=True)
    parser.add_argument(
        "--make-primary-publication-figures",
        dest="make_primary_publication_figures",
        action="store_true",
    )
    parser.add_argument(
        "--no-make-primary-publication-figures",
        dest="make_primary_publication_figures",
        action="store_false",
    )
    parser.set_defaults(make_primary_publication_figures=True)
    parser.add_argument(
        "--run-primary-leave-one-cluster",
        dest="run_primary_leave_one_cluster",
        action="store_true",
    )
    parser.add_argument(
        "--no-run-primary-leave-one-cluster",
        dest="run_primary_leave_one_cluster",
        action="store_false",
    )
    parser.set_defaults(run_primary_leave_one_cluster=False)
    parser.add_argument(
        "--fit-primary-behaviour-models",
        dest="fit_primary_behaviour_models",
        action="store_true",
        help=(
            "Fit the compact primary behavioural model sequence using local information rate and "
            "expected-relative cumulative information lagged by 300 ms."
        ),
    )
    parser.add_argument(
        "--no-fit-primary-behaviour-models",
        dest="fit_primary_behaviour_models",
        action="store_false",
    )
    parser.set_defaults(fit_primary_behaviour_models=True)
    parser.add_argument("--primary-information-rate-lag-ms", type=int, default=0)
    parser.add_argument("--primary-prop-expected-lag-ms", type=int, default=300)
    parser.add_argument(
        "--fit-neural-lowlevel-models",
        dest="fit_neural_lowlevel_models",
        action="store_true",
    )
    parser.add_argument(
        "--no-fit-neural-lowlevel-models",
        dest="fit_neural_lowlevel_models",
        action="store_false",
    )
    parser.set_defaults(fit_neural_lowlevel_models=False)
    parser.add_argument(
        "--neural-features",
        action="append",
        default=[],
        help="Path, directory, or glob for low-level neural TSV/CSV feature files. Repeatable.",
    )
    parser.add_argument("--neural-window-s", type=float, default=0.500)
    parser.add_argument("--neural-guard-s", type=float, default=0.100)
    parser.add_argument("--neural-pca-variance-threshold", type=float, default=0.90)
    parser.add_argument("--neural-pca-max-components", type=int, default=10)
    parser.add_argument(
        "--neural-feature-prefix",
        action="append",
        default=[],
        help="Repeatable prefix used to select neural feature columns, e.g. amp_, alpha_, beta_.",
    )
    parser.add_argument("--save-lagged-feature-table", dest="save_lagged_feature_table", action="store_true")
    parser.add_argument("--no-save-lagged-feature-table", dest="save_lagged_feature_table", action="store_false")
    parser.set_defaults(save_lagged_feature_table=False)


def run_hazard_behavior_fpp_command(args: argparse.Namespace) -> int:
    """Run the behavioural hazard command."""

    surprisal_paths = resolve_surprisal_paths(args.surprisal)
    neural_paths = resolve_neural_feature_paths(tuple(args.neural_features)) if args.neural_features else ()
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
        episode_anchor=str(args.episode_anchor),
        include_censored=bool(args.include_censored),
        unmatched_surprisal_strategy=str(args.unmatched_surprisal_strategy),
        token_availability=str(args.token_availability),
        expected_info_group=str(args.expected_info_group),
        target_fpp_label_prefix="FPP_",
        require_partner_offset_before_fpp=bool(args.require_partner_offset_before_fpp),
        partner_offset_fpp_tolerance_s=float(args.partner_offset_fpp_tolerance_s),
        overlapping_episode_strategy=str(args.overlapping_episode_strategy),
        cluster_column=args.cluster_column,
        overwrite=bool(args.overwrite),
        save_riskset=bool(args.save_riskset),
        lag_grid_ms=_parse_lag_grid_ms(args.lag_grid_ms),
        fit_primary_behaviour_models=bool(args.fit_primary_behaviour_models),
        fit_primary_stat_tests=bool(args.fit_primary_stat_tests),
        make_primary_publication_figures=bool(args.make_primary_publication_figures),
        run_primary_leave_one_cluster=bool(args.run_primary_leave_one_cluster),
        primary_information_rate_lag_ms=int(args.primary_information_rate_lag_ms),
        primary_prop_expected_lag_ms=int(args.primary_prop_expected_lag_ms),
        fit_neural_lowlevel_models=bool(args.fit_neural_lowlevel_models),
        neural_features=tuple(neural_paths),
        neural_window_s=float(args.neural_window_s),
        neural_guard_s=float(args.neural_guard_s),
        neural_pca_variance_threshold=float(args.neural_pca_variance_threshold),
        neural_pca_max_components=int(args.neural_pca_max_components),
        neural_feature_prefixes=tuple(args.neural_feature_prefix or ["amp_", "alpha_", "beta_"]),
        fit_lagged_models=bool(args.fit_lagged_models),
        save_lagged_feature_table=bool(args.save_lagged_feature_table),
    )
    result = run_behaviour_hazard_pipeline(config)
    print(result.out_dir)
    return 0


def _parse_lag_grid_ms(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",")]
    lags = [int(part) for part in parts if part]
    if not lags:
        raise ValueError("`--lag-grid-ms` must contain at least one integer.")
    return tuple(lags)
