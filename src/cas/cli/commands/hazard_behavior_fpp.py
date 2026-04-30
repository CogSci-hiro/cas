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
    parser.add_argument(
        "--include-quadratic-offset-timing",
        dest="include_quadratic_offset_timing",
        action="store_true",
    )
    parser.add_argument(
        "--no-include-quadratic-offset-timing",
        dest="include_quadratic_offset_timing",
        action="store_false",
    )
    parser.set_defaults(include_quadratic_offset_timing=True)
    parser.add_argument("--ipu-gap-threshold-s", type=float, default=0.300)
    parser.add_argument("--max-followup-s", type=float, default=6.0)
    parser.add_argument(
        "--episode-anchor",
        choices=["partner_ipu"],
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
    parser.set_defaults(require_partner_offset_before_fpp=False)
    parser.add_argument("--partner-offset-fpp-tolerance-s", type=float, default=0.020)
    parser.add_argument(
        "--overlapping-episode-strategy",
        choices=["exclude", "truncate", "keep"],
        default="keep",
    )
    parser.add_argument("--cluster-column", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-riskset", dest="save_riskset", action="store_true")
    parser.add_argument("--no-save-riskset", dest="save_riskset", action="store_false")
    parser.set_defaults(save_riskset=True)
    parser.add_argument(
        "--lag-grid-ms",
        default="0,50,100,150,200,300,500,700,1000",
        help="Comma-separated lag grid in milliseconds.",
    )
    parser.add_argument(
        "--fit-timing-control-models",
        dest="fit_timing_control_models",
        action="store_true",
        help=(
            "Fit partner-IPU anchored timing-control models with linear onset time, "
            "linear offset time, and a quadratic offset-time term."
        ),
    )
    parser.add_argument(
        "--no-fit-timing-control-models",
        dest="fit_timing_control_models",
        action="store_false",
    )
    parser.set_defaults(fit_timing_control_models=False)
    parser.add_argument(
        "--select-lags-with-timing-controls",
        dest="select_lags_with_timing_controls",
        action="store_true",
        help=(
            "Re-run behavioural lag selection against timing-controlled parent models and fit the "
            "resulting timing-controlled final model sequence."
        ),
    )
    parser.add_argument(
        "--no-select-lags-with-timing-controls",
        dest="select_lags_with_timing_controls",
        action="store_false",
    )
    parser.set_defaults(select_lags_with_timing_controls=False)
    parser.add_argument("--primary-information-rate-lag-ms", type=int, default=0)
    parser.add_argument("--primary-prop-expected-lag-ms", type=int, default=300)
    parser.add_argument("--run-r-glmm-lag-sweep", dest="run_r_glmm_lag_sweep", action="store_true")
    parser.add_argument("--no-run-r-glmm-lag-sweep", dest="run_r_glmm_lag_sweep", action="store_false")
    parser.set_defaults(run_r_glmm_lag_sweep=False)
    parser.add_argument(
        "--r-glmm-lag-grid-ms",
        default="0,50,100,150,200,300,500,700,1000",
        help="Comma-separated lag grid in milliseconds for the R GLMM lag sweep.",
    )
    parser.add_argument(
        "--r-glmm-include-quadratic-offset-timing",
        dest="r_glmm_include_quadratic_offset_timing",
        action="store_true",
    )
    parser.add_argument(
        "--no-r-glmm-include-quadratic-offset-timing",
        dest="r_glmm_include_quadratic_offset_timing",
        action="store_false",
    )
    parser.set_defaults(r_glmm_include_quadratic_offset_timing=True)
    parser.add_argument("--r-glmm-backend", choices=["glmmTMB", "glmer"], default="glmmTMB")
    parser.add_argument(
        "--r-glmm-include-run-random-effect",
        dest="r_glmm_include_run_random_effect",
        action="store_true",
    )
    parser.add_argument(
        "--no-r-glmm-include-run-random-effect",
        dest="r_glmm_include_run_random_effect",
        action="store_false",
    )
    parser.set_defaults(r_glmm_include_run_random_effect=False)
    parser.add_argument(
        "--r-glmm-prop-expected-mode",
        choices=["after_best_rate", "matched_lag"],
        default="after_best_rate",
    )
    parser.add_argument(
        "--r-glmm-include-prop-expected-in-final",
        dest="r_glmm_include_prop_expected_in_final",
        action="store_true",
    )
    parser.add_argument(
        "--no-r-glmm-include-prop-expected-in-final",
        dest="r_glmm_include_prop_expected_in_final",
        action="store_false",
    )
    parser.set_defaults(r_glmm_include_prop_expected_in_final=False)
    parser.add_argument("--save-lagged-feature-table", dest="save_lagged_feature_table", action="store_true")
    parser.add_argument("--no-save-lagged-feature-table", dest="save_lagged_feature_table", action="store_false")
    parser.set_defaults(save_lagged_feature_table=False)


def run_hazard_behavior_fpp_command(args: argparse.Namespace) -> int:
    """Run the behavioural hazard command."""

    surprisal_paths = resolve_surprisal_paths(args.surprisal)
    config = BehaviourHazardConfig(
        events_path=Path(args.events),
        surprisal_paths=tuple(surprisal_paths),
        out_dir=Path(args.out_dir),
        bin_size_s=float(args.bin_size_s),
        information_rate_window_s=float(args.information_rate_window_s),
        include_quadratic_offset_timing=bool(args.include_quadratic_offset_timing),
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
        fit_timing_control_models=bool(args.fit_timing_control_models or args.select_lags_with_timing_controls),
        select_lags_with_timing_controls=bool(args.select_lags_with_timing_controls),
        run_r_glmm_lag_sweep=bool(args.run_r_glmm_lag_sweep),
        r_glmm_lag_grid_ms=_parse_lag_grid_ms(args.r_glmm_lag_grid_ms),
        r_glmm_include_quadratic_offset_timing=bool(args.r_glmm_include_quadratic_offset_timing),
        r_glmm_backend=str(args.r_glmm_backend),
        r_glmm_include_run_random_effect=bool(args.r_glmm_include_run_random_effect),
        r_glmm_prop_expected_mode=str(args.r_glmm_prop_expected_mode),
        r_glmm_include_prop_expected_in_final=bool(args.r_glmm_include_prop_expected_in_final),
        primary_information_rate_lag_ms=int(args.primary_information_rate_lag_ms),
        primary_prop_expected_lag_ms=int(args.primary_prop_expected_lag_ms),
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
