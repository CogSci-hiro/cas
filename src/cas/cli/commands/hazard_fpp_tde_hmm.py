"""CLI command for the first-pass FPP hazard analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from cas.hazard import run_hazard_analysis
from cas.hazard.config import (
    NeuralInputConfig,
    NeuralPcaConfig,
    NeuralWindowConfig,
    load_hazard_analysis_config,
)
from cas.hazard_behavior.neural_lowlevel import run_neural_lowlevel_hazard_analysis
from cas.hazard_behavior.io import resolve_surprisal_paths


def add_hazard_fpp_tde_hmm_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``hazard-fpp-tde-hmm`` command."""

    parser = subparsers.add_parser(
        "hazard-fpp-tde-hmm",
        help="Run a first-pass pooled discrete-time hazard analysis anchored at partner onset using existing TDE-HMM outputs.",
    )
    parser.add_argument(
        "--config",
        default="config/hazard_fpp_tde_hmm.yaml",
        help="Path to the dedicated hazard-analysis config YAML.",
    )
    parser.add_argument("--neural-event-types", default=None, help="Comma-separated neural event types (fpp,spp).")
    parser.add_argument(
        "--neural-surprisal",
        default=None,
        help="Path/directory/glob for neural hazard surprisal TSVs.",
    )
    parser.add_argument(
        "--neural-lowlevel",
        default=None,
        help="Path/directory/glob for low-level neural TSVs.",
    )
    parser.add_argument("--neural-out-dir", default=None, help="Override neural output directory.")
    parser.add_argument("--neural-bin-size-s", type=float, default=None)
    parser.add_argument("--neural-max-followup-s", type=float, default=None)
    parser.add_argument("--neural-window-start-lag-s", type=float, default=None)
    parser.add_argument("--neural-window-end-lag-s", type=float, default=None)
    parser.add_argument("--select-neural-lags", dest="select_neural_lags", action="store_true")
    parser.add_argument("--no-select-neural-lags", dest="select_neural_lags", action="store_false")
    parser.set_defaults(select_neural_lags=None)
    parser.add_argument(
        "--neural-lag-grid-ms",
        default=None,
        help="Comma-separated lag windows in milliseconds, e.g. 50-250,100-500.",
    )
    parser.add_argument(
        "--neural-lag-selection-criterion",
        choices=["bic"],
        default=None,
        help="Lag-selection criterion for neural low-level models.",
    )
    parser.add_argument("--neural-null-permutations", type=int, default=None)
    parser.add_argument("--skip-spp-on-failure", dest="skip_spp_on_failure", action="store_true")
    parser.add_argument("--no-skip-spp-on-failure", dest="skip_spp_on_failure", action="store_false")
    parser.set_defaults(skip_spp_on_failure=None)
    parser.add_argument("--neural-pca-mode", choices=["count", "variance"], default=None)
    parser.add_argument("--neural-pca-count", type=int, default=None)
    parser.add_argument("--neural-pca-variance-threshold", type=float, default=None)


def run_hazard_fpp_tde_hmm_command(args: argparse.Namespace) -> int:
    """Run the first-pass FPP hazard analysis."""

    config = load_hazard_analysis_config(args.config)
    neural_config = config.neural
    if args.neural_event_types is not None:
        neural_config = replace(
            neural_config,
            event_types=tuple(
                value.strip().lower() for value in str(args.neural_event_types).split(",") if value.strip()
            ),
            enabled=True,
        )
    if args.neural_out_dir is not None:
        neural_config = replace(
            neural_config,
            out_dir=Path(str(args.neural_out_dir)),
            enabled=True,
        )
    if args.neural_surprisal is not None or args.neural_lowlevel is not None:
        surprisal_paths = (
            tuple(resolve_surprisal_paths(str(args.neural_surprisal)))
            if args.neural_surprisal is not None
            else (tuple(neural_config.input.surprisal_paths) if neural_config.input is not None else tuple())
        )
        lowlevel_paths = (
            tuple(resolve_surprisal_paths(str(args.neural_lowlevel)))
            if args.neural_lowlevel is not None
            else (tuple(neural_config.input.lowlevel_neural_paths) if neural_config.input is not None else tuple())
        )
        neural_config = replace(
            neural_config,
            input=NeuralInputConfig(
                surprisal_paths=tuple(Path(path) for path in surprisal_paths),
                lowlevel_neural_paths=tuple(Path(path) for path in lowlevel_paths),
            ),
            enabled=True,
        )
    if args.neural_bin_size_s is not None:
        neural_config = replace(neural_config, bin_size_s=float(args.neural_bin_size_s), enabled=True)
    if args.neural_max_followup_s is not None:
        neural_config = replace(
            neural_config,
            episode=replace(neural_config.episode, max_followup_s=float(args.neural_max_followup_s)),
            enabled=True,
        )
    if args.neural_window_start_lag_s is not None or args.neural_window_end_lag_s is not None:
        neural_config = replace(
            neural_config,
            window=NeuralWindowConfig(
                start_lag_s=(
                    float(args.neural_window_start_lag_s)
                    if args.neural_window_start_lag_s is not None
                    else neural_config.window.start_lag_s
                ),
                end_lag_s=(
                    float(args.neural_window_end_lag_s)
                    if args.neural_window_end_lag_s is not None
                    else neural_config.window.end_lag_s
                ),
                epsilon=neural_config.window.epsilon,
            ),
            enabled=True,
        )
    if (
        args.select_neural_lags is not None
        or args.neural_lag_grid_ms is not None
        or args.neural_lag_selection_criterion is not None
        or args.neural_null_permutations is not None
        or args.skip_spp_on_failure is not None
    ):
        neural_config = replace(
            neural_config,
            select_neural_lags=(
                bool(args.select_neural_lags)
                if args.select_neural_lags is not None
                else neural_config.select_neural_lags
            ),
            neural_lag_grid_ms=(
                _parse_neural_lag_grid_ms(args.neural_lag_grid_ms)
                if args.neural_lag_grid_ms is not None
                else neural_config.neural_lag_grid_ms
            ),
            neural_lag_selection_criterion=str(
                args.neural_lag_selection_criterion or neural_config.neural_lag_selection_criterion
            ),
            neural_null_permutations=(
                int(args.neural_null_permutations)
                if args.neural_null_permutations is not None
                else neural_config.neural_null_permutations
            ),
            skip_spp_on_failure=(
                bool(args.skip_spp_on_failure)
                if args.skip_spp_on_failure is not None
                else neural_config.skip_spp_on_failure
            ),
            enabled=True,
        )
    if (
        args.neural_pca_mode is not None
        or args.neural_pca_count is not None
        or args.neural_pca_variance_threshold is not None
    ):
        neural_config = replace(
            neural_config,
            pca=NeuralPcaConfig(
                mode=str(args.neural_pca_mode or neural_config.pca.mode),
                n_components=int(args.neural_pca_count or neural_config.pca.n_components),
                variance_threshold=float(
                    args.neural_pca_variance_threshold or neural_config.pca.variance_threshold
                ),
            ),
            enabled=True,
        )
    config = replace(config, neural=neural_config)
    if config.mode == "neural_lowlevel" or bool(config.neural.enabled):
        result = run_neural_lowlevel_hazard_analysis(config)
        print(result.output_dir)
        return 0
    result = run_hazard_analysis(config)
    print(result.output_dir)
    return 0


def _parse_neural_lag_grid_ms(value: str) -> tuple[tuple[int, int], ...]:
    windows: list[tuple[int, int]] = []
    for part in str(value).split(","):
        window_text = part.strip()
        if not window_text:
            continue
        pieces = window_text.split("-", 1)
        if len(pieces) != 2:
            raise ValueError("`--neural-lag-grid-ms` entries must look like start-end in milliseconds.")
        lag_start_ms = int(pieces[0].strip())
        lag_end_ms = int(pieces[1].strip())
        windows.append((lag_start_ms, lag_end_ms))
    if not windows:
        raise ValueError("`--neural-lag-grid-ms` must contain at least one lag window.")
    return tuple(windows)
