"""CLI command for publication-oriented source DICS plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.source_dics.config import load_source_dics_config
from cas.source_dics.plotting import run_source_dics_plotting


def _parse_csv_argument(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def add_plot_source_dics_fpp_spp_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``plot-source-dics-fpp-spp`` command."""

    parser = subparsers.add_parser(
        "plot-source-dics-fpp-spp",
        help="Generate publication-oriented source DICS lmeEEG surface/ROI figures.",
    )
    parser.add_argument(
        "--config",
        default="config/induced/source_localisation.yaml",
        help="Path to the source-level DICS YAML config.",
    )
    parser.add_argument(
        "--predictors",
        default=None,
        help="Optional comma-separated predictor list.",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Optional comma-separated frequency band list.",
    )
    parser.add_argument(
        "--time-summaries",
        default=None,
        help="Optional comma-separated time summary list (e.g. pre_event_full_mean).",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Generate only surface figures.",
    )
    parser.add_argument(
        "--roi-only",
        action="store_true",
        help="Generate only ROI summaries and ROI plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plotting outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def run_plot_source_dics_fpp_spp_command(args: argparse.Namespace) -> int:
    """Run source-level DICS plotting from normalized lmeEEG statistics."""

    config = load_source_dics_config(Path(args.config))
    index_path = run_source_dics_plotting(
        config,
        predictors=_parse_csv_argument(args.predictors),
        bands=_parse_csv_argument(args.bands),
        time_summaries=_parse_csv_argument(args.time_summaries),
        surface_only=bool(args.surface_only),
        roi_only=bool(args.roi_only),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )
    print(index_path)
    return 0
