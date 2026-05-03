"""CLI command for the pooled FPP/SPP source-level DICS alpha/beta pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cas.source_dics import load_source_dics_config, run_source_dics_pipeline


def _parse_csv_argument(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def add_source_dics_fpp_spp_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``source-dics-fpp-spp`` command."""

    parser = subparsers.add_parser(
        "source-dics-fpp-spp",
        help="Run the pooled FPP/SPP source-level DICS alpha/beta power pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config/source_dics_fpp_spp_alpha_beta.yaml",
        help="Path to the source-level DICS YAML config.",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject list, e.g. sub-001,sub-002.",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Optional comma-separated band list, e.g. alpha,beta.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite saved filters or export files when supported.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the source DICS run.",
    )


def run_source_dics_fpp_spp_command(args: argparse.Namespace) -> int:
    """Run the pooled FPP/SPP source-level DICS alpha/beta pipeline."""

    config = load_source_dics_config(Path(args.config))
    result = run_source_dics_pipeline(
        config,
        subjects=_parse_csv_argument(args.subjects),
        bands=_parse_csv_argument(args.bands),
        overwrite=bool(args.overwrite),
        verbose=True if args.verbose else None,
    )
    print(result.summary_path)
    return 0
