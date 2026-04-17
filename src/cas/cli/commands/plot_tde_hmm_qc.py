"""CLI command for plotting post-fit TDE-HMM QC outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from matplotlib import pyplot as plt

from cas.hmm.qc import _run_all_qc_plots_from_inputs, load_qc_inputs


def add_plot_tde_hmm_qc_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``plot-tde-hmm-qc`` command."""

    parser = subparsers.add_parser(
        "plot-tde-hmm-qc",
        help="Render post-fit QC plots for a saved causal TDE-HMM output directory.",
    )
    parser.add_argument("--output-dir", required=True, help="HMM output directory containing saved artifacts.")
    parser.add_argument("--selected-k", type=int, default=None, help="Optional selected K override.")
    parser.add_argument("--examples", type=int, default=3, help="Number of representative example runs to plot.")
    parser.add_argument(
        "--fpp-events",
        default=None,
        help="Optional CSV path overriding any fpp_events.csv in the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing qc/ subdirectory if it already exists.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output raster DPI for saved figures.")


def run_plot_tde_hmm_qc_command(args: argparse.Namespace) -> int:
    """Run the TDE-HMM QC plotting workflow."""

    output_dir = Path(args.output_dir).resolve()
    qc_output_dir = output_dir / "qc"
    if qc_output_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(
                f"QC directory already exists: {qc_output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(qc_output_dir)

    plt.rcParams["savefig.dpi"] = int(args.dpi)
    qc_inputs = load_qc_inputs(output_dir=output_dir, selected_k=args.selected_k)
    if args.fpp_events is not None:
        import pandas as pd

        fpp_events_path = Path(args.fpp_events).resolve()
        qc_inputs.fpp_events = pd.read_csv(fpp_events_path)
        qc_inputs.paths["fpp_events_override"] = fpp_events_path
    artifacts = _run_all_qc_plots_from_inputs(
        qc_inputs,
        qc_output_dir,
        n_examples=int(args.examples),
    )

    for artifact in artifacts.values():
        if artifact is None:
            continue
        if isinstance(artifact, list):
            for path in artifact:
                print(path)
            continue
        print(artifact)
    return 0
