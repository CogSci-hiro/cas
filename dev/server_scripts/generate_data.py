#!/usr/bin/env python3
"""
Batch-export time-aligned surprisal and entropy tables from pipeline outputs.

Expected input structure
------------------------
root/
  dyad-001/
    token_uncertainty.csv
    ...
  dyad-002/
    token_uncertainty.csv
    ...
  ...

Each dyad directory is the output of the uncertainty pipeline.

For each dyad directory, this script writes:
- timed_surprisal.csv
- timed_shannon_entropy.csv
- timed_renyi_entropy.csv
- export_metadata.json

to:

out_root/
  dyad-001/
    timed_surprisal.csv
    ...
  dyad-002/
    timed_surprisal.csv
    ...
  ...

Rényi handling
--------------
If the requested alpha grid already exists in token_uncertainty.csv, those
columns are reused.

Otherwise, the script tries to recompute the requested Rényi entropies from
full_logprobs_chunks/, which requires that the original pipeline run used:
    --save_full_logprobs_npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ###########################################################################
# CLI
# ###########################################################################

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory containing dyad-* pipeline output subdirectories.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory where exported tables will be written.",
    )
    parser.add_argument(
        "--renyi_alpha_start",
        type=float,
        default=0.25,
        help="Start of uniform Rényi alpha grid.",
    )
    parser.add_argument(
        "--renyi_alpha_end",
        type=float,
        default=8.0,
        help="End of uniform Rényi alpha grid.",
    )
    parser.add_argument(
        "--renyi_alpha_step",
        type=float,
        default=0.25,
        help="Step size of uniform Rényi alpha grid.",
    )
    parser.add_argument(
        "--dyad_glob",
        type=str,
        default="dyad-*",
        help="Glob pattern for dyad subdirectories. Default: %(default)s",
    )
    return parser.parse_args()


# ###########################################################################
# Helpers
# ###########################################################################

def make_alpha_grid(alpha_start: float, alpha_end: float, alpha_step: float) -> list[float]:
    """Create a uniform Rényi alpha grid."""
    if alpha_step <= 0:
        raise ValueError("renyi_alpha_step must be > 0")
    if alpha_end < alpha_start:
        raise ValueError("renyi_alpha_end must be >= renyi_alpha_start")

    alpha_values: list[float] = []
    current_alpha = alpha_start
    tolerance = alpha_step * 1e-9

    while current_alpha <= alpha_end + tolerance:
        if not math.isclose(
            current_alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=max(1e-9, alpha_step * 1e-6),
        ):
            alpha_values.append(round(current_alpha, 10))
        current_alpha += alpha_step

    if len(alpha_values) == 0:
        raise ValueError("Alpha grid is empty after excluding alpha=1")

    return alpha_values


def format_alpha_for_column(alpha: float) -> str:
    """Format alpha to match pipeline column naming."""
    alpha_text = f"{float(alpha):g}"
    return alpha_text.replace("-", "m").replace(".", "p")


def required_file(path: Path) -> Path:
    """Ensure a required file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def build_base_export_columns(token_df: pd.DataFrame) -> list[str]:
    """Return base identity columns for export."""
    preferred_columns = [
        "run",
        "speaker",
        "rendered_speaker",
        "start",
        "end",
        "token",
        "rendered_text",
        "token_kind",
        "annotation_index",
    ]
    return [column_name for column_name in preferred_columns if column_name in token_df.columns]


# ###########################################################################
# Rényi recomputation
# ###########################################################################

def renyi_columns_present(token_df: pd.DataFrame, alpha_values: Iterable[float]) -> bool:
    """Return whether all requested Rényi columns already exist."""
    for alpha in alpha_values:
        alpha_label = format_alpha_for_column(alpha)
        required_column = f"token_renyi_entropy_alpha_{alpha_label}_nats"
        if required_column not in token_df.columns:
            return False
    return True


def compute_renyi_from_log_probs(log_probs: np.ndarray, alpha: float) -> np.ndarray:
    """Compute Rényi entropy from full log-probability rows."""
    if math.isclose(alpha, 1.0):
        raise ValueError("Alpha=1 is Shannon entropy, not Rényi entropy.")

    scaled_log_probs = alpha * log_probs
    max_scaled = np.max(scaled_log_probs, axis=1, keepdims=True)
    power_sum_log = (
        np.log(np.sum(np.exp(scaled_log_probs - max_scaled), axis=1))
        + max_scaled[:, 0]
    )
    return power_sum_log / (1.0 - alpha)


def load_positionwise_renyi_from_chunks(
    full_logprobs_dir: Path,
    alpha_values: list[float],
) -> dict[float, dict[int, float]]:
    """Compute requested Rényi entropies per model position from saved chunks."""
    chunk_paths = sorted(full_logprobs_dir.glob("logprobs_chunk_*.npz"))
    if len(chunk_paths) == 0:
        raise FileNotFoundError(f"No full logprob chunks found in: {full_logprobs_dir}")

    renyi_by_alpha: dict[float, dict[int, float]] = {
        float(alpha): {}
        for alpha in alpha_values
    }

    for chunk_path in chunk_paths:
        with np.load(chunk_path) as chunk:
            absolute_positions = chunk["absolute_positions"]
            log_probs = chunk["log_probs"].astype(np.float32, copy=False)

        for alpha in alpha_values:
            renyi_values = compute_renyi_from_log_probs(log_probs=log_probs, alpha=float(alpha))
            alpha_map = renyi_by_alpha[float(alpha)]
            for absolute_position, renyi_value in zip(absolute_positions, renyi_values):
                alpha_map[int(absolute_position)] = float(renyi_value)

    return renyi_by_alpha


def aggregate_positionwise_renyi_to_annotations(
    token_df: pd.DataFrame,
    renyi_by_alpha: dict[float, dict[int, float]],
) -> pd.DataFrame:
    """Aggregate positionwise Rényi entropies back to annotation rows."""
    base_columns = build_base_export_columns(token_df)
    out_df = token_df[base_columns].copy()

    for alpha, alpha_map in renyi_by_alpha.items():
        values_nats: list[float] = []
        values_bits: list[float] = []

        for model_token_indices_json in token_df["model_token_indices_json"].astype(str):
            model_token_indices = json.loads(model_token_indices_json)
            model_token_indices = [int(index) for index in model_token_indices]

            if len(model_token_indices) == 0:
                values_nats.append(np.nan)
                values_bits.append(np.nan)
                continue

            per_position_values = [
                alpha_map[index]
                for index in model_token_indices
                if index in alpha_map
            ]

            if len(per_position_values) == 0:
                values_nats.append(np.nan)
                values_bits.append(np.nan)
                continue

            mean_value_nats = float(np.mean(per_position_values))
            values_nats.append(mean_value_nats)
            values_bits.append(mean_value_nats / math.log(2.0))

        alpha_label = format_alpha_for_column(alpha)
        out_df[f"renyi_entropy_alpha_{alpha_label}_nats"] = values_nats
        out_df[f"renyi_entropy_alpha_{alpha_label}_bits"] = values_bits

    return out_df


# ###########################################################################
# Per-dyad export
# ###########################################################################

def export_one_dyad(
    pipeline_output_dir: Path,
    out_dir: Path,
    alpha_values: list[float],
) -> dict[str, object]:
    """Export one dyad directory.

    Parameters
    ----------
    pipeline_output_dir
        One dyad pipeline-output directory.
    out_dir
        Output directory for exported tables.
    alpha_values
        Requested Rényi alpha grid.

    Returns
    -------
    dict[str, object]
        Metadata summary for this dyad.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    token_df = pd.read_csv(required_file(pipeline_output_dir / "token_uncertainty.csv")).copy()
    base_columns = build_base_export_columns(token_df)

    # i) Surprisal
    surprisal_columns = [
        column_name
        for column_name in [
            "token_surprisal_nats",
            "token_surprisal_bits",
            "n_model_pieces",
            "model_token_indices_json",
        ]
        if column_name in token_df.columns
    ]
    token_df[base_columns + surprisal_columns].to_csv(
        out_dir / "timed_surprisal.csv",
        index=False,
    )

    # ii) Shannon entropy
    shannon_columns = [
        column_name
        for column_name in [
            "token_shannon_entropy_nats",
            "token_shannon_entropy_bits",
            "n_model_pieces",
            "model_token_indices_json",
        ]
        if column_name in token_df.columns
    ]
    token_df[base_columns + shannon_columns].to_csv(
        out_dir / "timed_shannon_entropy.csv",
        index=False,
    )

    # iii) Rényi entropy
    if renyi_columns_present(token_df=token_df, alpha_values=alpha_values):
        renyi_columns: list[str] = []
        for alpha in alpha_values:
            alpha_label = format_alpha_for_column(alpha)
            renyi_columns.append(f"token_renyi_entropy_alpha_{alpha_label}_nats")
            bits_column = f"token_renyi_entropy_alpha_{alpha_label}_bits"
            if bits_column in token_df.columns:
                renyi_columns.append(bits_column)

        token_df[base_columns + renyi_columns].to_csv(
            out_dir / "timed_renyi_entropy.csv",
            index=False,
        )
        renyi_source = "precomputed_token_columns"
    else:
        full_logprobs_dir = required_file(pipeline_output_dir / "full_logprobs_chunks")
        renyi_by_alpha = load_positionwise_renyi_from_chunks(
            full_logprobs_dir=full_logprobs_dir,
            alpha_values=alpha_values,
        )
        renyi_df = aggregate_positionwise_renyi_to_annotations(
            token_df=token_df,
            renyi_by_alpha=renyi_by_alpha,
        )
        renyi_df.to_csv(out_dir / "timed_renyi_entropy.csv", index=False)
        renyi_source = "recomputed_from_full_logprobs"

    metadata = {
        "dyad_dir": str(pipeline_output_dir),
        "export_dir": str(out_dir),
        "n_rows": int(len(token_df)),
        "renyi_alpha_grid": alpha_values,
        "renyi_source": renyi_source,
    }
    (out_dir / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


# ###########################################################################
# Main
# ###########################################################################

def main() -> None:
    """Main entry point."""
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    alpha_values = make_alpha_grid(
        alpha_start=args.renyi_alpha_start,
        alpha_end=args.renyi_alpha_end,
        alpha_step=args.renyi_alpha_step,
    )

    dyad_dirs = sorted(
        path
        for path in args.input_root.glob(args.dyad_glob)
        if path.is_dir()
    )

    if len(dyad_dirs) == 0:
        raise FileNotFoundError(
            f"No dyad directories matching {args.dyad_glob!r} found under {args.input_root}"
        )

    batch_summary: list[dict[str, object]] = []

    for dyad_dir in dyad_dirs:
        print(f"Processing {dyad_dir.name} ...")
        out_dir = args.output_root / dyad_dir.name
        metadata = export_one_dyad(
            pipeline_output_dir=dyad_dir,
            out_dir=out_dir,
            alpha_values=alpha_values,
        )
        batch_summary.append(metadata)

    (args.output_root / "batch_export_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Wrote exports under: {args.output_root}")
    print(f"Wrote batch summary: {args.output_root / 'batch_export_summary.json'}")


if __name__ == "__main__":
    main()