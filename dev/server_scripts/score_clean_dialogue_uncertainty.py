#!/usr/bin/env python3
"""
Score clean dialogue uncertainty from SPPAS-derived tokens.

What this version changes
-------------------------
1. Fixes bfloat16 full-logprob saving by casting to float32 before NumPy export.
2. Treats each `run` as a separate conversation context.
3. Preserves original SPPAS token rows and timestamps while scoring only the
   clean LM-rendered text.
4. Supports tiny-model smoke tests and full 7B runs with the same interface.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "OpenLLM-France/Claire-Mistral-7B-0.1"
DEFAULT_MAX_NEWLINE_GAP_S = 0.75
DEFAULT_STRIDE = 512
DEFAULT_RENYI_ALPHAS: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    2.0,
    4.0,
    8.0,
    16.0,
)

SPACELESS_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "…", "%"}
APOSTROPHE_ENDINGS = ("'", "’")


@dataclass
class SpanRecord:
    """Character span of one inserted text fragment."""
    kind: str
    start_char: int
    end_char: int
    annotation_index: Optional[int] = None
    piece_index: Optional[int] = None
    payload: Optional[dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--tokens_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--newline_gap_s", type=float, default=DEFAULT_MAX_NEWLINE_GAP_S)
    parser.add_argument(
        "--speaker_prefix_mode",
        type=str,
        default="alphabetic",
        choices=["alphabetic", "original"],
    )
    parser.add_argument("--bos_token", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--renyi_alphas",
        type=float,
        nargs="*",
        default=list(DEFAULT_RENYI_ALPHAS),
    )
    parser.add_argument("--save_full_logprobs_npz", action="store_true")
    parser.add_argument(
        "--full_logprobs_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )

    return parser.parse_args()


def load_tokens_csv(tokens_csv: Path) -> pd.DataFrame:
    """Load and validate the token CSV."""
    tokens_df = pd.read_csv(tokens_csv).copy()

    required_columns = {
        "token",
        "speaker",
        "start",
        "end",
        "render_for_lm",
        "rendered_text",
        "rendered_piece_count",
        "rendered_pieces_json",
    }
    missing_columns = required_columns.difference(tokens_df.columns)
    if missing_columns:
        raise ValueError(f"tokens_csv is missing required columns: {sorted(missing_columns)}")

    tokens_df["annotation_index"] = np.arange(len(tokens_df), dtype=int)
    tokens_df["start"] = pd.to_numeric(tokens_df["start"], errors="raise")
    tokens_df["end"] = pd.to_numeric(tokens_df["end"], errors="raise")

    if "run" not in tokens_df.columns:
        tokens_df["run"] = 1

    sort_columns = ["run", "start", "end", "annotation_index"]
    tokens_df = tokens_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    tokens_df["annotation_index"] = np.arange(len(tokens_df), dtype=int)

    return tokens_df


def sanitize_speaker_name(name: str) -> str:
    """Sanitize speaker names for bracketed prompt format."""
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    cleaned = cleaned.replace("]", ")").replace("[", "(")
    return cleaned or "Speaker"


def number_to_letters(index: int) -> str:
    """Convert a zero-based integer to A, B, ..., Z, AA, AB, ..."""
    if index < 0:
        raise ValueError("index must be non-negative")

    letters: list[str] = []
    value = index

    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1

    return "".join(reversed(letters))


def canonicalize_speaker_labels(speakers: Sequence[str], mode: str) -> dict[str, str]:
    """Map raw speaker labels to prompt speaker labels."""
    unique_speakers = list(dict.fromkeys(str(speaker) for speaker in speakers))

    if mode == "original":
        return {speaker: sanitize_speaker_name(speaker) for speaker in unique_speakers}

    return {speaker: f"Speaker{number_to_letters(index)}" for index, speaker in enumerate(unique_speakers)}


def needs_space_before(token_text: str) -> bool:
    """Return whether a space should precede a rendered token piece."""
    if token_text in SPACELESS_BEFORE:
        return False
    if token_text.startswith("'") or token_text.startswith("’"):
        return False
    return True


def append_text(
    parts: list[str],
    spans: list[SpanRecord],
    text: str,
    kind: str,
    annotation_index: Optional[int] = None,
    piece_index: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    """Append text to the prompt and register its character span."""
    if text == "":
        return

    start_char = sum(len(part) for part in parts)
    parts.append(text)
    end_char = start_char + len(text)

    spans.append(
        SpanRecord(
            kind=kind,
            start_char=start_char,
            end_char=end_char,
            annotation_index=annotation_index,
            piece_index=piece_index,
            payload=payload,
        )
    )


def build_prompt_from_tokens(
    tokens_df: pd.DataFrame,
    speaker_prefix_mode: str,
    newline_gap_s: float,
) -> tuple[str, list[SpanRecord], pd.DataFrame, dict[str, str], pd.DataFrame]:
    """
    Render clean LM text from token rows while preserving annotation mapping.

    Each `run` is treated as a separate conversation context.
    """
    tokens_df = tokens_df.copy()

    speaker_map = canonicalize_speaker_labels(
        tokens_df["speaker"].astype(str).tolist(),
        mode=speaker_prefix_mode,
    )
    tokens_df["rendered_speaker"] = tokens_df["speaker"].astype(str).map(speaker_map)

    parts: list[str] = []
    spans: list[SpanRecord] = []
    piece_rows: list[dict[str, Any]] = []

    previous_run: Optional[int] = None
    previous_speaker: Optional[str] = None
    previous_end: Optional[float] = None
    previous_rendered_piece: Optional[str] = None
    at_line_start = True

    for row in tokens_df.itertuples(index=False):
        run_id = int(row.run)
        token_text = str(row.token)
        speaker = str(row.speaker)
        start_time = float(row.start)
        end_time = float(row.end)
        annotation_index = int(row.annotation_index)
        rendered_speaker = str(row.rendered_speaker)

        rendered_pieces = json.loads(str(row.rendered_pieces_json))
        rendered_pieces = [str(piece) for piece in rendered_pieces if str(piece) != ""]

        is_silence = token_text == "#"
        run_changed = previous_run is None or run_id != previous_run
        speaker_changed = previous_speaker is None or speaker != previous_speaker
        silence_gap_s = None if previous_end is None or run_changed else max(0.0, start_time - previous_end)

        force_newline = bool(
            run_changed
            or speaker_changed
            or (silence_gap_s is not None and silence_gap_s >= newline_gap_s)
            or is_silence
        )

        if run_changed and parts:
            if not parts[-1].endswith("\n"):
                append_text(parts, spans, "\n", "separator", payload={"reason": "run_boundary"})
            append_text(parts, spans, "\n", "separator", payload={"reason": "run_boundary_blank_line"})
            at_line_start = True
            previous_rendered_piece = None
            previous_end = None
            previous_speaker = None

        elif force_newline and parts and not parts[-1].endswith("\n"):
            append_text(parts, spans, "\n", "separator", payload={"reason": "turn_or_gap"})
            at_line_start = True

        if len(rendered_pieces) == 0:
            previous_run = run_id
            previous_end = end_time
            previous_speaker = speaker
            previous_rendered_piece = None
            continue

        if at_line_start:
            speaker_prefix = f"[{rendered_speaker}:] "
            append_text(
                parts,
                spans,
                speaker_prefix,
                "speaker_prefix",
                payload={"run": run_id, "speaker": speaker, "rendered_speaker": rendered_speaker},
            )
            at_line_start = False
            previous_rendered_piece = None

        for piece_index, piece_text in enumerate(rendered_pieces):
            if (
                previous_rendered_piece is not None
                and needs_space_before(piece_text)
                and not previous_rendered_piece.endswith(APOSTROPHE_ENDINGS)
            ):
                append_text(parts, spans, " ", "separator", payload={"reason": "intra_turn_space"})

            append_text(
                parts,
                spans,
                piece_text,
                "annotation_piece",
                annotation_index=annotation_index,
                piece_index=piece_index,
                payload={
                    "run": run_id,
                    "original_token": token_text,
                    "speaker": speaker,
                    "rendered_speaker": rendered_speaker,
                    "start": start_time,
                    "end": end_time,
                },
            )

            piece_rows.append(
                {
                    "run": run_id,
                    "annotation_index": annotation_index,
                    "piece_index": piece_index,
                    "original_token": token_text,
                    "rendered_piece": piece_text,
                    "speaker": speaker,
                    "rendered_speaker": rendered_speaker,
                    "start": start_time,
                    "end": end_time,
                }
            )

            previous_rendered_piece = piece_text

        previous_run = run_id
        previous_speaker = speaker
        previous_end = end_time

    prompt_text = "".join(parts)
    pieces_df = pd.DataFrame(piece_rows)

    return prompt_text, spans, tokens_df, speaker_map, pieces_df


def resolve_device(device: str) -> str:
    """Resolve requested device to an available backend."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    if device == "mps" and not (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available.")

    return device


def choose_torch_dtype(dtype_name: str, resolved_device: str) -> Optional[torch.dtype]:
    """Choose the torch dtype to use for model loading."""
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if resolved_device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return None


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype_name: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, str]:
    """Load tokenizer and model."""
    resolved_device = resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = choose_torch_dtype(dtype_name, resolved_device)

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if resolved_device != "cpu":
        model = model.to(resolved_device)

    model.eval()
    return tokenizer, model, resolved_device


def tokenize_prompt(tokenizer: Any, prompt_text: str, bos_token: bool) -> dict[str, Any]:
    """Tokenize prompt text and return token IDs plus offset mapping."""
    text_for_tokenization = prompt_text

    if bos_token and tokenizer.bos_token is not None:
        text_for_tokenization = tokenizer.bos_token + prompt_text

    encoded = tokenizer(
        text_for_tokenization,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    if bos_token and tokenizer.bos_token is not None:
        bos_length = len(tokenizer.bos_token)
        adjusted_offsets: list[tuple[int, int]] = []
        for start_char, end_char in encoded["offset_mapping"]:
            adjusted_offsets.append((max(0, start_char - bos_length), max(0, end_char - bos_length)))
        encoded["offset_mapping"] = adjusted_offsets

    return encoded


def infer_max_length(model: Any, tokenizer: Any, user_max_length: Optional[int]) -> int:
    """Infer usable model context length."""
    if user_max_length is not None:
        return int(user_max_length)

    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "sliding_window", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    candidates = [
        int(candidate)
        for candidate in candidates
        if candidate is not None and int(candidate) < 1_000_000
    ]

    if not candidates:
        return 2048

    return max(candidates)


def compute_shannon_entropy_from_log_probs(
    log_probs: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Compute Shannon entropy from log-probabilities."""
    return -(probs * log_probs).sum(dim=-1)


def compute_renyi_entropy_from_probs(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute Rényi entropy of order alpha from probabilities."""
    if math.isclose(alpha, 1.0):
        raise ValueError("Rényi alpha=1 is undefined here; use Shannon entropy instead.")
    power_sum = torch.sum(torch.pow(probs, alpha), dim=-1)
    return torch.log(power_sum) / (1.0 - alpha)


def infer_run_boundaries(tokens_df: pd.DataFrame, spans: Sequence[SpanRecord], offset_mapping: Sequence[Sequence[int]]) -> list[int]:
    """Infer model-token run boundaries from annotation spans."""
    annotation_run_map = {
        int(row.annotation_index): int(row.run)
        for row in tokens_df[["annotation_index", "run"]].itertuples(index=False)
    }

    candidate_boundaries: list[int] = []
    current_run = None

    ordered_piece_spans = [
        span for span in spans
        if span.kind == "annotation_piece" and span.annotation_index is not None
    ]

    for span in ordered_piece_spans:
        run_id = annotation_run_map[int(span.annotation_index)]
        if current_run is None:
            current_run = run_id
            continue

        if run_id != current_run:
            boundary_index = None
            for model_token_index, (offset_start, offset_end) in enumerate(offset_mapping):
                if int(offset_end) <= int(span.start_char):
                    continue
                boundary_index = model_token_index
                break

            if boundary_index is not None:
                candidate_boundaries.append(int(boundary_index))
            current_run = run_id

    return sorted(set(candidate_boundaries))


@torch.inference_mode()
def compute_distributional_metrics(
    model: Any,
    input_ids: Sequence[int],
    max_length: int,
    stride: int,
    device: str,
    renyi_alphas: Sequence[float],
    save_full_logprobs_npz: bool,
    full_logprobs_dtype: str,
    output_dir: Path,
    run_boundaries: Sequence[int],
) -> dict[str, Any]:
    """Compute surprisal and entropy metrics per model position with run resets."""
    sequence_length = len(input_ids)

    observed_logprob = np.full(sequence_length, np.nan, dtype=np.float64)
    surprisal_nats = np.full(sequence_length, np.nan, dtype=np.float64)
    shannon_entropy_nats = np.full(sequence_length, np.nan, dtype=np.float64)

    renyi_entropy_arrays: dict[float, np.ndarray] = {
        float(alpha): np.full(sequence_length, np.nan, dtype=np.float64)
        for alpha in renyi_alphas
    }

    saved_chunk_paths: list[str] = []

    if save_full_logprobs_npz:
        full_logprobs_dir = output_dir / "full_logprobs_chunks"
        full_logprobs_dir.mkdir(parents=True, exist_ok=True)
    else:
        full_logprobs_dir = None

    run_boundaries = sorted(set(int(index) for index in run_boundaries if 0 <= int(index) <= sequence_length))
    if 0 not in run_boundaries:
        run_boundaries = [0] + run_boundaries
    if sequence_length not in run_boundaries:
        run_boundaries = run_boundaries + [sequence_length]

    chunk_index = 0

    for run_start, run_end in zip(run_boundaries[:-1], run_boundaries[1:]):
        if run_end - run_start <= 1:
            continue

        run_input_ids = input_ids[run_start:run_end]
        run_length = len(run_input_ids)

        for target_start in range(1, run_length, stride):
            target_end = min(target_start + stride, run_length)
            window_start = max(0, target_end - max_length)
            window_ids = run_input_ids[window_start:target_end]

            window_tensor = torch.tensor([window_ids], dtype=torch.long)
            if device != "cpu":
                window_tensor = window_tensor.to(device)

            outputs = model(window_tensor)
            logits = outputs.logits[0]

            log_probs = torch.log_softmax(logits[:-1, :], dim=-1)
            probs = torch.exp(log_probs)

            chunk_relative_positions = list(range(max(target_start, window_start + 1), target_end))
            chunk_local_prev_indices = [
                relative_position - window_start - 1
                for relative_position in chunk_relative_positions
            ]
            chunk_absolute_positions = [run_start + relative_position for relative_position in chunk_relative_positions]

            selected_log_probs = log_probs[chunk_local_prev_indices, :]
            selected_probs = probs[chunk_local_prev_indices, :]

            shannon_entropy_chunk = compute_shannon_entropy_from_log_probs(
                log_probs=selected_log_probs,
                probs=selected_probs,
            )

            renyi_chunk_map: dict[float, torch.Tensor] = {}
            for alpha in renyi_alphas:
                renyi_chunk_map[float(alpha)] = compute_renyi_entropy_from_probs(
                    probs=selected_probs,
                    alpha=float(alpha),
                )

            for row_index, absolute_position in enumerate(chunk_absolute_positions):
                token_id = int(input_ids[absolute_position])
                token_logprob = float(selected_log_probs[row_index, token_id].item())

                observed_logprob[absolute_position] = token_logprob
                surprisal_nats[absolute_position] = -token_logprob
                shannon_entropy_nats[absolute_position] = float(shannon_entropy_chunk[row_index].item())

                for alpha in renyi_alphas:
                    renyi_entropy_arrays[float(alpha)][absolute_position] = float(
                        renyi_chunk_map[float(alpha)][row_index].item()
                    )

            if save_full_logprobs_npz and full_logprobs_dir is not None:
                selected_log_probs_cpu = (
                    selected_log_probs.detach().to(torch.float32).cpu().numpy()
                )

                if full_logprobs_dtype == "float16":
                    log_probs_to_save = selected_log_probs_cpu.astype(np.float16)
                else:
                    log_probs_to_save = selected_log_probs_cpu.astype(np.float32)

                chunk_path = full_logprobs_dir / f"logprobs_chunk_{chunk_index:05d}.npz"
                np.savez_compressed(
                    chunk_path,
                    absolute_positions=np.asarray(chunk_absolute_positions, dtype=np.int32),
                    log_probs=log_probs_to_save,
                )
                saved_chunk_paths.append(str(chunk_path.name))
                chunk_index += 1

    return {
        "observed_logprob": observed_logprob,
        "surprisal_nats": surprisal_nats,
        "shannon_entropy_nats": shannon_entropy_nats,
        "renyi_entropy_nats": renyi_entropy_arrays,
        "saved_full_logprobs_chunks": saved_chunk_paths,
    }


def map_annotation_spans_to_model_tokens(
    spans: Sequence[SpanRecord],
    offset_mapping: Sequence[Sequence[int]],
    tokens_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map original annotation rows to overlapping model token indices."""
    annotation_to_indices: dict[int, list[int]] = {
        int(annotation_index): []
        for annotation_index in tokens_df["annotation_index"].tolist()
    }

    annotation_piece_spans = [
        span for span in spans
        if span.kind == "annotation_piece" and span.annotation_index is not None
    ]

    for span in annotation_piece_spans:
        overlapping_indices: list[int] = []

        for model_token_index, (offset_start, offset_end) in enumerate(offset_mapping):
            overlap_start = max(int(offset_start), int(span.start_char))
            overlap_end = min(int(offset_end), int(span.end_char))
            if overlap_end > overlap_start:
                overlapping_indices.append(model_token_index)

        if len(overlapping_indices) > 0:
            annotation_to_indices[int(span.annotation_index)].extend(overlapping_indices)

    model_token_start: list[int] = []
    model_token_end: list[int] = []
    model_token_indices_json: list[str] = []

    for annotation_index in tokens_df["annotation_index"].tolist():
        unique_indices = sorted(set(annotation_to_indices[int(annotation_index)]))
        model_token_indices_json.append(json.dumps(unique_indices))

        if len(unique_indices) == 0:
            model_token_start.append(-1)
            model_token_end.append(-1)
        else:
            model_token_start.append(unique_indices[0])
            model_token_end.append(unique_indices[-1])

    mapped_df = tokens_df.copy()
    mapped_df["model_token_start"] = model_token_start
    mapped_df["model_token_end"] = model_token_end
    mapped_df["model_token_indices_json"] = model_token_indices_json

    return mapped_df


def format_renyi_alpha(alpha: float) -> str:
    """Format a Rényi order for column naming."""
    alpha_text = f"{float(alpha):g}"
    return alpha_text.replace("-", "m").replace(".", "p")


def aggregate_model_metrics_to_annotations(
    tokens_df: pd.DataFrame,
    distribution_metrics: dict[str, Any],
    tokenizer: Any,
    input_ids: Sequence[int],
) -> pd.DataFrame:
    """Aggregate model-position metrics back to original annotation rows."""
    aggregated_df = tokens_df.copy()

    observed_logprob = distribution_metrics["observed_logprob"]
    surprisal_nats = distribution_metrics["surprisal_nats"]
    shannon_entropy_nats = distribution_metrics["shannon_entropy_nats"]
    renyi_entropy_arrays = distribution_metrics["renyi_entropy_nats"]

    rendered_piece_texts: list[str] = []
    n_model_pieces: list[int] = []
    token_surprisal_values: list[float] = []
    token_shannon_entropy_values: list[float] = []
    token_observed_logprob_values: list[float] = []
    renyi_value_columns: dict[float, list[float]] = {
        float(alpha): []
        for alpha in renyi_entropy_arrays.keys()
    }

    for row in aggregated_df.itertuples(index=False):
        model_token_indices = json.loads(str(row.model_token_indices_json))
        model_token_indices = [int(index) for index in model_token_indices]

        if len(model_token_indices) == 0:
            rendered_piece_texts.append("")
            n_model_pieces.append(0)
            token_surprisal_values.append(np.nan)
            token_shannon_entropy_values.append(np.nan)
            token_observed_logprob_values.append(np.nan)
            for alpha in renyi_value_columns.keys():
                renyi_value_columns[alpha].append(np.nan)
            continue

        row_input_ids = [int(input_ids[index]) for index in model_token_indices]
        row_piece_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(row_input_ids)
        )

        row_surprisal = surprisal_nats[model_token_indices]
        row_observed_logprob = observed_logprob[model_token_indices]
        row_shannon = shannon_entropy_nats[model_token_indices]

        valid_surprisal = row_surprisal[~np.isnan(row_surprisal)]
        valid_observed_logprob = row_observed_logprob[~np.isnan(row_observed_logprob)]
        valid_shannon = row_shannon[~np.isnan(row_shannon)]

        rendered_piece_texts.append(row_piece_text)
        n_model_pieces.append(len(model_token_indices))
        token_surprisal_values.append(float(np.sum(valid_surprisal)) if len(valid_surprisal) > 0 else np.nan)
        token_observed_logprob_values.append(float(np.sum(valid_observed_logprob)) if len(valid_observed_logprob) > 0 else np.nan)
        token_shannon_entropy_values.append(float(np.mean(valid_shannon)) if len(valid_shannon) > 0 else np.nan)

        for alpha, alpha_array in renyi_entropy_arrays.items():
            row_alpha_values = alpha_array[model_token_indices]
            valid_alpha_values = row_alpha_values[~np.isnan(row_alpha_values)]
            renyi_value_columns[float(alpha)].append(
                float(np.mean(valid_alpha_values)) if len(valid_alpha_values) > 0 else np.nan
            )

    aggregated_df["model_token_piece_text"] = rendered_piece_texts
    aggregated_df["n_model_pieces"] = n_model_pieces
    aggregated_df["token_observed_logprob"] = token_observed_logprob_values
    aggregated_df["token_surprisal_nats"] = token_surprisal_values
    aggregated_df["token_surprisal_bits"] = aggregated_df["token_surprisal_nats"] / math.log(2.0)
    aggregated_df["token_shannon_entropy_nats"] = token_shannon_entropy_values
    aggregated_df["token_shannon_entropy_bits"] = aggregated_df["token_shannon_entropy_nats"] / math.log(2.0)

    for alpha, values in renyi_value_columns.items():
        alpha_label = format_renyi_alpha(alpha)
        aggregated_df[f"token_renyi_entropy_alpha_{alpha_label}_nats"] = values
        aggregated_df[f"token_renyi_entropy_alpha_{alpha_label}_bits"] = (
            aggregated_df[f"token_renyi_entropy_alpha_{alpha_label}_nats"] / math.log(2.0)
        )

    return aggregated_df


def aggregate_rendered_pieces_to_words(
    pieces_df: pd.DataFrame,
    distribution_metrics: dict[str, Any],
    offset_mapping: Sequence[Sequence[int]],
    prompt_text: str,
) -> pd.DataFrame:
    """Aggregate model-position metrics to rendered lexical pieces in order."""
    observed_logprob = distribution_metrics["observed_logprob"]
    surprisal_nats = distribution_metrics["surprisal_nats"]
    shannon_entropy_nats = distribution_metrics["shannon_entropy_nats"]
    renyi_entropy_arrays = distribution_metrics["renyi_entropy_nats"]

    word_rows: list[dict[str, Any]] = []
    search_pointer = 0

    for word_index, row in enumerate(pieces_df.itertuples(index=False)):
        rendered_piece = str(row.rendered_piece)

        start_char = prompt_text.find(rendered_piece, search_pointer)
        if start_char < 0:
            raise ValueError(f"Could not locate rendered piece {rendered_piece!r} in prompt text.")

        end_char = start_char + len(rendered_piece)
        search_pointer = end_char

        overlapping_model_indices: list[int] = []
        for model_token_index, (offset_start, offset_end) in enumerate(offset_mapping):
            overlap_start = max(int(offset_start), int(start_char))
            overlap_end = min(int(offset_end), int(end_char))
            if overlap_end > overlap_start:
                overlapping_model_indices.append(model_token_index)

        if len(overlapping_model_indices) == 0:
            row_record: dict[str, Any] = {
                "word_index": word_index,
                "run": int(row.run),
                "word": rendered_piece,
                "matched": False,
                "match_status": "no_model_token_overlap",
                "annotation_index": int(row.annotation_index),
                "piece_index": int(row.piece_index),
                "original_token": str(row.original_token),
                "start": float(row.start),
                "end": float(row.end),
                "speaker": str(row.speaker),
                "rendered_speaker": str(row.rendered_speaker),
                "word_observed_logprob": np.nan,
                "word_surprisal_nats": np.nan,
                "word_surprisal_bits": np.nan,
                "word_shannon_entropy_nats": np.nan,
                "word_shannon_entropy_bits": np.nan,
                "n_model_pieces": 0,
                "model_token_indices_json": "[]",
            }
            for alpha in renyi_entropy_arrays.keys():
                alpha_label = format_renyi_alpha(alpha)
                row_record[f"word_renyi_entropy_alpha_{alpha_label}_nats"] = np.nan
                row_record[f"word_renyi_entropy_alpha_{alpha_label}_bits"] = np.nan

            word_rows.append(row_record)
            continue

        row_surprisal = surprisal_nats[overlapping_model_indices]
        row_observed_logprob = observed_logprob[overlapping_model_indices]
        row_shannon = shannon_entropy_nats[overlapping_model_indices]

        valid_surprisal = row_surprisal[~np.isnan(row_surprisal)]
        valid_observed_logprob = row_observed_logprob[~np.isnan(row_observed_logprob)]
        valid_shannon = row_shannon[~np.isnan(row_shannon)]

        row_record = {
            "word_index": word_index,
            "run": int(row.run),
            "word": rendered_piece,
            "matched": True,
            "match_status": "ok",
            "annotation_index": int(row.annotation_index),
            "piece_index": int(row.piece_index),
            "original_token": str(row.original_token),
            "start": float(row.start),
            "end": float(row.end),
            "speaker": str(row.speaker),
            "rendered_speaker": str(row.rendered_speaker),
            "word_observed_logprob": float(np.sum(valid_observed_logprob)) if len(valid_observed_logprob) > 0 else np.nan,
            "word_surprisal_nats": float(np.sum(valid_surprisal)) if len(valid_surprisal) > 0 else np.nan,
            "word_surprisal_bits": float(np.sum(valid_surprisal)) / math.log(2.0) if len(valid_surprisal) > 0 else np.nan,
            "word_shannon_entropy_nats": float(np.mean(valid_shannon)) if len(valid_shannon) > 0 else np.nan,
            "word_shannon_entropy_bits": float(np.mean(valid_shannon)) / math.log(2.0) if len(valid_shannon) > 0 else np.nan,
            "n_model_pieces": len(overlapping_model_indices),
            "model_token_indices_json": json.dumps(overlapping_model_indices),
        }

        for alpha, alpha_array in renyi_entropy_arrays.items():
            row_alpha_values = alpha_array[overlapping_model_indices]
            valid_alpha_values = row_alpha_values[~np.isnan(row_alpha_values)]
            alpha_label = format_renyi_alpha(alpha)
            row_record[f"word_renyi_entropy_alpha_{alpha_label}_nats"] = (
                float(np.mean(valid_alpha_values)) if len(valid_alpha_values) > 0 else np.nan
            )
            row_record[f"word_renyi_entropy_alpha_{alpha_label}_bits"] = (
                float(np.mean(valid_alpha_values)) / math.log(2.0) if len(valid_alpha_values) > 0 else np.nan
            )

        word_rows.append(row_record)

    return pd.DataFrame(word_rows)


def build_summary(
    prompt_text: str,
    tokens_df: pd.DataFrame,
    words_df: pd.DataFrame,
    model_name: str,
    speaker_map: dict[str, str],
    max_length: int,
    stride: int,
    renyi_alphas: Sequence[float],
    saved_full_logprobs_chunks: Sequence[str],
    run_boundaries: Sequence[int],
) -> dict[str, Any]:
    """Create run-summary metadata."""
    n_lexical_tokens = int(tokens_df["render_for_lm"].fillna(False).sum())
    n_silence_markers = int((tokens_df["token"].astype(str) == "#").sum())
    n_matched_words = int(words_df["matched"].fillna(False).sum())

    return {
        "model_name": model_name,
        "n_annotations": int(len(tokens_df)),
        "n_runs": int(tokens_df["run"].nunique()),
        "n_lexical_tokens_for_lm": n_lexical_tokens,
        "n_silence_markers": n_silence_markers,
        "n_rendered_words": int(len(words_df)),
        "n_matched_rendered_words": n_matched_words,
        "speaker_map": speaker_map,
        "max_length": int(max_length),
        "stride": int(stride),
        "prompt_num_characters": int(len(prompt_text)),
        "renyi_alphas": [float(alpha) for alpha in renyi_alphas],
        "saved_full_logprobs_chunks": list(saved_full_logprobs_chunks),
        "run_boundaries": [int(index) for index in run_boundaries],
    }


def main() -> None:
    """Run the full scoring pipeline."""
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    tokens_df = load_tokens_csv(arguments.tokens_csv)

    prompt_text, spans, tokens_df, speaker_map, pieces_df = build_prompt_from_tokens(
        tokens_df=tokens_df,
        speaker_prefix_mode=arguments.speaker_prefix_mode,
        newline_gap_s=arguments.newline_gap_s,
    )

    tokenizer, model, resolved_device = load_model_and_tokenizer(
        model_name=arguments.model_name,
        device=arguments.device,
        dtype_name=arguments.dtype,
        trust_remote_code=arguments.trust_remote_code,
    )

    encoded = tokenize_prompt(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        bos_token=arguments.bos_token,
    )
    input_ids = encoded["input_ids"]
    offset_mapping = encoded["offset_mapping"]
    max_length = infer_max_length(
        model=model,
        tokenizer=tokenizer,
        user_max_length=arguments.max_length,
    )

    run_boundaries = infer_run_boundaries(
        tokens_df=tokens_df,
        spans=spans,
        offset_mapping=offset_mapping,
    )

    distribution_metrics = compute_distributional_metrics(
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        stride=arguments.stride,
        device=resolved_device,
        renyi_alphas=arguments.renyi_alphas,
        save_full_logprobs_npz=arguments.save_full_logprobs_npz,
        full_logprobs_dtype=arguments.full_logprobs_dtype,
        output_dir=arguments.output_dir,
        run_boundaries=run_boundaries,
    )

    tokens_df = map_annotation_spans_to_model_tokens(
        spans=spans,
        offset_mapping=offset_mapping,
        tokens_df=tokens_df,
    )
    token_out_df = aggregate_model_metrics_to_annotations(
        tokens_df=tokens_df,
        distribution_metrics=distribution_metrics,
        tokenizer=tokenizer,
        input_ids=input_ids,
    )
    word_out_df = aggregate_rendered_pieces_to_words(
        pieces_df=pieces_df,
        distribution_metrics=distribution_metrics,
        offset_mapping=offset_mapping,
        prompt_text=prompt_text,
    )

    model_token_df = pd.DataFrame(
        {
            "model_token_index": np.arange(len(input_ids), dtype=int),
            "input_id": input_ids,
            "token_text": tokenizer.convert_ids_to_tokens(input_ids),
            "offset_start": [pair[0] for pair in offset_mapping],
            "offset_end": [pair[1] for pair in offset_mapping],
            "observed_logprob": distribution_metrics["observed_logprob"],
            "surprisal_nats": distribution_metrics["surprisal_nats"],
            "surprisal_bits": distribution_metrics["surprisal_nats"] / math.log(2.0),
            "shannon_entropy_nats": distribution_metrics["shannon_entropy_nats"],
            "shannon_entropy_bits": distribution_metrics["shannon_entropy_nats"] / math.log(2.0),
        }
    )

    for alpha, alpha_array in distribution_metrics["renyi_entropy_nats"].items():
        alpha_label = format_renyi_alpha(alpha)
        model_token_df[f"renyi_entropy_alpha_{alpha_label}_nats"] = alpha_array
        model_token_df[f"renyi_entropy_alpha_{alpha_label}_bits"] = alpha_array / math.log(2.0)

    prompt_path = arguments.output_dir / "rendered_prompt.txt"
    spans_path = arguments.output_dir / "prompt_spans.json"
    summary_path = arguments.output_dir / "run_summary.json"
    token_out_path = arguments.output_dir / "token_uncertainty.csv"
    word_out_path = arguments.output_dir / "word_uncertainty.csv"
    model_token_out_path = arguments.output_dir / "model_token_uncertainty.csv"
    pieces_out_path = arguments.output_dir / "rendered_pieces.csv"

    prompt_path.write_text(prompt_text, encoding="utf-8")

    spans_payload = [
        {
            "kind": span.kind,
            "start_char": int(span.start_char),
            "end_char": int(span.end_char),
            "annotation_index": None if span.annotation_index is None else int(span.annotation_index),
            "piece_index": None if span.piece_index is None else int(span.piece_index),
            "payload": span.payload,
        }
        for span in spans
    ]
    spans_path.write_text(json.dumps(spans_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    token_out_df.to_csv(token_out_path, index=False)
    word_out_df.to_csv(word_out_path, index=False)
    model_token_df.to_csv(model_token_out_path, index=False)
    pieces_df.to_csv(pieces_out_path, index=False)

    summary = build_summary(
        prompt_text=prompt_text,
        tokens_df=token_out_df,
        words_df=word_out_df,
        model_name=arguments.model_name,
        speaker_map=speaker_map,
        max_length=max_length,
        stride=arguments.stride,
        renyi_alphas=arguments.renyi_alphas,
        saved_full_logprobs_chunks=distribution_metrics["saved_full_logprobs_chunks"],
        run_boundaries=run_boundaries,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved rendered prompt to: {prompt_path}")
    print(f"Saved prompt spans to: {spans_path}")
    print(f"Saved annotation-level uncertainty to: {token_out_path}")
    print(f"Saved rendered-word uncertainty to: {word_out_path}")
    print(f"Saved model-token uncertainty to: {model_token_out_path}")
    print(f"Saved rendered pieces to: {pieces_out_path}")
    print(f"Saved run summary to: {summary_path}")

    if arguments.save_full_logprobs_npz:
        print(f"Saved full log-probability chunks to: {arguments.output_dir / 'full_logprobs_chunks'}")


if __name__ == "__main__":
    main()
