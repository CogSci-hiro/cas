#!/usr/bin/env python3
"""Compute token- and word-level surprisal from Claire-Mistral-style dialogue text.

This script reads token-level dialogue annotations, converts them into a single
prompt formatted with speaker tags such as ``[SpeakerA:]``, computes model-token
surprisal with a causal language model, maps surprisals back to annotated
tokens, and then aggregates annotated tokens to canonical words from a second
CSV file.

The default model is the Mistral adaptation described in the Claire paper:
``OpenLLM-France/Claire-Mistral-7B-0.1``.

The script is designed to run on GPU in the cloud, but it can also be tested
locally on CPU with a much smaller causal LM, for example::

    python claire_surprisal.py \
        --tokens_csv tokens.csv \
        --words_csv words.csv \
        --output_dir out \
        --model_name sshleifer/tiny-gpt2 \
        --device cpu

Input token CSV must contain these columns:
    token, speaker, start, end

Silence markers marked as ``#`` are treated as non-lexical boundaries: they are
not sent to the language model as words, but they can trigger a turn break.

Canonical word CSV must contain at least:
    word

Notes
-----
- Surprisal is reported in natural log units (nats): ``-log p(token | context)``.
- Speaker prefixes inserted by the script are included in the LM context but are
  not exported as annotated tokens.
- Aggregation from annotated tokens to canonical words uses ordered greedy
  string matching after light normalization. This works well when the canonical
  word list preserves order and corresponds to the same transcript.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configuration constants
DEFAULT_MODEL_NAME = "OpenLLM-France/Claire-Mistral-7B-0.1"
DEFAULT_MAX_NEWLINE_GAP_S = 0.75
DEFAULT_STRIDE = 512
DEFAULT_BATCH_SIZE = 1
SPACELESS_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "…", "%"}
SPACELESS_AFTER = {"(", "[", "{", "«", "\"", "'"}
APOSTROPHE_ENDINGS = ("'", "’")


@dataclass
class SpanRecord:
    """Character span of one inserted text fragment.

    Parameters
    ----------
    kind
        Fragment type. One of ``annotation``, ``speaker_prefix`` or ``separator``.
    start_char
        Inclusive start character index in the built prompt.
    end_char
        Exclusive end character index in the built prompt.
    annotation_index
        Row index in the input annotation table when ``kind='annotation'``.
    payload
        Extra metadata.
    """

    kind: str
    start_char: int
    end_char: int
    annotation_index: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens_csv", type=Path, required=True, help="CSV with token, speaker, start, end columns.")
    parser.add_argument("--words_csv", type=Path, required=True, help="CSV with canonical words; must include a 'word' column.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HF model name or local path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Model dtype.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit loading when supported (GPU only).")
    parser.add_argument("--max_length", type=int, default=None, help="Override model context length.")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Number of new tokens scored per window.")
    parser.add_argument("--newline_gap_s", type=float, default=DEFAULT_MAX_NEWLINE_GAP_S, help="Insert a line break after silences/gaps at or above this duration.")
    parser.add_argument("--speaker_prefix_mode", type=str, default="alphabetic", choices=["alphabetic", "original"], help="Use [SpeakerA:] / [SpeakerB:] or original speaker labels.")
    parser.add_argument("--keep_case", action="store_true", help="Keep original token casing instead of lowercasing nothing; default already keeps case. Included for explicitness.")
    parser.add_argument("--bos_token", action="store_true", help="Prepend tokenizer BOS token if available.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    return parser.parse_args()


def load_inputs(tokens_csv: Path, words_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate input CSV files.

    Parameters
    ----------
    tokens_csv
        Path to token annotation CSV.
    words_csv
        Path to canonical word CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Token annotations and canonical words.
    """
    tokens_df = pd.read_csv(tokens_csv)
    words_df = pd.read_csv(words_csv)

    required_token_columns = {"token", "speaker", "start", "end"}
    missing_token_columns = required_token_columns.difference(tokens_df.columns)
    if missing_token_columns:
        raise ValueError(f"tokens_csv is missing required columns: {sorted(missing_token_columns)}")

    if "word" not in words_df.columns:
        raise ValueError("words_csv must contain a 'word' column.")

    tokens_df = tokens_df.copy()
    words_df = words_df.copy()
    tokens_df["annotation_index"] = np.arange(len(tokens_df), dtype=int)
    return tokens_df, words_df


def canonicalize_speaker_labels(speakers: Sequence[str], mode: str) -> Dict[str, str]:
    """Map raw speaker labels to prompt speaker labels.

    Parameters
    ----------
    speakers
        Unique speaker labels in encounter order.
    mode
        Either ``alphabetic`` or ``original``.

    Returns
    -------
    dict[str, str]
        Mapping from raw labels to rendered speaker names.
    """
    unique_speakers = list(dict.fromkeys(str(s) for s in speakers))
    if mode == "original":
        return {speaker: sanitize_speaker_name(speaker) for speaker in unique_speakers}

    labels: Dict[str, str] = {}
    for index, speaker in enumerate(unique_speakers):
        labels[speaker] = f"Speaker{number_to_letters(index)}"
    return labels


def number_to_letters(index: int) -> str:
    """Convert a zero-based index to A, B, ..., Z, AA, AB, ...

    Parameters
    ----------
    index
        Zero-based integer.

    Returns
    -------
    str
        Letter code.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    result = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        result.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(result))


def sanitize_speaker_name(name: str) -> str:
    """Sanitize speaker names for bracketed prompt format."""
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    cleaned = cleaned.replace("]", ")").replace("[", "(")
    return cleaned or "Speaker"


def needs_space_before(token: str) -> bool:
    """Return whether a space should precede a token in the rendered text."""
    if token in SPACELESS_BEFORE:
        return False
    if token.startswith("'") or token.startswith("’"):
        return False
    return True


def append_text(parts: List[str], spans: List[SpanRecord], text: str, kind: str, annotation_index: Optional[int] = None, payload: Optional[Dict[str, Any]] = None) -> None:
    """Append text and register its character span."""
    if not text:
        return
    start_char = sum(len(part) for part in parts)
    parts.append(text)
    end_char = start_char + len(text)
    spans.append(SpanRecord(kind=kind, start_char=start_char, end_char=end_char, annotation_index=annotation_index, payload=payload))


def build_prompt_from_annotations(tokens_df: pd.DataFrame, speaker_prefix_mode: str, newline_gap_s: float) -> Tuple[str, List[SpanRecord], pd.DataFrame, Dict[str, str]]:
    """Render token annotations into a Claire-style dialogue prompt.

    Parameters
    ----------
    tokens_df
        Token annotation table.
    speaker_prefix_mode
        Either ``alphabetic`` or ``original``.
    newline_gap_s
        Minimum silence/gap duration that forces a line break.

    Returns
    -------
    tuple[str, list[SpanRecord], pd.DataFrame, dict[str, str]]
        Prompt text, span records, updated token table, and speaker mapping.
    """
    tokens_df = tokens_df.copy()
    speaker_map = canonicalize_speaker_labels(tokens_df["speaker"].astype(str).tolist(), mode=speaker_prefix_mode)
    tokens_df["rendered_speaker"] = tokens_df["speaker"].astype(str).map(speaker_map)

    parts: List[str] = []
    spans: List[SpanRecord] = []

    previous_speaker: Optional[str] = None
    previous_end: Optional[float] = None
    previous_token_text: Optional[str] = None
    at_line_start = True

    for row in tokens_df.itertuples(index=False):
        token_text = str(row.token)
        speaker = str(row.speaker)
        start_time = float(row.start)
        end_time = float(row.end)
        annotation_index = int(row.annotation_index)

        is_silence = token_text == "#"
        speaker_changed = previous_speaker is None or speaker != previous_speaker
        silence_gap = None if previous_end is None else max(0.0, start_time - previous_end)
        force_newline = bool(speaker_changed or (silence_gap is not None and silence_gap >= newline_gap_s) or is_silence)

        if force_newline and parts and not parts[-1].endswith("\n"):
            append_text(parts, spans, "\n", kind="separator", payload={"reason": "turn_or_gap"})
            at_line_start = True

        if is_silence:
            previous_end = end_time
            previous_speaker = speaker
            previous_token_text = None
            continue

        if at_line_start:
            prefix = f"[{speaker_map[speaker]}:] "
            append_text(parts, spans, prefix, kind="speaker_prefix", payload={"speaker": speaker, "rendered_speaker": speaker_map[speaker]})
            at_line_start = False
            previous_token_text = None

        if previous_token_text is not None and needs_space_before(token_text) and not previous_token_text.endswith(APOSTROPHE_ENDINGS):
            append_text(parts, spans, " ", kind="separator", payload={"reason": "intra_turn_space"})

        append_text(
            parts,
            spans,
            token_text,
            kind="annotation",
            annotation_index=annotation_index,
            payload={
                "speaker": speaker,
                "rendered_speaker": speaker_map[speaker],
                "start": start_time,
                "end": end_time,
            },
        )

        previous_speaker = speaker
        previous_end = end_time
        previous_token_text = token_text

    prompt_text = "".join(parts)
    return prompt_text, spans, tokens_df, speaker_map


def choose_torch_dtype(dtype_name: str, device: str) -> Optional[torch.dtype]:
    """Choose the torch dtype to use for model loading."""
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return None


def load_model_and_tokenizer(model_name: str, device: str, dtype_name: str, load_in_4bit: bool, trust_remote_code: bool) -> Tuple[Any, Any, str]:
    """Load tokenizer and model.

    Returns
    -------
    tuple
        Tokenizer, model, resolved device string.
    """
    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if load_in_4bit:
        if resolved_device == "cpu":
            raise ValueError("--load_in_4bit is only useful on GPU-backed inference.")
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = choose_torch_dtype(dtype_name, resolved_device)
    else:
        torch_dtype = choose_torch_dtype(dtype_name, resolved_device)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if resolved_device != "cpu":
            model = model.to(resolved_device)
        model.eval()
        return tokenizer, model, resolved_device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model, resolved_device


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
    if device == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise ValueError("MPS was requested but is not available.")
    return device


def tokenize_prompt(tokenizer: Any, prompt_text: str, bos_token: bool) -> Dict[str, Any]:
    """Tokenize prompt and return token ids plus offsets."""
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
        adjusted_offsets: List[Tuple[int, int]] = []
        for start_char, end_char in encoded["offset_mapping"]:
            adjusted_offsets.append((max(0, start_char - bos_length), max(0, end_char - bos_length)))
        encoded["offset_mapping"] = adjusted_offsets

    return encoded


def infer_max_length(model: Any, tokenizer: Any, user_max_length: Optional[int]) -> int:
    """Infer usable context length."""
    if user_max_length is not None:
        return int(user_max_length)
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "sliding_window", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    candidates = [int(value) for value in candidates if value is not None and int(value) < 1_000_000]
    if not candidates:
        return 2048
    return max(candidates)


@torch.inference_mode()
def compute_per_model_token_surprisal(
    model: Any,
    input_ids: Sequence[int],
    max_length: int,
    stride: int,
    device: str,
) -> np.ndarray:
    """Compute per-model-token surprisal for a causal LM.

    Parameters
    ----------
    model
        Hugging Face causal LM.
    input_ids
        Full tokenized prompt as integer ids.
    max_length
        Maximum left-context window.
    stride
        Number of new tokens scored per chunk.
    device
        Execution device.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_tokens,)`` with surprisal in nats. Position 0 is NaN
        because no previous-token context exists for the first token.
    """
    sequence_length = len(input_ids)
    surprisal = np.full(sequence_length, np.nan, dtype=np.float64)

    for target_start in range(1, sequence_length, stride):
        target_end = min(target_start + stride, sequence_length)
        window_start = max(0, target_end - max_length)
        window_ids = input_ids[window_start:target_end]
        window_tensor = torch.tensor([window_ids], dtype=torch.long)
        if device != "cpu":
            window_tensor = window_tensor.to(device)

        outputs = model(window_tensor)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits[:-1, :], dim=-1)

        absolute_positions = range(max(target_start, window_start + 1), target_end)
        for absolute_position in absolute_positions:
            local_prev_index = absolute_position - window_start - 1
            token_id = int(input_ids[absolute_position])
            surprisal_value = -log_probs[local_prev_index, token_id].item()
            surprisal[absolute_position] = surprisal_value

    return surprisal


def map_annotation_spans_to_model_tokens(spans: List[SpanRecord], offset_mapping: Sequence[Tuple[int, int]], tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Map annotated token spans to overlapping model tokens.

    Returns
    -------
    pd.DataFrame
        Annotation table with start/end model token indices.
    """
    start_token_indices = np.full(len(tokens_df), -1, dtype=int)
    end_token_indices = np.full(len(tokens_df), -1, dtype=int)

    annotation_spans = [span for span in spans if span.kind == "annotation" and span.annotation_index is not None]
    token_pointer = 0
    n_model_tokens = len(offset_mapping)

    for span in annotation_spans:
        while token_pointer < n_model_tokens and offset_mapping[token_pointer][1] <= span.start_char:
            token_pointer += 1

        current_pointer = token_pointer
        overlapping_indices: List[int] = []
        while current_pointer < n_model_tokens and offset_mapping[current_pointer][0] < span.end_char:
            overlap_start = max(offset_mapping[current_pointer][0], span.start_char)
            overlap_end = min(offset_mapping[current_pointer][1], span.end_char)
            if overlap_end > overlap_start:
                overlapping_indices.append(current_pointer)
            current_pointer += 1

        if overlapping_indices:
            start_token_indices[span.annotation_index] = overlapping_indices[0]
            end_token_indices[span.annotation_index] = overlapping_indices[-1]

    mapped_df = tokens_df.copy()
    mapped_df["model_token_start"] = start_token_indices
    mapped_df["model_token_end"] = end_token_indices
    return mapped_df


def aggregate_model_to_annotation_tokens(tokens_df: pd.DataFrame, model_token_surprisal: np.ndarray, tokenizer: Any, input_ids: Sequence[int]) -> pd.DataFrame:
    """Aggregate model-token surprisals to annotated tokens."""
    aggregated = tokens_df.copy()
    token_piece_texts: List[str] = []
    token_piece_counts: List[int] = []
    token_surprisals: List[float] = []

    for row in aggregated.itertuples(index=False):
        start_index = int(row.model_token_start)
        end_index = int(row.model_token_end)
        token_text = str(row.token)

        if token_text == "#" or start_index < 0 or end_index < 0 or end_index < start_index:
            token_piece_texts.append("")
            token_piece_counts.append(0)
            token_surprisals.append(np.nan)
            continue

        span_indices = list(range(start_index, end_index + 1))
        span_surprisal = model_token_surprisal[span_indices]
        valid_surprisal = span_surprisal[~np.isnan(span_surprisal)]
        span_piece_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([input_ids[i] for i in span_indices]))

        token_piece_texts.append(span_piece_text)
        token_piece_counts.append(len(span_indices))
        token_surprisals.append(float(np.sum(valid_surprisal)) if len(valid_surprisal) else np.nan)

    aggregated["model_token_piece_text"] = token_piece_texts
    aggregated["n_model_pieces"] = token_piece_counts
    aggregated["token_surprisal_nats"] = token_surprisals
    aggregated["token_surprisal_bits"] = aggregated["token_surprisal_nats"] / math.log(2.0)
    return aggregated


def normalize_for_alignment(text: str) -> str:
    """Normalize strings for token-to-word alignment."""
    text = str(text)
    text = text.strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\-‐‑‒–—]", "", text)
    text = re.sub(r"^[\"'«»]+|[\"'«»]+$", "", text)
    return text


def aggregate_tokens_to_canonical_words(tokens_df: pd.DataFrame, words_df: pd.DataFrame) -> pd.DataFrame:
    """Greedily align annotated tokens to canonical words.

    Parameters
    ----------
    tokens_df
        Annotated token table with token surprisal already computed.
    words_df
        Canonical words in order.

    Returns
    -------
    pd.DataFrame
        Word-level table.
    """
    lexical_tokens = tokens_df.loc[tokens_df["token"].astype(str) != "#"].copy().reset_index(drop=True)
    lexical_tokens["normalized_token"] = lexical_tokens["token"].map(normalize_for_alignment)

    words_df = words_df.copy().reset_index(drop=True)
    words_df["normalized_word"] = words_df["word"].map(normalize_for_alignment)

    word_records: List[Dict[str, Any]] = []
    token_pointer = 0

    for word_index, word_row in words_df.iterrows():
        target = word_row["normalized_word"]
        original_word = word_row["word"]
        if target == "":
            word_records.append(
                {
                    "word_index": word_index,
                    "word": original_word,
                    "matched": False,
                    "match_status": "empty_canonical_word",
                    "token_start_index": np.nan,
                    "token_end_index": np.nan,
                    "start": np.nan,
                    "end": np.nan,
                    "speaker": np.nan,
                    "rendered_speaker": np.nan,
                    "word_surprisal_nats": np.nan,
                    "word_surprisal_bits": np.nan,
                    "n_source_tokens": 0,
                    "source_tokens": "",
                }
            )
            continue

        assembled = ""
        matched_indices: List[int] = []
        probe_pointer = token_pointer

        while probe_pointer < len(lexical_tokens) and len(assembled) < len(target):
            candidate_piece = lexical_tokens.loc[probe_pointer, "normalized_token"]
            if candidate_piece == "":
                probe_pointer += 1
                continue

            candidate_assembled = assembled + candidate_piece
            if not target.startswith(candidate_assembled):
                break

            assembled = candidate_assembled
            matched_indices.append(probe_pointer)
            probe_pointer += 1

            if assembled == target:
                break

        if matched_indices and assembled == target:
            matched_tokens = lexical_tokens.loc[matched_indices].copy()
            word_surprisal_nats = float(np.nansum(matched_tokens["token_surprisal_nats"].to_numpy(dtype=float)))
            first_token = matched_tokens.iloc[0]
            last_token = matched_tokens.iloc[-1]
            speaker_values = matched_tokens["speaker"].astype(str).unique().tolist()
            rendered_speaker_values = matched_tokens["rendered_speaker"].astype(str).unique().tolist()

            word_records.append(
                {
                    "word_index": word_index,
                    "word": original_word,
                    "matched": True,
                    "match_status": "ok",
                    "token_start_index": int(first_token["annotation_index"]),
                    "token_end_index": int(last_token["annotation_index"]),
                    "start": float(first_token["start"]),
                    "end": float(last_token["end"]),
                    "speaker": speaker_values[0] if len(speaker_values) == 1 else "MULTI",
                    "rendered_speaker": rendered_speaker_values[0] if len(rendered_speaker_values) == 1 else "MULTI",
                    "word_surprisal_nats": word_surprisal_nats,
                    "word_surprisal_bits": word_surprisal_nats / math.log(2.0),
                    "n_source_tokens": int(len(matched_tokens)),
                    "source_tokens": " ".join(matched_tokens["token"].astype(str).tolist()),
                }
            )
            token_pointer = matched_indices[-1] + 1
        else:
            word_records.append(
                {
                    "word_index": word_index,
                    "word": original_word,
                    "matched": False,
                    "match_status": "alignment_failed",
                    "token_start_index": np.nan,
                    "token_end_index": np.nan,
                    "start": np.nan,
                    "end": np.nan,
                    "speaker": np.nan,
                    "rendered_speaker": np.nan,
                    "word_surprisal_nats": np.nan,
                    "word_surprisal_bits": np.nan,
                    "n_source_tokens": 0,
                    "source_tokens": "",
                }
            )

    return pd.DataFrame(word_records)


def build_summary(
    prompt_text: str,
    tokens_df: pd.DataFrame,
    words_df: pd.DataFrame,
    model_name: str,
    speaker_map: Dict[str, str],
    max_length: int,
    stride: int,
) -> Dict[str, Any]:
    """Create run summary metadata."""
    n_lexical_tokens = int((tokens_df["token"].astype(str) != "#").sum())
    n_silence_markers = int((tokens_df["token"].astype(str) == "#").sum())
    n_matched_words = int(words_df["matched"].fillna(False).sum()) if "matched" in words_df.columns else None

    return {
        "model_name": model_name,
        "n_annotations": int(len(tokens_df)),
        "n_lexical_tokens": n_lexical_tokens,
        "n_silence_markers": n_silence_markers,
        "n_canonical_words": int(len(words_df)),
        "n_matched_canonical_words": n_matched_words,
        "speaker_map": speaker_map,
        "max_length": int(max_length),
        "stride": int(stride),
        "prompt_num_characters": int(len(prompt_text)),
    }


def main() -> None:
    """Run the full surprisal pipeline."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens_df, words_df = load_inputs(args.tokens_csv, args.words_csv)
    prompt_text, spans, tokens_df, speaker_map = build_prompt_from_annotations(
        tokens_df=tokens_df,
        speaker_prefix_mode=args.speaker_prefix_mode,
        newline_gap_s=args.newline_gap_s,
    )

    tokenizer, model, resolved_device = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    encoded = tokenize_prompt(tokenizer=tokenizer, prompt_text=prompt_text, bos_token=args.bos_token)
    input_ids = encoded["input_ids"]
    offset_mapping = encoded["offset_mapping"]
    max_length = infer_max_length(model=model, tokenizer=tokenizer, user_max_length=args.max_length)

    model_token_surprisal = compute_per_model_token_surprisal(
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        stride=args.stride,
        device=resolved_device,
    )

    tokens_df = map_annotation_spans_to_model_tokens(spans=spans, offset_mapping=offset_mapping, tokens_df=tokens_df)
    tokens_df = aggregate_model_to_annotation_tokens(tokens_df=tokens_df, model_token_surprisal=model_token_surprisal, tokenizer=tokenizer, input_ids=input_ids)
    words_out_df = aggregate_tokens_to_canonical_words(tokens_df=tokens_df, words_df=words_df)

    model_token_df = pd.DataFrame(
        {
            "model_token_index": np.arange(len(input_ids), dtype=int),
            "input_id": input_ids,
            "token_text": tokenizer.convert_ids_to_tokens(input_ids),
            "offset_start": [pair[0] for pair in offset_mapping],
            "offset_end": [pair[1] for pair in offset_mapping],
            "surprisal_nats": model_token_surprisal,
            "surprisal_bits": model_token_surprisal / math.log(2.0),
        }
    )

    prompt_path = args.output_dir / "rendered_prompt.txt"
    summary_path = args.output_dir / "run_summary.json"
    annotation_out_path = args.output_dir / "token_surprisal.csv"
    word_out_path = args.output_dir / "word_surprisal.csv"
    model_token_out_path = args.output_dir / "model_token_surprisal.csv"

    prompt_path.write_text(prompt_text, encoding="utf-8")
    tokens_df.to_csv(annotation_out_path, index=False)
    words_out_df.to_csv(word_out_path, index=False)
    model_token_df.to_csv(model_token_out_path, index=False)

    summary = build_summary(
        prompt_text=prompt_text,
        tokens_df=tokens_df,
        words_df=words_out_df,
        model_name=args.model_name,
        speaker_map=speaker_map,
        max_length=max_length,
        stride=args.stride,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved rendered prompt to: {prompt_path}")
    print(f"Saved annotation-level surprisals to: {annotation_out_path}")
    print(f"Saved canonical word surprisals to: {word_out_path}")
    print(f"Saved model-token surprisals to: {model_token_out_path}")
    print(f"Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
