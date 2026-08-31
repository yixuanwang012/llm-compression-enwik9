#!/usr/bin/env python3
"""
Run an autoregressive language-model compression estimate on enwik9.

This script is intended for CSF/SLURM GPU runs. It streams the input file,
applies the same ASCII normalisation used in the pilot notebook, computes
next-token negative log-probability, logs top-k target coverage, and writes
checkpoint/final JSON files.

The reported compressed size is a theoretical entropy estimate from model
log-probabilities. It is not an arithmetic-coded, decodable compressed file.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute theoretical BPC for ASCII-normalised enwik9 with a causal LM."
    )
    parser.add_argument("--input", required=True, help="Path to raw enwik9 or a test subset.")
    parser.add_argument("--output", required=True, help="Path to final JSON result.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON.")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model id or local model path.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Raw bytes read per model forward pass. The reference-paper setting is 2048.",
    )
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=1024,
        help="Maximum tokenized length scored per chunk.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top-k target hit-rate diagnostic.")
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=None,
        help="Optional byte limit for smoke tests, e.g. 1024000 for 500 paper-style chunks.",
    )
    parser.add_argument(
        "--fail-on-truncation",
        action="store_true",
        help=(
            "Stop if a tokenized chunk exceeds --max-model-tokens. Use this for paper-style "
            "runs so that context truncation is never hidden."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device selection. Use cuda on CSF GPU nodes.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype. float16 is usually appropriate on GPU.",
    )
    parser.add_argument(
        "--checkpoint-every-chunks",
        type=int,
        default=1000,
        help="Write checkpoint after this many processed chunks.",
    )
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=100,
        help="Print progress after this many processed chunks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint if it exists.",
    )
    parser.add_argument(
        "--classical-baselines",
        action="store_true",
        help="Also compute gzip/bz2/lzma baselines. Avoid this for repeated full 1GB runs.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small internal test that does not load transformers or a model.",
    )
    return parser.parse_args()


@dataclass
class RunState:
    byte_offset: int = 0
    chunks_processed: int = 0
    processed_bytes: int = 0
    total_bits: float = 0.0
    scored_tokens: int = 0
    truncated_chunks: int = 0
    skipped_chunks: int = 0
    topk_hits: int = 0
    topk_total: int = 0
    non_ascii_mapped: int = 0
    start_time: float = 0.0
    elapsed_before_resume: float = 0.0


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(requested: str, device: str):
    import torch

    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_checkpoint(path: Path) -> RunState:
    data = json.loads(path.read_text())
    return RunState(**data["state"])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_checkpoint(path: Path, state: RunState, args: argparse.Namespace) -> None:
    elapsed = time.time() - state.start_time + state.elapsed_before_resume
    payload = {
        "kind": "checkpoint",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "args": vars(args),
        "state": asdict(state),
        "metrics": metrics_from_state(state, elapsed),
    }
    write_json(path, payload)


def metrics_from_state(state: RunState, elapsed: float) -> dict[str, float | int]:
    theoretical_bytes = state.total_bits / 8.0
    bpc = state.total_bits / state.processed_bytes if state.processed_bytes else 0.0
    compression_rate = (
        100.0 * theoretical_bytes / state.processed_bytes if state.processed_bytes else 0.0
    )
    topk_hit_rate = 100.0 * state.topk_hits / state.topk_total if state.topk_total else 0.0
    bytes_per_second = state.processed_bytes / elapsed if elapsed > 0 else 0.0
    return {
        "processed_bytes": state.processed_bytes,
        "chunks_processed": state.chunks_processed,
        "scored_tokens": state.scored_tokens,
        "truncated_chunks": state.truncated_chunks,
        "skipped_chunks": state.skipped_chunks,
        "non_ascii_mapped": state.non_ascii_mapped,
        "topk_hits": state.topk_hits,
        "topk_total": state.topk_total,
        "topk_hit_rate_percent": topk_hit_rate,
        "total_bits": state.total_bits,
        "theoretical_compressed_bytes": theoretical_bytes,
        "bpc": bpc,
        "theoretical_compression_rate_percent": compression_rate,
        "elapsed_seconds": elapsed,
        "bytes_per_second": bytes_per_second,
    }


def ascii_normalise(raw_chunk: bytes) -> tuple[str, int]:
    non_ascii = sum(byte > 127 for byte in raw_chunk)
    ascii_chunk = bytes((byte & 0x7F) for byte in raw_chunk)
    return ascii_chunk.decode("ascii"), non_ascii


def classical_baselines(path: Path, limit_bytes: int | None) -> dict[str, dict[str, float | int]]:
    with path.open("rb") as handle:
        data = handle.read(limit_bytes) if limit_bytes else handle.read()
    data = bytes((byte & 0x7F) for byte in data)
    methods = {
        "gzip": gzip.compress(data, compresslevel=9),
        "bz2": bz2.compress(data, compresslevel=9),
        "lzma": lzma.compress(data, preset=9),
    }
    return {
        name: {
            "compressed_bytes": len(blob),
            "compression_rate_percent": 100.0 * len(blob) / len(data) if data else 0.0,
        }
        for name, blob in methods.items()
    }


def run_self_test() -> None:
    text, non_ascii = ascii_normalise(bytes([65, 200, 66, 255]))
    assert text == "AHB\x7f"
    assert non_ascii == 2
    state = RunState(processed_bytes=10, total_bits=20.0, topk_hits=8, topk_total=10)
    metrics = metrics_from_state(state, elapsed=2.0)
    assert abs(metrics["bpc"] - 2.0) < 1e-9
    assert abs(metrics["theoretical_compression_rate_percent"] - 25.0) < 1e-9
    assert abs(metrics["topk_hit_rate_percent"] - 80.0) < 1e-9
    print("Self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    if args.resume and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        state.elapsed_before_resume += time.time() - state.start_time
        print(f"Resuming from byte offset {state.byte_offset} in {checkpoint_path}")
    else:
        state = RunState()
    state.start_time = time.time()

    print(f"Model: {args.model}")
    print(f"Input: {input_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Chunk size: {args.chunk_size} bytes")
    print(f"Limit bytes: {args.limit_bytes if args.limit_bytes else 'full file'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    file_size = input_path.stat().st_size
    target_limit = min(file_size, args.limit_bytes) if args.limit_bytes else file_size
    if state.byte_offset >= target_limit:
        print("Checkpoint already reached requested byte limit.")

    with input_path.open("rb") as handle:
        handle.seek(state.byte_offset)
        while state.byte_offset < target_limit:
            bytes_left = target_limit - state.byte_offset
            raw_chunk = handle.read(min(args.chunk_size, bytes_left))
            if not raw_chunk:
                break

            text, non_ascii = ascii_normalise(raw_chunk)
            encoded = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = encoded["input_ids"]

            if input_ids.shape[1] > args.max_model_tokens:
                state.truncated_chunks += 1
                if args.fail_on_truncation:
                    raise RuntimeError(
                        "Tokenized chunk exceeds the model context limit. "
                        f"byte_offset={state.byte_offset}, raw_chunk_bytes={len(raw_chunk)}, "
                        f"tokenized_length={input_ids.shape[1]}, "
                        f"max_model_tokens={args.max_model_tokens}. "
                        "For a strict paper-style run, use a model with a longer context window "
                        "or reduce --chunk-size and report this deviation explicitly."
                    )
                input_ids = input_ids[:, : args.max_model_tokens]

            if input_ids.shape[1] < 2:
                state.skipped_chunks += 1
                state.byte_offset += len(raw_chunk)
                continue

            input_ids = input_ids.to(device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                state.total_bits += (-target_log_probs / math.log(2)).sum().item()

                k = min(args.top_k, logits.shape[-1])
                topk_indices = torch.topk(logits, k=k, dim=-1).indices
                hits = (topk_indices == targets.unsqueeze(-1)).any(dim=-1)
                state.topk_hits += hits.sum().item()
                state.topk_total += hits.numel()

            state.byte_offset += len(raw_chunk)
            state.processed_bytes += len(raw_chunk)
            state.non_ascii_mapped += non_ascii
            state.scored_tokens += targets.numel()
            state.chunks_processed += 1

            if device == "cuda" and state.chunks_processed % 500 == 0:
                torch.cuda.empty_cache()

            if state.chunks_processed % args.progress_every_chunks == 0:
                elapsed = time.time() - state.start_time + state.elapsed_before_resume
                metrics = metrics_from_state(state, elapsed)
                progress = 100.0 * state.byte_offset / target_limit
                print(
                    f"chunks={state.chunks_processed} "
                    f"bytes={state.processed_bytes} "
                    f"progress={progress:.2f}% "
                    f"BPC={metrics['bpc']:.4f} "
                    f"rate={metrics['theoretical_compression_rate_percent']:.2f}% "
                    f"top{args.top_k}={metrics['topk_hit_rate_percent']:.2f}%"
                )

            if state.chunks_processed % args.checkpoint_every_chunks == 0:
                write_checkpoint(checkpoint_path, state, args)

    elapsed = time.time() - state.start_time + state.elapsed_before_resume
    final_payload: dict[str, Any] = {
        "kind": "final_result",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "input_file_size_bytes": file_size,
        "target_limit_bytes": target_limit,
        "device": device,
        "dtype": str(dtype),
        "state": asdict(state),
        "metrics": metrics_from_state(state, elapsed),
        "note": (
            "The compressed size is a theoretical entropy estimate from model "
            "log-probabilities, not an arithmetic-coded decodable file. "
            "For paper-style comparison, chunk-size should be 2048 and truncated_chunks "
            "should remain zero."
        ),
    }
    if args.classical_baselines:
        final_payload["classical_baselines"] = classical_baselines(input_path, args.limit_bytes)

    write_json(output_path, final_payload)
    write_checkpoint(checkpoint_path, state, args)

    print("\nFinal metrics")
    print(json.dumps(final_payload["metrics"], indent=2, sort_keys=True))
    print(f"Result written to: {output_path}")
    print(f"Checkpoint written to: {checkpoint_path}")


if __name__ == "__main__":
    main()
