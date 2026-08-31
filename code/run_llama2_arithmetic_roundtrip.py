#!/usr/bin/env python3
"""Arithmetic-code a Llama 2 token stream and verify a byte-exact round trip."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import constriction
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAGIC = b"LMAC001\n"
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit-bytes", type=int, default=1024000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--progress-every-tokens", type=int, default=1000)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalise_and_pack_high_bits(raw: bytes) -> tuple[bytes, bytes, int]:
    mask = bytearray((len(raw) + 7) // 8)
    normalised = bytearray(len(raw))
    mapped = 0
    for index, value in enumerate(raw):
        normalised[index] = value & 0x7F
        if value > 127:
            mask[index // 8] |= 1 << (index % 8)
            mapped += 1
    return bytes(normalised), bytes(mask), mapped


def restore_high_bits(normalised: bytes, mask: bytes) -> bytes:
    restored = bytearray(normalised)
    for index in range(len(restored)):
        if mask[index // 8] & (1 << (index % 8)):
            restored[index] |= 0x80
    return bytes(restored)


def local_distribution(logits: torch.Tensor, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probabilities, k=top_k, sorted=True)
    tail = torch.clamp(1.0 - top_probs.sum(), min=torch.finfo(torch.float32).eps)
    local_probs = torch.cat((top_probs, tail.unsqueeze(0)))
    local_probs = local_probs / local_probs.sum()
    return (
        local_probs.detach().cpu().numpy().astype(np.float32, copy=False),
        top_ids.detach().cpu().numpy().astype(np.int32, copy=False),
    )


def categorical(probabilities: np.ndarray):
    return constriction.stream.model.Categorical(probabilities, perfect=False)


def write_archive(
    path: Path,
    metadata: dict,
    arithmetic_words: np.ndarray,
    escaped_ids: np.ndarray,
    compressed_mask: bytes,
) -> dict[str, int]:
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    words = np.asarray(arithmetic_words, dtype="<u4").tobytes()
    escapes = np.asarray(escaped_ids, dtype="<u2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(U32.pack(len(header)))
        handle.write(header)
        handle.write(U64.pack(len(words)))
        handle.write(words)
        handle.write(U64.pack(len(escapes)))
        handle.write(escapes)
        handle.write(U64.pack(len(compressed_mask)))
        handle.write(compressed_mask)
    return {
        "header_bytes": len(MAGIC) + U32.size + len(header) + 3 * U64.size,
        "arithmetic_bytes": len(words),
        "escape_bytes": len(escapes),
        "high_bit_mask_bytes": len(compressed_mask),
        "archive_bytes": path.stat().st_size,
    }


def read_archive(path: Path) -> tuple[dict, np.ndarray, np.ndarray, bytes]:
    with path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid archive magic")
        header = json.loads(handle.read(U32.unpack(handle.read(U32.size))[0]))
        words = np.frombuffer(handle.read(U64.unpack(handle.read(U64.size))[0]), dtype="<u4").copy()
        escapes = np.frombuffer(
            handle.read(U64.unpack(handle.read(U64.size))[0]), dtype="<u2"
        ).copy()
        compressed_mask = handle.read(U64.unpack(handle.read(U64.size))[0])
        if handle.read(1):
            raise ValueError("Unexpected trailing archive data")
    return header, words, escapes, compressed_mask


def model_step(model, token: torch.Tensor, cache):
    with torch.inference_mode():
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
    return output.logits[0, -1], output.past_key_values


def encode(
    raw: bytes,
    model,
    tokenizer,
    device: str,
    chunk_size: int,
    top_k: int,
    progress_every: int,
):
    normalised, high_mask, mapped = normalise_and_pack_high_bits(raw)
    encoder = constriction.stream.queue.RangeEncoder()
    escaped_ids: list[int] = []
    chunk_token_counts: list[int] = []
    chunk_byte_lengths: list[int] = []
    chunk_prefixes_hex: list[str] = []
    total_model_bits = 0.0
    topk_hits = 0
    scored_tokens = 0
    started = time.time()

    for chunk_start in range(0, len(normalised), chunk_size):
        chunk = normalised[chunk_start : chunk_start + chunk_size]
        text = chunk.decode("ascii")
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if tokenizer.bos_token_id is None or int(ids[0]) != tokenizer.bos_token_id:
            raise RuntimeError("Expected the Llama tokenizer to prepend exactly one BOS token")
        tokenizer_decoded = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).encode("ascii")
        if tokenizer_decoded == chunk:
            missing_prefix = b""
        elif chunk.endswith(tokenizer_decoded):
            missing_prefix = chunk[: len(chunk) - len(tokenizer_decoded)]
            if any(value not in {9, 10, 13, 32} for value in missing_prefix):
                raise RuntimeError(
                    f"Tokenizer changed non-whitespace data at byte offset {chunk_start}"
                )
        else:
            raise RuntimeError(
                "Tokenizer changed data beyond a removable leading-whitespace prefix at "
                f"byte offset {chunk_start}; original_prefix={chunk[:32]!r}, "
                f"decoded_prefix={tokenizer_decoded[:32]!r}"
            )

        targets = ids[1:].tolist()
        chunk_token_counts.append(len(targets))
        chunk_byte_lengths.append(len(chunk))
        chunk_prefixes_hex.append(missing_prefix.hex())
        current = torch.tensor([[tokenizer.bos_token_id]], device=device)
        cache = None

        for target in targets:
            logits, cache = model_step(model, current, cache)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            total_model_bits += float((-log_probs[target] / np.log(2.0)).item())
            probabilities, top_ids = local_distribution(logits, top_k)
            matches = np.flatnonzero(top_ids == target)
            if matches.size:
                local_symbol = int(matches[0])
                topk_hits += 1
            else:
                local_symbol = top_k
                if target > np.iinfo(np.uint16).max:
                    raise RuntimeError("Escaped token id does not fit in uint16")
                escaped_ids.append(target)
            encoder.encode(np.asarray([local_symbol], dtype=np.int32), categorical(probabilities))
            scored_tokens += 1
            current = torch.tensor([[target]], device=device)
            if scored_tokens % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"encode tokens={scored_tokens} chunks={len(chunk_token_counts)} "
                    f"top{top_k}={100.0 * topk_hits / scored_tokens:.2f}% "
                    f"tokens/s={scored_tokens / elapsed:.2f}",
                    flush=True,
                )

    return {
        "normalised": normalised,
        "high_mask": high_mask,
        "mapped": mapped,
        "words": encoder.get_compressed(),
        "escaped_ids": np.asarray(escaped_ids, dtype=np.uint16),
        "chunk_token_counts": chunk_token_counts,
        "chunk_byte_lengths": chunk_byte_lengths,
        "chunk_prefixes_hex": chunk_prefixes_hex,
        "total_model_bits": total_model_bits,
        "topk_hits": topk_hits,
        "scored_tokens": scored_tokens,
        "elapsed_seconds": time.time() - started,
    }


def decode(header, words, escaped_ids, compressed_mask, model, tokenizer, device, progress_every):
    decoder = constriction.stream.queue.RangeDecoder(words)
    escape_index = 0
    decoded_chunks: list[bytes] = []
    decoded_tokens = 0
    started = time.time()
    top_k = int(header["top_k"])

    for chunk_index, (token_count, byte_length, prefix_hex) in enumerate(
        zip(
            header["chunk_token_counts"],
            header["chunk_byte_lengths"],
            header["chunk_prefixes_hex"],
            strict=True,
        )
    ):
        token_ids = [tokenizer.bos_token_id]
        current = torch.tensor([[tokenizer.bos_token_id]], device=device)
        cache = None
        for _ in range(token_count):
            logits, cache = model_step(model, current, cache)
            probabilities, top_ids = local_distribution(logits, top_k)
            local_symbol = int(decoder.decode(categorical(probabilities)))
            if local_symbol == top_k:
                if escape_index >= len(escaped_ids):
                    raise RuntimeError("Escape side stream exhausted")
                token = int(escaped_ids[escape_index])
                escape_index += 1
            else:
                token = int(top_ids[local_symbol])
            token_ids.append(token)
            current = torch.tensor([[token]], device=device)
            decoded_tokens += 1
            if decoded_tokens % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"decode tokens={decoded_tokens} chunks={chunk_index + 1} "
                    f"tokens/s={decoded_tokens / elapsed:.2f}",
                    flush=True,
                )
        tokenizer_decoded = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).encode("ascii")
        chunk = bytes.fromhex(prefix_hex) + tokenizer_decoded
        if len(chunk) != byte_length:
            raise RuntimeError(
                f"Decoded chunk length mismatch at chunk {chunk_index}: "
                f"expected {byte_length}, got {len(chunk)}"
            )
        decoded_chunks.append(chunk)

    if escape_index != len(escaped_ids):
        raise RuntimeError("Unused values remain in escape side stream")
    normalised = b"".join(decoded_chunks)
    mask = gzip.decompress(compressed_mask)
    expected_mask_bytes = (len(normalised) + 7) // 8
    if len(mask) != expected_mask_bytes:
        raise RuntimeError("High-bit mask length mismatch")
    return restore_high_bits(normalised, mask), time.time() - started


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    archive_path = Path(args.archive)
    result_path = Path(args.result)
    raw = input_path.read_bytes()[: args.limit_bytes]
    if not raw:
        raise ValueError("Input is empty")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} with {dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    encoded = encode(
        raw,
        model,
        tokenizer,
        device,
        args.chunk_size,
        args.top_k,
        args.progress_every_tokens,
    )
    compressed_mask = gzip.compress(encoded["high_mask"], compresslevel=9, mtime=0)
    metadata = {
        "format": "LMAC001",
        "model": args.model,
        "top_k": args.top_k,
        "chunk_size_bytes": args.chunk_size,
        "input_bytes": len(raw),
        "input_sha256": sha256(raw),
        "normalised_sha256": sha256(encoded["normalised"]),
        "chunk_token_counts": encoded["chunk_token_counts"],
        "chunk_byte_lengths": encoded["chunk_byte_lengths"],
        "chunk_prefixes_hex": encoded["chunk_prefixes_hex"],
        "scored_tokens": encoded["scored_tokens"],
        "non_ascii_mapped": encoded["mapped"],
        "escape_count": len(encoded["escaped_ids"]),
    }
    sizes = write_archive(
        archive_path,
        metadata,
        encoded["words"],
        encoded["escaped_ids"],
        compressed_mask,
    )

    header, words, escaped_ids, mask = read_archive(archive_path)
    reconstructed, decode_seconds = decode(
        header,
        words,
        escaped_ids,
        mask,
        model,
        tokenizer,
        device,
        args.progress_every_tokens,
    )
    verified = reconstructed == raw
    reconstructed_hash = sha256(reconstructed)
    if not verified:
        raise RuntimeError("Final reconstructed bytes do not match the original input")

    result = {
        "kind": "arithmetic_round_trip",
        "input": str(input_path),
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "constriction_version": importlib.metadata.version("constriction"),
        "cuda_version": torch.version.cuda,
        "input_bytes": len(raw),
        "input_sha256": sha256(raw),
        "reconstructed_sha256": reconstructed_hash,
        "round_trip_verified": verified,
        "chunk_size_bytes": args.chunk_size,
        "chunks": len(encoded["chunk_token_counts"]),
        "scored_tokens": encoded["scored_tokens"],
        "top_k": args.top_k,
        "topk_hits": encoded["topk_hits"],
        "topk_hit_rate_percent": 100.0 * encoded["topk_hits"] / encoded["scored_tokens"],
        "escape_count": len(encoded["escaped_ids"]),
        "non_ascii_mapped": encoded["mapped"],
        "model_only_bpc": encoded["total_model_bits"] / len(raw),
        "model_only_theoretical_rate_percent": 100.0 * encoded["total_model_bits"] / (8 * len(raw)),
        "archive": str(archive_path),
        **sizes,
        "actual_archive_rate_percent": 100.0 * sizes["archive_bytes"] / len(raw),
        "encode_seconds": encoded["elapsed_seconds"],
        "decode_seconds": decode_seconds,
        "note": (
            "The archive uses a top-k categorical range code with an escape side stream. "
            "A gzip-compressed positional high-bit mask restores the original raw bytes. "
            "Per-chunk framing metadata restores leading whitespace removed by tokenizer decoding."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("Final result", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
