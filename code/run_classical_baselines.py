#!/usr/bin/env python3
"""Run and verify full-file gzip, bzip2, and LZMA2 baselines."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import shutil
import sys
import time
from pathlib import Path


BUFFER_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while block := handle.read(BUFFER_SIZE):
        digest.update(block)
    return digest.hexdigest()


def input_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def compress_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as raw_dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, compresslevel=9, mtime=0) as dst:
            shutil.copyfileobj(src, dst, length=BUFFER_SIZE)


def compress_bz2(source: Path, destination: Path) -> None:
    with source.open("rb") as src, bz2.open(destination, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst, length=BUFFER_SIZE)


def compress_lzma2(source: Path, destination: Path) -> None:
    with source.open("rb") as src, lzma.open(
        destination, "wb", format=lzma.FORMAT_XZ, preset=9
    ) as dst:
        shutil.copyfileobj(src, dst, length=BUFFER_SIZE)


def decoded_sha256(path: Path, method: str) -> str:
    opener = {"gzip": gzip.open, "bz2": bz2.open, "lzma2_xz": lzma.open}[method]
    with opener(path, "rb") as handle:
        return sha256_stream(handle)


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output_dir = Path(args.output_dir)
    result_path = Path(args.result)
    if not source.is_file():
        raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = source.stat().st_size
    original_hash = input_sha256(source)
    methods = {
        "gzip": (output_dir / "enwik9.gz", compress_gzip),
        "bz2": (output_dir / "enwik9.bz2", compress_bz2),
        "lzma2_xz": (output_dir / "enwik9.xz", compress_lzma2),
    }

    results = {}
    for name, (archive, compressor) in methods.items():
        print(f"Running {name}...", flush=True)
        started = time.time()
        compressor(source, archive)
        compression_seconds = time.time() - started

        print(f"Verifying {name}...", flush=True)
        verify_started = time.time()
        restored_hash = decoded_sha256(archive, name)
        verification_seconds = time.time() - verify_started
        verified = restored_hash == original_hash
        if not verified:
            raise RuntimeError(f"Decoded SHA-256 mismatch for {name}")

        compressed_size = archive.stat().st_size
        results[name] = {
            "archive": str(archive),
            "compressed_bytes": compressed_size,
            "compression_rate_percent": 100.0 * compressed_size / original_size,
            "compression_seconds": compression_seconds,
            "verification_seconds": verification_seconds,
            "decoded_sha256": restored_hash,
            "round_trip_verified": verified,
        }
        print(json.dumps({name: results[name]}, indent=2), flush=True)

    payload = {
        "input": str(source),
        "input_bytes": original_size,
        "input_sha256": original_hash,
        "python_version": sys.version,
        "scope": "Unchunked lossless compression of the original enwik9 byte stream.",
        "methods": results,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    main()
