#!/usr/bin/env python3
"""Exhaustively validate a complete corrected V1/V2 dataset root."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.crosshair_canonicalization import (  # noqa: E402
    validate_canonical_v1_shard,
    validate_migrated_v1_shard,
)


V1_COLLECTION = Path("resolution/v1/resolved-collection")
V2_COLLECTION = Path("v2-schema14-b49ddd0/resolution/resolved-collection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-v1-shards", type=int, default=192)
    parser.add_argument("--expected-v2-shards", type=int, default=171)
    parser.add_argument("--expected-v1-images", type=int, default=1_152_179)
    parser.add_argument("--expected-v2-images", type=int, default=2_229_369)
    return parser.parse_args()


def discover(collection: Path) -> list[Path]:
    return sorted(collection.glob("attempts/assignment-*/attempt-*/output/shard-*.tar"))


def migration_validation_job(source: str, corrected: str) -> dict[str, Any]:
    migration = validate_migrated_v1_shard(source, corrected)
    idempotence = validate_canonical_v1_shard(corrected)
    return {"migration": asdict(migration), "idempotence": asdict(idempotence)}


def sha256_pair(source: str, corrected: str) -> dict[str, Any]:
    source_path = Path(source)
    corrected_path = Path(corrected)
    if source_path.stat().st_size != corrected_path.stat().st_size:
        raise RuntimeError(f"V2 file size differs: {source_path} vs {corrected_path}")
    source_digest = hashlib.sha256()
    if os.path.samefile(source_path, corrected_path):
        with source_path.open("rb") as source_stream:
            for block in iter(lambda: source_stream.read(8 * 1024 * 1024), b""):
                source_digest.update(block)
        return {
            "source": str(source_path),
            "corrected": str(corrected_path),
            "bytes": source_path.stat().st_size,
            "sha256": source_digest.hexdigest(),
            "storage_mode": "shared_authoritative_reference",
        }
    corrected_digest = hashlib.sha256()
    with source_path.open("rb") as source_stream, corrected_path.open("rb") as corrected_stream:
        while True:
            source_block = source_stream.read(8 * 1024 * 1024)
            corrected_block = corrected_stream.read(8 * 1024 * 1024)
            if source_block != corrected_block:
                raise RuntimeError(f"V2 bytes differ: {source_path} vs {corrected_path}")
            if not source_block:
                break
            source_digest.update(source_block)
            corrected_digest.update(corrected_block)
    source_hash = source_digest.hexdigest()
    corrected_hash = corrected_digest.hexdigest()
    if source_hash != corrected_hash:
        raise RuntimeError(f"V2 SHA-256 differs: {source_path} vs {corrected_path}")
    return {
        "source": str(source_path),
        "corrected": str(corrected_path),
        "bytes": source_path.stat().st_size,
        "sha256": source_hash,
        "storage_mode": "independent_copy",
    }


def all_file_pairs(source: Path, corrected: Path) -> list[tuple[Path, Path]]:
    source_files = sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
    corrected_files = sorted(
        path.relative_to(corrected) for path in corrected.rglob("*") if path.is_file()
    )
    if source_files != corrected_files:
        missing = sorted(set(source_files) - set(corrected_files))
        extra = sorted(set(corrected_files) - set(source_files))
        raise RuntimeError(f"V2 file structure differs: missing={missing[:10]} extra={extra[:10]}")
    return [(source / relative, corrected / relative) for relative in source_files]


def checksum_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_v1_outer_metadata(corrected_shard: Path) -> dict[str, Any]:
    output_dir = corrected_shard.parent
    checksum_lines = (output_dir / "checksums.md5").read_text(encoding="utf-8").splitlines()
    matches = [line.split()[0] for line in checksum_lines if line.split()[-1] == corrected_shard.name]
    if len(matches) != 1:
        raise RuntimeError(f"Invalid corrected checksums.md5: {output_dir}")
    actual_md5 = checksum_md5(corrected_shard)
    if matches[0] != actual_md5:
        raise RuntimeError(f"Corrected checksum mismatch: {corrected_shard}")
    dataset = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))
    if dataset.get("observation_count") is None:
        raise RuntimeError(f"Corrected dataset.json lacks observation_count: {output_dir}")
    if "crosshair_canonicalization" not in dataset:
        raise RuntimeError(f"Corrected dataset.json lacks migration provenance: {output_dir}")
    return {
        "shard": str(corrected_shard),
        "md5": actual_md5,
        "observation_count": int(dataset["observation_count"]),
    }


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    started = time.monotonic()
    source_root = args.source_root.resolve()
    corrected_root = args.corrected_root.resolve()
    source_v1 = source_root / V1_COLLECTION
    corrected_v1 = corrected_root / V1_COLLECTION
    source_v2 = source_root / V2_COLLECTION
    corrected_v2 = corrected_root / V2_COLLECTION
    source_v1_shards = discover(source_v1)
    corrected_v1_shards = discover(corrected_v1)
    source_v2_shards = discover(source_v2)
    corrected_v2_shards = discover(corrected_v2)
    if len(source_v1_shards) != args.expected_v1_shards:
        raise SystemExit(f"Source V1 shard count is {len(source_v1_shards)}")
    if len(corrected_v1_shards) != args.expected_v1_shards:
        raise SystemExit(f"Corrected V1 shard count is {len(corrected_v1_shards)}")
    if len(source_v2_shards) != args.expected_v2_shards:
        raise SystemExit(f"Source V2 shard count is {len(source_v2_shards)}")
    if len(corrected_v2_shards) != args.expected_v2_shards:
        raise SystemExit(f"Corrected V2 shard count is {len(corrected_v2_shards)}")
    source_v1_by_relative = {path.relative_to(source_v1): path for path in source_v1_shards}
    corrected_v1_by_relative = {
        path.relative_to(corrected_v1): path for path in corrected_v1_shards
    }
    if source_v1_by_relative.keys() != corrected_v1_by_relative.keys():
        raise SystemExit("Corrected V1 shard identities differ from source")

    v1_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                migration_validation_job,
                str(source),
                str(corrected_v1_by_relative[relative]),
            ): relative
            for relative, source in source_v1_by_relative.items()
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            v1_results.append(result)
            print(
                f"V1 VALIDATION shards={len(v1_results)}/{len(futures)} "
                f"images={result['migration']['image_count']:,} path={futures[future]}",
                flush=True,
            )
    v1_results.sort(key=lambda result: result["migration"]["source_shard"])
    v1_images = sum(result["migration"]["image_count"] for result in v1_results)
    changed_pixels = sum(
        result["migration"]["changed_pixels_total"] for result in v1_results
    )
    if v1_images != args.expected_v1_images:
        raise SystemExit(f"Corrected V1 image count is {v1_images:,}")
    if changed_pixels != args.expected_v1_images * 17:
        raise SystemExit(f"Corrected V1 changed pixel total is {changed_pixels:,}")
    if any(result["idempotence"]["zero_repairs"] != 0 for result in v1_results):
        raise SystemExit("Corrected V1 idempotence pass requested repairs")

    v1_outer = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(verify_v1_outer_metadata, path) for path in corrected_v1_shards]
        for future in concurrent.futures.as_completed(futures):
            v1_outer.append(future.result())
    if sum(item["observation_count"] for item in v1_outer) != args.expected_v1_images:
        raise SystemExit("Corrected V1 dataset.json observation counts do not match")

    v2_pairs = all_file_pairs(source_v2, corrected_v2)
    v2_storage_mode = (
        "shared_authoritative_reference"
        if source_v2.resolve() == corrected_v2.resolve()
        else "independent_copy"
    )
    v2_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(sha256_pair, str(source), str(corrected)): source
            for source, corrected in v2_pairs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            v2_results.append(result)
            if len(v2_results) % 25 == 0 or len(v2_results) == len(futures):
                print(f"V2 BYTE VALIDATION files={len(v2_results)}/{len(futures)}", flush=True)
    v2_dataset_images = 0
    for shard in corrected_v2_shards:
        dataset = json.loads((shard.parent / "dataset.json").read_text(encoding="utf-8"))
        v2_dataset_images += int(dataset["observation_count"])
    if v2_dataset_images != args.expected_v2_images:
        raise SystemExit(f"V2 dataset.json image count is {v2_dataset_images:,}")

    report = {
        "validation_version": 1,
        "source_root": str(source_root),
        "corrected_root": str(corrected_root),
        "v1_shards": len(v1_results),
        "v1_images": v1_images,
        "v1_changed_pixels": changed_pixels,
        "v1_zero_repairs_second_pass": True,
        "v1_outer_checksums_verified": len(v1_outer),
        "v2_shards": len(corrected_v2_shards),
        "v2_images": v2_dataset_images,
        "v2_files_byte_identical": len(v2_results),
        "v2_storage_mode": v2_storage_mode,
        "combined_images": v1_images + v2_dataset_images,
        "elapsed_seconds": time.monotonic() - started,
        "all_gates_passed": True,
        "v1_results": v1_results,
        "v1_outer": sorted(v1_outer, key=lambda item: item["shard"]),
        "v2_results": sorted(v2_results, key=lambda item: item["source"]),
    }
    if report["combined_images"] != args.expected_v1_images + args.expected_v2_images:
        raise SystemExit("Combined image count differs from expected")
    write_atomic(args.output, report)
    print(
        f"VALIDATION COMPLETE v1={v1_images:,} v2={v2_dataset_images:,} "
        f"combined={report['combined_images']:,} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
