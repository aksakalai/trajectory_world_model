#!/usr/bin/env python3
"""Inventory canonicalization-relevant properties of V1/V2 tar collections."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_MEMBERS = (
    "frames.parquet",
    "transitions.parquet",
    "episodes.parquet",
    "manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-collection", type=Path, required=True)
    parser.add_argument("--v2-collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-shards", action="store_true")
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def discover_shards(collection: Path) -> list[Path]:
    return sorted(collection.glob("attempts/assignment-*/attempt-*/output/shard-*.tar"))


def file_hashes(path: Path, algorithms: tuple[str, ...]) -> dict[str, str]:
    digests = {algorithm: hashlib.new(algorithm) for algorithm in algorithms}
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    return {algorithm: digest.hexdigest() for algorithm, digest in digests.items()}


def schema_text(schema: pa.Schema) -> str:
    return schema.to_string(show_field_metadata=True, show_schema_metadata=True)


def schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema_text(schema).encode("utf-8")).hexdigest()


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"Could not read {name}")
    return extracted.read()


def expected_md5(output_dir: Path, shard: Path) -> str | None:
    checksum_path = output_dir / "checksums.md5"
    if not checksum_path.is_file():
        return None
    matches = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and Path(fields[-1]).name == shard.name:
            matches.append(fields[0])
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checksum for {shard}, found {len(matches)}")
    return matches[0]


def inventory_shard(
    shard: Path,
    *,
    curriculum: str,
    hash_shards: bool,
    verify_md5: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    with tarfile.open(shard, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        missing = sorted(set(REQUIRED_MEMBERS) - set(names))
        if duplicates or missing:
            raise RuntimeError(f"{shard}: duplicates={duplicates[:5]} missing={missing}")
        tables = {
            name: pq.read_table(io.BytesIO(member_bytes(archive, name)))
            for name in REQUIRED_MEMBERS[:3]
        }
        manifest = json.loads(member_bytes(archive, "manifest.json"))

    frames = tables["frames.parquet"]
    transitions = tables["transitions.parquet"]
    episodes = tables["episodes.parquet"]
    image_names = [name for name in names if name.startswith("episodes/") and name.endswith(".webp")]
    if len(image_names) != frames.num_rows:
        raise RuntimeError(
            f"{shard}: {len(image_names)} images != {frames.num_rows} frame rows"
        )
    frame_keys = frames["rgb_key"].to_pylist()
    if image_names != frame_keys:
        raise RuntimeError(f"{shard}: tar image order differs from frames rgb_key order")
    expected_plan_prefix = "movement-v1" if curriculum == "v1" else "trajectory-throw-v2"
    plan_versions = episodes["plan_version"].to_pylist()
    if not plan_versions or any(
        not isinstance(value, str) or not value.startswith(expected_plan_prefix)
        for value in plan_versions
    ):
        raise RuntimeError(f"{shard}: unexpected {curriculum} plan_version")

    states = Counter(frames["crosshair_state"].to_pylist())
    cooldowns = Counter(frames["cooldown_remaining_steps"].to_pylist())
    booleans = {}
    for field in ("q_visibility", "aim_lock_active", "trajectory_visible"):
        if field in frames.column_names:
            booleans[field] = dict(Counter(frames[field].to_pylist()))
    expected_counts = {
        "observation_count": frames.num_rows,
        "transition_count": transitions.num_rows,
        "episode_count": episodes.num_rows,
    }
    for key, actual in expected_counts.items():
        if manifest.get(key) != actual:
            raise RuntimeError(f"{shard}: manifest {key} mismatch")

    md5_expected = expected_md5(shard.parent, shard)
    algorithms = tuple(
        algorithm
        for algorithm, enabled in (("sha256", hash_shards), ("md5", verify_md5))
        if enabled
    )
    hashes = file_hashes(shard, algorithms) if algorithms else {}
    sha256 = hashes.get("sha256")
    md5_actual = hashes.get("md5")
    if verify_md5 and md5_expected != md5_actual:
        raise RuntimeError(f"{shard}: checksums.md5 mismatch")
    return {
        "path": str(shard),
        "bytes": shard.stat().st_size,
        "member_count": len(names),
        "first_member": names[0],
        "last_member": names[-1],
        "image_count": len(image_names),
        "frame_rows": frames.num_rows,
        "transition_rows": transitions.num_rows,
        "episode_rows": episodes.num_rows,
        "frame_schema_sha256": schema_fingerprint(frames.schema),
        "transition_schema_sha256": schema_fingerprint(transitions.schema),
        "episode_schema_sha256": schema_fingerprint(episodes.schema),
        "frame_schema": schema_text(frames.schema),
        "transition_schema": schema_text(transitions.schema),
        "episode_schema": schema_text(episodes.schema),
        "crosshair_states": dict(states),
        "cooldown_values": dict(cooldowns),
        "boolean_values": booleans,
        "plan_versions": sorted(set(plan_versions)),
        "sha256": sha256,
        "md5_expected": md5_expected,
        "md5_actual": md5_actual,
        "elapsed_seconds": time.monotonic() - started,
    }


def inventory_collection(
    collection: Path,
    *,
    curriculum: str,
    hash_shards: bool,
    verify_md5: bool,
    workers: int,
) -> dict[str, Any]:
    collection = collection.resolve()
    shards = discover_shards(collection)
    if not shards:
        raise RuntimeError(f"No shards found under {collection}")
    results = []
    schema_examples: dict[str, dict[str, str]] = defaultdict(dict)
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                inventory_shard,
                shard,
                curriculum=curriculum,
                hash_shards=hash_shards,
                verify_md5=verify_md5,
            ): shard
            for shard in shards
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"INVENTORY curriculum={curriculum} complete={len(results)}/{len(shards)} "
                f"frames={result['frame_rows']:,} elapsed={result['elapsed_seconds']:.1f}s "
                f"path={futures[future]}",
                flush=True,
            )
    results.sort(key=lambda result: result["path"])
    for result in results:
        for table in ("frame", "transition", "episode"):
            fingerprint = result[f"{table}_schema_sha256"]
            schema_examples[table].setdefault(fingerprint, result[f"{table}_schema"])

    aggregate_states: Counter[str] = Counter()
    aggregate_cooldowns: Counter[int] = Counter()
    aggregate_booleans: dict[str, Counter[bool]] = defaultdict(Counter)
    for result in results:
        aggregate_states.update(result["crosshair_states"])
        aggregate_cooldowns.update({int(key): value for key, value in result["cooldown_values"].items()})
        for field, counts in result["boolean_values"].items():
            aggregate_booleans[field].update(
                {key == "True" if isinstance(key, str) else bool(key): value for key, value in counts.items()}
            )
    return {
        "collection": str(collection),
        "curriculum": curriculum,
        "shard_count": len(results),
        "total_bytes": sum(result["bytes"] for result in results),
        "image_count": sum(result["image_count"] for result in results),
        "frame_rows": sum(result["frame_rows"] for result in results),
        "transition_rows": sum(result["transition_rows"] for result in results),
        "episode_rows": sum(result["episode_rows"] for result in results),
        "crosshair_states": dict(aggregate_states),
        "cooldown_values": {str(key): value for key, value in sorted(aggregate_cooldowns.items())},
        "boolean_values": {
            field: {str(key): value for key, value in sorted(counts.items())}
            for field, counts in aggregate_booleans.items()
        },
        "schema_variants": {
            table: {
                fingerprint: {
                    "schema": schema,
                    "shards": [
                        result["path"]
                        for result in results
                        if result[f"{table}_schema_sha256"] == fingerprint
                    ],
                }
                for fingerprint, schema in examples.items()
            }
            for table, examples in schema_examples.items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "shards": results,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    started = time.monotonic()
    v1_report = inventory_collection(
        args.v1_collection,
        curriculum="v1",
        hash_shards=args.hash_shards,
        verify_md5=args.verify_md5,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(args.output.suffix + ".v1-complete")
    checkpoint.write_text(
        json.dumps(
            {
                "report_version": 1,
                "status": "v1_complete_v2_pending",
                "hash_shards": args.hash_shards,
                "verify_md5": args.verify_md5,
                "v1": v1_report,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    v2_report = inventory_collection(
        args.v2_collection,
        curriculum="v2",
        hash_shards=args.hash_shards,
        verify_md5=args.verify_md5,
        workers=args.workers,
    )
    report = {
        "report_version": 1,
        "hash_shards": args.hash_shards,
        "verify_md5": args.verify_md5,
        "v1": v1_report,
        "v2": v2_report,
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
