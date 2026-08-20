#!/usr/bin/env python3
"""Create a deterministic episode-safe 90/5/5 V1+V2 split manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.tar_images import (  # noqa: E402
    AUTHORITATIVE_COLLECTIONS,
    discover_shards,
)


FRACTIONS = {"train": 0.90, "validation": 0.05, "test": 0.05}
SEED = 17082026


def stable_digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def read_episodes(shard: Path) -> list[dict[str, object]]:
    with tarfile.open(shard, "r") as archive:
        extracted = archive.extractfile("episodes.parquet")
        if extracted is None:
            raise RuntimeError(f"episodes.parquet is missing from {shard}")
        table = pq.read_table(
            io.BytesIO(extracted.read()), columns=["episode_id", "observation_count"]
        )
    result = []
    for episode_id, count in zip(
        table.column("episode_id").to_pylist(),
        table.column("observation_count").to_pylist(),
        strict=True,
    ):
        observations = int(count)
        if observations < 1:
            raise RuntimeError(f"Episode {episode_id} in {shard} is empty")
        result.append({"episode_id": str(episode_id), "observation_count": observations})
    return result


def _episode_quotas(episodes: int) -> dict[str, int]:
    """Largest-remainder quotas with deterministic handling of exact ties."""
    raw = {name: episodes * fraction for name, fraction in FRACTIONS.items()}
    quotas = {name: math.floor(value) for name, value in raw.items()}
    remaining = episodes - sum(quotas.values())
    order = sorted(
        FRACTIONS,
        key=lambda name: (-(raw[name] - quotas[name]), stable_digest(f"quota:{name}")),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def assign_grouped_episodes(
    records_by_collection: dict[str, list[dict[str, object]]],
    *,
    refinement_iterations: int = 3_000_000,
) -> dict[str, str]:
    """Assign each underlying episode ID once across every dataset version.

    Episode quotas prevent a few unusually long videos from making the held-out
    sets unrepresentative. A deterministic largest-first initialization and
    pair-swap refinement then make both the frame percentages and episode
    presence percentages in V1 and V2 independently approximate 90/5/5 as
    closely as their indivisible episode groups allow.
    """
    dimensions = ("v1_images", "v2_images", "v1_episodes", "v2_episodes")
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in dimensions}
    )
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for collection, records in records_by_collection.items():
        if collection not in ("v1", "v2"):
            raise ValueError(f"Unsupported collection: {collection}")
        for item in records:
            episode_id = str(item["episode_id"])
            occurrences[(collection, episode_id)] += 1
            if occurrences[(collection, episode_id)] > 1:
                raise RuntimeError(
                    f"Episode {episode_id} occurs more than once in {collection}"
                )
            grouped[episode_id][f"{collection}_images"] += int(item["observation_count"])
            grouped[episode_id][f"{collection}_episodes"] = 1
    if not grouped:
        raise RuntimeError("No underlying episodes were discovered")

    totals = {
        dimension: sum(values[dimension] for values in grouped.values())
        for dimension in dimensions
    }
    if any(value < 1 for value in totals.values()):
        raise RuntimeError("Both V1 and V2 must contain observations")
    targets = {
        collection: {
            split: totals[collection] * fraction
            for split, fraction in FRACTIONS.items()
        }
        for collection in totals
    }
    assigned = {
        collection: {split: 0 for split in FRACTIONS}
        for collection in totals
    }
    quotas = _episode_quotas(len(grouped))
    used = {split: 0 for split in FRACTIONS}
    members: dict[str, list[str]] = {split: [] for split in FRACTIONS}
    result: dict[str, str] = {}

    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -(item[1]["v1_images"] + item[1]["v2_images"]),
            stable_digest(item[0]),
        ),
    )
    for episode_id, observations in ordered:
        candidates = []
        for split in FRACTIONS:
            if used[split] >= quotas[split]:
                continue
            delta = 0.0
            for collection, value in observations.items():
                target = targets[collection][split]
                before = (assigned[collection][split] - target) / target
                after = (assigned[collection][split] + value - target) / target
                delta += after * after - before * before
            candidates.append((delta, stable_digest(f"{episode_id}:{split}"), split))
        split = min(candidates)[2]
        result[episode_id] = split
        members[split].append(episode_id)
        used[split] += 1
        for collection, value in observations.items():
            assigned[collection][split] += value

    # Swaps preserve the exact episode quotas. Only strictly improving swaps are
    # accepted, making the output deterministic and monotonically better.
    rng = random.Random(SEED)
    split_names = tuple(FRACTIONS)
    for _ in range(refinement_iterations):
        first, second = rng.sample(split_names, 2)
        first_index = rng.randrange(len(members[first]))
        second_index = rng.randrange(len(members[second]))
        first_id = members[first][first_index]
        second_id = members[second][second_index]
        first_values = grouped[first_id]
        second_values = grouped[second_id]
        old_score = new_score = 0.0
        for collection, total in totals.items():
            first_target = targets[collection][first]
            second_target = targets[collection][second]
            old_score += ((assigned[collection][first] - first_target) / total) ** 2
            old_score += ((assigned[collection][second] - second_target) / total) ** 2
            new_first = (
                assigned[collection][first]
                - first_values[collection]
                + second_values[collection]
            )
            new_second = (
                assigned[collection][second]
                - second_values[collection]
                + first_values[collection]
            )
            new_score += ((new_first - first_target) / total) ** 2
            new_score += ((new_second - second_target) / total) ** 2
        if new_score >= old_score:
            continue
        for collection in totals:
            assigned[collection][first] += (
                second_values[collection] - first_values[collection]
            )
            assigned[collection][second] += (
                first_values[collection] - second_values[collection]
            )
        members[first][first_index] = second_id
        members[second][second_index] = first_id
        result[first_id] = second
        result[second_id] = first

    if {name: len(values) for name, values in members.items()} != quotas:
        raise RuntimeError("Grouped episode quota invariant failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    shards = discover_shards(root)
    canonical_names: dict[Path, str] = {}
    for relative in AUTHORITATIVE_COLLECTIONS:
        for candidate in (root / relative).glob(
            "attempts/assignment-*/attempt-*/output/shard-*.tar"
        ):
            resolved = candidate.resolve()
            canonical = str(candidate.relative_to(root)).replace("\\", "/")
            if resolved in canonical_names and canonical_names[resolved] != canonical:
                raise RuntimeError(f"Resolved shard has multiple canonical links: {resolved}")
            canonical_names[resolved] = canonical
    if set(canonical_names) != set(shards):
        raise RuntimeError("Canonical shard links and discovered resolved shards differ")
    collection_for_shard: dict[Path, str] = {}
    for relative in AUTHORITATIVE_COLLECTIONS:
        collection_root = (root / relative).resolve()
        for shard in shards:
            if shard.is_relative_to(collection_root):
                collection_for_shard[shard] = "v1" if "v1" in str(relative) else "v2"
    if set(collection_for_shard) != set(shards):
        raise RuntimeError("Could not bind every shard to V1 or V2")

    records_by_collection: dict[str, list[dict[str, object]]] = defaultdict(list)
    shard_records: dict[Path, list[dict[str, object]]] = {}
    if args.workers < 1:
        raise ValueError("Metadata workers must be positive")
    scanned: dict[Path, list[dict[str, object]]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(shards))) as executor:
        futures = {executor.submit(read_episodes, shard): shard for shard in shards}
        completed = 0
        for future in as_completed(futures):
            shard = futures[future]
            scanned[shard] = future.result()
            completed += 1
            print(
                f"SPLIT_SCAN completed={completed}/{len(shards)} "
                f"episodes={len(scanned[shard])}",
                flush=True,
            )
    for shard in shards:
        relative_path = canonical_names[shard]
        episodes = scanned[shard]
        for episode in episodes:
            records_by_collection[collection_for_shard[shard]].append(episode)
        shard_records[shard] = episodes

    grouped_assignments = assign_grouped_episodes(dict(records_by_collection))
    for records in records_by_collection.values():
        for episode in records:
            episode["split"] = grouped_assignments[str(episode["episode_id"])]

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for collection in ("v1", "v2"):
        counts[collection] = {}
        for split in FRACTIONS:
            selected = [item for item in records_by_collection[collection] if item["split"] == split]
            counts[collection][split] = {
                "episodes": len(selected),
                "images": sum(int(item["observation_count"]) for item in selected),
            }
    counts["combined"] = {
        split: {
            "episodes": sum(value == split for value in grouped_assignments.values()),
            "images": sum(counts[c][split]["images"] for c in ("v1", "v2")),
        }
        for split in FRACTIONS
    }
    payload = {
        "manifest_version": 2,
        "dataset_root_name": root.name,
        "seed": SEED,
        "fractions": FRACTIONS,
        "assignment_unit": "underlying_episode_id_across_v1_v2",
        "assignment_algorithm": (
            "grouped-episode-quotas-largest-first-deterministic-pair-refinement-v2"
        ),
        "counts": counts,
        "shards": [
            {
                "relative_path": canonical_names[shard],
                "size_bytes": shard.stat().st_size,
                "collection": collection_for_shard[shard],
                "episodes": [
                    {
                        "episode_id": item["episode_id"],
                        "observation_count": item["observation_count"],
                        "split": item["split"],
                    }
                    for item in sorted(shard_records[shard], key=lambda value: str(value["episode_id"]))
                ],
            }
            for shard in shards
        ],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError("Refusing to overwrite a different immutable split manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(args.output), "sha256": digest, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
