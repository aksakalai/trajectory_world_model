#!/usr/bin/env python3
"""Build a balanced corrected-data probe for renderer-aware calibration."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.crosshair_canonicalization import decode_rgb  # noqa: E402


V1_COLLECTION = Path("resolution/v1/resolved-collection")
V2_COLLECTION = Path("v2-schema14-b49ddd0/resolution/resolved-collection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-per-category", type=int, default=32)
    parser.add_argument("--max-shards-per-collection", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def discover(collection: Path) -> list[Path]:
    shards = sorted(collection.glob("attempts/assignment-*/attempt-*/output/shard-*.tar"))
    if not shards:
        raise RuntimeError(f"No shards under {collection}")
    return shards


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise RuntimeError(f"Could not read {name}")
    return extracted.read()


def choose_evenly(indices: list[int], count: int) -> list[int]:
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [indices[int(position)] for position in positions]


def collect(
    collection: Path,
    categories: dict[str, Callable[[dict[str, object]], bool]],
    *,
    count: int,
    max_shards: int,
    seed: int,
) -> dict[str, list[tuple[np.ndarray, dict[str, object]]]]:
    found: dict[str, list[tuple[np.ndarray, dict[str, object]]]] = defaultdict(list)
    candidate_shards = discover(collection)
    random.Random(seed).shuffle(candidate_shards)
    for shard in candidate_shards[:max_shards]:
        remaining = {name for name in categories if len(found[name]) < count}
        if not remaining:
            break
        with tarfile.open(shard, "r:*") as archive:
            frames = pq.read_table(io.BytesIO(member_bytes(archive, "frames.parquet")))
            columns = frames.to_pydict()
            matches: dict[str, list[int]] = defaultdict(list)
            for index in range(frames.num_rows):
                row = {name: values[index] for name, values in columns.items()}
                for category in remaining:
                    if categories[category](row):
                        matches[category].append(index)
            for category in remaining:
                need = count - len(found[category])
                for index in choose_evenly(matches[category], need):
                    row = {name: values[index] for name, values in columns.items()}
                    payload = member_bytes(archive, str(row["rgb_key"]))
                    rgb = decode_rgb(payload)
                    metadata = {
                        "category": category,
                        "shard": str(shard),
                        "rgb_key": row["rgb_key"],
                        "crosshair_state": row["crosshair_state"],
                        "cooldown_remaining_steps": row["cooldown_remaining_steps"],
                        "trajectory_visible": row["trajectory_visible"],
                        "q_visibility": row["q_visibility"],
                    }
                    found[category].append((rgb, metadata))
    missing = {name: count - len(found[name]) for name in categories if len(found[name]) < count}
    if missing:
        raise RuntimeError(f"Could not fill calibration categories: {missing}")
    return found


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    v1 = collect(
        root / V1_COLLECTION,
        {
            "v1_ready_no_trajectory": lambda row: row["crosshair_state"] == "Ready"
            and row["cooldown_remaining_steps"] == 0
            and not row["trajectory_visible"],
        },
        count=args.images_per_category,
        max_shards=args.max_shards_per_collection,
        seed=args.seed,
    )
    v2 = collect(
        root / V2_COLLECTION,
        {
            "v2_ready_no_trajectory": lambda row: row["crosshair_state"] == "Ready"
            and not row["trajectory_visible"],
            "v2_ready_trajectory": lambda row: row["crosshair_state"] == "Ready"
            and row["trajectory_visible"],
            "v2_cooldown": lambda row: row["crosshair_state"] == "Cooldown",
        },
        count=args.images_per_category,
        max_shards=args.max_shards_per_collection,
        seed=args.seed,
    )
    all_samples = []
    for category in (
        "v1_ready_no_trajectory",
        "v2_ready_no_trajectory",
        "v2_ready_trajectory",
        "v2_cooldown",
    ):
        all_samples.extend((v1 | v2)[category])
    images = torch.from_numpy(
        np.stack([rgb.transpose(2, 0, 1) for rgb, _ in all_samples])
    ).to(torch.uint8)
    metadata = [item for _, item in all_samples]
    payload = {
        "probe_version": 1,
        "dataset_root": str(root),
        "images_per_category": args.images_per_category,
        "images_uint8": images,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    summary = {
        "output": str(args.output),
        "dataset_root": str(root),
        "image_count": len(images),
        "categories": {
            category: sum(item["category"] == category for item in metadata)
            for category in sorted({item["category"] for item in metadata})
        },
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
