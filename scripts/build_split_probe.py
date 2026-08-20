#!/usr/bin/env python3
"""Build an immutable trajectory-balanced probe from one manifest split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.episode_splits import load_episode_split_manifest
from world_model_trajectory.data.tar_images import TarImageDataset, discover_shards
from world_model_trajectory.models.final_factorized_autoencoder import TRAJECTORY_RGB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17082026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.images < 2 or args.images % 2:
        raise ValueError("Probe size must be a positive even number")
    # Tiny per-frame tensor operations become dramatically slower when PyTorch
    # launches a full CPU thread team for every decoded image.
    torch.set_num_threads(1)
    root = args.dataset_root.resolve()
    shards = discover_shards(root)
    selection = load_episode_split_manifest(
        args.split_manifest, dataset_root=root, discovered_shards=shards
    )
    dataset = TarImageDataset(
        shards, seed=args.seed, shuffle_shards=False, shuffle_buffer=1,
        episode_splits_by_shard=selection.episode_splits_by_shard,
        include_splits=(args.split,),
    )
    if args.workers < 0:
        raise ValueError("Probe workers cannot be negative")
    loader = DataLoader(
        dataset, batch_size=32, num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        in_order=False,
    )
    rng = random.Random(args.seed)
    capacity = args.images // 2
    reservoirs: dict[str, list[torch.Tensor]] = {"visible": [], "empty": []}
    seen = {"visible": 0, "empty": 0}
    color = torch.tensor(TRAJECTORY_RGB, dtype=torch.uint8).view(3, 1, 1)
    scanned = 0
    next_progress = 25000
    for batch in loader:
        batch_uint8 = batch.mul(255).round().to(torch.uint8)
        visible = (batch_uint8 == color).all(dim=1).flatten(1).any(dim=1)
        for uint8, is_visible in zip(batch_uint8, visible, strict=True):
            scanned += 1
            category = "visible" if bool(is_visible) else "empty"
            seen[category] += 1
            bucket = reservoirs[category]
            if len(bucket) < capacity:
                bucket.append(uint8)
            else:
                position = rng.randrange(seen[category])
                if position < capacity:
                    bucket[position] = uint8
        if scanned >= next_progress:
            print(f"PROBE_SCAN images={scanned} visible={seen['visible']} empty={seen['empty']}", flush=True)
            next_progress += 25000
    if any(len(bucket) != capacity for bucket in reservoirs.values()):
        raise RuntimeError(f"Could not fill balanced probe: {seen}")
    images = reservoirs["visible"] + reservoirs["empty"]
    rng.shuffle(images)
    payload = {
        "probe_version": 1,
        "split": args.split,
        "split_manifest": str(selection.manifest_path),
        "split_manifest_sha256": selection.manifest_sha256,
        "seed": args.seed,
        "sampling": "full-split-two-class-reservoir-exact-trajectory-core",
        "source_images_scanned": scanned,
        "source_class_counts": seen,
        "probe_class_counts": {"visible": capacity, "empty": capacity},
        "images_uint8": torch.stack(images),
    }
    if scanned != selection.image_counts[args.split]:
        raise RuntimeError(
            f"Manifest/stream count mismatch: {scanned} != {selection.image_counts[args.split]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite an existing immutable probe")
    torch.save(payload, args.output)
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(json.dumps({"output": str(args.output), "sha256": checksum,
                      "scanned": scanned, "source_class_counts": seen}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
