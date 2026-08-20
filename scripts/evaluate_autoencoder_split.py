#!/usr/bin/env python3
"""Evaluate a locked autoencoder checkpoint on a complete held-out episode split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.episode_splits import load_episode_split_manifest
from world_model_trajectory.data.tar_images import TarImageDataset, discover_shards
from world_model_trajectory.models.final_factorized_autoencoder import FinalFactorizedAutoencoder
from world_model_trajectory.models.final_factorized_autoencoder import fixed_crosshair_stencil
from world_model_trajectory.training.final_factorized_objective import (
    exact_core_confusion_counts, loss_components, trajectory_confusion_counts,
    trajectory_metrics_from_counts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--winner-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.split == "test":
        if args.winner_lock is None:
            raise RuntimeError("Test evaluation requires the immutable winner lock")
        lock = json.loads(args.winner_lock.read_text())
        if lock.get("status") != "winner_locked_before_test":
            raise RuntimeError("Winner lock has invalid status")
        if lock.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Test checkpoint differs from the locked winner")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract_path = args.checkpoint.parent / "run-contract.json"
    contract = json.loads(contract_path.read_text())
    config = contract["config"]
    model_config = config.get("model", config)
    model = FinalFactorizedAutoencoder(
        scene_width=int(model_config["scene_width"]),
        trajectory_width=int(model_config["trajectory_width"]),
        scene_latent_channels=int(model_config["scene_channels"]),
        trajectory_latent_channels=int(model_config["trajectory_channels"]),
        scene_latent_resolution=int(model_config["scene_resolution"]),
        trajectory_latent_resolution=int(model_config["trajectory_resolution"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device, memory_format=torch.channels_last).eval()
    root = args.dataset_root.resolve(); shards = discover_shards(root)
    selection = load_episode_split_manifest(
        args.split_manifest, dataset_root=root, discovered_shards=shards
    )
    if contract.get("split_manifest_sha256") != selection.manifest_sha256:
        raise RuntimeError("Checkpoint was trained against a different split manifest")
    dataset = TarImageDataset(
        shards, seed=int(config["seed"]), shuffle_shards=False, shuffle_buffer=1,
        episode_splits_by_shard=selection.episode_splits_by_shard,
        include_splits=(args.split,),
    )
    training = config["training"]
    batch_size = int(training.get("batch_size_per_rank", training.get("batch_size")))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=2 if args.workers else None)
    weights = {name: float(value) for name, value in config["loss_weights"].items()}
    objective_config = config["objective"]
    objective = {
        "target_score_threshold": float(objective_config["target_score_threshold"]),
        "antialias_support_radius": int(objective_config["antialias_support_radius"]),
        "positive_weight_cap": float(objective_config["positive_weight_cap"]),
    }
    sums = {name: 0.0 for name in weights}; images_seen = 0
    counts: dict[str, float] = defaultdict(float); exact: dict[str, float] = defaultdict(float)
    crosshair_quantized_mismatches = 0
    crosshair_max_abs_error = 0.0
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(images); components = loss_components(output, images, **objective)
            crosshair = fixed_crosshair_stencil(device=device)[0, 0]
            predicted_crosshair = output.image[:, :, crosshair]
            target_crosshair = images[:, :, crosshair]
            crosshair_max_abs_error = max(
                crosshair_max_abs_error,
                float((predicted_crosshair - target_crosshair).abs().max()),
            )
            crosshair_quantized_mismatches += int(
                (
                    predicted_crosshair.mul(255).round().to(torch.uint8)
                    != target_crosshair.mul(255).round().to(torch.uint8)
                ).sum()
            )
            images_seen += len(images)
            for name in weights: sums[name] += float(components[name]) * len(images)
            for destination, source in (
                (counts, trajectory_confusion_counts(
                    output.trajectory_probability, images,
                    target_score_threshold=float(objective["target_score_threshold"]),
                    antialias_support_radius=int(objective["antialias_support_radius"]),
                )),
                (exact, exact_core_confusion_counts(output.trajectory_probability, images)),
            ):
                for name, value in source.items(): destination[name] += value
            if images_seen % 16384 < batch_size:
                print(f"EVAL_PROGRESS split={args.split} images={images_seen}", flush=True)
    if images_seen != selection.image_counts[args.split]:
        raise RuntimeError(f"Evaluation count mismatch: {images_seen} != {selection.image_counts[args.split]}")
    means = {name: value / images_seen for name, value in sums.items()}
    report = {
        "status": "complete", "split": args.split, "images": images_seen,
        "episodes": selection.episode_counts[args.split],
        "split_manifest_sha256": selection.manifest_sha256,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "components": means,
        "weighted_total": sum(weights[name] * means[name] for name in weights),
        "trajectory": trajectory_metrics_from_counts(counts),
        "trajectory_exact_core": trajectory_metrics_from_counts(exact),
        "crosshair_audit": {
            "quantized_channel_mismatches": crosshair_quantized_mismatches,
            "max_abs_float_error": crosshair_max_abs_error,
        },
    }
    if crosshair_quantized_mismatches:
        raise RuntimeError(
            "Crosshair reconstruction changed "
            f"{crosshair_quantized_mismatches} quantized channel values"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EVALUATION_COMPLETE " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
