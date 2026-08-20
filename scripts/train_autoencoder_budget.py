#!/usr/bin/env python3
"""Train or resume one factorized-autoencoder candidate on an exact train budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.episode_splits import load_episode_split_manifest
from world_model_trajectory.data.tar_images import (
    AUTHORITATIVE_COLLECTIONS, TarImageDataset, discover_shards,
)
from world_model_trajectory.models.final_factorized_autoencoder import (
    FinalFactorizedAutoencoder,
)
from world_model_trajectory.training.final_factorized_objective import (
    exact_core_confusion_counts,
    loss_components,
    trajectory_confusion_counts,
    trajectory_metrics_from_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--validation-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-total-images", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_from_config(config: dict[str, object]) -> FinalFactorizedAutoencoder:
    return FinalFactorizedAutoencoder(
        scene_width=int(config["scene_width"]),
        trajectory_width=int(config["trajectory_width"]),
        scene_latent_channels=int(config["scene_channels"]),
        trajectory_latent_channels=int(config["trajectory_channels"]),
        scene_latent_resolution=int(config["scene_resolution"]),
        trajectory_latent_resolution=int(config["trajectory_resolution"]),
    )


def objective_kwargs(config: dict[str, object]) -> dict[str, float | int]:
    value = config["objective"]
    assert isinstance(value, dict)
    return {
        "target_score_threshold": float(value["target_score_threshold"]),
        "antialias_support_radius": int(value["antialias_support_radius"]),
        "positive_weight_cap": float(value["positive_weight_cap"]),
    }


def gradient_norms(model: FinalFactorizedAutoencoder) -> dict[str, float]:
    result = {}
    modules = {
        "scene_encoder": model.scene_branch.encoder,
        "scene_decoder": model.scene_branch.decoder,
        "trajectory_encoder": model.trajectory_branch.encoder,
        "trajectory_decoder": model.trajectory_branch.decoder,
    }
    for name, module in modules.items():
        gradients = [p.grad.float() for p in module.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(item).all() for item in gradients):
            raise RuntimeError(f"Missing or nonfinite gradient in {name}")
        norm = math.sqrt(sum(float(item.square().sum()) for item in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Nonpositive gradient norm in {name}: {norm}")
        result[name] = norm
    return result


@torch.no_grad()
def evaluate_probe(
    model: FinalFactorizedAutoencoder,
    images_uint8: torch.Tensor,
    *,
    batch_size: int,
    weights: dict[str, float],
    objective: dict[str, float | int],
    output_dir: Path,
    label: str,
) -> dict[str, object]:
    model.eval()
    sums = {name: 0.0 for name in weights}
    counts: dict[str, float] = defaultdict(float)
    exact: dict[str, float] = defaultdict(float)
    panels = []
    device = next(model.parameters()).device
    for start in range(0, len(images_uint8), batch_size):
        target = images_uint8[start : start + batch_size].to(
            device, dtype=torch.float32, non_blocking=True
        ).div_(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(target)
            components = loss_components(output, target, **objective)
        for name in weights:
            sums[name] += float(components[name]) * len(target)
        for destination, source in (
            (counts, trajectory_confusion_counts(
                output.trajectory_probability, target,
                target_score_threshold=float(objective["target_score_threshold"]),
                antialias_support_radius=int(objective["antialias_support_radius"]),
            )),
            (exact, exact_core_confusion_counts(output.trajectory_probability, target)),
        ):
            for name, value in source.items():
                destination[name] += value
        if not panels:
            panels = [target[:8].cpu(), output.image[:8].float().cpu(),
                      output.trajectory_probability[:8].float().expand(-1, 3, -1, -1).cpu()]
    means = {name: value / len(images_uint8) for name, value in sums.items()}
    if means["final_crosshair_rgb"] > 1e-8:
        raise RuntimeError("Deterministic crosshair reconstruction is not exact")
    save_image(torch.cat(panels), output_dir / f"validation-{label}.png", nrow=8)
    model.train()
    return {
        "images": len(images_uint8),
        "components": means,
        "weighted_total": sum(weights[name] * means[name] for name in weights),
        "trajectory": trajectory_metrics_from_counts(counts),
        "trajectory_exact_core": trajectory_metrics_from_counts(exact),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.target_total_images < 1:
        raise ValueError("Target image budget must be positive")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    dataset_root = args.dataset_root.resolve()
    shards = discover_shards(dataset_root)
    split = load_episode_split_manifest(
        args.split_manifest, dataset_root=dataset_root, discovered_shards=shards
    )
    if args.target_total_images > split.image_counts["train"]:
        raise ValueError("Target exceeds the complete training split")
    probe_bytes = args.validation_probe.read_bytes()
    probe_payload = torch.load(args.validation_probe, map_location="cpu", weights_only=False)
    if probe_payload.get("split") != "validation":
        raise RuntimeError("Selection probe is not validation-only")
    if probe_payload.get("split_manifest_sha256") != split.manifest_sha256:
        raise RuntimeError("Selection probe and episode split manifest differ")
    probe = probe_payload["images_uint8"]
    if probe.dtype != torch.uint8 or probe.ndim != 4 or len(probe) < 1:
        raise RuntimeError("Invalid validation probe")

    ordered_shards = list(shards)
    seed = int(config["seed"])
    random.Random(seed).shuffle(ordered_shards)
    canonical_names = {
        candidate.resolve(): str(candidate.relative_to(dataset_root)).replace("\\", "/")
        for collection in AUTHORITATIVE_COLLECTIONS
        for candidate in (dataset_root / collection).glob(
            "attempts/assignment-*/attempt-*/output/shard-*.tar"
        )
    }
    if set(canonical_names) != set(shards):
        raise RuntimeError("Canonical shard links and resolved training shards differ")
    source_files = (
        Path(__file__),
        REPOSITORY_ROOT / "src/world_model_trajectory/data/tar_images.py",
        REPOSITORY_ROOT / "src/world_model_trajectory/data/episode_splits.py",
        REPOSITORY_ROOT / "src/world_model_trajectory/models/final_factorized_autoencoder.py",
        REPOSITORY_ROOT / "src/world_model_trajectory/training/final_factorized_objective.py",
    )
    contract = {
        "contract_version": 1,
        "config": config,
        "config_sha256": digest(config_bytes),
        "dataset_root": str(dataset_root),
        "split_manifest": str(split.manifest_path),
        "split_manifest_sha256": split.manifest_sha256,
        "train_images": split.image_counts["train"],
        "validation_probe_sha256": digest(probe_bytes),
        "ordered_shards": [canonical_names[path] for path in ordered_shards],
        "source_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): digest(path.read_bytes())
            for path in source_files
        },
    }
    contract_hash = digest(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    contract["contract_sha256"] = contract_hash
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run-contract.json"
    checkpoint_path = args.output_dir / "latest.pt"
    if args.resume:
        previous = json.loads(contract_path.read_text())
        if previous.get("contract_sha256") != contract_hash:
            raise RuntimeError("Resume contract differs from the original candidate run")
    else:
        unexpected = [p for p in args.output_dir.iterdir() if p.name not in {"train.log"}]
        if unexpected:
            raise RuntimeError(f"Fresh candidate output is not empty: {unexpected[:4]}")
        atomic_json(contract, contract_path)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = model_from_config(config).to(device, memory_format=torch.channels_last)
    if model.contract_version != config["contract_version"]:
        raise RuntimeError("Model/config contract version mismatch")
    optimizer_config = config["optimizer"]
    assert isinstance(optimizer_config, dict)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    images_seen = optimizer_steps = 0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("contract_sha256") != contract_hash:
            raise RuntimeError("Checkpoint contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        images_seen = int(saved["images_seen"])
        optimizer_steps = int(saved["optimizer_steps"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
    if images_seen > args.target_total_images:
        raise RuntimeError("Checkpoint is beyond the requested target")

    training = config["training"]
    assert isinstance(training, dict)
    batch_size = int(training["batch_size"])
    accumulation = int(training["accumulation_steps"])
    weights = {name: float(value) for name, value in config["loss_weights"].items()}
    objective = objective_kwargs(config)
    resume_offset = images_seen
    dataset = TarImageDataset(
        ordered_shards, seed=seed, shuffle_shards=False,
        shuffle_buffer=int(training["shuffle_buffer"]),
        episode_splits_by_shard=split.episode_splits_by_shard,
        include_splits=("train",),
    )
    workers = int(training["workers"])
    if workers < 0:
        raise ValueError("Training workers cannot be negative")
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
        prefetch_factor=int(training["prefetch_factor"]) if workers else None,
        persistent_workers=False,
    )
    initial_images = images_seen
    run_started = time.perf_counter()
    next_checkpoint = (
        (images_seen // int(training["checkpoint_every_images"])) + 1
    ) * int(training["checkpoint_every_images"])
    optimizer.zero_grad(set_to_none=True)
    micro_batches = 0
    last_norms: dict[str, float] = {}

    def save_checkpoint() -> None:
        atomic_torch({
            "contract_sha256": contract_hash,
            "architecture": model.architecture_signature,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "images_seen": images_seen, "optimizer_steps": optimizer_steps,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }, checkpoint_path)

    model.train()
    stream_images = 0
    for images in loader:
        batch_end = stream_images + len(images)
        if batch_end <= resume_offset:
            stream_images = batch_end
            continue
        if stream_images < resume_offset:
            images = images[resume_offset - stream_images :]
        stream_images = batch_end
        remaining = args.target_total_images - images_seen
        if len(images) > remaining:
            images = images[:remaining]
        if not len(images):
            break
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images)
            components = loss_components(output, images, **objective)
            total = sum(weights[name] * components[name] for name in weights)
            (total / accumulation).backward()
        last_norms = gradient_norms(model)
        micro_batches += 1
        images_seen += len(images)
        if micro_batches == accumulation:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1; micro_batches = 0
            if optimizer_steps % int(training["log_every_optimizer_steps"]) == 0:
                elapsed = max(time.perf_counter() - run_started, 1e-9)
                print(f"PROGRESS images={images_seen} target={args.target_total_images} "
                      f"steps={optimizer_steps} rate={(images_seen-initial_images)/elapsed:.1f}img/s "
                      f"loss={float(total):.6f} gradients={json.dumps(last_norms, sort_keys=True)}",
                      flush=True)
            if images_seen >= next_checkpoint and images_seen < args.target_total_images:
                save_checkpoint()
                next_checkpoint += int(training["checkpoint_every_images"])
        if images_seen >= args.target_total_images:
            break
    if micro_batches:
        correction = accumulation / micro_batches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        last_norms = gradient_norms(model)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
    if images_seen != args.target_total_images:
        raise RuntimeError(f"Exact train budget failed: {images_seen} != {args.target_total_images}")
    save_checkpoint()
    metrics = evaluate_probe(
        model, probe, batch_size=batch_size, weights=weights, objective=objective,
        output_dir=args.output_dir, label=f"{images_seen:09d}",
    )
    record = {
        "status": "complete",
        "target_total_images": args.target_total_images,
        "images_seen": images_seen,
        "optimizer_steps": optimizer_steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": model.architecture_signature,
        "validation_probe": metrics,
        "last_gradient_norms": last_norms,
        "invocation_seconds": time.perf_counter() - run_started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
    }
    with (args.output_dir / "stage-history.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    atomic_json(record, args.output_dir / "latest-stage.json")
    print("STAGE_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
