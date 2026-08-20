#!/usr/bin/env python3
"""BF16 real-data smoke training for the clean final factorized autoencoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import torch
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.models.final_factorized_autoencoder import (  # noqa: E402
    FinalFactorizedAutoencoder,
)
from world_model_trajectory.training.final_factorized_objective import (  # noqa: E402
    loss_components,
    trajectory_confusion_counts,
    trajectory_diagnostics,
    trajectory_metrics_from_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stratified_order(metadata: list[dict[str, object]]) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row.get("category", "unknown"))].append(index)
    order: list[int] = []
    for offset in range(max(map(len, groups.values()))):
        for category in sorted(groups):
            if offset < len(groups[category]):
                order.append(groups[category][offset])
    return order


def weighted_loss(
    components: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    if set(components) != set(weights):
        raise RuntimeError(
            f"Loss contract mismatch: components={sorted(components)} "
            f"weights={sorted(weights)}"
        )
    return sum(weights[name] * components[name] for name in weights)


def objective_kwargs(config: dict[str, object]) -> dict[str, float | int]:
    objective = config["objective"]
    if not isinstance(objective, dict):
        raise ValueError("Config objective must be an object")
    return {
        "target_score_threshold": float(objective["target_score_threshold"]),
        "antialias_support_radius": int(objective["antialias_support_radius"]),
        "positive_weight_cap": float(objective["positive_weight_cap"]),
    }


def gradient_report(model: FinalFactorizedAutoencoder) -> dict[str, dict[str, float]]:
    modules = {
        "scene_encoder": model.scene_branch.encoder,
        "scene_decoder": model.scene_branch.decoder,
        "trajectory_encoder": model.trajectory_branch.encoder,
        "trajectory_decoder": model.trajectory_branch.decoder,
    }
    report: dict[str, dict[str, float]] = {}
    for name, module in modules.items():
        gradients = [
            parameter.grad.float()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"No gradients reached {name}")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Nonfinite gradient in {name}")
        absolute_sum = sum(float(gradient.abs().sum()) for gradient in gradients)
        squared_sum = sum(float(gradient.square().sum()) for gradient in gradients)
        maximum = max(float(gradient.abs().max()) for gradient in gradients)
        if absolute_sum <= 0 or not math.isfinite(absolute_sum + squared_sum + maximum):
            raise RuntimeError(f"Invalid zero/nonfinite gradient summary for {name}")
        report[name] = {
            "absolute_sum": absolute_sum,
            "l2": math.sqrt(squared_sum),
            "maximum": maximum,
        }
    return report


@torch.no_grad()
def evaluate(
    model: FinalFactorizedAutoencoder,
    images: torch.Tensor,
    weights: dict[str, float],
    objective: dict[str, float | int],
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    sums = {name: 0.0 for name in weights}
    confusion_sums: dict[str, float] = defaultdict(float)
    reconstructions = []
    scenes = []
    masks = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            components = loss_components(output, batch, **objective)
        for name in weights:
            sums[name] += float(components[name]) * len(batch)
        counts = trajectory_confusion_counts(
            output.trajectory_probability,
            batch,
            target_score_threshold=float(objective["target_score_threshold"]),
            antialias_support_radius=int(objective["antialias_support_radius"]),
        )
        for name, value in counts.items():
            confusion_sums[name] += value
        reconstructions.append(output.image.float().cpu())
        scenes.append(output.scene.float().cpu())
        masks.append(output.trajectory_probability.float().expand(-1, 3, -1, -1).cpu())
    component_means = {name: value / len(images) for name, value in sums.items()}
    diagnostics = trajectory_metrics_from_counts(confusion_sums)
    return {
        "components": component_means,
        "weighted_total": sum(weights[name] * component_means[name] for name in weights),
        "trajectory": diagnostics,
        "panel": torch.cat(
            (images[:8].float().cpu(), torch.cat(reconstructions)[:8], torch.cat(scenes)[:8], torch.cat(masks)[:8])
        ),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = {name: float(value) for name, value in config["loss_weights"].items()}
    objective = objective_kwargs(config)
    payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    order = stratified_order(payload["metadata"])
    images = payload["images_uint8"][order].to("cuda", dtype=torch.float32).div_(255)
    images = images.contiguous(memory_format=torch.channels_last)
    model = FinalFactorizedAutoencoder(
        scene_width=int(config["scene_width"]),
        trajectory_width=int(config["trajectory_width"]),
    ).to("cuda", memory_format=torch.channels_last)
    if model.contract_version != config["contract_version"]:
        raise RuntimeError("Model/config contract versions disagree")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, images, weights, objective, args.batch_size)
    save_image(initial.pop("panel"), args.output_dir / "initial.png", nrow=min(8, len(images)))
    history = []
    batches = cycle(range(0, len(images), args.batch_size))
    model.train()
    training_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        start = next(batches)
        batch = images[start : start + args.batch_size]
        if len(batch) < args.batch_size:
            batch = torch.cat((batch, images[: args.batch_size - len(batch)]))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            components = loss_components(output, batch, **objective)
            total = weighted_loss(components, weights)
        total.backward()
        gradients = gradient_report(model)
        diagnostics = trajectory_diagnostics(
            output.trajectory_probability,
            batch,
            target_score_threshold=float(objective["target_score_threshold"]),
            antialias_support_radius=int(objective["antialias_support_radius"]),
        )
        optimizer.step()
        row = {
            "step": step,
            "weighted_total": float(total.detach()),
            "components": {
                name: float(value.detach()) for name, value in components.items()
            },
            "gradients": gradients,
            "trajectory": diagnostics,
        }
        history.append(row)
        print(
            f"step={step:03d} loss={float(total.detach()):.6f} "
            f"traj_prob={diagnostics['predicted_probability_mean']:.5f} "
            f"traj_fg={diagnostics['predicted_foreground_fraction']:.5f} "
            f"traj_f1={diagnostics['f1']:.4f} "
            f"scene_grad={gradients['scene_encoder']['l2']:.4e} "
            f"trajectory_grad={gradients['trajectory_encoder']['l2']:.4e}",
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    final = evaluate(model, images, weights, objective, args.batch_size)
    save_image(final.pop("panel"), args.output_dir / "final.png", nrow=min(8, len(images)))
    if not final["weighted_total"] < initial["weighted_total"]:
        raise RuntimeError("Smoke objective did not improve")
    if final["trajectory"]["predicted_foreground_fraction"] >= 0.5:
        raise RuntimeError("Trajectory head collapsed toward full-screen foreground")
    if final["trajectory"]["predicted_foreground_fraction"] <= 0:
        raise RuntimeError("Trajectory head collapsed toward all-background")
    if final["trajectory"]["f1"] <= initial["trajectory"]["f1"]:
        raise RuntimeError("Trajectory F1 did not improve during smoke training")
    if final["components"]["trajectory"] >= initial["components"]["trajectory"]:
        raise RuntimeError("Direct trajectory objective did not improve")
    if final["components"]["final_crosshair_rgb"] > 1e-8:
        raise RuntimeError("Deterministic FP32 crosshair composition is not exact")
    source_files = (
        REPOSITORY_ROOT / "src/world_model_trajectory/models/final_factorized_autoencoder.py",
        REPOSITORY_ROOT / "src/world_model_trajectory/training/final_factorized_objective.py",
        Path(__file__).resolve(),
    )
    source_hashes = {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    report = {
        "smoke_version": 1,
        "architecture": model.architecture_signature,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "probe": str(args.probe),
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "gpu": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "training_seconds": training_seconds,
        "compute_only_training_images_per_second": (
            args.steps * args.batch_size / training_seconds
        ),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "initial": initial,
        "final": final,
        "history": history,
    }
    destination = args.output_dir / "smoke-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": model.architecture_signature,
            "config": config,
            "config_sha256": report["config_sha256"],
            "source_sha256": source_hashes,
            "seed": args.seed,
            "smoke_only": True,
        },
        args.output_dir / "smoke-checkpoint.pt",
    )
    print(f"SMOKE COMPLETE report={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
