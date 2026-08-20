#!/usr/bin/env python3
"""Fresh, exact-budget distributed training for the selected G autoencoder."""

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
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.distributed_tar_images import DistributedTarImageDataset
from world_model_trajectory.data.episode_splits import load_episode_split_manifest
from world_model_trajectory.data.tar_images import AUTHORITATIVE_COLLECTIONS, discover_shards
from world_model_trajectory.models.final_factorized_autoencoder import (
    FinalFactorizedAutoencoder,
    fixed_crosshair_stencil,
)
from world_model_trajectory.training.final_factorized_objective import (
    exact_core_confusion_counts,
    loss_components,
    trajectory_confusion_counts,
    trajectory_logit_loss,
    trajectory_metrics_from_counts,
    trajectory_target,
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
    parser.add_argument(
        "--uneven-smoke-base-images",
        type=int,
        help="Diagnostic only: rank r stops after base + r*batch images",
    )
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


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def unwrap(model: torch.nn.Module) -> FinalFactorizedAutoencoder:
    value = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(value, FinalFactorizedAutoencoder):
        raise TypeError("Unexpected autoencoder wrapper")
    return value


def model_from_config(config: dict[str, object]) -> FinalFactorizedAutoencoder:
    value = config["model"]
    assert isinstance(value, dict)
    return FinalFactorizedAutoencoder(
        scene_width=int(value["scene_width"]),
        trajectory_width=int(value["trajectory_width"]),
        scene_latent_channels=int(value["scene_channels"]),
        trajectory_latent_channels=int(value["trajectory_channels"]),
        scene_latent_resolution=int(value["scene_resolution"]),
        trajectory_latent_resolution=int(value["trajectory_resolution"]),
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
    modules = {
        "scene_encoder": model.scene_branch.encoder,
        "scene_decoder": model.scene_branch.decoder,
        "trajectory_encoder": model.trajectory_branch.encoder,
        "trajectory_decoder": model.trajectory_branch.decoder,
    }
    result = {}
    for name, module in modules.items():
        gradients = [parameter.grad.float() for parameter in module.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(item).all() for item in gradients):
            raise RuntimeError(f"Missing or nonfinite gradient in {name}")
        norm = math.sqrt(sum(float(item.square().sum()) for item in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Nonpositive gradient norm in {name}: {norm}")
        result[name] = norm
    return result


def all_reduce_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def gathered_int(value: int, device: torch.device, world_size: int) -> list[int]:
    local = torch.tensor([value], device=device, dtype=torch.int64)
    if world_size == 1:
        return [value]
    outputs = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(outputs, local)
    return [int(item.item()) for item in outputs]


def gather_objects(value: object, world_size: int) -> list[object]:
    if world_size == 1:
        return [value]
    values: list[object] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def leakage_audit(
    images_uint8: torch.Tensor, objective: dict[str, float | int], device: torch.device
) -> dict[str, float]:
    images = images_uint8[:16].to(device=device, dtype=torch.float32).div_(255)
    target = trajectory_target(
        images,
        score_threshold=float(objective["target_score_threshold"]),
        antialias_support_radius=int(objective["antialias_support_radius"]),
    )
    oracle = torch.where(target > 0.05, torch.full_like(target, 12), torch.full_like(target, -12))
    all_foreground = torch.full_like(target, 12)
    oracle_loss = float(trajectory_logit_loss(
        oracle, target, positive_weight_cap=float(objective["positive_weight_cap"])
    ))
    all_foreground_loss = float(trajectory_logit_loss(
        all_foreground, target, positive_weight_cap=float(objective["positive_weight_cap"])
    ))
    counts = trajectory_confusion_counts(
        all_foreground[:, None].sigmoid(), images,
        target_score_threshold=float(objective["target_score_threshold"]),
        antialias_support_radius=int(objective["antialias_support_radius"]),
    )
    metrics = trajectory_metrics_from_counts(counts)
    if not all_foreground_loss > oracle_loss + 1.0:
        raise RuntimeError("All-foreground leakage prediction is not strongly penalized")
    if metrics["predicted_foreground_fraction"] < 0.99 or metrics["precision"] > 0.05:
        raise RuntimeError("All-foreground leakage audit metrics are invalid")
    return {
        "oracle_loss": oracle_loss,
        "all_foreground_loss": all_foreground_loss,
        "all_foreground_precision": float(metrics["precision"]),
        "all_foreground_fraction": float(metrics["predicted_foreground_fraction"]),
    }


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
    panels: list[torch.Tensor] = []
    mismatches = 0
    max_crosshair_error = 0.0
    device = next(model.parameters()).device
    stencil = fixed_crosshair_stencil(device=device)[0, 0]
    for start in range(0, len(images_uint8), batch_size):
        target = images_uint8[start : start + batch_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(target)
            components = loss_components(output, target, **objective)
        predicted_crosshair = output.image[:, :, stencil]
        target_crosshair = target[:, :, stencil]
        max_crosshair_error = max(
            max_crosshair_error,
            float((predicted_crosshair - target_crosshair).abs().max()),
        )
        mismatches += int((
            predicted_crosshair.mul(255).round().to(torch.uint8)
            != target_crosshair.mul(255).round().to(torch.uint8)
        ).sum())
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
            panels = [
                target[:8].cpu(), output.image[:8].float().cpu(),
                output.trajectory_probability[:8].float().expand(-1, 3, -1, -1).cpu(),
            ]
    if mismatches:
        raise RuntimeError(f"Crosshair changed {mismatches} quantized channel values")
    means = {name: value / len(images_uint8) for name, value in sums.items()}
    save_image(torch.cat(panels), output_dir / f"validation-{label}.png", nrow=8)
    model.train()
    return {
        "images": len(images_uint8),
        "components": means,
        "weighted_total": sum(weights[name] * means[name] for name in weights),
        "trajectory": trajectory_metrics_from_counts(counts),
        "trajectory_exact_core": trajectory_metrics_from_counts(exact),
        "crosshair_audit": {
            "quantized_channel_mismatches": mismatches,
            "max_abs_float_error": max_crosshair_error,
        },
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, local_rank, device = distributed_context()
    is_main = rank == 0
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        training = config["training"]
        assert isinstance(training, dict)
        global_microbatches = int(training["global_accumulation_microbatches"])
        if global_microbatches % world_size:
            raise ValueError("world_size must divide global_accumulation_microbatches")
        local_accumulation = global_microbatches // world_size
        if local_accumulation < 1:
            raise ValueError("world_size exceeds the fixed global microbatch count")
        batch_size = int(training["batch_size_per_rank"])
        global_effective_batch_images = batch_size * global_microbatches
        if args.target_total_images < 1:
            raise ValueError("Target image count must be positive")

        dataset_root = args.dataset_root.resolve()
        shards = discover_shards(dataset_root)
        split = load_episode_split_manifest(
            args.split_manifest, dataset_root=dataset_root, discovered_shards=shards
        )
        if args.target_total_images > split.image_counts["train"]:
            raise ValueError("Target exceeds the immutable training split")
        distributed_wave_images = world_size * batch_size
        if (
            args.target_total_images < split.image_counts["train"]
            and args.uneven_smoke_base_images is None
            and args.target_total_images % distributed_wave_images
        ):
            raise ValueError(
                "A resumable staged target must align to one complete distributed wave"
            )
        probe_bytes = args.validation_probe.read_bytes()
        probe_payload = torch.load(args.validation_probe, map_location="cpu", weights_only=False)
        if probe_payload.get("split") != "validation":
            raise RuntimeError("Validation probe is not validation-only")
        if probe_payload.get("split_manifest_sha256") != split.manifest_sha256:
            raise RuntimeError("Validation probe and split manifest differ")
        probe = probe_payload["images_uint8"]

        source_files = (
            Path(__file__),
            REPOSITORY_ROOT / "src/world_model_trajectory/data/distributed_tar_images.py",
            REPOSITORY_ROOT / "src/world_model_trajectory/data/tar_images.py",
            REPOSITORY_ROOT / "src/world_model_trajectory/data/episode_splits.py",
            REPOSITORY_ROOT / "src/world_model_trajectory/models/final_factorized_autoencoder.py",
            REPOSITORY_ROOT / "src/world_model_trajectory/training/final_factorized_objective.py",
        )
        contract = {
            "contract_version": 1,
            "run_version": config["run_version"],
            "config": config,
            "config_sha256": digest(config_bytes),
            "dataset_root": str(dataset_root),
            "split_manifest": str(split.manifest_path),
            "split_manifest_sha256": split.manifest_sha256,
            "validation_probe_sha256": digest(probe_bytes),
            "full_train_images": split.image_counts["train"],
            "world_size": world_size,
            "local_accumulation_rounds": local_accumulation,
            "uneven_smoke_base_images": args.uneven_smoke_base_images,
            "source_sha256": {
                str(path.relative_to(REPOSITORY_ROOT)): digest(path.read_bytes())
                for path in source_files
            },
        }
        contract_hash = digest(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
        contract["contract_sha256"] = contract_hash
        contract_path = args.output_dir / "run-contract.json"
        checkpoint_path = args.output_dir / "latest.pt"
        if is_main:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if args.resume:
                previous = json.loads(contract_path.read_text())
                if previous.get("contract_sha256") != contract_hash:
                    raise RuntimeError("Resume contract differs from the original distributed run")
            else:
                unexpected = [path for path in args.output_dir.iterdir() if path.name != "train.log"]
                if unexpected:
                    raise RuntimeError(f"Fresh output directory is not empty: {unexpected[:4]}")
                atomic_json(contract, contract_path)
        if dist.is_initialized():
            dist.barrier()

        seed = int(config["seed"])
        random.seed(seed + rank)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed + rank)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        base_model = model_from_config(config).to(device, memory_format=torch.channels_last)
        model_config = config["model"]
        assert isinstance(model_config, dict)
        if base_model.contract_version != model_config["contract_version"]:
            raise RuntimeError("Model/config contract mismatch")
        model: torch.nn.Module = base_model
        if world_size > 1:
            model = DistributedDataParallel(
                base_model, device_ids=[local_rank], broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        optimizer_config = config["optimizer"]
        assert isinstance(optimizer_config, dict)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        images_seen = optimizer_steps = 0
        rank_stream_positions = [0] * world_size
        if args.resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved.get("contract_sha256") != contract_hash:
                raise RuntimeError("Checkpoint contract mismatch")
            unwrap(model).load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            images_seen = int(saved["images_seen"])
            optimizer_steps = int(saved["optimizer_steps"])
            rank_stream_positions = [int(value) for value in saved["rank_stream_positions"]]
            state = saved["rank_rng_states"][rank]
            torch.set_rng_state(state["torch"].cpu())
            torch.cuda.set_rng_state(state["cuda"].cpu(), device)
        if images_seen > args.target_total_images:
            raise RuntimeError("Checkpoint is beyond requested target")

        weights = {name: float(value) for name, value in config["loss_weights"].items()}
        objective = objective_kwargs(config)
        if is_main:
            audit = leakage_audit(probe, objective, device)
            atomic_json(audit, args.output_dir / "leakage-audit.json")
        if dist.is_initialized():
            dist.barrier()

        dataset = DistributedTarImageDataset(
            shards, rank=rank, world_size=world_size, seed=seed, shuffle_shards=True,
            shuffle_buffer=int(training["shuffle_buffer"]),
            episode_splits_by_shard=split.episode_splits_by_shard,
            include_splits=("train",),
        )
        workers = int(training["workers_per_rank"])
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
            prefetch_factor=int(training["prefetch_factor"]) if workers else None,
            persistent_workers=False,
        )
        loader_iterator = iter(loader)
        stream_position = 0
        resume_position = rank_stream_positions[rank]
        while stream_position < resume_position:
            candidate = next(loader_iterator)
            stream_position += len(candidate)
        if stream_position != resume_position:
            raise RuntimeError("Checkpoint rank stream position is not batch-aligned")
        if dist.is_initialized():
            dist.barrier()
        rank_limit = None
        if args.uneven_smoke_base_images is not None:
            rank_limit = args.uneven_smoke_base_images + rank * batch_size
        last_batch: torch.Tensor | None = None
        local_rank_images = 0
        micro_round = 0
        accumulated_images = 0
        last_norms: dict[str, float] = {}
        checkpoint_every = int(training["checkpoint_every_images"])
        checkpoint_thresholds = set(int(value) for value in training["checkpoint_milestones"])
        checkpoint_thresholds.update(range(checkpoint_every, split.image_counts["train"] + 1, checkpoint_every))
        pending_thresholds = sorted(value for value in checkpoint_thresholds if value > images_seen)
        run_started = time.perf_counter()
        initial_images = images_seen
        interval_images = 0.0
        interval_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        def save_checkpoint(label: str, evaluate: bool) -> dict[str, object] | None:
            local_state = {
                "stream_position": stream_position,
                "torch": torch.get_rng_state().cpu(),
                "cuda": torch.cuda.get_rng_state(device).cpu(),
            }
            states = gather_objects(local_state, world_size)
            if is_main:
                atomic_torch({
                    "contract_sha256": contract_hash,
                    "architecture": unwrap(model).architecture_signature,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "model": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "images_seen": images_seen,
                    "optimizer_steps": optimizer_steps,
                    "rank_stream_positions": [int(value["stream_position"]) for value in states],
                    "rank_rng_states": states,
                    "world_size": world_size,
                }, checkpoint_path)
            if dist.is_initialized():
                dist.barrier()
            metrics = None
            if is_main and evaluate:
                metrics = evaluate_probe(
                    unwrap(model), probe, batch_size=batch_size, weights=weights,
                    objective=objective, output_dir=args.output_dir, label=label,
                )
                atomic_json(metrics, args.output_dir / f"validation-{label}.json")
            if dist.is_initialized():
                dist.barrier()
            return metrics

        model.train()
        completed_metrics: dict[str, object] | None = None
        while images_seen < args.target_total_images:
            local_batch: torch.Tensor | None = None
            if rank_limit is None or local_rank_images < rank_limit:
                try:
                    candidate = next(loader_iterator)
                    raw_size = len(candidate)
                    batch_end = stream_position + raw_size
                    stream_position = batch_end
                    if rank_limit is not None:
                        candidate = candidate[: max(rank_limit - local_rank_images, 0)]
                    if len(candidate):
                        local_batch = candidate
                        last_batch = candidate
                except StopIteration:
                    pass
            local_size = 0 if local_batch is None else len(local_batch)
            sizes = gathered_int(local_size, device, world_size)
            if not any(sizes):
                break
            remaining = args.target_total_images - images_seen
            prefix = sum(sizes[:rank])
            allowed = min(local_size, max(remaining - prefix, 0))
            if allowed and local_batch is not None:
                local_batch = local_batch[:allowed]
                last_batch = local_batch
            else:
                local_batch = None
            allowed_sizes = gathered_int(allowed, device, world_size)
            global_batch_size = sum(allowed_sizes)
            if global_batch_size == 0:
                break
            if last_batch is None:
                raise RuntimeError("A rank has no replay batch while another rank has data")
            images = (local_batch if local_batch is not None else last_batch).to(
                device, non_blocking=True
            ).contiguous(memory_format=torch.channels_last)
            # Preserve the original full-batch numerical scale during BF16
            # backward. DDP's rank mean is cancelled by world_size; at optimizer
            # time a correction handles any step with fewer than the nominal
            # number of images, including the full split's final two images.
            gradient_scale = (
                world_size * allowed / global_effective_batch_images if allowed else 0.0
            )
            next_micro_round = micro_round + 1
            reaches_target = images_seen + global_batch_size >= args.target_total_images
            should_step = next_micro_round % local_accumulation == 0 or reaches_target
            sync_context = nullcontext() if should_step else model.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(images)
                    components = loss_components(output, images, **objective)
                    total = sum(weights[name] * components[name] for name in weights)
                    (total * gradient_scale).backward()
            micro_round += 1
            images_seen += global_batch_size
            accumulated_images += global_batch_size
            local_rank_images += allowed
            interval_images += allowed
            interval_loss += float(total.detach()) * allowed
            if should_step:
                if accumulated_images < 1:
                    raise RuntimeError("Optimizer step has no accumulated images")
                correction = global_effective_batch_images / accumulated_images
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                log_every = int(training["log_every_optimizer_steps"])
                audit_gradients = (
                    optimizer_steps == 0
                    or (optimizer_steps + 1) % log_every == 0
                    or images_seen >= args.target_total_images
                )
                if audit_gradients:
                    last_norms = gradient_norms(unwrap(model))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_images = 0
                optimizer_steps += 1
                if optimizer_steps % log_every == 0 or images_seen >= args.target_total_images:
                    count, loss_sum = all_reduce_sums([interval_images, interval_loss], device)
                    if is_main:
                        elapsed = max(time.perf_counter() - run_started, 1e-9)
                        rate = (images_seen - initial_images) / elapsed
                        print(
                            f"PROGRESS images={images_seen} target={args.target_total_images} "
                            f"steps={optimizer_steps} rate={rate:.1f}img/s "
                            f"loss={loss_sum / max(count, 1):.6f} "
                            f"gradients={json.dumps(last_norms, sort_keys=True)}",
                            flush=True,
                        )
                    interval_images = interval_loss = 0.0
                crossed = [value for value in pending_thresholds if value <= images_seen]
                if crossed and images_seen < args.target_total_images:
                    save_checkpoint(f"{images_seen:09d}", evaluate=True)
                    pending_thresholds = [value for value in pending_thresholds if value > images_seen]
            if images_seen >= args.target_total_images:
                break

        partial = micro_round % local_accumulation
        if images_seen < args.target_total_images and partial:
            if accumulated_images < 1:
                raise RuntimeError("Partial optimizer step has no accumulated images")
            correction = global_effective_batch_images / accumulated_images
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            last_norms = gradient_norms(unwrap(model))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if images_seen != args.target_total_images:
            raise RuntimeError(
                f"Exact distributed budget failed: {images_seen} != {args.target_total_images}"
            )
        completed_metrics = save_checkpoint(f"{images_seen:09d}", evaluate=True)
        record = {
            "status": "complete",
            "target_total_images": args.target_total_images,
            "images_seen": images_seen,
            "optimizer_steps": optimizer_steps,
            "world_size": world_size,
            "global_effective_batch_images": batch_size * global_microbatches,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "architecture": unwrap(model).architecture_signature,
            "validation_probe": completed_metrics,
            "last_gradient_norms": last_norms,
            "invocation_seconds": time.perf_counter() - run_started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
        }
        if is_main:
            with (args.output_dir / "stage-history.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            atomic_json(record, args.output_dir / "latest-stage.json")
            print("STAGE_COMPLETE " + json.dumps(record, sort_keys=True), flush=True)
        if dist.is_initialized():
            dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
