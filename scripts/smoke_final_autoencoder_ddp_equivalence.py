#!/usr/bin/env python3
"""Prove four-rank gradients match a serial average of the same microbatches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from train_final_autoencoder_ddp import gradient_norms, model_from_config, objective_kwargs
from world_model_trajectory.models.final_factorized_autoencoder import (
    COOLDOWN_RGB,
    READY_RGB,
    TRAJECTORY_RGB,
    fixed_crosshair_stencil,
)
from world_model_trajectory.training.final_factorized_objective import (
    loss_components,
    trajectory_logit_loss,
    trajectory_target,
)


def canonical_batch(rank: int, batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9000 + rank)
    images = torch.randint(0, 256, (batch_size, 3, 384, 384), generator=generator).float().div_(255)
    stencil = fixed_crosshair_stencil()[0, 0]
    for index in range(batch_size):
        color = READY_RGB if (rank + index) % 2 == 0 else COOLDOWN_RGB
        images[index, :, stencil] = images.new_tensor(color).div(255)[:, None]
        y = 40 + 3 * index
        images[index, :, y, 50:90] = images.new_tensor(TRAJECTORY_RGB).div(255)[:, None]
    return images


def weighted_loss(model: torch.nn.Module, images: torch.Tensor, config: dict[str, object]) -> torch.Tensor:
    weights = {name: float(value) for name, value in config["loss_weights"].items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(images)
        components = loss_components(output, images, **objective_kwargs(config))
        return sum(weights[name] * components[name] for name in weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        config = json.loads(args.config.read_text())
        torch.manual_seed(int(config["seed"]))
        model = model_from_config(config).to(device, memory_format=torch.channels_last)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        images = canonical_batch(rank, args.batch_size).to(device, memory_format=torch.channels_last)
        loss = weighted_loss(ddp, images, config)
        loss.backward()
        norms = gradient_norms(model)

        distributed_gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        comparison = None
        if rank == 0:
            torch.manual_seed(int(config["seed"]))
            reference = model_from_config(config).to(device, memory_format=torch.channels_last)
            reference.zero_grad(set_to_none=True)
            for source_rank in range(world_size):
                source = canonical_batch(source_rank, args.batch_size).to(
                    device, memory_format=torch.channels_last
                )
                (weighted_loss(reference, source, config) / world_size).backward()
            squared_error = squared_reference = 0.0
            max_abs = 0.0
            for name, parameter in reference.named_parameters():
                expected = parameter.grad.detach().float().cpu()
                actual = distributed_gradients[name]
                difference = actual - expected
                squared_error += float(difference.square().sum())
                squared_reference += float(expected.square().sum())
                max_abs = max(max_abs, float(difference.abs().max()))
            relative_l2 = (squared_error / max(squared_reference, 1e-30)) ** 0.5

            objective = objective_kwargs(config)
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
                all_foreground, target,
                positive_weight_cap=float(objective["positive_weight_cap"]),
            ))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
            stencil = fixed_crosshair_stencil(device=device)[0, 0]
            mismatches = int((
                output.image[:, :, stencil].mul(255).round().to(torch.uint8)
                != images[:, :, stencil].mul(255).round().to(torch.uint8)
            ).sum())
            comparison = {
                "world_size": world_size,
                "relative_gradient_l2": relative_l2,
                "maximum_gradient_abs_error": max_abs,
                "gradient_norms": norms,
                "oracle_trajectory_loss": oracle_loss,
                "all_foreground_trajectory_loss": all_foreground_loss,
                "crosshair_quantized_mismatches": mismatches,
            }
            if relative_l2 > 0.02 or max_abs > 0.02:
                raise RuntimeError(f"Distributed gradient mismatch: {comparison}")
            if all_foreground_loss <= oracle_loss + 1.0:
                raise RuntimeError("All-foreground trajectory leakage is not penalized")
            if mismatches:
                raise RuntimeError("Deterministic crosshair changed under DDP")
            print("DDP_EQUIVALENCE_COMPLETE " + json.dumps(comparison, sort_keys=True), flush=True)
        dist.barrier()

        # Prove exact per-image weighting across one full distributed wave and
        # an unequal final wave containing images only on rank zero. This is the
        # full-run edge case when the immutable training count is not divisible
        # by the distributed wave size.
        torch.manual_seed(int(config["seed"]))
        accumulated_model = model_from_config(config).to(
            device, memory_format=torch.channels_last
        )
        accumulated_ddp = DistributedDataParallel(
            accumulated_model, device_ids=[local_rank], broadcast_buffers=False
        )
        nominal_effective_images = args.batch_size * int(
            config["training"]["global_accumulation_microbatches"]
        )
        first = canonical_batch(rank, args.batch_size).to(
            device, memory_format=torch.channels_last
        )
        with accumulated_ddp.no_sync():
            (
                weighted_loss(accumulated_ddp, first, config)
                * world_size
                * len(first)
                / nominal_effective_images
            ).backward()
        final_allowed = 2 if rank == 0 else 0
        final = (
            canonical_batch(100, final_allowed).to(device, memory_format=torch.channels_last)
            if final_allowed
            else first
        )
        (
            weighted_loss(accumulated_ddp, final, config)
            * world_size
            * final_allowed
            / nominal_effective_images
        ).backward()
        accumulated_images = world_size * args.batch_size + 2
        for parameter in accumulated_model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(nominal_effective_images / accumulated_images)

        uneven_comparison = None
        if rank == 0:
            torch.manual_seed(int(config["seed"]))
            reference = model_from_config(config).to(device, memory_format=torch.channels_last)
            reference.zero_grad(set_to_none=True)
            for source_rank in range(world_size):
                source = canonical_batch(source_rank, args.batch_size).to(
                    device, memory_format=torch.channels_last
                )
                (weighted_loss(reference, source, config) * len(source) / accumulated_images).backward()
            final_reference = canonical_batch(100, 2).to(
                device, memory_format=torch.channels_last
            )
            (weighted_loss(reference, final_reference, config) * 2 / accumulated_images).backward()
            squared_error = squared_reference = 0.0
            max_abs = 0.0
            for (_, actual), (_, expected) in zip(
                accumulated_model.named_parameters(), reference.named_parameters(), strict=True
            ):
                difference = actual.grad.detach().float() - expected.grad.detach().float()
                squared_error += float(difference.square().sum())
                squared_reference += float(expected.grad.detach().float().square().sum())
                max_abs = max(max_abs, float(difference.abs().max()))
            uneven_comparison = {
                "accumulated_images": accumulated_images,
                "final_wave_images": 2,
                "maximum_gradient_abs_error": max_abs,
                "relative_gradient_l2": (
                    squared_error / max(squared_reference, 1e-30)
                ) ** 0.5,
            }
            if (
                uneven_comparison["relative_gradient_l2"] > 0.02
                or uneven_comparison["maximum_gradient_abs_error"] > 0.02
            ):
                raise RuntimeError(
                    f"Uneven accumulated gradient mismatch: {uneven_comparison}"
                )
            print(
                "DDP_UNEVEN_ACCUMULATION_COMPLETE "
                + json.dumps(uneven_comparison, sort_keys=True),
                flush=True,
            )
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
