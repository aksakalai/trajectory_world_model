#!/usr/bin/env python3
"""Measure raw renderer-loss values and shared-encoder gradient magnitudes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.models.comparison_autoencoders import (  # noqa: E402
    LayeredOutput,
    create_comparison_model,
)
from world_model_trajectory.training.losses import (  # noqa: E402
    edge_l1_loss,
    layered_renderer_losses,
    standard_renderer_losses,
)


MODELS = (
    "standard_residual_large_overlay_loss",
    "standard_residual_ceiling_overlay_loss",
    "layered_residual_large",
    "layered_residual_ceiling",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stratified_indices(metadata: list[dict[str, object]]) -> list[int]:
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        by_category[str(item["category"])].append(index)
    result = []
    for offset in range(max(map(len, by_category.values()))):
        for category in sorted(by_category):
            if offset < len(by_category[category]):
                result.append(by_category[category][offset])
    return result


def encoder_gradient_norm(
    component: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        component,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = component.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.float().square().sum()
    return float(squared.sqrt().item())


def components_for_model(
    model: torch.nn.Module, images: torch.Tensor
) -> dict[str, torch.Tensor]:
    output = model(images)
    if isinstance(output, LayeredOutput):
        return layered_renderer_losses(
            scene=output.scene,
            trajectory_mask=output.trajectory_mask,
            crosshair_logits=output.crosshair_logits,
            reconstruction=output.reconstruction,
            target=images,
            trajectory_logits=getattr(output, "trajectory_logits", None),
        )
    components = {
        "whole_l1": F.l1_loss(output, images),
        "edge": edge_l1_loss(output, images),
    }
    components.update(standard_renderer_losses(output, images))
    return components


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for representative gradient calibration")
    torch.manual_seed(args.seed)
    payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    images_uint8 = payload["images_uint8"]
    metadata = payload["metadata"]
    order = stratified_indices(metadata)
    required = args.batch_size * args.batches
    if len(order) < required:
        raise RuntimeError(f"Probe has {len(order)} images; calibration needs {required}")
    device = torch.device("cuda")
    report = {
        "calibration_version": 1,
        "probe": str(args.probe),
        "probe_dataset_root": payload.get("dataset_root"),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "models": {},
    }
    for model_name in MODELS:
        model = create_comparison_model(model_name).to(
            device, memory_format=torch.channels_last
        )
        model.train()
        encoder_parameters = tuple(model.encoder.parameters())
        values: dict[str, list[float]] = defaultdict(list)
        gradient_norms: dict[str, list[float]] = defaultdict(list)
        for batch_index in range(args.batches):
            indices = order[
                batch_index * args.batch_size : (batch_index + 1) * args.batch_size
            ]
            images = images_uint8[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            ).div_(255)
            images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components = components_for_model(model, images)
            names = list(components)
            for component_index, name in enumerate(names):
                component = components[name]
                values[name].append(float(component.detach().float().item()))
                gradient_norms[name].append(
                    encoder_gradient_norm(
                        component,
                        encoder_parameters,
                        retain_graph=component_index < len(names) - 1,
                    )
                )
            del components, images
        model_report = {
            "latent_shape": list(model.latent_shape),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "components": {
                name: {
                    "mean_value": sum(values[name]) / len(values[name]),
                    "mean_encoder_gradient_norm": sum(gradient_norms[name])
                    / len(gradient_norms[name]),
                    "values": values[name],
                    "encoder_gradient_norms": gradient_norms[name],
                }
                for name in values
            },
        }
        if not all(
            math.isfinite(component["mean_value"])
            and math.isfinite(component["mean_encoder_gradient_norm"])
            for component in model_report["components"].values()
        ):
            raise RuntimeError(f"Non-finite calibration measurement for {model_name}")
        report["models"][model_name] = model_report
        print(json.dumps({model_name: model_report}, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"CALIBRATION COMPLETE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
