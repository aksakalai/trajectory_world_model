#!/usr/bin/env python3
"""Run a matched short renderer-aware training calibration on one fixed probe."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.evaluation.trajectory_metrics import (  # noqa: E402
    crosshair_metrics,
    trajectory_metrics,
)
from world_model_trajectory.models.comparison_autoencoders import (  # noqa: E402
    LayeredOutput,
    create_comparison_model,
)
from world_model_trajectory.training.losses import renderer_targets  # noqa: E402

from calibrate_renderer_losses import MODELS, stratified_indices  # noqa: E402
from train_comparison_autoencoder import training_components  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--loss-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model names; defaults to the calibrated comparison set",
    )
    return parser.parse_args()


def scalar_metrics(reconstruction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = trajectory_metrics(reconstruction, target) | crosshair_metrics(
        reconstruction, target
    )
    return {name: float(value.item()) for name, value in metrics.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    images: torch.Tensor,
    weights: dict[str, float],
    *,
    batch_size: int,
) -> tuple[dict[str, float], LayeredOutput | torch.Tensor]:
    model.eval()
    reconstructions = []
    head_trajectory_sum = 0.0
    target_trajectory_sum = 0.0
    trajectory_pixel_count = 0
    crosshair_correct = 0
    component_sums = {name: 0.0 for name in weights}
    final_output: LayeredOutput | torch.Tensor | None = None
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            reconstruction, components, _ = training_components(output, batch)
        reconstructions.append(reconstruction.float())
        if isinstance(output, LayeredOutput):
            targets = renderer_targets(batch)
            head_trajectory_sum += float(output.trajectory_mask.float().sum().item())
            target_trajectory_sum += float(targets["trajectory"].float().sum().item())
            trajectory_pixel_count += targets["trajectory"].numel()
            crosshair_correct += int(
                (
                    (
                        (output.crosshair_logits.squeeze(1) >= 0).long()
                        if output.crosshair_logits.shape[1] == 1
                        else output.crosshair_logits.argmax(dim=1)
                    )
                    == targets["crosshair_class"]
                ).sum().item()
            )
        for name in weights:
            component_sums[name] += float(components[name].float().item()) * len(batch)
        final_output = output
    reconstruction = torch.cat(reconstructions)
    report = scalar_metrics(reconstruction, images)
    for name, total in component_sums.items():
        report[f"loss_{name}"] = total / len(images)
    report["weighted_total"] = sum(
        weights[name] * report[f"loss_{name}"] for name in weights
    )
    if trajectory_pixel_count:
        report["head_trajectory_mean_probability"] = (
            head_trajectory_sum / trajectory_pixel_count
        )
        report["target_trajectory_fraction"] = (
            target_trajectory_sum / trajectory_pixel_count
        )
        report["head_crosshair_accuracy"] = crosshair_correct / len(images)
    assert final_output is not None
    return report, final_output


def save_panel(
    model: torch.nn.Module,
    images: torch.Tensor,
    destination: Path,
) -> None:
    model.eval()
    sample = images[:8]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(sample)
    rows = [sample.float()]
    if isinstance(output, LayeredOutput):
        if output.crosshair_logits.shape[1] == 1:
            cooldown = output.crosshair_logits.float().sigmoid()
            probabilities = torch.cat((1 - cooldown, cooldown), dim=1)
        else:
            probabilities = output.crosshair_logits.softmax(dim=1).float()
        ready = sample.new_tensor((48, 242, 72)).div(255).view(1, 3, 1, 1)
        cooldown = sample.new_tensor((242, 48, 48)).div(255).view(1, 3, 1, 1)
        state = probabilities[:, :1, None, None] * ready
        state = state + probabilities[:, 1:, None, None] * cooldown
        state = state.expand(-1, -1, 384, 384)
        rows.extend(
            (
                output.reconstruction.float(),
                output.scene.float(),
                output.trajectory_mask.float().expand(-1, 3, -1, -1),
                state,
            )
        )
    else:
        rows.append(output.float())
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows).cpu(), destination, nrow=len(sample))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    torch.manual_seed(args.seed)
    payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    order = stratified_indices(payload["metadata"])
    images = payload["images_uint8"][order].to(device="cuda", dtype=torch.float32).div_(255)
    images = images.contiguous(memory_format=torch.channels_last)
    loss_config = json.loads(args.loss_config.read_text(encoding="utf-8"))
    models = tuple(args.models) if args.models else MODELS
    report = {
        "smoke_version": 1,
        "probe": str(args.probe),
        "loss_config": str(args.loss_config),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "models": {},
        "requested_models": list(models),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        torch.manual_seed(args.seed)
        model = create_comparison_model(model_name).to(
            "cuda", memory_format=torch.channels_last
        )
        weights = {
            name: float(value)
            for name, value in loss_config["models"][model_name].items()
        }
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        initial, _ = evaluate(model, images, weights, batch_size=args.batch_size)
        save_panel(model, images, args.output_dir / f"{model_name}-initial.png")
        model.train()
        indices = cycle(range(0, len(images), args.batch_size))
        gradient_sums: dict[str, float] = {}
        for step in range(args.steps):
            start = next(indices)
            batch = images[start : start + args.batch_size]
            if len(batch) < args.batch_size:
                batch = torch.cat((batch, images[: args.batch_size - len(batch)]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
                _, components, _ = training_components(output, batch)
                loss = sum(weights[name] * components[name] for name in weights)
            loss.backward()
            if step == 0:
                for name, module in model.named_children():
                    gradient_sums[name] = sum(
                        float(parameter.grad.float().abs().sum().item())
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    )
                required = (
                    "scene_encoder",
                    "trajectory_encoder",
                    "scene_decoder",
                    "trajectory_decoder",
                    "crosshair_classifier",
                )
                missing = [
                    name for name in required
                    if hasattr(model, name) and gradient_sums.get(name, 0.0) <= 0.0
                ]
                if missing:
                    raise RuntimeError(
                        "No gradient reached factorized modules: " + ", ".join(missing)
                    )
            optimizer.step()
        final, _ = evaluate(model, images, weights, batch_size=args.batch_size)
        save_panel(model, images, args.output_dir / f"{model_name}-final.png")
        if not final["weighted_total"] < initial["weighted_total"]:
            raise RuntimeError(f"Calibration loss did not improve for {model_name}")
        report["models"][model_name] = {
            "initial": initial,
            "final": final,
            "weighted_total_improvement": initial["weighted_total"]
            - final["weighted_total"],
            "first_step_gradient_abs_sums": gradient_sums,
        }
        print(
            f"SMOKE model={model_name} weighted_total="
            f"{initial['weighted_total']:.6f}->{final['weighted_total']:.6f} "
            f"trajectory_f1={initial['trajectory_f1']:.4f}->{final['trajectory_f1']:.4f} "
            f"crosshair_accuracy={initial['crosshair_state_accuracy']:.4f}->"
            f"{final['crosshair_state_accuracy']:.4f}",
            flush=True,
        )
        del model, optimizer
        torch.cuda.empty_cache()
    output = args.output_dir / "smoke-report.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"SMOKE COMPLETE output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
