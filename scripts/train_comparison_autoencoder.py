from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.tar_images import TarImageDataset, discover_shards
from world_model_trajectory.evaluation.trajectory_metrics import crosshair_metrics, trajectory_metrics
from world_model_trajectory.models.comparison_autoencoders import (
    LayeredOutput,
    VQOutput,
    create_comparison_model,
)
from world_model_trajectory.training.losses import (
    edge_l1_loss,
    layered_renderer_losses,
    standard_renderer_losses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one autoencoder comparison candidate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-probe", type=Path, required=True)
    parser.add_argument("--loss-config", type=Path)
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Split layered-model batches across every visible CUDA device",
    )
    parser.add_argument("--model-index", type=int, default=1)
    parser.add_argument("--model-count", type=int, default=1)
    parser.add_argument("--total-images", type=int, default=3_381_548)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation-steps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--crosshair-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--probe-every", type=int, default=2_500)
    parser.add_argument("--early-stop-min-images", type=int, default=1_500_000)
    parser.add_argument("--early-stop-patience-images", type=int, default=640_000)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-ema-alpha", type=float, default=0.20)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def unpack_model_output(
    output: torch.Tensor | VQOutput | LayeredOutput,
) -> tuple[torch.Tensor, torch.Tensor, dict, LayeredOutput | None]:
    if isinstance(output, VQOutput):
        return output.reconstruction, output.auxiliary_loss, output.metrics, None
    if isinstance(output, LayeredOutput):
        return output.reconstruction, output.reconstruction.new_zeros(()), {}, output
    return output, output.new_zeros(()), {}, None


class LayeredTensorAdapter(nn.Module):
    """Expose layered outputs as tensors that nn.DataParallel can gather."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(images)
        if not isinstance(output, LayeredOutput):
            raise TypeError("--data-parallel currently requires a layered model")
        return (
            output.scene,
            output.trajectory_mask,
            output.crosshair_logits,
            output.reconstruction,
        )


class LayeredLossAdapter(nn.Module):
    """Compute renderer losses on each GPU and gather only small reductions."""

    def __init__(self, model: nn.Module, loss_weights: dict[str, float]) -> None:
        super().__init__()
        self.model = model
        self.loss_names = tuple(loss_weights)
        self.loss_weights = tuple(loss_weights.values())

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(images)
        _, components, _ = training_components(output, images)
        missing = set(self.loss_names) - set(components)
        extra = set(components) - set(self.loss_names)
        if missing or extra:
            raise RuntimeError(
                f"Loss component mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        values = torch.stack([components[name] for name in self.loss_names])
        weights = values.new_tensor(self.loss_weights)
        return (values * weights).sum().unsqueeze(0), values.unsqueeze(0)


def forward_model(model: nn.Module, images: torch.Tensor) -> torch.Tensor | LayeredOutput:
    output = model(images)
    if isinstance(output, tuple) and len(output) == 4:
        scene, trajectory_mask, crosshair_logits, reconstruction = output
        return LayeredOutput(
            scene=scene,
            trajectory_mask=trajectory_mask,
            crosshair_logits=crosshair_logits,
            reconstruction=reconstruction,
            latent=reconstruction,
        )
    return output


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    probe_uint8: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    step: int,
    output_dir: Path,
) -> dict[str, float]:
    model.eval()
    targets: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    scenes: list[torch.Tensor] = []
    trajectory_masks: list[torch.Tensor] = []
    crosshair_probabilities: list[torch.Tensor] = []
    for start in range(0, len(probe_uint8), batch_size):
        target = probe_uint8[start : start + batch_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255.0)
        target = target.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_model(model, target)
            reconstruction, _, _, layered = unpack_model_output(output)
        targets.append(target.float().cpu())
        reconstructions.append(reconstruction.float().cpu())
        if layered is not None:
            scenes.append(layered.scene.float().cpu())
            trajectory_masks.append(layered.trajectory_mask.float().cpu())
            if layered.crosshair_logits.shape[1] == 1:
                cooldown = layered.crosshair_logits.float().sigmoid()
                probabilities = torch.cat((1 - cooldown, cooldown), dim=1)
            else:
                probabilities = layered.crosshair_logits.softmax(dim=1).float()
            crosshair_probabilities.append(probabilities.cpu())
    target_cpu = torch.cat(targets)
    reconstruction_cpu = torch.cat(reconstructions)
    metrics = {
        key: float(value.item())
        for key, value in {
            **trajectory_metrics(reconstruction_cpu, target_cpu),
            **crosshair_metrics(reconstruction_cpu, target_cpu),
        }.items()
    }
    panel_count = min(8, len(target_cpu))
    panel_rows = [target_cpu[:panel_count], reconstruction_cpu[:panel_count]]
    if scenes:
        scene_cpu = torch.cat(scenes)
        mask_cpu = torch.cat(trajectory_masks).expand(-1, 3, -1, -1)
        probabilities = torch.cat(crosshair_probabilities)
        state_swatch = torch.zeros_like(scene_cpu)
        ready = state_swatch.new_tensor((48, 242, 72)).div(255).view(1, 3, 1, 1)
        cooldown = state_swatch.new_tensor((242, 48, 48)).div(255).view(1, 3, 1, 1)
        ready_probability = probabilities[:, 0].view(-1, 1, 1, 1)
        cooldown_probability = probabilities[:, 1].view(-1, 1, 1, 1)
        state_swatch[:] = ready_probability * ready + cooldown_probability * cooldown
        panel_rows.extend(
            (
                scene_cpu[:panel_count],
                mask_cpu[:panel_count],
                state_swatch[:panel_count],
            )
        )
    save_image(
        torch.cat(panel_rows),
        output_dir / "panels" / f"probe-step-{step:08d}.png",
        nrow=panel_count,
    )
    with (output_dir / "probe_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
    print(
        "PROBE "
        f"step={step:,} trajectory_f1={metrics['trajectory_f1']:.4f} "
        f"trajectory_iou={metrics['trajectory_iou']:.4f} "
        f"trajectory_recall={metrics['trajectory_recall']:.4f} "
        f"trajectory_color_l1={metrics['trajectory_color_l1']:.5f} "
        f"crosshair_color_l1={metrics['crosshair_color_l1']:.5f} "
        f"crosshair_accuracy={metrics['crosshair_state_accuracy']:.4f} "
        f"center_l1={metrics['center_l1']:.5f}",
        flush=True,
    )
    model.train()
    return metrics


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    images_seen: int,
    config: dict,
    early_stop_state: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "images_seen": images_seen,
        "config": config,
        "early_stop_state": early_stop_state,
    }


def load_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    if args.loss_config is not None:
        payload = json.loads(args.loss_config.read_text(encoding="utf-8"))
        model_weights = payload.get("models", {}).get(args.model)
        if not isinstance(model_weights, dict) or not model_weights:
            raise ValueError(f"Loss config has no weights for {args.model}")
        weights = {str(name): float(value) for name, value in model_weights.items()}
        if any(value < 0 or not math.isfinite(value) for value in weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        return weights
    if args.model.startswith(
        ("standard_residual_", "layered_residual_", "factorized_residual_")
    ):
        raise ValueError("Renderer-aware models require --loss-config")
    weights = {
        "whole_l1": 1.0,
        "edge": args.edge_weight,
        "trajectory_rgb": args.trajectory_loss_weight,
        "crosshair_rgb": args.crosshair_loss_weight,
        "trajectory_overlap": 0.0,
    }
    if args.model == "residual_vq_large":
        weights["auxiliary"] = 1.0
    return weights


def training_components(
    output: torch.Tensor | VQOutput | LayeredOutput,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    reconstruction, auxiliary, model_metrics, layered = unpack_model_output(output)
    if layered is not None:
        components = layered_renderer_losses(
            scene=layered.scene,
            trajectory_mask=layered.trajectory_mask,
            crosshair_logits=layered.crosshair_logits,
            reconstruction=layered.reconstruction,
            target=target,
            trajectory_logits=getattr(layered, "trajectory_logits", None),
        )
    else:
        components = {
            "whole_l1": torch.nn.functional.l1_loss(reconstruction, target),
            "edge": edge_l1_loss(reconstruction, target),
            **standard_renderer_losses(reconstruction, target),
        }
    if isinstance(output, VQOutput):
        components["auxiliary"] = auxiliary
    return reconstruction, components, model_metrics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps must be positive")
    if not 1 <= args.model_index <= args.model_count:
        raise ValueError("model-index must be within model-count")
    loss_weights = load_loss_weights(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    panels = args.output_dir / "panels"
    checkpoints.mkdir(exist_ok=True)
    panels.mkdir(exist_ok=True)
    shards = discover_shards(args.dataset_root)
    if len(shards) != 363:
        raise RuntimeError(f"Expected 363 authoritative shards, found {len(shards)}")

    probe_payload = torch.load(args.trajectory_probe, map_location="cpu", weights_only=False)
    probe_uint8 = probe_payload["images_uint8"]
    if probe_uint8.dtype != torch.uint8 or probe_uint8.ndim != 4:
        raise ValueError("Trajectory probe must contain NCHW uint8 images")

    device = torch.device("cuda")
    base_model = create_comparison_model(args.model).to(
        device, memory_format=torch.channels_last
    )
    model: nn.Module = base_model
    evaluation_model: nn.Module = base_model
    if args.data_parallel:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = nn.DataParallel(LayeredLossAdapter(base_model, loss_weights))
        evaluation_model = nn.DataParallel(LayeredTensorAdapter(base_model))
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    resume_checkpoint = checkpoints / "latest.pt"
    resume_payload = None
    if args.resume_if_available and resume_checkpoint.exists():
        resume_payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        previous_config = resume_payload["config"]
        for key, expected in {
            "model": args.model,
            "dataset_root": str(args.dataset_root),
            "total_images": args.total_images,
        }.items():
            if previous_config.get(key) != expected:
                raise RuntimeError(
                    f"Resume checkpoint mismatch for {key}: "
                    f"{previous_config.get(key)!r} != {expected!r}"
                )
        if previous_config.get("loss_weights") != loss_weights:
            raise RuntimeError("Resume checkpoint loss weights do not match the current config")
        base_model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
    effective_batch = args.batch_size * args.accumulation_steps
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        effective_batch_size=effective_batch,
        parameter_count=parameter_count,
        latent_shape=list(getattr(base_model, "latent_shape", ())),
        architecture_signature=getattr(base_model, "architecture_signature", None),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0),
        shard_count=len(shards),
        precision="bfloat16",
        cuda_device_count=torch.cuda.device_count(),
        loss_weights=loss_weights,
        loss_config=str(args.loss_config) if args.loss_config else None,
        resumed_from=str(resume_checkpoint) if resume_payload is not None else None,
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)

    dataset = TarImageDataset(
        shards,
        seed=args.seed,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor,
    )

    early_stop_state: dict[str, float | int | None] = (
        resume_payload["early_stop_state"]
        if resume_payload is not None
        else {
            "ema": None,
            "best_ema": None,
            "last_improvement_images": 0,
        }
    )
    global_step = int(resume_payload["global_step"]) if resume_payload is not None else 0
    images_seen = int(resume_payload["images_seen"]) if resume_payload is not None else 0
    run_started = time.perf_counter()
    previous_end = run_started
    interval_started = run_started
    interval_images = 0
    interval_batches = 0
    interval_data_wait = 0.0
    interval_loss = torch.zeros((), device=device)
    interval_components = {
        name: torch.zeros((), device=device) for name in loss_weights
    }
    stopped_early = False
    stop_reason = None
    optimizer.zero_grad(set_to_none=True)

    if resume_payload is not None:
        print(
            f"RESUME checkpoint={resume_checkpoint} step={global_step:,} "
            f"images={images_seen:,}/{args.total_images:,}",
            flush=True,
        )

    evaluate_probe(
        evaluation_model,
        probe_uint8,
        device=device,
        batch_size=min(args.batch_size, 16),
        step=0,
        output_dir=args.output_dir,
    )

    for micro_step, images in enumerate(loader, start=1):
        data_ready = time.perf_counter()
        interval_data_wait += data_ready - previous_end
        remaining_images = args.total_images - images_seen
        if remaining_images <= 0:
            break
        if images.shape[0] > remaining_images:
            images = images[:remaining_images]
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.data_parallel:
                parallel_loss, parallel_components = model(images)
                total_loss = parallel_loss.mean()
                component_means = parallel_components.mean(dim=0)
                components = {
                    name: component_means[index]
                    for index, name in enumerate(loss_weights)
                }
                model_metrics = {}
            else:
                output = forward_model(model, images)
                _, components, model_metrics = training_components(output, images)
                missing_components = set(loss_weights) - set(components)
                if missing_components:
                    raise RuntimeError(
                        "Loss config references unavailable components: "
                        f"{sorted(missing_components)}"
                    )
                unweighted_components = set(components) - set(loss_weights)
                if unweighted_components:
                    raise RuntimeError(
                        f"Loss config omits components: {sorted(unweighted_components)}"
                    )
                total_loss = sum(
                    loss_weights[name] * components[name] for name in loss_weights
                )
            scaled_loss = total_loss / args.accumulation_steps
        scaled_loss.backward()

        batch_images = images.shape[0]
        images_seen += batch_images
        interval_images += batch_images
        interval_batches += 1
        interval_loss += total_loss.detach() * batch_images
        for name in interval_components:
            interval_components[name] += components[name].detach() * batch_images

        should_step = micro_step % args.accumulation_steps == 0 or images_seen >= args.total_images
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if should_step and (global_step == 1 or global_step % args.log_every == 0):
            torch.cuda.synchronize()
            now = time.perf_counter()
            mean_loss = float((interval_loss / interval_images).item())
            mean_components = {
                name: float((value / interval_images).item())
                for name, value in interval_components.items()
            }
            if "whole_l1" in mean_components:
                monitored_reconstruction = mean_components["whole_l1"] + loss_weights.get(
                    "edge", 0.0
                ) * mean_components.get("edge", 0.0)
            else:
                monitored_reconstruction = mean_components["final_l1"] + loss_weights.get(
                    "final_edge", 0.0
                ) * mean_components.get("final_edge", 0.0)
            alpha = args.early_stop_ema_alpha
            previous_ema = early_stop_state["ema"]
            ema = (
                monitored_reconstruction
                if previous_ema is None
                else alpha * monitored_reconstruction + (1 - alpha) * float(previous_ema)
            )
            early_stop_state["ema"] = ema
            best_ema = early_stop_state["best_ema"]
            if best_ema is None or float(best_ema) - ema >= args.early_stop_min_delta:
                early_stop_state["best_ema"] = ema
                early_stop_state["last_improvement_images"] = images_seen

            elapsed = now - run_started
            rate = images_seen / max(elapsed, 1e-9)
            remaining = max(args.total_images - images_seen, 0)
            eta = remaining / max(rate, 1e-9)
            recent_rate = interval_images / max(now - interval_started, 1e-9)
            percent = 100.0 * min(images_seen, args.total_images) / args.total_images
            model_metric_text = " ".join(
                f"{key}={float(value.item()):.4f}" for key, value in model_metrics.items()
            )
            component_text = " ".join(
                f"{name}={value:.6f}" for name, value in mean_components.items()
            )
            print(
                f"PROGRESS model={args.model_index}/{args.model_count}:{args.model} "
                f"step={global_step:,} "
                f"images={images_seen:,}/{args.total_images:,} ({percent:6.2f}%) "
                f"remaining={remaining:,} avg_rate={rate:,.1f} img/s "
                f"recent_rate={recent_rate:,.1f} img/s elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)} loss={mean_loss:.6f} reconstruction_ema={ema:.6f} "
                f"{component_text} "
                f"data_wait={1000 * interval_data_wait / interval_batches:.1f}ms/batch "
                f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB {model_metric_text}",
                flush=True,
            )
            interval_started = now
            interval_images = 0
            interval_batches = 0
            interval_data_wait = 0.0
            interval_loss.zero_()
            for value in interval_components.values():
                value.zero_()

            last_improvement = int(early_stop_state["last_improvement_images"])
            if (
                images_seen >= args.early_stop_min_images
                and images_seen - last_improvement >= args.early_stop_patience_images
            ):
                stopped_early = True
                stop_reason = (
                    f"smoothed reconstruction loss did not improve by "
                    f"{args.early_stop_min_delta} for "
                    f"{images_seen - last_improvement:,} images"
                )
                print(f"EARLY_STOP model={args.model} reason={stop_reason}", flush=True)

        if should_step and global_step % args.probe_every == 0:
            evaluate_probe(
                evaluation_model,
                probe_uint8,
                device=device,
                batch_size=min(args.batch_size, 16),
                step=global_step,
                output_dir=args.output_dir,
            )

        if should_step and global_step % args.checkpoint_every == 0:
            atomic_torch_save(
                checkpoint_payload(
                    base_model,
                    optimizer,
                    global_step=global_step,
                    images_seen=images_seen,
                    config=config,
                    early_stop_state=early_stop_state,
                ),
                checkpoints / "latest.pt",
            )

        previous_end = time.perf_counter()
        if stopped_early or images_seen >= args.total_images:
            break

    final_probe = evaluate_probe(
        evaluation_model,
        probe_uint8,
        device=device,
        batch_size=min(args.batch_size, 16),
        step=global_step,
        output_dir=args.output_dir,
    )
    final_checkpoint = checkpoints / ("early-stopped.pt" if stopped_early else "epoch-001.pt")
    atomic_torch_save(
        checkpoint_payload(
            base_model,
            optimizer,
            global_step=global_step,
            images_seen=images_seen,
            config=config,
            early_stop_state=early_stop_state,
        ),
        final_checkpoint,
    )
    summary = {
        "model": args.model,
        "status": "early_stopped" if stopped_early else "complete",
        "stop_reason": stop_reason,
        "images_seen": images_seen,
        "global_step": global_step,
        "elapsed_seconds": time.perf_counter() - run_started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "parameter_count": parameter_count,
        "latent_shape": list(getattr(base_model, "latent_shape", ())),
        "loss_weights": loss_weights,
        "final_probe": final_probe,
        "checkpoint": str(final_checkpoint),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"RUN_COMPLETE {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
