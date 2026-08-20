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
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.tar_images import TarImageDataset, discover_shards
from world_model_trajectory.models.plain_autoencoder import PlainAutoencoder
from world_model_trajectory.training.losses import reconstruction_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deterministic plain autoencoder")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-images", type=int, default=3_381_548)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--panel-every", type=int, default=5_000)
    parser.add_argument("--max-images-per-worker", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    return parser.parse_args()


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This timing run requires a CUDA GPU")

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

    config = vars(args).copy()
    config.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0),
        shard_count=len(shards),
        model="PlainAutoencoder",
        latent_shape=[8, 24, 24],
        precision="bfloat16",
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda")
    model = PlainAutoencoder().to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    global_step = 0
    images_seen = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        images_seen = state["images_seen"]
        print(
            "RESUME NOTE: model and optimizer state restored; iterable tar input restarts "
            "at an epoch boundary, so mid-epoch checkpoints are not exact data cursors.",
            flush=True,
        )
    if args.compile:
        model = torch.compile(model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({**config, "parameter_count": parameter_count}, indent=2), flush=True)

    nominal_steps_per_epoch = math.ceil(args.total_images / args.batch_size)
    total_target_images = args.total_images * args.epochs
    run_started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        dataset = TarImageDataset(
            shards,
            seed=args.seed + epoch,
            shuffle_shards=True,
            shuffle_buffer=args.shuffle_buffer,
            max_images=args.max_images_per_worker,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor if args.workers else None,
        )
        model.train()
        log_started = time.perf_counter()
        log_images = 0
        log_batches = 0
        data_wait_accumulator = 0.0
        step_accumulator = 0.0
        previous_end = time.perf_counter()
        fixed_panel: torch.Tensor | None = None
        epoch_images_seen = 0

        for epoch_step, images in enumerate(loader, start=1):
            data_ready = time.perf_counter()
            data_wait_accumulator += data_ready - previous_end
            images = images.to(device, non_blocking=True)
            if not args.no_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reconstruction = model(images)
                loss, pixel_loss, edge_loss = reconstruction_loss(
                    reconstruction, images, edge_weight=args.edge_weight
                )
            loss.backward()
            optimizer.step()
            step_end = time.perf_counter()
            step_accumulator += step_end - data_ready
            previous_end = step_end

            batch_images = images.shape[0]
            global_step += 1
            images_seen += batch_images
            epoch_images_seen += batch_images
            log_images += batch_images
            log_batches += 1
            if fixed_panel is None:
                fixed_panel = images[: min(8, batch_images)].detach().clone()

            if global_step == 1 or global_step % args.log_every == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                interval = max(now - log_started, 1e-9)
                recent_rate = log_images / interval
                average_rate = images_seen / max(now - run_started, 1e-9)
                epoch_images = min(epoch_images_seen, args.total_images)
                remaining = max(total_target_images - images_seen, 0)
                eta = remaining / max(average_rate, 1e-9)
                percent = 100.0 * epoch_images / args.total_images
                print(
                    f"PROGRESS epoch={epoch + 1}/{args.epochs} "
                    f"step={epoch_step:,} nominal_steps={nominal_steps_per_epoch:,} "
                    f"images={epoch_images:,}/{args.total_images:,} ({percent:6.2f}%) "
                    f"remaining={remaining:,} avg_rate={average_rate:,.1f} img/s "
                    f"recent_rate={recent_rate:,.1f} img/s "
                    f"elapsed={format_duration(now - run_started)} eta={format_duration(eta)} "
                    f"loss={loss.item():.5f} l1={pixel_loss.item():.5f} "
                    f"edge={edge_loss.item():.5f} "
                    f"data_wait={1000 * data_wait_accumulator / log_batches:.1f}ms/batch "
                    f"cpu_submit={1000 * step_accumulator / log_batches:.1f}ms/batch "
                    f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                    flush=True,
                )
                log_started = now
                log_images = 0
                log_batches = 0
                data_wait_accumulator = 0.0
                step_accumulator = 0.0

            if global_step % args.panel_every == 0 and fixed_panel is not None:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    panel_reconstruction = model(fixed_panel)
                comparison = torch.cat((fixed_panel, panel_reconstruction), dim=0)
                save_image(comparison.float(), panels / f"step-{global_step:08d}.jpg", nrow=len(fixed_panel))

            if global_step % args.checkpoint_every == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                atomic_torch_save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "images_seen": images_seen,
                        "config": config,
                    },
                    checkpoints / "latest.pt",
                )

            if epoch_images_seen >= args.total_images:
                break

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        epoch_checkpoint = checkpoints / f"epoch-{epoch + 1:03d}.pt"
        atomic_torch_save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "images_seen": images_seen,
                "config": config,
            },
            epoch_checkpoint,
        )
        print(
            f"EPOCH COMPLETE epoch={epoch + 1} images_seen={images_seen:,} "
            f"elapsed={format_duration(time.perf_counter() - run_started)} "
            f"checkpoint={epoch_checkpoint}",
            flush=True,
        )


if __name__ == "__main__":
    main()
