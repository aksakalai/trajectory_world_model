from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.tar_images import discover_shards, iter_tar_images
from world_model_trajectory.evaluation.trajectory_metrics import trajectory_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed trajectory-rich diagnostic probe")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--per-shard", type=int, default=4)
    parser.add_argument("--minimum-green-pixels", type=int, default=30)
    parser.add_argument("--minimum-frame-gap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shards = [
        shard for shard in discover_shards(args.dataset_root) if "v2-schema14-" in str(shard)
    ]
    random.Random(args.seed).shuffle(shards)

    selected: list[torch.Tensor] = []
    sources: list[dict[str, object]] = []
    for shard in shards:
        from_this_shard = 0
        last_selected_index = -args.minimum_frame_gap
        for frame_index, image in enumerate(iter_tar_images(shard)):
            if frame_index - last_selected_index < args.minimum_frame_gap:
                continue
            count = int(trajectory_mask(image.unsqueeze(0)).sum().item())
            if count < args.minimum_green_pixels:
                continue
            selected.append(image.mul(255).round().to(torch.uint8))
            sources.append(
                {
                    "shard": str(shard),
                    "stream_frame_index": frame_index,
                    "trajectory_green_pixels": count,
                }
            )
            last_selected_index = frame_index
            from_this_shard += 1
            if len(selected) >= args.images or from_this_shard >= args.per_shard:
                break
        print(
            f"PROBE_SCAN shard={shard.name} selected={len(selected)}/{args.images}",
            flush=True,
        )
        if len(selected) >= args.images:
            break

    if len(selected) != args.images:
        raise RuntimeError(f"Requested {args.images} trajectory frames, found {len(selected)}")
    images = torch.stack(selected)
    payload = {
        "images_uint8": images,
        "sources": sources,
        "selection": vars(args),
    }
    payload["selection"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload["selection"].items()
    }
    torch.save(payload, args.output)
    save_image(
        images[: min(32, len(images))].float().div(255),
        args.output.with_suffix(".jpg"),
        nrow=8,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"sources": sources, "selection": payload["selection"]}, indent=2),
        encoding="utf-8",
    )
    print(f"PROBE_COMPLETE images={len(images)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
