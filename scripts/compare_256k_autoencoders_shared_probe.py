#!/usr/bin/env python3
"""Compare old and final 256K checkpoints on one probe and one RGB metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from train_comparison_autoencoder import forward_model, unpack_model_output  # noqa: E402
from world_model_trajectory.evaluation.trajectory_metrics import (  # noqa: E402
    crosshair_metrics,
    trajectory_metrics,
)
from world_model_trajectory.models.comparison_autoencoders import (  # noqa: E402
    create_comparison_model,
)
from world_model_trajectory.models.final_factorized_autoencoder import (  # noqa: E402
    FinalFactorizedAutoencoder,
)


@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    images_uint8: torch.Tensor,
    *,
    kind: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(images_uint8), batch_size):
        target = images_uint8[start : start + batch_size].to(
            device=device, dtype=torch.float32
        ).div_(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "final":
                reconstruction = model(target).image
            else:
                reconstruction, _, _, _ = unpack_model_output(forward_model(model, target))
        outputs.append(reconstruction.float().cpu())
    return torch.cat(outputs)


def metrics(reconstruction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = {
        **trajectory_metrics(reconstruction, target),
        **crosshair_metrics(reconstruction, target),
        "final_l1": F.l1_loss(reconstruction, target),
    }
    return {name: float(value) for name, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    old_payload = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)
    new_payload = torch.load(args.new_checkpoint, map_location="cpu", weights_only=False)
    if int(old_payload["images_seen"]) != 256_000 or int(new_payload["images_seen"]) != 256_000:
        raise RuntimeError("Both checkpoints must be the exact 256,000-image checkpoints")
    probe_payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    images_uint8 = probe_payload["images_uint8"]
    target = images_uint8.float().div(255)
    device = torch.device("cuda")

    old_name = old_payload["config"]["model"]
    old_model = create_comparison_model(old_name).to(
        device, memory_format=torch.channels_last
    )
    old_model.load_state_dict(old_payload["model"], strict=True)
    new_config = new_payload["architecture"]
    new_model = FinalFactorizedAutoencoder(
        scene_width=int(new_config["scene_width"]),
        trajectory_width=int(new_config["trajectory_width"]),
    ).to(device, memory_format=torch.channels_last)
    new_model.load_state_dict(new_payload["model"], strict=True)

    result = {
        "comparison_contract": "same-probe-same-final-rgb-metric-v1",
        "probe": str(args.probe),
        "probe_images": len(images_uint8),
        "old": {
            "model": old_name,
            "checkpoint": str(args.old_checkpoint),
            **metrics(
                reconstruct(old_model, images_uint8, kind="old", device=device, batch_size=args.batch_size),
                target,
            ),
        },
        "new": {
            "model": "final_factorized_autoencoder_v2",
            "checkpoint": str(args.new_checkpoint),
            **metrics(
                reconstruct(new_model, images_uint8, kind="final", device=device, batch_size=args.batch_size),
                target,
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
