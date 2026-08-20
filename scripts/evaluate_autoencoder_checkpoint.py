from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from train_comparison_autoencoder import evaluate_probe
from world_model_trajectory.models.comparison_autoencoders import create_comparison_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_name = config["model"]
    model = create_comparison_model(model_name).to(
        "cuda", memory_format=torch.channels_last
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    probe_payload = torch.load(args.probe, map_location="cpu", weights_only=False)
    probe_uint8 = probe_payload["images_uint8"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(exist_ok=True)
    metrics = evaluate_probe(
        model,
        probe_uint8,
        device=torch.device("cuda"),
        batch_size=args.batch_size,
        step=int(checkpoint["global_step"]),
        output_dir=args.output_dir,
    )
    result = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "global_step": int(checkpoint["global_step"]),
        "images_seen": int(checkpoint["images_seen"]),
        "metrics": metrics,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
