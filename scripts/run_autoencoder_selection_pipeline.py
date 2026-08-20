#!/usr/bin/env python3
"""Run split creation, staged latent selection, final training, and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.data.episode_splits import load_episode_split_manifest
from world_model_trajectory.data.tar_images import TarImageDataset, discover_shards
from world_model_trajectory.models.final_factorized_autoencoder import (
    FactorizedState, FinalFactorizedAutoencoder,
)
from world_model_trajectory.training.final_factorized_objective import loss_components


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def execute(command: list[str]) -> None:
    print("EXEC " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def candidate_config(spec: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    config = json.loads(json.dumps(spec["base_model"]))
    config.update(candidate)
    config["seed"] = int(spec["seed"])
    config["candidate_id"] = candidate["id"]
    return config


def metric_score(metrics: dict[str, object], ranking: dict[str, object]) -> tuple[bool, float]:
    trajectory = metrics["trajectory"]
    exact = metrics["trajectory_exact_core"]
    eligible = (
        float(trajectory["predicted_foreground_fraction"])
        <= float(ranking["max_predicted_foreground_fraction"])
        and float(trajectory["f1"]) >= float(ranking["min_trajectory_f1"])
    )
    value = (
        float(metrics["weighted_total"])
        + 0.25 * (1 - float(trajectory["f1"]))
        + 0.10 * (1 - float(exact["f1"]))
    )
    return eligible, value


def score(record: dict[str, object], ranking: dict[str, object]) -> tuple[bool, float]:
    return metric_score(record["validation_probe"], ranking)


def latent_values(record: dict[str, object]) -> int:
    return int(record["architecture"]["total_spatial_values"])


def pareto_candidates(stage_results: list[dict[str, object]]) -> list[str]:
    eligible = [item for item in stage_results if item["eligible"]]
    frontier = []
    for item in eligible:
        item_latent = int(item["latent_values"])
        item_score = float(item["selection_score"])
        dominated = any(
            int(other["latent_values"]) <= item_latent
            and float(other["selection_score"]) <= item_score
            and (
                int(other["latent_values"]) < item_latent
                or float(other["selection_score"]) < item_score
            )
            for other in eligible
        )
        if not dominated:
            frontier.append(str(item["candidate"]))
    return frontier


def efficiency_winner(
    stage_results: list[dict[str, object]], relative_score_tolerance: float,
) -> tuple[str, dict[str, object]]:
    eligible = [item for item in stage_results if item["eligible"]]
    if not eligible:
        raise RuntimeError("Every candidate failed collapse gates")
    best_score = min(float(item["selection_score"]) for item in eligible)
    cutoff = best_score * (1.0 + relative_score_tolerance)
    equivalent = [
        item for item in eligible if float(item["selection_score"]) <= cutoff
    ]
    selected = min(
        equivalent,
        key=lambda item: (
            int(item["latent_values"]),
            float(item["selection_score"]),
            str(item["candidate"]),
        ),
    )
    decision = {
        "policy": "smallest_latent_within_relative_score_tolerance",
        "relative_score_tolerance": relative_score_tolerance,
        "best_score": best_score,
        "score_cutoff": cutoff,
        "quality_equivalent_candidates": [
            str(item["candidate"])
            for item in sorted(
                equivalent,
                key=lambda item: (
                    int(item["latent_values"]),
                    float(item["selection_score"]),
                ),
            )
        ],
        "selected": str(selected["candidate"]),
        "selected_latent_values": int(selected["latent_values"]),
        "selected_score": float(selected["selection_score"]),
    }
    return str(selected["candidate"]), decision


def latest_record(run_dir: Path, target: int) -> dict[str, object] | None:
    history = run_dir / "stage-history.jsonl"
    if not history.exists():
        return None
    matches = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    selected = [item for item in matches if int(item["target_total_images"]) == target]
    return selected[-1] if selected else None


def smoke_candidates(
    configs: dict[str, dict[str, object]], dataset_root: Path,
    split_manifest: Path, destination: Path,
) -> None:
    if destination.exists():
        return
    shards = discover_shards(dataset_root)
    split = load_episode_split_manifest(
        split_manifest, dataset_root=dataset_root, discovered_shards=shards
    )
    source = TarImageDataset(
        shards, shuffle_shards=False, shuffle_buffer=1, max_images=2,
        episode_splits_by_shard=split.episode_splits_by_shard,
        include_splits=("train",),
    )
    images = torch.stack(list(source)).cuda().contiguous(memory_format=torch.channels_last)
    reports = []
    for candidate_id, config in configs.items():
        model = FinalFactorizedAutoencoder(
            scene_width=int(config["scene_width"]),
            trajectory_width=int(config["trajectory_width"]),
            scene_latent_channels=int(config["scene_channels"]),
            trajectory_latent_channels=int(config["trajectory_channels"]),
            scene_latent_resolution=int(config["scene_resolution"]),
            trajectory_latent_resolution=int(config["trajectory_resolution"]),
        ).cuda().train()
        assert tuple(inspect.signature(model.encode).parameters) == ("images",)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images)
            components = loss_components(output, images, **{
                "target_score_threshold": float(config["objective"]["target_score_threshold"]),
                "antialias_support_radius": int(config["objective"]["antialias_support_radius"]),
                "positive_weight_cap": float(config["objective"]["positive_weight_cap"]),
            })
            total = sum(float(config["loss_weights"][name]) * value for name, value in components.items())
        total.backward()
        branch_norms = {}
        for name, module in {
            "scene_encoder": model.scene_branch.encoder,
            "scene_decoder": model.scene_branch.decoder,
            "trajectory_encoder": model.trajectory_branch.encoder,
            "trajectory_decoder": model.trajectory_branch.decoder,
        }.items():
            grads = [p.grad.float() for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"Smoke gradient failure {candidate_id}/{name}")
            branch_norms[name] = sum(float(g.square().sum()) for g in grads) ** 0.5
            if branch_norms[name] <= 0:
                raise RuntimeError(f"Zero smoke gradient {candidate_id}/{name}")
        state = model.encode(images)
        swapped = model.decode(FactorizedState(
            state.scene.flip(0), state.trajectory, state.crosshair
        ))
        torch.testing.assert_close(swapped.trajectory_logits, model.decode(state).trajectory_logits)
        reports.append({
            "candidate": candidate_id, "status": "pass",
            "architecture": model.architecture_signature,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "gradient_norms": branch_norms, "loss": float(total.detach()),
        })
        del model; torch.cuda.empty_cache()
    atomic_json({"status": "pass", "candidates": reports}, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--validation-probe", type=Path, required=True)
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.selection_config.read_text())
    python = sys.executable
    if not args.split_manifest.exists():
        execute([python, str(REPOSITORY_ROOT / "scripts/build_episode_split_manifest.py"),
                 "--dataset-root", str(args.dataset_root), "--output", str(args.split_manifest)])
    shards = discover_shards(args.dataset_root)
    split = load_episode_split_manifest(
        args.split_manifest, dataset_root=args.dataset_root, discovered_shards=shards
    )
    if not args.validation_probe.exists():
        execute([python, str(REPOSITORY_ROOT / "scripts/build_split_probe.py"),
                 "--dataset-root", str(args.dataset_root), "--split-manifest", str(args.split_manifest),
                 "--images", str(spec["validation_probe_images"]), "--seed", str(spec["seed"]),
                 "--output", str(args.validation_probe)])
    configs = {str(item["id"]): candidate_config(spec, item) for item in spec["candidates"]}
    config_dir = args.output_dir / "configs"; config_dir.mkdir(exist_ok=True)
    for candidate_id, config in configs.items():
        path = config_dir / f"candidate-{candidate_id}.json"
        encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise RuntimeError(f"Immutable candidate config changed: {path}")
        path.write_text(encoded)
    smoke_candidates(configs, args.dataset_root, args.split_manifest,
                     args.output_dir / "smoke-report.json")

    survivors = list(configs)
    selection_history = []
    stages = list(spec["stages"])
    for stage_index, stage in enumerate(stages):
        target = int(stage["target_total_images"])
        final_selection_stage = stage_index == len(stages) - 1
        stage_results = []
        for candidate_id in survivors:
            run_dir = args.output_dir / "candidates" / candidate_id
            record = latest_record(run_dir, target)
            if record is None:
                command = [python, str(REPOSITORY_ROOT / "scripts/train_autoencoder_budget.py"),
                           "--config", str(config_dir / f"candidate-{candidate_id}.json"),
                           "--dataset-root", str(args.dataset_root),
                           "--split-manifest", str(args.split_manifest),
                           "--validation-probe", str(args.validation_probe),
                           "--output-dir", str(run_dir), "--target-total-images", str(target)]
                if (run_dir / "run-contract.json").exists(): command.append("--resume")
                execute(command)
                record = latest_record(run_dir, target)
            if record is None: raise RuntimeError(f"Candidate {candidate_id} produced no stage record")
            eligible, value = score(record, spec["ranking"])
            stage_results.append({"candidate": candidate_id, "eligible": eligible,
                                  "selection_score": value,
                                  "latent_values": latent_values(record),
                                  "score_source": "validation_probe",
                                  "record": record})

        if final_selection_stage and bool(spec["ranking"].get("full_validation_finalists", False)):
            for item in stage_results:
                candidate_id = str(item["candidate"])
                run_dir = args.output_dir / "candidates" / candidate_id
                output = run_dir / f"selection-validation-{target:09d}.json"
                if not output.exists():
                    execute([
                        python,
                        str(REPOSITORY_ROOT / "scripts/evaluate_autoencoder_split.py"),
                        "--checkpoint", str(run_dir / "latest.pt"),
                        "--dataset-root", str(args.dataset_root),
                        "--split-manifest", str(args.split_manifest),
                        "--split", "validation", "--output", str(output),
                    ])
                validation = json.loads(output.read_text())
                eligible, value = metric_score(validation, spec["ranking"])
                item.update({
                    "eligible": eligible,
                    "selection_score": value,
                    "score_source": "full_validation",
                    "full_validation": validation,
                })

        stage_results.sort(key=lambda item: (not item["eligible"], item["selection_score"], item["candidate"]))
        if not any(item["eligible"] for item in stage_results):
            raise RuntimeError(f"Every candidate failed collapse gates at {stage['name']}")
        efficiency_decision = None
        if final_selection_stage:
            winner, efficiency_decision = efficiency_winner(
                stage_results,
                float(spec["ranking"].get("final_relative_score_tolerance", 0.0)),
            )
            survivors = [winner]
        else:
            survivors = [item["candidate"] for item in stage_results[:int(stage["keep"])]]
            pareto = pareto_candidates(stage_results)
            if bool(spec["ranking"].get("retain_pareto_candidates", False)):
                survivors.extend(candidate for candidate in pareto if candidate not in survivors)
        stage_report = {
            "stage": stage,
            "ranking": stage_results,
            "pareto_candidates": pareto_candidates(stage_results),
            "promoted": survivors,
        }
        if efficiency_decision is not None:
            stage_report["efficiency_decision"] = efficiency_decision
        selection_history.append(stage_report)
        atomic_json({"status": "selection_in_progress", "history": selection_history},
                    args.output_dir / "selection-report.json")

    winner = survivors[0]
    winner_dir = args.output_dir / "candidates" / winner
    full_target = int(split.image_counts["train"])
    if bool(spec["ranking"].get("require_approval_before_full_training", False)):
        selection_checkpoint = winner_dir / "latest.pt"
        approval_request = {
            "status": "awaiting_winner_approval",
            "recommended_winner": winner,
            "selection_checkpoint": str(selection_checkpoint.resolve()),
            "selection_checkpoint_sha256": hashlib.sha256(
                selection_checkpoint.read_bytes()
            ).hexdigest(),
            "full_training_target_images": full_target,
            "selection_history": selection_history,
        }
        atomic_json(
            approval_request,
            args.output_dir / "AWAITING_WINNER_APPROVAL.json",
        )
        print("AWAITING_WINNER_APPROVAL " + json.dumps(approval_request, sort_keys=True), flush=True)
        return 0
    if latest_record(winner_dir, full_target) is None:
        execute([python, str(REPOSITORY_ROOT / "scripts/train_autoencoder_budget.py"),
                 "--config", str(config_dir / f"candidate-{winner}.json"),
                 "--dataset-root", str(args.dataset_root), "--split-manifest", str(args.split_manifest),
                 "--validation-probe", str(args.validation_probe), "--output-dir", str(winner_dir),
                 "--target-total-images", str(full_target), "--resume"])
    checkpoint = winner_dir / "latest.pt"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    lock = {
        "status": "winner_locked_before_test", "winner": winner,
        "selection_version": spec["selection_version"],
        "selection_history": selection_history,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": split.manifest_sha256,
    }
    lock_path = args.output_dir / "WINNER_LOCKED_BEFORE_TEST.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("Existing winner lock differs")
    atomic_json(lock, lock_path)
    reports = {}
    for split_name in ("validation", "test"):
        output = args.output_dir / f"final-{split_name}-metrics.json"
        if not output.exists():
            command = [python, str(REPOSITORY_ROOT / "scripts/evaluate_autoencoder_split.py"),
                       "--checkpoint", str(checkpoint), "--dataset-root", str(args.dataset_root),
                       "--split-manifest", str(args.split_manifest), "--split", split_name,
                       "--output", str(output)]
            if split_name == "test": command.extend(("--winner-lock", str(lock_path)))
            execute(command)
        reports[split_name] = json.loads(output.read_text())
    final = {"status": "complete", "winner": winner, "winner_config": configs[winner],
             "train_images": full_target, "selection_history": selection_history,
             "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
             "held_out": reports}
    atomic_json(final, args.output_dir / "FINAL_REPORT.json")
    print("PIPELINE_COMPLETE " + json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
