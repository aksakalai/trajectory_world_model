#!/usr/bin/env python3
"""Create, verify, certify, and resolve the Windows 3,333,333-frame campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import dataset_controller as v1
import resolve_plan_certification as resolver
import v2_dataset_controller as v2


V1_FRAMES = 1_111_111
V2_FRAMES = 2_222_222
TOTAL_FRAMES = V1_FRAMES + V2_FRAMES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_v1(root: Path, args: argparse.Namespace) -> None:
    v1.create_plan(Namespace(
        collection=root,
        frame_budget=V1_FRAMES,
        workers=1,
        recipes_per_assignment=args.recipes_per_assignment,
        tail_single_recipes=64,
        episode_seconds=args.episode_seconds,
        observation_rate=args.observation_rate,
        width=args.width,
        height=args.height,
        storage_format="webp_parquet",
        webp_effort=0,
        seed_start=args.v1_seed_start,
        duration_calibration=v1.DEFAULT_DURATION_CALIBRATION,
        split="train",
        plan_id=None,
        allow_infeasible_diagnostic=False,
    ))
    v1.verify_plan(Namespace(collection=root))


def create_v2(root: Path, args: argparse.Namespace) -> None:
    v2.create_plan(Namespace(
        collection=root,
        frame_budget=V2_FRAMES,
        workers=1,
        recipes_per_assignment=args.recipes_per_assignment,
        episode_seconds=args.episode_seconds,
        observation_rate=args.observation_rate,
        width=args.width,
        height=args.height,
        storage_format="webp_parquet",
        webp_effort=0,
        seed_start=args.v2_seed_start,
        evaluation_percent=args.evaluation_percent,
        plan_id=None,
    ))
    verification = v2.verify_plan(root)
    write_json(root / "plan" / "structural-verification.json", verification)
    if not verification["valid"]:
        raise RuntimeError("V2 structural verification failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--episode-seconds", type=int, default=150)
    parser.add_argument("--observation-rate", type=int, default=20)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--recipes-per-assignment", type=int, default=32)
    parser.add_argument("--evaluation-percent", type=int, default=10)
    parser.add_argument("--v1-seed-start", type=int, default=410000)
    parser.add_argument("--v2-seed-start", type=int, default=510000)
    parser.add_argument("--max-replacement-attempts", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_replacement_attempts <= 10:
        raise ValueError("max-replacement-attempts must be between 1 and 10")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"campaign output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()

    v1_root = root / "candidate-plans" / "v1-1111111"
    v2_root = root / "candidate-plans" / "v2-2222222"
    create_v1(v1_root, args)
    create_v2(v2_root, args)
    v1_plan = v1.read_json(v1_root / "plan" / "collection-plan.json")
    v2_plan = v2.read_json(v2_root / "plan" / "collection-plan.json")
    v1_recipes = [
        recipe for recipe in v1.read_jsonl(v1_root / "plan" / "recipes.jsonl")
        if recipe.get("active")
    ]
    v2_recipes = v2.read_jsonl(v2_root / "plan" / "recipes.jsonl")
    v1_recipe_counts = Counter(str(recipe["mission"]) for recipe in v1_recipes)
    v2_recipe_counts = Counter(str(recipe["source"]) for recipe in v2_recipes)

    v1_resolution = resolver.resolve_version(
        "v1", v1_root, executable, root / "resolution" / "v1", args.max_replacement_attempts
    )
    v2_resolution = resolver.resolve_version(
        "v2", v2_root, executable, root / "resolution" / "v2", args.max_replacement_attempts
    )
    v1_targets = dict(v1_plan["target_frames_by_mission"])
    v2_distribution = dict(v2_plan["planned_distribution"])
    mission_frames = (
        sum(value for key, value in v1_targets.items() if key != "semi_markov")
        + int(v2_distribution["source_frames"]["mission"])
    )
    semi_frames = int(v1_targets["semi_markov"]) + int(
        v2_distribution["source_frames"]["semi_markov"]
    )
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "platform": "windows",
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "executable_size_bytes": executable.stat().st_size,
        "requested": {
            "total_frames": TOTAL_FRAMES,
            "v1_frames": V1_FRAMES,
            "v2_frames": V2_FRAMES,
            "mission_frames": 1_000_000,
            "semi_markov_frames": 2_333_333,
            "conceptual_source_shares": {"semi_markov": 0.7, "mission": 0.3},
        },
        "planned": {
            "total_frames": V1_FRAMES + V2_FRAMES,
            "mission_frames": mission_frames,
            "semi_markov_frames": semi_frames,
            "v1_frames_by_mission": v1_targets,
            "v2_frames_by_source": v2_distribution["source_frames"],
            "v2_frames_by_family": v2_distribution["family_frames"],
            "v2_frames_by_mission_type": v2_distribution["mission_type_frames"],
            "frame_percentages": {
                "mission": mission_frames / TOTAL_FRAMES,
                "semi_markov": semi_frames / TOTAL_FRAMES,
                "v1": V1_FRAMES / TOTAL_FRAMES,
                "v2": V2_FRAMES / TOTAL_FRAMES,
            },
            "v1_active_recipe_count": len(v1_recipes),
            "v1_active_recipe_counts_by_mission": dict(sorted(v1_recipe_counts.items())),
            "v2_active_recipe_count": len(v2_recipes),
            "v2_active_recipe_counts_by_source": dict(sorted(v2_recipe_counts.items())),
            "v2_mission_type_count": len(v2_distribution["mission_type_frames"]),
        },
        "candidate_plans": {
            "v1": {"plan_id": v1_plan["plan_id"], "path": str(v1_root)},
            "v2": {"plan_id": v2_plan["plan_id"], "path": str(v2_root)},
        },
        "resolution": {"v1": v1_resolution, "v2": v2_resolution},
        "complete": (
            mission_frames == 1_000_000
            and semi_frames == 2_333_333
            and v1_resolution["unresolved_count"] == 0
            and v2_resolution["unresolved_count"] == 0
            and v1_resolution["final_certification_complete"]
            and v2_resolution["final_certification_complete"]
        ),
    }
    write_json(root / "windows-plan-verification-report.json", report)
    print(json.dumps({
        "complete": report["complete"],
        "total_frames": report["planned"]["total_frames"],
        "mission_frames": mission_frames,
        "semi_markov_frames": semi_frames,
        "v1_rejected": v1_resolution["rejected_original_count"],
        "v2_rejected": v2_resolution["rejected_original_count"],
        "report": str(root / "windows-plan-verification-report.json"),
    }, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
