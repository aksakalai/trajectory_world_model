#!/usr/bin/env python3
"""Validate and render the planned 186-video V2 human-review set.

This tool never launches Unreal. Generate the immutable review assignments with
``v2_dataset_controller.py review-plan``, run them through ``dataset_worker.py``,
then use this script to validate or render their exact recorded frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_output(collection: Path, entry: dict[str, Any], plan_id: str) -> Path:
    assignment_id = str(entry["assignment_id"])
    assignment = read_json(collection / "assignments" / f"{assignment_id}.json")
    if assignment.get("plan_id") != plan_id or len(assignment.get("recipes", [])) != 1:
        raise ValueError(f"{assignment_id} is not the manifest's immutable assignment")
    recipe = assignment["recipes"][0]
    expected_binding = [{
        "recipe_id": recipe["recipe_id"],
        "replay_identity": recipe.get("replay_identity"),
        "mission_type": recipe.get("mission_type"),
        "mission_solution": recipe.get("mission_solution"),
    }]
    if (
        recipe.get("recipe_id") != entry.get("recipe_id")
        or recipe.get("mission_type") != entry.get("mission_type")
        or recipe.get("mission_solution") != entry.get("solution")
    ):
        raise ValueError(f"{assignment_id} recipe differs from the review manifest")
    matches = []
    for path in sorted((collection / "results").glob(f"{assignment_id}--attempt-*.json")):
        value = read_json(path)
        if value.get("technical_result") == "validated":
            output = Path(value["output_directory"]).resolve()
            dataset_json = output / "dataset.json"
            if (
                value.get("plan_id") != plan_id
                or value.get("assignment_id") != assignment_id
                or value.get("assignment_digest") != assignment.get("assignment_digest")
                or value.get("resolved_recipe_ids") != [recipe["recipe_id"]]
                or value.get("recipe_bindings") != expected_binding
                or not dataset_json.is_file()
                or value.get("output_dataset_sha256") != sha256_file(dataset_json)
            ):
                raise ValueError(
                    f"{assignment_id} validated result is not bound to its exact recipe"
                )
            dataset_metadata = read_json(dataset_json)
            if (
                dataset_metadata.get("plan_id") != plan_id
                or dataset_metadata.get("assignment_id") != assignment_id
            ):
                raise ValueError(f"{assignment_id} dataset metadata is not plan-bound")
            from dataset_worker import read_episode_rows
            rows = read_episode_rows(output)
            if len(rows) != 1:
                raise ValueError(f"{assignment_id} review output must contain exactly one episode")
            row = rows[0]
            if (
                row.get("plan_id") != plan_id
                or row.get("assignment_id") != assignment_id
                or row.get("recipe_id") != recipe["recipe_id"]
                or row.get("v2_replay_identity") != recipe.get("replay_identity")
                or row.get("v2_mission_type") != recipe.get("mission_type")
            ):
                raise ValueError(f"{assignment_id} episode is not the planned immutable replay")
            matches.append(output)
    if len(matches) != 1:
        raise ValueError(
            f"{assignment_id} must have exactly one validated immutable result; "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    collection = args.collection.resolve()
    manifest = read_json(collection / "v2-review-plan.json")
    entries = manifest.get("entries", [])
    if (
        len(entries) != 186
        or len({entry["mission_type"] for entry in entries}) != 62
        or any(
            sum(item["mission_type"] == entry["mission_type"] for item in entries) != 3
            for entry in entries
        )
    ):
        raise ValueError("review manifest is not exactly three examples for all 62 types")
    output = (args.output or collection / "videos").resolve()
    if not args.validate_only:
        output.mkdir(parents=True, exist_ok=True)
    review_script = Path(__file__).with_name("review_dataset.py")
    rendered: list[str] = []
    for entry in entries:
        dataset = result_output(collection, entry, manifest["plan_id"])
        command = [
            sys.executable, str(review_script), str(dataset),
            "--episode", entry["episode_id"],
        ]
        if args.validate_only:
            command.append("--validate-only")
        else:
            command.extend(["--output", str(output)])
            if args.ffmpeg:
                command.extend(["--ffmpeg", str(args.ffmpeg.resolve())])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        if not args.validate_only:
            source = output / f"{entry['episode_id']}.mp4"
            destination = output / entry["output_video"]
            if destination.exists():
                raise ValueError(f"refusing to overwrite review video: {destination}")
            source.replace(destination)
            rendered.append(destination.name)
    summary = {
        "validated_video_count": 186,
        "rendered_video_count": 0 if args.validate_only else len(rendered),
        "mission_type_count": 62,
        "examples_per_type": 3,
        "width": 384,
        "height": 384,
    }
    if not args.validate_only:
        (output / "review-set.json").write_text(
            json.dumps({**summary, "videos": rendered}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
