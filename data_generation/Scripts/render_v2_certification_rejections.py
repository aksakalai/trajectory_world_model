#!/usr/bin/env python3
"""Render explicitly invalid V2 recipes from a failed certification report.

This is a human-debugging tool. Its outputs are quarantined QA media and can
never be credited, validated as review media, or substituted into a plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("certification", type=Path)
    result.add_argument("--executable", type=Path, required=True)
    result.add_argument("--ffmpeg", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    certification = args.certification.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"diagnostic output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = certification / "certification-report-bound.json"
    if not report_path.exists():
        report_path = certification / "certification-report.json"
    report = read_json(report_path)
    rejected = [item for item in report["results"] if not item["certified"]]
    if not rejected:
        raise ValueError("certification report contains no rejected recipes")
    rejected_ids = {item["recipe_id"] for item in rejected}
    source_manifest = read_json(certification / "certification-manifest.json")
    recipes = [
        recipe for recipe in source_manifest["recipes"]
        if recipe["recipe_id"] in rejected_ids
    ]
    if {item["recipe_id"] for item in recipes} != rejected_ids:
        raise ValueError("certification manifest does not contain every rejected recipe")
    manifest = {
        **source_manifest,
        "assignment_id": "INVALID-REJECTED-RECIPE-DIAGNOSTIC",
        "attempt_id": "INVALID-DIAGNOSTIC",
        "executor_id": "INVALID-DIAGNOSTIC-RENDERER",
        "recipes": recipes,
    }
    manifest_path = output / "INVALID-diagnostic-request.json"
    write_new_json(manifest_path, manifest)
    dataset = output / "INVALID-dataset-source"
    executable = args.executable.resolve()
    command = [str(executable)]
    if "unrealeditor" in executable.name.lower():
        command.extend([str(ROOT / "he_grenade_game.uproject"), "-game"])
    command.extend([
        "-GenerateDataset",
        "-AllowUncertifiedV2Diagnostic",
        f"-RecipeManifest={manifest_path}",
        f"-Output={dataset}",
        "-RenderOffscreen",
        "-unattended",
        "-nosound",
        "-NoSplash",
        "-NoVSync",
    ])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"diagnostic Unreal capture exited {completed.returncode}")
    finalized = subprocess.run([
        sys.executable,
        str(ROOT / "Scripts" / "finalize_production_dataset.py"),
        str(dataset),
    ], check=False)
    if finalized.returncode != 0:
        raise RuntimeError("diagnostic dataset finalization failed")

    frames_root = output / "INVALID-frames"
    frames_root.mkdir()
    with tarfile.open(dataset / "shard-w000-000000.tar", "r") as archive:
        archive.extractall(frames_root, filter="data")
    videos = output / "INVALID-videos"
    videos.mkdir()
    rendered: list[dict[str, object]] = []
    by_id = {recipe["recipe_id"]: recipe for recipe in recipes}
    for result in rejected:
        recipe = by_id[result["recipe_id"]]
        episode_id = f"p-e{int(recipe['episode_index']):09d}"
        video_name = (
            f"INVALID_REJECTED__{recipe['mission_type']}__"
            f"episode-{int(recipe['episode_index']):03d}.mp4"
        )
        video = videos / video_name
        subprocess.run([
            str(args.ffmpeg.resolve()),
            "-y",
            "-framerate", str(source_manifest["generator"]["observation_rate"]),
            "-i", str(frames_root / "episodes" / episode_id / "frame-%06d.webp"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(video),
        ], check=True)
        rendered.append({
            "recipe_id": recipe["recipe_id"],
            "mission_type": recipe["mission_type"],
            "episode_id": episode_id,
            "certification_error": result["error"],
            "video": str(video),
            "valid_review_media": False,
            "training_eligible": False,
        })
    write_new_json(output / "INVALID-diagnostic-video-manifest.json", {
        "purpose": "rejected_v2_recipe_visual_diagnosis_only",
        "warning": "INVALID QA MEDIA: never use for training or review approval",
        "source_certification_report": str(report_path),
        "videos": rendered,
    })
    print(json.dumps({"invalid_videos": len(rendered), "output": str(videos)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError,
            subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
