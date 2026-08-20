#!/usr/bin/env python3
"""Capture and render failed V1 certification recipes for visual diagnosis."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    member = archive.getmember(name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"tar member cannot be read: {name}")
    with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_summary(recipe: dict[str, Any]) -> str:
    cell = recipe["cell"]
    if recipe["mission"] == "hoop_pass":
        return (
            f"Move {cell['direction']} on the {cell['path']} path, cross the hoop "
            "plane at X=700 with capsule center Y in [-790,-610] cm and Z in "
            "[80,145] cm, then reach and hold at the sampled goal. Camera behavior "
            f"is {cell['facing']}."
        )
    return (
        f"Approach the hoop using {cell['approach']} while facing {cell['facing']}; "
        "register verified rim contact, hold it for the prescribed duration, then "
        f"separate using {cell['recovery']} and hold at the recovery goal."
    )


def actual_summary(episode: dict[str, Any]) -> str:
    mission = episode["collection_mission"]
    termination = episode["termination_reason"]
    if mission == "hoop_pass":
        if episode["hoop_crossing_recorded"]:
            crossing = (
                f"The hoop plane was crossed at Y={episode['hoop_crossing_y']:.1f} cm, "
                f"Z={episode['hoop_crossing_z']:.1f} cm"
            )
            if int(episode["hoop_passes"]) == 0:
                crossing += ", outside the accepted passage corridor"
            else:
                crossing += ", inside the accepted passage corridor"
        else:
            crossing = "No hoop-plane crossing was recorded"
        return (
            f"{crossing}. The episode ended with {termination}; final distance to "
            f"the sampled goal was {episode['final_distance_to_goal_cm']:.1f} cm; "
            f"camera style was {episode['guided_camera_style']}."
        )
    return (
        f"The episode ended with {termination}. Verified contact frames: "
        f"{episode['verified_contact_steps']}; recovery frames: "
        f"{episode['recovery_steps']}; primary objective achieved: "
        f"{str(bool(episode['primary_objective_achieved'])).lower()}; final distance "
        f"to the recovery goal: {episode['final_distance_to_goal_cm']:.1f} cm; "
        f"camera style: {episode['guided_camera_style']}."
    )


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

    report = read_json(certification / "certification-report-bound.json")
    rejected = [item for item in report["results"] if not item["certified"]]
    if not rejected:
        raise ValueError("certification report contains no rejected V1 recipes")
    rejected_ids = {item["recipe_id"] for item in rejected}
    source_manifest = read_json(certification / "certification-manifest.json")
    recipes = [
        recipe for recipe in source_manifest["recipes"]
        if recipe["recipe_id"] in rejected_ids
    ]
    if {recipe["recipe_id"] for recipe in recipes} != rejected_ids:
        raise ValueError("certification manifest does not contain every rejection")

    manifest = {
        **source_manifest,
        "assignment_id": "INVALID-V1-CERTIFICATION-DIAGNOSTIC",
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

    shard = dataset / "shard-w000-000000.tar"
    frames_root = output / "INVALID-extracted-frames"
    frames_root.mkdir()
    with tarfile.open(shard, "r") as archive:
        episodes = read_jsonl_member(archive, "metadata/episodes.jsonl")
        frame_members = [
            member for member in archive.getmembers()
            if member.name.startswith("episodes/") and member.isfile()
        ]
        archive.extractall(frames_root, members=frame_members, filter="data")

    episode_by_recipe = {episode["recipe_id"]: episode for episode in episodes}
    recipe_by_id = {recipe["recipe_id"]: recipe for recipe in recipes}
    rejection_by_id = {item["recipe_id"]: item for item in rejected}
    videos = output / "videos"
    videos.mkdir()
    rendered: list[dict[str, Any]] = []
    fps = int(
        source_manifest["generator"].get(
            "observation_rate_hz",
            source_manifest["generator"].get("observation_rate", 20),
        )
    )
    for index, rejection in enumerate(rejected, start=1):
        recipe_id = rejection["recipe_id"]
        recipe = recipe_by_id[recipe_id]
        episode = episode_by_recipe[recipe_id]
        episode_id = episode["episode_id"]
        video_name = (
            f"{index:02d}__{recipe['mission']}__scenario-{int(recipe['scenario_index']):03d}"
            f"__repetition-{int(recipe['repetition_index']):03d}__{recipe_id}.mp4"
        )
        video = videos / video_name
        subprocess.run([
            str(args.ffmpeg.resolve()),
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_root / "episodes" / episode_id / "frame-%06d.webp"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(video),
        ], check=True)
        rendered.append({
            "recipe_id": recipe_id,
            "episode_id": episode_id,
            "mission": recipe["mission"],
            "scenario_index": recipe["scenario_index"],
            "repetition_index": recipe["repetition_index"],
            "active": recipe["active"],
            "cell": recipe["cell"],
            "certification_error": rejection_by_id[recipe_id]["error"],
            "termination_reason": episode["termination_reason"],
            "observation_count": episode["observation_count"],
            "video_seconds": episode["observation_count"] / fps,
            "expected": expected_summary(recipe),
            "actual": actual_summary(episode),
            "episode_metrics": {
                "guided_camera_style": episode["guided_camera_style"],
                "hoop_crossing_recorded": episode["hoop_crossing_recorded"],
                "hoop_crossing_y": episode["hoop_crossing_y"],
                "hoop_crossing_z": episode["hoop_crossing_z"],
                "hoop_passes": episode["hoop_passes"],
                "verified_contact_steps": episode["verified_contact_steps"],
                "recovery_steps": episode["recovery_steps"],
                "primary_objective_achieved": episode["primary_objective_achieved"],
                "final_distance_to_goal_cm": episode["final_distance_to_goal_cm"],
                "mission_parameters": episode["mission_parameters"],
            },
            "video": str(video),
            "valid_review_media": False,
            "training_eligible": False,
        })

    write_new_json(output / "expected-vs-actual.json", {
        "purpose": "failed V1 certification visual diagnosis only",
        "warning": "INVALID QA MEDIA: never use for training or human approval",
        "source_certification_report": str(
            certification / "certification-report-bound.json"
        ),
        "videos": rendered,
    })
    shutil.rmtree(frames_root)
    print(json.dumps({
        "diagnostic_videos": len(rendered),
        "videos": str(videos),
        "expected_vs_actual": str(output / "expected-vs-actual.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        RuntimeError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
