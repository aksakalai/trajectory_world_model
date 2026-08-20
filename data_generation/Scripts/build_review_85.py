#!/usr/bin/env python3
"""Render the fixed 85-video V1/V2 representative review selection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

from dataset_worker import read_episode_rows


def prepare_output_directory(root: Path) -> Path:
    """Return the persistent output directory without breaking render retries."""
    output = root / "videos"
    output.mkdir(parents=True, exist_ok=True)
    return output


def render(dataset: Path, episodes: list[str], output: Path, prefix: str) -> list[str]:
    temporary = output.parent / f".{prefix}-render"
    command = [
        sys.executable,
        str(Path(__file__).with_name("review_dataset.py")),
        str(dataset),
        "--output", str(temporary),
        "--ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(),
    ]
    for episode in episodes:
        command += ["--episode", episode]
    if subprocess.run(command, check=False).returncode:
        raise RuntimeError(f"rendering failed for {dataset}")
    manifest = json.loads((temporary / "review_manifest.json").read_text())
    names = []
    for index, item in enumerate(manifest["videos"]):
        source = temporary / item["video"]
        name = f"{prefix}-{index + 1:02d}-{source.name}"
        shutil.move(source, output / name)
        names.append(name)
    shutil.rmtree(temporary)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = prepare_output_directory(root)
    rendered: list[str] = []

    v1_guided = root / "v1-guided"
    guided = read_episode_rows(v1_guided)
    chosen: list[dict] = []
    quotas = {"object_view": 5, "contact_recovery": 4, "ramp_traverse": 4, "hoop_pass": 2}
    for mission, quota in quotas.items():
        rows = [row for row in guided if row.get("collection_mission") == mission and row.get("mission_success")]
        if len(rows) < quota:
            raise RuntimeError(f"V1 {mission} has only {len(rows)} successful examples")
        chosen.extend(rows[:quota])
    rendered += render(v1_guided, [row["episode_id"] for row in chosen], output, "v1-guided")

    v1_semi = root / "v1-semi-markov"
    semi_rows = read_episode_rows(v1_semi)
    if len(semi_rows) != 4:
        raise RuntimeError("V1 semi-Markov capture must contain four episodes")
    rendered += render(v1_semi, [row["episode_id"] for row in semi_rows], output, "v1-semi")

    v2 = root / "v2"
    mission_rows: dict[str, tuple[Path, dict]] = {}
    semi: list[tuple[Path, dict]] = []
    for result_path in sorted((v2 / "results").glob("*.json")):
        result = json.loads(result_path.read_text())
        if result.get("technical_result") != "validated":
            continue
        dataset = Path(result["output_directory"])
        for row in read_episode_rows(dataset):
            if row.get("v2_source") == "semi_markov" and len(semi) < 4:
                semi.append((dataset, row))
            elif row.get("v2_source") == "mission" and row.get("mission_success"):
                mission_rows.setdefault(row["v2_mission_type"], (dataset, row))
    if len(mission_rows) != 62 or len(semi) != 4:
        raise RuntimeError(f"V2 selection incomplete: {len(mission_rows)} missions, {len(semi)} semi-Markov")
    for index, (mission, (dataset, row)) in enumerate(sorted(mission_rows.items()), 1):
        rendered += render(dataset, [row["episode_id"]], output, f"v2-mission-{index:02d}-{mission}")
    for index, (dataset, row) in enumerate(semi, 1):
        rendered += render(dataset, [row["episode_id"]], output, f"v2-semi-{index:02d}")

    if len(rendered) != 85 or len(list(output.glob("*.mp4"))) != 85:
        raise RuntimeError(f"expected exactly 85 videos; rendered {len(rendered)}")
    (output / "review-85.json").write_text(json.dumps({"video_count": 85, "videos": rendered}, indent=2) + "\n")
    print(json.dumps({"video_count": 85, "videos": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
