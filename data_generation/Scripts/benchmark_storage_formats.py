#!/usr/bin/env python3
"""Run an identical PNG/JSONL versus WebP/Parquet generator benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--episode-seconds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=61001)
    parser.add_argument("--worker-id", type=int, default=210)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--observation-rate", type=int, default=20)
    parser.add_argument("--webp-effort", type=int, default=0)
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode, time.perf_counter() - started


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads((path / "dataset.json").read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    exe = args.exe.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    png_dir = output_root / "png-jsonl"
    webp_dir = output_root / "webp-parquet"
    for path in (png_dir, webp_dir):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite benchmark output: {path}")

    common = [
        str(exe),
        "-GenerateDataset",
        "-Stage=movement",
        f"-Episodes={args.episodes}",
        f"-EpisodeSeconds={args.episode_seconds}",
        f"-SeedStart={args.seed_start}",
        f"-WorkerId={args.worker_id}",
        f"-ObservationRate={args.observation_rate}",
        f"-Width={args.width}",
        f"-Height={args.height}",
        f"-WebPEffort={args.webp_effort}",
        "-Mission=semi_markov",
        "-BuildRevision=storage-format-ab-v1",
        "-RenderOffscreen",
        "-unattended",
        "-nosound",
        "-NoSplash",
        "-NoVSync",
    ]

    results: dict[str, dict[str, Any]] = {}
    for label, storage_format, output_dir in (
        ("png_jsonl", "png_jsonl", png_dir),
        ("webp_parquet", "webp_parquet", webp_dir),
    ):
        command = [
            *common,
            f"-StorageFormat={storage_format}",
            f"-Output={output_dir}",
        ]
        return_code, capture_seconds = run_logged(
            command, output_root / f"{label}-generator.log", exe.parent
        )
        if return_code != 0:
            raise SystemExit(
                f"{label} generator failed with exit code {return_code}; "
                f"see {label}-generator.log"
            )

        finalization_seconds = 0.0
        if storage_format == "webp_parquet":
            finalize_command = [
                sys.executable,
                str(SCRIPT_DIR / "finalize_production_dataset.py"),
                str(output_dir),
            ]
            return_code, finalization_seconds = run_logged(
                finalize_command,
                output_root / f"{label}-finalizer.log",
                SCRIPT_DIR.parent,
            )
            if return_code != 0:
                raise SystemExit(
                    f"Parquet finalization failed with exit code {return_code}; "
                    "see webp_parquet-finalizer.log"
                )

        validation_command = [
            sys.executable,
            str(SCRIPT_DIR / "review_dataset.py"),
            str(output_dir),
            "--validate-only",
        ]
        return_code, validation_seconds = run_logged(
            validation_command,
            output_root / f"{label}-validator.log",
            SCRIPT_DIR.parent,
        )
        if return_code != 0:
            raise SystemExit(
                f"{label} validation failed with exit code {return_code}; "
                f"see {label}-validator.log"
            )

        dataset = load_dataset(output_dir)
        shard_path = output_dir / dataset["shards"][0]
        simulated_seconds = (
            int(dataset["transition_count"]) / int(dataset["observation_rate_hz"])
        )
        end_to_end_seconds = capture_seconds + finalization_seconds
        results[label] = {
            "capture_seconds": capture_seconds,
            "parquet_finalization_seconds": finalization_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "validation_seconds": validation_seconds,
            "simulated_seconds": simulated_seconds,
            "simulated_seconds_per_wall_second": (
                simulated_seconds / end_to_end_seconds
            ),
            "episodes": int(dataset["completed_episode_count"]),
            "transitions": int(dataset["transition_count"]),
            "observations": int(dataset["observation_count"]),
            "shard_bytes": shard_path.stat().st_size,
            "dataset_directory_bytes": directory_bytes(output_dir),
            "bytes_per_observation": (
                directory_bytes(output_dir) / int(dataset["observation_count"])
            ),
        }

    compare_command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_dataset_formats.py"),
        str(png_dir),
        str(webp_dir),
        "--output",
        str(output_root / "equivalence.json"),
    ]
    return_code, comparison_seconds = run_logged(
        compare_command,
        output_root / "equivalence.log",
        SCRIPT_DIR.parent,
    )
    if return_code != 0:
        raise SystemExit(
            "Dataset equivalence failed; see equivalence.log for the first mismatch."
        )

    png = results["png_jsonl"]
    webp = results["webp_parquet"]
    report = {
        "benchmark_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "stage": "movement",
            "mission": "semi_markov",
            "episodes": args.episodes,
            "episode_seconds": args.episode_seconds,
            "seed_start": args.seed_start,
            "worker_id": args.worker_id,
            "observation_rate_hz": args.observation_rate,
            "rgb_width": args.width,
            "rgb_height": args.height,
            "generator_processes": 1,
            "webp_lossless_effort": args.webp_effort,
        },
        "results": results,
        "comparison_seconds": comparison_seconds,
        "webp_size_reduction_percent": (
            100.0
            * (png["dataset_directory_bytes"] - webp["dataset_directory_bytes"])
            / png["dataset_directory_bytes"]
        ),
        "webp_end_to_end_time_change_percent": (
            100.0
            * (webp["end_to_end_seconds"] - png["end_to_end_seconds"])
            / png["end_to_end_seconds"]
        ),
        "equivalence": json.loads(
            (output_root / "equivalence.json").read_text(encoding="utf-8")
        ),
    }
    (output_root / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = (
        "# Storage format benchmark\n\n"
        f"- Observations: {png['observations']:,}\n"
        f"- PNG/JSONL: {png['end_to_end_seconds']:.3f} s, "
        f"{png['dataset_directory_bytes'] / 1_000_000:.3f} MB\n"
        f"- WebP/Parquet: {webp['end_to_end_seconds']:.3f} s, "
        f"{webp['dataset_directory_bytes'] / 1_000_000:.3f} MB\n"
        f"- Size reduction: {report['webp_size_reduction_percent']:.2f}%\n"
        f"- End-to-end time change: "
        f"{report['webp_end_to_end_time_change_percent']:+.2f}%\n"
        f"- Decoded pixels and metadata equivalent: "
        f"{report['equivalence']['equivalent']}\n"
    )
    (output_root / "benchmark.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
