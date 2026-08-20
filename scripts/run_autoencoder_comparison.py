from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed one-epoch autoencoder screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectory-probe", type=Path, required=True)
    parser.add_argument("--loss-config", type=Path, required=True)
    return parser.parse_args()


def stream_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_root / "comparison_config.json")
    shutil.copy2(args.loss_config, args.output_root / "loss_config.json")

    if not args.trajectory_probe.exists():
        probe_command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_trajectory_probe.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--output",
            str(args.trajectory_probe),
            "--seed",
            str(config["seed"]),
        ]
        code = stream_process(probe_command, args.output_root / "probe_build.log")
        if code:
            raise SystemExit(f"Trajectory probe build failed with exit code {code}")

    models = config["models"]
    comparison_started = time.perf_counter()
    completed_durations: list[float] = []
    for index, candidate in enumerate(models, start=1):
        name = candidate["name"]
        run_dir = args.output_root / name
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            completed_durations.append(float(summary["elapsed_seconds"]))
            print(
                f"COMPARISON_SKIP model={name} reason=run_summary_exists "
                f"status={summary['status']}",
                flush=True,
            )
            continue

        remaining_models = len(models) - index + 1
        estimated_remaining = (
            sum(completed_durations) / len(completed_durations) * remaining_models
            if completed_durations
            else float("nan")
        )
        print(
            f"COMPARISON_START model={name} index={index}/{len(models)} "
            f"purpose={candidate['purpose']} estimated_remaining_seconds={estimated_remaining}",
            flush=True,
        )
        early = config["early_stop"]
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train_comparison_autoencoder.py"),
            "--model",
            name,
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(run_dir),
            "--trajectory-probe",
            str(args.trajectory_probe),
            "--loss-config",
            str(args.loss_config),
            "--resume-if-available",
            "--model-index",
            str(index),
            "--model-count",
            str(len(models)),
            "--total-images",
            str(config["total_images"]),
            "--batch-size",
            str(candidate["batch_size"]),
            "--accumulation-steps",
            str(candidate["accumulation_steps"]),
            "--workers",
            str(config["workers"]),
            "--prefetch-factor",
            str(config["prefetch_factor"]),
            "--shuffle-buffer",
            str(config["shuffle_buffer"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--edge-weight",
            str(config["edge_weight"]),
            "--seed",
            str(config["seed"]),
            "--log-every",
            str(config["log_every"]),
            "--checkpoint-every",
            str(config["checkpoint_every"]),
            "--probe-every",
            str(config["probe_every"]),
            "--early-stop-min-images",
            str(early["minimum_images"]),
            "--early-stop-patience-images",
            str(early["patience_images"]),
            "--early-stop-min-delta",
            str(early["minimum_delta"]),
            "--early-stop-ema-alpha",
            str(early["ema_alpha"]),
        ]
        candidate_started = time.perf_counter()
        exit_code = stream_process(command, run_dir / "train.log")
        if exit_code:
            print(
                f"COMPARISON_FAILED model={name} exit_code={exit_code} "
                f"rerun_command={' '.join(sys.argv)}",
                flush=True,
            )
            raise SystemExit(exit_code)
        completed_durations.append(time.perf_counter() - candidate_started)
        print(
            f"COMPARISON_FINISH model={name} index={index}/{len(models)} "
            f"elapsed_seconds={completed_durations[-1]:.1f}",
            flush=True,
        )

    summaries = []
    for candidate in models:
        summary_path = args.output_root / candidate["name"] / "run_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    summaries.sort(
        key=lambda item: item.get("final_probe", {}).get("trajectory_f1", -1.0), reverse=True
    )
    comparison_summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - comparison_started,
        "ranking_metric": "trajectory_f1",
        "runs": summaries,
    }
    (args.output_root / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2), encoding="utf-8"
    )
    print(f"COMPARISON_COMPLETE {json.dumps(comparison_summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
