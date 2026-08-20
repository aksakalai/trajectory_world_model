#!/usr/bin/env python3
"""Create a corrected Movement V1 collection without modifying its source."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.crosshair_canonicalization import (  # noqa: E402
    MIGRATION_VERSION,
    ShardReport,
    canonical_json,
    migrate_v1_shard,
    validate_migrated_v1_shard,
    write_checksum_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", type=Path, required=True)
    parser.add_argument("--output-collection", type=Path, required=True)
    parser.add_argument("--webp-method", type=int, default=0, choices=range(0, 7))
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--shard-subset",
        default="all",
        help="all, a comma/range ordinal expression such as 0-3,8, or assignment IDs",
    )
    parser.add_argument("--expected-shards", type=int, default=192)
    parser.add_argument("--minimum-free-gib", type=float, default=256.0)
    parser.add_argument(
        "--validation-level", choices=("write", "exhaustive"), default="write"
    )
    parser.add_argument("--resume-journal", type=Path)
    parser.add_argument("--jsonl-log", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_outputs(collection: Path) -> list[Path]:
    return sorted(collection.glob("attempts/assignment-*/attempt-*/output"))


def assignment_id(output_dir: Path) -> str:
    for part in output_dir.parts:
        if part.startswith("assignment-"):
            return part
    raise RuntimeError(f"No assignment ID in {output_dir}")


def parse_ordinal_expression(expression: str, count: int) -> set[int]:
    selected: set[int] = set()
    for item in expression.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range {item!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = sorted(index for index in selected if index < 0 or index >= count)
    if invalid:
        raise ValueError(f"Shard ordinals out of range: {invalid}")
    return selected


def select_outputs(outputs: list[Path], expression: str) -> list[Path]:
    if expression == "all":
        return outputs
    if "assignment-" in expression:
        requested = {value.strip() for value in expression.split(",") if value.strip()}
        selected = [output for output in outputs if assignment_id(output) in requested]
        missing = requested - {assignment_id(output) for output in selected}
        if missing:
            raise ValueError(f"Unknown assignment IDs: {sorted(missing)}")
        return selected
    ordinals = parse_ordinal_expression(expression, len(outputs))
    return [output for index, output in enumerate(outputs) if index in ordinals]


def source_artifacts(output_dir: Path) -> tuple[Path, Path, Path]:
    shards = sorted(output_dir.glob("shard-*.tar"))
    if len(shards) != 1:
        raise RuntimeError(f"Expected exactly one shard in {output_dir}, found {len(shards)}")
    dataset = output_dir / "dataset.json"
    checksum = output_dir / "checksums.md5"
    if not dataset.is_file() or not checksum.is_file():
        raise RuntimeError(f"Missing dataset.json or checksums.md5 in {output_dir}")
    return shards[0], dataset, checksum


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def copy_collection_metadata(source: Path, output: Path) -> int:
    """Copy collection files except the three rebuilt artifacts per output directory."""
    copied = 0
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination = output / relative
        if source_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if source_path.parent.name == "output" and (
            source_path.name == "dataset.json"
            or source_path.name == "checksums.md5"
            or (source_path.name.startswith("shard-") and source_path.suffix == ".tar")
        ):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if (
                destination.stat().st_size == source_path.stat().st_size
                and destination.read_bytes() == source_path.read_bytes()
            ):
                continue
            raise RuntimeError(f"Existing copied metadata differs: {destination}")
        shutil.copy2(source_path, destination)
        copied += 1
    return copied


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def publish_output_metadata(
    source_dataset_path: Path,
    destination_dir: Path,
    report: ShardReport,
) -> None:
    dataset = load_dataset(source_dataset_path)
    source_schema_version = dataset.get("schema_version")
    dataset["source_schema_version"] = source_schema_version
    dataset["schema_version"] = f"{source_schema_version}+{MIGRATION_VERSION}"
    dataset["webp_lossless_effort"] = report.webp_method
    dataset.setdefault("parquet_tables", {}).setdefault("frames.parquet", {})[
        "bytes"
    ] = _frames_parquet_size(report.corrected_shard)
    dataset["crosshair_canonicalization"] = {
        "migration_version": MIGRATION_VERSION,
        "source_shard": report.source_shard,
        "source_shard_sha256": report.source_sha256,
        "corrected_shard_sha256": report.corrected_sha256,
        "state_change": "Neutral -> Ready",
        "pixel_change": "17 fixed pixels: (242,242,242) -> (48,242,72)",
    }
    write_atomic(destination_dir / "dataset.json", canonical_json(dataset))
    write_checksum_file([Path(report.corrected_shard)], destination_dir / "checksums.md5")


def _frames_parquet_size(shard: str | Path) -> int:
    import tarfile

    with tarfile.open(shard, "r:*") as archive:
        return archive.getmember("frames.parquet").size


def migrate_job(
    source_shard: str,
    destination_shard: str,
    webp_method: int,
    validation_level: str,
    progress_every: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    report = migrate_v1_shard(
        source_shard,
        destination_shard,
        webp_method=webp_method,
        progress_every=progress_every,
    )
    validation = None
    if validation_level == "exhaustive":
        validation = asdict(validate_migrated_v1_shard(source_shard, destination_shard))
    return asdict(report), validation


class EventWriter:
    def __init__(self, *paths: Path | None):
        self.paths = [path.resolve() for path in paths if path is not None]
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True) + "\n"
        for path in self.paths:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    source = args.source_collection.resolve()
    output = args.output_collection.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("Source and output must be separate, non-nested collections")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every cannot be negative")
    outputs = discover_outputs(source)
    if len(outputs) != args.expected_shards:
        raise SystemExit(
            f"Expected {args.expected_shards} source outputs, found {len(outputs)} under {source}"
        )
    try:
        selected = select_outputs(outputs, args.shard_subset)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not selected:
        raise SystemExit("Shard subset selected no outputs")

    jobs = []
    total_frames = 0
    total_source_bytes = 0
    for source_dir in selected:
        shard, dataset_path, _ = source_artifacts(source_dir)
        relative = source_dir.relative_to(source)
        destination_dir = output / relative
        dataset = load_dataset(dataset_path)
        frames = int(dataset["observation_count"])
        total_frames += frames
        total_source_bytes += shard.stat().st_size
        jobs.append(
            {
                "assignment_id": assignment_id(source_dir),
                "source_dir": source_dir,
                "source_shard": shard,
                "source_dataset": dataset_path,
                "destination_dir": destination_dir,
                "destination_shard": destination_dir / shard.name,
                "frames": frames,
            }
        )

    required_bytes = total_source_bytes + int(args.minimum_free_gib * 1024**3)
    free_bytes = shutil.disk_usage(output.parent if output.parent.exists() else source.parent).free
    preflight = {
        "event": "preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "migration_version": MIGRATION_VERSION,
        "source_collection": str(source),
        "output_collection": str(output),
        "source_shards": len(outputs),
        "selected_shards": len(jobs),
        "selected_frames": total_frames,
        "selected_source_bytes": total_source_bytes,
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "workers": args.workers,
        "webp_method": args.webp_method,
        "tool_commit": args.tool_commit,
        "validation_level": args.validation_level,
        "dry_run": args.dry_run,
    }
    print(json.dumps(preflight, sort_keys=True), flush=True)
    if free_bytes < required_bytes:
        raise SystemExit(
            f"Insufficient free space: {free_bytes / 1024**3:.1f} GiB available, "
            f"{required_bytes / 1024**3:.1f} GiB required"
        )
    if args.dry_run:
        if args.summary:
            write_atomic(args.summary, canonical_json(preflight))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    copied_metadata_files = copy_collection_metadata(source, output)
    writer = EventWriter(args.resume_journal, args.jsonl_log)
    writer.write(preflight | {"copied_metadata_files": copied_metadata_files})
    completed_reports: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    completed_frames = 0
    completed_source_bytes = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_jobs = {}
        for job in jobs:
            job["destination_dir"].mkdir(parents=True, exist_ok=True)
            future = executor.submit(
                migrate_job,
                str(job["source_shard"]),
                str(job["destination_shard"]),
                args.webp_method,
                args.validation_level,
                args.progress_every,
            )
            future_jobs[future] = job

        try:
            for future in concurrent.futures.as_completed(future_jobs):
                job = future_jobs[future]
                report_dict, validation = future.result()
                report = ShardReport(**report_dict)
                publish_output_metadata(job["source_dataset"], job["destination_dir"], report)
                completed_reports.append(report_dict)
                if validation is not None:
                    validations.append(validation)
                completed_frames += report.image_count
                completed_source_bytes += report.source_bytes
                elapsed = time.monotonic() - started
                rate = completed_frames / elapsed if elapsed else 0.0
                remaining = total_frames - completed_frames
                event = {
                    "event": "shard_complete",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "assignment_id": job["assignment_id"],
                    "shards_complete": len(completed_reports),
                    "shards_total": len(jobs),
                    "frames_complete": completed_frames,
                    "frames_total": total_frames,
                    "frames_remaining": remaining,
                    "images_per_second": rate,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": remaining / rate if rate else None,
                    "source_bytes_complete": completed_source_bytes,
                    "source_bytes_total": total_source_bytes,
                    "changed_pixels_total": completed_frames * 17,
                    "errors": 0,
                    "temporary_path": str(job["destination_shard"]) + ".partial",
                    "output_path": str(job["destination_shard"]),
                    "report": report_dict,
                    "validation": validation,
                }
                writer.write(event)
                print(
                    "PROGRESS "
                    f"shards={len(completed_reports)}/{len(jobs)} "
                    f"frames={completed_frames:,}/{total_frames:,} "
                    f"rate={rate:.1f} img/s elapsed={elapsed:.1f}s "
                    f"eta={(remaining / rate if rate else 0):.1f}s "
                    f"output={job['destination_shard']}",
                    flush=True,
                )
        except BaseException as error:
            writer.write(
                {
                    "event": "failure",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "shards_complete": len(completed_reports),
                }
            )
            for future in future_jobs:
                future.cancel()
            raise

    summary = {
        "event": "migration_complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "migration_version": MIGRATION_VERSION,
        "source_collection": str(source),
        "corrected_collection": str(output),
        "shard_count": len(completed_reports),
        "image_count": completed_frames,
        "changed_pixels_total": completed_frames * 17,
        "source_bytes": sum(report["source_bytes"] for report in completed_reports),
        "corrected_bytes": sum(report["corrected_bytes"] for report in completed_reports),
        "elapsed_seconds": time.monotonic() - started,
        "webp_method": args.webp_method,
        "tool_commit": args.tool_commit,
        "validation_level": args.validation_level,
        "copied_metadata_files": copied_metadata_files,
        "v2_statement": "V2 is outside this V1 collection and was not modified.",
        "reports": completed_reports,
        "validations": validations,
    }
    writer.write(summary)
    summary_path = args.summary or output / "migration-release.json"
    write_atomic(summary_path, canonical_json(summary))
    print(
        f"COMPLETE corrected={completed_frames:,} images shards={len(completed_reports)} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
