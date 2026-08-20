#!/usr/bin/env python3
"""Assemble and verify a complete dataset root around corrected V1 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


V1_COLLECTION = Path("resolution/v1/resolved-collection")
V2_COLLECTION = Path("v2-schema14-b49ddd0/resolution/resolved-collection")
V2_ROOT = Path("v2-schema14-b49ddd0")
RELEASE_FILE = "CANONICAL_DATASET_RELEASE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "copy-unchanged", "link-v2", "verify-v2"),
        required=True,
    )
    parser.add_argument("--minimum-free-gib", type=float, default=512.0)
    return parser.parse_args()


def ensure_separate(source: Path, output: Path) -> None:
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("Source and corrected roots must be separate and non-nested")


def release_config(
    source: Path, output: Path, release_id: str, tool_commit: str
) -> dict[str, object]:
    return {
        "release_id": release_id,
        "tool_commit": tool_commit,
        "source_root": str(source),
        "corrected_root": str(output),
        "v1_source_collection": str(source / V1_COLLECTION),
        "v1_corrected_collection": str(output / V1_COLLECTION),
        "v2_source_collection": str(source / V2_COLLECTION),
        "v2_training_collection": str(output / V2_COLLECTION),
        "v2_storage_mode": "shared_authoritative_reference",
        "policy": (
            "Original V1 remains untouched, corrected V1 is stored separately, and the "
            "training root references the single authoritative untouched V2 collection."
        ),
        "source_mutation_forbidden": True,
        "v2_mutation_forbidden": True,
    }


def write_config(output: Path, config: dict[str, object]) -> None:
    destination = output / RELEASE_FILE
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        immutable_keys = {
            "release_id",
            "tool_commit",
            "source_root",
            "corrected_root",
            "v1_source_collection",
            "v1_corrected_collection",
            "v2_source_collection",
        }
        for key in immutable_keys:
            if existing.get(key) != config.get(key):
                raise SystemExit(f"Existing release config differs for {key}: {destination}")
        payload = existing | config | {
            "layout_updated_utc": datetime.now(timezone.utc).isoformat()
        }
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return
    output.mkdir(parents=True, exist_ok=True)
    payload = config | {"created_utc": datetime.now(timezone.utc).isoformat()}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def rsync_copy(source: Path, output: Path) -> None:
    command = [
        "rsync",
        "-a",
        "--human-readable",
        "--info=progress2",
        "--exclude=/resolution/v1/resolved-collection/***",
        "--exclude=/v2-schema14-b49ddd0/***",
        str(source) + "/",
        str(output) + "/",
    ]
    subprocess.run(command, check=True)


def verify_v2(source: Path, output: Path) -> None:
    source_v2 = source / V2_COLLECTION
    output_v2 = output / V2_COLLECTION
    if not output_v2.is_dir():
        raise SystemExit(f"Corrected dataset is missing V2: {output_v2}")
    command = [
        "rsync",
        "-aHnci",
        "--delete",
        "--out-format=%i %n%L",
        str(source_v2) + "/",
        str(output_v2) + "/",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    differences = [line for line in result.stdout.splitlines() if line.strip()]
    if differences:
        preview = "\n".join(differences[:20])
        raise SystemExit(f"V2 byte/structure verification failed:\n{preview}")
    print(f"V2 VERIFIED BYTE-IDENTICAL source={source_v2} output={output_v2}", flush=True)


def link_v2(source: Path, output: Path) -> None:
    source_v2_root = source / V2_ROOT
    output_v2_root = output / V2_ROOT
    if output_v2_root.parent.resolve() != output.resolve():
        raise SystemExit(f"Refusing unsafe V2 target: {output_v2_root}")
    if output_v2_root.is_symlink():
        if output_v2_root.resolve() != source_v2_root.resolve():
            raise SystemExit(f"Existing V2 link has an unexpected target: {output_v2_root}")
    else:
        if output_v2_root.exists():
            verify_v2(source, output)
            shutil.rmtree(output_v2_root)
        os.symlink(source_v2_root, output_v2_root, target_is_directory=True)
    if output_v2_root.resolve() != source_v2_root.resolve():
        raise SystemExit("V2 authoritative reference did not resolve to the source V2 root")
    print(
        f"V2 AUTHORITATIVE REFERENCE source={source_v2_root} link={output_v2_root}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    ensure_separate(source, output)
    if not (source / V1_COLLECTION).is_dir() or not (source / V2_COLLECTION).is_dir():
        raise SystemExit("Source root lacks an authoritative V1 or V2 collection")
    usage_path = output if output.exists() else output.parent
    free_bytes = shutil.disk_usage(usage_path).free
    minimum_bytes = int(args.minimum_free_gib * 1024**3)
    print(
        f"PREFLIGHT source={source} output={output} free_gib={free_bytes / 1024**3:.1f} "
        f"minimum_gib={args.minimum_free_gib:.1f}",
        flush=True,
    )
    if free_bytes < minimum_bytes:
        raise SystemExit("Insufficient free space for corrected dataset assembly")
    config = release_config(source, output, args.release_id, args.tool_commit)
    if args.mode == "preflight":
        print(json.dumps(config, indent=2, sort_keys=True), flush=True)
        return 0
    if args.mode == "copy-unchanged":
        write_config(output, config)
        rsync_copy(source, output)
        print(f"UNCHANGED DATA COPIED output={output}", flush=True)
    elif args.mode == "link-v2":
        link_v2(source, output)
        write_config(output, config)
    else:
        write_config(output, config)
        verify_v2(source, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
