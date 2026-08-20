#!/usr/bin/env python3
"""Compare aligned PNG/JSONL and WebP/Parquet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image

from review_dataset import read_json_lines, read_parquet_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("png_dataset", type=Path)
    parser.add_argument("webp_dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], tarfile.TarFile, dict[str, list[dict[str, Any]]]]:
    dataset = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    shard = dataset_dir / dataset["shards"][0]
    tar = tarfile.open(shard, mode="r:")
    if dataset["metadata_format"] == "jsonl":
        records = {
            "frames": read_json_lines(tar, "metadata/frames.jsonl"),
            "transitions": read_json_lines(tar, "metadata/transitions.jsonl"),
            "episodes": read_json_lines(tar, "metadata/episodes.jsonl"),
        }
    else:
        records = {
            "frames": read_parquet_records(tar, "frames.parquet"),
            "transitions": read_parquet_records(tar, "transitions.parquet"),
            "episodes": read_parquet_records(
                tar, "episodes.parquet", episodes=True
            ),
        }
    return dataset, tar, records


def normalized_frame(frame: dict[str, Any]) -> dict[str, Any]:
    result = dict(frame)
    result["rgb_key"] = str(result["rgb_key"]).rsplit(".", 1)[0]
    return result


def decoded_rgba(tar: tarfile.TarFile, key: str) -> bytes:
    extracted = tar.extractfile(key)
    if extracted is None:
        raise RuntimeError(f"Missing observation: {key}")
    with Image.open(extracted) as image:
        return image.convert("RGBA").tobytes()


def main() -> int:
    args = parse_args()
    png_dataset, png_tar, png_records = load_dataset(args.png_dataset.resolve())
    webp_dataset, webp_tar, webp_records = load_dataset(args.webp_dataset.resolve())
    try:
        counts = {}
        for name in ("frames", "transitions", "episodes"):
            left = png_records[name]
            right = webp_records[name]
            if len(left) != len(right):
                raise RuntimeError(
                    f"{name} count differs: PNG {len(left)} != WebP {len(right)}"
                )
            if name == "frames":
                pairs = zip(map(normalized_frame, left), map(normalized_frame, right))
            else:
                pairs = zip(left, right)
            for index, (left_record, right_record) in enumerate(pairs):
                if left_record != right_record:
                    raise RuntimeError(
                        f"{name} record {index} differs:\n"
                        f"PNG={left_record}\nWebP={right_record}"
                    )
            counts[name] = len(left)

        aggregate = hashlib.sha256()
        for index, (png_frame, webp_frame) in enumerate(
            zip(png_records["frames"], webp_records["frames"])
        ):
            png_pixels = decoded_rgba(png_tar, png_frame["rgb_key"])
            webp_pixels = decoded_rgba(webp_tar, webp_frame["rgb_key"])
            if png_pixels != webp_pixels:
                raise RuntimeError(f"Decoded RGB pixels differ at frame {index}.")
            aggregate.update(png_pixels)

        report = {
            "equivalent": True,
            "png_schema": png_dataset["schema_version"],
            "webp_schema": webp_dataset["schema_version"],
            "counts": counts,
            "decoded_rgba_sha256": aggregate.hexdigest(),
        }
        output = json.dumps(report, indent=2) + "\n"
        print(output, end="")
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        return 0
    finally:
        png_tar.close()
        webp_tar.close()


if __name__ == "__main__":
    raise SystemExit(main())
