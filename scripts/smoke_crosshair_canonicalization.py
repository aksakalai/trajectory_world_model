#!/usr/bin/env python3
"""Run real-frame V1/V2 crosshair smoke tests and build a loader-tested artifact."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.crosshair_canonicalization import (  # noqa: E402
    READY_GREEN,
    canonicalize_v1_rgb,
    crosshair_stencil,
    decode_rgb,
    encode_lossless_webp,
    inspect_stencil,
)
from world_model_trajectory.data.tar_images import iter_tar_images  # noqa: E402


@dataclass
class Sample:
    label: str
    shard: Path
    rgb_key: str
    payload: bytes
    rgb: np.ndarray
    row: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-collection", type=Path, required=True)
    parser.add_argument("--v2-collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v1-samples", type=int, default=6)
    return parser.parse_args()


def shards(collection: Path) -> list[Path]:
    result = sorted(collection.glob("attempts/assignment-*/attempt-*/output/shard-*.tar"))
    if not result:
        raise RuntimeError(f"No shards under {collection}")
    return result


def read_payload(archive: tarfile.TarFile, name: str) -> bytes:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise RuntimeError(f"Could not read {name}")
    return extracted.read()


def rows_by_index(table, indices: list[int]) -> list[dict[str, object]]:
    columns = table.to_pydict()
    return [{name: values[index] for name, values in columns.items()} for index in indices]


def load_v1_samples(collection: Path, count: int) -> list[Sample]:
    shard = shards(collection)[0]
    with tarfile.open(shard, "r:*") as archive:
        frames = pq.read_table(io.BytesIO(read_payload(archive, "frames.parquet")))
        if count > frames.num_rows:
            raise RuntimeError("Requested more V1 smoke samples than rows")
        indices = np.linspace(0, frames.num_rows - 1, count, dtype=int).tolist()
        rows = rows_by_index(frames, indices)
        samples = []
        for ordinal, row in enumerate(rows):
            payload = read_payload(archive, str(row["rgb_key"]))
            samples.append(
                Sample(
                    label=f"v1-{ordinal}",
                    shard=shard,
                    rgb_key=str(row["rgb_key"]),
                    payload=payload,
                    rgb=decode_rgb(payload),
                    row=row,
                )
            )
        return samples


def load_v2_samples(collection: Path) -> list[Sample]:
    wanted = {
        "ready_no_trajectory": lambda row: row["crosshair_state"] == "Ready"
        and not row["trajectory_visible"],
        "ready_trajectory": lambda row: row["crosshair_state"] == "Ready"
        and row["trajectory_visible"],
        "cooldown": lambda row: row["crosshair_state"] == "Cooldown",
    }
    found: dict[str, Sample] = {}
    for shard in shards(collection)[:16]:
        with tarfile.open(shard, "r:*") as archive:
            frames = pq.read_table(io.BytesIO(read_payload(archive, "frames.parquet")))
            columns = frames.to_pydict()
            for index in range(frames.num_rows):
                if len(found) == len(wanted):
                    return [found[name] for name in wanted]
                row = {name: values[index] for name, values in columns.items()}
                for label, predicate in wanted.items():
                    if label not in found and predicate(row):
                        payload = read_payload(archive, str(row["rgb_key"]))
                        found[label] = Sample(
                            label=f"v2-{label}",
                            shard=shard,
                            rgb_key=str(row["rgb_key"]),
                            payload=payload,
                            rgb=decode_rgb(payload),
                            row=row,
                        )
        if len(found) == len(wanted):
            break
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Could not find V2 smoke states: {sorted(missing)}")
    return [found[name] for name in wanted]


def add_payload(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(payload))


def build_mixed_tar(
    destination: Path,
    v1_samples: list[Sample],
    v2_samples: list[Sample],
) -> list[np.ndarray]:
    intended = []
    with tarfile.open(destination, "w") as archive:
        for ordinal, sample in enumerate(v1_samples):
            corrected, changed = canonicalize_v1_rgb(sample.rgb)
            if changed != 17:
                raise RuntimeError(f"V1 smoke sample did not change at 17 pixels: {sample.rgb_key}")
            payload = encode_lossless_webp(corrected, method=0)
            add_payload(archive, f"episodes/smoke-v1-{ordinal}/frame-000000.webp", payload)
            intended.append(corrected)
        for ordinal, sample in enumerate(v2_samples):
            add_payload(
                archive,
                f"episodes/smoke-v2-{ordinal}/frame-000000.webp",
                sample.payload,
            )
            intended.append(sample.rgb)
    loaded = list(iter_tar_images(destination))
    if len(loaded) != len(intended):
        raise RuntimeError("Production loader returned the wrong smoke image count")
    for index, (tensor, expected) in enumerate(zip(loaded, intended, strict=True)):
        actual_rgb = tensor.mul(255).round().byte().permute(1, 2, 0).numpy()
        if not np.array_equal(actual_rgb, expected):
            raise RuntimeError(f"Production loader changed smoke image {index}")
    return intended


def render_panel(
    destination: Path,
    originals: list[tuple[str, np.ndarray]],
) -> None:
    cell = 192
    crop_half = 12
    columns = len(originals)
    panel = Image.new("RGB", (columns * cell, cell * 2 + 28), "black")
    draw = ImageDraw.Draw(panel)
    for column, (label, rgb) in enumerate(originals):
        image = Image.fromarray(rgb)
        panel.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (column * cell, 20))
        crop = image.crop((192 - crop_half, 192 - crop_half, 192 + crop_half + 1, 192 + crop_half + 1))
        panel.paste(crop.resize((cell, cell), Image.Resampling.NEAREST), (column * cell, cell + 28))
        draw.text((column * cell + 4, 4), label, fill="white")
    panel.save(destination)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v1 = load_v1_samples(args.v1_collection.resolve(), args.v1_samples)
    v2 = load_v2_samples(args.v2_collection.resolve())
    corrected_v1 = []
    for sample in v1:
        if inspect_stencil(sample.rgb) != "legacy":
            raise RuntimeError(f"V1 source stencil is not white: {sample.rgb_key}")
        if sample.row["crosshair_state"] != "Neutral":
            raise RuntimeError(f"V1 source metadata is not Neutral: {sample.rgb_key}")
        if sample.row["cooldown_remaining_steps"] != 0:
            raise RuntimeError(f"V1 source cooldown is nonzero: {sample.rgb_key}")
        corrected, changed = canonicalize_v1_rgb(sample.rgb)
        difference = np.any(corrected != sample.rgb, axis=2)
        if changed != 17 or int(difference.sum()) != 17:
            raise RuntimeError(f"V1 smoke pixel count failed: {sample.rgb_key}")
        corrected_v1.append(corrected)

    ready_sample = next(sample for sample in v2 if sample.row["crosshair_state"] == "Ready")
    stencil = crosshair_stencil(384, 384)
    ready_pixels = np.asarray([ready_sample.rgb[y, x] for x, y in stencil])
    if not np.all(ready_pixels == READY_GREEN):
        raise RuntimeError("Actual V2 Ready stencil is not exact canonical green")
    for corrected in corrected_v1:
        corrected_pixels = np.asarray([corrected[y, x] for x, y in stencil])
        if not np.array_equal(corrected_pixels, ready_pixels):
            raise RuntimeError("Corrected V1 stencil differs from actual V2 Ready stencil")

    for sample in v2:
        state = str(sample.row["crosshair_state"])
        cooldown = int(sample.row["cooldown_remaining_steps"])
        expected = "Cooldown" if cooldown > 0 else "Ready"
        if state != expected:
            raise RuntimeError(f"V2 metadata mismatch for {sample.rgb_key}")

    mixed_tar = args.output_dir / "real-mixed-smoke.tar"
    build_mixed_tar(mixed_tar, v1, v2)
    panel_items = [
        ("V1 legacy", v1[0].rgb),
        ("V1 corrected", corrected_v1[0]),
    ] + [(sample.label, sample.rgb) for sample in v2]
    panel = args.output_dir / "crosshair-before-after.png"
    render_panel(panel, panel_items)
    report = {
        "v1_samples": len(v1),
        "v2_samples": len(v2),
        "v1_changed_pixels_each": 17,
        "corrected_matches_v2_ready": True,
        "production_loader_exact": True,
        "v2_payloads_untouched": True,
        "mixed_tar": str(mixed_tar),
        "panel": str(panel),
        "v1_source_keys": [sample.rgb_key for sample in v1],
        "v2_source_keys": [sample.rgb_key for sample in v2],
    }
    (args.output_dir / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
