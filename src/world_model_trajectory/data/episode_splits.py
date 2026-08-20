"""Immutable episode-level train/validation/test manifest support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class EpisodeSplitSelection:
    manifest_path: Path
    manifest_sha256: str
    episode_splits_by_shard: Mapping[Path, Mapping[str, str]]
    image_counts: Mapping[str, int]
    episode_counts: Mapping[str, int]

    def expected_images(self, include_splits: Sequence[str]) -> int:
        requested = {value.strip().lower() for value in include_splits}
        if not requested or not requested <= ALLOWED_SPLITS:
            raise ValueError(f"Invalid requested splits: {sorted(requested)}")
        return sum(self.image_counts[name] for name in requested)


def load_episode_split_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    discovered_shards: Sequence[str | Path],
) -> EpisodeSplitSelection:
    """Load and exhaustively bind a manifest to one canonical dataset release."""
    path = Path(manifest_path).resolve()
    payload_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("Immutable split manifest checksum sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != actual_sha256 or fields[1] != path.name:
        raise RuntimeError("Split manifest checksum sidecar does not match")
    payload = json.loads(payload_bytes)
    manifest_version = int(payload.get("manifest_version", -1))
    if manifest_version not in (1, 2):
        raise RuntimeError("Unsupported episode split manifest version")
    if manifest_version == 2 and payload.get("assignment_unit") != (
        "underlying_episode_id_across_v1_v2"
    ):
        raise RuntimeError("Grouped split manifest has an unsafe assignment unit")
    root = Path(dataset_root).resolve()
    if payload.get("dataset_root_name") != root.name:
        raise RuntimeError("Split manifest dataset identity does not match dataset root")
    fractions = payload.get("fractions")
    if fractions != {"test": 0.05, "train": 0.9, "validation": 0.05}:
        raise RuntimeError("Split manifest is not the required 90/5/5 contract")

    discovered = {Path(item).resolve() for item in discovered_shards}
    mappings: dict[Path, dict[str, str]] = {}
    image_counts = {name: 0 for name in ALLOWED_SPLITS}
    episode_counts = {name: 0 for name in ALLOWED_SPLITS}
    global_episode_splits: dict[str, str] = {}
    for shard_record in payload.get("shards", []):
        shard = (root / shard_record["relative_path"]).resolve()
        if shard in mappings:
            raise RuntimeError(f"Duplicate shard in split manifest: {shard}")
        if shard not in discovered:
            raise RuntimeError(f"Manifest references a noncanonical shard: {shard}")
        if int(shard_record["size_bytes"]) != shard.stat().st_size:
            raise RuntimeError(f"Shard size changed since split creation: {shard}")
        episode_mapping: dict[str, str] = {}
        for episode in shard_record.get("episodes", []):
            episode_id = str(episode["episode_id"])
            split = str(episode["split"]).strip().lower()
            observations = int(episode["observation_count"])
            if split not in ALLOWED_SPLITS or observations < 1:
                raise RuntimeError(f"Invalid episode split row in {shard}: {episode}")
            if episode_id in episode_mapping:
                raise RuntimeError(f"Duplicate episode in {shard}: {episode_id}")
            episode_mapping[episode_id] = split
            image_counts[split] += observations
            previous = global_episode_splits.get(episode_id)
            if previous is not None and previous != split:
                raise RuntimeError(
                    f"Underlying episode crosses splits: {episode_id} is {previous} and {split}"
                )
            if previous is None:
                global_episode_splits[episode_id] = split
                episode_counts[split] += 1
        if not episode_mapping:
            raise RuntimeError(f"Manifest shard has no episodes: {shard}")
        mappings[shard] = episode_mapping
    if set(mappings) != discovered:
        missing = sorted(str(item) for item in discovered - set(mappings))
        raise RuntimeError(f"Split manifest does not cover every shard: {missing[:8]}")

    declared = payload.get("counts", {}).get("combined", {})
    for split in ALLOWED_SPLITS:
        record = declared.get(split, {})
        if int(record.get("images", -1)) != image_counts[split]:
            raise RuntimeError(f"Manifest image count mismatch for {split}")
        if int(record.get("episodes", -1)) != episode_counts[split]:
            raise RuntimeError(f"Manifest episode count mismatch for {split}")
    return EpisodeSplitSelection(
        manifest_path=path,
        manifest_sha256=actual_sha256,
        episode_splits_by_shard=mappings,
        image_counts=image_counts,
        episode_counts=episode_counts,
    )
