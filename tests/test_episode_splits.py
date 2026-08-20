import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.episode_splits import load_episode_split_manifest


def write_manifest(root: Path, shard: Path, output: Path) -> None:
    payload = {
        "manifest_version": 1,
        "dataset_root_name": root.name,
        "fractions": {"train": 0.9, "validation": 0.05, "test": 0.05},
        "counts": {
            "combined": {
                "train": {"episodes": 1, "images": 9},
                "validation": {"episodes": 1, "images": 1},
                "test": {"episodes": 1, "images": 1},
            }
        },
        "shards": [{
            "relative_path": str(shard.relative_to(root)).replace("\\", "/"),
            "size_bytes": shard.stat().st_size,
            "collection": "v1",
            "episodes": [
                {"episode_id": "a", "observation_count": 9, "split": "train"},
                {"episode_id": "b", "observation_count": 1, "split": "validation"},
                {"episode_id": "c", "observation_count": 1, "split": "test"},
            ],
        }],
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    output.write_bytes(encoded)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {output.name}\n"
    )


def test_manifest_is_exhaustive_counted_and_checksum_bound(tmp_path: Path) -> None:
    root = tmp_path / "canonical"; root.mkdir()
    shard = root / "shard.tar"; shard.write_bytes(b"tar")
    manifest = tmp_path / "split.json"; write_manifest(root, shard, manifest)
    selection = load_episode_split_manifest(
        manifest, dataset_root=root, discovered_shards=[shard]
    )
    assert selection.image_counts == {"train": 9, "validation": 1, "test": 1}
    assert selection.episode_splits_by_shard[shard.resolve()]["b"] == "validation"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="checksum"):
        load_episode_split_manifest(manifest, dataset_root=root, discovered_shards=[shard])


def test_manifest_rejects_one_underlying_episode_crossing_splits(tmp_path: Path) -> None:
    root = tmp_path / "canonical"; root.mkdir()
    first = root / "v1.tar"; first.write_bytes(b"v1")
    second = root / "v2.tar"; second.write_bytes(b"v2")
    output = tmp_path / "split.json"
    payload = {
        "manifest_version": 2,
        "assignment_unit": "underlying_episode_id_across_v1_v2",
        "dataset_root_name": root.name,
        "fractions": {"train": 0.9, "validation": 0.05, "test": 0.05},
        "counts": {"combined": {
            "train": {"episodes": 1, "images": 9},
            "validation": {"episodes": 1, "images": 1},
            "test": {"episodes": 0, "images": 0},
        }},
        "shards": [
            {"relative_path": first.name, "size_bytes": first.stat().st_size,
             "collection": "v1", "episodes": [
                 {"episode_id": "shared", "observation_count": 9, "split": "train"}
             ]},
            {"relative_path": second.name, "size_bytes": second.stat().st_size,
             "collection": "v2", "episodes": [
                 {"episode_id": "shared", "observation_count": 1, "split": "validation"}
             ]},
        ],
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    output.write_bytes(encoded)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {output.name}\n"
    )
    with pytest.raises(RuntimeError, match="crosses splits"):
        load_episode_split_manifest(
            output, dataset_root=root, discovered_shards=[first, second]
        )
