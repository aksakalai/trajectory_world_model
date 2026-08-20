from __future__ import annotations

from pathlib import Path

import pytest

from world_model_trajectory.data.distributed_tar_images import (
    DistributedTarImageDataset,
    balanced_shard_allocations,
)


def test_balanced_allocations_are_deterministic_disjoint_and_complete(
    tmp_path: Path,
) -> None:
    shards = []
    for index, size in enumerate((101, 89, 73, 61, 47, 31, 23, 11)):
        path = tmp_path / f"shard-{index}.tar"
        path.write_bytes(b"x" * size)
        shards.append(path)
    first = balanced_shard_allocations(shards, partitions=4, seed=17)
    second = balanced_shard_allocations(shards, partitions=4, seed=17)
    assert first == second
    flattened = [path for allocation in first for path in allocation]
    assert len(flattened) == len(set(flattened)) == len(shards)
    assert set(flattened) == set(shards)
    sizes = [sum(path.stat().st_size for path in allocation) for allocation in first]
    assert max(sizes) - min(sizes) <= max(path.stat().st_size for path in shards)


def test_distributed_dataset_rejects_invalid_rank(tmp_path: Path) -> None:
    shard = tmp_path / "shard.tar"
    shard.write_bytes(b"x")
    with pytest.raises(ValueError, match="rank"):
        DistributedTarImageDataset([shard], rank=4, world_size=4)
