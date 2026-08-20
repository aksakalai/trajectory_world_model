"""Rank-aware image streaming for distributed autoencoder training."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import get_worker_info

from world_model_trajectory.data.tar_images import TarImageDataset


def balanced_shard_allocations(
    shards: Sequence[str | Path], *, partitions: int, seed: int
) -> tuple[tuple[Path, ...], ...]:
    """Assign every whole shard exactly once while balancing compressed bytes."""
    if partitions < 1:
        raise ValueError("partitions must be positive")
    paths = [Path(path) for path in shards]
    if not paths:
        raise ValueError("At least one shard is required")
    random.Random(seed).shuffle(paths)
    allocations: list[list[Path]] = [[] for _ in range(partitions)]
    allocated_bytes = [0] * partitions
    for shard in sorted(paths, key=lambda path: path.stat().st_size, reverse=True):
        destination = min(range(partitions), key=allocated_bytes.__getitem__)
        allocations[destination].append(shard)
        allocated_bytes[destination] += shard.stat().st_size
    return tuple(tuple(items) for items in allocations)


class DistributedTarImageDataset(TarImageDataset):
    """Tar image stream partitioned across every DDP rank and loader worker.

    Whole-shard ownership keeps network-volume reads sequential.  A shard can
    belong to exactly one global worker, so ranks cannot train on duplicate
    frames merely because each process constructed its own IterableDataset.
    """

    def __init__(
        self,
        shards: Sequence[str | Path],
        *,
        rank: int,
        world_size: int,
        **kwargs: object,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        super().__init__(shards, **kwargs)
        self.rank = rank
        self.world_size = world_size

    def _worker_shards(self) -> tuple[list[Path], int]:
        info = get_worker_info()
        local_worker = 0 if info is None else info.id
        local_workers = 1 if info is None else info.num_workers
        global_worker = self.rank * local_workers + local_worker
        global_workers = self.world_size * local_workers
        allocations = balanced_shard_allocations(
            self.shards, partitions=global_workers, seed=self.seed
        )
        selected = list(allocations[global_worker])
        if self.shuffle_shards:
            random.Random(self.seed + global_worker).shuffle(selected)
        return selected, global_worker
