from __future__ import annotations

import io
import random
import re
import tarfile
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms.functional import pil_to_tensor


AUTHORITATIVE_COLLECTIONS = (
    Path("resolution/v1/resolved-collection"),
    Path("v2-schema14-b49ddd0/resolution/resolved-collection"),
)

FRAME_MEMBER_PATTERN = re.compile(
    r"^episodes/(?P<episode>[^/]+)/frame-(?P<index>\d+)\.webp$"
)


def discover_shards(dataset_root: str | Path) -> list[Path]:
    """Return only tar shards from the two authoritative resolved collections."""
    root = Path(dataset_root)
    shards: list[Path] = []
    for relative_collection in AUTHORITATIVE_COLLECTIONS:
        collection = root / relative_collection
        if not collection.is_dir():
            raise FileNotFoundError(f"Authoritative collection is missing: {collection}")
        shards.extend(collection.glob("attempts/assignment-*/attempt-*/output/shard-*.tar"))
    shards = sorted(path.resolve() for path in shards)
    if not shards:
        raise FileNotFoundError(f"No authoritative tar shards found below {root}")
    return shards


def _is_rgb_frame(member: tarfile.TarInfo) -> bool:
    return (
        member.isfile()
        and member.name.startswith("episodes/")
        and member.name.endswith(".webp")
        and "/frame-" in member.name
    )


def iter_tar_images(
    shard: str | Path,
    *,
    episode_splits: Mapping[str, str] | None = None,
    include_splits: Sequence[str] = ("train",),
) -> Iterator[torch.Tensor]:
    """Sequentially decode frame images from one tar as float CHW tensors in [0, 1]."""
    accepted = {value.strip().lower() for value in include_splits}
    if episode_splits is not None and not accepted:
        raise ValueError("At least one image split must be selected")
    with tarfile.open(shard, mode="r|") as archive:
        for member in archive:
            if not _is_rgb_frame(member):
                continue
            if episode_splits is not None:
                match = FRAME_MEMBER_PATTERN.match(member.name)
                if match is None:
                    raise RuntimeError(f"Unexpected frame member name: {member.name}")
                episode = match.group("episode")
                try:
                    split = episode_splits[episode]
                except KeyError as error:
                    raise RuntimeError(
                        f"Frame references episode absent from split manifest: {episode}"
                    ) from error
                if split not in accepted:
                    continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not extract {member.name} from {shard}")
            with Image.open(io.BytesIO(extracted.read())) as image:
                image = image.convert("RGB")
                yield pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)


def _decode_rgb_frame(archive: tarfile.TarFile, member: tarfile.TarInfo) -> torch.Tensor:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"Could not extract {member.name}")
    with Image.open(io.BytesIO(extracted.read())) as image:
        return pil_to_tensor(image.convert("RGB"))


def iter_tar_frame_sequences(
    shard: str | Path,
    *,
    context_frames: int = 8,
    prediction_frames: int = 8,
    frame_stride: int = 1,
    window_step: int = 8,
    episode_splits: Mapping[str, str] | None = None,
    include_splits: Sequence[str] = ("train",),
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield episode-safe uint8 context/target sequences from one tar shard.

    The archive is read once, sequentially. Windows never cross episode or
    discontinuous frame boundaries. ``window_step=8`` makes adjacent 8->8
    examples share only the context/target boundary while avoiding the 16x
    image-decoding amplification of independently reading overlapping windows.
    """
    if context_frames < 1 or prediction_frames < 1:
        raise ValueError("context_frames and prediction_frames must be positive")
    if frame_stride < 1 or window_step < 1:
        raise ValueError("frame_stride and window_step must be positive")
    accepted = {value.strip().lower() for value in include_splits}
    if episode_splits is not None and not accepted:
        raise ValueError("At least one sequence split must be selected")
    sampled_length = context_frames + prediction_frames
    raw_span = 1 + (sampled_length - 1) * frame_stride
    buffer: deque[tuple[int, torch.Tensor]] = deque(maxlen=raw_span)
    current_episode: str | None = None
    previous_index: int | None = None
    next_start = 0

    with tarfile.open(shard, mode="r|") as archive:
        for member in archive:
            if not _is_rgb_frame(member):
                continue
            match = FRAME_MEMBER_PATTERN.match(member.name)
            if match is None:
                raise RuntimeError(f"Unexpected frame member name: {member.name}")
            episode = match.group("episode")
            frame_index = int(match.group("index"))
            if episode_splits is not None:
                try:
                    split = episode_splits[episode]
                except KeyError as error:
                    raise RuntimeError(
                        f"Frame references episode absent from split manifest: {episode}"
                    ) from error
                if split not in accepted:
                    current_episode = None
                    previous_index = None
                    buffer.clear()
                    continue
            if episode != current_episode or (
                previous_index is not None and frame_index != previous_index + 1
            ):
                buffer.clear()
                current_episode = episode
                next_start = frame_index
            buffer.append((frame_index, _decode_rgb_frame(archive, member)))
            previous_index = frame_index

            if len(buffer) < raw_span or buffer[0][0] < next_start:
                continue
            if buffer[0][0] != next_start:
                continue
            sampled = [buffer[offset][1] for offset in range(0, raw_span, frame_stride)]
            context = torch.stack(sampled[:context_frames])
            target = torch.stack(sampled[context_frames:])
            yield context, target
            next_start += window_step


def count_tar_frame_sequences(
    shard: str | Path,
    *,
    context_frames: int = 8,
    prediction_frames: int = 8,
    frame_stride: int = 1,
    window_step: int = 8,
    episode_splits: Mapping[str, str] | None = None,
    include_splits: Sequence[str] = ("train",),
) -> int:
    """Count exact episode-safe windows from the shard episode table."""
    import pyarrow.parquet as parquet

    sampled_length = context_frames + prediction_frames
    raw_span = 1 + (sampled_length - 1) * frame_stride
    with tarfile.open(shard, mode="r") as archive:
        try:
            member = archive.getmember("episodes.parquet")
        except KeyError as error:
            raise RuntimeError(f"episodes.parquet is missing from {shard}") from error
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract episodes.parquet from {shard}")
        columns = (
            ["episode_id", "observation_count"]
            if episode_splits is not None
            else ["observation_count"]
        )
        table = parquet.read_table(io.BytesIO(extracted.read()), columns=columns)
    accepted = {value.strip().lower() for value in include_splits}
    if episode_splits is not None and not accepted:
        raise ValueError("At least one sequence split must be selected")
    episode_ids = (
        table.column("episode_id").to_pylist()
        if episode_splits is not None
        else [None] * table.num_rows
    )
    rows = zip(episode_ids, table.column("observation_count").to_pylist(), strict=True)
    counts = []
    for episode_id, count in rows:
        if episode_splits is not None:
            episode = str(episode_id)
            if episode not in episode_splits:
                raise RuntimeError(f"Episode absent from split manifest: {episode}")
            if episode_splits[episode] not in accepted:
                continue
        counts.append(count)
    return sum(
        1 + (int(count) - raw_span) // window_step
        for count in counts
        if count is not None and int(count) >= raw_span
    )


def count_frame_sequences(
    shards: Sequence[str | Path],
    *,
    context_frames: int = 8,
    prediction_frames: int = 8,
    frame_stride: int = 1,
    window_step: int = 8,
    episode_splits_by_shard: Mapping[Path, Mapping[str, str]] | None = None,
    include_splits: Sequence[str] = ("train",),
) -> int:
    return sum(
        count_tar_frame_sequences(
            shard,
            context_frames=context_frames,
            prediction_frames=prediction_frames,
            frame_stride=frame_stride,
            window_step=window_step,
            episode_splits=(
                episode_splits_by_shard[Path(shard).resolve()]
                if episode_splits_by_shard is not None
                else None
            ),
            include_splits=include_splits,
        )
        for shard in shards
    )


class TarSequenceDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Worker-balanced episode-safe sequence stream over authoritative shards."""

    def __init__(
        self,
        shards: Sequence[str | Path],
        *,
        context_frames: int = 8,
        prediction_frames: int = 8,
        frame_stride: int = 1,
        window_step: int = 8,
        seed: int = 17,
        shuffle_shards: bool = True,
        max_sequences: int | None = None,
        episode_splits_by_shard: Mapping[Path, Mapping[str, str]] | None = None,
        include_splits: Sequence[str] = ("train",),
    ) -> None:
        super().__init__()
        if not shards:
            raise ValueError("At least one shard is required")
        self.shards = tuple(Path(path) for path in shards)
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.frame_stride = frame_stride
        self.window_step = window_step
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.max_sequences = max_sequences
        self.episode_splits_by_shard = (
            {Path(path).resolve(): dict(splits) for path, splits in episode_splits_by_shard.items()}
            if episode_splits_by_shard is not None
            else None
        )
        self.include_splits = tuple(value.strip().lower() for value in include_splits)
        if self.episode_splits_by_shard is not None:
            missing = {path.resolve() for path in self.shards} - set(self.episode_splits_by_shard)
            if missing:
                raise ValueError(f"Split manifest is missing shards: {sorted(map(str, missing))[:8]}")
            if not self.include_splits:
                raise ValueError("At least one sequence split must be selected")

    def _worker_shards(self) -> list[Path]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        worker_count = 1 if info is None else info.num_workers
        shards = list(self.shards)
        random.Random(self.seed).shuffle(shards)
        allocations: list[list[Path]] = [[] for _ in range(worker_count)]
        allocation_bytes = [0] * worker_count
        for shard in sorted(shards, key=lambda path: path.stat().st_size, reverse=True):
            destination = min(range(worker_count), key=allocation_bytes.__getitem__)
            allocations[destination].append(shard)
            allocation_bytes[destination] += shard.stat().st_size
        worker_shards = allocations[worker_id]
        if self.shuffle_shards:
            random.Random(self.seed + worker_id).shuffle(worker_shards)
        return worker_shards

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yielded = 0
        for shard in self._worker_shards():
            for sequence in iter_tar_frame_sequences(
                shard,
                context_frames=self.context_frames,
                prediction_frames=self.prediction_frames,
                frame_stride=self.frame_stride,
                window_step=self.window_step,
                episode_splits=(
                    self.episode_splits_by_shard[shard.resolve()]
                    if self.episode_splits_by_shard is not None
                    else None
                ),
                include_splits=self.include_splits,
            ):
                if self.max_sequences is not None and yielded >= self.max_sequences:
                    return
                yielded += 1
                yield sequence


class TarImageDataset(IterableDataset[torch.Tensor]):
    """Worker-partitioned, sequential tar reader with bounded local shuffling.

    Sequential reads are intentional: the source is a remote network volume and
    random image reads inside multi-gigabyte tar files are prohibitively costly.
    """

    def __init__(
        self,
        shards: Sequence[str | Path],
        *,
        seed: int = 17,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 256,
        max_images: int | None = None,
        skip_images: int = 0,
        episode_splits_by_shard: Mapping[Path, Mapping[str, str]] | None = None,
        include_splits: Sequence[str] = ("train",),
    ) -> None:
        super().__init__()
        if not shards:
            raise ValueError("At least one shard is required")
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be positive")
        if skip_images < 0:
            raise ValueError("skip_images must be nonnegative")
        self.shards = tuple(Path(path) for path in shards)
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.max_images = max_images
        self.skip_images = skip_images
        self.episode_splits_by_shard = (
            {Path(path).resolve(): dict(splits) for path, splits in episode_splits_by_shard.items()}
            if episode_splits_by_shard is not None
            else None
        )
        self.include_splits = tuple(value.strip().lower() for value in include_splits)
        if self.episode_splits_by_shard is not None:
            missing = {path.resolve() for path in self.shards} - set(self.episode_splits_by_shard)
            if missing:
                raise ValueError(f"Split manifest is missing shards: {sorted(map(str, missing))[:8]}")
            if not self.include_splits:
                raise ValueError("At least one image split must be selected")

    def _worker_shards(self) -> tuple[list[Path], int]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        worker_count = 1 if info is None else info.num_workers
        shards = list(self.shards)
        rng = random.Random(self.seed)
        rng.shuffle(shards)  # randomize equal-size tie handling

        # Greedy size balancing avoids a long tail when shards range from a few
        # hundred MiB to nearly 10 GiB. Each worker still reads whole shards.
        allocations: list[list[Path]] = [[] for _ in range(worker_count)]
        allocation_bytes = [0] * worker_count
        for shard in sorted(shards, key=lambda path: path.stat().st_size, reverse=True):
            destination = min(range(worker_count), key=allocation_bytes.__getitem__)
            allocations[destination].append(shard)
            allocation_bytes[destination] += shard.stat().st_size
        worker_shards = allocations[worker_id]
        if self.shuffle_shards:
            rng.shuffle(worker_shards)
        return worker_shards, worker_id

    def __iter__(self) -> Iterator[torch.Tensor]:
        shards, worker_id = self._worker_shards()
        rng = random.Random(self.seed + 1_000_003 * worker_id)
        yielded = 0
        skipped = 0
        buffer: list[torch.Tensor] = []

        def accepted(image: torch.Tensor) -> torch.Tensor | None:
            nonlocal skipped, yielded
            if skipped < self.skip_images:
                skipped += 1
                return None
            if self.max_images is not None and yielded >= self.max_images:
                return None
            yielded += 1
            return image

        for shard in shards:
            for image in iter_tar_images(
                shard,
                episode_splits=(
                    self.episode_splits_by_shard[shard.resolve()]
                    if self.episode_splits_by_shard is not None
                    else None
                ),
                include_splits=self.include_splits,
            ):
                if self.max_images is not None and yielded >= self.max_images:
                    return
                if self.shuffle_buffer == 1:
                    selected = accepted(image)
                    if selected is not None:
                        yield selected
                    continue
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(image)
                    continue
                index = rng.randrange(len(buffer))
                outgoing, buffer[index] = buffer[index], image
                selected = accepted(outgoing)
                if selected is not None:
                    yield selected

        rng.shuffle(buffer)
        for image in buffer:
            if self.max_images is not None and yielded >= self.max_images:
                return
            selected = accepted(image)
            if selected is not None:
                yield selected
