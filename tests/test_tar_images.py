import io
import sys
import tarfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.tar_images import iter_tar_images


def test_iter_tar_images_ignores_metadata(tmp_path: Path) -> None:
    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_bytes, format="WEBP", lossless=True)
    shard = tmp_path / "shard.tar"
    with tarfile.open(shard, "w") as archive:
        payload = image_bytes.getvalue()
        frame = tarfile.TarInfo("episodes/e-1/frame-000000.webp")
        frame.size = len(payload)
        archive.addfile(frame, io.BytesIO(payload))
        metadata = b"{}"
        manifest = tarfile.TarInfo("manifest.json")
        manifest.size = len(metadata)
        archive.addfile(manifest, io.BytesIO(metadata))

    images = list(iter_tar_images(shard))
    assert len(images) == 1
    assert images[0].shape == (3, 16, 16)
    assert images[0].min() >= 0
    assert images[0].max() <= 1


def test_episode_split_filter_and_exact_skip_budget(tmp_path: Path) -> None:
    shard = tmp_path / "shard.tar"
    with tarfile.open(shard, "w") as archive:
        for episode, red in (("train-episode", 10), ("validation-episode", 20)):
            for frame_index in range(3):
                stream = io.BytesIO()
                Image.new("RGB", (16, 16), (red + frame_index, 0, 0)).save(
                    stream, format="WEBP", lossless=True
                )
                payload = stream.getvalue()
                member = tarfile.TarInfo(
                    f"episodes/{episode}/frame-{frame_index:06d}.webp"
                )
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    mapping = {"train-episode": "train", "validation-episode": "validation"}
    validation = list(
        iter_tar_images(
            shard, episode_splits=mapping, include_splits=("validation",)
        )
    )
    assert len(validation) == 3
    assert [round(float(image[0, 0, 0]) * 255) for image in validation] == [20, 21, 22]

    from world_model_trajectory.data.tar_images import TarImageDataset

    resumed = list(
        TarImageDataset(
            [shard], shuffle_buffer=1, shuffle_shards=False, skip_images=1,
            max_images=1, episode_splits_by_shard={shard.resolve(): mapping},
            include_splits=("train",),
        )
    )
    assert len(resumed) == 1
    assert round(float(resumed[0][0, 0, 0]) * 255) == 11
