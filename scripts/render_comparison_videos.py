from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from world_model_trajectory.models.comparison_autoencoders import (  # noqa: E402
    LayeredOutput,
    VQOutput,
    create_comparison_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render trajectory-rich target/reconstruction comparison videos"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def is_frame(member: tarfile.TarInfo) -> bool:
    return (
        member.isfile()
        and member.name.startswith("episodes/")
        and member.name.endswith(".webp")
        and "/frame-" in member.name
    )


def episode_name(member_name: str) -> str:
    return member_name.split("/", 2)[1]


def select_clips(
    sources: list[dict[str, object]], count: int, frame_count: int
) -> list[tuple[Path, str, list[tarfile.TarInfo], int]]:
    by_shard: dict[Path, list[dict[str, object]]] = defaultdict(list)
    for source in sources:
        by_shard[Path(str(source["shard"]))].append(source)

    selected: list[tuple[Path, str, list[tarfile.TarInfo], int]] = []
    seen_episodes: set[tuple[Path, str]] = set()
    for shard, shard_sources in by_shard.items():
        with tarfile.open(shard, "r:*") as archive:
            members = [member for member in archive.getmembers() if is_frame(member)]
        for source in shard_sources:
            stream_index = int(source["stream_frame_index"])
            if stream_index >= len(members):
                raise IndexError(f"Probe index {stream_index} exceeds {shard}")
            center_member = members[stream_index]
            episode = episode_name(center_member.name)
            key = (shard, episode)
            if key in seen_episodes:
                continue
            seen_episodes.add(key)
            episode_members = [
                member for member in members if episode_name(member.name) == episode
            ]
            center_in_episode = next(
                index
                for index, member in enumerate(episode_members)
                if member.name == center_member.name
            )
            start = max(0, center_in_episode - frame_count // 2)
            start = min(start, max(0, len(episode_members) - frame_count))
            selected.append(
                (
                    shard,
                    episode,
                    episode_members[start : start + frame_count],
                    center_in_episode - start,
                )
            )
            if len(selected) == count:
                return selected
    raise RuntimeError(f"Found only {len(selected)} distinct trajectory-rich episodes")


def decode_members(shard: Path, members: list[tarfile.TarInfo]) -> torch.Tensor:
    frames: list[torch.Tensor] = []
    with tarfile.open(shard, "r:*") as archive:
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name} from {shard}")
            with Image.open(io.BytesIO(extracted.read())) as image:
                frames.append(pil_to_tensor(image.convert("RGB")))
    return torch.stack(frames)


@torch.inference_mode()
def reconstruct(
    model: torch.nn.Module, frames_uint8: torch.Tensor, batch_size: int
) -> torch.Tensor:
    reconstructed: list[torch.Tensor] = []
    for start in range(0, len(frames_uint8), batch_size):
        inputs = (
            frames_uint8[start : start + batch_size]
            .to(device="cuda", dtype=torch.float32)
            .div_(255)
            .contiguous(memory_format=torch.channels_last)
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(inputs)
        if isinstance(output, (LayeredOutput, VQOutput)):
            output = output.reconstruction
        reconstructed.append(
            output.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        )
    return torch.cat(reconstructed)


def labeled_pair(target: torch.Tensor, reconstruction: torch.Tensor) -> bytes:
    target_image = Image.fromarray(target.permute(1, 2, 0).numpy())
    reconstruction_image = Image.fromarray(reconstruction.permute(1, 2, 0).numpy())
    canvas = Image.new("RGB", (768, 416), "black")
    canvas.paste(target_image, (0, 32))
    canvas.paste(reconstruction_image, (384, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "TARGET", fill=(255, 255, 255))
    draw.text((396, 9), "LAYERED RESIDUAL CEILING", fill=(255, 255, 255))
    draw.line((384, 0, 384, 416), fill=(255, 255, 255), width=1)
    return canvas.tobytes()


def write_video(
    destination: Path,
    targets: torch.Tensor,
    reconstructions: torch.Tensor,
    fps: int,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "768x416",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    for target, reconstruction in zip(targets, reconstructions, strict=True):
        process.stdin.write(labeled_pair(target, reconstruction))
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {destination}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = torch.load(args.probe, map_location="cpu", weights_only=False)
    clips = select_clips(probe["sources"], args.clips, args.frames)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_comparison_model(args.model)
    model.load_state_dict(payload["model"])
    model = model.cuda().eval().to(memory_format=torch.channels_last)
    print(f"MODEL model={args.model} latent_shape={model.latent_shape}", flush=True)

    for index, (shard, episode, members, center) in enumerate(clips, start=1):
        print(
            f"CLIP_START index={index}/{len(clips)} episode={episode} "
            f"frames={len(members)} trajectory_center={center}",
            flush=True,
        )
        targets = decode_members(shard, members)
        reconstructions = reconstruct(model, targets, args.batch_size)
        destination = args.output_dir / f"best-model-trajectory-{index:02d}-{episode}.mp4"
        write_video(destination, targets, reconstructions, args.fps)
        print(f"CLIP_COMPLETE path={destination} bytes={destination.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
