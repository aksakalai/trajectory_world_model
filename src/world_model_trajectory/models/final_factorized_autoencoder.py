"""Clean final factorized autoencoder implementation.

This module intentionally does not depend on the interrupted factorized-v3 or
legacy layered-autoencoder implementations.  The only inputs to both learned
encoders are RGB images.  Crosshair state is read and rendered by a fixed,
validated inverse of the known renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


CONTRACT_VERSION = "final-independent-rgb-encoders-deterministic-crosshair-v2"
IMAGE_SHAPE = (3, 384, 384)
READY_RGB = (48, 242, 72)
COOLDOWN_RGB = (242, 48, 48)
TRAJECTORY_RGB = (40, 255, 65)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def fixed_crosshair_stencil(
    *, device: torch.device | None = None, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Return the renderer's exact 17-pixel stencil as 1x1x384x384."""
    stencil = torch.zeros(1, 1, 384, 384, device=device, dtype=dtype)
    stencil[:, :, 192, 188:197] = 1
    stencil[:, :, 188:197, 192] = 1
    return stencil


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"Expected Bx3x384x384 RGB images, found {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("RGB inputs must be floating point tensors normalized to [0, 1]")
    if not torch.isfinite(images).all():
        raise ValueError("RGB inputs contain NaN or infinity")
    if (images < 0).any() or (images > 1).any():
        raise ValueError("RGB inputs must be normalized to [0, 1]")


def crosshair_pixels(images: torch.Tensor) -> torch.Tensor:
    """Return the 17 RGB stencil pixels as Bx3x17."""
    _validate_images(images)
    stencil = fixed_crosshair_stencil(device=images.device)[0, 0]
    return images[:, :, stencil]


def read_crosshair_state(images: torch.Tensor) -> torch.Tensor:
    """Read exact Ready=0/Cooldown=1 state and reject noncanonical frames."""
    pixels = crosshair_pixels(images).float()
    palette = pixels.new_tensor((READY_RGB, COOLDOWN_RGB)).div(255.0)
    difference = (pixels[:, None] - palette[None, :, :, None]).abs()
    pixel_matches = (difference <= (0.5 / 255.0)).all(dim=2)
    class_matches = pixel_matches.all(dim=2)
    exactly_one = class_matches.sum(dim=1) == 1
    if not exactly_one.all():
        bad = (~exactly_one).nonzero().flatten().tolist()
        raise ValueError(
            "Noncanonical or inconsistent crosshair stencil at batch indices "
            f"{bad[:8]}"
        )
    return class_matches.to(torch.int64).argmax(dim=1, keepdim=True).to(images.dtype)


def compose_layers(
    scene: torch.Tensor,
    trajectory_probability: torch.Tensor,
    crosshair_state: torch.Tensor,
) -> torch.Tensor:
    """Compose scene -> trajectory -> crosshair; supports soft state for later WM use."""
    if scene.ndim != 4 or tuple(scene.shape[1:]) != IMAGE_SHAPE:
        raise ValueError("Scene must have shape Bx3x384x384")
    batch = scene.shape[0]
    if trajectory_probability.shape != (batch, 1, 384, 384):
        raise ValueError("Trajectory probability must have shape Bx1x384x384")
    if crosshair_state.shape != (batch, 1):
        raise ValueError("Crosshair state must have shape Bx1")
    # The learned branches may run under BF16 autocast, but known renderer
    # colors must remain exact normalized FP32 values. This also gives the
    # frozen decoder a stable differentiable interface for the world model.
    with torch.autocast(device_type=scene.device.type, enabled=False):
        scene_float = scene.float()
        trajectory_color = scene_float.new_tensor(TRAJECTORY_RGB).div(255).view(
            1, 3, 1, 1
        )
        trajectory = trajectory_probability.float().clamp(0, 1)
        composed = scene_float * (1 - trajectory) + trajectory_color * trajectory
        ready = scene_float.new_tensor(READY_RGB).div(255).view(1, 3, 1, 1)
        cooldown = scene_float.new_tensor(COOLDOWN_RGB).div(255).view(1, 3, 1, 1)
        state = crosshair_state.float().view(batch, 1, 1, 1).clamp(0, 1)
        crosshair_color = ready * (1 - state) + cooldown * state
        stencil = fixed_crosshair_stencil(device=scene.device, dtype=torch.float32)
        return composed * (1 - stencil) + crosshair_color * stencil


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class DownsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
        self.residual = ResidualBlock(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.residual(self.downsample(inputs))


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.residual = ResidualBlock(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(inputs, scale_factor=2, mode="nearest")
        return self.residual(self.convolution(hidden))


class SpatialAutoencoderBranch(nn.Module):
    """One independent RGB encoder and its isolated decoder."""

    def __init__(
        self,
        *,
        base_channels: int,
        latent_channels: int,
        output_channels: int,
        downsample_stages: int = 2,
    ) -> None:
        super().__init__()
        if downsample_stages not in (2, 3):
            raise ValueError("Only 96x96 and 48x48 spatial bottlenecks are supported")
        self.latent_resolution = 96 if downsample_stages == 2 else 48
        upper_channels = base_channels * 2
        self.encoder = nn.Sequential(
            DownsampleBlock(3, base_channels),
            DownsampleBlock(base_channels, upper_channels),
            nn.GroupNorm(_groups(upper_channels), upper_channels),
            nn.SiLU(),
            nn.Conv2d(upper_channels, latent_channels, 3, padding=1),
        )
        self.from_latent = nn.Conv2d(latent_channels, upper_channels, 3, padding=1)
        self.decoder = nn.Sequential(
            ResidualBlock(upper_channels),
            UpsampleBlock(upper_channels, base_channels),
            UpsampleBlock(base_channels, base_channels),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, 3, padding=1),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(images)
        if self.latent_resolution == 48:
            latent = F.avg_pool2d(latent, kernel_size=2, stride=2)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_resolution == 48:
            latent = F.interpolate(latent, size=(96, 96), mode="nearest")
        return self.decoder(self.from_latent(latent))


@dataclass(frozen=True)
class FactorizedState:
    scene: torch.Tensor
    trajectory: torch.Tensor
    crosshair: torch.Tensor


@dataclass(frozen=True)
class FactorizedReconstruction:
    state: FactorizedState
    scene: torch.Tensor
    trajectory_logits: torch.Tensor
    trajectory_probability: torch.Tensor
    image: torch.Tensor


class FinalFactorizedAutoencoder(nn.Module):
    """Independent scene/trajectory spatial latents plus deterministic HUD state."""

    contract_version = CONTRACT_VERSION
    def __init__(
        self,
        *,
        scene_width: int = 24,
        trajectory_width: int = 12,
        scene_latent_channels: int = 6,
        trajectory_latent_channels: int = 2,
        scene_latent_resolution: int = 96,
        trajectory_latent_resolution: int = 96,
    ) -> None:
        super().__init__()
        if min(scene_width, trajectory_width, scene_latent_channels, trajectory_latent_channels) < 1:
            raise ValueError("Branch widths and latent channels must be positive")
        if scene_latent_resolution not in (48, 96) or trajectory_latent_resolution not in (48, 96):
            raise ValueError("Latent resolutions must be 48 or 96")
        self.scene_width = scene_width
        self.trajectory_width = trajectory_width
        self.scene_latent_channels = scene_latent_channels
        self.trajectory_latent_channels = trajectory_latent_channels
        self.scene_latent_resolution = scene_latent_resolution
        self.trajectory_latent_resolution = trajectory_latent_resolution
        self.scene_latent_shape = (
            scene_latent_channels, scene_latent_resolution, scene_latent_resolution
        )
        self.trajectory_latent_shape = (
            trajectory_latent_channels,
            trajectory_latent_resolution,
            trajectory_latent_resolution,
        )
        self.scene_branch = SpatialAutoencoderBranch(
            base_channels=scene_width,
            latent_channels=scene_latent_channels,
            output_channels=3,
            downsample_stages=2 if scene_latent_resolution == 96 else 3,
        )
        self.trajectory_branch = SpatialAutoencoderBranch(
            base_channels=trajectory_width,
            latent_channels=trajectory_latent_channels,
            output_channels=1,
            downsample_stages=2 if trajectory_latent_resolution == 96 else 3,
        )

    @property
    def architecture_signature(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "learned_encoder_inputs": "rgb_only",
            "independent_encoder_parameters": True,
            "scene_latent_shape": list(self.scene_latent_shape),
            "trajectory_latent_shape": list(self.trajectory_latent_shape),
            "scene_width": self.scene_width,
            "trajectory_width": self.trajectory_width,
            "total_spatial_channels": (
                self.scene_latent_channels + self.trajectory_latent_channels
            ),
            "total_spatial_values": (
                self.scene_latent_channels
                * self.scene_latent_resolution
                * self.scene_latent_resolution
                + self.trajectory_latent_channels
                * self.trajectory_latent_resolution
                * self.trajectory_latent_resolution
            ),
            "crosshair_state": "deterministic_scalar_ready_0_cooldown_1",
            "learned_crosshair_parameters": 0,
            "composition_order": ["scene", "trajectory", "crosshair"],
        }

    def encode(self, images: torch.Tensor) -> FactorizedState:
        _validate_images(images)
        return FactorizedState(
            scene=self.scene_branch.encode(images),
            trajectory=self.trajectory_branch.encode(images),
            crosshair=read_crosshair_state(images),
        )

    def decode(self, state: FactorizedState) -> FactorizedReconstruction:
        batch = state.scene.shape[0]
        if state.scene.shape != (batch, *self.scene_latent_shape):
            raise ValueError(f"Scene latent violates the {self.scene_latent_shape} contract")
        if state.trajectory.shape != (batch, *self.trajectory_latent_shape):
            raise ValueError(
                f"Trajectory latent violates the {self.trajectory_latent_shape} contract"
            )
        if state.crosshair.shape != (batch, 1):
            raise ValueError("Crosshair state violates the scalar contract")
        scene = self.scene_branch.decode(state.scene).sigmoid()
        trajectory_logits = self.trajectory_branch.decode(state.trajectory)
        trajectory_probability = trajectory_logits.float().sigmoid()
        image = compose_layers(scene, trajectory_probability, state.crosshair)
        return FactorizedReconstruction(
            state=state,
            scene=scene,
            trajectory_logits=trajectory_logits,
            trajectory_probability=trajectory_probability,
            image=image,
        )

    def forward(self, images: torch.Tensor) -> FactorizedReconstruction:
        return self.decode(self.encode(images))
