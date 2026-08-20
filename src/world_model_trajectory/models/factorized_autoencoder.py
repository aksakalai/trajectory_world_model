from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .comparison_autoencoders import (
    LayeredOutput,
    ResidualAutoencoder,
    fixed_crosshair_stencil,
)


FACTORIZED_CONTRACT_VERSION = "independent-encoders-binary-crosshair-v3"


@dataclass
class FactorizedLatents:
    """Independent states consumed by the frozen-AE world model."""

    scene: torch.Tensor
    trajectory: torch.Tensor
    # Shape Bx1 (or BxTx1 for sequences): Ready/green=0, Cooldown/red=1.
    crosshair_state: torch.Tensor

    def detach(self) -> "FactorizedLatents":
        return FactorizedLatents(
            self.scene.detach(),
            self.trajectory.detach(),
            self.crosshair_state.detach(),
        )


@dataclass
class FactorizedOutput(LayeredOutput):
    latent: FactorizedLatents
    trajectory_logits: torch.Tensor
    crosshair_state: torch.Tensor


def crosshair_pixels(images: torch.Tensor) -> torch.Tensor:
    """Return only the fixed 17-pixel RGB crosshair stencil as Bx51."""
    if images.ndim != 4 or images.shape[1:] != (3, 384, 384):
        raise ValueError(f"Expected Bx3x384x384 images, found {tuple(images.shape)}")
    stencil = fixed_crosshair_stencil(device=images.device, dtype=torch.bool)[0, 0]
    return images[:, :, stencil].reshape(images.shape[0], -1)


def deterministic_crosshair_state(images: torch.Tensor) -> torch.Tensor:
    """Extract Ready=0/Cooldown=1 from the canonical fixed-color stencil."""
    pixels = crosshair_pixels(images).float().view(images.shape[0], 3, 17)
    mean_color = pixels.mean(dim=2)
    palette = mean_color.new_tensor(((48, 242, 72), (242, 48, 48))).div(255)
    distances = (mean_color[:, None] - palette[None]).abs().mean(dim=2)
    return distances.argmin(dim=1, keepdim=True).to(dtype=images.dtype)


def compose_factorized_layers(
    scene: torch.Tensor,
    trajectory_mask: torch.Tensor,
    crosshair_state: torch.Tensor,
) -> torch.Tensor:
    """Compose an exact known renderer from a binary or soft state scalar."""
    trajectory_color = scene.new_tensor((40, 255, 65)).div(255).view(1, 3, 1, 1)
    composed = scene * (1 - trajectory_mask) + trajectory_color * trajectory_mask
    ready = scene.new_tensor((48, 242, 72)).div(255).view(1, 3, 1, 1)
    cooldown = scene.new_tensor((242, 48, 48)).div(255).view(1, 3, 1, 1)
    state = crosshair_state.to(dtype=scene.dtype).view(-1, 1, 1, 1).clamp(0, 1)
    color = ready * (1 - state) + cooldown * state
    stencil = fixed_crosshair_stencil(device=scene.device, dtype=scene.dtype)
    return composed * (1 - stencil) + color * stencil


class FactorizedResidualAutoencoder(nn.Module):
    """Renderer-aware AE with independent RGB encoders and isolated decoders.

    Both spatial encoders see RGB, never renderer targets. Crosshair state is
    read deterministically from only the 17 known pixels and has no learned AE
    parameters. Supervision, not a target-derived encoder input, teaches the
    trajectory branch which RGB structure represents the trajectory.
    """

    contract_version = FACTORIZED_CONTRACT_VERSION

    def __init__(
        self,
        *,
        scene_base_channels: int = 24,
        trajectory_base_channels: int = 12,
        scene_latent_channels: int = 6,
        trajectory_latent_channels: int = 2,
        downsample_stages: int = 2,
        blocks_per_stage: int = 1,
        scene_max_channels: int = 192,
        trajectory_max_channels: int = 96,
    ) -> None:
        super().__init__()
        scene_backbone = ResidualAutoencoder(
            base_channels=scene_base_channels,
            latent_channels=scene_latent_channels,
            downsample_stages=downsample_stages,
            blocks_per_stage=blocks_per_stage,
            max_channels=scene_max_channels,
        )
        trajectory_backbone = ResidualAutoencoder(
            base_channels=trajectory_base_channels,
            latent_channels=trajectory_latent_channels,
            downsample_stages=downsample_stages,
            blocks_per_stage=blocks_per_stage,
            max_channels=trajectory_max_channels,
        )

        self.scene_encoder = scene_backbone.encoder
        self.scene_from_latent = scene_backbone.from_latent
        self.scene_decoder = scene_backbone.decoder
        self.trajectory_encoder = trajectory_backbone.encoder
        self.trajectory_from_latent = trajectory_backbone.from_latent

        # Reuse the residual decoder structure but expose one raw mask logit.
        trajectory_layers = list(trajectory_backbone.decoder.children())
        final = trajectory_layers[-2]
        if not isinstance(final, nn.Conv2d) or not isinstance(
            trajectory_layers[-1], nn.Sigmoid
        ):
            raise TypeError("Unexpected residual decoder output structure")
        trajectory_layers[-2] = nn.Conv2d(
            final.in_channels, 1, kernel_size=3, padding=1
        )
        trajectory_layers.pop()
        self.trajectory_decoder = nn.Sequential(*trajectory_layers)

        spatial = 384 // 2**downsample_stages
        self.scene_latent_shape = (scene_latent_channels, spatial, spatial)
        self.trajectory_latent_shape = (
            trajectory_latent_channels,
            spatial,
            spatial,
        )
        # Compatibility summary: the spatial world-model state has eight
        # channels total, matching the successful single-latent AE budget.
        self.latent_shape = (
            scene_latent_channels + trajectory_latent_channels,
            spatial,
            spatial,
        )

    @property
    def architecture_signature(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "encoder_input": "rgb_only",
            "independent_encoders": True,
            "scene_latent_shape": list(self.scene_latent_shape),
            "trajectory_latent_shape": list(self.trajectory_latent_shape),
            "total_spatial_latent_channels": self.latent_shape[0],
            "trajectory_decoder_output": "logits",
            "crosshair_state_dimensions": 1,
            "crosshair_input": "fixed_17_pixel_rgb_stencil",
            "crosshair_encoding": "deterministic_nearest_canonical_color",
            "crosshair_rendering": "deterministic_fixed_stencil_and_palette",
            "crosshair_states": ["Ready", "Cooldown"],
            "composition_order": ["scene", "trajectory", "crosshair"],
        }

    def encode(self, images: torch.Tensor) -> FactorizedLatents:
        if images.ndim != 4 or images.shape[1:] != (3, 384, 384):
            raise ValueError(f"Expected Bx3x384x384 images, found {tuple(images.shape)}")
        return FactorizedLatents(
            scene=self.scene_encoder(images),
            trajectory=self.trajectory_encoder(images),
            crosshair_state=deterministic_crosshair_state(images),
        )

    def _validate_latents(self, latent: FactorizedLatents) -> None:
        batch = latent.scene.shape[0]
        if latent.scene.shape != (batch, *self.scene_latent_shape):
            raise ValueError("Scene latent does not match the configured contract")
        if latent.trajectory.shape != (batch, *self.trajectory_latent_shape):
            raise ValueError("Trajectory latent does not match the configured contract")
        if latent.crosshair_state.shape != (batch, 1):
            raise ValueError("Expected one crosshair state per image")

    def decode_latent(self, latent: FactorizedLatents) -> FactorizedOutput:
        self._validate_latents(latent)
        scene = self.scene_decoder(self.scene_from_latent(latent.scene))
        trajectory_logits = self.trajectory_decoder(
            self.trajectory_from_latent(latent.trajectory)
        )
        trajectory_mask = trajectory_logits.float().sigmoid().to(scene.dtype)
        reconstruction = compose_factorized_layers(
            scene, trajectory_mask, latent.crosshair_state
        )
        # Layered training/evaluation APIs accept logits. These fixed logits
        # are a diagnostic view only; no AE parameter learns the crosshair.
        crosshair_logits = (latent.crosshair_state.float() * 2 - 1) * 16
        return FactorizedOutput(
            scene=scene,
            trajectory_mask=trajectory_mask,
            crosshair_logits=crosshair_logits,
            reconstruction=reconstruction,
            latent=latent,
            trajectory_logits=trajectory_logits,
            crosshair_state=latent.crosshair_state,
        )

    def forward(self, images: torch.Tensor) -> FactorizedOutput:
        return self.decode_latent(self.encode(images))
