from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            *[ResidualBlock(out_channels) for _ in range(blocks)],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(blocks)])
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.residual(inputs)
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.convolution(inputs)


class PlainDownsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class PlainUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.normalization = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return F.silu(self.normalization(self.convolution(inputs)), inplace=True)


class ConfigurablePlainAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int,
        latent_channels: int,
        downsample_stages: int,
    ) -> None:
        super().__init__()
        channels = [min(base_channels * 2**index, 256) for index in range(downsample_stages)]
        encoder: list[nn.Module] = []
        previous = 3
        for current in channels:
            encoder.append(PlainDownsampleBlock(previous, current))
            previous = current
        encoder.append(nn.Conv2d(previous, latent_channels, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [
            nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        reversed_channels = list(reversed(channels))
        for index, current in enumerate(reversed_channels):
            output = reversed_channels[index + 1] if index + 1 < len(reversed_channels) else base_channels
            decoder.append(PlainUpsampleBlock(current, output))
        decoder.extend((nn.Conv2d(base_channels, 3, kernel_size=3, padding=1), nn.Sigmoid()))
        self.decoder = nn.Sequential(*decoder)
        spatial = 384 // 2**downsample_stages
        self.latent_shape = (latent_channels, spatial, spatial)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class ResidualAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int,
        latent_channels: int,
        downsample_stages: int,
        blocks_per_stage: int,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        channels = [
            min(base_channels * 2**index, max_channels) for index in range(downsample_stages)
        ]
        encoder: list[nn.Module] = []
        previous = 3
        for current in channels:
            encoder.append(DownsampleBlock(previous, current, blocks_per_stage))
            previous = current
        encoder.extend(
            (
                nn.GroupNorm(_groups(previous), previous),
                nn.SiLU(inplace=True),
                nn.Conv2d(previous, latent_channels, kernel_size=3, padding=1),
            )
        )
        self.encoder = nn.Sequential(*encoder)

        self.from_latent = nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1)
        decoder: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        for index, current in enumerate(reversed_channels):
            output = reversed_channels[index + 1] if index + 1 < len(reversed_channels) else base_channels
            decoder.append(UpsampleBlock(current, output, blocks_per_stage))
        decoder.extend(
            (
                nn.GroupNorm(_groups(base_channels), base_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
        )
        self.decoder = nn.Sequential(*decoder)
        spatial = 384 // 2**downsample_stages
        self.latent_shape = (latent_channels, spatial, spatial)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latents))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


@dataclass
class VQOutput:
    reconstruction: torch.Tensor
    auxiliary_loss: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_weight: float) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, height, width = latents.shape
        flat = latents.permute(0, 2, 3, 1).reshape(-1, channels)
        embeddings = self.codebook.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + embeddings.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ embeddings.t()
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, embeddings).view(batch, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        straight_through = latents + (quantized - latents).detach()

        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            utilization = (counts > 0).float().mean()
        return straight_through, loss, {
            "vq_perplexity": perplexity,
            "vq_utilization": utilization,
            "vq_dead_fraction": 1.0 - utilization,
        }


class ResidualVQVAE(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 32,
        downsample_stages: int = 3,
        blocks_per_stage: int = 2,
        embedding_dim: int = 64,
        codebook_size: int = 2048,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        backbone = ResidualAutoencoder(
            base_channels=base_channels,
            latent_channels=embedding_dim,
            downsample_stages=downsample_stages,
            blocks_per_stage=blocks_per_stage,
        )
        self.encoder = backbone.encoder
        self.from_latent = backbone.from_latent
        self.decoder = backbone.decoder
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim, commitment_weight)
        spatial = 384 // 2**downsample_stages
        self.latent_shape = (spatial, spatial)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> VQOutput:
        encoded = self.encoder(images)
        quantized, auxiliary_loss, metrics = self.quantizer(encoded)
        reconstruction = self.decoder(self.from_latent(quantized))
        return VQOutput(reconstruction, auxiliary_loss, metrics)


@dataclass
class LayeredOutput:
    scene: torch.Tensor
    trajectory_mask: torch.Tensor
    crosshair_logits: torch.Tensor
    reconstruction: torch.Tensor
    latent: torch.Tensor


def fixed_crosshair_stencil(
    *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    mask = torch.zeros(1, 1, 384, 384, device=device, dtype=dtype)
    mask[:, :, 192, 188:197] = 1
    mask[:, :, 188:197, 192] = 1
    return mask


def compose_renderer_layers(
    scene: torch.Tensor,
    trajectory_mask: torch.Tensor,
    crosshair_logits: torch.Tensor,
) -> torch.Tensor:
    """Differentiably compose scene -> trajectory -> fixed crosshair."""
    if scene.ndim != 4 or scene.shape[1:] != (3, 384, 384):
        raise ValueError(f"Expected Bx3x384x384 scene, found {tuple(scene.shape)}")
    if trajectory_mask.shape != (scene.shape[0], 1, 384, 384):
        raise ValueError("Trajectory mask shape does not match scene")
    if crosshair_logits.shape not in ((scene.shape[0], 1), (scene.shape[0], 2)):
        raise ValueError("Crosshair logits must have shape Bx1 or Bx2")
    trajectory_color = scene.new_tensor((40, 255, 65)).div(255).view(1, 3, 1, 1)
    composed = scene * (1 - trajectory_mask) + trajectory_color * trajectory_mask
    # Class order is Ready, Cooldown. Neutral/white does not exist canonically.
    crosshair_colors = scene.new_tensor(((48, 242, 72), (242, 48, 48))).div(255)
    if crosshair_logits.shape[1] == 1:
        cooldown = crosshair_logits.sigmoid()
        probabilities = torch.cat((1 - cooldown, cooldown), dim=1)
    else:
        probabilities = crosshair_logits.softmax(dim=1)
    crosshair_color = (probabilities @ crosshair_colors).view(-1, 3, 1, 1)
    stencil = fixed_crosshair_stencil(device=scene.device, dtype=scene.dtype)
    return composed * (1 - stencil) + crosshair_color * stencil


class LayeredResidualAutoencoder(nn.Module):
    """Shared encoder with scene, trajectory-mask, and crosshair-state branches."""

    def __init__(
        self,
        *,
        base_channels: int,
        latent_channels: int,
        downsample_stages: int,
        blocks_per_stage: int,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        scene_backbone = ResidualAutoencoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
            downsample_stages=downsample_stages,
            blocks_per_stage=blocks_per_stage,
            max_channels=max_channels,
        )
        trajectory_backbone = ResidualAutoencoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
            downsample_stages=downsample_stages,
            blocks_per_stage=blocks_per_stage,
            max_channels=max_channels,
        )
        self.encoder = scene_backbone.encoder
        self.scene_from_latent = scene_backbone.from_latent
        self.scene_decoder = scene_backbone.decoder
        self.trajectory_from_latent = trajectory_backbone.from_latent
        trajectory_layers = list(trajectory_backbone.decoder.children())
        final_convolution = trajectory_layers[-2]
        if not isinstance(final_convolution, nn.Conv2d):
            raise TypeError("Unexpected residual decoder output structure")
        trajectory_layers[-2] = nn.Conv2d(
            final_convolution.in_channels, 1, kernel_size=3, padding=1
        )
        self.trajectory_decoder = nn.Sequential(*trajectory_layers)
        self.crosshair_classifier = nn.Linear(latent_channels, 2)
        spatial = 384 // 2**downsample_stages
        self.latent_shape = (latent_channels, spatial, spatial)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode_latent(self, latent: torch.Tensor) -> LayeredOutput:
        """Decode an existing latent for frozen-encoder world-model evaluation."""
        scene = self.scene_decoder(self.scene_from_latent(latent))
        trajectory_mask = self.trajectory_decoder(self.trajectory_from_latent(latent))
        # The state is rendered only on the fixed 17-pixel center stencil.
        # Global pooling diluted that tiny signal enough for the classifier to
        # settle on the Ready majority class, especially at 96x96. Pool the
        # corresponding central latent neighborhood instead.
        center_y = latent.shape[-2] // 2
        center_x = latent.shape[-1] // 2
        crosshair_features = latent[
            :, :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2
        ].mean(dim=(2, 3))
        crosshair_logits = self.crosshair_classifier(crosshair_features)
        reconstruction = compose_renderer_layers(scene, trajectory_mask, crosshair_logits)
        return LayeredOutput(
            scene=scene,
            trajectory_mask=trajectory_mask,
            crosshair_logits=crosshair_logits,
            reconstruction=reconstruction,
            latent=latent,
        )

    def forward(self, images: torch.Tensor) -> LayeredOutput:
        return self.decode_latent(self.encode(images))


def create_comparison_model(name: str) -> nn.Module:
    if name == "factorized_residual_ceiling":
        # Local import avoids a module cycle: the factorized implementation
        # intentionally reuses the residual backbone and layered output API.
        from .factorized_autoencoder import FactorizedResidualAutoencoder

        return FactorizedResidualAutoencoder(
            scene_base_channels=24,
            trajectory_base_channels=12,
            scene_latent_channels=6,
            trajectory_latent_channels=2,
            downsample_stages=2,
            blocks_per_stage=1,
            scene_max_channels=192,
            trajectory_max_channels=96,
        )
    if name == "plain_large":
        return ConfigurablePlainAutoencoder(
            base_channels=32, latent_channels=4, downsample_stages=3
        )
    if name == "residual_medium":
        return ResidualAutoencoder(
            base_channels=32,
            latent_channels=8,
            downsample_stages=4,
            blocks_per_stage=1,
        )
    if name == "residual_large":
        return ResidualAutoencoder(
            base_channels=32,
            latent_channels=4,
            downsample_stages=3,
            blocks_per_stage=1,
        )
    if name == "residual_vq_large":
        return ResidualVQVAE(blocks_per_stage=1)
    if name == "residual_ceiling":
        return ResidualAutoencoder(
            base_channels=32,
            latent_channels=8,
            downsample_stages=2,
            blocks_per_stage=1,
            max_channels=256,
        )
    if name == "standard_residual_large_overlay_loss":
        return ResidualAutoencoder(
            base_channels=32,
            latent_channels=4,
            downsample_stages=3,
            blocks_per_stage=1,
        )
    if name == "standard_residual_ceiling_overlay_loss":
        return ResidualAutoencoder(
            base_channels=32,
            latent_channels=8,
            downsample_stages=2,
            blocks_per_stage=1,
            max_channels=256,
        )
    if name == "layered_residual_large":
        return LayeredResidualAutoencoder(
            base_channels=32,
            latent_channels=4,
            downsample_stages=3,
            blocks_per_stage=1,
        )
    if name == "layered_residual_ceiling":
        return LayeredResidualAutoencoder(
            base_channels=32,
            latent_channels=8,
            downsample_stages=2,
            blocks_per_stage=1,
            max_channels=256,
        )
    raise ValueError(f"Unknown comparison model: {name}")
