from __future__ import annotations

import torch
from torch import nn


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        ),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class PlainAutoencoder(nn.Module):
    """Deterministic 384x384 autoencoder with a 24x24x8 bottleneck."""

    latent_shape = (8, 24, 24)

    def __init__(self, base_channels: int = 32, latent_channels: int = 8) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            _down_block(3, c),       # 384 -> 192
            _down_block(c, 2 * c),   # 192 -> 96
            _down_block(2 * c, 4 * c),  # 96 -> 48
            _down_block(4 * c, 8 * c),  # 48 -> 24
            nn.Conv2d(8 * c, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 8 * c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            _up_block(8 * c, 4 * c),
            _up_block(4 * c, 2 * c),
            _up_block(2 * c, c),
            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))
