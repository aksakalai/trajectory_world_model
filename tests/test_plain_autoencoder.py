import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.models.plain_autoencoder import PlainAutoencoder
from world_model_trajectory.training.losses import reconstruction_loss


def test_shape_and_backward() -> None:
    model = PlainAutoencoder(base_channels=8)
    images = torch.rand(2, 3, 384, 384)
    latents = model.encode(images)
    reconstruction = model.decode(latents)
    loss, pixel, edge = reconstruction_loss(reconstruction, images, edge_weight=0.05)
    assert latents.shape == (2, 8, 24, 24)
    assert reconstruction.shape == images.shape
    assert pixel.item() > 0
    assert edge.item() > 0
    loss.backward()
