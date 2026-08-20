from __future__ import annotations

import torch

from world_model_trajectory.training.losses import (
    CROSSHAIR_RGBS,
    crosshair_class_targets,
    fixed_crosshair_mask,
)


def trajectory_mask(images: torch.Tensor) -> torch.Tensor:
    """Detect the simulator's saturated green trajectory pixels.

    This is deliberately an evaluation heuristic, not a training target. It is
    calibrated for the current renderer and must be audited if presentation
    colors change.
    """
    red, green, blue = images.unbind(dim=1)
    mask = (green > 0.55) & ((green - red) > 0.20) & ((green - blue) > 0.10)
    if images.shape[-2:] == (384, 384):
        mask = mask & ~fixed_crosshair_mask(images)
    return mask


def trajectory_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    target_mask = trajectory_mask(target)
    reconstruction_mask = trajectory_mask(reconstruction)
    true_positive = (target_mask & reconstruction_mask).sum().float()
    false_positive = ((~target_mask) & reconstruction_mask).sum().float()
    false_negative = (target_mask & (~reconstruction_mask)).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1.0)

    mask_rgb = target_mask.unsqueeze(1).expand_as(target)
    color_values = (reconstruction - target).abs()[mask_rgb]
    color_error = color_values.mean() if color_values.numel() else reconstruction.new_zeros(())

    center = target.shape[-1] // 2
    radius = 12
    target_center = target[..., center - radius : center + radius + 1, center - radius : center + radius + 1]
    reconstruction_center = reconstruction[
        ..., center - radius : center + radius + 1, center - radius : center + radius + 1
    ]
    center_l1 = (reconstruction_center - target_center).abs().mean()

    return {
        "trajectory_precision": precision,
        "trajectory_recall": recall,
        "trajectory_f1": f1,
        "trajectory_iou": iou,
        "trajectory_color_l1": color_error,
        "center_l1": center_l1,
        "trajectory_target_pixels": target_mask.sum().float(),
        "trajectory_reconstructed_pixels": reconstruction_mask.sum().float(),
    }


def crosshair_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Report centre-HUD reconstruction on the exact stamped crosshair pixels."""
    target_mask = fixed_crosshair_mask(target)
    expanded_mask = target_mask.unsqueeze(1).expand_as(target)
    values = (reconstruction - target).abs()[expanded_mask]
    color_l1 = values.mean() if values.numel() else reconstruction.new_zeros(())
    target_class = crosshair_class_targets(target)
    palette = reconstruction.new_tensor(CROSSHAIR_RGBS).div(255)
    reconstructed_center = reconstruction[:, :, 192, 192]
    distance = (reconstructed_center[:, None, :] - palette[None, :, :]).abs().mean(dim=2)
    predicted_class = distance.argmin(dim=1)
    state_accuracy = (predicted_class == target_class).float().mean()
    target_pixels = target.permute(0, 2, 3, 1)[target_mask]
    reconstructed_pixels = reconstruction.permute(0, 2, 3, 1)[target_mask]
    exact_color_fraction = (
        (reconstructed_pixels - target_pixels).abs().le(0.5 / 255).all(dim=1).float().mean()
    )
    return {
        "crosshair_color_l1": color_l1,
        "crosshair_state_accuracy": state_accuracy,
        "crosshair_exact_color_fraction": exact_color_fraction,
        "crosshair_target_pixels": target_mask.sum().float(),
    }
