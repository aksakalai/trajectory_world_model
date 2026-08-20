from __future__ import annotations

import torch
import torch.nn.functional as F


# These RGB values are stamped directly into the captured image by the current
# simulator. Keep them here (rather than inferring a broad colour range) so
# ordinary scene pixels are never accidentally up-weighted.
TRAJECTORY_RGB = (40, 255, 65)
CROSSHAIR_RGBS = ((48, 242, 72), (242, 48, 48))
# Exhaustive source inventory: 2,817,290 Ready and 564,258 Cooldown frames in
# the corrected combined corpus (4.993x imbalance). The correction is capped at
# 3x: full inverse-frequency weighting overcorrected on the balanced calibration
# probe, while 3x prevents the layered head from minimizing loss as Ready-only.
CROSSHAIR_CLASS_WEIGHTS = (1.0, 3.0)


def exact_rgb_mask(
    images: torch.Tensor, colors: tuple[tuple[int, int, int], ...]
) -> torch.Tensor:
    """Return pixels that match one renderer-stamped RGB colour exactly.

    Inputs are normalized [0, 1] RGB tensors. WebP storage is lossless in this
    dataset, but the half-quantum tolerance also makes the check robust to the
    float conversion.
    """
    palette = images.new_tensor(colors).div_(255.0)
    difference = (images.unsqueeze(1) - palette[None, :, :, None, None]).abs()
    return (difference <= (0.5 / 255.0)).all(dim=2).any(dim=1)


def overlay_l1_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean RGB L1 only at labelled overlay pixels, zero when none exist."""
    expanded_mask = mask.unsqueeze(1).expand_as(target)
    selected = (reconstruction - target).abs()[expanded_mask]
    return selected.mean() if selected.numel() else reconstruction.new_zeros(())


def edge_l1_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between horizontal and vertical first-order image gradients."""
    recon_dx = reconstruction[..., :, 1:] - reconstruction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = reconstruction[..., 1:, :] - reconstruction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy))


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel = F.l1_loss(reconstruction, target)
    edge = edge_l1_loss(reconstruction, target)
    return pixel + edge_weight * edge, pixel, edge


def overlay_reconstruction_losses(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trajectory and crosshair RGB losses for the deterministic HUD overlay."""
    trajectory = overlay_l1_loss(
        reconstruction, target, exact_rgb_mask(target, (TRAJECTORY_RGB,))
    )
    crosshair = overlay_l1_loss(
        reconstruction, target, exact_rgb_mask(target, CROSSHAIR_RGBS)
    )
    return trajectory, crosshair


def fixed_crosshair_mask(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-2:] != (384, 384):
        raise ValueError("Expected BxCx384x384 images")
    mask = torch.zeros(
        images.shape[0], 384, 384, dtype=torch.bool, device=images.device
    )
    mask[:, 192, 188:197] = True
    mask[:, 188:197, 192] = True
    return mask


def crosshair_class_targets(target: torch.Tensor) -> torch.Tensor:
    """Return canonical class indices: Ready=0, Cooldown=1."""
    center = target[:, :, 192, 192]
    palette = target.new_tensor(CROSSHAIR_RGBS).div(255)
    difference = (center[:, None, :] - palette[None, :, :]).abs()
    matches = (difference <= (0.5 / 255.0)).all(dim=2)
    if not matches.any(dim=1).all() or not (matches.sum(dim=1) == 1).all():
        raise ValueError("Target contains a non-canonical crosshair center pixel")
    return matches.to(torch.int64).argmax(dim=1)


def soft_trajectory_score(images: torch.Tensor) -> torch.Tensor:
    """Differentiable saturated-green score including antialiased trajectory edges."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW images")
    trajectory = images.new_tensor(TRAJECTORY_RGB).div(255).view(1, 3, 1, 1)
    color_distance = (images - trajectory).abs().mean(dim=1)
    similarity = torch.exp(-color_distance / 0.18)
    red, green, blue = images.unbind(dim=1)
    green_excess = green - torch.maximum(red, blue)
    dominance = torch.sigmoid((green_excess - 0.12) / 0.04)
    score = similarity * dominance
    if images.shape[-2:] == (384, 384):
        score = score.masked_fill(fixed_crosshair_mask(images), 0)
    return score.clamp(0, 1)


def trajectory_target_mask(target: torch.Tensor) -> torch.Tensor:
    """Soft target combining exact renderer cores and antialiased green edges."""
    soft = soft_trajectory_score(target)
    exact = exact_rgb_mask(target, (TRAJECTORY_RGB,)).to(dtype=target.dtype)
    if target.shape[-2:] == (384, 384):
        exact = exact.masked_fill(fixed_crosshair_mask(target), 0)
    return torch.maximum(soft, exact)


def soft_tversky_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    dimensions = tuple(range(1, prediction.ndim))
    true_positive = (prediction * target).sum(dim=dimensions)
    false_positive = (prediction * (1 - target)).sum(dim=dimensions)
    false_negative = ((1 - prediction) * target).sum(dim=dimensions)
    score = (true_positive + epsilon) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + epsilon
    )
    return (1 - score).mean()


def trajectory_overlap_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Probability-space BCE is intentionally used because both standard and
    # layered paths provide bounded soft masks. PyTorch disallows BCE under
    # CUDA autocast, so keep this small reduction in float32 while preserving
    # gradients to the mixed-precision producer.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_float = prediction.float().clamp(1e-6, 1 - 1e-6)
        target_float = target.float()
        bce = F.binary_cross_entropy(prediction_float, target_float)
        return bce + soft_tversky_loss(
            prediction_float,
            target_float,
            false_positive_weight=0.4,
            false_negative_weight=0.6,
        )


def trajectory_logits_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BF16-safe trajectory supervision that cannot lose saturated gradients."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_float = logits.float()
        target_float = target.float()
        bce = F.binary_cross_entropy_with_logits(logits_float, target_float)
        probability = logits_float.sigmoid()
        return bce + soft_tversky_loss(
            probability,
            target_float,
            false_positive_weight=0.4,
            false_negative_weight=0.6,
        )


def masked_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    expanded = valid_mask.unsqueeze(1).expand_as(target)
    selected = (prediction - target).abs()[expanded]
    return selected.mean() if selected.numel() else prediction.new_zeros(())


def renderer_targets(target: torch.Tensor) -> dict[str, torch.Tensor]:
    trajectory = trajectory_target_mask(target)
    crosshair = fixed_crosshair_mask(target)
    return {
        "trajectory": trajectory,
        "crosshair": crosshair,
        "crosshair_class": crosshair_class_targets(target),
        "scene_valid": ~(crosshair | (trajectory > 0.05)),
    }

def standard_renderer_losses(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    targets = renderer_targets(target)
    trajectory_rgb = overlay_l1_loss(
        reconstruction, target, targets["trajectory"] > 0.05
    )
    crosshair_rgb = overlay_l1_loss(reconstruction, target, targets["crosshair"])
    predicted_trajectory = soft_trajectory_score(reconstruction)
    overlap = trajectory_overlap_loss(predicted_trajectory, targets["trajectory"])
    return {
        "trajectory_rgb": trajectory_rgb,
        "crosshair_rgb": crosshair_rgb,
        "trajectory_overlap": overlap,
    }


def layered_renderer_losses(
    *,
    scene: torch.Tensor,
    trajectory_mask: torch.Tensor,
    crosshair_logits: torch.Tensor,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    trajectory_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    targets = renderer_targets(target)
    scene_loss = masked_l1_loss(scene, target, targets["scene_valid"])
    trajectory_loss = (
        trajectory_logits_loss(
            trajectory_logits.squeeze(1), targets["trajectory"]
        )
        if trajectory_logits is not None
        else trajectory_overlap_loss(
            trajectory_mask.squeeze(1), targets["trajectory"]
        )
    )
    if crosshair_logits.shape == (target.shape[0], 1):
        crosshair_loss = F.binary_cross_entropy_with_logits(
            crosshair_logits.squeeze(1).float(),
            targets["crosshair_class"].to(dtype=torch.float32),
            pos_weight=crosshair_logits.new_tensor(CROSSHAIR_CLASS_WEIGHTS[1]),
        )
    elif crosshair_logits.shape == (target.shape[0], 2):
        crosshair_loss = F.cross_entropy(
            crosshair_logits,
            targets["crosshair_class"],
            weight=crosshair_logits.new_tensor(CROSSHAIR_CLASS_WEIGHTS),
        )
    else:
        raise ValueError("Crosshair logits must have shape Bx1 or Bx2")
    final_l1 = F.l1_loss(reconstruction, target)
    final_edge = edge_l1_loss(reconstruction, target)
    final_overlay = standard_renderer_losses(reconstruction, target)
    scene_trajectory_suppression = soft_trajectory_score(scene).mean()
    return {
        "scene_l1": scene_loss,
        "scene_trajectory_suppression": scene_trajectory_suppression,
        "trajectory": trajectory_loss,
        "crosshair": crosshair_loss,
        "final_l1": final_l1,
        "final_edge": final_edge,
        "final_trajectory_rgb": final_overlay["trajectory_rgb"],
        "final_crosshair_rgb": final_overlay["crosshair_rgb"],
        "final_trajectory_overlap": final_overlay["trajectory_overlap"],
    }
