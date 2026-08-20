"""Standalone objectives and diagnostics for the final factorized autoencoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from world_model_trajectory.models.final_factorized_autoencoder import (
    TRAJECTORY_RGB,
    FactorizedReconstruction,
    fixed_crosshair_stencil,
)


DEFAULT_TARGET_SCORE_THRESHOLD = 0.05
DEFAULT_ANTIALIAS_SUPPORT_RADIUS = 2
DEFAULT_POSITIVE_WEIGHT_CAP = 128.0


def exact_trajectory_core(images: torch.Tensor) -> torch.Tensor:
    color = images.new_tensor(TRAJECTORY_RGB).div(255).view(1, 3, 1, 1)
    matches = ((images - color).abs() <= (0.5 / 255.0)).all(dim=1)
    return matches.masked_fill(fixed_crosshair_stencil(device=images.device)[0], False)


def trajectory_target(
    images: torch.Tensor,
    *,
    score_threshold: float = DEFAULT_TARGET_SCORE_THRESHOLD,
    antialias_support_radius: int = DEFAULT_ANTIALIAS_SUPPORT_RADIUS,
) -> torch.Tensor:
    """Extract renderer cores plus only their locally connected antialiased edges.

    The renderer emits an exact (40, 255, 65) core.  Its one-pixel blended
    antialias fringe is admitted only near that core; no whole-frame green-color
    heuristic is used.  The fixed crosshair stencil is excluded because it is
    composited after, and can therefore occlude, trajectory pixels.
    """
    if not 0 < score_threshold < 1:
        raise ValueError("Trajectory score threshold must be between zero and one")
    if antialias_support_radius < 0:
        raise ValueError("Antialias support radius cannot be negative")
    exact = exact_trajectory_core(images)
    color = images.new_tensor(TRAJECTORY_RGB).div(255).view(1, 3, 1, 1)
    similarity = torch.exp(-(images - color).abs().mean(dim=1) / 0.18)
    red, green, blue = images.unbind(dim=1)
    green_excess = green - torch.maximum(red, blue)
    dominance = torch.sigmoid((green_excess - 0.12) / 0.04)
    soft = similarity * dominance
    soft = soft.masked_fill(fixed_crosshair_stencil(device=images.device)[0], 0)
    kernel = 1 + 2 * antialias_support_radius
    local_support = F.max_pool2d(
        exact[:, None].to(torch.float32), kernel_size=kernel, stride=1,
        padding=antialias_support_radius,
    )[:, 0].bool()
    soft = torch.where(
        (soft > score_threshold) & local_support, soft, torch.zeros_like(soft)
    )
    return torch.maximum(soft, exact.to(images.dtype)).clamp(0, 1)


def soft_tversky_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    false_positive_weight: float = 0.4,
    false_negative_weight: float = 0.6,
) -> torch.Tensor:
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1 - target)).sum(dim=dimensions)
    false_negative = ((1 - probability) * target).sum(dim=dimensions)
    score = (true_positive + 1e-6) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + 1e-6
    )
    return (1 - score).mean()


def trajectory_logit_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight_cap: float = DEFAULT_POSITIVE_WEIGHT_CAP,
) -> torch.Tensor:
    """Full-frame BF16-safe loss that penalizes every background false positive."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_float = logits.float()
        target_float = target.float()
        positive_support = (target_float > 0).sum().float()
        negative_support = target_float.numel() - positive_support
        if not math.isfinite(positive_weight_cap) or positive_weight_cap < 1:
            raise ValueError("Positive weight cap must be finite and at least one")
        positive_weight = (
            negative_support / positive_support.clamp_min(1)
        ).clamp(min=1, max=positive_weight_cap)
        bce = F.binary_cross_entropy_with_logits(
            logits_float, target_float, pos_weight=positive_weight
        )
        return bce + soft_tversky_loss(logits_float.sigmoid(), target_float)


def edge_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    dx = F.l1_loss(prediction_dx, target_dx)
    dy = F.l1_loss(prediction_dy, target_dy)
    return 0.5 * (dx + dy)


def _masked_rgb_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    selected = (prediction - target).abs()[mask.unsqueeze(1).expand_as(target)]
    return selected.mean() if selected.numel() else prediction.new_zeros(())


def loss_components(
    output: FactorizedReconstruction,
    target_rgb: torch.Tensor,
    *,
    target_score_threshold: float = DEFAULT_TARGET_SCORE_THRESHOLD,
    antialias_support_radius: int = DEFAULT_ANTIALIAS_SUPPORT_RADIUS,
    positive_weight_cap: float = DEFAULT_POSITIVE_WEIGHT_CAP,
) -> dict[str, torch.Tensor]:
    trajectory = trajectory_target(
        target_rgb,
        score_threshold=target_score_threshold,
        antialias_support_radius=antialias_support_radius,
    )
    crosshair = fixed_crosshair_stencil(device=target_rgb.device).expand(
        target_rgb.shape[0], -1, -1, -1
    )[:, 0]
    scene_valid = ~(crosshair | (trajectory > target_score_threshold))
    return {
        "scene_l1": _masked_rgb_l1(output.scene, target_rgb, scene_valid),
        "trajectory": trajectory_logit_loss(
            output.trajectory_logits[:, 0], trajectory,
            positive_weight_cap=positive_weight_cap,
        ),
        "final_l1": F.l1_loss(output.image, target_rgb),
        "final_edge": edge_l1(output.image, target_rgb),
        "final_trajectory_rgb": _masked_rgb_l1(
            output.image, target_rgb, trajectory > target_score_threshold
        ),
        "final_crosshair_rgb": _masked_rgb_l1(output.image, target_rgb, crosshair),
        "final_trajectory_overlap": soft_tversky_loss(
            output.trajectory_probability[:, 0].float(), trajectory.float()
        ),
    }


@torch.no_grad()
def trajectory_confusion_counts(
    probability: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    target_score_threshold: float = DEFAULT_TARGET_SCORE_THRESHOLD,
    antialias_support_radius: int = DEFAULT_ANTIALIAS_SUPPORT_RADIUS,
) -> dict[str, float]:
    target = trajectory_target(
        target_rgb,
        score_threshold=target_score_threshold,
        antialias_support_radius=antialias_support_radius,
    ) > target_score_threshold
    predicted = probability[:, 0].float() >= 0.5
    return {
        "true_positive": float((predicted & target).sum()),
        "false_positive": float((predicted & ~target).sum()),
        "false_negative": float((~predicted & target).sum()),
        "predicted_positive": float(predicted.sum()),
        "target_positive": float(target.sum()),
        "probability_sum": float(probability.float().sum()),
        "pixel_count": float(predicted.numel()),
    }


@torch.no_grad()
def exact_core_confusion_counts(
    probability: torch.Tensor,
    target_rgb: torch.Tensor,
) -> dict[str, float]:
    """Global strict-core counts; every predicted background pixel is an FP."""
    target = exact_trajectory_core(target_rgb)
    predicted = probability[:, 0].float() >= 0.5
    return {
        "true_positive": float((predicted & target).sum()),
        "false_positive": float((predicted & ~target).sum()),
        "false_negative": float((~predicted & target).sum()),
        "predicted_positive": float(predicted.sum()),
        "target_positive": float(target.sum()),
        "probability_sum": float(probability.float().sum()),
        "pixel_count": float(predicted.numel()),
    }


def trajectory_metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    pixel_count = max(counts["pixel_count"], 1)
    return {
        "predicted_probability_mean": counts["probability_sum"] / pixel_count,
        "predicted_foreground_fraction": counts["predicted_positive"] / pixel_count,
        "target_foreground_fraction": counts["target_positive"] / pixel_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


@torch.no_grad()
def trajectory_diagnostics(
    probability: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    target_score_threshold: float = DEFAULT_TARGET_SCORE_THRESHOLD,
    antialias_support_radius: int = DEFAULT_ANTIALIAS_SUPPORT_RADIUS,
) -> dict[str, float]:
    counts = trajectory_confusion_counts(
        probability,
        target_rgb,
        target_score_threshold=target_score_threshold,
        antialias_support_radius=antialias_support_radius,
    )
    return trajectory_metrics_from_counts(counts)
