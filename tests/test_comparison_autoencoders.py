import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.training.losses import (
    CROSSHAIR_CLASS_WEIGHTS,
    CROSSHAIR_RGBS,
    TRAJECTORY_RGB,
    exact_rgb_mask,
    crosshair_class_targets,
    fixed_crosshair_mask,
    layered_renderer_losses,
    overlay_reconstruction_losses,
    renderer_targets,
    standard_renderer_losses,
    trajectory_target_mask,
)


def test_exact_overlay_masks_and_losses() -> None:
    target = torch.zeros(1, 3, 4, 4)
    target[0, :, 1, 1] = torch.tensor(TRAJECTORY_RGB).float() / 255.0
    target[0, :, 2, 2] = torch.tensor(CROSSHAIR_RGBS[1]).float() / 255.0
    trajectory_mask = exact_rgb_mask(target, (TRAJECTORY_RGB,))
    crosshair_mask = exact_rgb_mask(target, CROSSHAIR_RGBS)
    assert trajectory_mask.sum().item() == 1
    assert crosshair_mask.sum().item() == 1
    trajectory_loss, crosshair_loss = overlay_reconstruction_losses(torch.zeros_like(target), target)
    assert trajectory_loss > 0
    assert crosshair_loss > 0

from world_model_trajectory.models.comparison_autoencoders import (
    LayeredOutput,
    VQOutput,
    compose_renderer_layers,
    create_comparison_model,
)


@pytest.mark.parametrize(
    ("name", "latent_shape"),
    (
        ("plain_large", (4, 48, 48)),
        ("residual_medium", (8, 24, 24)),
        ("residual_large", (4, 48, 48)),
        ("residual_vq_large", (48, 48)),
        ("residual_ceiling", (8, 96, 96)),
        ("standard_residual_large_overlay_loss", (4, 48, 48)),
        ("standard_residual_ceiling_overlay_loss", (8, 96, 96)),
        ("layered_residual_large", (4, 48, 48)),
        ("layered_residual_ceiling", (8, 96, 96)),
    ),
)
def test_candidate_output_shape(name: str, latent_shape: tuple[int, ...]) -> None:
    model = create_comparison_model(name)
    model.eval()
    images = torch.rand(1, 3, 384, 384)
    with torch.no_grad():
        output = model(images)
    reconstruction = output.reconstruction if isinstance(output, (VQOutput, LayeredOutput)) else output
    assert reconstruction.shape == images.shape
    assert tuple(model.latent_shape) == latent_shape
    if isinstance(output, VQOutput):
        assert output.auxiliary_loss.item() >= 0


def canonical_target(*, cooldown: bool = False, trajectory: bool = False) -> torch.Tensor:
    target = torch.zeros(1, 3, 384, 384)
    target[:] = torch.tensor((20, 30, 40)).view(1, 3, 1, 1) / 255
    if trajectory:
        target[0, :, 192, 180:205] = torch.tensor(TRAJECTORY_RGB).float().view(3, 1) / 255
    color = CROSSHAIR_RGBS[1 if cooldown else 0]
    mask = fixed_crosshair_mask(target)[0]
    target[0, :, mask] = torch.tensor(color).float().view(3, 1) / 255
    return target


def test_fixed_compositor_order_absence_and_crosshair_precedence() -> None:
    scene = torch.full((1, 3, 384, 384), 0.25)
    absent = torch.zeros(1, 1, 384, 384)
    ready_logits = torch.tensor([[20.0, -20.0]])
    no_trajectory = compose_renderer_layers(scene, absent, ready_logits)
    assert torch.allclose(no_trajectory[0, :, 100, 100], scene[0, :, 100, 100])
    ready = torch.tensor(CROSSHAIR_RGBS[0]).float() / 255
    assert torch.allclose(no_trajectory[0, :, 192, 192], ready)

    present = torch.zeros_like(absent)
    present[:, :, 192, 180:205] = 1
    composed = compose_renderer_layers(scene, present, ready_logits)
    trajectory = torch.tensor(TRAJECTORY_RGB).float() / 255
    assert torch.allclose(composed[0, :, 192, 180], trajectory)
    assert torch.allclose(composed[0, :, 192, 192], ready)


def test_renderer_targets_use_canonical_crosshair_and_separate_trajectory() -> None:
    ready = canonical_target(trajectory=True)
    cooldown = canonical_target(cooldown=True)
    assert crosshair_class_targets(ready).tolist() == [0]
    assert crosshair_class_targets(cooldown).tolist() == [1]
    assert fixed_crosshair_mask(ready).sum().item() == 17
    trajectory = trajectory_target_mask(ready)
    assert trajectory[0, 192, 180] > 0.99
    assert trajectory[0, 192, 192] == 0
    targets = renderer_targets(ready)
    assert not targets["scene_valid"][0, 192, 180]
    assert not targets["scene_valid"][0, 192, 192]


def test_crosshair_class_weight_matches_corrected_corpus_inventory() -> None:
    ready = 2_817_290
    cooldown = 564_258
    assert CROSSHAIR_CLASS_WEIGHTS == (1.0, 3.0)
    assert ready / cooldown == pytest.approx(5.0, rel=0.002)


def test_standard_and_layered_losses_are_finite_and_branch_gradients_flow() -> None:
    target = canonical_target(trajectory=True)
    standard = target.clone().requires_grad_(True)
    standard_losses = standard_renderer_losses(standard, target)
    assert all(torch.isfinite(value) for value in standard_losses.values())
    sum(standard_losses.values()).backward()
    assert standard.grad is not None

    scene = target.clone().requires_grad_(True)
    trajectory_logits = torch.zeros(1, 1, 384, 384, requires_grad=True)
    trajectory_mask = trajectory_logits.sigmoid()
    crosshair_logits = torch.zeros(1, 2, requires_grad=True)
    reconstruction = compose_renderer_layers(scene, trajectory_mask, crosshair_logits)
    losses = layered_renderer_losses(
        scene=scene,
        trajectory_mask=trajectory_mask,
        crosshair_logits=crosshair_logits,
        reconstruction=reconstruction,
        target=target,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert trajectory_logits.grad is not None and trajectory_logits.grad.abs().sum() > 0
    assert crosshair_logits.grad is not None and crosshair_logits.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_renderer_losses_support_cuda_autocast_backward() -> None:
    target = canonical_target(trajectory=True).cuda()
    prediction = target.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = standard_renderer_losses(prediction, target)
        total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_each_layered_branch_and_final_compositor_reach_shared_encoder() -> None:
    model = create_comparison_model("layered_residual_large")
    images = canonical_target(trajectory=True)
    output = model(images)
    assert isinstance(output, LayeredOutput)
    objectives = (
        output.scene.mean(),
        output.trajectory_mask.mean(),
        output.crosshair_logits[:, 0].mean(),
        output.reconstruction.mean(),
    )
    encoder_parameters = tuple(model.encoder.parameters())
    for index, objective in enumerate(objectives):
        model.zero_grad(set_to_none=True)
        objective.backward(retain_graph=index < len(objectives) - 1)
        encoder_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in encoder_parameters
            if parameter.grad is not None
        )
        assert encoder_gradient > 0


@pytest.mark.parametrize("name", ("layered_residual_large", "layered_residual_ceiling"))
def test_layered_crosshair_head_uses_central_latent_signal(name: str) -> None:
    model = create_comparison_model(name)
    observed = {}

    def capture_features(_module, inputs, _output):
        observed["features"] = inputs[0].detach()

    handle = model.crosshair_classifier.register_forward_hook(capture_features)
    with torch.no_grad():
        output = model(canonical_target())
    handle.remove()
    assert isinstance(output, LayeredOutput)
    assert observed["features"].shape == (1, model.latent_shape[0])
    center_y = output.latent.shape[-2] // 2
    center_x = output.latent.shape[-1] // 2
    expected = output.latent[
        :, :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2
    ].mean(dim=(2, 3))
    assert torch.equal(observed["features"], expected)
