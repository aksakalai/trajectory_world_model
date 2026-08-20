import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.models.final_factorized_autoencoder import (
    COOLDOWN_RGB,
    READY_RGB,
    FactorizedState,
    FinalFactorizedAutoencoder,
    compose_layers,
    fixed_crosshair_stencil,
    read_crosshair_state,
)
from world_model_trajectory.training.final_factorized_objective import (
    loss_components,
    trajectory_confusion_counts,
    exact_core_confusion_counts,
    trajectory_logit_loss,
    trajectory_metrics_from_counts,
    trajectory_target,
)


def canonical_frame(*, trajectory: bool, cooldown: bool) -> torch.Tensor:
    image = torch.full((1, 3, 384, 384), 0.25)
    if trajectory:
        image[:, :, 250, 80:180] = image.new_tensor((40, 255, 65)).div(255)[:, None]
    color = COOLDOWN_RGB if cooldown else READY_RGB
    stencil = fixed_crosshair_stencil()[0, 0]
    image[:, :, stencil] = image.new_tensor(color).div(255)[:, None]
    return image


def tiny_model() -> FinalFactorizedAutoencoder:
    return FinalFactorizedAutoencoder(scene_width=4, trajectory_width=2)


def test_contract_shapes_independent_parameters_and_rgb_only_api() -> None:
    model = tiny_model()
    state = model.encode(canonical_frame(trajectory=True, cooldown=True))
    assert state.scene.shape == (1, 6, 96, 96)
    assert state.trajectory.shape == (1, 2, 96, 96)
    assert state.crosshair.shape == (1, 1)
    assert state.crosshair.item() == 1
    scene_parameters = {id(parameter) for parameter in model.scene_branch.parameters()}
    trajectory_parameters = {
        id(parameter) for parameter in model.trajectory_branch.parameters()
    }
    assert not scene_parameters & trajectory_parameters
    assert tuple(inspect.signature(model.encode).parameters) == ("images",)
    assert model.architecture_signature["total_spatial_channels"] == 8
    assert model.architecture_signature["learned_crosshair_parameters"] == 0
    assert not any("crosshair" in name for name, _ in model.named_parameters())


def test_mixed_resolution_candidate_contract_and_decode_shape() -> None:
    model = FinalFactorizedAutoencoder(
        scene_width=4, trajectory_width=2,
        scene_latent_channels=8, scene_latent_resolution=48,
        trajectory_latent_channels=1, trajectory_latent_resolution=96,
    )
    output = model(canonical_frame(trajectory=True, cooldown=False))
    assert output.state.scene.shape == (1, 8, 48, 48)
    assert output.state.trajectory.shape == (1, 1, 96, 96)
    assert output.image.shape == (1, 3, 384, 384)
    assert model.architecture_signature["total_spatial_values"] == 27648


def test_crosshair_reader_is_exact_local_and_rejects_noncanonical_data() -> None:
    ready = canonical_frame(trajectory=True, cooldown=False)
    cooldown = canonical_frame(trajectory=False, cooldown=True)
    assert read_crosshair_state(ready).item() == 0
    assert read_crosshair_state(cooldown).item() == 1
    changed_elsewhere = 1 - ready
    stencil = fixed_crosshair_stencil(dtype=torch.bool)
    changed_elsewhere = torch.where(stencil, ready, changed_elsewhere)
    torch.testing.assert_close(
        read_crosshair_state(changed_elsewhere), read_crosshair_state(ready)
    )
    corrupt = ready.clone()
    corrupt[:, :, 192, 192] = 0.5
    with pytest.raises(ValueError, match="Noncanonical"):
        read_crosshair_state(corrupt)


def test_crosshair_renderer_changes_exactly_the_fixed_17_pixels() -> None:
    scene = torch.full((1, 3, 384, 384), 0.2)
    trajectory = torch.zeros(1, 1, 384, 384)
    ready = compose_layers(scene, trajectory, torch.zeros(1, 1))
    cooldown = compose_layers(scene, trajectory, torch.ones(1, 1))
    changed = (ready != cooldown).any(dim=1)
    assert int(changed.sum()) == 17
    torch.testing.assert_close(changed, fixed_crosshair_stencil()[0])
    bf16_ready = compose_layers(
        scene.to(torch.bfloat16), trajectory.to(torch.bfloat16), torch.zeros(1, 1)
    )
    assert bf16_ready.dtype == torch.float32
    ready_color = (
        torch.tensor(READY_RGB, dtype=torch.float32).div(255)[:, None].expand(-1, 17)
    )
    torch.testing.assert_close(
        bf16_ready[:, :, fixed_crosshair_stencil()[0, 0]][0], ready_color,
        rtol=0,
        atol=0,
    )


def test_decoder_interventions_are_branch_local() -> None:
    torch.manual_seed(5)
    model = tiny_model().eval()
    first = model.encode(canonical_frame(trajectory=True, cooldown=False))
    second = model.encode(canonical_frame(trajectory=False, cooldown=True))
    baseline = model.decode(first)
    trajectory_swap = model.decode(
        FactorizedState(first.scene, second.trajectory, first.crosshair)
    )
    torch.testing.assert_close(trajectory_swap.scene, baseline.scene)
    assert not torch.allclose(
        trajectory_swap.trajectory_logits, baseline.trajectory_logits
    )
    scene_swap = model.decode(
        FactorizedState(second.scene, first.trajectory, first.crosshair)
    )
    torch.testing.assert_close(
        scene_swap.trajectory_logits, baseline.trajectory_logits
    )
    assert not torch.allclose(scene_swap.scene, baseline.scene)
    crosshair_swap = model.decode(
        FactorizedState(first.scene, first.trajectory, second.crosshair)
    )
    torch.testing.assert_close(crosshair_swap.scene, baseline.scene)
    torch.testing.assert_close(
        crosshair_swap.trajectory_logits, baseline.trajectory_logits
    )


def test_full_frame_trajectory_loss_rejects_all_foreground_loophole() -> None:
    empty = trajectory_target(canonical_frame(trajectory=False, cooldown=False))
    all_background = trajectory_logit_loss(torch.full_like(empty, -10), empty)
    all_foreground = trajectory_logit_loss(torch.full_like(empty, 10), empty)
    assert all_foreground > all_background * 5

    present = trajectory_target(canonical_frame(trajectory=True, cooldown=False))
    matched_logits = torch.where(present > 0.05, 10.0, -10.0)
    matched = trajectory_logit_loss(matched_logits, present)
    missing = trajectory_logit_loss(torch.full_like(present, -10), present)
    assert matched < missing


def test_trajectory_target_rejects_unanchored_green_scene_pixels() -> None:
    image = canonical_frame(trajectory=False, cooldown=False)
    image[:, :, 50:80, 50:80] = image.new_tensor((0, 255, 0)).div(255)[:, None, None]
    assert not (trajectory_target(image)[:, 50:80, 50:80] > 0).any()


def test_exact_core_metric_penalizes_predicting_everything() -> None:
    image = canonical_frame(trajectory=True, cooldown=False)
    probability = torch.ones((1, 1, 384, 384))
    metrics = trajectory_metrics_from_counts(
        exact_core_confusion_counts(probability, image)
    )
    assert metrics["recall"] == 1.0
    assert metrics["precision"] < 0.01
    assert metrics["f1"] < 0.02

    image[:, :, 200, 100] = image.new_tensor((40, 255, 65)).div(255)[None]
    image[:, :, 200, 101] = image.new_tensor((80, 230, 90)).div(255)[None]
    target = trajectory_target(image)
    assert target[:, 200, 100].item() == 1
    assert target[:, 200, 101].item() > 0


def test_saturated_bf16_logits_keep_finite_corrective_gradients() -> None:
    logits = torch.tensor([[[-20.0, 20.0]]], dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0]]])
    trajectory_logit_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 0, 1] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_saturated_logits_and_exact_compositor_under_cuda_bf16() -> None:
    logits = torch.tensor(
        [[[-20.0, 20.0]]], device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.tensor([[[1.0, 0.0]]], device="cuda")
    trajectory_logit_loss(logits, target).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 0, 1] > 0
    scene = torch.zeros(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
    composed = compose_layers(
        scene, torch.zeros(1, 1, 384, 384, device="cuda"),
        torch.zeros(1, 1, device="cuda"),
    )
    expected = (
        torch.tensor(READY_RGB, device="cuda").float().div(255)[:, None].expand(-1, 17)
    )
    torch.testing.assert_close(
        composed[:, :, fixed_crosshair_stencil(device=torch.device("cuda"))[0, 0]][0],
        expected,
        rtol=0,
        atol=0,
    )


def test_global_trajectory_metrics_are_computed_from_global_counts() -> None:
    positive = canonical_frame(trajectory=True, cooldown=False)
    empty = canonical_frame(trajectory=False, cooldown=False)
    positive_prediction = torch.ones(1, 1, 384, 384)
    empty_prediction = torch.zeros(1, 1, 384, 384)
    first = trajectory_confusion_counts(positive_prediction, positive)
    second = trajectory_confusion_counts(empty_prediction, empty)
    combined = {name: first[name] + second[name] for name in first}
    metrics = trajectory_metrics_from_counts(combined)
    direct = trajectory_confusion_counts(
        torch.cat((positive_prediction, empty_prediction)),
        torch.cat((positive, empty)),
    )
    assert metrics == trajectory_metrics_from_counts(direct)


def test_scene_loss_masks_overlays_but_not_ordinary_pixels() -> None:
    target = canonical_frame(trajectory=True, cooldown=False)
    model = tiny_model().eval()
    output = model(target)
    components = loss_components(output, target)
    baseline = components["scene_l1"]
    stencil = fixed_crosshair_stencil(dtype=torch.bool)
    modified_scene = output.scene.clone()
    modified_scene = torch.where(stencil, 1 - modified_scene, modified_scene)
    trajectory = trajectory_target(target) > 0.05
    modified_scene = torch.where(
        trajectory[:, None], 1 - modified_scene, modified_scene
    )
    masked_output = type(output)(
        state=output.state,
        scene=modified_scene,
        trajectory_logits=output.trajectory_logits,
        trajectory_probability=output.trajectory_probability,
        image=output.image,
    )
    torch.testing.assert_close(loss_components(masked_output, target)["scene_l1"], baseline)
    modified_scene[:, :, 10, 10] = 1 - modified_scene[:, :, 10, 10]
    ordinary_output = type(output)(
        state=output.state,
        scene=modified_scene,
        trajectory_logits=output.trajectory_logits,
        trajectory_probability=output.trajectory_probability,
        image=output.image,
    )
    assert loss_components(ordinary_output, target)["scene_l1"] != baseline


def test_joint_objective_reaches_all_four_learned_subpaths() -> None:
    torch.manual_seed(11)
    model = tiny_model().train()
    images = torch.cat(
        (
            canonical_frame(trajectory=True, cooldown=False),
            canonical_frame(trajectory=False, cooldown=True),
        )
    )
    output = model(images)
    sum(loss_components(output, images).values()).backward()
    modules = {
        "scene_encoder": model.scene_branch.encoder,
        "scene_decoder": model.scene_branch.decoder,
        "trajectory_encoder": model.trajectory_branch.encoder,
        "trajectory_decoder": model.trajectory_branch.decoder,
    }
    for name, module in modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients, f"No gradients reached {name}"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_direct_branch_losses_have_no_cross_branch_gradient_path() -> None:
    model = tiny_model().train()
    images = canonical_frame(trajectory=True, cooldown=False)
    output = model(images)
    components = loss_components(output, images)
    components["scene_l1"].backward(retain_graph=True)
    assert any(parameter.grad is not None for parameter in model.scene_branch.parameters())
    assert all(parameter.grad is None for parameter in model.trajectory_branch.parameters())

    model.zero_grad(set_to_none=True)
    components["trajectory"].backward()
    assert all(parameter.grad is None for parameter in model.scene_branch.parameters())
    assert any(
        parameter.grad is not None for parameter in model.trajectory_branch.parameters()
    )
