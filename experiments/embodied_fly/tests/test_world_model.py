import numpy as np
import pytest
import torch

from embodied_fly.world_data import (
    HISTORY,
    HORIZON,
    quaternion_matrix,
    split_for_episode,
    window_arrays,
)
from embodied_fly.world_model import FlyWorldModel, constant_velocity, sigreg


def test_action_state_alignment_and_cold_history_do_not_wrap_or_cross_episodes():
    features = np.arange(160, dtype=np.float32)[:, None]
    metrics = np.repeat(features, 30, axis=1)
    actions = features + 1000
    batch = window_arrays(features, metrics, actions, [0, 30])
    np.testing.assert_array_equal(batch["sequence"][0, : HISTORY - 1, -1], 0)
    np.testing.assert_array_equal(batch["sequence"][0, :HISTORY, 0], 0)
    assert batch["sequence"][1, HISTORY - 1, 0] == 30
    assert batch["actions"][1, 0, 0] == 1030
    assert batch["target"][1, 0, 0] == 31
    assert batch["target"][1, -1, 0] == 130
    with pytest.raises(ValueError):
        window_arrays(features, metrics, actions, [60])
    assert [split_for_episode(i) for i in (0, 1, 8, 9)] == [
        "train",
        "validation",
        "test",
        "test",
    ]


def test_future_actions_and_target_states_cannot_change_earlier_predictions():
    torch.manual_seed(713)
    model = FlyWorldModel(8, 6, 16)
    sequence = torch.randn(4, HISTORY + HORIZON, 8)
    actions = torch.randn(4, HORIZON, 6)
    original = model.encode(sequence)
    modified = sequence.clone()
    modified[:, HISTORY:] += 50
    torch.testing.assert_close(original[:, :HISTORY], model.encode(modified)[:, :HISTORY])
    history = sequence[:, :HISTORY]
    prediction = model.latent_rollout(history, actions)
    changed = actions.clone()
    changed[:, 50:] += 10
    actual = model.latent_rollout(history, changed)
    torch.testing.assert_close(prediction[:, :50], actual[:, :50])
    assert not torch.allclose(prediction[:, 50:], actual[:, 50:])


def test_regularizer_detects_collapsed_latents_and_has_finite_gradients():
    torch.manual_seed(717)
    varied = torch.randn(256, 3, 8, requires_grad=True)
    directions = torch.randn(8, 32)
    value = sigreg(varied, directions)
    assert value < sigreg(torch.zeros_like(varied), directions) / 5
    value.backward()
    assert torch.isfinite(varied.grad).all()
    assert varied.grad.abs().sum() > 0


def test_metric_prediction_has_action_gradients_and_coordinate_translation_equivariance():
    torch.manual_seed(719)
    model = FlyWorldModel(8, 6, 16)
    history = torch.randn(3, HISTORY, 8)
    actions = torch.randn(3, HORIZON, 6, requires_grad=True)
    initial = torch.randn(3, 30)
    original = model.metric_rollout(history, actions, initial)
    shifted = initial.clone()
    shift = torch.tensor([7.0, -4.0, 0.0])
    shifted[:, :3] += shift
    result = model.metric_rollout(history, actions, shifted)
    torch.testing.assert_close(result[..., :3], original[..., :3] + shift)
    torch.testing.assert_close(result[..., 3:], original[..., 3:])
    original[..., 3:6].square().mean().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0


def test_constant_velocity_and_quaternion_conventions():
    initial = np.zeros((2, 30), np.float32)
    initial[:, 3:6] = [1, 2, 3]
    prediction = constant_velocity(initial, HORIZON)
    np.testing.assert_allclose(prediction[0, -1, :3], [0.2, 0.4, 0.6])
    q = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [2**-0.5, 0, 0, 2**-0.5]])
    r = quaternion_matrix(q).reshape(-1, 3, 3)
    np.testing.assert_allclose(r[0], np.eye(3))
    np.testing.assert_allclose(r[0], r[1])
    np.testing.assert_allclose(r[2] @ [1, 0, 0], [0, 1, 0], atol=1e-15)


def test_intervention_changes_only_named_wings_and_effect_gate_checks_magnitude():
    from embodied_fly.world_evaluate import action_variants, effect_errors

    actions = np.full((100, 78), 0.1, np.float32)
    wings = np.arange(14, 20)
    variants, names, clipped = action_variants(actions, wings)
    np.testing.assert_array_equal(variants[0], actions)
    np.testing.assert_array_equal(variants[15], actions)
    nonwing = np.r_[0:14, 20:78]
    np.testing.assert_array_equal(
        variants[..., nonwing], np.repeat(actions[None, :, nonwing], 16, axis=0)
    )
    assert len(names) == 16 and clipped == 0
    np.testing.assert_allclose(variants[1, :, 14], 0.103)
    np.testing.assert_allclose(variants[13, :, 14], 0.103)
    actual = np.array([1.0, -2.0, 0.01])
    assert effect_errors(actual, actual, 0.2)["passed"]
    assert not effect_errors(-actual, actual, 0.2)["passed"]
    assert not effect_errors(2 * actual, actual, 0.2)["passed"]


def test_analytical_wing_integrator_matches_decoupled_mujoco_hinges():
    from embodied_fly.velocity_demonstrations import environment, initialize_worlds
    from embodied_fly.wing_position import normalize_targets
    from embodied_fly.world_analytic import AnalyticalFly, wing_step

    env = environment(3, 3)
    base = initialize_worlds(env, np.zeros(3))
    analytic = AnalyticalFly(env.model)
    wings = analytic.indices["wing_action"]
    q = env.fields["qpos"][:, analytic.indices["wing_qpos"]].copy()
    v = env.fields["qvel"][:, analytic.indices["wing_qvel"]].copy()
    action = np.repeat(base[None], 3, axis=0)
    target = q + np.array([[0.01], [0.1], [-0.1]])
    action[:, wings] = normalize_targets(env.model, target)
    for _ in range(10):
        for _ in range(2):
            q, v = wing_step(q, v, action[:, wings], analytic.parameters, 0.001)
        env.step(action)
    np.testing.assert_allclose(
        q, env.fields["qpos"][:, analytic.indices["wing_qpos"]], atol=1e-10
    )
    np.testing.assert_allclose(
        v, env.fields["qvel"][:, analytic.indices["wing_qvel"]], atol=1e-9
    )
