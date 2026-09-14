import numpy as np
import pytest
import torch

from embodied_fly.velocity_demonstrations import environment, initialize_worlds
from embodied_fly.world_analytic import AnalyticalFly
from embodied_fly.world_data import physical_metrics
from embodied_fly.world_residual_physics import DifferentiableFly


@pytest.fixture(scope="module")
def physical_case():
    env = environment(2, 2)
    base = initialize_worlds(env, np.array([0.0, 0.3]))
    analytic = AnalyticalFly(env.model)
    qpos = env.fields["qpos"].copy()
    qvel = env.fields["qvel"].copy()
    qvel[:, :6] = [[0.3, -0.2, 0.1, 0.1, -0.2, 0.3], [-0.4, 0.5, -0.2, -0.2, 0.1, -0.3]]
    initial = physical_metrics(qpos, qvel, analytic.indices).astype(np.float64)
    actions = np.tile(base, (2, 100, 1))
    t = np.arange(100) * 0.002
    for i in (0, 3):
        actions[:, :, analytic.indices["wing_action"][i]] += 0.12 * np.sin(2 * np.pi * 30 * t)
    return env.model, qpos, initial, actions


def test_zero_residual_matches_full_numpy_analytical_rollout(physical_case):
    model, qpos, initial, actions = physical_case
    plant = DifferentiableFly(model).double()
    x, u = torch.tensor(initial), torch.tensor(actions, dtype=torch.float64)
    inertia, inverse = plant.inertia(qpos, x)
    wings = plant.wing_rollout(x, u)
    actual, lift = plant(
        x, wings, inertia, inverse, torch.zeros(2, 100, 6, dtype=torch.float64)
    )
    expected, expected_lift = AnalyticalFly(model).predict(
        initial, actions.astype(np.float64), qpos
    )
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(lift.detach(), expected_lift, atol=1e-10)


def test_integrated_residual_and_action_gradients_match_finite_differences(physical_case):
    model, qpos, initial, actions = physical_case
    plant = DifferentiableFly(model).double()
    x = torch.tensor(initial[:1])
    # A full wingbeat window exercises wing forcing, drag and body rotation.
    u = torch.tensor(actions[:1, :25], dtype=torch.float64, requires_grad=True)
    residual = torch.zeros(1, 25, 6, dtype=torch.float64, requires_grad=True)
    inertia, inverse = plant.inertia(qpos[:1], x)

    def objective(commands, correction):
        path, _ = plant(x, plant.wing_rollout(x, commands), inertia, inverse, correction)
        return path[0, -1, 3:6].sum() + path[0, -1, 15:18].sum() * 0.1

    value = objective(u, residual)
    value.backward()
    assert torch.isfinite(residual.grad).all() and torch.isfinite(u.grad).all()
    for kind, tensor, idx in (
        ("action", u, (0, 10, int(plant.wing_ids[0]))),
        ("linear", residual, (0, 10, 0)),
        ("angular", residual, (0, 10, 3)),
    ):
        plus, minus = tensor.detach().clone(), tensor.detach().clone()
        epsilon = 1e-5
        plus[idx] += epsilon
        minus[idx] -= epsilon
        with torch.no_grad():
            if kind == "action":
                finite = (objective(plus, residual) - objective(minus, residual)) / (
                    2 * epsilon
                )
            else:
                finite = (objective(u, plus) - objective(u, minus)) / (2 * epsilon)
        assert abs(float(tensor.grad[idx])) > 1e-8
        torch.testing.assert_close(tensor.grad[idx], finite, atol=1e-7, rtol=2e-4)


def test_residual_readout_is_zero_initialized_and_has_no_direct_state_input():
    from embodied_fly.world_residual_model import ResidualWorldModel

    model = ResidualWorldModel()
    z = torch.randn(2, 100, 64)
    assert model.prober[0].in_features == 64
    torch.testing.assert_close(model.accelerations(z), torch.zeros(2, 100, 6))
    with pytest.raises(RuntimeError):
        model.metric_rollout(None)
