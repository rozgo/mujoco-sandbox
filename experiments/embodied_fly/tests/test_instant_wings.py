import mujoco
import numpy as np
import pytest
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.instant_migrate import transfer
from embodied_fly.physical_contract import physical_contract
from embodied_fly.wing_motion import CONFIG, INSTANT_CONFIG, WingMotionForces, config_for_model


@pytest.fixture(scope="module", params=(5000, 1000))
def worlds(request):
    return FlyBatch(
        3, 3, 14, preset="wing_position", wing_response="instant", physics_hz=request.param
    )


def test_current_wing_motion_sets_force_immediately_without_history_or_dt_dependence(worlds):
    a, b = WingMotionForces(worlds.model), WingMotionForces(worlds.model)
    angles = np.array([[0, 0.7, -1, 0, 0.7, -1]])
    rotation, body_speed = np.eye(3)[None], np.zeros((1, 6))
    speed = np.array([[50, 0, 0, 50, 0, 0]])
    a.activity[:] = 0
    b.activity[:] = 100  # Prior state cannot influence the instantaneous force.
    x = a.advance(angles, speed, rotation, body_speed, 0.0002).copy()
    y = b.advance(angles, speed, rotation, body_speed, 0.002).copy()
    np.testing.assert_array_equal(x, y)
    assert x[0, 2] == pytest.approx(1.6 * a.weight)
    np.testing.assert_array_equal(
        a.advance(angles, speed * 0, rotation, body_speed, 0.0002), 0
    )
    # Differential response and wing pitch both use this tick, with no ramp.
    speed[0, 3] = 0
    unpitched = a.advance(angles, speed, rotation, body_speed, 0.0002).copy()
    angles[0, 2] += np.pi
    pitched = a.advance(angles, speed, rotation, body_speed, 0.0002).copy()
    assert pitched[0, 2] == pytest.approx(unpitched[0, 2] * 0.6)
    assert np.linalg.norm(unpitched[0, 3:]) > 0


def test_physics_marker_survives_native_batch_and_binary_reload(worlds, tmp_path):
    assert physical_contract(worlds.model) == physical_contract(worlds.template.model)
    assert config_for_model(worlds.model) == INSTANT_CONFIG
    path = tmp_path / "instant.mjb"
    mujoco.mj_saveModel(worlds.model, str(path))
    loaded = mujoco.MjModel.from_binary_path(str(path))
    assert config_for_model(loaded) == INSTANT_CONFIG
    assert physical_contract(loaded) == physical_contract(worlds.model)
    legacy = FlyBatch(
        1, 1, 14, preset="wing_position", physics_hz=1 / worlds.model.opt.timestep
    )
    assert config_for_model(legacy.model) == CONFIG
    assert physical_contract(legacy.model) != physical_contract(worlds.model)
    parent = {
        "physical_contract": physical_contract(legacy.model),
        "motor_only": True,
        "config": {"internal_steps": 4, "preset": "wing_position"},
        "state_dict": {"weight": torch.randn(3)},
        "critic_state_dict": {"weight": torch.randn(3)},
        "optimizer_state_dict": {"exp_avg": torch.randn(3)},
        "value_optimizer_state_dict": {"step": torch.tensor(7)},
        "log_std": torch.randn(6),
    }
    child, _ = transfer(parent, legacy.model, worlds.model, "test-parent")
    for key in (
        "state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "value_optimizer_state_dict",
        "log_std",
    ):
        assert child[key] is parent[key]
    assert parent["config"].get("wing_response") is None
    with pytest.raises(ValueError, match="recorded filtered"):
        transfer(child, legacy.model, worlds.model, "wrong-contract")


def test_every_physics_tick_reads_evolving_physical_wings(worlds, monkeypatch):
    worlds.reset(np.arange(3))
    old = worlds.wing_forces.advance
    calls = []

    def trace(angles, velocities, rotation, body_velocity, dt):
        np.testing.assert_array_equal(
            velocities, worlds.fields["qvel"][:, worlds.template.wing_velocity_indices]
        )
        calls.append((angles.copy(), velocities.copy(), dt))
        return old(angles, velocities, rotation, body_velocity, dt)

    monkeypatch.setattr(worlds.wing_forces, "advance", trace)
    action = np.repeat(worlds.template.passive_action[None], 3, axis=0)
    action[:, 14:20] = 0.1
    worlds.step(action)
    assert len(calls) == worlds.substeps == round(0.002 / worlds.model.opt.timestep)
    assert all(dt == worlds.model.opt.timestep for _, _, dt in calls)
    assert not np.array_equal(calls[0][1], calls[-1][1])
    expected = np.clip(np.abs(calls[-1][1].reshape(3, 2, 3)[:, :, 0]) / 50, 0, 1.4)
    np.testing.assert_array_equal(worlds.wing_forces.activity, expected)
