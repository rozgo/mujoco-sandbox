import numpy as np
import pytest

from embodied_fly.body import FlyEnvironment
from embodied_fly.motion_flight import initialize
from embodied_fly.physical_contract import physical_contract
from embodied_fly.pid_hover import HoverPID, PIDConfig


def test_pid_changes_only_wing_commands_and_keeps_a_bounded_integral():
    env = FlyEnvironment("wing_position", wing_response="instant", physics_hz=1000)
    base = initialize(env)
    pid = HoverPID(env, base, env.data.qpos[:3])
    fields = {
        k: getattr(env.data, k).copy() for k in ("qpos", "qvel", "act", "ctrl", "xfrc_applied")
    }
    initial = pid.act()
    pid.target[2] += 0.1
    for _ in range(1000):
        changed = pid.act()
    assert not np.array_equal(changed[pid.channels], initial[pid.channels])
    assert np.max(np.abs(pid.integral)) <= pid.config.integral_limit_cm_s
    assert np.max(np.abs(changed)) <= 1
    other = np.setdiff1d(np.arange(env.model.nu), pid.channels)
    np.testing.assert_array_equal(changed[other], base[other])
    for k, expected in fields.items():
        np.testing.assert_array_equal(getattr(env.data, k), expected)
    assert env.control_dt == 0.002 and env.substeps == 2


def test_pid_reference_has_no_hidden_body_support_when_wing_force_is_cut(monkeypatch):
    env = FlyEnvironment("wing_position", wing_response="instant", physics_hz=1000)
    base = initialize(env)
    pid = HoverPID(env, base, env.data.qpos[:3])
    monkeypatch.setattr(env.wing_forces, "advance", lambda *args: np.zeros((1, 6)))
    for _ in range(30):
        env.step(pid.act())
    assert env.data.qpos[2] < 0.5  # Free-fall from 2 cm despite active wing controller.
    assert not env.data.xfrc_applied.any() and not env.data.qfrc_applied.any()
    assert abs(env.data.qvel[env.wing_velocity_indices]).max() > 1


def test_one_khz_pid_cold_start_sustains_without_body_pose_or_force_overrides():
    env = FlyEnvironment("wing_position", wing_response="instant", physics_hz=1000)
    base = initialize(env, 1.86651647)
    target = env.data.qpos[:3].copy()
    controller = HoverPID(
        env, base, target, PIDConfig(height_kp=900, height_ki=1600, height_kd=50)
    )
    contract = physical_contract(env.model)
    positions = []
    for _ in range(1500):
        env.step(controller.act())
        positions.append(env.data.qpos[:3].copy())
        assert abs(env.data.actuator_force[controller.channels]).max() <= 0.030000001
        assert env.data.xmat[env.thorax_id, 8] > 0.99
    settled = np.asarray(positions)[500:]
    assert np.ptp(settled[:, 2]) * 10 < 0.2
    assert np.linalg.norm(settled - target, axis=1).max() * 10 < 0.5
    assert env.maximum_disallowed_ground_force == 0
    assert not env.data.warning.number.any()
    assert physical_contract(env.model) == contract


@pytest.mark.parametrize("hz", (0, -1000, 750, float("nan")))
def test_invalid_physics_clock_is_rejected(hz):
    with pytest.raises(ValueError, match="integer multiple"):
        FlyEnvironment("wing_position", wing_response="instant", physics_hz=hz)
