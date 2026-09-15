import mujoco
import numpy as np
import pytest

from embodied_fly.physical_contract import ARRAYS, physical_contract
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.wing_motion import WingMotionForces, config_for_model


@pytest.fixture(scope="module")
def env():
    return environment(2, 2, reduced_coupling=True)


def test_physical_identity_and_recorded_version(env, tmp_path):
    old = environment(1, 1)
    for key in ARRAYS:
        np.testing.assert_array_equal(getattr(old.model, key), getattr(env.model, key))
    assert physical_contract(old.model) != physical_contract(env.model)
    assert physical_contract(env.model) == physical_contract(env.template.model)
    path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(env.model, str(path))
    assert config_for_model(mujoco.MjModel.from_binary_path(str(path))).version.endswith("v6")


def test_yaw_pitch_does_not_change_lift_and_sweep_does_not_change_thrust(env):
    f = WingMotionForces(env.model, 2)
    angles = np.tile([0, 0.8, -1, 0, 0.9, -1], (2, 1))
    velocities = np.tile([35.0, 0, 0, 35, 0, 0], (2, 1))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    body = np.zeros((2, 6))
    angles[1, [2, 5]] += [-0.2, 0.2]
    w = f.advance(angles, velocities, rotations, body, 0.001).copy()
    np.testing.assert_allclose(w[0, :3], w[1, :3], atol=1e-12)
    assert w[1, 5] > w[0, 5]
    angles[1] = angles[0]
    velocities[1, [0, 3]] = 45
    w = f.advance(angles, velocities, rotations, body, 0.001).copy()
    np.testing.assert_allclose(w[0, :2], w[1, :2], atol=1e-12)
    assert w[1, 2] > w[0, 2]
    velocities[:] = 0
    np.testing.assert_array_equal(f.advance(angles, velocities, rotations, body, 0.001), 0)


def test_native_batch_share_force_law_and_no_wing_body_mass_coupling(env):
    single = env.template
    env.reset(np.arange(2))
    single.reset()
    for step in range(30):
        action = single.passive_action.copy()
        action[14:20] = 0.1 * np.sin(step * 0.35 + np.arange(6))
        single.step(action)
        env.step(np.repeat(action[None], 2, axis=0))
    # Same native/batch numerical tolerance as the existing wing-force tests.
    np.testing.assert_allclose(single.data.qpos, env.fields["qpos"][0], atol=3e-6, rtol=0)
    mass = np.empty((single.model.nv, single.model.nv))
    mujoco.mj_fullM(single.model, single.data, mass)
    other = np.setdiff1d(np.arange(single.model.nv), single.wing_velocity_indices)
    np.testing.assert_array_equal(mass[np.ix_(single.wing_velocity_indices, other)], 0)
