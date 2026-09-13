import numpy as np
import pytest

from embodied_fly.body import FlyEnvironment


def test_fast_wing_speed_is_distinguishable_without_changing_old_observations():
    env = FlyEnvironment("flight")
    values = np.array([1100, 1500, 2000, -1100, -1500, -2000])
    env.data.qvel[env.wing_velocity_indices] = values
    legacy = env.observation()
    extended = env.observation(extended_wing_velocity=True)
    np.testing.assert_array_equal(extended[:383], legacy)
    np.testing.assert_allclose(extended[383:], values / 2000)
    assert np.max(np.abs(extended[383:])) <= 1


def test_wing_angles_are_continuous_and_preserve_all_existing_channels():
    env = FlyEnvironment("flight")
    angles = np.array([-1.0, 0.2, 2.0, 1.1, -0.3, 0.5])
    env.data.qpos[env.wing_angle_indices] = angles
    previous = env.observation(True)
    extended = env.observation(True, wing_angles=True)
    assert extended.shape == (395,)
    np.testing.assert_array_equal(extended[:389], previous)
    np.testing.assert_allclose(extended[389:], angles / np.pi)
    env.data.qpos[env.wing_angle_indices] += 0.01
    changed = env.observation(True, wing_angles=True)
    np.testing.assert_allclose(changed[389:] - extended[389:], 0.01 / np.pi, rtol=1e-5)
    with pytest.raises(ValueError, match="requires"):
        env.observation(wing_angles=True)
