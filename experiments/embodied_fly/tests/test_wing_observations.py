import numpy as np

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
