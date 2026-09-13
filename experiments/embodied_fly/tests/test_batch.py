import numpy as np
import pytest

from embodied_fly.batch import FlyBatch


@pytest.mark.parametrize("sensor_extension_size", (0, 6, 12))
def test_native_batch_matches_single_physics_observation_and_partial_reset(
    sensor_extension_size,
):
    env = FlyBatch(2, 2, sensor_extension_size)
    single = env.template
    env.reset([0], yaw=[0.17])
    single.reset(yaw=0.17)
    np.testing.assert_array_equal(
        single.observation(
            sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
        ),
        env.observation()[0],
    )
    rng = np.random.default_rng(7104)
    peak = 0.0
    for _ in range(100):
        action = single.walking_action(rng.normal(0, 0.03, 78))
        single.step(action)
        env.step(np.stack([action, -action]))
        peak = max(peak, env.forbidden_peak[0])
        np.testing.assert_allclose(single.data.qpos, env.fields["qpos"][0], atol=1e-12)
        np.testing.assert_array_equal(
            single.observation(
                sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
            ),
            env.observation()[0],
        )
        np.testing.assert_allclose(single.anatomical_velocity(), env.velocity()[0], atol=1e-10)
    np.testing.assert_allclose(peak, single.maximum_disallowed_ground_force, atol=1e-9)
    other = env.fields["qpos"][1].copy()
    env.reset([0], yaw=[0.17])
    env.reset([0], yaw=[0.17])  # identical repeated requested poses must survive reset
    single.reset(yaw=0.17)
    np.testing.assert_array_equal(
        single.observation(
            sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
        ),
        env.observation()[0],
    )
    np.testing.assert_array_equal(other, env.fields["qpos"][1])
    assert env.model.nv == 108 and env.model.nu == 78
    np.testing.assert_array_equal(
        env.model.actuator_forcerange, single.model.actuator_forcerange
    )
