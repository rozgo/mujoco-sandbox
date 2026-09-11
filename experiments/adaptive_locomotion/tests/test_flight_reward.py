import numpy as np

from adaptive_locomotion.env import DogEnv


def test_flight_cost_survives_healthy_stride_gating_without_changing_physics():
    envs = [
        DogEnv(
            32,
            seed=2,
            limb_stage="consolidate",
            randomize=False,
            faults=False,
            threads=1,
            stride_weight=1,
            damage_flight_weight=weight,
        )
        for weight in (0, 0.5)
    ]
    saw_flight = saw_support = False
    try:
        for env in envs:
            env.commands[:] = [0.55, 0, 0]
            for g in env.groups:
                g.qpos[:, 2] += 0.04
                g.batch.forward()
            env.refresh()
        for _ in range(100):
            rewards = [env.step(np.zeros((32, 12)))[0] for env in envs]
            base = envs[0]
            damaged = (base.valid < 1).any(1)
            supported = (base.tip_forces > 1).any(1)
            saw_flight |= np.any(damaged & ~supported)
            saw_support |= np.any(damaged & supported)
            np.testing.assert_allclose(
                rewards[0] - rewards[1],
                0.5 * damaged * ~supported,
                atol=2e-6,
            )
            np.testing.assert_array_equal(base.obs(), envs[1].obs())
            for left, right in zip(base.groups, envs[1].groups, strict=True):
                np.testing.assert_array_equal(left.qpos, right.qpos)
                np.testing.assert_array_equal(left.qvel, right.qvel)
        assert saw_flight and saw_support
    finally:
        for env in envs:
            env.close()
