import numpy as np

from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.rear_overlap import overlap_cost, rear_swing_metrics


def test_overlap_is_symmetric_damage_aware_and_allows_stance():
    valid = np.ones((6, 12))
    valid[0, 2] = valid[1, 5] = 0
    valid[2, :3] = 0
    valid[3, 3:6] = 0
    valid[4, 8] = 0  # Rear loss must not receive this objective.
    force = np.zeros((6, 4))
    force[:, 0] = 40
    cmd = np.tile([0.55, 0, 0], (6, 1))
    np.testing.assert_array_equal(overlap_cost(force, valid, cmd), [1, 1, 1, 1, 0, 0])
    force[:, 2] = 40  # One rear foot down, then both rear feet down.
    assert not overlap_cost(force, valid, cmd).any()
    force[:, 3] = 40
    assert not overlap_cost(force, valid, cmd).any()
    assert not overlap_cost(force * 0, valid, cmd * 0).any()


def test_phase_measure_distinguishes_alternation_and_synchrony():
    p = np.zeros((300, 4, 3))
    p[..., 2] = 0.022
    for offset, expected in [(0, 0), (10, 0.5)]:
        p[..., 2] = 0.022
        for leg, lag in [(2, 0), (3, offset)]:
            p[:, leg, 2] += 0.06 * (((np.arange(300) - lag) % 20) < 8)
        d = rear_swing_metrics(p, np.full(4, 0.022))
        assert np.isclose(d["separation_fraction"], expected)
        assert d["phase_concentration"] > 0.99
        assert (d["both_rear_feet_above_1cm_fraction"] == 0) == bool(offset)
    p[..., 2] = 0.022
    assert rear_swing_metrics(p, np.full(4, 0.022))["phase_fraction"] is None


def test_overlap_reward_does_not_change_the_physics():
    envs = [
        DogEnv(32, seed=2, limb_stage="consolidate", threads=1, rear_overlap_weight=w)
        for w in (0, 1)
    ]
    difference = 0
    try:
        for _ in range(50):
            rewards = [e.step(np.zeros((32, 12)))[0] for e in envs]
            difference += abs(rewards[0] - rewards[1]).sum()
            np.testing.assert_array_equal(envs[0].obs(), envs[1].obs())
            for a, b in zip(envs[0].groups, envs[1].groups, strict=True):
                np.testing.assert_array_equal(a.qpos, b.qpos)
                np.testing.assert_array_equal(a.qvel, b.qvel)
        assert difference > 0
    finally:
        for e in envs:
            e.close()
