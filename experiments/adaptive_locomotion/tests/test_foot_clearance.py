import numpy as np
import pytest

from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.foot_clearance import clearance_cost, clearance_metrics


def test_clearance_cost_distinguishes_dragging_from_support_and_swing():
    p = np.zeros((4, 4, 3))
    p[..., 2] = 0.022
    q = p.copy()
    q[1:, :, 0] += 0.02
    valid = np.ones((4, 12))
    valid[:, 6:9] = 0
    command = np.tile([0.55, 0, 0], (4, 1))
    p[2, :, 2] += 0.03
    q[2, :, 2] += 0.03
    valid[3] = 1  # The healthy dog keeps its existing objective.
    cost = clearance_cost(p, q, 0.022, valid, command)
    assert cost[0] == cost[2] == cost[3] == 0
    assert cost[1] > 0.9
    q[:, 2] = 100  # Missing leg cannot alter the reward.
    np.testing.assert_array_equal(clearance_cost(p, q, 0.022, valid, command), cost)
    assert not clearance_cost(p, q, 0.022, valid, np.zeros_like(command)).any()


def test_clearance_metrics_measure_physical_ground_travel():
    p = np.zeros((100, 1, 4, 3))
    p[..., 2] = 0.022
    p[:, :, 0, 0] = np.arange(100)[:, None] * 0.02
    p[:, :, 1, 0] = p[:, :, 0, 0]
    p[:, :, 1, 2] += 0.025
    r = clearance_metrics(p, np.full((1, 4), 0.022))
    np.testing.assert_allclose(r["drag_travel_m"], [[1, 0, 0, 0]])
    np.testing.assert_allclose(r["moving_clearance_m"], [[0, 0.025, 0, 0]])


def test_rear_scope_leaves_front_damage_and_front_feet_out_of_the_objective():
    p = np.zeros((3, 4, 3))
    p[..., 2] = 0.022
    q = p.copy()
    q[..., 0] += 0.02
    valid = np.ones((3, 12))
    valid[0, 2] = 0  # Front calf removed: no rear correction requested.
    valid[1, 8] = 0  # Rear calf removed.
    valid[2, 6:9] = 0  # Whole rear leg removed.
    commands = np.tile([0.55, 0, 0], (3, 1))
    cost = clearance_cost(p, q, 0.022, valid, commands, scope="surviving_rear")
    assert cost[0] == 0 and cost[1] == cost[2] > 0.9
    p[1:, 3, 2] += 0.03
    q[1:, 3, 2] += 0.03
    assert not clearance_cost(
        p, q, 0.022, valid, commands, scope="surviving_rear"
    ).any()


def test_clearance_reward_changes_learning_signal_without_changing_physics():
    envs = [
        DogEnv(
            32, seed=2, limb_stage="consolidate", threads=1, damage_clearance_weight=w
        )
        for w in (0, 1)
    ]
    difference = 0
    try:
        for _ in range(50):
            a = np.zeros((32, 12))
            rewards = [env.step(a)[0] for env in envs]
            np.testing.assert_array_equal(envs[0].obs(), envs[1].obs())
            difference += abs(rewards[0] - rewards[1]).sum()
            for g, h in zip(envs[0].groups, envs[1].groups, strict=True):
                np.testing.assert_array_equal(g.qpos, h.qpos)
                np.testing.assert_array_equal(g.qvel, h.qvel)
        assert difference > 0
    finally:
        for env in envs:
            env.close()
    with pytest.raises(ValueError, match="flat ground"):
        DogEnv(damage_clearance_weight=1, terrain="steps")
