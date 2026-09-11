import numpy as np
import pytest

from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.visible_steps import VisibleSteps, swing_metrics


def trajectory(height=0.06, advance=0.20):
    p = np.zeros((90, 1, 4, 3))
    p[..., 2] = 0.022
    force = np.full((90, 1, 4), 40.0)
    for i in range(60, 70):
        p[i, 0, 0] += [
            advance * (i - 59) / 10,
            0,
            height * np.sin(np.pi * (i - 59) / 11),
        ]
        force[i, 0, 0] = 0
    p[70:, 0, 0, 0] = advance
    return p, force


def earned(height, advance=0.20, missing=False):
    p, f = trajectory(height, advance)
    tracker = VisibleSteps(1)
    tracker.reset([0], p[0], f[0])
    valid = np.ones((1, 12))
    if missing:
        valid[0, 2] = 0
    return sum(
        tracker.update(x, y, 0.022, valid, np.array([[1.0, 0.0]]), True)[0]
        for x, y in zip(p[1:], f[1:], strict=True)
    )


def test_completed_lift_beats_shuffle_backward_and_hover():
    assert earned(0.06) > earned(0.01)
    assert earned(0.06) > earned(0.06, -0.20)
    assert earned(0.06, missing=True) == 0
    t = VisibleSteps(1)
    p = np.zeros((1, 4, 3))
    p[..., 2] = 0.022
    f = np.full((1, 4), 40.0)
    t.reset([0], p, f)
    p[..., 2] += 0.06
    # Liftoff itself never earns a landing bonus, nor does continued hovering.
    for _ in range(30):
        assert (
            t.update(p, f * 0, 0.022, np.ones((1, 12)), np.array([[1.0, 0.0]]), True)[0]
            == 0
        )
    t.reset([0], p, f * 0)
    assert not t.active.any() and not t.air.any()


def test_independent_measure_requires_a_completed_visible_swing():
    for h, expected in ((0.06, 1), (0.01, 0)):
        p, f = trajectory(h)
        result = swing_metrics(p[:, 0, 0], f[:, 0, 0], 0.022)
        assert result["completed_swings"] == 1
        assert result["visible_swings"] == expected
    assert swing_metrics(p[:65, 0, 0], f[:65, 0, 0], 0.022)["completed_swings"] == 0


def test_reward_does_not_move_the_robot():
    envs = [
        DogEnv(32, seed=2, limb_stage="consolidate", threads=1, visible_step_weight=w)
        for w in (0, 2)
    ]
    diff = 0
    try:
        for _ in range(50):
            rewards = [e.step(np.zeros((32, 12)))[0] for e in envs]
            np.testing.assert_array_equal(envs[0].obs(), envs[1].obs())
            diff += abs(rewards[0] - rewards[1]).sum()
            for a, b in zip(envs[0].groups, envs[1].groups, strict=True):
                np.testing.assert_array_equal(a.qpos, b.qpos)
                np.testing.assert_array_equal(a.qvel, b.qvel)
        assert diff > 0
    finally:
        for e in envs:
            e.close()
    with pytest.raises(ValueError, match="flat ground"):
        DogEnv(visible_step_weight=1, terrain="steps")
