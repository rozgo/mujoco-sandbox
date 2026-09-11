import numpy as np

from adaptive_locomotion.stride import StrideTracker


def landing_reward(distance):
    tracker = StrideTracker(1)
    positions = np.zeros((1, 4, 3))
    force = np.full((1, 4), 40.0)
    tracker.reset([0], positions, force)
    direction = np.array([[1.0, 0]])
    force[0, 0] = 0
    for i in range(10):
        positions[0, 0] = [distance * (i + 1) / 10, 0, 0.04]
        tracker.update(positions, force, direction, np.array([True]))
    positions[0, 0, 2] = 0
    force[0, 0] = 40
    return tracker.update(positions, force, direction, np.array([True]))[0]


def test_long_forward_swing_preferred_to_shuffle_or_backward_step():
    assert landing_reward(0.35) > landing_reward(0.20) > landing_reward(-0.1)


def test_hover_and_slide_do_not_earn_stride_reward_and_reset_clears_history():
    tracker = StrideTracker(2)
    positions = np.zeros((2, 4, 3))
    forces = np.full((2, 4), 40.0)
    tracker.reset([0, 1], positions, forces)
    forces[0] = 0
    direction = np.array([[1.0, 0], [1.0, 0]])
    for _ in range(20):
        positions[1, :, 0] += 0.02
        reward = tracker.update(positions, forces, direction, np.ones(2, bool))
        assert np.all(reward <= 0)
    tracker.reset([0], positions, forces)
    assert not tracker.air[0].any()
    np.testing.assert_array_equal(tracker.previous, positions)
    assert np.all(tracker.update(positions, forces, direction, np.zeros(2, bool)) == 0)
