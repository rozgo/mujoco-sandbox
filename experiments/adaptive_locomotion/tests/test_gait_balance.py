import numpy as np

from adaptive_locomotion.bodies import PRESETS
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.gait_balance import ContactTiming


def timing_cost(duties, offsets):
    timing = ContactTiming(1)
    values = []
    for step in range(150):
        contact = (step + np.asarray(offsets)) % 30 < duties
        value = timing.update(contact[None, :] * 50.0)
        if step > 90:
            values.append(value[0])
    return np.mean(values)


def test_completed_duration_balance_does_not_prescribe_footfall_phase():
    assert timing_cost([15, 15, 15, 15], [0, 15, 15, 0]) < 1e-12
    assert timing_cost([15, 15, 15, 15], [0, 7, 14, 21]) < 1e-12
    assert timing_cost([14, 20, 20, 8], [0, 15, 15, 0]) > 0.01
    timing = ContactTiming(2)
    for _ in range(20):
        timing.update(np.array([[50, 0, 50, 0], [0, 50, 0, 50]]))
    untouched = timing.air[1].copy()
    timing.reset([0], np.ones((2, 4)) * 50)
    assert not timing.air[0].any()
    np.testing.assert_array_equal(timing.air[1], untouched)


def test_gait_balance_is_inactive_for_a_shortened_body_and_preserves_physics():
    envs = [
        DogEnv(
            1,
            bodies=[PRESETS["short_fl"]],
            randomize=False,
            faults=False,
            threads=1,
            balance_weight=weight,
        )
        for weight in (0, 20)
    ]
    try:
        for i in range(50):
            action = np.full((1, 12), 0.15 * np.sin(i * 0.3))
            results = [e.step(action) for e in envs]
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_array_equal(envs[0].obs(), envs[1].obs())
            np.testing.assert_array_equal(
                envs[0].groups[0].qpos, envs[1].groups[0].qpos
            )
    finally:
        for e in envs:
            e.close()
