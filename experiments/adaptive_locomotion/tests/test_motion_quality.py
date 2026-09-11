import numpy as np

from adaptive_locomotion.motion_quality import measure


def test_absolute_spectral_rms_detects_jitter_even_with_a_larger_slow_gait():
    time = np.arange(550) * 0.02
    fast = 0.1 * np.sin(2 * np.pi * 9 * time)
    slow = 2 * np.sin(2 * np.pi * 2 * time)
    measures = []
    for q in (fast, fast + slow):
        empty = np.zeros((550, 1, 3))
        measures.append(
            measure(
                q[:, None, None],
                q[:, None, None],
                empty,
                empty,
                np.zeros((550, 1)),
                np.zeros((550, 1, 4)),
            )
        )
    for value in measures:
        assert (
            abs(value["summary"]["joint_above_6hz_rms_rad"] - 0.1 / np.sqrt(2)) < 1e-4
        )
    assert measures[1]["summary"]["joint_above_6hz_power_fraction"] < 0.01
