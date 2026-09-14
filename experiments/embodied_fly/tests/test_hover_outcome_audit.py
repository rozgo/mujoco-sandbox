import numpy as np
import pytest

from embodied_fly.hover_outcome_audit import flight_windows


def capture(velocities):
    qpos = np.zeros((len(velocities), 7))
    qpos[:, 2] = 2
    position = np.zeros((len(velocities), 3))
    position[:, 2] = 2 + np.cumsum(velocities) * 0.002
    return {
        "time": np.arange(len(velocities)) * 0.002,
        "measured_velocity": np.column_stack((np.zeros((len(velocities), 2)), velocities)),
        "post_position": position,
        "qpos": qpos,
    }


def test_initial_drop_does_not_count_as_better_later_hover():
    steady = capture(np.full(5000, 0.5))
    dipping = capture(np.r_[np.full(1000, -1.0), np.full(4000, 0.875)])
    assert steady["post_position"][-1, 2] == pytest.approx(dipping["post_position"][-1, 2])
    a, b = flight_windows(steady, None), flight_windows(dipping, None)
    assert a["settled_2_10"]["mean_height_rate_mm_s"] == pytest.approx(5)
    assert b["settled_2_10"]["mean_height_rate_mm_s"] == pytest.approx(8.75)
    assert b["startup_0_2"]["minimum_height_mm"] < a["startup_0_2"]["minimum_height_mm"]


def test_failure_does_not_create_a_full_hover_metric():
    result = flight_windows(capture(np.zeros(5000)), 1.5)
    assert not result["startup_0_2"]["complete"]
    assert not result["settled_2_10"]["complete"]
    assert "velocity_rms_mm_s" not in result["settled_2_10"]
