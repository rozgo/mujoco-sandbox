import numpy as np
import pytest

from embodied_fly.record import replay_clock


def test_walking_and_flight_replay_share_real_time_with_distinct_state_rates():
    for hz in (500, 5000):
        states = {"qpos": np.zeros((hz, 1)), "time": np.arange(hz) / hz}
        indices, timestamps, rate = replay_clock(states, {"control_hz": hz})
        assert rate == hz
        assert len(indices) == 50
        np.testing.assert_allclose(timestamps[indices], np.arange(50) / 50)
    legacy = {"qpos": np.zeros((1000, 1))}
    indices, _, _ = replay_clock(legacy, {})
    np.testing.assert_array_equal(indices, np.arange(0, 1000, 10))


def test_replay_rejects_mislabeled_clock():
    states = {"qpos": np.zeros((100, 1)), "time": np.arange(100) / 5000}
    with pytest.raises(ValueError, match="control rate"):
        replay_clock(states, {"control_hz": 500})
