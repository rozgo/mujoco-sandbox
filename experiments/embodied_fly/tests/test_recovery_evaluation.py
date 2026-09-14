import numpy as np
import pytest

from embodied_fly.evaluate_recovery import recovery_windows


def test_recovery_metrics_measure_actual_velocity_reduction():
    velocity = np.zeros((1500, 3))
    velocity[:500, 2] = 1
    velocity[500:, 2] = 0.25
    result = recovery_windows(
        {"time": np.arange(1500) * 0.002, "measured_velocity": velocity}, None, 3
    )
    assert result["complete_recovery_window"]
    assert result["first_second"]["vertical_rms_mm_s"] == pytest.approx(10)
    assert result["last_second"]["vertical_rms_mm_s"] == pytest.approx(2.5)


def test_failed_recovery_has_no_complete_window_score():
    arrays = {"time": np.arange(1500) * 0.002, "measured_velocity": np.zeros((1500, 3))}
    assert recovery_windows(arrays, 1.9, 3) == {"complete_recovery_window": False}
