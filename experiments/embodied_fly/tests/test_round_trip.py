import numpy as np
import pytest

from embodied_fly.round_trip import CASES, target_at


def test_six_ordered_trips_hit_both_targets_and_return_exactly_without_moving_origin():
    origin = np.array([0.2, -0.1, 1.8665])
    initial = origin.copy()
    for _, axis, sign in CASES:
        a, b = initial.copy(), initial.copy()
        a[axis] += 0.15 * sign
        b[axis] -= 0.15 * sign
        np.testing.assert_allclose(target_at(3, origin, axis, sign), a)
        np.testing.assert_allclose(target_at(6, origin, axis, sign), b)
        for t in (0, 1, 8, 11, 12, 20):
            np.testing.assert_array_equal(target_at(t, origin, axis, sign), initial)
    np.testing.assert_array_equal(origin, initial)


def test_round_trip_reversals_have_continuous_targets_and_zero_boundary_speed():
    origin = np.zeros(3)
    for t in (1, 2, 4, 5, 7, 8):
        before = target_at(t - 1e-5, origin, 0, 1)
        after = target_at(t + 1e-5, origin, 0, 1)
        assert np.linalg.norm(after - before) / 2e-5 < 1e-7
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            target_at(bad, origin, 0, 1)
