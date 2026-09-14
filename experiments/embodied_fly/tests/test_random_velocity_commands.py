import numpy as np

from embodied_fly.random_velocity_commands import SmoothRandomCommands


def test_random_command_bounds_reproducibility_and_braking():
    source = SmoothRandomCommands()
    times = np.r_[np.arange(0, 200, 0.017), [1e6, 1e7]]
    values = np.array([source.command(t) for t in times])
    assert np.isfinite(values).all()
    assert np.all(np.abs(values) <= [1.5, 1.5, 1.5, 4.5])
    other = SmoothRandomCommands()
    for t in times[::193]:
        np.testing.assert_array_equal(source.command(t), other.command(t))
    for segment in range(0, 100, 5):
        np.testing.assert_array_equal(source.command(segment * 2 + 1.5), np.zeros(4))
    assert len(np.unique(values.round(3), axis=0)) > 100
    assert not np.allclose(source.command(3), SmoothRandomCommands(seed=7).command(3))


def test_command_and_acceleration_have_no_step_at_knots():
    source = SmoothRandomCommands()
    h = 1e-4
    # Ramps end halfway through each segment. Check both those joins and the
    # next segment's start rather than accepting bounded but jerky white noise.
    for t in np.arange(2, 30, 1, dtype=float):
        left = source.command(t - h)
        center = source.command(t)
        right = source.command(t + h)
        assert np.max(np.abs(right - left)) < 1e-8
        assert np.max(np.abs((right - 2 * center + left) / h**2)) < 0.02
