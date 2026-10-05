"""Independent physics checks for the contact audit, not relaxed quality gates."""

import mujoco
import numpy as np
import pytest

from sixlegs.neural_insertion.contact_audit import frozen_normal, normal_acceleration, normal_fixed_step, normal_reference
from sixlegs.neural_insertion.contact_diagnostics import kernel, load_fixture


def test_independent_normal_force_matches_frozen_mujoco_contact():
    result = frozen_normal()
    assert result["passed"]
    assert normal_acceleration(1e-9, -.3) == -9.81  # separated: no force
    assert normal_acceleration(-1e-7, 1.) == -9.81  # unloading: no tensile contact


def test_independent_free_fall_matches_discrete_closed_form():
    dt, duration = 1e-7, .001
    state, _ = normal_fixed_step(dt, duration)
    n = round(duration/dt)
    expected_gap = .006-.5*9.81*dt*dt*n*(n+1)
    np.testing.assert_allclose(state[-1], [expected_gap, -9.81*duration], rtol=0, atol=1e-14)


def test_independent_recurrence_reproduces_a_complete_small_rigid_impact():
    dt = 1e-7
    c, m, d, _ = load_fixture("segment_normal", dt)
    initial_gap, initial_velocity = 1e-8, -.343
    d.qpos[2] = (c.floor_z_m+c.radius_m+initial_gap)*1000
    d.qvel[2] = initial_velocity*1000
    mujoco.mj_forward(m, d)
    independent, _ = normal_fixed_step(dt, .0002, initial_gap, initial_velocity)
    measured = []
    for _ in range(2):
        out = np.zeros(15)
        kernel().contact_chunk(m._address, d._address, 1000, m.geom("support").id, m.body("segment").id, out)
        assert not out[14]
        measured.append(d.qpos[2]/1000-(c.floor_z_m+c.radius_m))
    np.testing.assert_allclose(measured, independent[1:, 0], rtol=0, atol=1e-10)


def test_continuous_reference_resolves_bounce_under_tighter_integration():
    pytest.importorskip("scipy")
    times = np.arange(601)*.0001
    a, _ = normal_reference(times)
    b, _ = normal_reference(times, tighter=True)
    assert np.max(abs(a[:, 0]-b[:, 0])) < 1e-11
    # Actual rebound after the first impact, not a reference that stays in flight.
    window = (times > .0351) & (times < .043)
    assert b[window, 0].max() > 50e-6
