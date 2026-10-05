"""Checks that the smoothing experiment changes only the intended numerics."""

import mujoco
import numpy as np
import pytest

from sixlegs.neural_insertion.contact_audit import normal_acceleration, normal_fixed_step, normal_reference
from sixlegs.neural_insertion.contact_diagnostics import kernel, load_fixture
from sixlegs.neural_insertion.contact_laws import LAWS, ContactLaw
from sixlegs.neural_insertion.contact_smoothing import RIGID, CHAINS, frozen_law, law_audit, load_variant
from sixlegs.neural_insertion.units import MM_G


def test_ramped_impedance_is_continuous_at_entry_and_matches_pinned_spline():
    law = LAWS["ramp_5us"]
    assert law.continuous_entry and law.impedance(0) == mujoco.mjMINIMP
    assert law.impedance(law.width_m) == law.d_width
    assert law.impedance(.5*law.width_m) == pytest.approx(.5*(mujoco.mjMINIMP+law.d_width))
    assert not LAWS["flat"].continuous_entry and LAWS["flat"].impedance(0) == .9999
    # At a grazing entry the ramp applies only the impedance-floor fraction of the
    # damping force (mjMINIMP), while the flat law applies the full B*v jump.
    grazing = normal_acceleration(-1e-12, -.01, law=law)+9.81
    assert 0 < grazing < 1 and normal_acceleration(-1e-12, -.01, law=LAWS["flat"])+9.81 > 1000*grazing
    with pytest.raises(ValueError):
        ContactLaw("bad", d0=0)


def test_variants_change_only_contact_law_and_integrator():
    for kind in RIGID+CHAINS:
        for law, integrator in (("ramp_5us", "RK4"), ("flat", "RK4"), ("ramp_5us", "implicitfast")):
            result = law_audit(kind, LAWS[law], integrator)
            assert result["passed"], (kind, law, integrator, result)
    _, baseline, _, _ = load_fixture("der", 7.8125e-8)
    _, variant, _, _ = load_variant("der", 7.8125e-8, MM_G, LAWS["flat"], "implicitfast")
    np.testing.assert_array_equal(baseline.geom_solimp, variant.geom_solimp)
    np.testing.assert_array_equal(baseline.geom_solref, variant.geom_solref)


def test_frozen_ramped_law_matches_independent_formula_across_units():
    result = frozen_law(LAWS["ramp_5us"])
    assert result["passed"], result["maximum_relative_error"]
    shallow = result["cases"][0]
    assert all(c["contacts"] == 2 for c in shallow["cases"])
    assert mujoco.mjMINIMP < shallow["independent_impedance"] < .03  # inside the ramp, not saturated


def test_kernel_rk4_path_matches_mj_step_bit_for_bit():
    for kind, steps in (("segment_oblique", 1500), ("der", 400)):
        _, model, a, _ = load_variant(kind, 1.5625e-7, MM_G, LAWS["ramp_5us"], "RK4")
        if kind == "segment_oblique":
            a.qpos[2] -= 5.999e-3*MM_G.length  # test initialization close to contact
            a.qvel[2] = -.3*MM_G.length
            mujoco.mj_forward(model, a)
        else:
            # Advance physically into the first impact before comparing step paths.
            out = np.zeros(15)
            kernel().contact_chunk(model._address, a._address, 22900, model.geom("support").id, -1, out)
            assert out[10] > 0 and out[14] == 0
        b = mujoco.MjData(model)
        mujoco.mj_copyData(b, model, a)
        out = np.zeros(15)
        kernel().contact_chunk(model._address, a._address, steps, model.geom("support").id,
                               model.body("segment").id if kind.startswith("segment") else -1, out)
        for _ in range(steps):
            mujoco.mj_step(model, b)
        assert out[0] == steps and out[10] > 0 and out[14] == 0
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)
        np.testing.assert_array_equal(a.plugin_state, b.plugin_state)
        assert a.time == b.time


def test_independent_rk4_recurrence_converges_where_euler_does_not():
    pytest.importorskip("scipy")
    times = np.arange(451)*.0001
    law = LAWS["ramp_5us"]
    reference, _ = normal_reference(times, tighter=True, law=law)
    errors = {}
    for method in ("euler", "rk4"):
        errors[method] = [float(np.max(abs(normal_fixed_step(dt, .045, law=law, method=method)[0][:, 0]-reference[:, 0]))*1e6)
                          for dt in (1.5625e-7, 7.8125e-8)]
    assert errors["euler"][0] > 1 and errors["euler"][1] == pytest.approx(errors["euler"][0]/2, rel=.1)
    assert errors["rk4"][1] < .01 and errors["rk4"][1] < errors["euler"][1]/100
