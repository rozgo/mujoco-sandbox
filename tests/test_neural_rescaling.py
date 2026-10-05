"""Unit/contact regressions; full accuracy acceptance is the benchmark CLI."""

from dataclasses import replace

import numpy as np
import pytest

from sixlegs.neural_insertion.benchmarks import matched_error_um
from sixlegs.neural_insertion.rescaling import CableConfig, load_cable, trial
from sixlegs.neural_insertion.units import SI, MM_G, CM_G, Units


def test_rescaling_preserves_physical_mass_inertia_and_geometry():
    # The unit-equivalence fixture has binary-exact spacing in both systems.
    # Non-binary composite vertex rounding is documented separately.
    config = CableConfig(length_m=.01, segments=16)
    a, da = load_cable(config, MM_G)
    b, db = load_cable(config, CM_G)
    np.testing.assert_allclose(a.body_mass / MM_G.mass, b.body_mass / CM_G.mass,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(a.body_inertia / MM_G.torque, b.body_inertia / CM_G.torque,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(da.site_xpos / MM_G.length, db.site_xpos / CM_G.length,
                               rtol=0, atol=1e-15)
    ids = [i for i in range(a.nbody) if a.body(i).name.startswith("thread_B")]
    assert a.body_mass[ids].sum() / MM_G.mass == pytest.approx(config.mass_kg, rel=1e-12)
    assert np.count_nonzero(a.dof_armature) == 0
    with pytest.raises(ValueError, match="inertia"):
        load_cable(config, SI)


def test_composite_contact_settings_do_not_fall_back_to_internal_defaults():
    c = CableConfig(dt_s=1.25e-6, contact_time_s=1e-5)
    m, _ = load_cable(c)
    for i in range(m.ngeom):
        if m.geom(i).name.startswith("thread_") or m.geom(i).name == "support":
            np.testing.assert_allclose(m.geom_solref[i], (c.contact_time_s, 1))
            assert m.geom_solimp[i, 0] == c.contact_impedance
            assert m.geom_solimp[i, 1] == c.contact_impedance
            assert m.geom_friction[i, 0] == c.friction
            assert m.geom_size[i, 0] > 0


def test_loaded_bending_is_invariant_under_consistent_unit_conversion():
    c = CableConfig(length_m=.01, segments=16, gravity_m_s2=0, floor_z_m=-.02)
    ra, a = trial(c, MM_G, duration=.03, force_n=(0, 0, -1e-8))
    rb, b = trial(c, CM_G, duration=.03, force_n=(0, 0, -1e-8))
    assert ra["completed"] and rb["completed"]
    assert a["tip"][-1, 2] < -1e-6  # A real loaded deflection, not just zero-state equivalence.
    assert matched_error_um(a, b) < .01
    assert a["time"][-1] == pytest.approx(.03)


def test_invalid_or_mismatched_experiments_are_rejected():
    with pytest.raises(ValueError):
        Units("invalid", 0, 1)
    with pytest.raises(ValueError):
        CableConfig(segments=1)
    with pytest.raises(ValueError):
        trial(duration=.00015)
    with pytest.raises(ValueError, match="RK4"):
        trial(replace(CableConfig(), integrator="RK4"), force_n=(1e-8, 0, 0))
    a = {"time": np.array([0.1]), "tip": np.zeros((1, 3))}
    b = {"time": np.array([0.2]), "tip": np.zeros((1, 3))}
    with pytest.raises(ValueError, match="timestamps"):
        matched_error_um(a, b)
