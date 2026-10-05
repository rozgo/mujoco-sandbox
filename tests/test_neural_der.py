"""Meaningful physical and state-management regressions for the DER port."""

from dataclasses import replace

import mujoco
import numpy as np

from sixlegs.neural_insertion.der import elastic_metrics, load_der
from sixlegs.neural_insertion.der_benchmarks import audit
from sixlegs.neural_insertion.rescaling import CableConfig, load_cable
from sixlegs.neural_insertion.units import MM_G


def test_direct_projection_matches_elastic_energy_and_preserves_lifecycle():
    result = audit("direct")
    assert result["max_virtual_work_relative_error"] < 1e-4
    for field in ("repeat_force_error_nm", "copy_force_error_nm", "restore_force_error_nm",
                  "reset_force_error_nm", "query_plugin_state_error"):
        assert result[field] < 1e-15
    assert result["unit_force_relative_error"] < 1e-9
    assert result["native_mass_max_difference_kg"] == 0
    assert result["native_inertia_max_difference_kg_m2"] == 0


def test_published_force_limitation_is_retained_and_direct_torque_has_correct_sign():
    config = CableConfig(length_m=.01, segments=8, gravity_m_s2=0, relaxation_s=0)
    observed = {}
    for projection in ("published", "direct"):
        model, data = load_der(config, projection=projection)
        direction = np.zeros(model.nv)
        direction[9] = .5
        mujoco.mj_integratePos(model, data.qpos, direction, 1.)
        mujoco.mj_forward(model, data)
        observed[projection] = data.qfrc_passive.copy()/MM_G.torque
        assert elastic_metrics(model, data)["twisting_j"] > 0
    # This is a recorded limitation, not a passing torsion claim for upstream.
    np.testing.assert_array_equal(observed["published"], 0)
    assert observed["direct"][9] < 0
    gj = config.ei/(1+config.poisson)
    expected = -gj*.5/(config.length_m*(config.segments-1)/config.segments)
    np.testing.assert_allclose(observed["direct"][::3], expected, rtol=1e-12, atol=1e-20)


def test_extra_queries_and_copied_restart_do_not_change_dynamics():
    config = CableConfig(length_m=.01, segments=8, dt_s=1e-7, gravity_m_s2=0, relaxation_s=0)
    model, a = load_der(config)
    direction = np.zeros(model.nv)
    direction[11] = .2
    direction[18] = .1
    mujoco.mj_integratePos(model, a.qpos, direction, 1.)
    mujoco.mj_forward(model, a)
    b = mujoco.MjData(model)
    mujoco.mj_copyData(b, model, a)
    for _ in range(100):
        mujoco.mj_step(model, a)
        mujoco.mj_forward(model, b)
        mujoco.mj_forward(model, b)
        mujoco.mj_step(model, b)
    np.testing.assert_allclose(a.qpos, b.qpos, rtol=0, atol=1e-13)
    np.testing.assert_allclose(a.qvel, b.qvel, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.plugin_state, b.plugin_state, rtol=0, atol=1e-13)


def test_native_geometry_contacts_and_inertia_are_preserved():
    config = replace(CableConfig(), segments=32)
    native, _ = load_cable(config)
    port, _ = load_der(config)
    assert port.nv == native.nv
    assert port.ngeom == native.ngeom
    assert port.body_mass[-1] == 0  # Fixed endpoint frame, not a physical support.
    assert not port.dof_armature.any()
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        np.testing.assert_array_equal(getattr(native, name), getattr(port, name)[:-1])
    for name in ("geom_size", "geom_pos", "geom_quat", "geom_solref", "geom_solimp", "geom_friction"):
        np.testing.assert_allclose(getattr(native, name), getattr(port, name), rtol=0, atol=1e-15)
