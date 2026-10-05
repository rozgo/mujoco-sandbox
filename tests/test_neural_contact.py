"""Checks that diagnostic controls preserve physics and logging preserves stepping."""

import mujoco
import numpy as np

from sixlegs.neural_insertion.contact_diagnostics import (
    KINDS, kernel, load_fixture, momentum, unit_audit,
)
from sixlegs.neural_insertion.rescaling import load_cable
from sixlegs.neural_insertion.units import MM_G


def test_unit_conversion_preserves_compiled_physical_fixtures():
    for kind in KINDS:
        assert all(v["passed"] for v in unit_audit(kind).values())


def test_controls_remove_only_the_intended_mechanics():
    config, rigid, rd, _ = load_fixture("segment_oblique", 1.5625e-7)
    native, _ = load_cable(config)
    b = native.body("thread_B_1").id
    np.testing.assert_array_equal(rigid.body_inertia[1], native.body_inertia[b])
    assert rigid.body_mass[1] == native.body_mass[b]
    np.testing.assert_allclose(rigid.body_ipos[1], 0, atol=0)
    assert rd.qvel[0]/MM_G.length == .05
    _, flexible, fd, _ = load_fixture("der", 1.5625e-7)
    _, chain, cd, _ = load_fixture("chain_no_elastic", 1.5625e-7)
    for field in ("body_mass", "body_inertia", "geom_size", "geom_pos", "geom_solref", "geom_solimp", "geom_friction"):
        np.testing.assert_array_equal(getattr(flexible, field), getattr(chain, field))
    np.testing.assert_array_equal(fd.qpos, cd.qpos)
    np.testing.assert_array_equal(fd.qvel, cd.qvel)
    assert flexible.nplugin == 1 and chain.nplugin == 0
    assert not chain.dof_damping.any() and not chain.dof_armature.any()
    assert not cd.qfrc_passive.any()
    assert np.linalg.norm(fd.qfrc_passive) > 0


def test_batch_logger_matches_python_steps_and_contact_impulse_balance():
    _, model, a, _ = load_fixture("segment_oblique", 1.5625e-7)
    # Explicit test initialization close to contact; no live pose assignment.
    a.qpos[2] -= 5.999e-3*MM_G.length
    a.qvel[2] = -.3*MM_G.length
    mujoco.mj_forward(model, a)
    b = mujoco.MjData(model)
    mujoco.mj_copyData(b, model, a)
    p0, _ = momentum(model, a, 1, MM_G)
    out = np.zeros(15)
    steps = 1280
    kernel().contact_chunk(model._address, a._address, steps, model.geom("support").id, 1, out)
    for _ in range(steps):
        mujoco.mj_step1(model, b)
        b.qfrc_applied[:] = 0
        mujoco.mj_step2(model, b)
    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)
    assert out[0] == steps and out[10] > 0 and out[14] == 0
    mujoco.mj_forward(model, a)
    p1, _ = momentum(model, a, 1, MM_G)
    jg = model.body_mass[1]/MM_G.mass*model.opt.gravity/MM_G.length*a.time
    np.testing.assert_allclose(p1-p0-jg, out[4:7]/MM_G.force, rtol=1e-11, atol=1e-22)


def test_batch_stepping_preserves_der_plugin_history():
    _, model, a, _ = load_fixture("der", 1.5625e-7)
    out = np.zeros(15)
    support = model.geom("support").id
    # Advance physically into the first contact before comparing step paths.
    kernel().contact_chunk(model._address, a._address, 23040, support, -1, out)
    assert out[10] > 0 and out[14] == 0
    b = mujoco.MjData(model)
    mujoco.mj_copyData(b, model, a)
    kernel().contact_chunk(model._address, a._address, 640, support, -1, out)
    for _ in range(640):
        mujoco.mj_step1(model, b)
        b.qfrc_applied[:] = 0
        mujoco.mj_step2(model, b)
    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)
    np.testing.assert_array_equal(a.plugin_state, b.plugin_state)
