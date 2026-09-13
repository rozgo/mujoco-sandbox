import mujoco
import numpy as np
import pytest

from embodied_fly.batch import FlyBatch
from embodied_fly.body import FlyEnvironment
from embodied_fly.motion_flight import WingReference, initialize
from embodied_fly.wing_motion import CONFIG, WingMotionForces


@pytest.fixture(scope="module")
def env():
    return FlyEnvironment("wing_motion")


def test_wing_model_keeps_interface_with_exactly_decoupled_inertia(env, tmp_path):
    m = env.model
    assert (m.nu, m.nv, m.na, len(env.observation(True, wing_angles=True))) == (
        78,
        108,
        72,
        395,
    )
    assert env.control_dt == 0.002 and m.opt.timestep == 0.0002
    wing = [m.body(f"walker/wing_{side}").id for side in ("left", "right")]
    np.testing.assert_array_equal(m.body_mass[wing], 0)
    np.testing.assert_array_equal(m.body_inertia[wing], 0)
    assert not m.geom_fluid.any() and m.opt.density == m.opt.viscosity == 0
    env.reset()
    mass = np.empty((m.nv, m.nv))
    mujoco.mj_fullM(m, env.data, mass)
    other = np.setdiff1d(np.arange(m.nv), env.wing_velocity_indices)
    np.testing.assert_array_equal(mass[np.ix_(env.wing_velocity_indices, other)], 0)
    assert np.linalg.eigvalsh(mass).min() > 0
    p = tmp_path / "model.mjb"
    mujoco.mj_saveModel(m, str(p))
    loaded = mujoco.MjModel.from_binary_path(str(p))
    data = mujoco.MjData(loaded)
    mujoco.mj_forward(loaded, data)
    np.testing.assert_array_equal(loaded.body_mass, m.body_mass)
    assert np.isfinite(data.qacc).all()


def test_wing_motion_has_no_mechanical_body_reaction_when_force_law_disabled():
    a, b = FlyEnvironment("wing_motion"), FlyEnvironment("wing_motion")
    for e in (a, b):
        e.wing_forces = None  # diagnostic only: cut the declared coupling
        e.data.qpos[2] = 5
        mujoco.mj_forward(e.model, e.data)
    channels = np.array(["wing_" in name for name in a.action_names])
    for step in range(50):
        quiet = a.passive_action.copy()
        moving = quiet.copy()
        moving[channels] = 0.5 * np.sin(step * 0.3 + np.arange(6))
        a.step(quiet)
        b.step(moving)
    np.testing.assert_allclose(a.data.qpos[:7], b.data.qpos[:7], atol=1e-10, rtol=0)
    assert np.linalg.norm(b.data.qvel[b.wing_velocity_indices]) > 1


def test_causal_wing_lift_mirroring_decay_and_damping(env):
    force = WingMotionForces(env.model, 3)
    angle = np.tile([0, 0.7, -1, 0, 0.7, -1], (3, 1))
    velocity = np.zeros((3, 6))
    rotation = np.tile(np.eye(3), (3, 1, 1))
    body_velocity = np.zeros((3, 6))
    np.testing.assert_array_equal(
        force.advance(angle, velocity, rotation, body_velocity, 0.002), 0
    )
    velocity[:, (0, 3)] = [[40, 40], [60, 20], [20, 60]]
    for _ in range(100):
        wrench = force.advance(angle, velocity, rotation, body_velocity, 0.002).copy()
    assert wrench[0, 2] > force.weight
    assert abs(wrench[0, 3:]).max() == 0
    assert wrench[1, 2] == pytest.approx(wrench[2, 2])
    np.testing.assert_allclose(wrench[1, 3:], -wrench[2, 3:])
    assert force.lift.max() <= 2.24 * force.weight
    body_velocity[0, :3] = (1, -2, 3)
    damped = force.advance(angle, velocity, rotation, body_velocity, 0.002)
    assert np.dot(damped[0, 3:], body_velocity[0, :3]) < 0
    velocity[:] = 0
    body_velocity[:] = 0
    for _ in range(150):
        force.advance(angle, velocity, rotation, body_velocity, 0.002)
    assert force.lift.max() < 1e-9 * force.weight
    force.reset([0, 1, 2])
    np.testing.assert_array_equal(force.activity, 0)


def test_native_batch_force_law_feedback_and_selective_reset():
    batch = FlyBatch(2, 2, 12, preset="wing_motion")
    e = batch.template
    state = {name: getattr(e.data, name).copy() for name in ("qpos", "qvel", "act", "ctrl")}
    state["qpos"][2] = 3
    batch.reset([0, 1], state={k: np.stack([v, v]) for k, v in state.items()})
    for k, v in state.items():
        getattr(e.data, k)[:] = v
    mujoco.mj_forward(e.model, e.data)
    channels = np.array(["wing_" in n for n in e.action_names])
    for step in range(40):
        action = e.passive_action.copy()
        action[channels] = 0.6 * np.sin(step * 0.3 + np.arange(6))
        e.step(action)
        batch.step(np.stack([action, -action]))
        np.testing.assert_allclose(batch.fields["qpos"][0], e.data.qpos, atol=3e-6)
        np.testing.assert_allclose(
            batch.wing_forces.activity[0], e.wing_forces.activity[0], atol=3e-6
        )
        np.testing.assert_allclose(
            batch.observation()[0], e.observation(True, wing_angles=True), atol=3e-5
        )
    assert np.any(batch.wing_forces.lift > 0)
    before = batch.wing_forces.activity[1].copy()
    batch.reset([0])
    np.testing.assert_array_equal(batch.wing_forces.activity[0], 0)
    np.testing.assert_array_equal(batch.wing_forces.activity[1], before)
    np.testing.assert_array_equal(batch.fields["xfrc_applied"][0], 0)
    assert (
        batch.model.actuator_gainprm[batch.model.actuator("walker/wing_yaw_left").id, 0]
        == CONFIG.joint_torque_limit
    )


def test_reference_wings_support_free_body_without_direct_body_controller():
    env = FlyEnvironment("wing_motion")
    reference = WingReference(env, initialize(env))
    for _ in range(500):
        env.step(reference.act())
        assert env.data.xmat[env.thorax_id, 8] > 0.95
        assert 0.015 < env.data.qpos[2] * 0.01 < 0.025
    assert not env.data.qfrc_applied.any()
    assert env.maximum_disallowed_ground_force == 0
    assert not env.data.warning.number.any()
