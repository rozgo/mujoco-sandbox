"""Physical flight prerequisites, independent of a trained flight controller."""

import mujoco
import numpy as np
import pytest

from embodied_fly.body import FlyEnvironment


def test_flight_keeps_full_body_and_contact_with_compatible_observation_shape():
    walking = FlyEnvironment()
    flight = FlyEnvironment("flight")
    assert flight.model.nv == walking.model.nv == 108
    assert flight.model.nu == walking.model.nu == 78
    assert flight.model.na == 72
    assert flight.observation().shape == walking.observation().shape == (383,)
    np.testing.assert_array_equal(walking.actuator_activation(), walking.data.act)
    np.testing.assert_allclose(flight.model.body_mass, walking.model.body_mass)
    np.testing.assert_array_equal(flight.model.geom_contype, walking.model.geom_contype)
    np.testing.assert_array_equal(
        flight.model.geom_conaffinity, walking.model.geom_conaffinity
    )
    assert flight.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    wing = np.array(["wing_" in n for n in flight.action_names])
    assert wing.sum() == 6
    assert np.all(flight.model.actuator_dyntype[wing] == mujoco.mjtDyn.mjDYN_NONE)
    np.testing.assert_array_equal(flight.model.actuator_gainprm[wing, 0], 18)
    flight.data.ctrl[wing] = 0.1
    np.testing.assert_array_equal(flight.actuator_activation()[wing], 0.1)
    for _ in range(50):
        flight.step(flight.passive_action)
    assert flight.data.time == pytest.approx(0.01)
    assert not np.any(flight.data.contact.geom < 0)
    assert len(flight.data.contact) > 0
    assert not flight.data.warning.number.any()
    assert not flight.data.xfrc_applied.any()
    assert not flight.data.qfrc_applied.any()


def test_enabled_wing_aerodynamics_dissipate_energy_at_same_physical_state():
    flight = FlyEnvironment("flight")
    model, data = flight.model, flight.data
    geoms = [i for i in range(model.ngeom) if "fluid" in model.geom(i).name]
    assert len(geoms) == 2
    enabled = model.geom_fluid[geoms].copy()
    assert np.all(enabled[:, 0] == 1)
    data.qvel[:3] = [7, 2, 1]
    for i in flight.joint_ids:
        if "wing_" in model.joint(i).name:
            data.qvel[model.jnt_dofadr[i]] = 30
    mujoco.mj_forward(model, data)
    force_on = data.qfrc_passive.copy()
    model.geom_fluid[geoms] = 0
    mujoco.mj_forward(model, data)
    aero_difference = force_on - data.qfrc_passive
    assert np.linalg.norm(aero_difference) > 1e-8
    assert aero_difference @ data.qvel < 0
    model.geom_fluid[geoms] = enabled


def test_original_actor_interval_preserves_fine_physics_and_elapsed_time():
    held = FlyEnvironment("flight")
    explicit = FlyEnvironment("flight")
    observation = held.advance(held.passive_action, 0.002)
    for _ in range(10):
        expected = explicit.step(explicit.passive_action)
    np.testing.assert_array_equal(held.data.qpos, explicit.data.qpos)
    np.testing.assert_array_equal(observation, expected)
    assert held.data.time == pytest.approx(0.002)
    before = held.data.qpos.copy()
    with pytest.raises(ValueError):
        held.advance(held.passive_action, 0.0003)
    np.testing.assert_array_equal(held.data.qpos, before)
