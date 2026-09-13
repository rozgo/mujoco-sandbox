import mujoco
import numpy as np
import pytest

from embodied_fly.batch import FlyBatch


@pytest.mark.parametrize("sensor_extension_size", (0, 6, 12))
def test_native_batch_matches_single_physics_observation_and_partial_reset(
    sensor_extension_size,
):
    env = FlyBatch(2, 2, sensor_extension_size)
    single = env.template
    env.reset([0], yaw=[0.17])
    single.reset(yaw=0.17)
    np.testing.assert_array_equal(
        single.observation(
            sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
        ),
        env.observation()[0],
    )
    rng = np.random.default_rng(7104)
    peak = 0.0
    for _ in range(100):
        action = single.walking_action(rng.normal(0, 0.03, 78))
        single.step(action)
        env.step(np.stack([action, -action]))
        peak = max(peak, env.forbidden_peak[0])
        np.testing.assert_allclose(single.data.qpos, env.fields["qpos"][0], atol=1e-12)
        np.testing.assert_array_equal(
            single.observation(
                sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
            ),
            env.observation()[0],
        )
        np.testing.assert_allclose(single.anatomical_velocity(), env.velocity()[0], atol=1e-10)
    np.testing.assert_allclose(peak, single.maximum_disallowed_ground_force, atol=1e-9)
    other = env.fields["qpos"][1].copy()
    env.reset([0], yaw=[0.17])
    env.reset([0], yaw=[0.17])  # identical repeated requested poses must survive reset
    single.reset(yaw=0.17)
    np.testing.assert_array_equal(
        single.observation(
            sensor_extension_size >= 6, wing_angles=sensor_extension_size == 12
        ),
        env.observation()[0],
    )
    np.testing.assert_array_equal(other, env.fields["qpos"][1])
    assert env.model.nv == 108 and env.model.nu == 78
    np.testing.assert_array_equal(
        env.model.actuator_forcerange, single.model.actuator_forcerange
    )


@pytest.mark.parametrize("extension", (0, 12))
def test_flight_batch_matches_airborne_native_actuation_aerodynamics_and_reset(extension):
    env = FlyBatch(2, 2, extension, preset="flight")
    single = env.template
    assert env.control_dt == 0.0002 and env.substeps == 4
    assert env.model.opt.timestep == 0.00005
    assert env.model.nu == 78 and env.model.na == 72
    for name in (
        "body_mass",
        "body_inertia",
        "geom_fluid",
        "geom_contype",
        "geom_conaffinity",
        "actuator_gainprm",
        "actuator_dynprm",
        "actuator_dyntype",
        "actuator_forcerange",
        "dof_damping",
        "jnt_stiffness",
    ):
        np.testing.assert_array_equal(getattr(env.model, name), getattr(single.model, name))
    initial = {
        name: getattr(single.data, name).copy() for name in ("qpos", "qvel", "act", "ctrl")
    }
    initial["qpos"][2] = 1.0  # 10 mm airborne reset only; no live pose writes.
    initial["qpos"][single.wing_angle_indices] = (0.4, -0.7, 0.2, 0.4, -0.7, 0.2)
    initial["qvel"][single.wing_velocity_indices] = (500, -200, 800, -500, 200, -800)
    selected_state = {name: value[None] for name, value in initial.items()}
    env.reset([0], state=selected_state)
    for name, value in initial.items():
        getattr(single.data, name)[:] = value
    mujoco.mj_forward(single.model, single.data)
    peak_aerodynamic_load = 0.0
    for step in range(100):
        action = single.passive_action.copy()
        wings = np.array(["wing_" in name for name in single.action_names])
        action[wings] = 0.2 * np.sin(step * 0.27 + np.arange(6))
        single.step(action)
        env.step(np.stack([action, action]))
        np.testing.assert_array_equal(env.previous_action[0], action)
        np.testing.assert_allclose(single.data.qpos, env.fields["qpos"][0], atol=1e-12)
        np.testing.assert_allclose(single.data.qvel, env.fields["qvel"][0], atol=1e-10)
        np.testing.assert_allclose(
            single.data.qfrc_passive, env.fields["qfrc_passive"][0], atol=1e-10
        )
        np.testing.assert_array_equal(
            single.observation(extension > 0, wing_angles=extension == 12),
            env.observation()[0],
        )
        peak_aerodynamic_load = max(
            peak_aerodynamic_load, np.linalg.norm(single.data.qfrc_passive[:3])
        )
    assert single.data.time == pytest.approx(0.02)
    assert peak_aerodynamic_load > 0.01 * env.body_weight
    assert not single.data.xfrc_applied.any() and not single.data.qfrc_applied.any()
    other = {name: values[1].copy() for name, values in env.fields.items()}
    env.reset([0], state=selected_state)
    env.reset([0], state=selected_state)
    np.testing.assert_array_equal(env.fields["qpos"][0], initial["qpos"])
    for name, expected in other.items():
        np.testing.assert_array_equal(env.fields[name][1], expected)
    before = env.fields["qpos"].copy()
    with pytest.raises(ValueError, match="Invalid reset state"):
        env.reset([0], state={"qpos": np.full((1, env.model.nq), np.nan)})
    np.testing.assert_array_equal(env.fields["qpos"], before)
