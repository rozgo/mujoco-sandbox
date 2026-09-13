import mujoco
import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.motor_focus import MotorTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.wing_motion import CONFIG
from embodied_fly.wing_position import (
    POSITION,
    normalize_targets,
    reference_torque_to_position,
    wing_actuators,
)


def test_position_actuators_hold_rest_and_enforce_torque_limits_without_pose_writes():
    env = FlyBatch(3, 3, 14, preset="wing_position")
    task = MotorTasks(env, 95001)
    model, native = env.model, env.template
    assert physical_contract(model) == physical_contract(native.model)
    assert physical_contract(model)["preset"] == "wing_position"
    posture = GroundPosture(env, task.ground["qpos"])
    ids = wing_actuators(model)
    # Verify the actual compiled actuator force, including saturation.
    d = native.data
    d.qvel[native.wing_velocity_indices] = np.arange(6) * 20
    d.ctrl[ids] = model.actuator_ctrlrange[ids, 0]
    mujoco.mj_forward(native.model, d)
    expected = np.clip(
        POSITION.kp * (d.ctrl[ids] - d.qpos[native.wing_angle_indices])
        - POSITION.kv * d.qvel[native.wing_velocity_indices],
        -CONFIG.joint_torque_limit,
        CONFIG.joint_torque_limit,
    )
    np.testing.assert_allclose(d.actuator_force[ids], expected, atol=1e-12)
    assert np.abs(d.actuator_force[ids]).max() == CONFIG.joint_torque_limit
    reference = posture.wing_targets()
    before = env.fields["qpos"].copy()
    np.testing.assert_array_equal(reference, posture.wing_targets())
    np.testing.assert_array_equal(before, env.fields["qpos"])
    # Physical hold at the ground resting pose for both ground worlds.
    action = np.repeat(posture.rest_action[None], 3, axis=0)
    for _ in range(150):
        env.step(action)
    np.testing.assert_allclose(
        env.fields["qpos"][:2, native.wing_angle_indices],
        np.repeat(task.ground["qpos"][native.wing_angle_indices][None], 2, axis=0),
        atol=1e-5,
    )
    assert not env.fields["warning"].any()


def test_reference_conversion_matches_requested_torque_at_current_state():
    env = FlyBatch(3, 3, 14, preset="wing_position")
    t, m = env.template, env.model
    q = np.tile([0, 0.7, -1, 0, 0.7, -1], (3, 1))
    v = np.tile([10, 1, -1, -10, 2, -2], (3, 1))
    requested = np.linspace(-0.4, 0.4, 18).reshape(3, 6)
    action = reference_torque_to_position(m, requested, q, v)
    limits = m.actuator_ctrlrange[wing_actuators(m)]
    target = limits[:, 0] + (action + 1) / 2 * (limits[:, 1] - limits[:, 0])
    np.testing.assert_allclose(
        POSITION.kp * (target - q) - POSITION.kv * v,
        requested * CONFIG.joint_torque_limit,
        atol=2e-8,
    )
    np.testing.assert_array_equal(normalize_targets(m, limits[:, 0]), -1)
    np.testing.assert_array_equal(normalize_targets(m, limits[:, 1]), 1)
    assert physical_contract(m) == physical_contract(t.model)


def test_full_stand_labels_do_not_retain_the_parents_crouched_body_targets():
    from pathlib import Path

    from embodied_fly.motor_focus import MotorTeacher

    env = FlyBatch(3, 3, 14, preset="wing_position")
    tasks = MotorTasks(env, 95021)
    teacher = MotorTeacher(
        tasks,
        Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers/walking.npz",
        "cpu",
        ground_posture=True,
        hover_reference="state",
        stand_initial_form=True,
    )
    supplied = np.full((3, 78), 0.43, np.float32)
    before = env.fields["qpos"].copy()
    targets = teacher.act(supplied)
    np.testing.assert_array_equal(targets[0], teacher.posture.rest_action)
    other = np.r_[0:14, 20:78]
    np.testing.assert_array_equal(targets[1, other], supplied[1, other])
    np.testing.assert_array_equal(env.fields["qpos"], before)


def test_anchored_walking_teacher_matches_native_path_and_preserves_physics():
    from pathlib import Path

    from embodied_fly.body import FlyEnvironment
    from embodied_fly.motor_focus import MotorTeacher
    from embodied_fly.teacher import TeacherOracle

    env = FlyBatch(3, 3, 14, preset="wing_position")
    tasks = MotorTasks(env, 96001)
    path = Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers/walking.npz"
    teacher = MotorTeacher(tasks, path, "cpu", True, "state", True, "anchored")
    native = FlyEnvironment("wing_position")
    oracle = TeacherOracle(native, path)
    oracle.set_reference(1.0, 0.0, 1.0, heading=tasks.heading[1])
    contract = physical_contract(env.model)
    for step in range(20):
        env.batch.forward()
        before = env.fields["qpos"].copy()
        actual = teacher.act()
        for key in ("qpos", "qvel", "act", "ctrl"):
            getattr(native.data, key)[:] = env.fields[key][1]
        native.data.time = step * env.control_dt
        mujoco.mj_forward(native.model, native.data)
        native.mean_sensors = env.mean_sensors[1, : native.model.nsensordata].copy()
        expected = oracle.act(step, "world_path")
        expected[wing_actuators(native.model)] = teacher.posture.wing_targets([1])[0]
        np.testing.assert_allclose(actual[1], expected, atol=5e-5, rtol=5e-5)
        np.testing.assert_array_equal(env.fields["qpos"], before)
        env.step(actual)
    assert physical_contract(env.model) == contract


def test_hover_start_weight_and_per_task_assistance_are_training_only():
    import pytest

    from embodied_fly.motor_retention import hover_start_weights, task_mixtures

    ids = np.array([0, 1, 2, 2, 2])
    ages = np.array([0, 0, 0, 24, 25])
    np.testing.assert_array_equal(task_mixtures(ids, 0.0, 0.0, 1.0), [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(
        hover_start_weights(ids, ages, 0.002, 0.05, 10), [1, 1, 10, 10, 1]
    )
    for weight in (-1, 0.5, float("nan")):
        with pytest.raises(ValueError):
            hover_start_weights(ids, ages, 0.002, 0.05, weight)
