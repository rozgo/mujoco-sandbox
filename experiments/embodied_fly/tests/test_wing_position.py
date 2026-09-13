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
