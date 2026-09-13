import numpy as np
from test_motor_focus import TEACHER

from embodied_fly.batch import FlyBatch
from embodied_fly.motor_focus import MotorTasks, MotorTeacher
from embodied_fly.physical_contract import physical_contract
from embodied_fly.state_hover import wing_commands


def test_measured_state_teacher_ignores_timer_and_responds_to_wing_and_height():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    tasks = MotorTasks(env, 92001)
    teacher = MotorTeacher(tasks, TEACHER, "cpu", True, "state")
    contract = physical_contract(env.model)
    before = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    initial = teacher.act()[2]
    for age in (10, 21, 31, 1000):
        env.ages[2] = age
        np.testing.assert_array_equal(teacher.act()[2], initial)
    for key, value in before.items():
        np.testing.assert_array_equal(env.fields[key], value)
    assert physical_contract(env.model) == contract
    assert (initial[teacher.channels[[0, 3]]] > 0).all()  # start from zero wing speed
    env.requested_height_cm[2] += 0.2
    high = teacher.act()[2]
    assert (high[teacher.channels[[0, 3]]] > initial[teacher.channels[[0, 3]]]).all()
    env.fields["qvel"][2, env.template.wing_velocity_indices] = (50, 0, 0, 50, 0, 0)
    assert not np.array_equal(teacher.act()[2, teacher.channels], high[teacher.channels])


def test_state_reference_starts_hover_through_bounded_physical_wing_motion():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    tasks = MotorTasks(env, 92001)
    tasks.task_ids[:] = 2
    tasks.reset(np.arange(3))
    contract = physical_contract(env.model)
    channels = [
        env.model.actuator(env.model.joint(j).name).id for j in env.template.wing_joint_ids
    ]
    minimum = np.inf
    for _ in range(250):
        action = np.tile(tasks.air_action, (3, 1))
        action[:, channels] = wing_commands(
            env.fields["qpos"][:, env.template.wing_angle_indices],
            env.fields["qvel"][:, env.template.wing_velocity_indices],
            env.velocity(),
            env.fields["qpos"][:, 2],
            env.requested_height_cm,
            env.model.qpos_spring[env.template.wing_angle_indices],
        )
        assert np.isfinite(action).all() and abs(action).max() <= 1
        env.step(action)
        minimum = min(minimum, env.fields["qpos"][:, 2].min())
    assert minimum > 1.0  # CGS: remain at least 1 cm above the floor
    assert (env.wing_forces.activity > 0.3).all()
    assert np.all(env.fields["xmat"][:, env.template.thorax_id, 8] > 0.99)
    assert not env.fields["warning"].any()
    assert physical_contract(env.model) == contract
