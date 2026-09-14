import mujoco
import numpy as np
import torch
from scipy import sparse

from embodied_fly.batch import FlyBatch
from embodied_fly.fresh_velocity_actor import initialize, parameter_digest
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import ARRAYS, physical_contract
from embodied_fly.velocity_exercise import DURATION, STAGES, command_at
from embodied_fly.velocity_motor import VelocityPID, heading_rotation, observation
from embodied_fly.wing_motion import WingMotionForces


def test_velocity_frame_rotates_heading_without_tilting_world_vertical():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    heading = heading_rotation(rotation)[0]
    np.testing.assert_allclose(heading @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(heading @ [0, 0, 1], [0, 0, 1])


def test_fresh_brain_reproducible_without_policy_baggage():
    adjacency = sparse.csr_matrix((np.ones(4), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6))
    a = initialize(adjacency, [0, 1], [2, 3], [4, 5], 20)
    b = initialize(adjacency, [0, 1], [2, 3], [4, 5], 20)
    c = initialize(adjacency, [0, 1], [2, 3], [4, 5], 21)
    assert parameter_digest(a) == parameter_digest(b) != parameter_digest(c)
    assert a.observation_size == 391 and a.sensor_extension_size == 0
    assert torch.all(a.observation_mean == 0) and torch.all(a.observation_std == 1)
    assert torch.count_nonzero(a.wing_residual.network[-1].weight)
    assert not a.initial_state(2).any()
    torch.testing.assert_close(a.core.adjacency.to_dense(), c.core.adjacency.to_dense())


def test_no_position_or_needs_leak_into_velocity_commands_and_teacher():
    env = FlyBatch(
        1,
        1,
        12,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
        heading_control=True,
    )
    tasks = HoverOnlyTasks(env, 31)
    command = np.array([[0.1, -0.1, 0.2, 0.0]])
    before = observation(env, command)
    env.command[:] = 9
    env.needs[:] = 7
    env.requested_xy_cm[:] = 400
    env.requested_height_cm[:] = 200
    np.testing.assert_array_equal(before, observation(env, command))
    np.testing.assert_allclose(before[:, 375:379], command)
    env.template.data.time = 0.1
    left, right = (
        VelocityPID(env.template, tasks.air_action),
        VelocityPID(env.template, tasks.air_action),
    )
    action = left.act(command[0])
    env.template.data.qpos[:3] += [100, -200, 50]
    # Absolute root position cannot influence a pure velocity teacher.
    np.testing.assert_array_equal(action, right.act(command[0]))


def test_exercise_contains_every_direction_and_hover_without_resets():
    ends = np.cumsum([stage[1] for stage in STAGES])
    commands = np.array([command_at(end - 0.1)[0] for end in ends])
    for axis in range(4):
        assert np.any(commands[:, axis] > 0) and np.any(commands[:, axis] < 0)
    np.testing.assert_array_equal(command_at(0)[0], np.zeros(4))
    np.testing.assert_array_equal(command_at(DURATION)[0], np.zeros(4))
    for end in ends[:-1]:
        np.testing.assert_allclose(command_at(end - 1e-8)[0], command_at(end)[0], atol=1e-8)


def test_zero_velocity_commands_keep_wings_active_to_support_weight():
    env = FlyBatch(
        1,
        1,
        12,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
        heading_control=True,
    )
    tasks = HoverOnlyTasks(env, 32)
    controller = VelocityPID(env.template, tasks.air_action)
    env.template.data.qvel[:] = 0
    env.template.data.time = 0.005
    first = controller.act(np.zeros(4))
    env.template.data.time = 0.015
    second = controller.act(np.zeros(4))
    assert not np.array_equal(
        first[controller.wings.channels], second[controller.wings.channels]
    )
    assert np.all(controller.wings.amplitudes > 0)


def test_independent_yaw_comes_from_wing_pitch_and_preserves_body_mechanics(tmp_path):
    kwargs = {"preset": "wing_position", "wing_response": "instant", "physics_hz": 1000}
    old, new = FlyBatch(1, 1, 12, **kwargs), FlyBatch(1, 1, 12, heading_control=True, **kwargs)
    for key in ARRAYS:
        np.testing.assert_array_equal(getattr(old.model, key), getattr(new.model, key))
    assert physical_contract(old.model) != physical_contract(new.model)
    assert physical_contract(new.model)["independent_heading_control"]
    m = new.model
    force = WingMotionForces(m, 3)
    angles = np.tile([0, 0.7, -1, 0, 0.7, -1], (3, 1)).astype(float)
    angles[1, [2, 5]] += (-0.1, 0.1)
    angles[2, [2, 5]] += (0.1, -0.1)
    speed = np.tile([40, 0, 0, 40, 0, 0], (3, 1))
    wrench = force.advance(
        angles, speed, np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 6)), 0.001
    ).copy()
    # cos((-1 +/- .1) + 1) can differ by floating-point roundoff.
    np.testing.assert_allclose(wrench[:, 3:5], 0, atol=1e-15, rtol=0)
    assert wrench[0, 5] == 0 and wrench[1, 5] > 0 and wrench[2, 5] < 0
    np.testing.assert_allclose(wrench[1, :5], wrench[2, :5], atol=1e-15)
    np.testing.assert_array_equal(
        force.advance(
            angles,
            np.zeros_like(speed),
            np.tile(np.eye(3), (3, 1, 1)),
            np.zeros((3, 6)),
            0.001,
        ),
        0,
    )
    path = tmp_path / "heading.mjb"
    mujoco.mj_saveModel(m, str(path))
    assert physical_contract(mujoco.MjModel.from_binary_path(str(path))) == physical_contract(
        m
    )
