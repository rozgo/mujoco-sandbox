import numpy as np
import pytest
import torch
from scipy import sparse
from test_motor_focus import TEACHER

from embodied_fly.batch import FlyBatch
from embodied_fly.brain import EmbodiedBrain
from embodied_fly.motor_focus import MotorTasks, MotorTeacher
from embodied_fly.motor_retention import FrozenMotorReference, task_loss, task_mixtures


def test_frozen_reference_copies_parameters_shares_only_graph_and_resets_each_world():
    actor = EmbodiedBrain(
        sparse.eye(6, dtype=np.float32, format="csr"),
        [0, 1],
        [2, 3],
        [4, 5],
        397,
        78,
        sensor_extension_size=14,
        motor_only=True,
    )
    reference = FrozenMotorReference(actor, 3)
    assert reference.actor.core.adjacency is actor.core.adjacency
    assert reference.actor.core.transpose is actor.core.transpose
    assert all(
        a.data_ptr() != b.data_ptr() and not b.requires_grad
        for a, b in zip(actor.parameters(), reference.actor.parameters())
    )
    observations = torch.randn(3, 397)
    original = {k: v.clone() for k, v in reference.actor.state_dict().items()}
    before = reference.act(observations)
    assert not before.requires_grad and not reference.memory.requires_grad
    memory = reference.memory.clone()
    reference.reset(torch.tensor([False, True, False]))
    assert not reference.memory[:, 1].any()
    torch.testing.assert_close(reference.memory[:, [0, 2]], memory[:, [0, 2]])
    loss = actor(observations, actor.initial_state(3)).action.square().mean()
    loss.backward()
    torch.optim.Adam(actor.parameters(), lr=0.01).step()
    assert all(torch.equal(reference.actor.state_dict()[k], v) for k, v in original.items())
    assert all(p.grad is None for p in reference.actor.parameters())


def test_retained_ground_labels_keep_body_actions_and_restore_only_ground_wings():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    tasks = MotorTasks(env, 123)
    teacher = MotorTeacher(tasks, TEACHER, "cpu", True, "state")
    actions = (
        np.random.default_rng(123).uniform(-0.2, 0.2, (3, env.model.nu)).astype(np.float32)
    )
    original = actions.copy()
    qpos = env.fields["qpos"].copy()
    result = teacher.act(actions)
    nonwing = np.setdiff1d(np.arange(env.model.nu), teacher.channels)
    np.testing.assert_array_equal(result[:2, nonwing], original[:2, nonwing])
    np.testing.assert_array_equal(
        result[:2, teacher.channels], teacher.posture.wing_targets([0, 1])
    )
    np.testing.assert_array_equal(result[2, nonwing], tasks.air_action[nonwing])
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(env.fields["qpos"], qpos)


def test_assistance_and_loss_are_accounted_for_per_task():
    ids = np.array([0, 1, 2, 0, 1, 2])
    mixes = task_mixtures(ids, 0, 0.8)
    np.testing.assert_allclose(mixes, [0, 0, 0.8, 0, 0, 0.8])
    student, target = np.ones((6, 78)), -np.ones((6, 78))
    executed = (1 - mixes[:, None]) * student + mixes[:, None] * target
    np.testing.assert_array_equal(executed[ids != 2], student[ids != 2])
    np.testing.assert_allclose(executed[ids == 2], -0.6)
    losses = {i: torch.tensor(float(i + 1), requires_grad=True) for i in range(3)}
    loss = task_loss(losses, 4)
    torch.testing.assert_close(loss, torch.tensor(15 / 9))
    loss.backward()
    for i, weight in enumerate([4, 4, 1]):
        torch.testing.assert_close(losses[i].grad, torch.tensor(weight / 9))
    for invalid in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            task_mixtures(ids, 0, invalid)
    for invalid in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            task_loss(losses, invalid)
