import numpy as np
import torch
from test_motor_focus import TEACHER, tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.hover_feedback import examples, response_loss
from embodied_fly.motor_focus import MotorTasks, MotorTeacher


def test_counterfactual_reference_matches_teacher_and_preserves_physical_state():
    env = FlyBatch(3, 3, 14, preset="wing_position")
    tasks = MotorTasks(env, 99203)
    teacher = MotorTeacher(tasks, TEACHER, "cpu", True, "state", True)
    for _ in range(12):
        env.batch.forward()
        before = {k: v.copy() for k, v in env.fields.items()}
        obs = env.observation()
        x, y = examples(
            env.model, obs[2], *[env.fields[k][2] for k in ("qpos", "qvel", "act", "ctrl")]
        )
        np.testing.assert_allclose(y[0], teacher.act()[2, teacher.channels], atol=1e-6)
        np.testing.assert_array_equal(x[:, 297:375], np.repeat(obs[2:3, 297:375], 5, axis=0))
        assert all(np.array_equal(v, env.fields[k]) for k, v in before.items())
        assert np.isfinite(y).all() and abs(y).max() <= 1
        env.step(teacher.act())
    actor = tiny_brain()
    actor.set_motor_only()
    memory = actor.initial_state(3)
    original = memory.clone()
    before = {k: v.copy() for k, v in env.fields.items()}
    loss, count = response_loss(
        actor, memory, env, env.observation(), tasks.task_ids, 8, np.random.default_rng(99)
    )
    loss.backward()
    assert count == 2 and torch.isfinite(loss)
    assert actor.sensor_extension.weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in actor.core.parameters())
    assert all(p.grad is None for p in actor.utility_head.parameters())
    assert torch.equal(memory, original)
    assert all(np.array_equal(v, env.fields[k]) for k, v in before.items())
