from types import SimpleNamespace

import numpy as np
import torch
from test_motor_focus import TEACHER, tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.motor_curriculum import GroundCommandCurriculum
from embodied_fly.motor_focus import MotorTasks, MotorTeacher


def test_command_change_preserves_body_history_and_uses_correct_label_source():
    env = FlyBatch(9, 3, 14, preset="wing_position")
    tasks = MotorTasks(env, 99303)
    teacher = MotorTeacher(tasks, TEACHER, "cpu", True, "state", True)
    curriculum = GroundCommandCurriculum(tasks, 2, 0.004)
    actor = tiny_brain()
    actor.set_motor_only()
    memory = actor.initial_state(9)
    for _ in range(2):
        output = actor(torch.as_tensor(env.observation()), memory)
        memory = output.state.detach()
        env.step(output.action.detach().numpy())
    before = {k: v.copy() for k, v in env.fields.items()}
    old_memory = memory.clone()
    old_actions = env.previous_action.copy()
    changed = curriculum.advance(memory)
    np.testing.assert_array_equal(changed, [0, 1])
    assert all(np.array_equal(v, env.fields[k]) for k, v in before.items())
    np.testing.assert_array_equal(env.previous_action, old_actions)
    assert torch.equal(memory, old_memory)
    np.testing.assert_array_equal(tasks.task_ids[:3], [1, 0, 2])
    np.testing.assert_allclose(env.observation()[:2, 375], [0.1, 0])
    parent = np.full((9, 78), 0.123, dtype=np.float32)
    original_parent = parent.copy()
    actual = curriculum.supervised_actions(parent, teacher)
    labels = teacher.posture.targets(teacher.ground.act().numpy(), tasks.task_ids)
    np.testing.assert_array_equal(actual[:2], labels[:2])
    np.testing.assert_array_equal(actual[2:], parent[2:])
    np.testing.assert_array_equal(parent, original_parent)
    curriculum.prepare_reset([0])
    assert tasks.task_ids[0] == 0 and tasks.task_ids[1] == 0
    assert len(curriculum.events) == 2
    assert all(e["neural_state_l2_at_boundary"] > 0 for e in curriculum.events)


def test_training_collects_command_changes_without_teacher_actions(tmp_path, monkeypatch):
    from embodied_fly import motor_focus

    actor = tiny_brain()
    actor.set_motor_only()
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"test")
    graph = tmp_path / "graph"
    graph.mkdir()
    (graph / "brain.npz").write_bytes(b"test metadata")
    monkeypatch.setattr(
        motor_focus, "load_motor_actor", lambda *a: (actor, {"graph_sha256": "test"})
    )
    report = motor_focus.train(
        SimpleNamespace(
            output=tmp_path / "run",
            device="cpu",
            worlds=9,
            threads=3,
            sequence=4,
            seconds=0.01,
            lr=1e-5,
            teacher_mix=0.0,
            resume=parent,
            graph=graph,
            teacher=TEACHER,
            seed=99303,
            episode_seconds=0.008,
            preset="wing_position",
            ground_posture=True,
            stand_initial_form=True,
            hover_reference="state",
            retain_ground=True,
            ground_retention_weight=4.0,
            nonwing_retention_weight=4.0,
            ground_wing_loss=10.0,
            ground_switch_seconds=0.004,
            transition_worlds=2,
        )
    )
    c = report["command_curriculum"]
    assert c["transition_worlds"] == 2
    assert c["fixed_rehearsal_worlds"] == {"stand": 2, "walk": 2, "hover": 3}
    assert len(c["command_events"]) == 2 * report["updates"]
    assert all(not e["physical_or_neural_reset"] for e in c["command_events"])
    assert report["ground_retention"]["weights_unchanged"]
    assert report["utility_and_intention_weights_unchanged"]
    assert all(e["teacher_mix"] == 0 for e in report["completed_episodes"])
    assert (
        sum(e["command_transition_world"] for e in report["completed_episodes"])
        == 2 * report["updates"]
    )
