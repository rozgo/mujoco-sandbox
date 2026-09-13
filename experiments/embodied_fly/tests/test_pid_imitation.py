from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy import sparse

from embodied_fly.batch import FlyBatch
from embodied_fly.body import FlyEnvironment
from embodied_fly.brain import EmbodiedBrain
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.motion_flight import initialize
from embodied_fly.physical_contract import physical_contract
from embodied_fly.pid_hover import HoverPID
from embodied_fly.pid_imitation import BatchPID, imitation_loss, teacher_fraction


def test_batched_teacher_matches_accepted_single_pid_without_writing_live_world():
    batch = FlyBatch(
        2, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000
    )
    tasks = HoverOnlyTasks(batch, 8)
    teacher = BatchPID(batch, tasks)
    single = FlyEnvironment("wing_position", wing_response="instant", physics_hz=1000)
    base = initialize(single, 1.86651647)
    reference = HoverPID(single, base, single.data.qpos[:3])
    for _ in range(100):
        before = {k: v.copy() for k, v in batch.fields.items()}
        targets = teacher.act()
        for k, v in before.items():
            np.testing.assert_array_equal(batch.fields[k], v)
        # Match the comparison renderer's current-state scratch feedback, rather
        # than MuJoCo's pre-integration derived fields left by mj_step.
        mujoco.mj_forward(single.model, single.data)
        np.testing.assert_allclose(targets[0], reference.act(), atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(targets[0], targets[1])
        single.step(targets[0])
        batch.step(targets)
    np.testing.assert_allclose(single.data.qpos[:7], batch.fields["qpos"][0, :7], atol=1e-7)
    np.testing.assert_allclose(single.data.qpos, batch.fields["qpos"][0], atol=1e-5)
    assert single.substeps == 2 and batch.substeps == 2


def test_handoff_ends_with_student_and_wing_loss_has_body_posture_gradient():
    np.testing.assert_allclose(
        [teacher_fraction(p) for p in (0, 0.5, 0.65, 0.8, 1)], [1, 1, 0.5, 0, 0]
    )
    actions = torch.ones(2, 78, requires_grad=True)
    loss, wing, posture = imitation_loss(actions, torch.zeros_like(actions), np.arange(14, 20))
    loss.backward()
    assert wing == 1 and posture == 1
    assert actions.grad[:, 14:20].mean() > actions.grad[:, :14].mean() > 0


def test_pid_imitation_checkpoint_preserves_graph_body_and_discards_ppo_optimizer(
    tmp_path, monkeypatch
):
    from embodied_fly import pid_imitation

    graph = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    brain = EmbodiedBrain(
        graph, [0, 1], [2, 3], [4, 5], 399, 78, sensor_extension_size=16, motor_only=True
    )
    env = FlyBatch(2, 1, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    parent = {
        "config": {"internal_steps": 4},
        "physical_contract": physical_contract(env.model),
        "motor_only": True,
        "sensor_extension_size": 16,
        "observation_size": 399,
        "action_size": 78,
        "graph_sha256": "fixture",
        "optimizer_state_dict": {"stale": True},
        "critic_state_dict": {"stale": True},
        "value_optimizer_state_dict": {"stale": True},
    }
    monkeypatch.setattr(pid_imitation, "load_actor", lambda *args: (brain, parent))
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"fixture")
    args = SimpleNamespace(
        resume=checkpoint,
        graph=tmp_path,
        output=tmp_path / "run",
        device="cpu",
        seconds=0.02,
        worlds=2,
        threads=1,
        sequence=4,
        episode_seconds=0.05,
        lr=0.0001,
        seed=120401,
    )
    report = pid_imitation.train(args)
    saved = torch.load(args.output / "actor.pt", weights_only=True)
    assert saved["physical_contract"] == parent["physical_contract"]
    assert saved["graph_sha256"] == "fixture" and saved["motor_only"]
    assert all(
        k not in saved
        for k in ("optimizer_state_dict", "critic_state_dict", "value_optimizer_state_dict")
    )
    assert "imitation_optimizer_state_dict" in saved
    assert report["transitions"] > 0 and report["updates"] > 0
    assert not report["actor_receives_teacher_clock_or_integral"]
    assert all(v["finite"] and v["l2"] > 0 for v in report["core_gradient_audit"].values())
