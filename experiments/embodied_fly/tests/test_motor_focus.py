import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scipy import sparse

from embodied_fly.batch import FlyBatch
from embodied_fly.body import FlyEnvironment
from embodied_fly.brain import EmbodiedBrain
from embodied_fly.motion_flight import WingReference
from embodied_fly.motor_focus import MotorTasks, MotorTeacher
from embodied_fly.physical_contract import physical_contract
from embodied_fly.teacher import TeacherOracle

TEACHER = Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers/walking.npz"


def test_ground_wing_accuracy_adds_gradient_without_changing_other_targets():
    from embodied_fly.motor_focus import motor_error

    x = torch.ones(3, 78, requires_grad=True)
    target = torch.zeros_like(x)
    flight = torch.tensor([False, False, True])
    wings = np.arange(14, 20)
    previous = motor_error(x, target, wings, flight)
    corrected = motor_error(x, target, wings, flight, 2.0)
    a = torch.autograd.grad(previous.sum(), x)[0]
    b = torch.autograd.grad(corrected.sum(), x)[0]
    torch.testing.assert_close(a[:, :14], b[:, :14])
    torch.testing.assert_close(a[:, 20:], b[:, 20:])
    torch.testing.assert_close(a[2], b[2])
    assert torch.all(b[:2, wings] > a[:2, wings])
    torch.testing.assert_close(motor_error(target, target, wings, flight, 2), torch.zeros(3))
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            motor_error(x, target, wings, flight, bad)


def tiny_brain():
    graph = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    return EmbodiedBrain(graph, [0, 1], [2, 3], [4, 5], 397, 78, sensor_extension_size=14)


def test_motor_context_matches_fixed_parent_and_cannot_be_changed_by_utilities():
    parent = tiny_brain().eval()
    actor = tiny_brain().eval()
    actor.load_state_dict(parent.state_dict())
    actor.set_motor_only()
    a, b = parent.initial_state(2), actor.initial_state(2)
    for _ in range(20):
        obs = torch.randn(2, 397)
        old = parent(obs, a, activity_override=torch.ones(2, dtype=torch.long))
        new = actor(obs, b, sample_activity=True)
        torch.testing.assert_close(new.action, old.action, atol=0, rtol=0)
        torch.testing.assert_close(new.state, old.state, atol=0, rtol=0)
        assert not new.utility_scores.any()
        a, b = old.state.detach(), new.state.detach()
    before = actor(obs, b)
    with torch.no_grad():
        for p in actor.utility_head.parameters():
            p.fill_(100)
    changed = actor(obs, b)
    torch.testing.assert_close(changed.action, before.action, atol=0, rtol=0)
    before.action.square().sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in actor.utility_head.parameters())
    assert all(
        p.grad is None and not p.requires_grad for p in actor.intention_encoder.parameters()
    )
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.core.parameters())
    with pytest.raises(ValueError, match="fixed context"):
        actor(obs, b, activity_override=torch.zeros(2, dtype=torch.long))


def test_canonical_physics_matches_native_batch_and_rejects_mechanical_wings():
    batch = FlyBatch(3, 3, 14, preset="wing_motion")
    native = batch.template
    before = physical_contract(batch.model)
    assert before == physical_contract(native.model)
    tasks = MotorTasks(batch, 71001)
    assert physical_contract(batch.model) == before
    np.testing.assert_array_equal(batch.requested_height_cm[:2], 0)
    assert batch.requested_height_cm[2] > 1.8
    retained = batch.fields["qpos"][2].copy()
    target = batch.requested_height_cm[2]
    tasks.reset([0, 1])
    np.testing.assert_array_equal(batch.fields["qpos"][2], retained)
    assert batch.requested_height_cm[2] == target
    changed = copy.copy(batch.model)
    wing = changed.body("walker/wing_left").id
    changed.body_mass[wing] = 1e-6
    with pytest.raises(ValueError, match="massless"):
        physical_contract(changed)
    changed = copy.copy(batch.model)
    geom = np.flatnonzero(changed.geom_bodyid == wing)[0]
    changed.geom_contype[geom] = 1
    with pytest.raises(ValueError, match="non-colliding"):
        physical_contract(changed)


def test_motor_references_match_single_world_teachers_without_pose_writes():
    batch = FlyBatch(3, 3, 14, preset="wing_motion")
    tasks = MotorTasks(batch, 71001)
    teacher = MotorTeacher(tasks, TEACHER, "cpu")
    single = FlyEnvironment("wing_motion")
    oracle = TeacherOracle(single, TEACHER)
    for _ in range(12):
        # Refresh both scratch views at the same integration time before
        # comparing the current-state reference adapters.
        batch.batch.forward()
        before = batch.fields["qpos"].copy()
        actual = teacher.act()
        np.testing.assert_array_equal(batch.fields["qpos"], before)
        for i in range(3):
            for name in ("qpos", "qvel", "act", "ctrl"):
                getattr(single.data, name)[:] = batch.fields[name][i]
            single.data.time = batch.ages[i] * batch.control_dt
            mujoco.mj_forward(single.model, single.data)
            single.mean_sensors = batch.mean_sensors[i, : single.model.nsensordata].copy()
            if i < 2:
                oracle.set_reference(i, 0, 2)
                expected = oracle.act(0, "receding")
            else:
                ref = WingReference(
                    single, tasks.air_action, height=batch.requested_height_cm[i]
                )
                expected = ref.act()
            np.testing.assert_allclose(actual[i], expected, atol=3e-5, rtol=3e-5)
        batch.step(actual)


def test_motor_training_freezes_intentions_and_saves_the_shared_physics(tmp_path, monkeypatch):
    from embodied_fly import motor_focus
    from embodied_fly.provenance import sha256

    actor = tiny_brain()
    actor.set_motor_only()
    fixed = {
        k: v.clone()
        for k, v in actor.state_dict().items()
        if k.startswith(("utility_head.", "intention_encoder."))
    }
    resume = tmp_path / "parent.pt"
    resume.write_bytes(b"test parent")
    graph = tmp_path / "graph"
    graph.mkdir()
    (graph / "brain.npz").write_bytes(b"test metadata")
    monkeypatch.setattr(
        motor_focus, "load_motor_actor", lambda *args: (actor, {"graph_sha256": "test"})
    )
    args = SimpleNamespace(
        output=tmp_path / "run",
        device="cpu",
        worlds=3,
        threads=3,
        sequence=2,
        seconds=0.01,
        lr=1e-5,
        teacher_mix=1.0,
        episode_seconds=2.0,
        seed=7,
        resume=resume,
        graph=graph,
        teacher=TEACHER,
    )
    report = motor_focus.train(args)
    checkpoint = torch.load(args.output / "actor.pt", weights_only=True)
    assert checkpoint["motor_only"] and report["utility_and_intention_weights_unchanged"]
    assert report["worlds_by_task"] == {"stand": 1, "walk": 1, "hover": 1}
    assert report["transitions"] >= 6
    assert report["checkpoint_sha256"] == sha256(args.output / "actor.pt")
    assert all(torch.equal(checkpoint["state_dict"][k], v) for k, v in fixed.items())
    loaded = mujoco.MjModel.from_binary_path(str(args.output / "model.mjb"))
    assert checkpoint["physical_contract"] == physical_contract(loaded)
    json.dumps(report)


def test_review_retains_early_forbidden_load_in_full_clip_gate(tmp_path, monkeypatch):
    from embodied_fly import motor_focus

    class EarlyContactBatch(FlyBatch):
        def step(self, action):
            super().step(action)
            self.forbidden_peak[:] = 3 * self.body_weight if self.ages[0] == 1 else 0

    monkeypatch.setattr(motor_focus, "FlyBatch", EarlyContactBatch)
    args = SimpleNamespace(
        output=tmp_path / "reference",
        mode="reference",
        teacher=TEACHER,
        device="cpu",
        seed=71001,
        seconds=0.004,
    )
    report = motor_focus.review(args)
    assert all(r["max_forbidden_ground_force_over_weight"] == 3 for r in report["results"])
    assert not any(r["success"] for r in report["results"])
