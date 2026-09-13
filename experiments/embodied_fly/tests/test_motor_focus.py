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


@pytest.mark.parametrize(
    "task_set,retain_ground,subset",
    [
        ("all", False, "all"),
        ("ground", False, "all"),
        ("all", True, "all"),
        ("all", True, "wing-output"),
        ("all", True, "wing-residual"),
        ("all", True, "wing-feedback"),
    ],
)
def test_motor_training_freezes_intentions_and_saves_the_shared_physics(
    tmp_path, monkeypatch, task_set, retain_ground, subset
):
    from embodied_fly import motor_focus
    from embodied_fly.provenance import sha256

    actor = tiny_brain()
    actor.set_motor_only()
    if subset in ("wing-residual", "wing-feedback"):
        actor.enable_wing_residual(8)
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
        trainable_subset=subset,
        feedback_lr=0.003,
        nonwing_retention_weight=4.0 if subset == "wing-feedback" else 0.0,
        teacher_mix=0.0 if retain_ground else 1.0,
        hover_teacher_mix=0.8 if retain_ground else None,
        hover_reference="state" if retain_ground else "clock",
        retain_ground=retain_ground,
        ground_retention_weight=4.0 if retain_ground else 1.0,
        ground_posture=True,
        ground_wing_loss=10.0,
        wing_response_loss=1.0,
        wing_response_worlds=2,
        episode_seconds=0.004 if retain_ground else 2.0,
        seed=7,
        resume=resume,
        graph=graph,
        teacher=TEACHER,
        task_set=task_set,
        wing_angle_perturbation=0.15 if task_set == "ground" else 0.0,
        wing_speed_perturbation=2.0 if task_set == "ground" else 0.0,
    )
    report = motor_focus.train(args)
    checkpoint = torch.load(args.output / "actor.pt", weights_only=True)
    assert checkpoint["motor_only"] and report["utility_and_intention_weights_unchanged"]
    assert report["worlds_by_task"] == (
        {"stand": 1, "walk": 1, "hover": 1}
        if task_set == "all"
        else {"stand": 2, "walk": 1, "hover": 0}
    )
    assert report["training_tasks"] == (
        ["stand", "walk", "hover"] if task_set == "all" else ["stand", "walk"]
    )
    assert report["ground_posture"]["runtime_override"] is False
    assert checkpoint["config"]["ground_posture"] is True
    assert report["wing_response_supervision"]["extra_physics_worlds"] == 0
    assert checkpoint["wing_response_supervision"]["weight"] == 1
    assert report["ground_wing_loss_weight"] == 10
    assert report["ground_retention"]["enabled"] == retain_ground
    assert report["parameter_subset"]["mode"] == subset
    if subset in ("wing-output", "wing-residual", "wing-feedback"):
        assert report["core_gradient_audit"] is None
        assert report["subset_gradient_audit"]
        assert report["parameter_subset"]["upstream_parameters_and_buffers_unchanged"] == (
            subset != "wing-feedback"
        )
        assert report["parameter_subset"]["nonwing_output_rows_unchanged"]
        if subset != "wing-feedback":
            assert report["parameter_subset"]["same_history_nonwing_action_max_delta"] < 1e-5
        else:
            assert (
                report["parameter_subset"]["selected_parameter_changes_l2"][
                    "sensor_extension.weight"
                ]
                > 0
            )
            assert (
                report["subset_gradient_audit"]["sensor_extension.weight"]["gradient_l2"] > 0
            )
            assert report["ground_retention"]["additional_nonwing_mse_weight"] == 4
        assert (
            report["parameter_subset"]["same_history_world_actions_checked"]
            == report["transitions"]
        )
        assert not any(report["core_parameter_changes"].values())
    if retain_ground:
        assert report["ground_retention"]["weights_unchanged"]
        assert report["ground_retention"]["checkpoint_sha256"] == sha256(resume)
        assert report["ground_retention"]["runtime_module"] is False
        assert checkpoint["config"]["state_hover_recipe"]["clock_or_external_phase"] is False
        assert len(report["completed_episodes"]) == 3
        for ep in report["completed_episodes"]:
            assert ep["teacher_mix"] == pytest.approx(0.8 if ep["task"] == "hover" else 0)
    assert report["transitions"] >= 6
    assert report["checkpoint_sha256"] == sha256(args.output / "actor.pt")
    assert all(torch.equal(checkpoint["state_dict"][k], v) for k, v in fixed.items())
    if subset in ("wing-residual", "wing-feedback"):
        assert checkpoint["wing_residual_enabled"]
        assert checkpoint["wing_residual_hidden"] == 8
        restored = tiny_brain()
        restored.enable_wing_residual(8)
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
    loaded = mujoco.MjModel.from_binary_path(str(args.output / "model.mjb"))
    assert checkpoint["physical_contract"] == physical_contract(loaded)
    json.dumps(report, allow_nan=False)
    for line in (args.output / "progress.jsonl").read_text().splitlines():
        row = json.loads(line)
        json.dumps(row, allow_nan=False)
        if task_set == "ground":
            assert row["task_motor_loss"]["hover"] is None
            assert row["posture_by_task"]["hover"] is None


def test_ground_reset_disturbances_change_only_wings_and_preserve_nominal_target_and_body():
    a, b = (FlyBatch(6, 2, 14, preset="wing_motion") for _ in range(2))
    reference = MotorTasks(a, 72009, "ground")
    disturbed = MotorTasks(b, 72009, "ground", 0.15, 2)
    assert np.array_equal(disturbed.task_ids, [0, 1, 0, 1, 0, 1])
    assert physical_contract(a.model) == physical_contract(b.model)
    np.testing.assert_array_equal(reference.ground["qpos"], disturbed.ground["qpos"])
    t = b.template
    other = np.setdiff1d(np.arange(b.model.nq), t.wing_angle_indices)
    np.testing.assert_array_equal(a.fields["qpos"][:, other], b.fields["qpos"][:, other])
    difference = (
        b.fields["qpos"][:, t.wing_angle_indices] - a.fields["qpos"][:, t.wing_angle_indices]
    )
    assert np.abs(difference).max() <= 0.150001 and np.abs(difference).max() > 0.05
    assert np.abs(b.fields["qvel"][:, t.wing_velocity_indices]).max() <= 2
    for column, joint in enumerate(t.wing_joint_ids):
        if b.model.jnt_limited[joint]:
            q = b.fields["qpos"][:, t.wing_angle_indices[column]]
            assert (q >= b.model.jnt_range[joint, 0]).all() and (
                q <= b.model.jnt_range[joint, 1]
            ).all()
    retained = b.fields["qpos"][2:].copy()
    disturbed.reset([0, 1])
    np.testing.assert_array_equal(b.fields["qpos"][2:], retained)
    np.testing.assert_array_equal(disturbed.ground["qpos"], reference.ground["qpos"])


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
