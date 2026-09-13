import json
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from test_motor_focus import tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.motor_focus import MotorTasks
from embodied_fly.motor_outcome import GroundMotorReward
from embodied_fly.physical_contract import physical_contract


def test_rewards_measure_live_pose_without_fixing_walking_legs_or_writing_state():
    env = FlyBatch(2, 2, 14, preset="wing_motion")
    tasks = MotorTasks(env, 19, "ground")
    reward = GroundMotorReward(tasks)
    before = {k: v.copy() for k, v in env.fields.items()}
    _, _, original = reward(env.previous_action)
    for k, v in before.items():
        np.testing.assert_array_equal(env.fields[k], v)
    env.fields["qpos"][:, env.template.wing_angle_indices] += 0.2
    env.fields["qvel"][:, env.template.wing_velocity_indices] += 4
    env.fields["qpos"][:, env.template.qpos_indices[reward.posture.groups["legs"]]] += 0.2
    env.fields["qpos"][:, 2] *= 0.85
    total, failed, changed = reward(env.previous_action)
    assert np.all(changed["wing_pose"] < original["wing_pose"])
    assert np.all(changed["wing_stillness"] < original["wing_stillness"])
    assert changed["stand_leg_pose"][0] < original["stand_leg_pose"][0]
    assert changed["stand_leg_pose"][1] == original["stand_leg_pose"][1] == 0
    assert np.all(changed["body_height"] < original["body_height"])
    np.testing.assert_allclose(
        total, sum(changed.values()) * env.control_dt - failed, rtol=1e-6
    )


def test_motor_ppo_runs_physics_freezes_utility_and_saves_resumable_canonical_actor(
    tmp_path, monkeypatch
):
    from embodied_fly import ppo

    brain = tiny_brain()
    brain.set_motor_only()
    frozen = {
        k: v.clone()
        for k, v in brain.state_dict().items()
        if k.startswith(("utility_head.", "intention_encoder."))
    }
    env = FlyBatch(2, 2, 14, preset="wing_motion")
    parent = {"graph_sha256": "test", "physical_contract": physical_contract(env.model)}
    monkeypatch.setattr(ppo, "load_actor", lambda *args: (brain, parent))
    resume = tmp_path / "parent.pt"
    resume.write_bytes(b"test parent")
    graph = tmp_path / "graph"
    graph.mkdir()
    (graph / "brain.npz").write_bytes(b"test graph metadata")
    args = SimpleNamespace(
        resume=resume,
        graph=graph,
        output=tmp_path / "run",
        device="cpu",
        motor_ground=True,
        preset="wing_motion",
        rehearsal=None,
        flight_resets=None,
        worlds=2,
        threads=2,
        horizon=8,
        sequence=4,
        epochs=1,
        seconds=0.001,
        gamma=0.998,
        gae_lambda=0.99,
        lr=1e-5,
        noise=0.02,
        target_kl=0.03,
        entropy=0.001,
        stationary_cost=0,
        stationary_turn_cost=0,
        episode_seconds=0.004,
        seed=81001,
    )
    report = ppo.train(args)
    assert report["transitions"] == 16 and report["ppo_updates"] > 0
    assert report["active_motor_channels"] == 78
    assert report["rehearsal_frames"] == report["rehearsal_seconds"] == 0
    assert report["teacher_present_during_collection"] is False
    assert report["utility_and_intention_weights_unchanged"]
    assert report["worlds_by_task"] == {"stand": 1, "walk": 1, "hover": 0}
    assert len(report["completed_episodes"]) == 8
    assert all(
        x["l2"] > 0 and x["finite"]
        for x in report["core_gradient_audit_from_physical_reward"].values()
    )
    checkpoint = torch.load(args.output / "actor.pt", weights_only=True)
    assert checkpoint["motor_only"]
    assert all(torch.equal(checkpoint["state_dict"][k], v) for k, v in frozen.items())
    model = mujoco.MjModel.from_binary_path(str(args.output / "model.mjb"))
    assert checkpoint["physical_contract"] == physical_contract(model)
    json.dumps(report, allow_nan=False)
    # The second run must consume the optimizer and critic from the first, while
    # retaining exactly the same physical identity and inactive utility weights.
    parent.update(checkpoint)
    args.resume, args.output = args.output / "actor.pt", tmp_path / "continuation"
    continued = ppo.train(args)
    assert continued["optimizer_resumed"]
    assert continued["utility_and_intention_weights_unchanged"]
