import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "wing_weight,all_motor,warmup,physical,independent,force_kl_stop,response",
    [
        (0.0, False, 0, False, False, False, "filtered"),
        (100.0, False, 0, False, False, False, "filtered"),
        (0.0, True, 0, False, False, False, "filtered"),
        (0.0, True, 1, False, False, False, "filtered"),
        (0.0, True, 0, True, False, False, "filtered"),
        (0.0, True, 0, True, True, False, "filtered"),
        (0.0, True, 0, True, True, True, "filtered"),
        (0.0, True, 1, True, True, False, "filtered"),
        (0.0, True, 0, True, True, False, "instant"),
        (0.0, True, 1, True, True, False, "hover-only"),
    ],
)
def test_motor_ppo_runs_physics_freezes_utility_and_saves_resumable_canonical_actor(
    tmp_path,
    monkeypatch,
    wing_weight,
    all_motor,
    warmup,
    physical,
    independent,
    force_kl_stop,
    response,
):
    from embodied_fly import ppo

    hover_only = response == "hover-only"
    if hover_only:
        response = "instant"

    if force_kl_stop:
        original_logp = ppo.joint_log_probability

        def shifted_update_logp(*args, **kwargs):
            result = original_logp(*args, **kwargs)
            # Force actor KL rejection on the first replayed chunk, while
            # leaving all physical collection probabilities untouched.
            return result + 2 if torch.is_grad_enabled() else result

        monkeypatch.setattr(ppo, "joint_log_probability", shifted_update_logp)

    brain = tiny_brain()
    if hover_only:
        from embodied_fly.brain import initialize_extended_actor

        state = brain.state_dict()
        brain.observation_size = 399
        brain.sensor_extension_size = 16
        brain.sensor_extension = torch.nn.Linear(16, 128, bias=False)
        brain.observation_mean = torch.zeros(399)
        brain.observation_std = torch.ones(399)
        initialize_extended_actor(brain, state)
    brain.set_motor_only()
    if all_motor:
        brain.enable_wing_residual(8)
    frozen = {
        k: v.clone()
        for k, v in brain.state_dict().items()
        if k.startswith(("utility_head.", "intention_encoder."))
    }
    initial_actor = {k: v.clone() for k, v in brain.state_dict().items()}
    worlds = 3 if all_motor else 2
    if physical:
        worlds = 4
    preset = "wing_position" if all_motor else "wing_motion"
    env = FlyBatch(
        worlds,
        2,
        brain.sensor_extension_size,
        preset=preset,
        wing_response=response,
        physics_hz=1000 if hover_only else None,
    )
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
        motor_ground=not all_motor,
        motor_all=all_motor,
        motor_retention_weight=4.0 if all_motor and not physical else 0.0,
        hover_physical=physical,
        hover_only=hover_only,
        critic_lr=1e-4,
        checkpoint_activations=physical,
        independent_critic=independent,
        critic_warmup_rollouts=warmup,
        wing_supervision=wing_weight,
        preset=preset,
        rehearsal=None,
        flight_resets=None,
        worlds=worlds,
        threads=2,
        horizon=8,
        sequence=4,
        epochs=2 if independent else 1,
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
    assert report["transitions"] == worlds * 8
    assert report["critic_updates"] > 0
    assert (report["ppo_updates"] == 0) == bool(warmup or force_kl_stop)
    if warmup or force_kl_stop:
        assert all(torch.equal(brain.state_dict()[k], v) for k, v in initial_actor.items())
        assert report["core_gradient_audit_from_physical_reward"] is None
        assert all(v == 0 for v in report["core_changes"].values())
    if independent:
        assert report["critic_schedule"]["independent_of_actor_kl"]
        assert report["critic_updates"] == 4
        assert report["critic_sample_presentations"] == worlds * 8 * 2
        assert not report["critic_schedule"]["actor_or_physics_replay_for_value_fit"]
    if force_kl_stop:
        assert report["actor_kl_stops"] == 1
        assert report["critic_updates_after_actor_stop"] == 4
    assert report["active_motor_channels"] == 78
    assert report["rehearsal_frames"] == report["rehearsal_seconds"] == 0
    assert report["teacher_present_during_collection"] == bool(
        wing_weight or (all_motor and not physical)
    )
    assert not report["executed_teacher_actions"]
    assert report["wing_supervision"]["weight"] == wing_weight
    assert report["wing_supervised_presentations"] == (
        report["ppo_updates"] * 8 if wing_weight else 0
    )
    if wing_weight:
        assert all(
            x["finite"] and x["l2"] > 0
            for x in report["core_gradient_audit_from_weighted_wing_supervision"].values()
        )
    assert report["utility_and_intention_weights_unchanged"]
    assert report["worlds_by_task"] == (
        {"stand": 0, "walk": 0, "hover": 4}
        if hover_only
        else {
            "stand": 1,
            "walk": 1,
            "hover": 2 if physical else int(all_motor),
        }
    )
    assert sum(report["transitions_by_task"].values()) == report["transitions"]
    assert len(report["completed_episodes"]) == worlds * 4
    assert (
        warmup
        or force_kl_stop
        or all(
            x["l2"] > 0 and x["finite"]
            for x in report["core_gradient_audit_from_physical_reward"].values()
        )
    )
    checkpoint = torch.load(args.output / "actor.pt", weights_only=True)
    assert checkpoint["motor_only"] and checkpoint["config"]["ground_posture"]
    assert all(torch.equal(checkpoint["state_dict"][k], v) for k, v in frozen.items())
    model = mujoco.MjModel.from_binary_path(str(args.output / "model.mjb"))
    assert checkpoint["physical_contract"] == physical_contract(model)
    if all_motor:
        assert checkpoint["wing_residual_enabled"] and checkpoint["wing_residual_hidden"] == 8
        assert report["motor_retention"]["weights_unchanged"] == (not physical)
        assert not report["motor_retention"]["runtime_module"]
        restored = tiny_brain()
        if hover_only:
            restored.observation_size = 399
            restored.sensor_extension_size = 16
            restored.sensor_extension = torch.nn.Linear(16, 128, bias=False)
            restored.observation_mean = torch.zeros(399)
            restored.observation_std = torch.ones(399)
        restored.enable_wing_residual(checkpoint["wing_residual_hidden"])
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        # The physical reward gradient was audited before adding the retention loss.
        assert (
            warmup
            or force_kl_stop
            or physical
            or report["core_gradient_audit_from_ground_retention"] is not None
        )
    json.dumps(report, allow_nan=False)
    # The second run must consume the optimizer and critic from the first, while
    # retaining exactly the same physical identity and inactive utility weights.
    parent.update(checkpoint)
    args.resume, args.output = args.output / "actor.pt", tmp_path / "continuation"
    continued = ppo.train(args)
    assert continued["optimizer_resumed"]
    assert continued["utility_and_intention_weights_unchanged"]


def test_all_motor_rewards_select_task_once_and_do_not_move_physics():
    from embodied_fly.motor_outcome import AllMotorReward

    env = FlyBatch(3, 3, 14, preset="wing_position")
    tasks = MotorTasks(env, 98001)
    reward = AllMotorReward(tasks)
    reward.reset(np.arange(3))
    before = {k: v.copy() for k, v in env.fields.items()}
    original, failed, terms = reward(env.previous_action)
    for key, value in before.items():
        np.testing.assert_array_equal(env.fields[key], value)
    np.testing.assert_allclose(original, sum(terms.values()) * env.control_dt - failed)
    assert all(value[2] == 0 for key, value in terms.items() if key.startswith("ground/"))
    assert all(not value[:2].any() for key, value in terms.items() if key.startswith("hover/"))
    # A moving hover is worse than a stationary hover at the same height.
    env.fields["qvel"][2, 0] = 0.5
    drifting, _, _ = reward(env.previous_action)
    assert drifting[2] < original[2]
    # Hover failure uses the declared 0.5 cm floor and incurs one -1 penalty.
    env.fields["qpos"][2, 2] = 0.4
    total, failed, terms = reward(env.previous_action)
    assert failed.tolist() == [False, False, True]
    np.testing.assert_allclose(total, sum(terms.values()) * env.control_dt - failed, rtol=1e-6)


def test_wing_corrective_labels_restore_angle_brake_speed_and_do_not_move_body():
    env = FlyBatch(2, 2, 14, preset="wing_motion")
    tasks = MotorTasks(env, 19, "ground")
    posture = GroundMotorReward(tasks).posture
    np.testing.assert_allclose(posture.wing_targets(), 0, atol=1e-12)
    env.fields["qpos"][0, env.template.wing_angle_indices] += 0.1
    env.fields["qvel"][1, env.template.wing_velocity_indices] += 2
    state = {k: v.copy() for k, v in env.fields.items()}
    target = posture.wing_targets()
    np.testing.assert_allclose(target[0], -0.02 * 0.1 / 0.03, rtol=1e-6)
    np.testing.assert_allclose(target[1], -0.00015 * 2 / 0.03, rtol=1e-6)
    for k, v in state.items():
        np.testing.assert_array_equal(env.fields[k], v)
    np.testing.assert_array_equal(posture.wing_targets([1]), target[1:])
    env.fields["qvel"][:, env.template.wing_velocity_indices] = -1000
    assert (posture.wing_targets() == 1).all()
