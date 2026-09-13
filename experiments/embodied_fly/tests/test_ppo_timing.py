import numpy as np
import torch
from test_motor_focus import tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.hover_ppo import HoverPPOReward, HoverPPOTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import advantages
from embodied_fly.ppo_timing import physical_timescales, recurrent_forward


def test_recomputation_preserves_recurrent_outputs_gradients_and_episode_boundaries():
    torch.manual_seed(718)
    observations = torch.randn(64, 2, 397)
    a, b = tiny_brain(), tiny_brain()
    a.set_motor_only()
    b.set_motor_only()
    b.load_state_dict(a.state_dict())
    actions = []
    for brain, recompute in ((a, False), (b, True)):
        state, outputs = brain.initial_state(2), []
        for t, obs in enumerate(observations):
            result = recurrent_forward(brain, obs, state, torch.ones(2).long(), 1, recompute)
            state = brain.reset_worlds(result.state, torch.tensor([t == 25, False]))
            outputs.append(result.action)
        stacked = torch.stack(outputs)
        (stacked.square().mean() + state.square().mean()).backward()
        actions.append(stacked.detach())
    torch.testing.assert_close(*actions, rtol=0, atol=0)
    for (name, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
        else:
            assert left.grad is right.grad is None


def test_flight_timing_retains_delayed_advantage_and_stops_at_real_resets():
    time = physical_timescales(0.002, 0.999, 0.995, 512, 128)
    assert time["rollout_seconds"] == 1.024
    assert time["recurrent_gradient_seconds"] == 0.256
    assert 0.3 < time["gae_trace_efold_seconds"] < 0.34
    rewards, done = torch.zeros(128, 2), torch.zeros(128, 2, dtype=torch.bool)
    rewards[100] = 1
    done[50, 1] = True
    advantage, _ = advantages(
        rewards, torch.zeros_like(rewards), torch.zeros(2), done, 0.999, 0.995
    )
    assert advantage[0, 0] > 0.5
    assert advantage[0, 1] == 0


def test_hover_physical_cost_prefers_quiet_recovery_and_never_controls_the_body():
    env = FlyBatch(4, 2, 14, preset="wing_position")
    tasks = HoverPPOTasks(env, 712)
    reward = HoverPPOReward(env)
    original = {k: v.copy() for k, v in env.fields.items()}
    r, failed, terms = reward(env.previous_action)
    for key, value in original.items():
        np.testing.assert_array_equal(env.fields[key], value)
    np.testing.assert_allclose(r, sum(terms.values()) * env.control_dt - failed, atol=1e-6)
    initial = r[2]
    env.fields["qvel"][2, 2] = 5
    moving = reward(env.previous_action)[0][2]
    env.fields["qvel"][2, 2] = 10
    faster = reward(env.previous_action)[0][2]
    assert initial > moving > faster
    env.fields["qvel"][2, 2] = 0
    env.requested_height_cm[2] += 0.5
    # The reward tracks the actual command, not a target frozen at reset.
    assert reward(env.previous_action)[0][2] < initial
    env.fields["qpos"][2, 2] = 0.4
    r, failed, terms = reward(env.previous_action)
    assert failed[2]
    np.testing.assert_allclose(r, sum(terms.values()) * env.control_dt - failed, atol=1e-6)
    assert tasks.task_ids.tolist() == [0, 1, 2, 2]


def test_hover_reset_curriculum_keeps_the_same_body_and_untouched_other_worlds():
    env = FlyBatch(8, 2, 14, preset="wing_position")
    contract = physical_contract(env.model)
    tasks = HoverPPOTasks(env, 720)
    tasks.widening = 1
    before = {k: v[:7].copy() for k, v in env.fields.items()}
    tasks.reset([7])
    for key, value in before.items():
        np.testing.assert_array_equal(env.fields[key][:7], value)
    assert physical_contract(env.model) == contract
    assert abs(env.fields["qpos"][7, 2] - env.requested_height_cm[7]) <= 0.2
    assert abs(env.fields["qvel"][7, 2]) <= 5
    assert 1.8 <= env.requested_height_cm[7] <= 2.2
    assert env.ages[7] == 0 and not env.wing_forces.activity[7].any()
