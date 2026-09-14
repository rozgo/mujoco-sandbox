import numpy as np
import pytest
import torch

from embodied_fly.recovery_starts import capture, restore
from embodied_fly.velocity_demonstrations import environment, initialize_worlds
from embodied_fly.velocity_hover import HoverReward
from embodied_fly.velocity_motor import observation


def test_full_state_and_reward_ring_restore_reproduce_continuation_and_isolate_worlds():
    env = environment(3, 3)
    action = initialize_worlds(env, [0, 0.1, -0.1])
    reward = HoverReward(env, 2)
    memory = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    rng = np.random.default_rng(1643)
    actions = np.clip(action + rng.normal(0, 0.001, (83, 3, env.model.nu)), -1, 1)
    for a in actions[:53]:
        env.step(a)
        reward()
    saved = capture(env, reward, memory)
    obs = observation(env, np.zeros((3, 4)))
    reference = []
    for a in actions[53:]:
        env.step(a)
        score, _, _ = reward()
        reference.append((env.fields["qpos"].copy(), score.copy(), reward.mean().copy()))
    # Different ring index and unrelated live worlds must be retained.
    before = capture(env, reward, memory)
    ids = np.array([0, 2])
    selected = {k: v[ids] for k, v in saved.items()}
    restore(env, reward, memory, ids, selected)
    np.testing.assert_array_equal(observation(env, np.zeros((3, 4)))[ids], obs[ids])
    after = capture(env, reward, memory)
    for key in before:
        np.testing.assert_array_equal(after[key][1], before[key][1], err_msg=key)
    assert np.all(env.ages[ids] == 0)
    for a, (qpos, expected_reward, means) in zip(actions[53:], reference, strict=True):
        env.step(a)
        score, _, _ = reward()
        np.testing.assert_allclose(env.fields["qpos"][ids], qpos[ids], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(score[ids], expected_reward[ids])
        np.testing.assert_allclose(reward.mean()[ids], means[ids], atol=1e-12, rtol=0)


def test_bad_world_mapping_is_rejected_before_restoration():
    env = environment(2, 2)
    reward = HoverReward(env, 2)
    memory = torch.zeros(7, 2)
    saved = capture(env, reward, memory)
    for ids in ([0, 0], [-1, 1], [0, 2]):
        with pytest.raises(ValueError):
            restore(env, reward, memory, ids, saved)


def test_recurrent_replay_uses_actual_recovery_memory_inside_sequences():
    from test_brain import tiny_graph

    from embodied_fly.brain import EmbodiedBrain
    from embodied_fly.ppo import joint_log_probability, motor_distribution
    from embodied_fly.velocity_hover_ppo import replay_audit

    torch.manual_seed(5123)
    actor = EmbodiedBrain(tiny_graph(), [0, 1], [2, 3], [4, 5], 4, 78)
    actor.set_motor_only(True)
    obs = torch.randn(8, 3, 4)
    state = actor.initial_state(3)
    initial = state.clone()
    active, std = torch.ones(78, dtype=torch.bool), torch.full((78,), -6.5)
    data = {k: [] for k in ("obs", "latent", "mean_action", "logp", "done")}
    resets = {}
    with torch.no_grad():
        for t in range(8):
            result = actor(obs[t], state)
            dist = motor_distribution(result, active, std, 0.0005)
            latent = dist.sample()
            done = torch.tensor([t == 2, False, t == 5])
            for k, v in {
                "obs": obs[t],
                "latent": latent,
                "mean_action": result.action,
                "logp": joint_log_probability(result, dist, latent, None, motor_only=True),
                "done": done,
            }.items():
                data[k].append(v)
            state = actor.reset_worlds(result.state, done)
            if done.any():
                ids = torch.where(done)[0]
                restored = torch.rand(state.shape[0], len(ids)) * 0.2
                state[:, ids] = restored
                resets[t] = (ids, restored.clone())
    data = {k: torch.stack(v) for k, v in data.items()}
    with pytest.raises(RuntimeError):
        replay_audit(actor, data, [initial], 8, active, std)
    data["reset_memories"] = resets
    assert replay_audit(actor, data, [initial], 8, active, std)["passed"]
