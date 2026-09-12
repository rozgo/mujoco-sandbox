import torch
from test_brain import make_brain

from embodied_fly.ppo import advantages, joint_log_probability, motor_distribution


def test_recurrent_sample_replay_has_unit_ratio_and_reward_gradient_reaches_core():
    brain = make_brain().train()
    active = torch.tensor([True, False, True])
    log_std = torch.full((2,), -3.0)
    obs = torch.randn(8, 3, 4)
    memory = brain.initial_state(3)
    samples = []
    done = torch.zeros(8, 3, dtype=torch.bool)
    done[3, 1] = True
    with torch.no_grad():
        for t in range(8):
            result = brain(obs[t], memory, sample_activity=True)
            dist = motor_distribution(result, active, log_std)
            latent = dist.sample()
            probability = joint_log_probability(result, dist, latent, result.activity)
            samples.append((latent, result.activity, probability))
            memory = brain.reset_worlds(result.state, done[t])
    memory = brain.initial_state(3)
    probabilities = []
    for t, (latent, activity, old) in enumerate(samples):
        result = brain(obs[t], memory, activity_override=activity)
        dist = motor_distribution(result, active, log_std)
        new = joint_log_probability(result, dist, latent, activity)
        torch.testing.assert_close(new, old)
        probabilities.append(new)
        memory = brain.reset_worlds(result.state, done[t])
    loss = -(torch.stack(probabilities) * torch.randn(8, 3)).mean()
    loss.backward()
    for parameter in brain.core.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_gae_resets_do_not_leak_and_timeouts_use_terminal_state_bootstrap():
    # First world fails at t=0. Second is truncated there, so its reward already
    # contains gamma * V(final observation), not V(next episode reset state).
    rewards = torch.tensor([[1.0, 1.0 + 0.9 * 4.0], [100.0, 100.0]])
    values = torch.tensor([[2.0, 2.0], [9.0, 9.0]])
    done = torch.tensor([[True, True], [False, False]])
    advantage, returns = advantages(
        rewards, values, torch.tensor([10.0, 10.0]), done, 0.9, 1.0
    )
    torch.testing.assert_close(advantage[0], torch.tensor([-1.0, 2.6]))
    torch.testing.assert_close(returns[0], torch.tensor([1.0, 4.6]))
    torch.testing.assert_close(returns[1], torch.tensor([109.0, 109.0]))
