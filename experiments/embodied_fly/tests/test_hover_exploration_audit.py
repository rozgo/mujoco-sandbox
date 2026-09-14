from types import SimpleNamespace

import pytest
import torch

from embodied_fly.hover_exploration_audit import sampled_action
from embodied_fly.ppo import motor_distribution
from embodied_fly.ppo_step import motor_kl
from embodied_fly.velocity_hover_ppo import scale_exploration


def test_paired_noise_has_exact_half_latent_perturbation_and_deployed_zero():
    output = SimpleNamespace(action=torch.tensor([[0.2, -0.3], [0.4, -0.5]]))
    log_std = torch.full((2,), -4.0)
    actions = [
        sampled_action(output, log_std, scale, torch.Generator().manual_seed(42))
        for scale in (0, 0.5, 1)
    ]
    assert torch.equal(actions[0], output.action)
    mean = output.action.atanh()
    torch.testing.assert_close(actions[1].atanh() - mean, (actions[2].atanh() - mean) / 2)


def test_current_noise_matches_original_ppo_distribution():
    output = SimpleNamespace(action=torch.tensor([[0.2, -0.3], [0.4, -0.5]]))
    log_std = torch.full((2,), -4.0)
    torch.manual_seed(42)
    expected = motor_distribution(output, torch.ones(2, dtype=torch.bool), log_std, 0.0005)
    actual = sampled_action(output, log_std, 1, torch.Generator().manual_seed(42))
    torch.testing.assert_close(actual, expected.sample().tanh(), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_invalid_noise_rejected(scale):
    with pytest.raises(ValueError):
        sampled_action(
            SimpleNamespace(action=torch.zeros(1, 2)), torch.zeros(2), scale, torch.Generator()
        )


def test_training_noise_transition_preserves_rng_and_changes_probability_and_kl():
    log_std = torch.full((2,), torch.tensor(0.003).log().item())
    before = log_std.clone()
    rng = torch.get_rng_state().clone()
    scale_exploration(log_std, 1)
    assert torch.equal(log_std, before)
    scale_exploration(log_std, 0.5)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(log_std.exp(), before.exp() / 2)
    action = torch.zeros(3, 2)
    old = motor_distribution(
        SimpleNamespace(action=action), torch.ones(2, dtype=torch.bool), before, 0.0005
    )
    new = motor_distribution(
        SimpleNamespace(action=action), torch.ones(2, dtype=torch.bool), log_std, 0.0005
    )
    expected = torch.distributions.kl_divergence(old, new).sum(-1).mean().item()
    assert motor_kl(action, action, before, log_std) == pytest.approx(expected, rel=1e-5)
    for invalid in (0, -1, float("nan"), 0.001, 1000):
        with pytest.raises(ValueError):
            scale_exploration(log_std, invalid)
