from types import SimpleNamespace

import pytest
import torch

from embodied_fly.hover_exploration_audit import sampled_action
from embodied_fly.ppo import motor_distribution


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
