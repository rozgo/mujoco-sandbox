import copy

import pytest
import torch

from embodied_fly.ppo_step import bounded_step, motor_kl


def test_analytic_tanh_distribution_kl_is_zero_identical_and_detects_small_joint_shift():
    mean = torch.zeros(3, 2, 78)
    log_std = torch.full((78,), -6.0)
    assert motor_kl(mean, mean, log_std, log_std) == 0
    assert motor_kl(mean, mean + 0.001, log_std, log_std) > 1
    assert motor_kl(mean, mean + 0.000001, log_std, log_std) < 1e-4


def test_backtracking_adam_matches_one_small_step_not_several_updates():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    result = bounded_step(
        optimizer, [parameter], lambda: parameter.square().item() * 100, limit=0.02
    )
    assert result["accepted"] and result["attempts"] > 1
    assert result["accepted_kl"] <= 0.02
    assert optimizer.state[parameter]["step"].item() == 1
    reference = torch.nn.Parameter(torch.tensor([0.0]))
    reference_optimizer = torch.optim.Adam([reference], lr=result["lr"])
    reference.grad = parameter.grad.clone()
    reference_optimizer.step()
    torch.testing.assert_close(reference, parameter, rtol=0, atol=0)


@pytest.mark.parametrize("measured", [float("inf"), 1.0])
def test_rejected_step_restores_weights_and_existing_adam_history(measured):
    parameter = torch.nn.Parameter(torch.tensor([0.2]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    weights = parameter.detach().clone()
    history = copy.deepcopy(optimizer.state[parameter])
    result = bounded_step(optimizer, [parameter], lambda: measured, max_attempts=3)
    assert not result["accepted"]
    torch.testing.assert_close(parameter, weights, rtol=0, atol=0)
    for key, value in history.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
