import numpy as np
import torch
from test_brain import make_brain

from embodied_fly.ppo import Critic
from embodied_fly.ppo_critic import fit_critic, value_quality


def test_critic_calibration_is_training_only_fixed_and_checkpointed():
    torch.manual_seed(202)
    brain = make_brain()
    critic = Critic(brain, standardize_inputs=True)
    original = {k: v.clone() for k, v in brain.state_dict().items()}
    features = torch.randn(16, 3, critic.network[0].in_features) * 0.2 + 7
    features[..., -1] = 4  # constant features must stay finite
    targets = 1 + features[..., 0] - 7
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)
    fit = fit_critic(critic, optimizer, features, targets, 4, 4, np.random.default_rng(5))
    assert fit["input_calibrated_this_fit"]
    assert fit["fit_mse_after"] < fit["fit_mse_before"]
    assert all(torch.equal(original[k], v) for k, v in brain.state_dict().items())
    mean = critic.network.input_mean.clone()
    scale = critic.network.input_scale.clone()
    torch.testing.assert_close(mean, features.flatten(0, 1).mean(0))
    assert scale.min() >= 0.05
    fit = fit_critic(critic, optimizer, features + 3, targets, 4, 1, np.random.default_rng(6))
    assert not fit["input_calibrated_this_fit"]
    torch.testing.assert_close(mean, critic.network.input_mean)
    torch.testing.assert_close(scale, critic.network.input_scale)
    restored = Critic(brain, standardize_inputs=True)
    restored.load_state_dict(critic.state_dict(), strict=True)
    torch.testing.assert_close(restored.network(features), critic.network(features))
    assert not restored.network.calibrate(features - 10)


def test_legacy_value_network_stays_compatible_and_calibration_rejects_nonfinite():
    import pytest

    brain = make_brain()
    legacy = Critic(brain)
    assert list(legacy.state_dict()) == [
        f"network.{i}.{k}" for i in (0, 2, 4) for k in ("weight", "bias")
    ]
    critic = Critic(brain, standardize_inputs=True)
    with pytest.raises(ValueError, match="Nonfinite"):
        critic.network.calibrate(
            torch.full((2, 1, critic.network[0].in_features), float("nan"))
        )


def test_saved_critic_inputs_reproduce_values_and_do_not_change_with_actor_weights():
    torch.manual_seed(197)
    brain = make_brain()
    critic = Critic(brain)
    state = brain.initial_state(3)
    features, values = [], []
    for t in range(8):
        obs = torch.randn(3, 4, requires_grad=True)
        features.append(critic.features(brain, obs, state))
        values.append(critic(brain, obs, state).detach())
        state = brain.reset_worlds(
            brain(obs, state).state, torch.tensor([t == 3, False, False])
        )
    cached = torch.stack(features)
    assert not cached.requires_grad
    expected = torch.stack(values)
    torch.testing.assert_close(critic.network(cached).squeeze(-1), expected)
    with torch.no_grad():
        for parameter in brain.parameters():
            parameter.add_(0.5)
    torch.testing.assert_close(critic.network(cached).squeeze(-1), expected)


def test_critic_fits_every_sample_without_backpropagating_into_inputs_or_targets():
    torch.manual_seed(198)
    brain = make_brain()
    critic = Critic(brain)
    initial = {k: v.clone() for k, v in brain.state_dict().items()}
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)
    features = torch.randn(16, 3, critic.network[0].in_features, requires_grad=True)
    returns = (0.2 * features[..., 0].detach() + 1).requires_grad_()
    fit = fit_critic(critic, optimizer, features, returns, 4, 2, np.random.default_rng(4))
    assert fit["updates"] == 8 and fit["sample_presentations"] == 96
    assert fit["fit_mse_after"] < fit["fit_mse_before"]
    assert features.grad is returns.grad is None
    assert all(p.grad is None for p in brain.parameters())
    assert all(torch.equal(initial[k], v) for k, v in brain.state_dict().items())
    assert {int(s["step"]) for s in optimizer.state.values()} == {8}


def test_value_quality_keeps_task_errors_separate():
    targets = torch.tensor([[0.0, 2.0, 3.0], [1.0, 3.0, 4.0]])
    predictions = targets + torch.tensor([0.0, 1.0, 2.0])
    report = value_quality(predictions, targets, np.arange(3), ("stand", "walk", "hover"))
    assert [v["prediction_rmse"] for v in report.values()] == [0, 1, 2]
