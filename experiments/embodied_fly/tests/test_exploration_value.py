import numpy as np
import pytest
import torch
from torch import nn

from embodied_fly.exploration_value import fit_capacity, quality, window_returns


def test_complete_realized_windows_match_direct_sum_and_preserve_failure_penalty():
    reward = np.arange(14, dtype=float).reshape(7, 2) / 10
    reward[2, 1] = -1
    reward[3:, 1] = 0  # absorbing failure, as used by frozen collection
    expected = np.stack(
        [(reward[i : i + 3] * 0.9 ** np.arange(3)[:, None]).sum(0) for i in range(5)]
    )
    np.testing.assert_allclose(window_returns(reward, 0.9, 3), expected, atol=1e-6)
    assert window_returns(reward, 0.9, 3)[2, 1] == -1
    assert window_returns(reward, 0.9, 3)[3, 1] == 0
    with pytest.raises(ValueError):
        window_returns(reward, 0.9, 8)


def test_target_normalization_folds_back_to_same_network_without_mutating_parent():
    torch.manual_seed(177)
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(32, 4)
    y = 100 + 2 * x[:, 0]
    train = np.arange(32) < 24
    fitted, report = fit_capacity(
        model, x, y, train, ~train, steps=30, seed=42, normalized=True
    )
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
    assert report["test"]["rmse"] < 5
    with torch.no_grad():
        actual = quality(fitted(x[~train]).squeeze(-1), y[~train])
    assert actual["rmse"] == pytest.approx(report["test"]["rmse"], abs=1e-5)
    assert list(fitted.state_dict()) == list(model.state_dict())


def test_quality_does_not_invent_correlation_for_constant_predictions():
    result = quality(np.zeros(5), np.arange(5))
    assert result["pearson"] is None
    assert result["explained_variance"] == pytest.approx(0)
