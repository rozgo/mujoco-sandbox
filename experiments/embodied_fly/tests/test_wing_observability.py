import numpy as np
import torch

from embodied_fly.wing_observability import metrics, ridge_readout
from embodied_fly.wing_readout import replace_wing_rows


def test_readout_uses_training_statistics_and_recovers_a_known_held_out_mapping():
    rng = np.random.default_rng(491)
    train = rng.normal(size=(200, 12))
    test = rng.normal(3, 2, (60, 12))  # changed test distribution must not refit normalization
    matrix = rng.normal(size=(12, 6))
    prediction = ridge_readout(train, train @ matrix + 4, test, alpha=1e-8)
    np.testing.assert_allclose(prediction, test @ matrix + 4, atol=2e-6)
    result = metrics(test @ matrix + 4, prediction, (train @ matrix + 4).mean(0))
    np.testing.assert_allclose(result["r2_against_training_mean"], 1, atol=1e-10)


def test_wing_calibration_leaves_other_actions_and_all_upstream_state_exactly_intact():
    weights, bias = torch.randn(78, 256), torch.randn(78)
    original = {
        "motor_decoder.3.weight": weights,
        "motor_decoder.3.bias": bias,
        "core.bias": torch.randn(20),
    }
    wings = np.array([2, 3, 4, 9, 10, 11])
    other = [i for i in range(78) if i not in wings]
    changed = replace_wing_rows(original, weights[wings] + 1, bias[wings] - 2, wings)
    hidden = torch.randn(5, 256)
    old = (hidden @ weights.T + bias).tanh()
    new = (
        hidden @ changed["motor_decoder.3.weight"].T + changed["motor_decoder.3.bias"]
    ).tanh()
    torch.testing.assert_close(old[:, other], new[:, other], rtol=0, atol=0)
    torch.testing.assert_close(original["core.bias"], changed["core.bias"], rtol=0, atol=0)
    assert not torch.equal(old[:, wings], new[:, wings])
