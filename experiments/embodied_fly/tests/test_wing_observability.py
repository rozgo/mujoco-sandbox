import numpy as np

from embodied_fly.wing_observability import metrics, ridge_readout


def test_readout_uses_training_statistics_and_recovers_a_known_held_out_mapping():
    rng = np.random.default_rng(491)
    train = rng.normal(size=(200, 12))
    test = rng.normal(3, 2, (60, 12))  # changed test distribution must not refit normalization
    matrix = rng.normal(size=(12, 6))
    prediction = ridge_readout(train, train @ matrix + 4, test, alpha=1e-8)
    np.testing.assert_allclose(prediction, test @ matrix + 4, atol=2e-6)
    result = metrics(test @ matrix + 4, prediction, (train @ matrix + 4).mean(0))
    np.testing.assert_allclose(result["r2_against_training_mean"], 1, atol=1e-10)
