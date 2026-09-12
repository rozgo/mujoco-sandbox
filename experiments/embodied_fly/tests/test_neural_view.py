import numpy as np
import torch

from embodied_fly.neural_view import NeuralProjection


def test_anatomical_binning_preserves_world_identity_sign_and_magnitude():
    p = NeuralProjection(np.array([[0, 0, 0], [0, 0, 0], [2, 0, 2], [np.nan, 0, 0]]), size=3)
    state = torch.tensor([[0.5, -0.5], [-0.25, -0.5], [0.2, 0.8], [1, 1]])
    before = state.clone()
    result = p.project(state)
    np.testing.assert_allclose(result[:, 0, 0], [[0.125, 0.375], [-0.5, 0.5]])
    np.testing.assert_allclose(result[:, 2, 2], [[0.2, 0.2], [0.8, 0.8]])
    assert p.report()["neurons_without_locations"] == 1
    torch.testing.assert_close(state, before)
