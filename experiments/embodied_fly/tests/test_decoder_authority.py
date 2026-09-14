import copy

import numpy as np
import pytest
import torch

from embodied_fly.decoder_authority import (
    parameter_variants,
    response_jacobian,
    varied_linear_output,
)


def test_batched_probes_equal_actual_parameter_copies_and_leave_base_unchanged():
    torch.manual_seed(9)
    layer = torch.nn.Linear(128, 6).double()
    x = torch.randn(16, 128, dtype=torch.float64)
    offsets, _ = parameter_variants(0.001)
    changes = torch.as_tensor(offsets, dtype=x.dtype)
    before = copy.deepcopy(layer.state_dict())
    base = layer(x)
    actual = varied_linear_output(layer, (x,), base, changes)
    expected = []
    for i in range(16):
        variant = copy.deepcopy(layer)
        with torch.no_grad():
            variant.bias += changes[i, :6]
            variant.weight[[0, 3]] *= 1 + changes[i, 6]
        expected.append(variant(x[i]))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[[0, 15]], base[[0, 15]], rtol=0, atol=0)
    for k, v in layer.state_dict().items():
        assert torch.equal(v, before[k])


def test_probe_order_recovers_known_coupled_authority():
    delta = 0.001
    changes, names = parameter_variants(delta)
    matrix = np.array([[1, 2, 0, 0, 0, 1, 0], [0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 3]])
    values = changes @ matrix.T + np.array([0.1, -0.1, 1])
    np.testing.assert_allclose(response_jacobian(values, delta), matrix, rtol=1e-6, atol=1e-12)
    assert names[0] == "unchanged" and names[-1] == "unchanged_duplicate"
    assert np.linalg.matrix_rank(response_jacobian(values, delta)) == 3


@pytest.mark.parametrize("delta", [0, -0.1, 0.1, np.inf, np.nan])
def test_invalid_probes_are_rejected(delta):
    with pytest.raises(ValueError):
        parameter_variants(delta)
