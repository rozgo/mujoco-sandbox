import numpy as np
import pytest
import torch
from scipy import sparse

from embodied_fly.brain import EmbodiedBrain, NeuralCore


def tiny_graph():
    # Two input -> two descending -> two motor cells, with recurrent feedback.
    return sparse.csr_matrix(
        (
            np.array([1, -1, 0.7, 0.8, 0.2, -0.1], np.float32),
            ([2, 3, 4, 5, 2, 3], [0, 1, 2, 3, 4, 5]),
        ),
        shape=(6, 6),
    )


def test_sparse_core_matches_dense_values_and_gradients():
    torch.manual_seed(2)
    core = NeuralCore(tiny_graph()).double()
    state = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    drive = torch.randn_like(state)
    actual = core(state, drive)
    gain = (0.25 + 3.75 * core.excitability.sigmoid())[:, None]
    leak = (0.02 + 0.96 * core.leak.sigmoid())[:, None]
    expected = state + leak * (
        (gain * (core.adjacency.to_dense() @ state) + core.bias[:, None] + drive).tanh()
        - state
    )
    parameters = (state, core.excitability, core.leak, core.bias)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    torch.testing.assert_close(actual, expected)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)


def make_brain():
    torch.manual_seed(4)
    return EmbodiedBrain(tiny_graph(), [0, 1], [2, 3], [4, 5], 4, 3)


def test_core_learns_motor_loss_and_utility_stays_in_same_actor():
    brain = make_brain()
    before = {n: p.detach().clone() for n, p in brain.core.named_parameters()}
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
    output = brain(torch.randn(3, 4), brain.initial_state(3))
    loss = output.action.square().mean() + torch.nn.functional.cross_entropy(
        output.utility_logits, torch.tensor([0, 1, 2])
    )
    loss.backward()
    for parameter in brain.core.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in brain.core.named_parameters():
        assert not torch.equal(before[name], parameter)
    assert output.action.abs().max() <= 1
    torch.testing.assert_close(output.utility_scores.sum(1), torch.ones(3))


def test_memory_is_independent_persistent_and_selectively_reset():
    brain = make_brain().eval()
    observation = torch.randn(2, 4)
    first = brain(observation, brain.initial_state(2))
    second = brain(observation, first.state)
    assert not torch.allclose(first.state, second.state)
    for i in range(2):
        alone = brain(observation[i : i + 1], first.state[:, i : i + 1])
        torch.testing.assert_close(alone.action[0], second.action[i])
    reset = brain.reset_worlds(second.state, torch.tensor([True, False]))
    assert not reset[:, 0].any()
    torch.testing.assert_close(reset[:, 1], second.state[:, 1])


def test_no_sensory_to_motor_bypass_without_connectivity():
    graph = sparse.csr_matrix((6, 6), dtype=np.float32)
    brain = EmbodiedBrain(graph, [0, 1], [2, 3], [4, 5], 4, 3).eval()
    result = brain(torch.randn(2, 4), brain.initial_state(2))
    torch.testing.assert_close(result.action[0], result.action[1])


def test_neural_clock_preserves_held_target_relaxation_and_legacy_values():
    core = NeuralCore(sparse.csr_matrix((6, 6), dtype=np.float32)).double()
    with torch.no_grad():
        core.leak.copy_(torch.linspace(-2, 2, 6))
    state = torch.randn(6, 2, dtype=torch.float64)
    drive = torch.randn_like(state)
    original = core(state, drive)
    assert torch.equal(original, core(state, drive, time_scale=1.0))
    fine = state
    for _ in range(10):
        fine = core(fine, drive, time_scale=0.1)
    torch.testing.assert_close(fine, original, atol=1e-12, rtol=1e-12)
    # The graph is still frozen; only state/cell parameters receive gradients.
    gradients = torch.autograd.grad(fine.square().sum(), (core.leak, core.bias))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="time scale"):
            core(state, drive, time_scale=invalid)
