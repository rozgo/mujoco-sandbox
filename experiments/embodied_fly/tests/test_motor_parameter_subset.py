import pytest
import torch
from test_motor_focus import tiny_brain

from embodied_fly.motor_parameter_subset import WingOutputSubset


def test_wing_output_training_changes_only_selected_rows_and_preserves_same_history_body_outputs():
    torch.manual_seed(314)
    parent = tiny_brain().eval()
    parent.set_motor_only()
    actor = tiny_brain().eval()
    actor.load_state_dict(parent.state_dict())
    actor.set_motor_only()
    subset = WingOutputSubset(actor, list(range(14, 20)))
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=1e-3)
    observations = torch.randn(12, 3, 397)
    memory = actor.initial_state(3)
    for observation in observations[:4]:
        result = actor(observation, memory)
        memory = result.state
        assert not memory.requires_grad
        # Deliberately request changes to all outputs; gradient selection must
        # still protect body rows even when their targets differ.
        target = torch.ones_like(result.action) * 0.6
        optimizer.zero_grad(set_to_none=True)
        (result.action - target).square().mean().backward()
        subset.gradient_audit()
        optimizer.step()
    report = subset.verify_and_report()
    assert report["effective_trainable_parameters"] == 6 * 257
    assert report["tensor_requires_grad_parameters"] == 78 * 257
    assert all(v > 0 for v in report["selected_parameter_changes_l2"].values())
    assert all(p.grad is None for p in actor.core.parameters())
    assert report["runtime_action_mask"] is False
    a, b = parent.initial_state(3), actor.initial_state(3)
    maximum_wing_difference = 0.0
    with torch.no_grad():
        for observation in observations:
            old, new = parent(observation, a), actor(observation, b)
            torch.testing.assert_close(old.state, new.state, atol=0, rtol=0)
            torch.testing.assert_close(
                old.action[:, subset.other], new.action[:, subset.other], atol=0, rtol=0
            )
            maximum_wing_difference = max(
                maximum_wing_difference,
                float(
                    (old.action[:, subset.channels] - new.action[:, subset.channels])
                    .abs()
                    .max()
                ),
            )
            a, b = old.state, new.state
    assert maximum_wing_difference > 0.001
    with torch.no_grad():
        actor.motor_decoder[3].weight[0, 0] += 0.01
    with pytest.raises(RuntimeError, match="Non-wing"):
        subset.verify_and_report()


def test_wing_output_subset_rejects_invalid_row_selection():
    for channels in ([0] * 6, [-1, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 78], [0, 1]):
        with pytest.raises(ValueError):
            WingOutputSubset(tiny_brain(), channels)
