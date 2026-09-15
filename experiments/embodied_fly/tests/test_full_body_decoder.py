import numpy as np
import pytest
import torch
from scipy import sparse

from embodied_fly.brain import EmbodiedBrain
from embodied_fly.full_body_decoder import (
    consolidate_actor,
    decode_features,
    freeze_for_decoder_training,
    motor_features,
)


def legacy_actor():
    torch.manual_seed(311)
    graph = sparse.csr_matrix(
        (np.ones(4), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6), dtype=np.float32
    )
    actor = EmbodiedBrain(
        graph,
        [0, 1],
        [2, 3],
        [4, 5],
        4,
        78,
        motor_only=True,
        wing_residual_enabled=True,
        wing_residual_hidden=128,
    ).double()
    with torch.no_grad():
        actor.wing_residual.network[-1].weight.normal_(0, 0.1)
        actor.wing_residual.feature_scale.fill_(0.06)
        actor.wing_residual.feature_mean.copy_(torch.tensor([0.5, -0.3]))
    return actor


def test_consolidation_preserves_actions_recurrence_and_clipped_normalization():
    old = legacy_actor()
    new = legacy_actor()
    new.load_state_dict(old.state_dict())
    consolidate_actor(new)
    assert new.wing_residual is None
    assert not any(k.startswith("wing_residual.") for k in new.state_dict())
    a, b = old.initial_state(3), new.initial_state(3)
    for _ in range(15):
        obs = torch.randn(3, 4, dtype=torch.float64)
        before, after = old(obs, a), new(obs, b)
        torch.testing.assert_close(before.action, after.action, atol=1e-13, rtol=1e-13)
        assert torch.equal(before.state, after.state)
        a, b = before.state.detach(), after.state.detach()
    # Known extreme features hit clipping; preservation is not restricted to
    # a calibration dataset or the unclipped approximation.
    features = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float64)
    z = old.motor_decoder[0](features)
    logits = old.motor_decoder[3](old.motor_decoder[2](old.motor_decoder[1](z)))
    logits[:, 14:20] += old.wing_residual(z)
    torch.testing.assert_close(
        new.motor_decoder(features), logits.tanh(), atol=1e-13, rtol=1e-13
    )
    with pytest.raises(ValueError, match="isolated"):
        new.enable_wing_residual(128)


def test_all_actuator_rows_can_learn_from_all_shared_hidden_units():
    actor = legacy_actor()
    consolidate_actor(actor)
    freeze_for_decoder_training(actor)
    assert all(
        p.requires_grad == name.startswith("motor_decoder.")
        for name, p in actor.named_parameters()
    )
    state = torch.randn(6, 5, dtype=torch.float64)
    features = motor_features(actor, state)
    assert features.shape == (5, 2)  # Raw cells, not stale cached normalization.
    target = torch.linspace(-0.8, 0.8, 78, dtype=torch.float64).expand(5, -1)
    optimizer = torch.optim.SGD(actor.motor_decoder.parameters(), lr=0.01)
    optimizer.zero_grad()
    (decode_features(actor, features) - target).square().mean().backward()
    grad = actor.motor_decoder[3].weight.grad
    assert (grad.abs().sum(1) > 0).all()  # Every actuator, not just six wings.
    assert grad[np.r_[0:14, 20:78], 256:].abs().sum() > 0
    optimizer.step()
    assert actor.motor_decoder[3].weight[np.r_[0:14, 20:78], 256:].abs().sum() > 0
    assert all(
        p.grad is None
        for name, p in actor.named_parameters()
        if not name.startswith("motor_decoder.")
    )


def test_new_random_velocity_actor_defaults_to_general_decoder():
    from embodied_fly.fresh_velocity_actor import initialize

    old = legacy_actor()
    graph = old.core.adjacency.to_dense().numpy()
    actor = initialize(sparse.csr_matrix(graph), [0, 1], [2, 3], [4, 5], 319)
    assert actor.wing_residual is None and actor.shared_decoder_hidden == 384
    assert actor.motor_decoder[3].out_features == 78
