import numpy as np
import torch

from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.retention import healthy_mask, reference_bonus, reference_loss


def test_reference_does_not_constrain_damaged_states_or_leak_into_actor():
    context = np.ones((3, 20), dtype=np.float32)
    context[1, 0] = 0.7
    context[2, 4] = 0.6
    healthy = healthy_mask(context)
    np.testing.assert_array_equal(healthy, [True, False, False])
    actions = np.ones((3, 12), dtype=np.float32)
    target = np.zeros_like(actions)
    bonus = reference_bonus(actions, target, healthy)
    assert 0 < bonus[0] < reference_bonus(target, target, healthy)[0]
    assert not bonus[1:].any()
    mean = torch.tensor(actions, requires_grad=True)
    loss = reference_loss(mean, torch.tensor(target), torch.tensor(healthy))
    loss.backward()
    assert mean.grad[0].abs().sum() > 0
    assert mean.grad[1:].abs().sum() == 0
    assert reference_loss(mean, torch.tensor(target), torch.zeros(3)) == 0


def test_retention_curriculum_preserves_healthy_episodes_and_separates_damage():
    env = DogEnv(64, seed=19, threads=4, retention_curriculum=True)
    try:
        assert [g.n for g in env.groups] == [48, 4, 4, 4, 4]
        healthy = healthy_mask(env.context)
        permanent_healthy = healthy & (env.fault_at == 100000)
        assert 20 <= permanent_healthy.sum() <= 42
        assert (env.strength[48:] == 1).all()
        assert (env.fault_at[48:] == 100000).all()
        assert (env.context[48:, :4].min(1) == np.float32(0.7)).all()
        # Force a scheduled loss on an intact/full-strength episode. Its health
        # mask changes immediately and the target torque cap actually changes.
        i = np.flatnonzero(permanent_healthy)[0]
        before = env.obs()[i].copy()
        env.fault_at[i], env.fault_joint[i], env.fault_strength[i] = 0, 4, 0.5
        # Private schedule/strength metadata isn't an observation channel.
        np.testing.assert_array_equal(env.obs()[i], before)
        env.step(np.zeros((64, 12)))
        assert not healthy_mask(env.context)[i]
        g = env.groups[0]
        slot = np.flatnonzero(g.slot == 4)[0]
        assert np.isclose(g.force_range[i, slot, 1], 23.7 * 0.5)
        assert env.obs().shape == (64, 66)
    finally:
        env.close()
