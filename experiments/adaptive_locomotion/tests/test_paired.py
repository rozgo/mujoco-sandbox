import numpy as np
import torch

from adaptive_locomotion.bodies import build_model
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.paired import HELDOUT_PAIRS, TRAIN_PAIRS, training_pairs
from adaptive_locomotion.retention import reference_loss, single_damage_mask


def test_pair_curriculum_keeps_groups_and_heldout_combinations_separate():
    assert not set(TRAIN_PAIRS) & set(HELDOUT_PAIRS)
    env = DogEnv(64, threads=4, seed=7, retention_curriculum=True, pair_level="hard")
    try:
        assert [g.n for g in env.groups] == [32, 4, 4, 4, 4, 4, 4, 4, 4]
        np.testing.assert_array_equal(
            (env.context[:, :4] < 1).sum(1), [0] * 32 + [1] * 16 + [2] * 16
        )
        assert (env.strength[32:] == 1).all() and (env.fault_at[32:] == 100000).all()
        qpos = [g.qpos.copy() for g in env.groups]
        env.reset([33])
        for g, q in zip(env.groups[5:], qpos[5:], strict=True):
            np.testing.assert_array_equal(g.qpos, q)
        assert env.obs().shape == (64, 66)
    finally:
        env.close()
    for mild, hard in zip(training_pairs("mild"), training_pairs("hard"), strict=True):
        a, b = build_model(mild), build_model(hard)
        assert a.nq == b.nq == 19 and a.nu == b.nu == 12
        assert b.body_mass.sum() < a.body_mass.sum()
        assert (b.body_inertia[1:] > 0).all()


def test_single_skill_reference_leaves_pair_actions_unconstrained():
    context = np.ones((3, 20), dtype=np.float32)
    context[1, 0] = 0.7
    context[2, [0, 3]] = [0.75, 0.65]
    mask = single_damage_mask(context)
    np.testing.assert_array_equal(mask, [False, True, False])
    mean = torch.ones(3, 12, requires_grad=True)
    loss = reference_loss(mean, torch.zeros_like(mean), torch.tensor(mask))
    loss.backward()
    assert mean.grad[1].abs().sum() > 0
    assert mean.grad[[0, 2]].abs().sum() == 0
