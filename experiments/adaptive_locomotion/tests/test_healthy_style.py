import numpy as np
import pytest
import torch

from adaptive_locomotion.healthy_style import (
    HealthyMotion,
    style_bonus,
    style_loss,
    style_mask,
    teacher_observation,
)


def test_teacher_encoding_is_private_and_preserves_surviving_joint_feedback():
    obs = np.arange(2 * 66, dtype=np.float32).reshape(2, 66)
    obs[:, 45:57] = 1
    obs[1, 48:51] = 0
    original = obs.copy()
    target_obs = teacher_observation(obs)
    np.testing.assert_array_equal(obs, original)
    np.testing.assert_array_equal(target_obs[0], obs[0])
    for start in (0, 12, 30):
        assert not target_obs[1, start + 3 : start + 6].any()
        np.testing.assert_array_equal(
            target_obs[1, start : start + 3], obs[1, start : start + 3]
        )
    np.testing.assert_array_equal(target_obs[1, 45:57], np.ones(12))
    np.testing.assert_array_equal(target_obs[:, 24:30], obs[:, 24:30])
    np.testing.assert_array_equal(target_obs[:, 42:45], obs[:, 42:45])


def test_healthy_style_only_constrains_surviving_joints_on_damaged_bodies():
    obs = np.zeros((2, 66), np.float32)
    obs[:, 45:57] = 1
    obs[1, 48:51] = 0
    mask = style_mask(obs)
    mean = torch.ones((2, 12), requires_grad=True)
    target = torch.zeros((2, 12))
    style_loss(mean, target, torch.tensor(mask)).backward()
    assert not mean.grad[0].any() and not mean.grad[1, 3:6].any()
    assert mean.grad[1, :3].abs().sum() > 0
    action = np.zeros((2, 12))
    target = np.zeros_like(action)
    assert style_bonus(action, target, mask)[1] == 1
    action[1, 3:6] = 100
    assert style_bonus(action, target, mask)[1] == 1
    action[1, 0] = 1
    assert 0 < style_bonus(action, target, mask)[1] < 1
    assert style_bonus(action, target, mask)[0] == 0


def test_motion_matching_allows_independent_leg_phases_and_masks_removed_joints():
    bank = np.zeros((2, 66), np.float32)
    bank[:, 45:57] = 1
    bank[1, :24] = 1
    actions = np.arange(24, dtype=np.float32).reshape(2, 12)
    reference = HealthyMotion(bank, actions)
    obs = bank[:1].copy()
    for leg in (1, 3):
        obs[:, leg * 3 : leg * 3 + 3] = 1
        obs[:, 12 + leg * 3 : 15 + leg * 3] = 1
    obs[0, 45 + 5] = 0
    obs[0, 5] = obs[0, 17] = 1000
    original = obs.copy()
    target, distance = reference.query(obs)
    for leg, phase in enumerate((0, 1, 0, 1)):
        np.testing.assert_array_equal(
            target[0, leg * 3 : leg * 3 + 3], actions[phase, leg * 3 : leg * 3 + 3]
        )
    np.testing.assert_array_equal(distance, 0)
    np.testing.assert_array_equal(obs, original)


def test_motion_sequences_distinguish_identical_poses_with_different_histories():
    obs = np.zeros((2, 66), np.float32)
    obs[:, 45:57] = 1
    past = obs.copy()
    past[1, :24] = 0.4
    actions = np.array([[0] * 12, [1] * 12], np.float32)
    reference = HealthyMotion(obs, actions, past)
    target, distance = reference.query(obs[:1], past[1:])
    np.testing.assert_array_equal(target, actions[1:])
    np.testing.assert_allclose(distance, 0, atol=1e-6)
    with pytest.raises(ValueError, match="same causal history"):
        reference.query(obs)


def test_frozen_motion_reference_preserves_queries_and_rejects_wrong_history(tmp_path):
    rng = np.random.default_rng(123)
    obs = rng.normal(size=(8, 66)).astype(np.float32)
    obs[:, 45:57] = 1
    past = rng.normal(size=obs.shape).astype(np.float32)
    actions = rng.normal(size=(8, 12)).astype(np.float32)
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, observations=obs, actions=actions, past_observations=past)
    reference = HealthyMotion(obs, actions, past)
    loaded = HealthyMotion.load(path, sequence=True)
    for expected, actual in zip(reference.query(obs, past), loaded.query(obs, past)):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="history mode"):
        HealthyMotion.load(path, sequence=False)


@pytest.mark.parametrize("sequence", [False, True])
def test_fast_masked_distance_agrees_with_brute_force(sequence):
    rng = np.random.default_rng(14)
    bank = rng.normal(size=(64, 66)).astype(np.float32)
    actions = rng.normal(size=(64, 12)).astype(np.float32)
    obs = rng.normal(size=(8, 66)).astype(np.float32)
    obs[:, 45:57] = rng.integers(0, 2, (8, 12))
    obs[0, 45:48] = 0  # A completely absent leg must also be harmless.
    past_bank = rng.normal(size=bank.shape).astype(np.float32) if sequence else None
    past = rng.normal(size=obs.shape).astype(np.float32) if sequence else None
    prior = HealthyMotion(bank, actions, past_bank)
    target, errors = prior.query(obs, past)
    f = prior.feature(obs, past)
    valid = obs[:, 45:57].reshape(-1, 4, 3)
    channels = (valid, valid, np.ones((8, 4, 1)))
    mask = np.concatenate(channels + ((valid, valid) if sequence else ()), axis=2)
    for leg in range(4):
        delta = (f[:, None, leg] - prior.features[None, :, leg]) / prior.scale[leg]
        distance = (delta**2 * mask[:, None, leg]).sum(2) / mask[:, leg].sum(1)[:, None]
        idx = distance.argmin(1)
        np.testing.assert_allclose(
            errors[:, leg], distance[np.arange(8), idx], atol=2e-6
        )
        np.testing.assert_array_equal(
            target.reshape(8, 4, 3)[:, leg], actions.reshape(64, 4, 3)[idx, leg]
        )
