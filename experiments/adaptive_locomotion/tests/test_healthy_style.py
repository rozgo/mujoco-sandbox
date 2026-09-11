import numpy as np
import torch

from adaptive_locomotion.healthy_style import (
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
