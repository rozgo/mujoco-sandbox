"""Training-only healthy-policy targets for surviving joints after removal."""

import numpy as np


def teacher_observation(obs):
    """Query the intact teacher without its out-of-distribution validity bits.

    Missing joints are represented at nominal pose, zero speed and zero previous
    action. All other measurements are actual current feedback. Only the teacher
    gets this hypothetical intact encoding; the deployed actor sees real inputs.
    """
    result = obs.copy()
    valid = obs[:, 45:57]
    for start in (0, 12, 30):
        result[:, start : start + 12] *= valid
    result[:, 45:57] = 1
    return result


def style_mask(obs):
    valid = obs[:, 45:57]
    return valid * (valid < 1).any(1)[:, None]


def style_bonus(action, target, mask):
    error = (((np.clip(action, -3, 3) - target) * mask) ** 2).sum(1)
    count = mask.sum(1)
    return (count > 0) * np.exp(-error / np.maximum(count, 1) / 0.25)


def style_loss(mean, target, mask):
    count = mask.sum(-1)
    error = (((mean - target) * mask).square().sum(-1)) / count.clamp_min(1)
    return error.sum() / (count > 0).sum().clamp_min(1)
