"""Training-only frozen motor reference and explicit task mixture accounting."""

import copy

import numpy as np
import torch


class FrozenMotorReference:
    def __init__(self, actor, worlds):
        # Share immutable graph buffers only. Every learned parameter is copied.
        memo = {id(x): x for x in (actor.core.adjacency, actor.core.transpose)}
        self.actor = copy.deepcopy(actor, memo).eval().requires_grad_(False)
        self.memory = self.actor.initial_state(worlds)

    @torch.no_grad()
    def act(self, observation):
        result = self.actor(observation, self.memory)
        self.memory = result.state
        return result.action

    def reset(self, done):
        self.memory = self.actor.reset_worlds(self.memory, done)


def task_mixtures(task_ids, teacher_mix, hover_teacher_mix=None, walk_teacher_mix=None):
    hover_mix = teacher_mix if hover_teacher_mix is None else hover_teacher_mix
    walk_mix = teacher_mix if walk_teacher_mix is None else walk_teacher_mix
    if not np.isfinite([teacher_mix, hover_mix, walk_mix]).all() or not (
        0 <= teacher_mix <= 1 and 0 <= hover_mix <= 1 and 0 <= walk_mix <= 1
    ):
        raise ValueError("Teacher mixtures must be finite and in [0,1]")
    ids = np.asarray(task_ids)
    return np.where(ids == 2, hover_mix, np.where(ids == 1, walk_mix, teacher_mix)).astype(
        np.float32
    )


def hover_start_weights(task_ids, ages, interval, duration, weight):
    """Weight rare initial hover corrections in training; never an actor input."""
    if (
        not np.isfinite([interval, duration, weight]).all()
        or interval <= 0
        or duration < 0
        or weight < 1
    ):
        raise ValueError("Positive interval, nonnegative window and weight >= 1 required")
    early = (np.asarray(task_ids) == 2) & (np.asarray(ages) * interval < duration)
    return np.where(early, weight, 1).astype(np.float32)


def task_loss(group_losses, ground_weight=1.0):
    if not np.isfinite(ground_weight) or ground_weight <= 0:
        raise ValueError("Ground retention weight must be finite and positive")
    weights = {i: ground_weight if i != 2 else 1.0 for i in group_losses}
    return sum(group_losses[i] * w for i, w in weights.items()) / sum(weights.values())


def ground_nonwing_loss(action, reference, task_ids, nonwing_channels):
    """Distill the frozen parent's ground body commands; never overwrite actions."""
    ground = torch.as_tensor(np.asarray(task_ids) != 2, device=action.device)
    channels = torch.as_tensor(nonwing_channels, device=action.device)
    reference = torch.as_tensor(reference, device=action.device)
    if reference.shape != action.shape or not ground.any():
        raise ValueError("Matching reference actions and at least one ground world required")
    return (action[ground][:, channels] - reference[ground][:, channels]).square().mean()
