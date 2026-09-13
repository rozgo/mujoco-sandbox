"""Teach local wing feedback through the existing graph, without runtime helpers.

Paired counterfactual measurements supplement nominal physical-state imitation.
They are synthetic sensory examples, not additional simulated experience.
"""

import numpy as np
import torch

from embodied_fly.motor_response import perturb_feedback
from embodied_fly.wing_motion import CONFIG


def response_examples(env, posture, observation, ids, axis, velocity):
    """Return +/- observations and bounded reference wing commands for each row."""
    q = env.fields["qpos"][ids].copy()
    v = env.fields["qvel"][ids].copy()
    variants = perturb_feedback(env.model, observation[ids], q, v)
    columns = np.where(velocity, 13, 1) + 2 * axis
    obs = np.stack(
        [variants[np.arange(len(ids)), columns + offset] for offset in (0, 1)], axis=1
    )
    angle = np.repeat(q[:, None, env.template.wing_angle_indices], 2, axis=1)
    speed = np.repeat(v[:, None, env.template.wing_velocity_indices], 2, axis=1)
    for row, (a, vel) in enumerate(zip(axis, velocity)):
        (speed if vel else angle)[row, :, a] += np.array([1, -1]) * (2.0 if vel else 0.05)
    torque = (
        posture.wing_kp * (posture.qref[env.template.wing_angle_indices] - angle)
        - posture.wing_kd * speed
    )
    targets = np.clip(torque / CONFIG.joint_torque_limit, -1, 1).astype(np.float32)
    # Normalize the requested finite difference to magnitude1 in an unsaturated
    # perturbed axis, with0 in the other five axes. This fits response, not bias.
    scale = (
        np.where(velocity, 4 * posture.wing_kd, 0.1 * posture.wing_kp)
        / CONFIG.joint_torque_limit
    )
    return obs, targets, scale.astype(np.float32)


def response_loss(
    actor, memory_before_action, env, posture, observation, task_ids, worlds, rng
):
    ids = np.flatnonzero(task_ids != 2)
    ids = rng.choice(ids, min(len(ids), worlds), replace=False)
    axis = rng.integers(0, 6, len(ids))
    velocity = rng.integers(0, 2, len(ids)).astype(bool)
    obs, target, scale = response_examples(env, posture, observation, ids, axis, velocity)
    device = memory_before_action.device
    # Copies of actual prior memory. Gradients update the current encoder/core/
    # decoder; they do not invent a preceding counterfactual physical trajectory.
    state = memory_before_action[:, ids].detach().repeat_interleave(2, dim=1)
    action = actor(torch.as_tensor(obs.reshape(-1, 397), device=device), state).action
    action = action[:, posture.wings].reshape(len(ids), 2, 6)
    target = torch.as_tensor(target, device=device)
    scale = torch.as_tensor(scale[:, None], device=device)
    response = (action[:, 0] - action[:, 1]) / scale
    expected = (target[:, 0] - target[:, 1]) / scale
    return (response - expected).square().mean()
