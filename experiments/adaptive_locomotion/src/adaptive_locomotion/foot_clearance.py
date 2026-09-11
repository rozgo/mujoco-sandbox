# Inspired by Isaac Lab's speed-weighted foot_clearance_reward, BSD-3-Clause.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# See third_party/isaaclab_rewards/PROVENANCE.md and LICENSE.
"""Flat-ground foot clearance, with no contact schedule or commanded foot path."""

import numpy as np

from .bodies import CONTROL_DT


def clearance_cost(previous, current, radii, valid, commands, target=0.03, scope="all"):
    """Penalize low moving intact feet on damaged bodies, not stationary support.

    Velocity is measured in the world frame. Using contact as a swing gate would
    exempt the very dragging we want to discourage. Stumps/absent limbs do not
    get an intact-foot target. Geometry is never modified by this reward.
    """
    speed = np.linalg.norm((current[..., :2] - previous[..., :2]) / CONTROL_DT, axis=2)
    height = (current[..., 2] + previous[..., 2]) * 0.5 - radii
    deficit = np.clip((target - height) / target, 0, 1)
    intact = valid.reshape(-1, 4, 3).all(2)
    damaged = ~intact.all(1)
    if scope == "surviving_rear":
        damaged = ~intact[:, 2:].all(1)
        intact[:, :2] = False
    elif scope != "all":
        raise ValueError("Unknown clearance scope")
    moving = np.linalg.norm(commands[:, :2], axis=1) > 0.15
    cost = (deficit**2 * np.tanh(2 * speed) * intact).sum(1)
    return cost / np.maximum(intact.sum(1), 1) * damaged * moving


def clearance_metrics(positions, radii):
    """Per-leg measures after 1 s, independent of the training reward.

    Positions have time x trials x legs x xyz axes. Heights refer to the sphere
    bottom above the z=0 plane. Do not interpret absent semantic slots as feet.
    """
    start = round(1 / CONTROL_DT)
    height = positions[start:, ..., 2] - radii
    speed = np.linalg.norm(np.diff(positions[start - 1 :, ..., :2], axis=0), axis=-1)
    speed /= CONTROL_DT
    moving = speed > 0.2
    drag = moving & (height < 0.005)
    return {
        "moving_clearance_m": (np.maximum(height, 0) * speed).sum(0)
        / np.maximum(speed.sum(0), 1e-8),
        "clearance_p95_m": np.quantile(height, 0.95, axis=0),
        "drag_fraction": drag.mean(0),
        "drag_travel_m": (speed * drag).sum(0) * CONTROL_DT,
    }
