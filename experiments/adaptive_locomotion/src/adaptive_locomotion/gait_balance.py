# Reward formulas adapted from Isaac Lab's Spot rewards, BSD-3-Clause.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# See third_party/isaaclab_rewards/PROVENANCE.md and LICENSE.
"""Measured stance/swing duration regularization, without a prescribed gait."""

import numpy as np

from .bodies import CONTROL_DT


class ContactTiming:
    def __init__(self, n):
        self.contact = np.zeros((n, 4), bool)
        self.air = np.zeros((n, 4))
        self.stance = self.air.copy()
        self.last_air = self.air.copy()
        self.last_stance = self.air.copy()

    def reset(self, ids, forces):
        self.contact[ids] = forces[ids] > 5
        for state in (self.air, self.stance, self.last_air, self.last_stance):
            state[ids] = 0

    def update(self, forces):
        contact = forces > np.where(self.contact, 1, 5)
        landing = contact & ~self.contact
        takeoff = ~contact & self.contact
        self.last_air[landing] = self.air[landing]
        self.last_stance[takeoff] = self.stance[takeoff]
        self.air = np.where(contact, 0, self.air + CONTROL_DT)
        self.stance = np.where(contact, self.stance + CONTROL_DT, 0)
        self.contact = contact
        # Match torch.var's sample variance and the upstream half-second clamp.
        return np.var(np.minimum(self.last_air, 0.5), axis=1, ddof=1) + np.var(
            np.minimum(self.last_stance, 0.5), axis=1, ddof=1
        )


def body_motion_cost(velocity, angular_velocity):
    return 0.8 * velocity[:, 2] ** 2 + 0.2 * np.abs(angular_velocity[:, :2]).sum(1)
