"""Optional event-based stride preference; no clock or prescribed footfall order."""

import numpy as np

from .bodies import CONTROL_DT


class StrideTracker:
    def __init__(self, n):
        self.contact = np.zeros((n, 4), bool)
        self.air = np.zeros((n, 4))
        self.lift = np.zeros((n, 4, 3))
        self.previous = self.lift.copy()

    def reset(self, ids, positions, forces):
        self.contact[ids] = forces[ids] > 5
        self.air[ids] = 0
        self.lift[ids] = self.previous[ids] = positions[ids]

    def update(self, positions, forces, direction, moving):
        contact = forces > np.where(self.contact, 1, 5)
        takeoff = self.contact & ~contact
        self.lift[takeoff] = self.previous[takeoff]
        self.air += CONTROL_DT
        touchdown = contact & ~self.contact & (self.air >= 0.06)
        advance = np.sum(
            (positions - self.lift)[:, :, :2] * direction[:, None, :], axis=-1
        )
        # A per-landing offset discourages many tiny steps. Reward saturates at
        # 40 cm; time in the air alone earns nothing. Supporting sliding costs.
        landing = (touchdown * (np.clip(advance, 0, 0.4) - 0.30)).sum(1)
        speed2 = np.sum(((positions - self.previous)[:, :, :2] / CONTROL_DT) ** 2, 2)
        slip = np.minimum(speed2, 4) * contact * self.contact
        flight = ~contact.any(1)
        reward = moving * (8 * landing - 0.15 * slip.sum(1) - 0.5 * flight)
        self.air[contact] = 0
        self.contact = contact
        self.previous[:] = positions
        return reward
