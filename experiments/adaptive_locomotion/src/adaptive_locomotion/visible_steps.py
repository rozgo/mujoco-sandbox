"""Reward physical swing-and-landing events, without imposing a gait phase."""

import numpy as np

from .bodies import CONTROL_DT


class VisibleSteps:
    """All intact feet, including healthy bodies; stumps are support only.

    A landing earns quality for height, time clear, and forward replacement.
    Hovering earns no landing reward; low sliding has a dense cost. All state
    here is training bookkeeping, never an actor input or a physical command.
    """

    def __init__(self, n):
        self.contact = np.zeros((n, 4), bool)
        self.active = np.zeros((n, 4), bool)
        self.air = np.zeros((n, 4))
        self.peak = np.zeros((n, 4))
        self.lift = np.zeros((n, 4, 3))
        self.previous = self.lift.copy()

    def reset(self, ids, positions, forces):
        self.contact[ids] = forces[ids] > 5
        self.active[ids] = False
        self.air[ids] = self.peak[ids] = 0
        self.lift[ids] = self.previous[ids] = positions[ids]

    def update(self, positions, forces, radii, valid, direction, moving):
        intact = valid.reshape(-1, 4, 3).all(2)
        contact = forces > np.where(self.contact, 1, 5)
        takeoff = self.contact & ~contact
        self.active[takeoff] = True
        self.lift[takeoff] = self.previous[takeoff]
        self.air[takeoff] = self.peak[takeoff] = 0
        height = positions[..., 2] - radii
        airborne = ~contact & self.active
        self.air += airborne * CONTROL_DT
        self.peak = np.where(airborne, np.maximum(self.peak, height), self.peak)
        touchdown = contact & ~self.contact & self.active
        advance = np.sum((positions - self.lift)[..., :2] * direction[:, None], 2)
        quality = (
            np.clip(self.peak / 0.06, 0, 1)
            * np.clip(self.air / 0.12, 0, 1)
            * np.clip(advance / 0.18, 0, 1)
        )
        landing = 8 * touchdown * (quality - 0.5)
        speed = (
            np.linalg.norm((positions - self.previous)[..., :2], axis=2) / CONTROL_DT
        )
        low_travel = np.clip((0.06 - height) / 0.06, 0, 1) ** 2 * np.tanh(2 * speed)
        reward = ((landing - 2 * low_travel) * intact).sum(1)
        reward /= np.maximum(intact.sum(1), 1)
        self.active[contact] = False
        self.contact = contact
        self.previous[:] = positions
        return reward * moving


def swing_metrics(positions, forces, radius, start=50):
    """Offline completed swings for one leg; force debounce and physical height.

    Independent from reward hysteresis. Ignore partial opening/ending swings.
    A visible swing must peak >=3 cm, clear 1 cm for >=60 ms, last >=80 ms,
    and advance >=8 cm. Return zero counts if a foot never actually steps.
    """
    contact = forces > 1
    contact[1:-1] |= contact[:-2] & contact[2:]
    height = positions[:, 2] - radius
    events = []
    lift = None
    for i in range(max(start, 1), len(contact)):
        if contact[i - 1] and not contact[i]:
            lift = i
        if contact[i] and not contact[i - 1] and lift is not None:
            peak = float(height[lift:i].max())
            clear = float((height[lift:i] >= 0.01).sum() * CONTROL_DT)
            duration = (i - lift) * CONTROL_DT
            advance = float(positions[i, 0] - positions[lift - 1, 0])
            events.append((peak, clear, duration, advance))
            lift = None
    good = [
        p >= 0.03 and c >= 0.06 and t >= 0.08 and a >= 0.08 for p, c, t, a in events
    ]
    return {
        "completed_swings": len(events),
        "visible_swings": sum(good),
        "visible_swing_fraction": float(np.mean(good)) if good else 0.0,
        "mean_swing_peak_m": float(np.mean([e[0] for e in events])) if events else 0.0,
        "mean_swing_clear_1cm_s": float(np.mean([e[1] for e in events]))
        if events
        else 0.0,
    }
