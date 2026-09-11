"""Soft rear swing overlap cost and independent physical phase diagnostics."""

import numpy as np


def overlap_cost(forces, valid, commands):
    """Prefer some rear support after front damage; allow double stance.

    Both rear feet must exist. Front-left/right and calf/whole-leg loss use
    the identical mask. No commanded phase, foot trajectory or actor input.
    """
    intact = valid.reshape(-1, 4, 3).all(2)
    applicable = intact[:, 2:].all(1) & ~intact[:, :2].all(1)
    moving = np.linalg.norm(commands[:, :2], axis=1) > 0.15
    rear_unsupported = ~(forces[:, 2:] > 1).any(1)
    return (applicable & moving & rear_unsupported).astype(float)


def rear_swing_metrics(positions, radii, start=50):
    """One trial's phase from actual upward 1 cm clearance crossings.

    0/1 is synchronous; .5 is alternating. The diagnostic does not use the
    reward's force threshold. Returns no phase if there are no complete cycles.
    """
    clear = positions[:, 2:, 2] - np.asarray(radii)[2:] > 0.01
    rising = [np.flatnonzero(clear[1:, i] & ~clear[:-1, i]) + 1 for i in range(2)]
    phases = []
    for t in rising[1]:
        j = np.searchsorted(rising[0], t, side="right") - 1
        if j >= 0 and j + 1 < len(rising[0]) and rising[0][j] >= start:
            phases.append((t - rising[0][j]) / (rising[0][j + 1] - rising[0][j]))
    mean = np.mean(np.exp(2j * np.pi * np.asarray(phases))) if phases else None
    phase = float(np.angle(mean) / (2 * np.pi) % 1) if mean is not None else None
    return {
        "phase_fraction": phase,
        "phase_concentration": float(abs(mean)) if mean is not None else None,
        "separation_fraction": min(phase, 1 - phase) if phase is not None else None,
        "phase_samples": len(phases),
        "both_rear_feet_above_1cm_fraction": float(clear[start:].all(1).mean()),
    }
