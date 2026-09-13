"""Versioned causal wing-speed input, alongside unchanged walking observations."""

import numpy as np

WING_JOINTS = tuple(
    f"walker/wing_{axis}_{side}"
    for side in ("left", "right")
    for axis in ("yaw", "roll", "pitch")
)
WING_SPEED_SCALE_RAD_S = 2000.0


def wing_velocity_indices(model):
    return np.array([model.jnt_dofadr[model.joint(name).id] for name in WING_JOINTS])


def append_wing_velocity(observation, qvel, indices):
    """No velocity clipping here; encoder safety bound is ±20,000 rad/s."""
    velocity = np.asarray(qvel)[..., indices] / WING_SPEED_SCALE_RAD_S
    if not np.isfinite(velocity).all():
        raise ValueError("Nonfinite measured wing speed")
    return np.concatenate((observation, velocity), axis=-1).astype(np.float32)
