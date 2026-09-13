"""Continuous causal wing feedback alongside unchanged walking observations."""

import numpy as np

WING_JOINTS = tuple(
    f"walker/wing_{axis}_{side}"
    for side in ("left", "right")
    for axis in ("yaw", "roll", "pitch")
)
WING_SPEED_SCALE_RAD_S = 2000.0
WING_ANGLE_SCALE_RAD = np.pi


def wing_velocity_indices(model):
    return np.array([model.jnt_dofadr[model.joint(name).id] for name in WING_JOINTS])


def wing_angle_indices(model):
    return np.array([model.jnt_qposadr[model.joint(name).id] for name in WING_JOINTS])


def append_wing_velocity(observation, qvel, indices):
    """No velocity clipping here; encoder safety bound is ±20,000 rad/s."""
    velocity = np.asarray(qvel)[..., indices] / WING_SPEED_SCALE_RAD_S
    if not np.isfinite(velocity).all():
        raise ValueError("Nonfinite measured wing speed")
    return np.concatenate((observation, velocity), axis=-1).astype(np.float32)


def append_wing_angles(observation, qpos, indices):
    """Measured hinge angles / pi, without the walking normalizer's saturation."""
    angles = np.asarray(qpos)[..., indices] / WING_ANGLE_SCALE_RAD
    if not np.isfinite(angles).all():
        raise ValueError("Nonfinite measured wing angle")
    return np.concatenate((observation, angles), axis=-1).astype(np.float32)


HEIGHT_SCALE_CM = 2.0


def append_height(observation, qpos, requested_height_cm):
    """Ideal root altitude above the z=0 floor and a current command, both / 2 cm.

    This is simulator sensing, not image perception or a biological organ model.
    The requested height is an explicit current task input, never a future state.
    """
    height = np.asarray(qpos)[..., 2]
    target = np.broadcast_to(np.asarray(requested_height_cm), height.shape)
    measured = np.stack((height, target), axis=-1) / HEIGHT_SCALE_CM
    if not np.isfinite(measured).all():
        raise ValueError("Nonfinite altitude or requested height")
    return np.concatenate((observation, measured), axis=-1).astype(np.float32)


def actor_observation(env, actor):
    """Load each preserved schema explicitly; extensions always keep the old prefix."""
    extension = actor.sensor_extension_size
    if extension not in (0, 6, 12, 14):
        raise ValueError("Unsupported live actor observation schema")
    return env.observation(
        extension >= 6, wing_angles=extension >= 12, height_inputs=extension == 14
    )
