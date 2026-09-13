"""Training-only hover reference from measured state, without a phase clock.

The real wing coordinates supply oscillation phase. This code produces teaching
commands; it does not alter MuJoCo physics or run inside the deployed actor.
"""

import numpy as np

from embodied_fly.wing_motion import CONFIG

RECIPE = {
    "frequency_hz": 12.0,
    "energy_gain_per_s": 60.0,
    "energy_floor_rad": 0.05,
    "startup_radius_fraction": 0.1,
    "startup_velocity_time_s": 0.01,
    "amplitude_height_gain_rad_per_cm": 0.4,
    "amplitude_vertical_speed_gain": 0.02,
    "clock_or_external_phase": False,
    "deployed_module": False,
}


def wing_commands(angles, velocities, body_velocity, altitude, requested_altitude, spring):
    """All inputs are current observations, except fixed mechanical constants."""
    c = CONFIG
    q = np.asarray(angles).reshape(-1, 2, 3)
    v = np.asarray(velocities).reshape(-1, 2, 3)
    body_velocity = np.asarray(body_velocity)
    omega = 2 * np.pi * RECIPE["frequency_hz"]
    amplitude = np.clip(
        c.reference_sweep_speed / (c.lift_weight_multiplier * 4 * RECIPE["frequency_hz"])
        + RECIPE["amplitude_height_gain_rad_per_cm"] * (requested_altitude - altitude)
        - RECIPE["amplitude_vertical_speed_gain"] * body_velocity[:, 5],
        0.12,
        1.15,
    )[:, None]
    sweep, speed = q[:, :, 0], v[:, :, 0]
    radius2 = sweep**2 + (speed / omega) ** 2
    energy = (amplitude**2 - radius2) / (radius2 + RECIPE["energy_floor_rad"] ** 2)
    acceleration = -(omega**2) * sweep + RECIPE["energy_gain_per_s"] * energy * speed
    # A resting wing has no phase. A bounded, state-triggered positive startup
    # command leaves that equilibrium; no elapsed time enters the decision.
    starting = radius2 < (RECIPE["startup_radius_fraction"] * amplitude) ** 2
    acceleration = np.where(
        starting, (omega * amplitude - speed) / RECIPE["startup_velocity_time_s"], acceleration
    )
    desired = np.zeros_like(q)
    desired[:, :, 0] = sweep
    desired[:, :, 1] = (0.7 + np.clip(-0.1 * body_velocity[:, 3], -0.6, 0.6))[:, None]
    desired[:, :, 2] = -1
    spring = np.asarray(spring).reshape(1, 2, 3)
    torque = 0.02 * (desired - q) - 0.00015 * v
    torque[:, :, 0] = (
        c.angular_armature * acceleration
        + c.joint_damping * speed
        + c.joint_stiffness * (sweep - spring[:, :, 0])
    )
    return np.clip(torque.reshape(-1, 6) / c.joint_torque_limit, -1, 1).astype(np.float32)
