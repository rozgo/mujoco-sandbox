"""Versioned wing position actuation; the measured-motion flight law is unchanged.

MuJoCo supplies local actuator feedback, with the same joint torque cap as v1.
No oscillator, task command, brain bypass or body target enters this module.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from embodied_fly.wing_motion import CONFIG, configure_model


@dataclass(frozen=True)
class WingPositionConfig:
    version: str = "wing_position_v1"
    kp: float = 0.1  # g cm^2 / s^2 / rad
    kv: float = 0.0002  # g cm^2 / s / rad; additional actuator damping


POSITION = WingPositionConfig()


def wing_actuators(model):
    return np.array(
        [
            model.actuator(f"walker/wing_{axis}_{side}").id
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        ]
    )


def is_position(model):
    return bool(np.all(model.actuator_biasprm[wing_actuators(model), 1] < 0))


def configure_position(model):
    configure_model(model)
    ids = wing_actuators(model)
    joints = model.actuator_trnid[ids, 0]
    if not np.all(model.actuator_gear[ids, 0] == 1) or not np.all(model.jnt_limited[joints]):
        raise ValueError("Position pilot requires limited unit-gear wing hinges")
    model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[ids] = 0
    model.actuator_gainprm[ids, 0] = POSITION.kp
    model.actuator_biasprm[ids] = 0
    model.actuator_biasprm[ids, 1] = -POSITION.kp
    model.actuator_biasprm[ids, 2] = -POSITION.kv
    model.actuator_ctrllimited[ids] = True
    model.actuator_ctrlrange[ids] = model.jnt_range[joints]
    mujoco.mj_setConst(model, mujoco.MjData(model))


def normalize_targets(model, targets):
    limits = model.actuator_ctrlrange[wing_actuators(model)]
    return (
        2
        * (np.clip(targets, limits[:, 0], limits[:, 1]) - limits[:, 0])
        / (limits[:, 1] - limits[:, 0])
        - 1
    ).astype(np.float32)


def reference_torque_to_position(model, normalized_torque, angles, velocities, interval=0.0):
    """Training-only reference conversion. Never called in the deployed actor.

    Match the reference torque at the measured state, subject to position limits.
    A half-interval velocity lead compensates the held position target's lag.
    This is reference supervision, never a deployed extrapolator.
    """
    if not np.isfinite(interval) or interval < 0:
        raise ValueError("Finite nonnegative reference hold interval required")
    target = (
        np.asarray(angles)
        + (
            np.asarray(normalized_torque) * CONFIG.joint_torque_limit
            + POSITION.kv * np.asarray(velocities)
        )
        / POSITION.kp
        + 0.5 * interval * np.asarray(velocities)
    )
    return normalize_targets(model, target)


def report():
    return {
        **asdict(POSITION),
        "torque_limit_CGS": CONFIG.joint_torque_limit,
        "command": "six normalized absolute wing angle targets",
        "force": "clip(kp * (target - angle) - kv * angular_velocity, +/- torque_limit)",
        "target_limits": "unchanged mechanical joint ranges",
        "task_dependent_actuation": False,
        "flight_force_law": CONFIG.version,
    }
