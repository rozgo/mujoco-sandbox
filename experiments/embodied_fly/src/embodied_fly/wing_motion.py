"""Flight dynamics driven by measured wing motion, in FlyBody CGS units.

The six wing hinges retain bounded actuation and sensory feedback. Their diagonal
armature defines an independent angular response; zero wing-body mass/inertia
removes inertial coupling to the thorax. A causal force law is the only coupling.
No time, command, target, policy action or externally supplied phase enters it.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class WingMotionConfig:
    version: str = "wing_motion_v1"
    angular_armature: float = 2e-6  # g cm^2; independent wing actuator response
    joint_damping: float = 0.0002  # g cm^2 / s
    joint_stiffness: float = 0.001  # g cm^2 / s^2 / rad
    joint_torque_limit: float = 0.03  # g cm^2 / s^2
    activity_filter_seconds: float = 0.012
    reference_sweep_speed: float = 50.0  # rad/s, per wing
    maximum_activity: float = 1.4
    lift_weight_multiplier: float = 1.6  # max lift 2.24 body weights
    horizontal_drag_seconds: float = 0.12
    vertical_drag_seconds: float = 0.08
    angular_drag_seconds: float = 0.025
    attitude_stiffness: float = 100.0  # angular acceleration / up-vector error
    steering_acceleration: float = 60.0  # rad/s^2 per activity difference
    maximum_angular_acceleration: float = 250.0
    inertia_radius_cm: float = 0.06
    forward_force_fraction: float = 0.25


CONFIG = WingMotionConfig()


def configure_model(model):
    """Apply after compilation, including mjbatch's sensor-augmented compilation.

    MuJoCo's compiler requires positive inertials on moving bodies. The compiled
    model supports massless hinges with positive diagonal armature. This explicit
    postcompile configuration is saved in MJB and tested for positive mass matrix
    and exactly zero wing/non-wing inertial coupling. Recompiling XML requires this
    function again. Original aerodynamic and walking models do not call it.
    """
    bodies = [model.body(f"walker/wing_{side}").id for side in ("left", "right")]
    joints = [
        model.joint(f"walker/wing_{axis}_{side}").id
        for side in ("left", "right")
        for axis in ("yaw", "roll", "pitch")
    ]
    dofs = model.jnt_dofadr[joints]
    model.body_mass[bodies] = 0
    model.body_inertia[bodies] = 0
    model.dof_armature[dofs] = CONFIG.angular_armature
    model.dof_damping[dofs] = CONFIG.joint_damping
    model.jnt_stiffness[joints] = CONFIG.joint_stiffness
    model.jnt_solref[joints] = (0.001, 1)
    # Wing surfaces are visible, with no aerodynamic or contact coupling.
    geoms = np.isin(model.geom_bodyid, bodies)
    model.geom_contype[geoms] = 0
    model.geom_conaffinity[geoms] = 0
    model.geom_fluid[:] = 0
    model.opt.density = model.opt.viscosity = 0
    for j in joints:
        actuator = model.actuator(model.joint(j).name)
        model.actuator_gainprm[actuator.id, 0] = CONFIG.joint_torque_limit
        model.actuator_forcelimited[actuator.id] = True
        model.actuator_forcerange[actuator.id] = (
            -CONFIG.joint_torque_limit,
            CONFIG.joint_torque_limit,
        )
    mujoco.mj_setConst(model, mujoco.MjData(model))


class WingMotionForces:
    """One independent causal activity state per world; no cross-world mixing."""

    def __init__(self, model, worlds=1):
        self.config = CONFIG
        self.mass = float(model.body_mass.sum())
        self.weight = self.mass * abs(float(model.opt.gravity[2]))
        self.inertia = self.mass * CONFIG.inertia_radius_cm**2
        self.activity = np.zeros((worlds, 2))
        self.wrench = np.zeros((worlds, 6))  # world force then world torque
        self.lift = np.zeros(worlds)

    def reset(self, ids):
        self.activity[ids] = 0
        self.wrench[ids] = 0
        self.lift[ids] = 0

    def advance(self, angles, velocities, rotation, body_velocity, dt):
        """Inputs have [world, ...] axes; velocity is angular/linear in body axes."""
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Positive finite physical interval required")
        angles = np.asarray(angles).reshape(-1, 2, 3)
        velocities = np.asarray(velocities).reshape(-1, 2, 3)
        rotation = np.asarray(rotation).reshape(-1, 3, 3)
        body_velocity = np.asarray(body_velocity).reshape(-1, 6)
        if len(angles) != len(self.activity) or not all(
            np.isfinite(x).all() for x in (angles, velocities, rotation, body_velocity)
        ):
            raise ValueError("Invalid wing/body state")
        c = self.config
        sweep = np.clip(
            np.abs(velocities[:, :, 0]) / c.reference_sweep_speed,
            0,
            c.maximum_activity,
        )
        alpha = -np.expm1(-dt / c.activity_filter_seconds)
        self.activity += alpha * (sweep - self.activity)
        # Pitch changes stroke effectiveness; sweep must move to produce lift.
        efficiency = 0.8 + 0.2 * np.cos(angles[:, :, 2] + 1.0)
        effort = self.activity * efficiency
        collective = effort.mean(axis=1)
        differential = effort[:, 1] - effort[:, 0]
        self.lift[:] = c.lift_weight_multiplier * self.weight * collective
        engagement = np.clip(collective * 2, 0, 1)
        # Roll-axis stroke orientation modulates forward thrust, relative to the
        # inherited 0.7 rad folded reference. All steering is wing-state derived.
        forward = np.tanh((angles[:, :, 1].mean(axis=1) - 0.7) * 2)
        force_local = np.zeros((len(angles), 3))
        force_local[:, 0] = self.lift * c.forward_force_fraction * forward
        # Upright-biased thrust and a damped restoring torque are declared body
        # response choices. They provide no translational target or hover servo.
        force_world = np.einsum("nij,nj->ni", rotation, force_local)
        force_world += self.lift[:, None] * (0.2 * rotation[:, :, 2])
        force_world[:, 2] += 0.8 * self.lift
        linear_world = np.einsum("nij,nj->ni", rotation, body_velocity[:, 3:])
        force_world -= (
            self.mass
            * engagement[:, None]
            * linear_world
            / (c.horizontal_drag_seconds, c.horizontal_drag_seconds, c.vertical_drag_seconds)
        )
        angular_world = np.einsum("nij,nj->ni", rotation, body_velocity[:, :3])
        acceleration = -angular_world / c.angular_drag_seconds
        acceleration += c.attitude_stiffness * np.cross(rotation[:, :, 2], (0, 0, 1))
        steer_local = np.stack(
            [
                -c.steering_acceleration * differential,
                np.zeros_like(differential),
                c.steering_acceleration * differential,
            ],
            axis=1,
        )
        acceleration += np.einsum("nij,nj->ni", rotation, steer_local)
        norm = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(
            1, c.maximum_angular_acceleration / np.maximum(norm, 1e-12)
        )[:, None]
        self.wrench[:, :3] = force_world
        self.wrench[:, 3:] = self.inertia * engagement[:, None] * acceleration
        return self.wrench

    def report(self):
        return {
            "name": "Flight dynamics driven by measured wing motion",
            "config": asdict(self.config),
            "inputs": "six actual wing angles and speeds; measured body orientation/velocity",
            "body_mass_kg": self.mass * 0.001,
            "wing_mass_and_body_inertial_coupling": "zero; independent diagonal angular response",
            "aerodynamic_force_model": False,
            "causal_activity_state_shape": list(self.activity.shape),
            "attitude_assistance": "activity-dependent angular damping and restoring torque",
            "translation": "bounded wing-motion lift/thrust and drag; no target-position servo",
        }
