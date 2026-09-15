"""PID plant reference: bounded wing targets, instantaneous forces, free body.

This reference has a declared wing trajectory clock. It is not an RL actor,
and neither this controller nor its phase is inserted into a learned policy.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.body import FlyEnvironment
from embodied_fly.motion_flight import initialize
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.wing_position import POSITION, normalize_targets, wing_actuators


@dataclass(frozen=True)
class PIDConfig:
    frequency_hz: float = 30.0
    height_kp: float = 900.0
    height_ki: float = 1600.0
    height_kd: float = 50.0
    horizontal_kp: float = 15.0
    horizontal_ki: float = 3.0
    horizontal_kd: float = 5.0
    integral_limit_cm_s: float = 1.0
    acceleration_limit_cm_s2: float = 300.0
    roll_kp: float = 100.0
    roll_kd: float = 15.0
    wing_velocity_feedback: float = 0.0008

    def __post_init__(self):
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or min(self.frequency_hz, self.integral_limit_cm_s, self.acceleration_limit_cm_s2)
            <= 0
        ):
            raise ValueError(
                "PID settings must be finite and nonnegative with positive rate/limits"
            )


class HoverPID:
    def __init__(self, env, base_action, target, config=None):
        self.env, self.base, self.target = env, base_action.copy(), np.asarray(target).copy()
        self.config = config or PIDConfig()
        self.integral = np.zeros(3)
        self.channels = wing_actuators(env.model)
        self.amplitudes = np.zeros(2)
        self.desired_acceleration = np.zeros(3)

    def act(self):
        e, c = self.env, self.config
        dt = e.control_dt
        error = self.target - e.data.qpos[:3]
        velocity = e.data.qvel[:3].copy()  # free-root linear velocity in world axes
        kp = np.array([c.horizontal_kp, c.horizontal_kp, c.height_kp])
        ki = np.array([c.horizontal_ki, c.horizontal_ki, c.height_ki])
        kd = np.array([c.horizontal_kd, c.horizontal_kd, c.height_kd])
        proposed_integral = np.clip(
            self.integral + error * dt, -c.integral_limit_cm_s, c.integral_limit_cm_s
        )
        raw = kp * error + ki * proposed_integral - kd * velocity
        # Conditional integration: do not wind up while demanding beyond the cap.
        accept = (np.abs(raw) <= c.acceleration_limit_cm_s2) | (raw * error < 0)
        self.integral[accept] = proposed_integral[accept]
        acceleration = np.clip(
            kp * error + ki * self.integral - kd * velocity,
            -c.acceleration_limit_cm_s2,
            c.acceleration_limit_cm_s2,
        )
        return self.action_for_acceleration(acceleration)

    def action_for_acceleration(self, acceleration, yaw_acceleration=None):
        """Shared bounded wing mapping; no position/velocity goal is read here."""
        e, c, law = self.env, self.config, self.env.wing_forces.config
        dt = e.control_dt
        acceleration = np.asarray(acceleration, dtype=np.float64)
        if acceleration.shape != (3,) or not np.isfinite(acceleration).all():
            raise ValueError("Expected a finite world-frame acceleration vector")
        self.desired_acceleration[:] = acceleration
        rotation = e.data.xmat[e.thorax_id].reshape(3, 3)
        angular = e.anatomical_velocity()[:3]
        gravity = abs(e.model.opt.gravity[2])
        # Weight feedforward. The plant retains its passive drag; velocity
        # callers may request wing thrust to overcome that resistance.
        vertical = gravity + acceleration[2]
        collective = np.clip(
            vertical / (gravity * law.lift_weight_multiplier * (0.8 + 0.2 * rotation[2, 2])),
            0.1,
            1.15,
        )
        lift_acceleration = gravity * law.lift_weight_multiplier * collective
        reduced = law.version == "wing_motion_reduced_coupling_v6"
        thrust_acceleration = (
            gravity * np.clip(2 * collective, 0, 1) if reduced else lift_acceleration
        )
        desired_horizontal = acceleration[:2]
        forward_axis = rotation[:2, 0]
        side_axis = rotation[:2, 1]
        # Counter body tilt and command forward acceleration via stroke orientation.
        horizontal_request = (
            (desired_horizontal - 0.2 * lift_acceleration * rotation[:2, 2])
            / thrust_acceleration
            if reduced
            else desired_horizontal / lift_acceleration - 0.2 * rotation[:2, 2]
        )
        forward = np.dot(horizontal_request, forward_axis) / max(
            np.dot(forward_axis, forward_axis), 0.5
        )
        stroke = 0.7 + 0.5 * np.arctanh(
            np.clip(forward / law.forward_force_fraction, -0.8, 0.8)
        )
        # Earlier plants use roll for lateral force. The agile plant additionally
        # allows direct side thrust through differential stroke orientation.
        roll_target = np.clip(
            -np.dot(desired_horizontal, side_axis) / (0.2 * lift_acceleration), -0.2, 0.2
        )
        stroke_offset = 0.0
        if getattr(law, "lateral_force_fraction", 0):
            side = np.dot(horizontal_request, side_axis) / max(
                np.dot(side_axis, side_axis), 0.5
            )
            stroke_offset = 0.25 * np.arctanh(
                np.clip(side / law.lateral_force_fraction, -0.8, 0.8)
            )
            roll_target = 0.0
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        desired_roll_acc = c.roll_kp * (roll_target - roll) - c.roll_kd * angular[0]
        restoring_world = law.attitude_stiffness * np.cross(rotation[:, 2], (0, 0, 1))
        restoring_roll = np.dot(restoring_world, rotation[:, 0])
        differential = np.clip(
            (-angular[0] / law.angular_drag_seconds + restoring_roll - desired_roll_acc)
            / law.steering_acceleration,
            -0.25,
            0.25,
        )
        efforts = np.clip(collective + np.array([-0.5, 0.5]) * differential, 0.05, 1.3)
        amplitude = efforts * law.reference_sweep_speed / (4 * c.frequency_hz)
        self.amplitudes[:] = amplitude
        omega = 2 * np.pi * c.frequency_hz
        phase = omega * (e.data.time + dt / 2)
        desired = np.column_stack((amplitude * np.sin(phase), np.full(2, stroke), -np.ones(2)))
        desired[:, 1] += (-stroke_offset, stroke_offset)
        if yaw_acceleration is not None:
            authority = getattr(law, "yaw_pitch_acceleration", 0)
            if not authority or not np.isfinite(yaw_acceleration):
                raise ValueError("Independent yaw request requires the heading-control plant")
            restoring_yaw = np.dot(restoring_world, rotation[:, 2])
            yaw_effort = (
                yaw_acceleration
                + angular[2] / getattr(law, "yaw_drag_seconds", law.angular_drag_seconds)
                - restoring_yaw
                - (0 if reduced else law.steering_acceleration * differential)
            ) / authority
            offset = 0.5 * np.arcsin(np.clip(yaw_effort, -np.sin(0.5), np.sin(0.5)))
            desired[:, 2] += (-offset, offset)
        speed = np.zeros((2, 3))
        speed[:, 0] = amplitude * omega * np.cos(phase)
        wing_acceleration = np.zeros((2, 3))
        wing_acceleration[:, 0] = -amplitude * omega**2 * np.sin(phase)
        spring = e.model.qpos_spring[e.wing_angle_indices].reshape(2, 3)
        # Feed forward the wing joint's own inertia, spring and damping. MuJoCo
        # still applies its position feedback, joint limits and torque cap.
        target = (
            desired
            + (
                law.angular_armature * wing_acceleration
                + (law.joint_damping + POSITION.kv) * speed
                + law.joint_stiffness * (desired - spring)
                + c.wing_velocity_feedback
                * (speed - e.data.qvel[e.wing_velocity_indices].reshape(2, 3))
            )
            / POSITION.kp
        )
        action = self.base.copy()
        action[self.channels] = normalize_targets(e.model, target.reshape(6))
        return action


def run(args):
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        raise ValueError("Positive finite capture duration required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    env = FlyEnvironment("wing_position", wing_response="instant", physics_hz=args.physics_hz)
    base = initialize(env, args.height, args.heading)
    env.requested_height_cm = args.height
    target = env.data.qpos[:3].copy()
    config = PIDConfig(
        frequency_hz=args.frequency,
        height_kp=args.kp,
        height_ki=args.ki,
        height_kd=args.kd,
        wing_velocity_feedback=args.wing_kd,
    )
    controller = HoverPID(env, base, target, config)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    rows, actual = [], []
    started = time.perf_counter()
    failure = None
    for _ in range(round(args.seconds / env.control_dt)):
        action = controller.act()
        rows.append(
            {
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "activation": env.data.act.copy(),
                "ctrl": env.data.ctrl.copy(),
                "action": action.copy(),
                "time": env.data.time,
                "requested_height_cm": args.height,
                "wing_activity": env.wing_forces.activity[0].copy(),
                "wing_wrench": env.wing_forces.wrench[0].copy(),
                "pid_integral": controller.integral.copy(),
                "pid_acceleration": controller.desired_acceleration.copy(),
                "wing_amplitude": controller.amplitudes.copy(),
            }
        )
        try:
            env.step(action)
        except RuntimeError as error:
            failure = str(error)
            break
        actual.append(
            {
                "position": env.data.qpos[:3].copy(),
                "velocity": env.data.qvel[:3].copy(),
                "upright": env.data.xmat[env.thorax_id, 8],
                "wing_torque_peak": abs(env.data.actuator_force[controller.channels]).max(),
            }
        )
    elapsed = time.perf_counter() - started
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(args.output / "hover.npz", **arrays)
    measured = {k: np.stack([row[k] for row in actual]) for k in actual[0]}
    windows = {}
    for name, mask in (
        ("whole_capture", np.ones(len(actual), bool)),
        ("after_first_second", np.arange(len(actual)) * env.control_dt >= 1),
    ):
        if not mask.any():
            continue
        q, v = measured["position"][mask], measured["velocity"][mask]
        windows[name] = {
            "height_span_mm": float(np.ptp(q[:, 2]) * 10),
            "altitude_rms_mm": float(np.sqrt(np.mean((q[:, 2] - args.height) ** 2)) * 10),
            "position_rms_mm": float(np.sqrt(np.mean(np.sum((q - target) ** 2, axis=1))) * 10),
            "position_peak_mm": float(np.linalg.norm(q - target, axis=1).max() * 10),
            "vertical_speed_rms_mm_s": float(np.sqrt(np.mean(v[:, 2] ** 2)) * 10),
            "minimum_upright": float(measured["upright"][mask].min()),
        }
    whole, settled = windows["whole_capture"], windows.get("after_first_second", {})
    stable = measured["upright"].min() > 0.99 and measured["position"][:, 2].min() > 0.5
    support = env.maximum_disallowed_ground_force / env.wing_forces.weight
    # Tight demonstration gates; do not substitute old loose flight survival.
    passed = bool(
        stable
        and failure is None
        and support == 0
        and settled
        and settled["height_span_mm"] < 0.2
        and settled["position_peak_mm"] < 0.5
        and settled["vertical_speed_rms_mm_s"] < 15
    )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "mode": "reference",
        "controller": "reference",
        "student_present": False,
        "motor_only": False,
        "reference_kind": "PID hover plant controller",
        "pid": asdict(config),
        "reference_phase_clock": True,
        "physical_preset": env.preset,
        "physical_contract": physical_contract(env.model),
        "force_model": env.wing_forces.report(),
        "environment": env.report(),
        "physics_hz": 1 / env.model.opt.timestep,
        "control_hz": 1 / env.control_dt,
        "setup_seconds": setup,
        "stepping_and_capture_seconds": elapsed,
        "parallel_physics_worlds": 1,
        "simulated_seconds_per_world": len(actual) * env.control_dt,
        "physical_transitions": len(actual),
        "training_seconds": 0,
        "optimizer_updates": 0,
        "model_sha256": sha256(args.output / "model.mjb"),
        "checkpoint_sha256": None,
        "warning_count": int(env.data.warning.number.sum()),
        "numerical_failure": failure,
        "windows": windows,
        "initial_position_cm": target.tolist(),
        "final_position_cm": env.data.qpos[:3].tolist(),
        "maximum_wing_torque_CGS": float(measured["wing_torque_peak"].max()),
        "body_force_or_pose_control": False,
        "scope": "reference controller, airborne start; not learned flight or takeoff",
        "results": [
            {
                "case": "hover",
                "stable": bool(stable),
                "success": passed,
                "root_tracking_rmse_m": whole["position_rms_mm"] / 1000,
                "max_forbidden_ground_force_over_weight": float(support),
            }
        ],
        "gates": {
            "settled_height_span_mm": 0.2,
            "settled_position_peak_mm": 0.5,
            "settled_vertical_speed_rms_mm_s": 15,
            "minimum_upright": 0.99,
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "windows",
                    "results",
                    "stepping_and_capture_seconds",
                    "numerical_failure",
                )
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--physics-hz", type=float, default=1000)
    p.add_argument("--height", type=float, default=2)
    p.add_argument("--heading", type=float, default=0)
    p.add_argument("--frequency", type=float, default=30)
    p.add_argument("--kp", type=float, default=900)
    p.add_argument("--ki", type=float, default=1600)
    p.add_argument("--kd", type=float, default=50)
    p.add_argument("--wing-kd", type=float, default=0.0008)
    args = p.parse_args()
    result = run(args)
    raise SystemExit(0 if result["results"][0]["success"] else 2)
