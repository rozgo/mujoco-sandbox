"""Velocity commands and a training-only PID on the unchanged flight plant.

No position goal, supplied neural clock, historical policy or observation
normalizer is part of the new student interface. Coordinates use heading-relative
forward/left and world-up; the same convention applies to command and feedback.
"""

import numpy as np

from embodied_fly.pid_hover import HoverPID, PIDConfig

SCHEMA = "flight_velocity_motor_v1"
OBSERVATION_SIZE = 391
COMMAND_SCALE_CM_S = 1.0


def heading_rotation(rotation):
    rotation = np.asarray(rotation).reshape(-1, 3, 3)
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    result = np.zeros_like(rotation)
    result[:, 0, 0] = result[:, 1, 1] = np.cos(yaw)
    result[:, 1, 0] = np.sin(yaw)
    result[:, 0, 1] = -np.sin(yaw)
    result[:, 2, 2] = 1
    return result


def measured_velocity(env):
    rotation = heading_rotation(env.fields["xmat"][:, env.template.thorax_id])
    return np.einsum("nji,nj->ni", rotation, env.fields["qvel"][:, :3])


def observation(env, command_cm_s):
    """Fresh 391-feature schema; no position/needs inputs or legacy normalizer.

    0:282 joints + effective actuator inputs; 282:285 body angular velocity;
    285:288 measured velocity; 288:375 upright/contact/previous joint actions;
    375:379 requested translation/yaw velocity; 379:391 measured wing speeds/angles.
    """
    command = np.asarray(command_cm_s)
    if command.shape != (env.n, 4) or not np.isfinite(command).all():
        raise ValueError("Expected four finite translation/yaw velocity commands per world")
    if env.sensor_extension_size < 12:
        raise ValueError("The velocity schema requires measured wing angles and speeds")
    old = env.observation()
    result = np.concatenate(
        (
            old[:, :282],
            env.velocity()[:, :3] / 20,
            measured_velocity(env) / COMMAND_SCALE_CM_S,
            old[:, 288:375],
            command / COMMAND_SCALE_CM_S,
            old[:, 383:395],
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (env.n, OBSERVATION_SIZE) or not np.isfinite(result).all():
        raise ValueError("Invalid velocity observation layout")
    return result


class VelocityPID:
    """PI velocity feedback with the accepted PID's bounded wing actuation.

    The derivative coefficient is zero. Integrating velocity error supplies bias
    rejection; no absolute position or position target enters this controller.
    PID phase/integral are teacher state only, never student observations.
    """

    def __init__(self, env, base_action, *, motion_feedforward=False):
        self.env = env
        self.motion_feedforward = motion_feedforward
        self.wings = HoverPID(
            env,
            base_action,
            np.zeros(3),
            PIDConfig(roll_kp=600.0, roll_kd=40.0)
            if motion_feedforward
            else PIDConfig(roll_kp=300.0, roll_kd=35.0),
        )
        self.integral = np.zeros(3)
        self.yaw_integral = 0.0
        # Stronger horizontal velocity damping and faster roll correction handle
        # braking after yaw motion; vertical gains retain the accepted response.
        self.kp = np.array([20.0, 20.0, 50.0])
        self.ki = np.array([20.0, 20.0, 900.0])
        if motion_feedforward:
            self.kp[:2] = 40.0
            self.ki[:2] = 200.0
        self.previous_requested_world = np.zeros(3)
        self.previous_yaw_command = 0.0
        self.yaw_kp = 60.0 if motion_feedforward else 20.0
        self.yaw_ki = 30.0

    def reset(self):
        self.integral[:] = 0
        self.yaw_integral = 0.0
        self.previous_requested_world[:] = 0
        self.previous_yaw_command = 0.0

    def act(self, command_cm_s):
        command = np.asarray(command_cm_s, dtype=np.float64)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("Expected finite forward/left/up/yaw velocity commands")
        e = self.env
        rotation = heading_rotation(e.data.xmat[e.thorax_id])[0]
        requested_world = rotation @ command[:3]
        feedforward = np.zeros(3)
        yaw_feedforward = 0.0
        if self.motion_feedforward:
            law = e.wing_forces.config
            # Causal command differences include both speed changes and the
            # changing world direction of heading-relative velocity during yaw.
            # Counter declared resistance with wing thrust, never direct forces.
            feedforward = (requested_world - self.previous_requested_world) / e.control_dt
            feedforward += requested_world / np.array(
                [
                    law.horizontal_drag_seconds,
                    law.horizontal_drag_seconds,
                    law.vertical_drag_seconds,
                ]
            )
            yaw_feedforward = (command[3] - self.previous_yaw_command) / e.control_dt
        self.previous_requested_world[:] = requested_world
        self.previous_yaw_command = float(command[3])
        error = requested_world - e.data.qvel[:3]
        proposed = np.clip(self.integral + error * e.control_dt, -1, 1)
        raw = feedforward + self.kp * error + self.ki * proposed
        accept = (np.abs(raw) <= 300) | (raw * error < 0)
        self.integral[accept] = proposed[accept]
        acceleration = np.clip(
            feedforward + self.kp * error + self.ki * self.integral, -300, 300
        )
        yaw_error = command[3] - e.anatomical_velocity()[2]
        self.yaw_integral = float(
            np.clip(self.yaw_integral + yaw_error * e.control_dt, -0.5, 0.5)
        )
        yaw_cap = 60 if self.motion_feedforward else 30
        yaw_acceleration = np.clip(
            yaw_feedforward + self.yaw_kp * yaw_error + self.yaw_ki * self.yaw_integral,
            -yaw_cap,
            yaw_cap,
        )
        return self.wings.action_for_acceleration(acceleration, yaw_acceleration)


def schema_report():
    return {
        "schema": SCHEMA,
        "observation_size": OBSERVATION_SIZE,
        "command": ["forward_cm_s", "left_cm_s", "up_cm_s", "yaw_rad_s"],
        "frame": "heading-relative horizontal axes; world vertical",
        "command_indices": [375, 376, 377, 378],
        "command_scale_cm_s": COMMAND_SCALE_CM_S,
        "yaw_command_scale_rad_s": 1.0,
        "position_target_inputs": False,
        "absolute_position_inputs": False,
        "needs_inputs": False,
        "teacher_phase_or_integral_inputs": False,
        "zero_command": "actively brake translation and keep supporting weight",
        "actions": "78 bounded physical joint/adhesion commands, including six wing angle targets",
        "return_to_origin_required": False,
        "recurrent_controller": "same fixed MaleCNS graph; fresh trainable parameters",
    }
