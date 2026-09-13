"""Training-only inherited flight expert on our complete, physically free body.

The upstream expert uses a wingbeat generator. Neither it nor this adapter is a
deployed MaleCNS student. The adapter supplies complete bounded motor labels.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from flybody.fly_envs import flight_imitation
from flybody.quaternions import get_dquat_local
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

from embodied_fly.body import FlyEnvironment
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.teacher import ConvertedTeacher


class FlightTeacherOracle:
    def __init__(self, env, checkpoint, wing_pattern):
        if env.preset != "flight":
            raise ValueError("Flight physical preset required")
        self.env = env
        self.policy = ConvertedTeacher(checkpoint).eval()
        template = flight_imitation(wpg_pattern_path=str(wing_pattern))
        template.reset()
        fly, task = template.task.walker, template.task
        self.joints = [env.model.joint("walker/" + j.name).id for j in fly.observable_joints]
        names = template.action_spec().name.split("\t")
        self.user_index = task._user_idx_action
        self.physical_action_indices = [i for i in range(len(names)) if i != self.user_index]
        self.actuators = [
            env.model.actuator("walker/" + names[i]).id for i in self.physical_action_indices
        ]
        self.wing_indices = task._wing_inds_action
        self.wing_joints = [
            env.model.joint("walker/" + names[i]).id for i in self.wing_indices
        ]
        self.pattern = WingBeatPatternGenerator(
            base_pattern_path=wing_pattern, dt_ctrl=env.control_dt
        )
        self.sensors = {}
        for kind in ("accelerometer", "gyro", "velocimeter"):
            indices = []
            for sensor in getattr(fly.mjcf_model.sensor, kind):
                sid = env.model.sensor("walker/" + sensor.name).id
                indices.extend(
                    range(
                        env.model.sensor_adr[sid],
                        env.model.sensor_adr[sid] + env.model.sensor_dim[sid],
                    )
                )
            self.sensors[kind] = np.array(indices)
        self.leg_joints = [
            j
            for j in env.joint_ids
            if any(s in env.model.joint(j).name for s in ("coxa", "femur", "tibia", "tarsus"))
        ]
        self.retracted_controls = np.zeros(env.model.nu)
        for i in range(env.model.nu):
            if env.model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = env.model.actuator_trnid[i, 0]
            if jid in self.leg_joints:
                self.retracted_controls[i] = env.model.qpos_spring[env.model.jnt_qposadr[jid]]

    def initialize(self, speed, seconds, phase=0):
        env = self.env
        self.reference, velocity = constant_speed_trajectory(
            n_steps=round(seconds / env.control_dt) + 10,
            speed=speed,
            body_rot_angle_y=-47.5,
            init_pos=(0, 0, 1),
            control_timestep=env.control_dt,
        )
        env.reset()
        env.command[0] = speed
        env.data.qpos[:7] = self.reference[0]
        env.data.qvel[:6] = velocity[0]
        addresses = env.model.jnt_qposadr[self.leg_joints]
        env.data.qpos[addresses] = env.model.qpos_spring[addresses]
        wing_q, wing_v = self.pattern.reset(initial_phase=phase, return_qvel=True)
        env.data.qpos[env.model.jnt_qposadr[self.wing_joints]] = wing_q
        env.data.qvel[env.model.jnt_dofadr[self.wing_joints]] = wing_v
        env.data.ctrl[:] = self.retracted_controls
        filtered = env.model.actuator_actadr >= 0
        env.data.act[env.model.actuator_actadr[filtered]] = self.retracted_controls[filtered]
        mujoco.mj_forward(env.model, env.data)
        env.mean_sensors = env.data.sensordata.copy()

    def observation(self, step):
        env, model, data = self.env, self.env.model, self.env.data
        rotation = data.xmat[env.thorax_id].reshape(3, 3)
        future = self.reference[step : step + 6]
        observation = {
            "walker/" + k: env.mean_sensors[i].copy() for k, i in self.sensors.items()
        }
        observation.update(
            {
                "walker/actuator_activation": np.zeros(0, np.float32),
                "walker/joints_pos": data.qpos[model.jnt_qposadr[self.joints]].copy(),
                "walker/joints_vel": data.qvel[model.jnt_dofadr[self.joints]].copy(),
                "walker/world_zaxis": rotation[2].copy(),
                "walker/ref_displacement": (future[:, :3] - data.qpos[:3]) @ rotation,
                "walker/ref_root_quat": get_dquat_local(data.qpos[3:7], future[:, 3:]),
            }
        )
        for k, shape in self.policy.manifest["observation_shapes"].items():
            if observation[k].shape != tuple(shape):
                raise ValueError(f"Flight teacher schema mismatch: {k}")
        return observation

    def act(self, step):
        residual = self.policy.act(self.observation(step))
        frequency = self.pattern.base_beat_freq * (
            1 + self.pattern.rel_freq_range * np.clip(residual[self.user_index], -1, 1)
        )
        goal = self.pattern.step(frequency)
        residual[self.wing_indices] += (
            goal - self.env.data.qpos[self.env.model.jnt_qposadr[self.wing_joints]]
        )
        raw = self.retracted_controls.copy()
        raw[self.actuators] = residual[self.physical_action_indices]
        return np.clip(
            2 * (raw - self.env.low) / (self.env.high - self.env.low) - 1, -1, 1
        ).astype(np.float32)


def probe(args):
    if args.seconds <= 0 or not np.isfinite([args.seconds, args.speed]).all():
        raise ValueError("Positive duration and finite probe parameters required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    setup = time.perf_counter()
    provenance = evidence()
    env = FlyEnvironment("flight", wing_limits=getattr(args, "wing_limits", "original"))
    teacher = FlightTeacherOracle(env, args.teacher, args.wing_pattern)
    teacher.initialize(args.speed, args.seconds)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - setup
    started = time.perf_counter()
    rows = {
        k: [] for k in ("qpos", "qvel", "activation", "ctrl", "observation", "action", "time")
    }
    failure = None
    errors, heights, upright = [], [], []
    try:
        for i in range(round(args.seconds / env.control_dt)):
            action = teacher.act(i)
            for k, v in (
                ("qpos", env.data.qpos),
                ("qvel", env.data.qvel),
                ("activation", env.data.act),
                ("ctrl", env.data.ctrl),
                ("observation", env.observation()),
                ("action", action),
                ("time", env.data.time),
            ):
                rows[k].append(np.array(v, copy=True))
            env.step(action)
            errors.append(np.linalg.norm(env.data.qpos[:3] - teacher.reference[i + 1, :3]))
            heights.append(float(env.data.qpos[2] * 0.01))
            upright.append(float(env.data.xmat[env.thorax_id, 8]))
            assert not env.data.xfrc_applied.any() and not env.data.qfrc_applied.any()
    except RuntimeError as e:
        failure = str(e)
    np.savez_compressed(
        args.output / "flight.npz", **{k: np.asarray(v) for k, v in rows.items()}
    )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "setup_seconds": setup_seconds,
        "stepping_capture_and_state_write_wall_seconds": time.perf_counter() - started,
        "controller": "inherited flight policy plus upstream wingbeat generator; retracted leg servo targets",
        "student_present": False,
        "policy_acceptance_eligible": False,
        "wing_limit_profile": env.wing_limits,
        "maximum_wing_limit_violation_rad": env.maximum_wing_limit_violation.tolist(),
        "physics_hz": 1 / env.model.opt.timestep,
        "control_hz": 1 / env.control_dt,
        "speed_cm_s": args.speed,
        "requested_seconds": args.seconds,
        "simulated_seconds": env.data.time,
        "final_root_position_m": (env.data.qpos[:3] * 0.01).tolist(),
        "root_tracking_rmse_m": float(np.sqrt(np.mean(np.square(errors))) * 0.01)
        if errors
        else None,
        "maximum_root_tracking_error_m": float(max(errors) * 0.01) if errors else None,
        "minimum_root_height_m": min(heights) if heights else None,
        "minimum_upright": min(upright) if upright else None,
        "warning_count": int(env.data.warning.number.sum()),
        "numerical_failure": failure,
        "max_forbidden_ground_force_over_weight": env.maximum_disallowed_ground_force
        / (env.model.body_mass.sum() * 981),
        "teacher_sha256": sha256(args.teacher),
        "teacher_metadata_sha256": sha256(args.teacher.with_suffix(".json")),
        "wing_pattern_sha256": sha256(args.wing_pattern),
        "model_sha256": sha256(args.output / "model.mjb"),
        "state_sha256": sha256(args.output / "flight.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("teacher", "wing-pattern", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.2)
    parser.add_argument("--speed", type=float, default=0)
    parser.add_argument("--wing-limits", choices=("original", "firm"), default="original")
    probe(parser.parse_args())
