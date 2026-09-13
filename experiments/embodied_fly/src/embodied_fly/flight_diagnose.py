"""Compare measured wing motion and replayed passive forces before contact.

Force values are sampled at captured control boundaries, not integrated over
every physics substep. This is a failure diagnostic, not a new live simulation.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.provenance import evidence, sha256


def diagnose(args):
    if args.output.exists():
        raise FileExistsError("Preserve the previous diagnostic")
    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    wing = np.array([j for j in range(model.njnt) if "wing_" in (model.joint(j).name or "")])
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    angle_inputs = np.array([np.flatnonzero(hinges == j)[0] for j in wing])
    velocity_inputs = len(hinges) + angle_inputs
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mean = checkpoint["state_dict"]["observation_mean"].numpy()[velocity_inputs]
    std = checkpoint["state_dict"]["observation_std"].numpy()[velocity_inputs].clip(0.05)
    angle_mean = checkpoint["state_dict"]["observation_mean"].numpy()[angle_inputs]
    angle_std = checkpoint["state_dict"]["observation_std"].numpy()[angle_inputs].clip(0.05)
    results = {}
    initial = None
    for name, path in (("teacher", args.teacher_capture), ("student", args.student_capture)):
        with np.load(path) as capture:
            count = int(np.searchsorted(capture["time"], args.seconds - 1e-10))
            if count < 2 or count >= len(capture["qpos"]):
                raise ValueError("Diagnostic window must be inside both captures")
            if initial is None:
                initial = capture["qpos"][0].copy()
            else:
                np.testing.assert_array_equal(initial, capture["qpos"][0])
            forces = []
            for i in range(count):
                data.qpos[:] = capture["qpos"][i]
                data.qvel[:] = capture["qvel"][i]
                data.act[:] = capture["activation"][i]
                data.ctrl[:] = capture["ctrl"][i]
                mujoco.mj_forward(model, data)
                forces.append(data.qfrc_passive[2] / (981 * model.body_mass.sum()))
            velocity = capture["qvel"][:count, model.jnt_dofadr[wing]]
            encoded_velocity = (capture["observation"][:count, velocity_inputs] - mean) / std
            encoded_angle = (
                capture["observation"][:count, angle_inputs] - angle_mean
            ) / angle_std
            below = np.flatnonzero(capture["qpos"][:, 2] < 0.8)
            results[name] = {
                "state_sha256": sha256(path),
                "frames": count,
                "window_seconds": float(capture["time"][count] - capture["time"][0]),
                "sampled_mean_upward_passive_force_over_weight": float(np.mean(forces)),
                "wing_joint_peak_to_peak_rad": np.ptp(
                    capture["qpos"][:count, model.jnt_qposadr[wing]], axis=0
                ).tolist(),
                "wing_speed_rms_rad_s": np.sqrt(np.mean(velocity**2, axis=0)).tolist(),
                "wing_velocity_physical_sensor_clip_fraction": np.mean(
                    np.abs(velocity) >= 1000, axis=0
                ).tolist(),
                "wing_velocity_encoder_clip_fraction": np.mean(
                    np.abs(encoded_velocity) >= 10, axis=0
                ).tolist(),
                "wing_angle_encoder_clip_fraction": np.mean(
                    np.abs(encoded_angle) >= 10, axis=0
                ).tolist(),
                "root_height_after_window_m": float(capture["qpos"][count, 2] * 0.01),
                "first_below_8mm_seconds": float(capture["time"][below[0]])
                if len(below)
                else None,
            }
            if checkpoint.get("sensor_extension_size", 0) >= 6:
                extended_mean = checkpoint["state_dict"]["observation_mean"].numpy()[383:389]
                extended_std = (
                    checkpoint["state_dict"]["observation_std"].numpy()[383:389].clip(0.05)
                )
                extended = (velocity / 2000 - extended_mean) / extended_std
                results[name]["extended_wing_velocity_encoder_clip_fraction"] = np.mean(
                    np.abs(extended) >= 10, axis=0
                ).tolist()
            if checkpoint.get("sensor_extension_size") == 12:
                angle_extension_mean = checkpoint["state_dict"]["observation_mean"].numpy()[
                    389:395
                ]
                angle_extension_std = (
                    checkpoint["state_dict"]["observation_std"].numpy()[389:395].clip(0.05)
                )
                raw_angle = capture["qpos"][:count, model.jnt_qposadr[wing]] / np.pi
                results[name]["extended_wing_angle_encoder_clip_fraction"] = np.mean(
                    np.abs((raw_angle - angle_extension_mean) / angle_extension_std) >= 10,
                    axis=0,
                ).tolist()
    report = {
        "provenance": evidence(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.model),
        "method": "Native MuJoCo mj_forward replay at captured control boundaries; no new physics integration or substep-averaged force claim",
        "wing_joints": [model.joint(j).name for j in wing],
        "sensor_extension_size": checkpoint.get("sensor_extension_size", 0),
        "sensor_extension_weight_l2": float(
            checkpoint["state_dict"]["sensor_extension.weight"].norm()
        )
        if checkpoint.get("sensor_extension_size")
        else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model", "checkpoint", "teacher-capture", "student-capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.03)
    diagnose(parser.parse_args())
