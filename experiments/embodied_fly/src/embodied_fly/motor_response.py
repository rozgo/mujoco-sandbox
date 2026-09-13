"""Frozen neural response to small wing-feedback changes; not a physics rollout."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.observations import WING_JOINTS, WING_SPEED_SCALE_RAD_S
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now


def perturb_feedback(model, observation, qpos, qvel, angle_delta=0.05, speed_delta=2.0):
    """25 variants: unchanged, +/- each of six angles, +/- each of six speeds.

    Recompute both measured input paths from the perturbed coordinates. Other
    inputs, including actual previous commands, remain unchanged. This is a
    sensory intervention, not a claim the held observations describe a rollout.
    """
    obs = np.asarray(observation)
    if obs.shape[-1] != 397:
        raise ValueError("Expected canonical 397-input motor observation")
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joints = np.array([model.joint(n).id for n in WING_JOINTS])
    columns = np.array([np.flatnonzero(hinges == j)[0] for j in joints])
    result = np.repeat(obs[:, None], 25, axis=1)
    for i, (joint, column) in enumerate(zip(joints, columns)):
        lo, hi = model.jnt_range[joint]
        for sign, offset in ((1, 0), (-1, 1)):
            a = 1 + i * 2 + offset
            angle = qpos[:, model.jnt_qposadr[joint]] + sign * angle_delta
            result[:, a, column] = np.clip(2 * (angle - lo) / max(hi - lo, 1e-4) - 1, -5, 5)
            result[:, a, 389 + i] = angle / np.pi
            a = 13 + i * 2 + offset
            speed = qvel[:, model.jnt_dofadr[joint]] + sign * speed_delta
            result[:, a, len(hinges) + column] = np.clip(speed / 100, -10, 10)
            result[:, a, 383 + i] = speed / WING_SPEED_SCALE_RAD_S
    return result


@torch.no_grad()
def diagnose(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    source = json.loads((args.capture / "report.json").read_text())
    if source["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Capture belongs to a different checkpoint")
    model = mujoco.MjModel.from_binary_path(str(args.capture / "model.mjb"))
    if source["model_sha256"] != sha256(args.capture / "model.mjb") or checkpoint[
        "physical_contract"
    ] != physical_contract(model):
        raise ValueError("Capture/checkpoint physical identity differs")
    if not actor.motor_only or actor.observation_size != 397:
        raise ValueError("Requires the canonical motor-only actor")
    cases = [row["case"] for row in source["results"]]
    captures = []
    for row in source["results"]:
        path = args.capture / (row["case"] + ".npz")
        if sha256(path) != row["state_sha256"]:
            raise ValueError("Capture hash mismatch")
        with np.load(path) as d:
            captures.append(
                {k: d[k].copy() for k in ("observation", "qpos", "qvel", "action")}
            )
    wings = np.array([model.actuator(n).id for n in WING_JOINTS])
    memory = actor.initial_state(len(cases))
    setup = time.perf_counter() - start
    start = time.perf_counter()
    snapshots = []
    replay_error = 0.0
    for step in range(max(args.frames) + 1):
        obs = np.stack([d["observation"][step] for d in captures])
        if step in args.frames:
            variants = perturb_feedback(
                model,
                obs,
                np.stack([d["qpos"][step] for d in captures]),
                np.stack([d["qvel"][step] for d in captures]),
            )
            inputs = torch.as_tensor(variants.reshape(-1, 397), device=device)
            state = memory.repeat_interleave(25, dim=1)
            normalized = (inputs - actor.observation_mean) / actor.observation_std.clamp_min(
                0.05
            )
            for held_step in range(25):
                output = actor(inputs, state)
                state = output.state
                if held_step not in (0, 4, 24):
                    continue
                action = output.action.cpu().numpy().reshape(len(cases), 25, 78)[:, :, wings]
                angle_response = ((action[:, 1:13:2] - action[:, 2:13:2]) / 0.1).transpose(
                    0, 2, 1
                )
                speed_response = ((action[:, 13:25:2] - action[:, 14:25:2]) / 4).transpose(
                    0, 2, 1
                )
                for i, case in enumerate(cases):
                    snapshots.append(
                        {
                            "case": case,
                            "captured_time_seconds": step / 500,
                            "held_input_updates": held_step + 1,
                            "baseline_action": action[i, 0].tolist(),
                            "angle_response_action_per_rad": angle_response[i].tolist(),
                            "velocity_response_action_per_rad_s": speed_response[i].tolist(),
                            "restoring_angle_diagonals": np.diag(angle_response[i]).tolist(),
                            "damping_velocity_diagonals": np.diag(speed_response[i]).tolist(),
                        }
                    )
        output = actor(torch.as_tensor(obs, device=device), memory)
        memory = output.state
        replay_error = max(
            replay_error,
            float(
                np.max(
                    np.abs(
                        output.action.cpu().numpy()
                        - np.stack([d["action"][step] for d in captures])
                    )
                )
            ),
        )
    if replay_error > 3e-4:
        raise RuntimeError(f"Recorded-history replay differs: {replay_error}")
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "capture_report_sha256": sha256(args.capture / "report.json"),
        "physical_contract": checkpoint["physical_contract"],
        "setup_seconds": setup,
        "diagnostic_seconds": time.perf_counter() - start,
        "physical_transitions": 0,
        "training_updates": 0,
        "parallel_neural_sequences": len(cases) * 25,
        "replay_max_action_error": replay_error,
        "wing_channels": wings.tolist(),
        "expected_ground_angle_diagonal": -0.02 / 0.03,
        "expected_ground_velocity_diagonal": -0.00015 / 0.03,
        "observation_mean": actor.observation_mean.cpu().tolist(),
        "observation_std": actor.observation_std.cpu().tolist(),
        "extension_weight_column_norm": actor.sensor_extension.weight.norm(dim=0)
        .cpu()
        .tolist(),
        "last_probe_encoder_clipped_fraction": float((normalized.abs() >= 10).float().mean()),
        "snapshots": snapshots,
        "limitations": "Counterfactual held sensory inputs with copied captured-history neural state. No physical rollout, feedback correction, training or policy acceptance.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("provenance", "observation_mean", "observation_std", "snapshots")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "graph", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 250, 500])
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())
