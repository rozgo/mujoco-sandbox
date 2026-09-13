"""Audit altitude feedback through frozen MaleCNS on recorded motor histories.

Counterfactual sensor readings are neural probes, not physical experience. The
training reference is queried for comparison only and never drives a body here.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.observations import wing_angle_indices, wing_velocity_indices
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.state_hover import wing_commands
from embodied_fly.wing_position import reference_torque_to_position, wing_actuators


def examples(model, observation, qpos, qvel, activation, ctrl):
    """Nominal, height +/-0.2 cm, world vertical speed +/-2 cm/s."""
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = qpos, qvel, activation, ctrl
    mujoco.mj_forward(model, data)
    thorax = model.body("walker/thorax").id
    anatomical = data.xmat[thorax].reshape(3, 3)
    inertial = data.ximat[thorax].reshape(3, 3)
    body_v = np.r_[observation[282:285] * 20, observation[285:288] * 10]
    rotate = anatomical.T @ inertial
    body_v = np.r_[rotate @ body_v[:3], rotate @ body_v[3:]]
    obs = np.repeat(observation[None], 5, axis=0)
    h = np.full(5, qpos[2])
    v = np.repeat(body_v[None], 5, axis=0)
    for row, sign in ((1, 1), (2, -1)):
        h[row] += sign * 0.2
        obs[row, 395] += sign * 0.1
    for row, sign in ((3, 1), (4, -1)):
        delta = np.array([0, 0, sign * 2.0])
        obs[row, 285:288] += inertial.T @ delta / 10
        v[row, 3:6] += anatomical.T @ delta
    angles = np.repeat(qpos[wing_angle_indices(model)][None], 5, axis=0)
    speeds = np.repeat(qvel[wing_velocity_indices(model)][None], 5, axis=0)
    torque = wing_commands(
        angles,
        speeds,
        v,
        h,
        obs[:, 396] * 2,
        model.qpos_spring[wing_angle_indices(model)],
    )
    return obs, reference_torque_to_position(model, torque, angles, speeds, 0.002)


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    report = json.loads((args.capture / "report.json").read_text())
    if report["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Checkpoint/capture mismatch")
    model = mujoco.MjModel.from_binary_path(str(args.capture / "model.mjb"))
    if report["model_sha256"] != sha256(args.capture / "model.mjb"):
        raise ValueError("Captured model hash mismatch")
    if physical_contract(model) != checkpoint["physical_contract"] or not actor.motor_only:
        raise ValueError("Canonical motor body required")
    captures = []
    names = [r["case"] for r in report["results"]]
    for row in report["results"]:
        path = args.capture / (row["case"] + ".npz")
        if sha256(path) != row["state_sha256"]:
            raise ValueError("Capture hash mismatch")
        with np.load(path) as z:
            captures.append(
                {
                    k: z[k].copy()
                    for k in ("observation", "qpos", "qvel", "activation", "ctrl", "action")
                }
            )
    hover = names.index("hover")
    if min(args.frames) < 0 or max(args.frames) >= min(len(c["qpos"]) for c in captures):
        raise ValueError("Probe frame outside capture")
    memory = actor.initial_state(len(captures))
    wings = wing_actuators(model)
    setup = time.perf_counter() - start
    start = time.perf_counter()
    rows, replay_error = [], 0.0
    for frame in range(max(args.frames) + 1):
        obs = np.stack([c["observation"][frame] for c in captures])
        if frame in args.frames:
            c = captures[hover]
            inputs, expected = examples(
                model,
                *[c[k][frame] for k in ("observation", "qpos", "qvel", "activation", "ctrl")],
            )
            state = memory[:, hover : hover + 1].repeat_interleave(5, dim=1)
            for held in range(25):
                output = actor(torch.as_tensor(inputs, device=device), state)
                state = output.state
                if held in (0, 4, 24):
                    action = output.action[:, wings].cpu().numpy()
                    rows.append(
                        {
                            "frame": frame,
                            "time_seconds": frame / 500,
                            "held_input_updates": held + 1,
                            "actual_height_cm": float(c["qpos"][frame, 2]),
                            "requested_height_cm": float(obs[hover, 396] * 2),
                            "actor_actions": action.tolist(),
                            "reference_actions": expected.tolist(),
                            "actor_height_response_per_cm": (
                                (action[1] - action[2]) / 0.4
                            ).tolist(),
                            "reference_height_response_per_cm": (
                                (expected[1] - expected[2]) / 0.4
                            ).tolist(),
                            "actor_vertical_speed_response_per_cm_s": (
                                (action[3] - action[4]) / 4
                            ).tolist(),
                            "reference_vertical_speed_response_per_cm_s": (
                                (expected[3] - expected[4]) / 4
                            ).tolist(),
                        }
                    )
        output = actor(torch.as_tensor(obs, device=device), memory)
        memory = output.state
        replay_error = max(
            replay_error,
            float(
                np.abs(
                    output.action.cpu().numpy()
                    - np.stack([c["action"][frame] for c in captures])
                ).max()
            ),
        )
    if replay_error > 3e-4:
        raise RuntimeError(f"History replay mismatch: {replay_error}")
    result = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "capture_report_sha256": sha256(args.capture / "report.json"),
        "physical_contract": checkpoint["physical_contract"],
        "setup_seconds": setup,
        "diagnostic_seconds": time.perf_counter() - start,
        "physical_transitions": 0,
        "training_updates": 0,
        "replay_max_action_error": replay_error,
        "snapshots": rows,
        "scope": "Frozen actor, copied true preceding memory, counterfactual sensors. Held-input responses are not physical rollouts or complete closed-loop stability tests.",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("provenance", "snapshots")})
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "graph", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[0, 100, 250, 500, 1000, 2000]
    )
    run(parser.parse_args())
