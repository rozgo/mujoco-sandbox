"""Matched PID/PPO hover capture, with three predeclared learned-policy starts."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import REFERENCE_HEIGHT_CM, HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.pid_hover import HoverPID, PIDConfig
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


@torch.no_grad()
def run(args):
    started, provenance = time.perf_counter(), evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    env = FlyBatch(
        4,
        4,
        actor.sensor_extension_size,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
    )
    if parent["physical_contract"] != physical_contract(env.model):
        raise ValueError("Comparison must match the recorded accepted physics")
    tasks = HoverOnlyTasks(env, 120091)
    target = tasks.start.copy()
    state = {k: env.fields[k][1:3].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    state["qpos"][0, 2] -= 0.02
    state["qvel"][0, 2] = -1
    state["qpos"][1, 0] += 0.02
    state["qvel"][1, 0] = 0.5
    env.reset([1, 2], state=state)
    env.requested_height_cm[:] = REFERENCE_HEIGHT_CM
    env.requested_xy_cm[:] = target[:, :2]
    reference = HoverPID(env.template, tasks.air_action, target[3], PIDConfig())
    memory = actor.initial_state(3)
    args.output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    synchronize(device)
    setup = time.perf_counter() - started
    names = ["hover", "lower_start", "lateral_start", "pid"]
    rows, measured = [], []
    failure, first_failure = None, [None] * 4
    started = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        obs = env.observation()
        output = actor(torch.as_tensor(obs[:3], device=device), memory)
        memory = output.state
        # Controller scratch state follows world 3. This does not write a live
        # world's physical state; both PID and actor execute bounded actions.
        for key in ("qpos", "qvel", "act", "ctrl"):
            getattr(env.template.data, key)[:] = env.fields[key][3]
        env.template.data.time = step * env.control_dt
        mujoco.mj_forward(env.template.model, env.template.data)
        action = np.vstack([output.action.cpu().numpy(), reference.act()])
        rows.append(
            {
                "qpos": env.fields["qpos"].copy(),
                "qvel": env.fields["qvel"].copy(),
                "activation": env.fields["act"].copy(),
                "ctrl": env.fields["ctrl"].copy(),
                "action": action.copy(),
                "time": np.full(4, step * env.control_dt),
                "requested_height_cm": env.requested_height_cm.copy(),
                "requested_xy_cm": env.requested_xy_cm.copy(),
                "observation": obs.copy(),
            }
        )
        try:
            env.step(action)
        except RuntimeError as error:
            failure = str(error)
            break
        position = env.fields["qpos"][:, :3].copy()
        upright = env.fields["xmat"][:, env.template.thorax_id, 8].copy()
        for i in range(4):
            if first_failure[i] is None and (
                position[i, 2] < 0.5
                or upright[i] < 0.5
                or env.forbidden_peak[i] / env.body_weight > 0.1
            ):
                first_failure[i] = (step + 1) * env.control_dt
        measured.append(
            {
                "position": position,
                "velocity": env.fields["qvel"][:, :3].copy(),
                "upright": upright,
                "forbidden": env.forbidden_peak.copy() / env.body_weight,
            }
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    results = []
    for i, name in enumerate(names):
        arrays = {k: np.stack([r[k][i] for r in rows]) for k in rows[0]}
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
        actual = {k: np.stack([r[k][i] for r in measured]) for k in measured[0]}
        windows = {}
        for label, mask in [
            ("whole_capture", np.ones(len(measured), bool)),
            ("after_first_second", np.arange(len(measured)) * env.control_dt >= 1),
        ]:
            if not mask.any():
                continue
            pos, vel = actual["position"][mask], actual["velocity"][mask]
            windows[label] = {
                "height_span_mm": float(np.ptp(pos[:, 2]) * 10),
                "position_peak_mm": float(np.linalg.norm(pos - target[i], axis=1).max() * 10),
                "position_rms_mm": float(
                    np.sqrt(np.mean(np.sum((pos - target[i]) ** 2, axis=1))) * 10
                ),
                "altitude_rms_mm": float(
                    np.sqrt(np.mean((pos[:, 2] - target[i, 2]) ** 2)) * 10
                ),
                "vertical_speed_rms_mm_s": float(np.sqrt(np.mean(vel[:, 2] ** 2)) * 10),
            }
        settled = windows.get("after_first_second", {})
        quality = bool(
            settled
            and settled["height_span_mm"] < 0.2
            and settled["position_peak_mm"] < 0.5
            and settled["vertical_speed_rms_mm_s"] < 15
            and actual["upright"].min() > 0.99
        )
        results.append(
            {
                "case": name,
                "controller": "PID reference" if i == 3 else "learned MaleCNS actor",
                "windows": windows,
                "stable": first_failure[i] is None and failure is None,
                "first_failure_seconds": first_failure[i],
                "reference_quality_gate": quality
                and first_failure[i] is None
                and failure is None,
                "minimum_upright": float(actual["upright"].min()),
                "maximum_forbidden_load_bodyweights": float(actual["forbidden"].max()),
            }
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "results": results,
        "physical_contract": physical_contract(env.model),
        "force_model": env.wing_forces.report(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.output / "model.mjb"),
        "setup_seconds": setup,
        "capture_seconds": elapsed,
        "control_hz": 500,
        "physics_hz": 1000,
        "parallel_physics_worlds": 4,
        "physical_transitions": len(measured) * 4,
        "seconds_per_world": len(measured) * env.control_dt,
        "numerical_failure": failure,
        "pid_config": vars(PIDConfig()),
        "matched_nominal_initial_state": True,
        "actor_has_pid_or_oscillator": False,
        "actor_evaluation": "deterministic mean, no exploration, no resets",
        "policy_case_offsets": {
            "hover": "none",
            "lower_start": "z -0.02 cm, vz -1 cm/s",
            "lateral_start": "x +0.02 cm, vx +0.5 cm/s",
        },
        "reference_controller_scope": "world 3 only; read-only scratch state, outputs wing actuator commands",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture_seconds": elapsed, "results": results}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seconds", type=float, default=10)
    run(p.parse_args())
