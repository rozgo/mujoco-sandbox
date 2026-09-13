"""Bounded, training-only recovery audit on the canonical position-actuated fly.

Every world runs the complete interval without resets. Initial height/vertical
velocity differ; the requested height remains 2 cm. No brain learning occurs.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.motion_flight import initialize
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256
from embodied_fly.state_hover import RECIPE, wing_commands
from embodied_fly.wing_position import reference_torque_to_position, wing_actuators


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = list(itertools.product(args.heights, args.vertical_speeds))
    env = FlyBatch(len(cases), threads=4, sensor_extension_size=14, preset="wing_position")
    base = initialize(env.template)
    state = {
        key: np.repeat(getattr(env.template.data, key)[None], len(cases), axis=0)
        for key in ("qpos", "qvel", "act", "ctrl")
    }
    state["qpos"][:, 2] = [case[0] for case in cases]
    state["qvel"][:, 2] = [case[1] for case in cases]
    env.reset(np.arange(env.n), state=state)
    env.requested_height_cm[:] = 2.0
    channels = wing_actuators(env.model)
    angles, speeds = env.template.wing_angle_indices, env.template.wing_velocity_indices
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    rows = {key: [] for key in ("qpos", "qvel", "action", "velocity", "upright")}
    first_failure = [None] * env.n
    started = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        q, v = env.fields["qpos"][:, angles], env.fields["qvel"][:, speeds]
        velocity = env.velocity()
        torque = wing_commands(
            q, v, velocity, env.fields["qpos"][:, 2], env.requested_height_cm,
            env.model.qpos_spring[angles],
        )
        action = np.repeat(base[None], env.n, axis=0)
        action[:, channels] = reference_torque_to_position(
            env.model, torque, q, v, env.control_dt
        )
        upright = env.fields["xmat"][:, env.template.thorax_id, 8].copy()
        for key in ("qpos", "qvel"):
            rows[key].append(env.fields[key].copy())
        rows["action"].append(action.copy())
        rows["velocity"].append(velocity)
        rows["upright"].append(upright)
        for i in range(env.n):
            if first_failure[i] is None and (env.fields["qpos"][i, 2] < 0.5 or upright[i] < 0.5):
                first_failure[i] = step * env.control_dt
        env.step(action)
    seconds = time.perf_counter() - started
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    results = []
    for i, (height, speed) in enumerate(cases):
        z = arrays["qpos"][:, i, 2]
        last = z[-round(1 / env.control_dt):]
        result = {
            "initial_height_cm": height, "initial_vertical_speed_cm_s": speed,
            "target_height_cm": 2.0, "first_failure_seconds": first_failure[i],
            "airborne_upright": first_failure[i] is None,
            "height_min_cm": float(z.min()), "height_max_cm": float(z.max()),
            "final_height_cm": float(env.fields["qpos"][i, 2]),
            "final_second_height_rmse_cm": float(np.sqrt(np.mean((last - 2) ** 2))),
            "final_second_vertical_speed_rms_cm_s": float(
                np.sqrt(np.mean(arrays["qvel"][-500:, i, 2] ** 2))
            ),
        }
        # Recovery-specific diagnostic gate, not the unchanged motor release gate.
        result["recovery_pass"] = bool(
            result["airborne_upright"] and result["final_second_height_rmse_cm"] < 0.3
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "provenance": evidence(), "physical_contract": physical_contract(env.model),
        "learning": False, "reference_recipe": RECIPE, "initialization_only": True,
        "live_resets": 0, "setup_seconds": setup, "capture_seconds": seconds,
        "seconds_per_world": args.seconds, "world_action_transitions": len(arrays["qpos"]) * env.n,
        "warnings": int(env.fields["warning"].sum()), "cases": results,
        "model_sha256": sha256(args.output / "model.mjb"),
        "trajectory_sha256": sha256(args.output / "trajectories.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--heights", type=float, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--vertical-speeds", type=float, nargs="+", default=[-20, 0, 20])
    args = parser.parse_args()
    if args.seconds < 1 or not np.isfinite([args.seconds, *args.heights, *args.vertical_speeds]).all():
        parser.error("Finite inputs and at least one second required")
    run(args)
