"""Teacher-free, direct-wing student evaluation from a declared airborne reset.

An expert capture supplies only the initial physical pose/velocity/filter state.
No teacher, oscillator, future reference or root forces drive the live rollout.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT, FlyEnvironment
from embodied_fly.evaluate import load_actor
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.observations import actor_observation
from embodied_fly.provenance import evidence, sha256, utc_now


def initialize_from_capture(env, path, speed):
    env.reset()
    with np.load(path, allow_pickle=False) as capture:
        for name, array in (
            ("qpos", env.data.qpos),
            ("qvel", env.data.qvel),
            ("activation", env.data.act),
            ("ctrl", env.data.ctrl),
        ):
            state = capture[name][0]
            if state.shape != array.shape or not np.isfinite(state).all():
                raise ValueError("Initial capture and flight model differ")
            array[:] = state
    env.command[:] = (speed, 0, 0)
    env.requested_height_cm = float(env.data.qpos[2])
    mujoco.mj_forward(env.model, env.data)
    env.mean_sensors = env.data.sensordata.copy()


@torch.no_grad()
def evaluate(args):
    if not np.isfinite([args.seconds, args.speed]).all() or args.seconds <= 0:
        raise ValueError("Finite positive duration and finite command required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    provenance = evidence()
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    env = FlyEnvironment("flight", wing_limits=getattr(args, "wing_limits", "original"))
    initialize_from_capture(env, args.initial, args.speed)
    start_pos = env.data.qpos[:3].copy()
    memory = actor.initial_state(1)
    projection = NeuralProjection.from_graph(args.graph, device) if args.neural_view else None
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rows = {
        k: []
        for k in (
            "qpos",
            "qvel",
            "activation",
            "ctrl",
            "observation",
            "action",
            "time",
            "utility",
        )
    }
    maps = []
    map_stride = round(0.02 / env.control_dt)
    errors, heights, upright = [], [], []
    failure = None
    stepping_started = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        observation = actor_observation(env, actor)
        result = actor(
            torch.as_tensor(observation[None], device=device),
            memory,
            time_scale=env.control_dt / CONTROL_DT,
        )
        memory = result.state
        action = result.action[0].cpu().numpy()
        for k, value in (
            ("qpos", env.data.qpos),
            ("qvel", env.data.qvel),
            ("activation", env.data.act),
            ("ctrl", env.data.ctrl),
            ("observation", observation),
            ("action", action),
            ("time", env.data.time),
            ("utility", result.utility_scores[0].cpu().numpy()),
        ):
            rows[k].append(np.array(value, copy=True))
        if projection is not None and step % map_stride == 0:
            maps.append(projection.project(memory)[0].astype(np.float16))
        try:
            env.step(action)
        except RuntimeError as error:
            failure = str(error)
            break
        target = start_pos + (args.speed * env.data.time, 0, 0)
        errors.append(np.linalg.norm(env.data.qpos[:3] - target) * 0.01)
        heights.append(float(env.data.qpos[2] * 0.01))
        upright.append(float(env.data.xmat[env.thorax_id, 8]))
        if env.data.xfrc_applied.any() or env.data.qfrc_applied.any():
            raise RuntimeError("Student flight must not use applied root forces")
    if projection is not None:
        rows.update(
            neural_map=maps,
            neural_occupancy=projection.occupancy,
            neural_map_stride=map_stride,
        )
    np.savez_compressed(args.output / "flight.npz", **rows)
    ratio = env.maximum_disallowed_ground_force / (env.model.body_mass.sum() * 981)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    stable = bool(heights) and min(heights) > 0.008 and min(upright) > 0.5
    success = stable and rmse < 0.001 and ratio < 0.1 and failure is None
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "setup_seconds": setup_seconds,
        "stepping_capture_and_state_write_wall_seconds": time.perf_counter()
        - stepping_started,
        "checkpoint_sha256": sha256(args.checkpoint),
        "training_source_commit": checkpoint["source_commit"],
        "graph_sha256": checkpoint["graph_sha256"],
        "initial_capture_sha256": sha256(args.initial),
        "initialization": "First physical frame of declared expert capture; memory/previous actions reset; no subsequent teacher access",
        "student_present": True,
        "observation_size": actor.observation_size,
        "sensor_extension_size": actor.sensor_extension_size,
        "teacher_present": False,
        "scripted_gait_present": False,
        "policy_acceptance_eligible": True,
        "scope": "Airborne direct-wing development probe; not takeoff, landing or survival",
        "physics": "native MuJoCo CPU",
        "physical_preset": "flight",
        "wing_limit_profile": env.wing_limits,
        "maximum_wing_limit_violation_rad": env.maximum_wing_limit_violation.tolist(),
        "device": str(device),
        "physics_hz": 1 / env.model.opt.timestep,
        "control_hz": 1 / env.control_dt,
        "neural_time_scale": env.control_dt / CONTROL_DT,
        "active_actuators": env.model.nu,
        "speed_cm_s": args.speed,
        "requested_seconds": args.seconds,
        "simulated_seconds": env.data.time,
        "root_tracking_rmse_m": rmse,
        "minimum_root_height_m": min(heights) if heights else None,
        "final_root_position_m": (env.data.qpos[:3] * 0.01).tolist(),
        "max_forbidden_ground_force_over_weight": ratio,
        "warning_count": int(env.data.warning.number.sum()),
        "numerical_failure": failure,
        "success": success,
        "gates": {
            "minimum_height_m": 0.008,
            "minimum_upright": 0.5,
            "maximum_root_rmse_m": 0.001,
            "maximum_prohibited_support_over_weight": 0.1,
        },
        "results": [
            {
                "case": "flight",
                "stable": stable,
                "success": success,
                "gate_label": "Airborne tracking gate",
            }
        ],
        "neural_view": projection.report() if projection is not None else None,
        "model_sha256": sha256(args.output / "model.mjb"),
        "state_sha256": sha256(args.output / "flight.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("graph", "checkpoint", "initial", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=0)
    parser.add_argument("--neural-view", action="store_true")
    parser.add_argument("--wing-limits", choices=("original", "firm"), default="original")
    raise SystemExit(0 if evaluate(parser.parse_args())["success"] else 2)
