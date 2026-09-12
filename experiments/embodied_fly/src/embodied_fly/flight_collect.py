"""Complete, physical flight demonstrations for the shared MaleCNS student.

The inherited expert/WPG is training-only. Capture the executed individual wing
commands, not a frequency label that would require a runtime oscillator.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import FlyEnvironment
from embodied_fly.flight_teacher import FlightTeacherOracle
from embodied_fly.provenance import evidence, sha256, utc_now


def collect(args):
    if args.seconds <= 0 or args.episodes < 4:
        raise ValueError("Positive duration and at least four episodes required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    env = FlyEnvironment("flight")
    teacher = FlightTeacherOracle(env, args.teacher, args.wing_pattern)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rng = np.random.default_rng(args.seed)
    collection_started = time.perf_counter()
    reports = []
    for episode in range(args.episodes):
        speed = (0, 5, 10, 20)[episode % 4]
        phase = float(rng.random())
        teacher.initialize(speed, args.seconds, phase)
        rows = {
            k: []
            for k in ("qpos", "qvel", "activation", "ctrl", "observation", "action", "time")
        }
        errors, heights, tilts = [], [], []
        failure = physical_failure = None
        episode_started = time.perf_counter()
        for step in range(round(args.seconds / env.control_dt)):
            action = teacher.act(step)
            for k, value in (
                ("qpos", env.data.qpos),
                ("qvel", env.data.qvel),
                ("activation", env.data.act),
                ("ctrl", env.data.ctrl),
                ("observation", env.observation()),
                ("action", action),
                ("time", env.data.time),
            ):
                rows[k].append(np.array(value, copy=True))
            try:
                env.step(action)
            except RuntimeError as error:
                failure = str(error)
                break
            errors.append(
                np.linalg.norm(env.data.qpos[:3] - teacher.reference[step + 1, :3]) * 0.01
            )
            heights.append(env.data.qpos[2] * 0.01)
            tilts.append(float(env.data.xmat[env.thorax_id, 8]))
            if heights[-1] < 0.008 or tilts[-1] < 0.5 or errors[-1] > 0.001:
                physical_failure = "outside_declared_flight_demonstration_envelope"
            if env.data.xfrc_applied.any() or env.data.qfrc_applied.any():
                raise RuntimeError("Flight demonstrations must not use applied root forces")
        path = args.output / f"episode_{episode:03d}.npz"
        np.savez_compressed(path, **rows, activity=np.ones(len(rows["action"]), np.int64))
        report = {
            "episode": episode,
            "speed_cm_s": speed,
            "initial_wing_phase": phase,
            "steps": len(rows["action"]),
            "simulated_seconds": env.data.time,
            "wall_seconds": time.perf_counter() - episode_started,
            "failure": failure,
            "physical_failure": physical_failure,
            "final_upright": float(env.data.xmat[env.thorax_id, 8]),
            "minimum_upright": min(tilts) if tilts else None,
            "minimum_height_m": min(heights) if heights else None,
            "root_tracking_rmse_m": float(np.sqrt(np.mean(np.square(errors))))
            if errors
            else None,
            "warning_count": int(env.data.warning.number.sum()),
            "max_forbidden_ground_force_over_weight": env.maximum_disallowed_ground_force
            / (env.model.body_mass.sum() * 981),
            "state_sha256": sha256(path),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    manifest = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "seed": args.seed,
        "source": "Inherited flight teacher plus supplied wing pattern; complete body and individual bounded actuator labels",
        "student_present": False,
        "policy_acceptance_eligible": False,
        "setup_seconds": setup_seconds,
        "collection_seconds": time.perf_counter() - collection_started,
        "teacher_sha256": sha256(args.teacher),
        "wing_pattern_sha256": sha256(args.wing_pattern),
        "model_sha256": sha256(args.output / "model.mjb"),
        "environment": env.report(),
        "control_hz": 1 / env.control_dt,
        "physics_backend": "native MuJoCo CPU",
        "worlds": 1,
        "utility_labels": "explore for this commanded flight warm start; not yet learned survival selection",
        "observations_are_causal": True,
        "future_reference_used_by_teacher_only": True,
        "student_observation_limit": "383 existing proprioceptive inputs; no altitude target or camera pixels yet; initial wing-control curriculum",
        "episodes": reports,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("teacher", "wing-pattern", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42001)
    collect(parser.parse_args())
