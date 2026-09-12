"""Collect physical, full-body teacher demonstrations before brain imitation."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT, FlyEnvironment
from embodied_fly.provenance import evidence, utc_now
from embodied_fly.teacher import TeacherOracle


def collect(teacher_path, output, seconds=2.0, episodes=8, seed=2001):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    run_evidence = evidence()
    torch.set_num_threads(2)
    environment = FlyEnvironment()
    oracle = TeacherOracle(environment, teacher_path)
    setup = time.perf_counter() - started
    rng = np.random.default_rng(seed)
    mujoco.mj_saveModel(environment.model, str(output / "model.mjb"))
    reports = []
    for episode in range(episodes):
        speed = (0.0, 0.5, 1.0, 2.0)[episode % 4]
        # First motor curriculum: hold, straight walking, and broad walking turns.
        # Diagnostic 01 showed the inherited teacher topples during some in-place
        # and slow tight turns. Those failed demonstrations remain archived.
        yaw = (0.0, 0.0, -0.75, 0.75)[(episode // 4) % 4] if speed >= 1 else 0.0
        heading = float(rng.uniform(-0.2, 0.2))
        environment.reset(yaw=heading)
        environment.command[:] = (speed, 0, yaw)
        oracle.set_reference(speed, yaw, seconds, heading=heading)
        observations, actions, qpos, qvel, activations, controls = [], [], [], [], [], []
        initial_pos = environment.data.qpos[:3].copy()
        episode_start = time.perf_counter()
        failure = None
        physical_failure = None
        for step in range(int(seconds / CONTROL_DT)):
            observations.append(environment.observation())
            action = oracle.act(step)
            actions.append(action)
            qpos.append(environment.data.qpos.copy())
            qvel.append(environment.data.qvel.copy())
            activations.append(environment.data.act.copy())
            controls.append(environment.data.ctrl.copy())
            try:
                environment.step(action)
            except RuntimeError as error:
                failure = str(error)
                break
            if environment.data.xmat[environment.thorax_id, 8] < 0.5:
                physical_failure = "body_tilt_over_60_degrees"
        distance = environment.data.qpos[:2] - initial_pos[:2]
        report = {
            "episode": episode,
            "speed_cm_s": speed,
            "yaw_rad_s": yaw,
            "steps": len(actions),
            "simulated_seconds": len(actions) * CONTROL_DT,
            "wall_seconds": time.perf_counter() - episode_start,
            "displacement_m": (distance * 0.01).tolist(),
            "final_height_m": float(environment.data.qpos[2] * 0.01),
            "final_upright": float(environment.data.xmat[environment.thorax_id, 8]),
            "failure": failure,
            "physical_failure": physical_failure,
            "warning_count": int(environment.data.warning.number.sum()),
        }
        np.savez_compressed(
            output / f"episode_{episode:03d}.npz",
            observation=observations,
            action=actions,
            qpos=qpos,
            qvel=qvel,
            activation=activations,
            ctrl=controls,
            activity=np.full(len(actions), 0 if speed == 0 else 1, np.int64),
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    manifest = {
        "provenance": run_evidence,
        "completed_utc": utc_now(),
        "source": "upstream learned teacher through full 78-actuator FlyBody",
        "seed": seed,
        "setup_seconds": setup,
        "collection_seconds": time.perf_counter() - started - setup,
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "teacher_golden_max_error": oracle.policy.golden_max_error,
        "physics_backend": "native MuJoCo CPU",
        "worlds": 1,
        "environment": environment.report(),
        "episodes": reports,
        "observations_are_causal": True,
        "future_reference_used_by_teacher_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    collect(args.teacher, args.output, args.seconds, args.episodes)
