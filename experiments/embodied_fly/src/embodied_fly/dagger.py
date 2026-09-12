"""Training-only teacher feedback on states visited by a student/teacher mixture.

This is dataset aggregation for imitation, not policy-gradient RL. All action
mixtures drive bounded actuators through the same physical environment. The
student receives causal feedback; the teacher alone has a future reference.
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
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.teacher import TeacherOracle


@torch.no_grad()
def collect(args):
    if not 0 <= args.student_fraction <= 1:
        raise ValueError("Student fraction must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_evidence = evidence()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    actor, _ = load_actor(args.checkpoint, args.graph, device)
    environment = FlyEnvironment()
    oracle = TeacherOracle(environment, args.teacher)
    mujoco.mj_saveModel(environment.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rng = np.random.default_rng(args.seed)
    reports = []
    collection_start = time.perf_counter()
    for episode in range(args.episodes):
        speed = (0.0, 0.5, 1.0, 2.0)[episode % 4]
        turn = (0.0, 0.0, -0.75, 0.75)[(episode // 4) % 4] if speed >= 1 else 0.0
        heading = float(rng.uniform(-0.2, 0.2))
        environment.reset(yaw=heading)
        environment.command[:] = (speed, 0, turn)
        oracle.set_reference(speed, turn, args.seconds, heading=heading)
        memory = actor.initial_state(1)
        arrays = {
            key: []
            for key in (
                "observation",
                "action",
                "executed_action",
                "student_action",
                "activity",
                "qpos",
                "qvel",
                "activation",
                "ctrl",
            )
        }
        episode_start = time.perf_counter()
        initial_position = environment.data.qpos[:3].copy()
        failure = physical_failure = None
        upright = []
        for step in range(round(args.seconds / CONTROL_DT)):
            observation = environment.observation()
            result = actor(torch.as_tensor(observation[None], device=device), memory)
            memory = result.state
            student_action = environment.walking_action(result.action[0].cpu().numpy())
            teacher_action = oracle.act(step, reference_mode=args.reference_mode)
            action = environment.walking_action(
                args.student_fraction * student_action
                + (1 - args.student_fraction) * teacher_action
            )
            arrays["observation"].append(observation)
            arrays["action"].append(teacher_action)
            arrays["executed_action"].append(action)
            arrays["student_action"].append(student_action)
            arrays["activity"].append(0 if speed == 0 else 1)
            for key, value in (
                ("qpos", environment.data.qpos),
                ("qvel", environment.data.qvel),
                ("activation", environment.data.act),
                ("ctrl", environment.data.ctrl),
            ):
                arrays[key].append(value.copy())
            try:
                environment.step(action)
            except RuntimeError as error:
                failure = str(error)
                break
            upright.append(float(environment.data.xmat[environment.thorax_id, 8]))
            if upright[-1] < 0.5:
                physical_failure = "body_tilt_over_60_degrees"
                break
        np.savez_compressed(args.output / f"episode_{episode:03d}.npz", **arrays)
        report = {
            "episode": episode,
            "speed_cm_s": speed,
            "yaw_rad_s": turn,
            "steps": len(arrays["action"]),
            "simulated_seconds": len(arrays["action"]) * CONTROL_DT,
            "wall_seconds": time.perf_counter() - episode_start,
            "displacement_m": (
                (environment.data.qpos[:2] - initial_position[:2]) * 0.01
            ).tolist(),
            "final_upright": float(environment.data.xmat[environment.thorax_id, 8]),
            "minimum_upright": min(upright) if upright else None,
            "max_disallowed_ground_force_over_weight": environment.maximum_disallowed_ground_force
            / (environment.model.body_mass.sum() * 981),
            "failure": failure,
            "physical_failure": physical_failure,
            "warning_count": int(environment.data.warning.number.sum()),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    manifest = {
        "provenance": run_evidence,
        "completed_utc": utc_now(),
        "source": "teacher labels on student/teacher-mixture physical states",
        "method": "DAgger-style dataset aggregation; teacher queried only during training",
        "seed": args.seed,
        "student_fraction": args.student_fraction,
        "walking_action_mask": True,
        "active_actuators": int((~environment.walking_inactive).sum()),
        "setup_seconds": setup_seconds,
        "collection_seconds": time.perf_counter() - collection_start,
        "checkpoint_sha256": sha256(args.checkpoint),
        "teacher_sha256": sha256(args.teacher),
        "physics_backend": "native MuJoCo CPU",
        "brain_device": str(device),
        "worlds": 1,
        "environment": environment.report(),
        "episodes": reports,
        "action_field": "teacher supervised target; executed_action is the physical mixture",
        "observations_are_causal": True,
        "future_reference_used_by_teacher_only": True,
        "teacher_reference_mode": args.reference_mode,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in ("environment", "episodes", "provenance")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=22001)
    parser.add_argument("--student-fraction", type=float, default=0.25)
    parser.add_argument(
        "--reference-mode", choices=("world_path", "receding"), default="world_path"
    )
    collect(parser.parse_args())
