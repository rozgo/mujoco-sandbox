"""Collect the retained student's own causal closed-loop behavior for rehearsal.

The actor remains frozen during collection. Keep failed episodes with explicit
labels; training can reject them without erasing the record. No teacher is used.
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


@torch.no_grad()
def collect(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, _ = load_actor(args.checkpoint, args.graph, device)
    if actor.observation_size != 383:
        raise ValueError("This retention corpus must use the declared legacy reference actor")
    env = FlyEnvironment()
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    fixed = [
        ("hold", 0, 0),
        ("slow_walk", 0.5, 0),
        ("walk", 1, 0),
        ("fast_walk", 2, 0),
        ("left", 1.5, 0.75),
        ("right", 1.5, -0.75),
    ]
    schedules = [(name, [(2, (v, 0, yaw))]) for name, v, yaw in fixed]
    for name in ("transition_a", "transition_b"):
        schedules.append((name, [(2, (1, 0, 0)), (2, (0, 0, 0)), (2, (1, 0, 0))]))
    rng = np.random.default_rng(args.seed)
    setup_seconds = time.perf_counter() - started
    collection_start = time.perf_counter()
    reports = []
    for episode, (name, phases) in enumerate(schedules):
        heading = float(rng.uniform(-0.2, 0.2))
        env.reset(yaw=heading)
        memory = actor.initial_state(1)
        rows = {
            k: []
            for k in (
                "observation",
                "action",
                "activity",
                "qpos",
                "qvel",
                "activation",
                "ctrl",
                "time",
                "command",
            )
        }
        failure = physical_failure = None
        minimum_upright = 1.0
        episode_start = time.perf_counter()
        for duration, command in phases:
            env.command[:] = command
            for _ in range(round(duration / CONTROL_DT)):
                observation = env.observation()
                result = actor(torch.as_tensor(observation[None], device=device), memory)
                memory = result.state
                action = env.walking_action(result.action[0].cpu().numpy())
                for key, value in (
                    ("observation", observation),
                    ("action", action),
                    ("activity", int(result.activity[0])),
                    ("qpos", env.data.qpos),
                    ("qvel", env.data.qvel),
                    ("activation", env.data.act),
                    ("ctrl", env.data.ctrl),
                    ("time", env.data.time),
                    ("command", env.command),
                ):
                    rows[key].append(np.array(value, copy=True))
                try:
                    env.step(action)
                except RuntimeError as error:
                    failure = str(error)
                    break
                minimum_upright = min(minimum_upright, float(env.data.xmat[env.thorax_id, 8]))
                if (
                    minimum_upright < 0.5
                    or env.maximum_disallowed_ground_force
                    > 0.1 * env.model.body_mass.sum() * 981
                ):
                    physical_failure = "unstable_or_prohibited_support"
            if failure:
                break
        path = args.output / f"episode_{episode:03d}.npz"
        np.savez_compressed(path, **rows)
        report = {
            "episode": episode,
            "case": name,
            "initial_heading_rad": heading,
            "phases": [
                {"seconds": duration, "command_cm_s_rad_s": command}
                for duration, command in phases
            ],
            "failure": failure,
            "physical_failure": physical_failure,
            "final_upright": float(env.data.xmat[env.thorax_id, 8]),
            "minimum_upright": minimum_upright,
            "max_disallowed_ground_force_over_weight": env.maximum_disallowed_ground_force
            / (env.model.body_mass.sum() * 981),
            "warning_count": int(env.data.warning.number.sum()),
            "steps": len(rows["action"]),
            "simulated_seconds": env.data.time,
            "wall_seconds": time.perf_counter() - episode_start,
            "state_sha256": sha256(path),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    manifest = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "seed": args.seed,
        "source": "Retained student's own live physical observations and executed actions",
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.output / "model.mjb"),
        "teacher_present": False,
        "walking_action_mask": True,
        "command_switches_reset_memory": False,
        "observations_are_causal": True,
        "setup_seconds": setup_seconds,
        "collection_seconds": time.perf_counter() - collection_start,
        "environment": env.report(),
        "control_hz": 1 / CONTROL_DT,
        "physics": "native MuJoCo CPU",
        "brain_device": str(device),
        "live_physics_worlds": 1,
        "episodes": reports,
        "training_rejection_rule": "Reject whole episodes with numerical failure, unstable body or prohibited support; retain their files",
        "does_not_establish_tracking_acceptance": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("graph", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44001)
    collect(parser.parse_args())
