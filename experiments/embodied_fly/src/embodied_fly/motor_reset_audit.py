"""Compare frozen motor inference with training batch size and initial sensors.

This is a physical diagnostic, not learning. Reset assignments reproduce existing
evaluation starts; all subsequent movement uses the frozen actor and normal forces.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.motor_focus import TASKS, MotorTasks, MotorTeacher
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def audit(args):
    if args.seconds <= 0 or args.worlds < 3:
        raise ValueError("Need a positive duration and at least three worlds")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    source_report = json.loads((args.captures / "report.json").read_text())
    if source_report["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Saved comparison must belong to this exact checkpoint")
    env = FlyBatch(args.worlds, 16, 14, preset="wing_motion")
    if checkpoint["physical_contract"] != physical_contract(env.model):
        raise ValueError("Audit requires exact checkpoint physics")
    tasks = MotorTasks(env, 72001)
    captures = {task: np.load(args.captures / f"{task}.npz") for task in TASKS}
    mapping = tasks.task_ids
    state = {
        key: np.stack([captures[TASKS[i]][field][0] for i in mapping])
        for key, field in (
            ("qpos", "qpos"),
            ("qvel", "qvel"),
            ("act", "activation"),
            ("ctrl", "ctrl"),
        )
    }
    env.reset(np.arange(args.worlds), state=state)
    env.command[:] = np.stack([captures[TASKS[i]]["command"][0] for i in mapping])
    env.requested_height_cm[:] = [
        captures[TASKS[i]]["requested_height_cm"][0] for i in mapping
    ]
    expected_obs = np.stack([captures[TASKS[i]]["observation"][0] for i in mapping])
    initial_obs = env.observation().copy()
    initial_state = {k: env.fields[k].copy() for k in state}
    # This constructor exists during training but is absent in normal evaluation.
    MotorTeacher(tasks, args.teacher, device, True, "state")
    teacher_setup_obs = env.observation().copy()
    setup_state_unchanged = all(
        np.array_equal(env.fields[k], v) for k, v in initial_state.items()
    )
    observation_delta = teacher_setup_obs - initial_obs
    obs = torch.as_tensor(teacher_setup_obs, device=device)
    with torch.no_grad():
        actor.train()
        train_mode = actor(obs, actor.initial_state(args.worlds)).action
        actor.eval()
        eval_mode = actor(obs, actor.initial_state(args.worlds)).action
        small = actor(obs[:3], actor.initial_state(3)).action
    expected_action = np.stack([captures[task]["action"][0] for task in TASKS])
    checks = {
        "initial_observation_max_delta_from_saved_evaluation": float(
            abs(initial_obs - expected_obs).max()
        ),
        "teacher_setup_observation_max_delta": float(abs(observation_delta).max()),
        "teacher_setup_changed_input_columns": np.flatnonzero(
            np.any(observation_delta != 0, axis=0)
        ).tolist(),
        "teacher_setup_physical_state_unchanged": setup_state_unchanged,
        "train_eval_mode_action_max_delta": float((train_mode - eval_mode).abs().max()),
        "three_world_first_action_max_delta_from_saved": float(
            abs(small.cpu().numpy() - expected_action).max()
        ),
        "batch_size_first_action_max_delta": float((eval_mode[:3] - small).abs().max()),
    }
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    memory = actor.initial_state(args.worlds)
    first_failure = np.full(args.worlds, np.nan)
    minimum_height = np.full(args.worlds, np.inf)
    minimum_upright = np.full(args.worlds, np.inf)
    maximum_load = np.zeros(args.worlds)
    rows = []
    synchronize(device)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        observation = env.observation()
        with torch.no_grad():
            result = actor(torch.as_tensor(observation, device=device), memory)
        memory = result.state
        action = result.action.cpu().numpy()
        rows.append(
            {
                "qpos": env.fields["qpos"][:3].copy(),
                "qvel": env.fields["qvel"][:3].copy(),
                "observation": observation[:3].copy(),
                "action": action[:3].copy(),
            }
        )
        env.step(action)
        failed = tasks.failed()
        first_failure[np.isnan(first_failure) & failed] = (step + 1) * env.control_dt
        minimum_height = np.minimum(minimum_height, env.fields["qpos"][:, 2])
        minimum_upright = np.minimum(
            minimum_upright, env.fields["xmat"][:, env.template.thorax_id, 8]
        )
        maximum_load = np.maximum(maximum_load, env.forbidden_peak / env.body_weight)
    synchronize(device)
    elapsed = time.perf_counter() - started
    trace = args.output / "first_three_worlds.npz"
    np.savez_compressed(trace, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_evaluation_sha256": sha256(args.captures / "report.json"),
        "physical_contract": physical_contract(env.model),
        "checks": checks,
        "worlds": args.worlds,
        "seconds_per_world": len(rows) * env.control_dt,
        "physical_transitions": len(rows) * args.worlds,
        "training_updates": 0,
        "runtime_teacher_actions": False,
        "setup_seconds": setup,
        "frozen_rollout_seconds": elapsed,
        "brain_device": str(device),
        "physics_backend": "native CPU MuJoCo/mjbatch",
        "warnings": int(env.fields["warning"].sum()),
        "trace_sha256": sha256(trace),
        "model_sha256": sha256(args.output / "model.mjb"),
        "results": [
            {
                "world": i,
                "task": TASKS[mapping[i]],
                "first_envelope_failure_seconds": float(t) if np.isfinite(t) else None,
                "minimum_height_cm": float(minimum_height[i]),
                "minimum_upright": float(minimum_upright[i]),
                "maximum_forbidden_load_over_weight": float(maximum_load[i]),
            }
            for i, t in enumerate(first_failure)
        ],
        "scope": "Fixed final checkpoint, matched saved starts; batch and setup diagnostic, not learned success or generalization",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "graph", "teacher", "captures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=1.0)
    audit(parser.parse_args())
