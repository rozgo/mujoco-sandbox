"""Complete, continuous PID exercises for fresh velocity motor imitation."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_exercise import DURATION, STAGES, command_at, metrics
from embodied_fly.velocity_motor import VelocityPID, measured_velocity, observation


def environment(worlds, threads):
    return FlyBatch(
        worlds,
        threads,
        12,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
        heading_control=True,
        fast_flight=True,
        lateral_control=True,
    )


def initialize_worlds(env, headings):
    tasks = HoverOnlyTasks(env, 121101)
    # Initialization only. Preserve the accepted full-body posture and cold wings.
    state = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    state["qpos"][:, 3:7] = 0
    state["qpos"][:, 3] = np.cos(np.asarray(headings) / 2)
    state["qpos"][:, 6] = np.sin(np.asarray(headings) / 2)
    env.reset(np.arange(env.n), state=state)
    return tasks.air_action


def pre_row(env, commands, seconds, stage, *, refresh=True):
    if refresh:
        env.batch.forward()
    rows = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    rows.update(
        time=np.full(env.n, seconds),
        stage=np.full(env.n, stage, dtype=np.int32),
        command=commands.copy(),
        measured_velocity=measured_velocity(env).copy(),
        yaw_rate=env.velocity()[:, 2].copy(),
        heading=np.arctan2(
            env.fields["xmat"][:, env.template.thorax_id, 3],
            env.fields["xmat"][:, env.template.thorax_id, 0],
        ),
    )
    return rows


def post_row(env):
    return {
        "post_position": env.fields["qpos"][:, :3].copy(),
        "upright": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
        "forbidden": env.forbidden_peak.copy() / env.body_weight,
    }


def collect(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    env = environment(10, 10)
    accepted = json.loads(args.contract_report.read_text())
    assert physical_contract(env.model) == accepted["physical_contract"]
    # All ten episodes have all 50 stages. The last two never supply gradients.
    headings = np.array([0, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.25, -0.15, 0.15])
    phases = np.array([0, -0.35, 0.25, -0.2, 0.1, 0.35, -0.1, 0.2, 0.17, -0.17])
    base = initialize_worlds(env, headings)
    teachers = [VelocityPID(env.template, base, motion_feedforward=True) for _ in headings]
    steps = round(DURATION / env.control_dt)
    observations = np.lib.format.open_memmap(
        args.output / "observations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(env.n, steps, 391),
    )
    actions = np.lib.format.open_memmap(
        args.output / "actions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(env.n, steps, env.model.nu),
    )
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    capture_started = time.perf_counter()
    captures = None
    for step in range(steps):
        seconds = step * env.control_dt
        command, stage = command_at(seconds, 1.5, 4.5)
        commands = np.repeat(command[None], env.n, axis=0)
        row = pre_row(env, commands, seconds, stage)
        observations[:, step] = observation(env, commands)
        targets = []
        for world, teacher in enumerate(teachers):
            for key in ("qpos", "qvel", "act", "ctrl"):
                getattr(env.template.data, key)[:] = env.fields[key][world]
            # Offset only the teacher's oscillator clock, never the physical clock.
            env.template.data.time = seconds + phases[world] / (2 * np.pi * 30)
            mujoco.mj_forward(env.template.model, env.template.data)
            targets.append(teacher.act(command))
        targets = np.asarray(targets, np.float32)
        actions[:, step] = targets
        row["action"] = targets
        env.step(targets)
        row.update(post_row(env))
        if captures is None:
            captures = {k: np.empty((steps, *v.shape), v.dtype) for k, v in row.items()}
        for key, value in row.items():
            captures[key][step] = value
        if step % 2500 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "simulated_seconds": seconds,
                        "capture_seconds": time.perf_counter() - capture_started,
                    }
                ),
                flush=True,
            )
    capture_seconds = time.perf_counter() - capture_started
    observations.flush()
    actions.flush()
    cases = []
    for world in range(env.n):
        arrays = {k: v[:, world] for k, v in captures.items()}
        file = args.output / f"episode_{world:02d}.npz"
        np.savez_compressed(file, **arrays)
        cases.append(
            dict(
                world=world,
                split="train" if world < 8 else "validation",
                heading_rad=float(headings[world]),
                phase_rad=float(phases[world]),
                file=file.name,
                sha256=sha256(file),
                **metrics(arrays, 10),
            )
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "accepted_reference_sha256": sha256(args.contract_report),
        "observations_sha256": sha256(args.output / "observations.npy"),
        "actions_sha256": sha256(args.output / "actions.npy"),
        "train_episodes": 8,
        "validation_episodes": 2,
        "physical_worlds": 10,
        "cpu_threads": 10,
        "duration_per_episode_seconds": DURATION,
        "steps_per_episode": steps,
        "action_transitions": steps * env.n,
        "physics_steps": steps * env.n * 2,
        "neural_training_updates": 0,
        "episode_resets_after_initialization": 0,
        "stage_names": [s[0] for s in STAGES],
        "teacher_action_fraction": 1,
        "physics_hz": 1000,
        "control_hz": 500,
        "physical_command_speed_scale": 10,
        "setup_seconds": setup,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "cases": cases,
        "passed": all(c["passed"] for c in cases),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("provenance", "cases")}),
        flush=True,
    )
    if not report["passed"]:
        raise RuntimeError(
            "Teacher variation failed; retained all data and reports for review"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract-report", required=True, type=Path)
    collect(parser.parse_args())
