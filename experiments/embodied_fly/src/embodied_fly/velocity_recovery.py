"""Reordered complete PID exercises, starting from observed student mistakes."""

import argparse
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import (
    environment,
    initialize_worlds,
    post_row,
    pre_row,
)
from embodied_fly.velocity_exercise import (
    DURATION,
    STAGES,
    command_at,
    metrics,
    rolling_velocity,
)
from embodied_fly.velocity_motor import VelocityPID, observation


def exercise_order(seed):
    """Shuffle inverse movement pairs; keep each move's brake and safe vertical order.

    Vertical pairs climb before descending so reordering commands cannot request
    a deliberate floor collision from the accepted low airborne start.
    """
    pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 10),
        (9, 11),
        (12, 14),
        (13, 15),
        (16, 18),
        (17, 19),
        (20, 21),
        (22, 23),
    ]
    rng = np.random.default_rng(seed)
    order = [0]
    for index in rng.permutation(len(pairs)):
        pair = pairs[index]
        vertical = any(STAGES[1 + 2 * block][2][2] for block in pair)
        if not vertical and rng.integers(2):
            pair = pair[::-1]
        for block in pair:
            order.extend((1 + 2 * block, 2 + 2 * block))
    order.append(49)
    assert sorted(order) == list(range(50))
    return order


def stage_windows(orders):
    windows = np.empty((len(orders), 50, 2), dtype=np.int64)
    for world, order in enumerate(orders):
        end = 0
        for stage in order:
            start = end
            end += round(STAGES[stage][1] * 500)
            windows[world, stage] = start, end
    return windows


def recovery_starts(env, dataset, student):
    """Reset-only exact saved body/joint states; never alter live poses/velocities."""
    headings = [0.2, -0.25, 0, 0, 0, 0, 0, 0, 0, 0]
    base = initialize_worlds(env, headings)
    specification = [
        (0, 0.25),
        (0, 0.5),
        (0, 1.0),
        (0, 2.0),
        (0, 3.0),
        (0, 4.0),
        (8, 0.75),
        (8, 2.5),
    ]
    sources = [
        {"kind": "cold", "heading_rad": headings[i], "phase_rad": p}
        for i, p in enumerate((0.22, -0.27))
    ]
    reset = {key: env.fields[key].copy() for key in ("qpos", "qvel", "act", "ctrl")}
    previous = env.previous_action.copy()
    phases = [0.22, -0.27]
    angular_ids = env.template.wing_angle_indices
    speed_ids = env.template.wing_velocity_indices
    for world, (episode, seconds) in enumerate(specification, 2):
        path = student / f"episode_{episode:02d}.npz"
        with np.load(path) as a:
            index = round(seconds * 500)
            assert index < len(a["time"])
            for key, values in reset.items():
                values[world] = a[key][index]
            previous[world] = a["action"][index - 1]
        # Align the teacher oscillator from observed sweep angle and speed.
        # This phase is teacher state only; the actor receives measured sensors.
        angle = reset["qpos"][world, angular_ids].reshape(2, 3)[:, 0].mean()
        speed = reset["qvel"][world, speed_ids].reshape(2, 3)[:, 0].mean()
        phase = float(np.arctan2(angle, speed / (2 * np.pi * 30)))
        phases.append(phase)
        sources.append(
            {
                "kind": "student_physical_state",
                "source_episode": episode,
                "source_seconds": seconds,
                "source_sha256": sha256(path),
                "phase_rad": phase,
                "phase_source": "measured sweep angle/speed",
            }
        )
    env.reset(np.arange(env.n), state=reset)
    env.previous_action[:] = previous
    return base, np.asarray(phases), sources


def recovery_metrics(arrays):
    selected = arrays["time"] >= 1.6
    v = rolling_velocity(arrays["measured_velocity"] * 10)
    yaw = rolling_velocity(arrays["yaw_rate"][:, None])[:, 0]
    velocity_error = float(np.linalg.norm(v[selected], axis=1).max())
    yaw_error = float(abs(yaw[selected]).max())
    physical = bool(
        arrays["upright"].min() > 0.85
        and arrays["post_position"][:, 2].min() > 0.5
        and arrays["forbidden"].max() < 0.1
    )
    return {
        "passed": bool(physical and velocity_error < 0.5 and yaw_error < 0.12),
        "physical_envelope_passed": physical,
        "settled_velocity_error_peak_mm_s": velocity_error,
        "settled_yaw_error_peak_rad_s": yaw_error,
        "minimum_upright": float(arrays["upright"].min()),
        "minimum_altitude_mm": float(arrays["post_position"][:, 2].min() * 10),
        "initial_altitude_mm": float(arrays["qpos"][0, 2] * 10),
        "final_altitude_mm": float(arrays["post_position"][-1, 2] * 10),
        "gate": "After 1.6s: 100ms-mean speed <0.5mm/s, yaw <0.12rad/s; upright >0.85, height >5mm, forbidden contact <0.1 bodyweights throughout",
    }


def collect(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    original = json.loads((args.dataset / "report.json").read_text())
    student = json.loads((args.student / "report.json").read_text())
    env = environment(10, 10)
    assert (
        original["physical_contract"]
        == student["physical_contract"]
        == physical_contract(env.model)
    )
    for case in student["cases"]:
        assert sha256(args.student / case["file"]) == case["capture_sha256"]
    base, phases, starts = recovery_starts(env, args.dataset, args.student)
    teachers = [VelocityPID(env.template, base, motion_feedforward=True) for _ in starts]
    orders = [exercise_order(141000 + i) for i in range(10)]
    steps = 1000 if args.probe else round(DURATION / env.control_dt)
    if not args.probe:
        probe = json.loads(args.probe_report.read_text())
        assert (
            probe["passed"]
            and probe["student_checkpoint_sha256"] == student["checkpoint_sha256"]
        )
        assert probe["physical_contract"] == physical_contract(env.model)
    # Capture small physical traces separately; large observations use disk memmaps.
    obs = np.lib.format.open_memmap(
        args.output / "observations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(10 if args.probe else 20, steps, 391),
    )
    targets = np.lib.format.open_memmap(
        args.output / "actions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(10 if args.probe else 20, steps, 78),
    )
    offset = 0 if args.probe else 10
    if not args.probe:
        for name, destination in (("observations", obs), ("actions", targets)):
            assert sha256(args.dataset / f"{name}.npy") == original[f"{name}_sha256"]
            destination[:10] = np.load(args.dataset / f"{name}.npy", mmap_mode="r")
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    begin = time.perf_counter()
    captures = None
    for step in range(steps):
        seconds = step * 0.002
        schedules = [command_at(seconds, 1.5, 4.5, order) for order in orders]
        commands = np.asarray([c for c, _ in schedules])
        stages = np.asarray([s for _, s in schedules])
        row = pre_row(env, commands, seconds, stages)
        obs[offset:, step] = observation(env, commands)
        actions = []
        for world, teacher in enumerate(teachers):
            for key in ("qpos", "qvel", "act", "ctrl"):
                getattr(env.template.data, key)[:] = env.fields[key][world]
            env.template.data.time = seconds + phases[world] / (2 * np.pi * 30)
            mujoco.mj_forward(env.template.model, env.template.data)
            actions.append(teacher.act(commands[world]))
        actions = np.asarray(actions, np.float32)
        row["action"] = actions
        targets[offset:, step] = actions
        env.step(actions)
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
                        "capture_seconds": time.perf_counter() - begin,
                    }
                ),
                flush=True,
            )
    capture_seconds = time.perf_counter() - begin
    obs.flush()
    targets.flush()
    cases = []
    if not args.probe:
        for case in original["cases"]:
            shutil.copy2(args.dataset / case["file"], args.output / case["file"])
            cases.append(dict(case, source="preserved original teacher exercise"))
    for world in range(10):
        arrays = {k: v[:, world] for k, v in captures.items()}
        file = args.output / f"episode_{world + offset:02d}.npz"
        np.savez_compressed(file, **arrays)
        result = recovery_metrics(arrays) if args.probe else metrics(arrays, 10, orders[world])
        cases.append(
            dict(
                world=world + offset,
                file=file.name,
                sha256=sha256(file),
                split="train" if world < 8 else "validation",
                start=starts[world],
                stage_order=orders[world],
                **result,
            )
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "probe_only": args.probe,
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "student_checkpoint_sha256": student["checkpoint_sha256"],
        "source_dataset_report_sha256": sha256(args.dataset / "report.json"),
        "observations_sha256": sha256(args.output / "observations.npy"),
        "actions_sha256": sha256(args.output / "actions.npy"),
        "physical_worlds": 10,
        "cpu_threads": 10,
        "duration_per_episode_seconds": steps * 0.002,
        "steps_per_episode": steps,
        "new_action_transitions": steps * 10,
        "new_physics_steps": steps * 20,
        "stage_names": [s[0] for s in STAGES],
        "stage_orders": {str(i + offset): order for i, order in enumerate(orders)},
        "stage_windows": stage_windows(
            ([] if args.probe else [list(range(50))] * 10) + orders
        ).tolist(),
        "train_episode_ids": list(range(8)) + ([] if args.probe else list(range(10, 18))),
        "validation_episode_ids": [8, 9] if args.probe else [8, 9, 18, 19],
        "train_episodes": 8 if args.probe else 16,
        "validation_episodes": 2 if args.probe else 4,
        "original_dataset_rehearsal_fraction": 0 if args.probe else 0.5,
        "setup_seconds": setup,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "physics_hz": 1000,
        "control_hz": 500,
        "teacher_action_fraction": 1,
        "neural_training_updates": 0,
        "episode_resets_after_initialization": 0,
        "starts_are_saved_physical_states": True,
        "phase_alignment": "Teacher-only initial phase inferred from measured wing angle/speed; no phase sensor or phase action inserted",
        "cases": cases,
        "passed": all(c["passed"] for c in cases),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("provenance", "stage_windows", "stage_orders", "cases")
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise RuntimeError(
            "Teacher recovery/order gate failed; capture retained, training prohibited"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--probe-report", type=Path)
    args = p.parse_args()
    if not args.probe and args.probe_report is None:
        p.error("Full collection requires --probe-report")
    collect(args)
