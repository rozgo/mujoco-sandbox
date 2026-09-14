"""Student-only physical flight on the identical velocity-teacher plant."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment, post_row, pre_row
from embodied_fly.velocity_exercise import (
    DURATION,
    STAGES,
    command_at,
    metrics,
    rolling_velocity,
)
from embodied_fly.velocity_motor import SCHEMA, observation


def completed_stage_metrics(arrays, failure_seconds=None):
    """Score only complete settled windows before any physical failure."""
    velocity = rolling_velocity(arrays["measured_velocity"] * 10)
    yaw = rolling_velocity(arrays["yaw_rate"][:, None])[:, 0]
    valid_until = min(
        float(arrays["time"][-1]) + 0.002,
        failure_seconds if failure_seconds is not None else float("inf"),
    )
    ends = np.cumsum([s[1] for s in STAGES])
    result = []
    for stage, (end, (_, _, command)) in enumerate(zip(ends, STAGES, strict=True)):
        if end > valid_until + 1e-9:
            continue
        selected = (arrays["time"] >= end - 0.4) & (arrays["time"] < end)
        if not selected.any():
            continue
        velocity_error = float(
            np.linalg.norm(
                velocity[selected] - arrays["command"][selected, :3] * 10, axis=1
            ).max()
        )
        yaw_error = float(np.abs(yaw[selected] - arrays["command"][selected, 3]).max())
        velocity_limit = max(0.5, 1.5 * np.linalg.norm(command[:3]))
        yaw_limit = max(0.12, 0.45 * abs(command[3]))
        result.append(
            {
                "stage": stage,
                "name": STAGES[stage][0],
                "settled_velocity_error_peak_mm_s": velocity_error,
                "settled_yaw_error_peak_rad_s": yaw_error,
                "passed": velocity_error < velocity_limit and yaw_error < yaw_limit,
            }
        )
    return result


@torch.no_grad()
def evaluate(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if checkpoint.get("observation_schema") != SCHEMA:
        raise ValueError("Expected the velocity-command actor")
    teacher = json.loads((args.dataset / "report.json").read_text())
    env = environment(1, 1)
    if (
        physical_contract(env.model) != checkpoint["physical_contract"]
        or checkpoint["physical_contract"] != teacher["physical_contract"]
    ):
        raise ValueError("Student does not match the exact teacher physics")
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    cases = []
    for episode in (0, 8):
        reference = args.dataset / f"episode_{episode:02d}.npz"
        with np.load(reference) as data:
            reset = {k: data[k][0:1].copy() for k in ("qpos", "qvel", "act", "ctrl")}
        env.reset(np.array([0]), state=reset)
        memory = actor.initial_state(1)
        rows = []
        failure = None
        first_failure = None
        synchronize(device)
        begin = time.perf_counter()
        for step in range(round(DURATION / env.control_dt)):
            seconds = step * env.control_dt
            command, stage = command_at(seconds, 1.5, 4.5)
            row = pre_row(env, command[None], seconds, stage)
            obs = observation(env, command[None])
            result = actor(torch.as_tensor(obs, device=device), memory)
            memory = result.state
            action = result.action.cpu().numpy()
            row["action"] = action
            row["observation"] = obs
            try:
                env.step(action)
            except RuntimeError as error:
                failure = str(error)
                break
            row.update(post_row(env))
            rows.append({k: v[0].copy() for k, v in row.items()})
            reason = (
                "altitude below 5 mm"
                if row["post_position"][0, 2] < 0.5
                else "upright below 0.85"
                if row["upright"][0] < 0.85
                else "forbidden ground contact"
                if row["forbidden"][0] > 0.1
                else None
            )
            if reason and first_failure is None:
                first_failure = (step + 1) * env.control_dt
                failure = reason
            # Brief physical continuation makes the failure visible, without resets.
            if (
                first_failure is not None
                and (step + 1) * env.control_dt >= first_failure + 0.4
            ):
                break
            if step and step % 5000 == 0:
                print(
                    json.dumps({"episode": episode, "simulated_seconds": seconds}), flush=True
                )
        synchronize(device)
        evaluation_seconds = time.perf_counter() - begin
        if not rows:
            raise RuntimeError("No valid student trajectory was captured")
        arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        file = args.output / f"episode_{episode:02d}.npz"
        np.savez_compressed(file, **arrays)
        velocity = rolling_velocity(arrays["measured_velocity"] * 10)
        duration = len(rows) * env.control_dt
        complete = failure is None and len(rows) == round(DURATION / env.control_dt)
        full = metrics(arrays, 10) if complete else None
        completed_stages = completed_stage_metrics(arrays, first_failure)
        summary = {
            "episode": episode,
            "file": file.name,
            "capture_sha256": sha256(file),
            "teacher_capture_sha256": sha256(reference),
            "duration_seconds": duration,
            "evaluation_wall_seconds": evaluation_seconds,
            "failure": failure,
            "first_failure_seconds": first_failure,
            "completed_full_exercise": complete,
            "passed": bool(full and full["passed"]),
            "stage_count": 50,
            "stages_reached": np.unique(arrays["stage"]).tolist(),
            "stages_passed": sum(s["passed"] for s in completed_stages),
            "completed_stage_metrics": completed_stages,
            "incomplete_stage_metrics_not_scored": not complete,
            "minimum_altitude_mm": float(arrays["post_position"][:, 2].min() * 10),
            "height_span_mm": float(np.ptp(arrays["post_position"][:, 2]) * 10),
            "final_altitude_mm": float(arrays["post_position"][-1, 2] * 10),
            "minimum_upright": float(arrays["upright"].min()),
            "conditional_velocity_rms_mm_s": float(
                np.sqrt(
                    np.mean(np.sum((velocity - arrays["command"][:, :3] * 10) ** 2, axis=1))
                )
            ),
            "metrics_scope": "Recorded interval only; early termination is a failure, never full-task tracking success",
            "full_exercise_metrics": full,
        }
        cases.append(summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "full_exercise_metrics"}),
            flush=True,
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.output / "model.mjb"),
        "physical_contract": checkpoint["physical_contract"],
        "cumulative_training_seconds": checkpoint.get("cumulative_training_seconds", 0),
        "cumulative_updates": checkpoint["training_updates"],
        "physics_hz": 1000,
        "control_hz": 500,
        "no_teacher_actions": True,
        "no_phase_input": True,
        "no_action_masks": True,
        "all_78_actions_from_actor": True,
        "physical_resets": "initialization only, once per case",
        "exercise_duration_seconds": DURATION,
        "stage_names": [s[0] for s in STAGES],
        "cases": cases,
        "passed": all(c["passed"] for c in cases),
        "setup_seconds": setup_seconds,
        "total_wall_seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())
