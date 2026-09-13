"""Continuous ground commands in the current full fly, without motor assistance."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.motor_focus import MotorTasks
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.transitions import GATES, summarize_phase

PHASES = (("stand", 0.0), ("walk", 1.0), ("stop", 0.0), ("resume", 1.0))


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = FlyBatch(3, 3, 14, preset="wing_position")
    if not actor.motor_only or checkpoint["physical_contract"] != physical_contract(env.model):
        raise ValueError("The motor actor and canonical physical body must match")
    tasks = MotorTasks(env, args.seed, task_set="ground")
    tasks.task_ids[:] = 0
    tasks.reset(np.arange(3))
    posture = GroundPosture(env, tasks.ground["qpos"])
    memory = actor.initial_state(3)
    projection = NeuralProjection.from_graph(args.graph, device) if args.neural_view else None
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    started = time.perf_counter()
    frames, maps, phases, events = [], [], [], []
    numerical_failure = None
    steps = round(args.phase_seconds / env.control_dt)
    for phase_id, (name, forward) in enumerate(PHASES):
        before = {k: v.copy() for k, v in env.fields.items()}
        memory_before = memory.clone()
        env.command[:] = (forward, 0, 0)
        tasks.task_ids[:] = int(forward != 0)
        # A command boundary changes sensory requests only, never state or memory.
        assert all(np.array_equal(v, env.fields[k]) for k, v in before.items())
        assert torch.equal(memory, memory_before)
        events.append(
            {
                "phase": name,
                "time_seconds": len(frames) * env.control_dt,
                "physical_state_unchanged_at_boundary": True,
                "neural_memory_unchanged_at_boundary": True,
                "preceding_memory_l2": float(memory.norm()),
            }
        )
        initial = env.fields["qpos"][:, :3].copy()
        measured = []
        for _ in range(steps):
            observation = env.observation()
            result = actor(torch.as_tensor(observation, device=device), memory)
            memory = result.state
            if projection is not None and len(frames) % 10 == 0:
                maps.append(projection.project(memory).astype(np.float16))
            action = result.action.cpu().numpy()
            frames.append(
                {
                    "qpos": env.fields["qpos"].copy(),
                    "qvel": env.fields["qvel"].copy(),
                    "activation": env.fields["act"].copy(),
                    "ctrl": env.fields["ctrl"].copy(),
                    "action": action.copy(),
                    "observation": observation.copy(),
                    "command": env.command.copy(),
                    "time": np.full(3, len(frames) * env.control_dt),
                    "phase": np.full(3, phase_id),
                    "requested_height_cm": env.requested_height_cm.copy(),
                    "initial_qpos": np.repeat(tasks.ground["qpos"][None], 3, axis=0),
                    "wing_activity": env.wing_forces.activity.copy(),
                    "wing_wrench": env.wing_forces.wrench.copy(),
                }
            )
            try:
                env.step(action)
            except RuntimeError as error:
                numerical_failure = {
                    "message": str(error),
                    "phase": name,
                    "action_index": len(frames) - 1,
                }
            frames[-1]["post_action_qpos"] = env.fields["qpos"].copy()
            measured.append(
                {
                    "position": env.fields["qpos"][:, :3].copy(),
                    "velocity": env.velocity(),
                    "upright": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
                    "support": env.forbidden_peak.copy() / env.body_weight,
                    **posture.measure(),
                }
            )
            if numerical_failure is not None:
                break
        arrays = {k: np.stack([x[k] for x in measured]) for k in measured[0]}
        for i in range(3):
            row = summarize_phase(
                "stop" if forward == 0 else name,
                [forward, 0, 0],
                initial[i],
                arrays["position"][:, i],
                arrays["velocity"][:, i],
                arrays["upright"][:, i],
                arrays["position"][:, i, 2] * 0.01,
                float(arrays["support"][:, i].max()),
                expected_frames=steps,
            )
            row.update(
                phase=name,
                world=i,
                wing_max_deviation_rad=float(arrays["wing_max_deviation_rad"][:, i].max()),
                wing_speed_rms_rad_s=float(
                    np.sqrt(arrays["wings_velocity_mse_rad2_s2"][:, i].mean())
                ),
                leg_pose_rms_rad=float(np.sqrt(arrays["legs_angle_mse_rad2"][:, i].mean())),
            )
            row.setdefault("stable", False)
            row["resting_wings_pass"] = (
                row["wing_max_deviation_rad"] < 0.2 and row["wing_speed_rms_rad_s"] < 2
            )
            row["success"] &= row["resting_wings_pass"]
            if forward == 0:
                late = round(GATES["settle_seconds"] / env.control_dt)
                row["late_hold_form_rms_rad"] = {
                    group: float(np.sqrt(arrays[group + "_angle_mse_rad2"][late:, i].mean()))
                    if len(measured) > late
                    else None
                    for group in posture.groups
                }
                row["hold_form_pass"] = all(
                    value is not None and value < 0.15
                    for value in row["late_hold_form_rms_rad"].values()
                )
                row["maximum_height_loss_fraction"] = float(
                    arrays["body_height_loss_fraction"][:, i].max()
                )
                row["hold_form_pass"] &= row["maximum_height_loss_fraction"] < 0.1
                row["success"] &= row["hold_form_pass"]
            phases.append(row)
        if numerical_failure is not None:
            break
    elapsed = time.perf_counter() - started
    cases = []
    for i in range(3):
        capture = {key: np.stack([f[key][i] for f in frames]) for key in frames[0]}
        np.testing.assert_array_equal(capture["qpos"][1:], capture["post_action_qpos"][:-1])
        if projection is not None:
            capture.update(
                neural_map=np.stack([m[i] for m in maps]),
                neural_occupancy=projection.occupancy,
                neural_map_stride=10,
            )
        case = f"sequence_{i}"
        np.savez_compressed(args.output / f"{case}.npz", **capture)
        rows = [p for p in phases if p["world"] == i]
        cases.append(
            {
                "case": case,
                "success": numerical_failure is None
                and len(rows) == len(PHASES)
                and all(p["success"] for p in rows),
                "stable": all(p["stable"] for p in rows),
                "gate_label": "Continuous command gates",
                "phases": rows,
                "state_sha256": sha256(args.output / f"{case}.npz"),
            }
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": checkpoint["physical_contract"],
        "model_sha256": sha256(args.output / "model.mjb"),
        "physical_preset": "wing_position",
        "motor_only": True,
        "teacher_present": False,
        "student_present": True,
        "controller": "student",
        "one_checkpoint_for_all_cases": True,
        "policy_acceptance_eligible": True,
        "walking_action_mask": False,
        "active_actuators": 78,
        "live_resets": 0,
        "neural_resets_at_commands": 0,
        "phase_schedule": list(PHASES),
        "phase_seconds": args.phase_seconds,
        "boundary_events": events,
        "seed": args.seed,
        "gates": GATES,
        "posture_gates": posture.report()["gates"],
        "numerical_failure": numerical_failure,
        "setup_seconds": setup,
        "stepping_and_capture_seconds": elapsed,
        "parallel_worlds": 3,
        "transitions": len(frames) * 3,
        "simulated_seconds_per_world": len(frames) * env.control_dt,
        "physics_hz": 5000,
        "control_hz": 500,
        "warning_count": int(env.fields["warning"].sum()),
        "neural_view": projection.report() if projection is not None else None,
        "results": cases,
        "scope": "Continuous commanded ground motors; no utility selection, turns or flight transitions",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["transitions", "stepping_and_capture_seconds", "results"]}
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase-seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=99103)
    parser.add_argument("--neural-view", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.phase_seconds) or args.phase_seconds < 0.5:
        parser.error("At least half a second per phase required")
    r = run(args)
    raise SystemExit(0 if all(c["success"] for c in r["results"]) else 2)
