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

from embodied_fly.body import CONTROL_DT, FlyEnvironment
from embodied_fly.evaluate import load_actor
from embodied_fly.flight_teacher import FlightTeacherOracle
from embodied_fly.observations import append_wing_angles, append_wing_velocity
from embodied_fly.provenance import evidence, sha256, utc_now


@torch.no_grad()
def collect(args):
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.episodes < 4:
        raise ValueError("Positive duration and at least four episodes required")
    if not 0 <= args.student_fraction <= 1:
        raise ValueError("Student actuator fraction must be between zero and one")
    if bool(args.student_checkpoint) != bool(args.graph):
        raise ValueError("Student checkpoint and graph must be supplied together")
    if args.student_fraction and not args.student_checkpoint:
        raise ValueError("A nonzero student fraction requires a student checkpoint")
    warmup = getattr(args, "teacher_warmup_seconds", 0.0)
    if not np.isfinite(warmup) or not 0 <= warmup < args.seconds:
        raise ValueError("Teacher warmup must be finite and shorter than the capture")
    if warmup and not args.student_checkpoint:
        raise ValueError("Teacher warmup requires a continuously observed student")
    reset_at_release = getattr(args, "reset_memory_at_release", False)
    if reset_at_release and not warmup:
        raise ValueError("Release memory ablation requires a teacher warmup")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    env = FlyEnvironment("flight")
    warmup_steps = round(warmup / env.control_dt)
    if not np.isclose(warmup, warmup_steps * env.control_dt, atol=1e-12, rtol=0):
        raise ValueError("Teacher warmup must be an integer number of action intervals")
    teacher = FlightTeacherOracle(env, args.teacher, args.wing_pattern)
    device = torch.device(args.device)
    actor = None
    if args.student_checkpoint:
        actor, _ = load_actor(args.student_checkpoint, args.graph, device)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rng = np.random.default_rng(args.seed)
    collection_started = time.perf_counter()
    reports = []
    for episode in range(args.episodes):
        speed = (0, 5, 10, 20)[episode % 4]
        phase = float(rng.random())
        teacher.initialize(speed, args.seconds, phase)
        memory = actor.initial_state(1) if actor is not None else None
        rows = {
            k: []
            for k in ("qpos", "qvel", "activation", "ctrl", "observation", "action", "time")
        }
        if actor is not None:
            rows.update(student_action=[], executed_action=[], executed_student_fraction=[])
        errors, heights, tilts = [], [], []
        failure = physical_failure = None
        first_release_failure = None
        episode_started = time.perf_counter()
        for step in range(round(args.seconds / env.control_dt)):
            observation = env.observation()
            action = teacher.act(step)
            executed = action
            if actor is not None:
                if warmup and step == warmup_steps:
                    if reset_at_release:
                        memory = actor.initial_state(1)
                    rows["release_neural_state"] = memory.cpu().numpy().copy()
                actor_observation = (
                    append_wing_velocity(observation, env.data.qvel, env.wing_velocity_indices)
                    if actor.sensor_extension_size >= 6
                    else observation
                )
                if actor.sensor_extension_size == 12:
                    actor_observation = append_wing_angles(
                        actor_observation, env.data.qpos, env.wing_angle_indices
                    )
                result = actor(
                    torch.as_tensor(actor_observation[None], device=device),
                    memory,
                    time_scale=env.control_dt / CONTROL_DT,
                )
                memory = result.state
                student = result.action[0].cpu().numpy()
                fraction = 0.0 if step < warmup_steps else args.student_fraction
                executed = (
                    student.copy()
                    if fraction == 1
                    else fraction * student + (1 - fraction) * action
                )
                if not np.isfinite(executed).all() or np.max(np.abs(executed)) > 1.000001:
                    raise RuntimeError("Nonfinite or unbounded mixed actuator command")
                rows["student_action"].append(student.copy())
                rows["executed_action"].append(executed.copy())
                rows["executed_student_fraction"].append(fraction)
            for k, value in (
                ("qpos", env.data.qpos),
                ("qvel", env.data.qvel),
                ("activation", env.data.act),
                ("ctrl", env.data.ctrl),
                ("observation", observation),
                ("action", action),
                ("time", env.data.time),
            ):
                rows[k].append(np.array(value, copy=True))
            try:
                env.step(executed)
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
                if step >= warmup_steps and first_release_failure is None:
                    first_release_failure = float(env.data.time - warmup)
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
            "after_teacher_warmup": {
                "frames": max(0, len(errors) - warmup_steps),
                "minimum_height_m": min(heights[warmup_steps:])
                if len(heights) > warmup_steps
                else None,
                "minimum_upright": min(tilts[warmup_steps:])
                if len(tilts) > warmup_steps
                else None,
                "root_tracking_rmse_m": float(
                    np.sqrt(np.mean(np.square(errors[warmup_steps:])))
                )
                if len(errors) > warmup_steps
                else None,
                "first_envelope_failure_seconds_after_release": first_release_failure,
            }
            if warmup
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
        "source": "Inherited flight teacher plus supplied wing pattern; complete body and individual bounded actuator labels"
        if actor is None
        else "Teacher corrections on states visited by a physical student/teacher actuator mixture; supervised dataset aggregation, not RL or student acceptance",
        "student_present": actor is not None,
        "student_checkpoint_sha256": sha256(args.student_checkpoint)
        if actor is not None
        else None,
        "graph_sha256": sha256(args.graph / "weights.npz") if actor is not None else None,
        "student_fraction": args.student_fraction,
        "teacher_warmup_seconds": warmup,
        "reset_memory_at_release": reset_at_release,
        "release_diagnostic": bool(warmup),
        "release_diagnostic_scope": "Student observes causal physical teacher-driven warmup with persistent memory, then takes its declared actuator fraction. Teacher still supplies counterfactual training labels, never an acceptance result."
        if warmup
        else None,
        "student_inference_device": str(device) if actor is not None else None,
        "executed_action": "teacher label"
        if actor is None
        else "Per-frame executed_student_fraction * student_action + (1-executed_student_fraction) * teacher label; saved separately. Fraction 1 executes student alone.",
        "recurrent_memory": "Diagnostic ablation clears memory exactly at teacher release; no physical state reset"
        if reset_at_release
        else "Reset only between episodes; persists at every 5 kHz decision",
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
        "recorded_observation_size": len(env.observation()),
        "student_observation_size": actor.observation_size if actor is not None else None,
        "student_sensor_extension_size": actor.sensor_extension_size
        if actor is not None
        else None,
        "student_observation_limit": "Legacy capture stores 383 fields; student appends its declared measured wing velocity/angle extension. No altitude target, camera pixels or teacher phase enters student input.",
        "episodes": reports,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("teacher", "wing-pattern", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--student-fraction", type=float, default=0)
    parser.add_argument("--teacher-warmup-seconds", type=float, default=0)
    parser.add_argument("--reset-memory-at-release", action="store_true")
    parser.add_argument("--device", default="cuda")
    collect(parser.parse_args())
