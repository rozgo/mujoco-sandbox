"""One continuous walk/stop/walk trial; no physical or neural reset at commands."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT, FlyEnvironment
from embodied_fly.evaluate import load_actor
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.provenance import evidence, sha256, utc_now

PHASES = (("walk", 2.0, (1, 0, 0)), ("stop", 2.0, (0, 0, 0)), ("resume", 2.0, (1, 0, 0)))
GATES = {
    "settle_seconds": 0.25,
    "minimum_upright": 0.5,
    "minimum_height_m": 0.0006,
    "max_forbidden_load_over_weight": 0.1,
    "moving_mean_forward_error_cm_s": 0.25,
    "moving_mean_lateral_error_cm_s": 0.25,
    "moving_mean_yaw_error_rad_s": 0.35,
    "stop_late_planar_speed_rms_cm_s": 0.1,
    "stop_block_yaw_rms_rad_s": 0.3,
    "stop_late_drift_m": 0.001,
    "stop_total_drift_m": 0.002,
}


def summarize_phase(
    name,
    command,
    before,
    positions,
    velocities,
    upright,
    heights,
    support_ratio,
    expected_frames=1000,
):
    velocity, positions = np.asarray(velocities), np.asarray(positions)
    skip = round(GATES["settle_seconds"] / CONTROL_DT)
    late = velocity[skip:]
    if len(late) < round(0.1 / CONTROL_DT):
        return {"phase": name, "success": False, "reason": "incomplete phase"}
    block = round(0.1 / CONTROL_DT)
    usable = len(late) // block * block
    yaw_blocks = late[:usable, 2].reshape(-1, block).mean(1)
    mean = late.mean(0)
    total_drift = float(np.linalg.norm(positions[-1, :2] - before[:2]) * 0.01)
    late_drift = float(np.linalg.norm(positions[-1, :2] - positions[skip, :2]) * 0.01)
    planar_rms = float(np.sqrt(np.mean(np.sum(late[:, 3:5] ** 2, axis=1))))
    yaw_rms = float(np.sqrt(np.mean(yaw_blocks**2)))
    stable = (
        min(upright) > GATES["minimum_upright"] and min(heights) > GATES["minimum_height_m"]
    )
    support_valid = support_ratio < GATES["max_forbidden_load_over_weight"]
    tracking = (
        planar_rms < GATES["stop_late_planar_speed_rms_cm_s"]
        and yaw_rms < GATES["stop_block_yaw_rms_rad_s"]
        and late_drift < GATES["stop_late_drift_m"]
        and total_drift < GATES["stop_total_drift_m"]
        if name == "stop"
        else abs(mean[3] - command[0]) < GATES["moving_mean_forward_error_cm_s"]
        and abs(mean[4] - command[1]) < GATES["moving_mean_lateral_error_cm_s"]
        and abs(mean[2] - command[2]) < GATES["moving_mean_yaw_error_rad_s"]
    )
    return {
        "phase": name,
        "command_cm_s_rad_s": list(command),
        "frames": len(velocity),
        "post_settle_frames": len(late),
        "stable": bool(stable),
        "support_valid": bool(support_valid),
        "success": bool(
            len(velocity) == expected_frames and stable and support_valid and tracking
        ),
        "complete": len(velocity) == expected_frames,
        "mean_velocity_after_settle_cm_s_rad_s": mean[[3, 4, 2]].tolist(),
        "late_planar_speed_rms_cm_s": planar_rms,
        "late_block_yaw_rms_rad_s": yaw_rms,
        "total_planar_displacement_m": total_drift,
        "late_planar_displacement_m": late_drift,
        "minimum_upright": float(min(upright)),
        "minimum_height_m": float(min(heights)),
        "max_disallowed_ground_force_over_weight": float(support_ratio),
    }


@torch.no_grad()
def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    provenance = evidence()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    environment = FlyEnvironment()
    environment.reset(yaw=0.1)
    memory = actor.initial_state(1)
    projection = NeuralProjection.from_graph(args.graph, device) if args.neural_view else None
    mujoco.mj_saveModel(environment.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - setup_start
    captures = {
        k: []
        for k in (
            "qpos",
            "qvel",
            "activation",
            "ctrl",
            "action",
            "utility",
            "command",
            "phase",
            "velocity",
            "position_after_action",
        )
    }
    neural_maps, phase_results, boundary_events = [], [], []
    started = time.perf_counter()
    frame = 0
    numerical_failure = None
    for phase_id, (name, seconds, command) in enumerate(PHASES):
        environment.command[:] = command
        before = environment.data.qpos[:3].copy()
        boundary_events.append(
            {
                "phase": name,
                "physics_time": float(environment.data.time),
                "root_position_cm": before.tolist(),
                "preceding_memory_l2": float(memory.norm()),
                "reset_performed": False,
            }
        )
        # Reset a diagnostic accumulator only, never body state or neural memory.
        environment.maximum_disallowed_ground_force = 0.0
        velocities, positions, upright, heights = [], [], [], []
        for _ in range(round(seconds / CONTROL_DT)):
            result = actor(
                torch.as_tensor(environment.observation()[None], device=device), memory
            )
            memory = result.state
            action = environment.walking_action(result.action[0].cpu().numpy())
            for key, value in (
                ("qpos", environment.data.qpos),
                ("qvel", environment.data.qvel),
                ("activation", environment.data.act),
                ("ctrl", environment.data.ctrl),
                ("action", action),
                ("utility", result.utility_scores[0].cpu().numpy()),
                ("command", environment.command),
            ):
                captures[key].append(value.copy())
            captures["phase"].append(phase_id)
            if projection is not None and frame % 10 == 0:
                neural_maps.append(projection.project(memory)[0].astype(np.float16))
            try:
                environment.step(action)
            except RuntimeError as error:
                numerical_failure = str(error)
                break
            velocity = environment.anatomical_velocity()
            position = environment.data.qpos[:3].copy()
            captures["velocity"].append(velocity)
            captures["position_after_action"].append(position)
            velocities.append(velocity)
            positions.append(position)
            upright.append(float(environment.data.xmat[environment.thorax_id, 8]))
            heights.append(float(position[2] * 0.01))
            frame += 1
        phase_results.append(
            summarize_phase(
                name,
                command,
                before,
                positions,
                velocities,
                upright,
                heights,
                environment.maximum_disallowed_ground_force
                / (environment.model.body_mass.sum() * 981),
            )
        )
        if numerical_failure:
            break
    arrays = {key: np.asarray(value) for key, value in captures.items()}
    if projection is not None:
        arrays.update(
            neural_map=neural_maps, neural_occupancy=projection.occupancy, neural_map_stride=10
        )
    np.savez_compressed(args.output / "walk_stop_walk.npz", **arrays)
    success = (
        numerical_failure is None
        and len(phase_results) == 3
        and all(p["success"] for p in phase_results)
    )
    result = {
        "case": "walk_stop_walk",
        "success": success,
        "stable": all(p.get("stable", False) for p in phase_results),
        "gate_label": "Transition gates",
        "phases": phase_results,
        "simulated_seconds": float(environment.data.time),
        "wall_seconds": time.perf_counter() - started,
        "numerical_failure": numerical_failure,
        "warning_count": int(environment.data.warning.number.sum()),
        "state_sha256": sha256(args.output / "walk_stop_walk.npz"),
    }
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "setup_seconds": setup_seconds,
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.output / "model.mjb"),
        "training_method": checkpoint.get("method"),
        "one_checkpoint_for_all_cases": True,
        "teacher_present": False,
        "scripted_gait_present": False,
        "walking_action_mask": True,
        "active_actuators": 59,
        "neural_view": projection.report() if projection else None,
        "gates": GATES,
        "command_boundary_events": boundary_events,
        "physics": "native MuJoCo CPU",
        "neural_device": str(device),
        "scope": "development transition test, separate from original six fixed-command gates",
        "results": [result],
        "physical_success_count": int(success),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neural-view", action="store_true")
    raise SystemExit(0 if evaluate(parser.parse_args()) else 2)
