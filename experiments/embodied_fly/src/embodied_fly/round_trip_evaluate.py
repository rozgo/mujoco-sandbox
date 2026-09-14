"""Deterministic learned round trips: same actor, seven commands, no teacher."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.round_trip import CASES
from embodied_fly.round_trip_tasks import batched_targets
from embodied_fly.train import synchronize


def metrics(arrays, origins):
    """Post-step target/state pairs; retain raw speed and sustained drift separately."""
    times = arrays["time"] + 0.002
    results = []
    drift = np.full(arrays["actual_position"].shape[:2], np.nan)
    drift[50:] = (
        np.linalg.norm(
            arrays["actual_position"][50:] - arrays["actual_position"][:-50], axis=2
        )
        * 100
    )
    for i, (name, axis, sign) in enumerate(CASES):
        position = arrays["actual_position"][:, i]
        error = np.linalg.norm(position - arrays["post_target"][:, i], axis=1) * 10
        bad = (
            (position[:, 2] < 0.5)
            | (arrays["upright"][:, i] < 0.5)
            | (arrays["forbidden"][:, i] > 0.1)
        )
        windows = {}
        for label, a, b in (
            ("first_target", 3.5, 4),
            ("opposite_target", 6.5, 7),
            ("returned_hold", 11, 12.001),
        ):
            mask = (times >= a) & (times < b)
            windows[label] = {
                "position_peak_mm": float(error[mask].max()),
                "position_rms_mm": float(np.sqrt(np.mean(error[mask] ** 2))),
                "raw_speed_rms_mm_s": float(
                    np.sqrt(np.mean(np.sum(arrays["actual_velocity"][mask, i] ** 2, axis=1)))
                    * 10
                ),
                "100ms_drift_peak_mm_s": float(np.nanmax(drift[mask, i])),
            }
        passed = (
            not bad.any()
            and all(w["position_peak_mm"] < 0.5 for w in windows.values())
            and windows["returned_hold"]["100ms_drift_peak_mm_s"] < 1.5
        )
        after_start = times >= 1
        results.append(
            {
                "case": name,
                "axis": axis,
                "first_sign": sign,
                "passed": bool(passed),
                "first_failure_seconds": float(times[np.flatnonzero(bad)[0]])
                if bad.any()
                else None,
                "windows": windows,
                "whole_path_position_rms_mm": float(np.sqrt(np.mean(error**2))),
                "after_first_second_position_rms_mm": float(
                    np.sqrt(np.mean(error[after_start] ** 2))
                ),
                "final_origin_error_mm": float(np.linalg.norm(position[-1] - origins[i]) * 10),
                "minimum_upright": float(arrays["upright"][:, i].min()),
                "maximum_forbidden_bodyweights": float(arrays["forbidden"][:, i].max()),
            }
        )
    return results


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = FlyBatch(
        7,
        7,
        actor.sensor_extension_size,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
    )
    if parent["physical_contract"] != physical_contract(env.model):
        raise ValueError("Learned round trips must match parent physical contract")
    tasks = HoverOnlyTasks(env, 120901)
    origins = tasks.start.copy()
    routes, amplitudes = np.arange(7), np.full(7, 0.15)
    memory = actor.initial_state(7)
    args.output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    synchronize(device)
    setup = time.perf_counter() - started
    start = time.perf_counter()
    rows = []
    for step in range(6000):
        t = step * env.control_dt
        target = batched_targets(np.full(7, t), origins, routes, amplitudes)
        env.requested_xy_cm[:] = target[:, :2]
        env.requested_height_cm[:] = target[:, 2]
        output = actor(torch.as_tensor(env.observation(), device=device), memory)
        memory = output.state
        action = output.action.cpu().numpy()
        row = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
        row.update(time=t, target=target, action=action.copy())
        env.step(action)
        row.update(
            post_target=batched_targets(
                np.full(7, t + env.control_dt), origins, routes, amplitudes
            ),
            actual_position=env.fields["qpos"][:, :3].copy(),
            actual_velocity=env.fields["qvel"][:, :3].copy(),
            upright=env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
            forbidden=env.forbidden_peak.copy() / env.body_weight,
        )
        rows.append(row)
    synchronize(device)
    elapsed = time.perf_counter() - start
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(args.output / "capture.npz", **arrays)
    cases = metrics(arrays, origins)
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "mode": "learned MaleCNS actor; deterministic mean, no exploration, no resets",
        "checkpoint_sha256": sha256(args.checkpoint),
        "actor_training_method": parent.get("method"),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "worlds": 7,
        "physical_transitions": 42000,
        "simulated_seconds_per_world": 12,
        "physics_hz": 1000,
        "control_hz": 500,
        "setup_seconds": setup,
        "capture_seconds": elapsed,
        "total_wall_seconds": time.perf_counter() - started,
        "origins_cm": origins.tolist(),
        "amplitude_cm": 0.15,
        "cases": cases,
        "passed": all(c["passed"] for c in cases),
        "actor_has_pid_or_external_oscillator": False,
        "future_waypoints_or_route_ids_in_observation": False,
        "gate": "Each waypoint window peak error <0.5 mm; final 100 ms displacement drift peak <1.5 mm/s; no physical failure. Full 12 s retained, including failures.",
        "scope": "Fixed development paths. One shared actor. Targets only; no body pose writes or helper forces. Raw velocity retained separately from measurement-only sustained drift. Does not overwrite the earlier PID raw-speed gate.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
