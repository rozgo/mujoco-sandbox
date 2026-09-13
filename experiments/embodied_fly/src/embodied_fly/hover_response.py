"""Frozen, phase-matched physical perturbations of the learned hover actor.

Branch initialization deliberately changes height or vertical velocity once.
After initialization, every branch advances through bounded actor commands and
the unchanged physics. No PID, optimizer, force override or continuing reset.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def first_sustained(values, threshold, dt, *, post_step=False):
    above = np.asarray(values) > threshold
    for i in range(len(above) - 2):
        if above[i : i + 3].all():
            return float((i + int(post_step)) * dt * 1000)
    return None


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    provenance, started = evidence(), time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    if not actor.motor_only:
        raise ValueError("The response probe requires a motor-only actor")

    def environment(n):
        e = FlyBatch(
            n,
            min(n, 16),
            actor.sensor_extension_size,
            preset="wing_position",
            wing_response="instant",
            physics_hz=1000,
        )
        assert physical_contract(e.model) == parent["physical_contract"]
        HoverOnlyTasks(e, 120301)
        return e

    warm = environment(1)
    calibration = np.load(args.calibration)
    measured = calibration["qpos"][
        calibration["time"] >= 6, warm.template.wing_angle_indices[0]
    ]
    center, amplitude = (measured.max() + measured.min()) / 2, np.ptp(measured) / 2
    if amplitude <= 0:
        raise ValueError("Phase calibration needs a moving wing")
    phase_targets = np.arange(4) * np.pi / 2
    snapshots = {}
    memory = actor.initial_state(1)
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for step in range(3500):
        angle = warm.fields["qpos"][0, warm.template.wing_angle_indices[0]]
        speed = warm.fields["qvel"][0, warm.template.wing_velocity_indices[0]]
        phase = np.arctan2(
            (angle - center) / amplitude, speed / (amplitude * 2 * np.pi * args.phase_hz)
        ) % (2 * np.pi)
        for i, target in enumerate(phase_targets):
            error = np.angle(np.exp(1j * (phase - target)))
            if step >= 3000 and i not in snapshots and abs(error) < 0.15:
                snapshots[i] = {
                    "physical": {
                        k: warm.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")
                    },
                    "memory": memory.clone(),
                    "previous_action": warm.previous_action.copy(),
                    "time": step * warm.control_dt,
                    "phase_rad": float(phase),
                    "target_xy": warm.requested_xy_cm.copy(),
                    "target_height": warm.requested_height_cm.copy(),
                }
        if len(snapshots) == 4:
            break
        result = actor(torch.as_tensor(warm.observation(), device=device), memory)
        memory = result.state
        warm.step(result.action.cpu().numpy())
    if len(snapshots) != 4:
        raise RuntimeError(f"Found only {len(snapshots)} wing phases; no probe was run")
    synchronize(device)
    warmup_seconds = time.perf_counter() - started
    warmup_transitions = step
    del warm, memory

    env = environment(20)
    memory = actor.initial_state(20)
    labels = ("control", "higher_0.5mm", "lower_0.5mm", "rising_10mm_s", "falling_10mm_s")
    for phase, snap in sorted(snapshots.items()):
        ids = np.arange(phase * 5, phase * 5 + 5)
        physical = {k: np.repeat(v, 5, axis=0) for k, v in snap["physical"].items()}
        physical["qpos"][1, 2] += 0.05
        physical["qpos"][2, 2] -= 0.05
        physical["qvel"][3, 2] += 1
        physical["qvel"][4, 2] -= 1
        env.reset(ids, state=physical)
        env.previous_action[ids] = snap["previous_action"]
        env.requested_xy_cm[ids] = snap["target_xy"]
        env.requested_height_cm[ids] = snap["target_height"]
        memory[:, ids] = snap["memory"]
    # All worlds, including each control, receive the same reset/forward procedure.
    rows = []
    started = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        obs = env.observation()
        result = actor(torch.as_tensor(obs, device=device), memory)
        memory = result.state
        action = result.action.cpu().numpy()
        env.step(action)
        rows.append(
            {
                "action": action.copy(),
                "position_cm": env.fields["qpos"][:, :3].copy(),
                "velocity_cm_s": env.fields["qvel"][:, :3].copy(),
                "lift_bodyweights": env.wing_forces.lift.copy() / env.body_weight,
                "forbidden_bodyweights": env.forbidden_peak.copy() / env.body_weight,
            }
        )
    synchronize(device)
    capture_seconds = time.perf_counter() - started
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(args.output / "capture.npz", **arrays)
    results = []
    for phase in range(4):
        base = 5 * phase
        for j in range(1, 5):
            wing_delta = np.sqrt(
                np.mean(
                    (arrays["action"][:, base + j, 14:20] - arrays["action"][:, base, 14:20])
                    ** 2,
                    axis=1,
                )
            )
            lift_delta = (
                arrays["lift_bodyweights"][:, base + j] - arrays["lift_bodyweights"][:, base]
            )
            desired_sign = -1 if j in (1, 3) else 1
            windows = {}
            for ms in (20, 50, 100, 200):
                n = min(len(rows), round(ms / 1000 / env.control_dt))
                windows[str(ms)] = {
                    "mean_lift_delta_bodyweights": float(lift_delta[:n].mean()),
                    "corrective_sign": bool(desired_sign * lift_delta[:n].mean() > 0),
                }
            results.append(
                {
                    "phase_degrees": phase * 90,
                    "perturbation": labels[j],
                    "first_wing_command_change_ms": first_sustained(
                        wing_delta, 1e-5, env.control_dt
                    ),
                    "first_lift_change_ms": first_sustained(
                        abs(lift_delta), 0.005, env.control_dt, post_step=True
                    ),
                    "first_corrective_lift_ms": first_sustained(
                        desired_sign * lift_delta, 0.005, env.control_dt, post_step=True
                    ),
                    "windows_ms": windows,
                    "final_height_delta_mm": float(
                        (
                            arrays["position_cm"][-1, base + j, 2]
                            - arrays["position_cm"][-1, base, 2]
                        )
                        * 10
                    ),
                }
            )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "calibration_sha256": sha256(args.calibration),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "physical_contract": parent["physical_contract"],
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "capture_seconds": capture_seconds,
        "warmup_transitions": warmup_transitions,
        "probe_transitions": len(rows) * env.n,
        "optimizer_updates": 0,
        "phase_source": "Measured sweep angle and speed; diagnostic selection only, never actor input",
        "phase_snapshots": [
            {k: s[k] for k in ("time", "phase_rad")} for _, s in sorted(snapshots.items())
        ],
        "initialization": "Five branches per phase share neural/physical state except one explicit perturbation; all are reset once",
        "latency_thresholds": "Three consecutive controls: wing action RMS >1e-5, lift difference >.005 bodyweights; 2 ms resolution",
        "lift_scope": "Last physics tick per action, wing-generated scalar lift before passive body drag; not cycle-averaged or net vertical force",
        "limitation": "Four phases of one nominal trajectory; short local feedback test, not global control bandwidth or robustness",
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--phase-hz", type=float, default=9.25)
    p.add_argument("--seconds", type=float, default=0.3)
    run(p.parse_args())
