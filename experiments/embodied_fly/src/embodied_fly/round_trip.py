"""Closed target trips on the accepted wing-driven plant; PID reference only."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.pid_hover import HoverPID, PIDConfig
from embodied_fly.provenance import evidence, sha256, utc_now

CASES = (
    ("left_right", 1, 1),
    ("right_left", 1, -1),
    ("forward_backward", 0, 1),
    ("backward_forward", 0, -1),
    ("up_down", 2, 1),
    ("down_up", 2, -1),
    ("hover", 0, 0),
)


def target_at(seconds, origin, axis, sign, amplitude_cm=0.15):
    """Start -> signed target -> opposite target -> start, with smooth ramps.

    Coordinates are world axes at the canonical heading: +x forward, +y left,
    +z up. This sets a requested position, never a body pose or motor action.
    """
    if axis not in (0, 1, 2) or sign not in (-1, 0, 1):
        raise ValueError("Valid world axis and direction required")
    if not np.isfinite([seconds, amplitude_cm]).all() or min(seconds, amplitude_cm) < 0:
        raise ValueError("Finite nonnegative time and amplitude required")
    knots = (0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 12.0)
    values = (0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0)
    i = min(int(np.searchsorted(knots, seconds, side="right")) - 1, len(knots) - 2)
    u = np.clip((seconds - knots[i]) / (knots[i + 1] - knots[i]), 0, 1)
    smooth = u**3 * (10 - 15 * u + 6 * u * u)
    target = np.asarray(origin, dtype=np.float64).copy()
    target[axis] += sign * amplitude_cm * (values[i] + (values[i + 1] - values[i]) * smooth)
    return target


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    provenance = evidence()
    env = FlyBatch(7, 7, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    expected = json.loads(args.contract_report.read_text())
    assert physical_contract(env.model) == expected["physical_contract"]
    tasks = HoverOnlyTasks(env, 120801)
    origins = tasks.start.copy()
    controllers = [
        HoverPID(env.template, tasks.air_action, origin, PIDConfig()) for origin in origins
    ]
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    assert sha256(args.output / "model.mjb") == expected["model_sha256"]
    setup = time.perf_counter() - total_started
    start = time.perf_counter()
    rows = []
    failed = np.zeros(7, dtype=bool)
    first_failure = np.full(7, np.nan)
    for step in range(6000):
        t = step * env.control_dt
        targets = np.stack(
            [
                target_at(t, origins[i], axis, sign, args.amplitude)
                for i, (_, axis, sign) in enumerate(CASES)
            ]
        )
        env.requested_xy_cm[:] = targets[:, :2]
        env.requested_height_cm[:] = targets[:, 2]
        actions = []
        for i, c in enumerate(controllers):
            # Controller scratch state only: live batch state is never assigned.
            for key in ("qpos", "qvel", "act", "ctrl"):
                getattr(env.template.data, key)[:] = env.fields[key][i]
            env.template.data.time = t
            mujoco.mj_forward(env.template.model, env.template.data)
            c.target[:] = targets[i]
            actions.append(c.act())
        actions = np.stack(actions)
        row = {key: env.fields[key].copy() for key in ("qpos", "qvel", "act", "ctrl")}
        row.update(time=t, target=targets.copy(), action=actions.copy())
        env.step(actions)
        up = env.fields["xmat"][:, env.template.thorax_id, 8]
        bad = (
            (env.fields["qpos"][:, 2] < 0.5)
            | (up < 0.5)
            | (env.forbidden_peak / env.body_weight > 0.1)
        )
        first_failure[bad & ~failed] = (step + 1) * env.control_dt
        failed |= bad
        row.update(
            actual_position=env.fields["qpos"][:, :3].copy(),
            actual_velocity=env.fields["qvel"][:, :3].copy(),
            upright=up.copy(),
            forbidden=env.forbidden_peak.copy() / env.body_weight,
        )
        rows.append(row)
    capture_seconds = time.perf_counter() - start
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(args.output / "capture.npz", **arrays)
    cases = []
    for i, (name, axis, sign) in enumerate(CASES):
        errors = (
            np.linalg.norm(arrays["actual_position"][:, i] - arrays["target"][:, i], axis=1)
            * 10
        )
        windows = {}
        for label, a, b in (
            ("first_target", 3.5, 4),
            ("opposite_target", 6.5, 7),
            ("returned_hold", 11, 12),
        ):
            mask = (arrays["time"] >= a) & (arrays["time"] < b)
            windows[label] = {
                "position_peak_mm": float(errors[mask].max()),
                "position_rms_mm": float(np.sqrt(np.mean(errors[mask] ** 2))),
                "speed_rms_mm_s": float(
                    np.sqrt(np.mean(np.sum(arrays["actual_velocity"][mask, i] ** 2, axis=1)))
                    * 10
                ),
            }
        passed = (
            not failed[i]
            and all(w["position_peak_mm"] < 0.5 for w in windows.values())
            and windows["returned_hold"]["speed_rms_mm_s"] < 1.5
        )
        cases.append(
            {
                "case": name,
                "axis": axis,
                "first_sign": sign,
                "passed": bool(passed),
                "first_failure_seconds": float(first_failure[i]) if failed[i] else None,
                "windows": windows,
                "final_offset_from_origin_mm": (
                    (arrays["actual_position"][-1, i] - origins[i]) * 10
                ).tolist(),
                "minimum_upright": float(arrays["upright"][:, i].min()),
                "maximum_forbidden_bodyweights": float(arrays["forbidden"][:, i].max()),
            }
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "mode": "PID reference only; no learned actor",
        "body_force_or_pose_control": False,
        "pid": asdict(PIDConfig()),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "amplitude_cm": args.amplitude,
        "simulated_seconds_per_world": 12,
        "worlds": 7,
        "physical_transitions": 42000,
        "physics_hz": 1000,
        "control_hz": 500,
        "setup_seconds": setup,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - total_started,
        "actor_updates": 0,
        "origins_cm": origins.tolist(),
        "cases": cases,
        "passed": all(c["passed"] for c in cases),
        "scope": "Plant control-authority check for a proposed same-actor round-trip curriculum. Intermediate targets must be reached, then return to the original position at low speed. No fly brain, PID teaching, live reset or body-force helper.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--amplitude", type=float, default=0.15)
    p.add_argument(
        "--contract-report",
        type=Path,
        default=Path("docs/embodied_fly/runs/hover_explore_01/training.json"),
    )
    result = run(p.parse_args())
    raise SystemExit(0 if result["passed"] else 2)
