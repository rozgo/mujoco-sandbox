"""One continuous full-flight velocity exercise on the accepted physical plant."""

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
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_motor import (
    VelocityPID,
    measured_velocity,
    observation,
    schema_report,
)

# Every episode contains every movement. This is a command schedule, not a
# physical pose trajectory. Nothing is reset between its stages.
MOVES = (
    ("FORWARD", (1, 0, 0, 0)),
    ("BACKWARD", (-1, 0, 0, 0)),
    ("LEFT", (0, 1, 0, 0)),
    ("RIGHT", (0, -1, 0, 0)),
    ("UP", (0, 0, 1, 0)),
    ("DOWN", (0, 0, -1, 0)),
)
STAGES = [("HOVER", 2.0, (0, 0, 0, 0))]
for label, command in MOVES:
    STAGES.extend(((label + " / NO TURN", 2.0, command), ("BRAKE / HOVER", 1.0, (0, 0, 0, 0))))
for label, sign in (("TURN LEFT IN PLACE", 1), ("TURN RIGHT IN PLACE", -1)):
    STAGES.extend(((label, 2.0, (0, 0, 0, sign)), ("BRAKE / HOVER", 1.0, (0, 0, 0, 0))))
for label, command in MOVES:
    for turn, sign in (("TURN LEFT", 1), ("TURN RIGHT", -1)):
        STAGES.extend(
            (
                (label + " + " + turn, 1.8, (*command[:3], sign)),
                ("BRAKE / HOVER", 0.8, (0, 0, 0, 0)),
            )
        )
for label, command in (
    ("DIAGONAL FORWARD + LEFT", (1, 1, 0, 0)),
    ("DIAGONAL BACK + RIGHT", (-1, -1, 0, 0)),
    ("CLIMBING DIAGONAL + TURN", (1, 1, 1, 1)),
    ("DESCENDING DIAGONAL + TURN", (-1, -1, -1, -1)),
):
    STAGES.extend(((label, 2.0, command), ("BRAKE / HOVER", 1.0, (0, 0, 0, 0))))
STAGES.append(("FINAL HOVER", 2.0, (0, 0, 0, 0)))
STAGES = tuple(STAGES)
DURATION = sum(stage[1] for stage in STAGES)
COMMAND_CM_S = 0.15  # 1.5 mm/s on each commanded translation axis.
YAW_RAD_S = 0.45  # 25.8 degrees/s.
RAMP_SECONDS = 0.25


def command_at(seconds, speed=COMMAND_CM_S, yaw_speed=YAW_RAD_S):
    if (
        not np.isfinite([seconds, speed, yaw_speed]).all()
        or min(seconds, speed, yaw_speed) < 0
    ):
        raise ValueError("Finite nonnegative time/speed required")
    ends = np.cumsum([stage[1] for stage in STAGES])
    index = min(int(np.searchsorted(ends, seconds, side="right")), len(STAGES) - 1)
    start = 0 if index == 0 else ends[index - 1]
    previous = np.asarray(STAGES[max(0, index - 1)][2], dtype=float) * (
        speed,
        speed,
        speed,
        yaw_speed,
    )
    target = np.asarray(STAGES[index][2], dtype=float) * (speed, speed, speed, yaw_speed)
    u = np.clip((seconds - start) / RAMP_SECONDS, 0, 1)
    blend = u**3 * (10 - 15 * u + 6 * u * u)
    return previous + (target - previous) * blend, index


def rolling_velocity(velocities, width=50):
    """Observer-only 100 ms mean; raw 500 Hz velocity is retained separately."""
    count = np.arange(1, len(velocities) + 1)
    sums = np.vstack((np.zeros((1, velocities.shape[1])), np.cumsum(velocities, axis=0)))
    starts = np.maximum(0, count - width)
    return (sums[count] - sums[starts]) / (count - starts)[:, None]


def metrics(arrays, speed_scale=1.0):
    measured = arrays["measured_velocity"] * 10
    requested = arrays["command"][:, :3] * 10
    yaw_mean = rolling_velocity(arrays["yaw_rate"][:, None])[:, 0]
    mean = rolling_velocity(measured)
    results = []
    ends = np.cumsum([stage[1] for stage in STAGES])
    for i, (name, duration, command) in enumerate(STAGES):
        end = ends[i]
        # Settled performance is predeclared, not selected after seeing a run.
        selected = (arrays["time"] >= end - 0.4) & (arrays["time"] < end)
        error = mean[selected] - requested[selected]
        peak = float(np.linalg.norm(error, axis=1).max())
        # Predeclared faster-motion gates: 10% command magnitude, with the
        # original absolute floors. Braking/hover retain the slow gates.
        velocity_limit = max(
            0.5, COMMAND_CM_S * 10 * speed_scale * 0.1 * np.linalg.norm(command[:3])
        )
        yaw_limit = max(0.12, YAW_RAD_S * speed_scale * 0.1 * abs(command[3]))
        results.append(
            {
                "index": i,
                "stage": name,
                "start_seconds": float(end - duration),
                "end_seconds": float(end),
                "settled_window_seconds": 0.4,
                "settled_mean_velocity_mm_s": mean[selected].mean(0).tolist(),
                "requested_velocity_mm_s": requested[selected][-1].tolist(),
                "settled_velocity_error_peak_mm_s": peak,
                "settled_velocity_error_rms_mm_s": float(
                    np.sqrt(np.mean(np.sum(error**2, axis=1)))
                ),
                "raw_velocity_error_rms_mm_s": float(
                    np.sqrt(
                        np.mean(
                            np.sum((measured[selected] - requested[selected]) ** 2, axis=1)
                        )
                    )
                ),
                "settled_yaw_error_peak_rad_s": float(
                    np.abs(yaw_mean[selected] - arrays["command"][selected, 3]).max()
                ),
                "heading_change_degrees": float(
                    np.degrees(
                        np.unwrap(arrays["heading"])[arrays["stage"] == i][-1]
                        - np.unwrap(arrays["heading"])[arrays["stage"] == i][0]
                    )
                ),
                "velocity_error_limit_mm_s": float(velocity_limit),
                "yaw_error_limit_rad_s": float(yaw_limit),
                "passed": bool(
                    peak < velocity_limit
                    and float(
                        np.abs(yaw_mean[selected] - arrays["command"][selected, 3]).max()
                    )
                    < yaw_limit
                ),
            }
        )
    physical = (
        arrays["upright"].min() > 0.85
        and arrays["post_position"][:, 2].min() > 0.5
        and arrays["forbidden"].max() < 0.1
    )
    return {
        "stages": results,
        "passed": bool(physical and all(stage["passed"] for stage in results)),
        "physical_envelope_passed": bool(physical),
        "minimum_upright": float(arrays["upright"].min()),
        "minimum_altitude_mm": float(arrays["post_position"][:, 2].min() * 10),
        "maximum_forbidden_contact_bodyweights": float(arrays["forbidden"].max()),
        "raw_velocity_peak_mm_s": np.abs(measured).max(0).tolist(),
        "mean_velocity_error_whole_exercise_rms_mm_s": float(
            np.sqrt(np.mean(np.sum((mean - requested) ** 2, axis=1)))
        ),
        "gate": "Each stage's final 0.4 s: 100 ms mean vector velocity error <max(0.5 mm/s, 10% requested vector speed) and yaw-rate error <max(0.12 rad/s, 10% requested yaw rate); upright >0.85; altitude >5 mm; forbidden contact <0.1 bodyweights. Raw wingbeat velocity retained; no position-return gate.",
    }


def run(args):
    if not np.isfinite(args.speed_scale) or args.speed_scale <= 0:
        raise ValueError("Positive finite physical command speed scale required")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    e = FlyBatch(
        1,
        1,
        12,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
        heading_control=True,
        fast_flight=args.fast_flight,
    )
    expected = json.loads(args.contract_report.read_text())["physical_contract"]
    base = FlyBatch(1, 1, 12, preset="wing_position", wing_response="instant", physics_hz=1000)
    if physical_contract(base.model) != expected:
        raise ValueError("Original physical plant no longer matches its accepted fingerprint")
    from embodied_fly.physical_contract import ARRAYS, OPTIONS

    for key in ARRAYS:
        np.testing.assert_array_equal(getattr(e.model, key), getattr(base.model, key))
    for key in OPTIONS:
        np.testing.assert_array_equal(getattr(e.model.opt, key), getattr(base.model.opt, key))
    del base
    tasks = HoverOnlyTasks(e, 121102)
    controller = VelocityPID(e.template, tasks.air_action, motion_feedforward=args.fast_flight)
    mujoco.mj_saveModel(e.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    begin = time.perf_counter()
    rows = []
    for step in range(round(DURATION / e.control_dt)):
        t = step * e.control_dt
        command, stage = command_at(
            t, COMMAND_CM_S * args.speed_scale, YAW_RAD_S * args.speed_scale
        )
        e.batch.forward()  # Current derived feedback, without advancing time.
        observed = observation(e, command[None])
        for key in ("qpos", "qvel", "act", "ctrl"):
            getattr(e.template.data, key)[:] = e.fields[key][0]
        e.template.data.time = t
        mujoco.mj_forward(e.template.model, e.template.data)
        action = controller.act(command)
        row = {k: e.fields[k][0].copy() for k in ("qpos", "qvel", "act", "ctrl")}
        row.update(
            time=t,
            stage=stage,
            command=command,
            measured_velocity=measured_velocity(e)[0].copy(),
            yaw_rate=float(e.velocity()[0, 2]),
            heading=float(
                np.arctan2(
                    e.fields["xmat"][0, e.template.thorax_id, 3],
                    e.fields["xmat"][0, e.template.thorax_id, 0],
                )
            ),
            observation=observed[0],
            action=action.copy(),
            pid_integral=controller.integral.copy(),
        )
        e.step(action[None])
        row.update(
            post_position=e.fields["qpos"][0, :3].copy(),
            upright=float(e.fields["xmat"][0, e.template.thorax_id, 8]),
            forbidden=float(e.forbidden_peak[0] / e.body_weight),
        )
        rows.append(row)
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    capture_seconds = time.perf_counter() - begin
    np.savez_compressed(args.output / "capture.npz", **arrays)
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "controller": "Velocity PID teacher; PI feedback with bounded wing-joint commands",
        "learned_actor": False,
        "training_updates": 0,
        "worlds": 1,
        "physical_transitions": len(rows),
        "physics_hz": 1000,
        "control_hz": 500,
        "duration_seconds": DURATION,
        "episode_resets_after_initialization": 0,
        "all_movements_in_one_continuous_episode": True,
        "physical_command_speed_scale": args.speed_scale,
        "fast_flight": args.fast_flight,
        "command_speed_mm_s": COMMAND_CM_S * 10 * args.speed_scale,
        "command_ramp_seconds": RAMP_SECONDS,
        "controller_kp": controller.kp.tolist(),
        "controller_ki": controller.ki.tolist(),
        "causal_motion_feedforward": controller.motion_feedforward,
        "wing_controller": asdict(controller.wings.config),
        "position_target": False,
        "heading_target": False,
        "yaw_command_rad_s": YAW_RAD_S * args.speed_scale,
        "body_mechanics_match_previous_plant": True,
        "force_law_change": (
            "Versioned v4: v3 measured-wing heading authority with yaw damping time constant 0.25 s instead of 0.025 s; roll/pitch damping unchanged"
            if args.fast_flight
            else "Versioned v3: independent yaw authority from measured left/right wing pitch asymmetry; previous v2 preserved"
        ),
        "wing_force_config": asdict(e.template.wing_forces.config),
        "previous_physical_contract": expected,
        "teacher_clock_hz": controller.wings.config.frequency_hz,
        "observation_schema": schema_report(),
        "physical_contract": physical_contract(e.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "setup_seconds": setup,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        **metrics(arrays, args.speed_scale),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract-report", type=Path, required=True)
    p.add_argument("--speed-scale", type=float, default=1.0)
    p.add_argument("--fast-flight", action="store_true")
    run(p.parse_args())
