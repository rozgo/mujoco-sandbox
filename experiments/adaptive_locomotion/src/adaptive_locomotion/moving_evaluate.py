"""Independent full-window balance acceptance; failed trials never auto-reset."""

import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from .bodies import CONTROL_DT, LEGS, LIMITS, PRESETS
from .limb_loss import LOSS_BODIES
from .moving_env import MOTIONS, MovingEnv
from .train import load_checkpoint

BODY_MAP = {b.name: b for b in (PRESETS["healthy"], *LOSS_BODIES)}


def run_case(
    net,
    body="healthy",
    surface="combined",
    trials=4,
    seconds=10,
    seed=9301,
    physics_backend="mjbatch",
    transition=False,
    capture=False,
    support_substeps=True,
    timestep=0.002,
    capture_trial=0,
    support_friction=None,
    relative=True,
    motion_scale=1.0,
):
    if trials < 1 or seconds <= 2 or (transition and seconds < 10):
        raise ValueError(
            "Acceptance needs positive trials, >2 s standing or >=10 s transitions"
        )
    if capture and not 0 <= capture_trial < trials:
        raise ValueError("Captured trial must be a valid batch index")
    torch.set_num_threads(1)
    env = MovingEnv(
        trials,
        seed,
        cases=[(BODY_MAP[body], "moving")],
        motion=surface,
        relative=relative,
        motion_scale=motion_scale,
        substep_support=True,
        schedule=False,
        randomize=True,
        threads=4,
        physics_backend=physics_backend,
        support_substeps=support_substeps,
        timestep=timestep,
        support_friction=support_friction,
    )
    g = env.groups[0]
    net = net.cpu().eval()
    alive = np.ones(trials, bool)
    bad_peak = np.zeros(trials)
    torque_ratio = np.zeros(trials)
    fail_time = np.full(trials, np.nan)
    speed, drift, tilt, air, supported = [], [], [], [], []
    deck_error, deck_torque, world_speed, slips = [], [], [], []
    frames = []
    resume_x = None
    pen = np.zeros(trials)
    warnings = np.zeros(trials, dtype=int)
    scratch = mujoco.MjData(g.model)
    start = time.perf_counter()
    for k in range(round(seconds / CONTROL_DT)):
        t = k * CONTROL_DT
        walking = transition and (t < 3 or t >= 8)
        env.set_commands([0.55 if walking else 0, 0, 0])
        if transition and k == 400:
            resume_x = env.pos[:, 0].copy()
        obs = env.obs()
        with torch.no_grad():
            action, _ = net(
                torch.as_tensor(obs), support=torch.as_tensor(env.support_obs())
            )
        _, _, fell, info = env.step(action.numpy())
        deck_error.append(np.abs(g.qpos[:, g.deck_qadr] - info["deck_target"]))
        deck_torque.append(
            np.abs(g.torque[:, g.deck_aadr])
            / g.model.actuator_forcerange[g.deck_aadr, 1]
        )
        warnings = np.maximum(warnings, g.warnings[:, :, 0].sum(1))
        fail_time[alive & fell] = (k + 1) * CONTROL_DT
        alive &= ~fell
        bad_peak = np.maximum(bad_peak, g.support_peaks[:, ~g.support_allowed].max(1))
        torque_ratio = np.maximum(torque_ratio, (np.abs(env.torque) / LIMITS).max(1))
        post_settle = (5 <= t < 8) if transition else t >= 2
        contacts = np.zeros((trials, 4), bool)
        contacts[:, g.tip_slots] = g.support_peaks[:, g.tip_sensors] > 1
        if post_settle:
            speed.append(info["relative_speed_m_s"])
            world_speed.append(np.linalg.norm(env.vel[:, :2], axis=1))
            slips.append(info["relative_slip_m_s"] * (env.tip_forces > 5))
            drift.append(info["relative_drift_m"])
            tilt.append(np.rad2deg(np.arccos(np.clip(env.up[:, 2], -1, 1))))
            supported.append(contacts.sum(1) >= 2)
            if surface.startswith("gap_"):
                air.append(~contacts[:, LEGS.index(surface[-2:].upper())])
        # Independent CPU forward detects actual geometry penetration, even for
        # GPU rollouts. This diagnostic is sampled at 50 Hz; support is 500 Hz.
        step_pen = np.zeros(trials)
        for i in range(trials):
            scratch.qpos[:] = g.qpos[i]
            scratch.qvel[:] = g.qvel[i]
            mujoco.mj_forward(g.model, scratch)
            if scratch.ncon:
                step_pen[i] = max(0, -float(scratch.contact.dist.min()))
                pen[i] = max(pen[i], step_pen[i])
        if capture:
            i = capture_trial
            frames.append(
                {
                    "time": (k + 1) * CONTROL_DT,
                    "qpos": g.qpos[i].copy(),
                    "qvel": g.qvel[i].copy(),
                    "action": env.action[i].copy(),
                    "ctrl": g.ctrl[i].copy(),
                    "torque": env.torque[i].copy(),
                    "commands": env.commands[i].copy(),
                    "tip_positions": env.tip_positions[i].copy(),
                    "tip_forces": env.tip_forces[i].copy(),
                    "support_peak_forces_n": g.support_peaks[i].copy(),
                    "hold_anchor": env.hold_anchor[i].copy(),
                    "relative_anchor": env.local_anchor[i].copy(),
                    "deck_origin": env.deck_origin[i].copy(),
                    "deck_rotation": env.deck_rot[i].copy(),
                    "deck_linear": env.deck_linear[i].copy(),
                    "deck_angular": env.deck_angular[i].copy(),
                    "relative_drift_m": info["relative_drift_m"][i],
                    "relative_speed_m_s": info["relative_speed_m_s"][i],
                    "deck_actuator_force": g.torque[i, g.deck_aadr].copy(),
                    "alive": bool(alive[i]),
                    "support_valid": bool(bad_peak[i] <= 1),
                    "sampled_penetration_m": step_pen[i],
                }
            )
    mean_speed = np.mean(speed, axis=0)
    max_drift, max_tilt = np.max(drift, axis=0), np.max(tilt, axis=0)
    air_fraction = np.mean(air, axis=0) if air else np.ones(trials)
    support_fraction = np.mean(supported, axis=0)
    resumed = env.pos[:, 0] - resume_x if resume_x is not None else None
    passed = (
        alive
        & (bad_peak <= 1)
        & (mean_speed <= 0.06)
        & (max_drift <= 0.15)
        & (max_tilt <= 20)
        & (torque_ratio <= 1.0001)
        & (pen <= 0.008)
        & (air_fraction >= 0.95)
        & (support_fraction >= 0.95)
        & (warnings == 0)
    )
    if resumed is not None:
        passed &= resumed >= 1
    rows = [
        {
            "trial": i,
            "cpu_warning_count": int(warnings[i])
            if physics_backend == "mjbatch"
            else None,
            "mean_world_speed_mps": float(np.mean(world_speed, axis=0)[i]),
            "peak_deck_force_limit_ratio": float(np.max(deck_torque, axis=(0, 2))[i]),
            "max_deck_joint_tracking_error": np.max(deck_error, axis=0)[i].tolist(),
            "mean_loaded_tip_slip_mps": float(np.mean(slips, axis=(0, 2))[i]),
            "passed": bool(passed[i]),
            "survived": bool(alive[i]),
            "failure_time_s": None if np.isnan(fail_time[i]) else float(fail_time[i]),
            "mean_idle_speed_mps": float(mean_speed[i]),
            "max_idle_drift_m": float(max_drift[i]),
            "max_idle_tilt_deg": float(max_tilt[i]),
            "peak_unintended_support_n": float(bad_peak[i]),
            "peak_torque_limit_ratio": float(torque_ratio[i]),
            "max_penetration_m": float(pen[i]),
            "unsupported_foot_fraction": float(air_fraction[i]) if air else None,
            "at_least_two_contacts_fraction": float(support_fraction[i]),
            "resumed_distance_m": None if resumed is None else float(resumed[i]),
        }
        for i in range(trials)
    ]
    result = {
        "body": body,
        "motion": surface,
        "support_relative_observations": relative,
        "motion_scale": motion_scale,
        "transition": transition,
        "seed": seed,
        "seconds": seconds,
        "trials": trials,
        "passed": int(passed.sum()),
        "survived": int(alive.sum()),
        "rows": rows,
        "support_checked_every_physics_step": support_substeps,
        "physics_timestep_s": timestep,
        "physics_backend": physics_backend,
        "penetration_sample_hz": 50,
        "numerical_checks": "CPU warnings and finite state"
        if physics_backend == "mjbatch"
        else "Warp finite state, velocity envelope and capacity overflow (not CPU warning counters)",
        "metric_window_s": [5, 8] if transition else [2, seconds],
        "execution_seconds": time.perf_counter() - start,
    }
    model = g.model
    if support_friction is not None:
        result["support_sliding_friction"] = support_friction
    env.close()
    return result, frames, model


def evaluate(
    checkpoint,
    output,
    motions="all",
    trials=4,
    seconds=10,
    seed=9401,
    physics_backend="mjbatch",
    relative=True,
    body="healthy",
    motion_scale=1.0,
):
    net, _ = load_checkpoint(checkpoint)
    selected = list(MOTIONS) if motions == "all" else motions.split(",")
    results = []
    for motion in selected:
        result, _, _ = run_case(
            net,
            body=body,
            surface=motion,
            trials=trials,
            seconds=seconds,
            seed=seed,
            physics_backend=physics_backend,
            relative=relative,
            motion_scale=motion_scale,
        )
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "seed": seed,
        "passed": sum(r["passed"] for r in results),
        "survived": sum(r["survived"] for r in results),
        "trials": sum(r["trials"] for r in results),
        "cases": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
