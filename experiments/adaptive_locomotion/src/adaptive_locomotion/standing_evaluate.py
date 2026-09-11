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
from .standing_env import StandingEnv
from .standing_surfaces import AGGRESSIVE, GENTLE, SURFACES
from .train import load_checkpoint

BODY_MAP = {b.name: b for b in (PRESETS["healthy"], *LOSS_BODIES)}


def run_case(
    net,
    body="healthy",
    surface="flat",
    trials=4,
    seconds=10,
    seed=9301,
    physics_backend="mjbatch",
    transition=False,
    capture=False,
    support_substeps=True,
    timestep=0.002,
):
    torch.set_num_threads(1)
    env = StandingEnv(
        trials,
        seed,
        cases=[(BODY_MAP[body], surface)],
        schedule=False,
        randomize=True,
        threads=4,
        physics_backend=physics_backend,
        support_substeps=support_substeps,
        timestep=timestep,
    )
    g = env.groups[0]
    net = net.cpu().eval()
    alive = np.ones(trials, bool)
    bad_peak = np.zeros(trials)
    torque_ratio = np.zeros(trials)
    fail_time = np.full(trials, np.nan)
    speed, drift, tilt, air, supported = [], [], [], [], []
    frames = []
    resume_x = None
    pen = np.zeros(trials)
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
        _, _, fell, _ = env.step(action.numpy())
        fail_time[alive & fell] = (k + 1) * CONTROL_DT
        alive &= ~fell
        bad_peak = np.maximum(bad_peak, g.support_peaks[:, ~g.support_allowed].max(1))
        torque_ratio = np.maximum(torque_ratio, (np.abs(env.torque) / LIMITS).max(1))
        post_settle = (5 <= t < 8) if transition else t >= 2
        contacts = np.zeros((trials, 4), bool)
        contacts[:, g.tip_slots] = g.support_peaks[:, g.tip_sensors] > 1
        if post_settle:
            speed.append(np.linalg.norm(env.vel[:, :2], axis=1))
            drift.append(np.linalg.norm(env.pos[:, :2] - env.hold_anchor, axis=1))
            tilt.append(np.rad2deg(np.arccos(np.clip(env.up[:, 2], -1, 1))))
            supported.append(contacts.sum(1) >= 2)
            if surface.startswith("gap_"):
                air.append(~contacts[:, LEGS.index(surface[-2:].upper())])
        # Independent CPU forward detects actual geometry penetration, even for
        # GPU rollouts. This diagnostic is sampled at 50 Hz; support is 500 Hz.
        for i in range(trials):
            scratch.qpos[:] = g.qpos[i]
            scratch.qvel[:] = g.qvel[i]
            mujoco.mj_forward(g.model, scratch)
            if scratch.ncon:
                pen[i] = max(pen[i], -float(scratch.contact.dist.min()))
        if capture:
            frames.append(
                {
                    "time": (k + 1) * CONTROL_DT,
                    "qpos": g.qpos[0].copy(),
                    "qvel": g.qvel[0].copy(),
                    "action": env.action[0].copy(),
                    "ctrl": g.ctrl[0].copy(),
                    "torque": env.torque[0].copy(),
                    "commands": env.commands[0].copy(),
                    "tip_positions": env.tip_positions[0].copy(),
                    "tip_forces": env.tip_forces[0].copy(),
                    "support_peak_forces_n": g.support_peaks[0].copy(),
                    "hold_anchor": env.hold_anchor[0].copy(),
                    "alive": bool(alive[0]),
                    "support_valid": bool(bad_peak[0] <= 1),
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
    )
    if resumed is not None:
        passed &= resumed >= 1
    rows = [
        {
            "trial": i,
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
        "surface": surface,
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
        "metric_window_s": [5, 8] if transition else [2, seconds],
        "execution_seconds": time.perf_counter() - start,
    }
    model = g.model
    env.close()
    return result, frames, model


def evaluate(
    checkpoint,
    output,
    surfaces="gentle",
    bodies="healthy",
    trials=4,
    seconds=10,
    seed=9301,
    physics_backend="mjbatch",
    transitions=True,
):
    net, _ = load_checkpoint(checkpoint)
    selected = {"gentle": GENTLE, "aggressive": AGGRESSIVE, "all": SURFACES}.get(
        surfaces
    )
    selected = selected if selected is not None else surfaces.split(",")
    cases = [(b, s, False) for b in bodies.split(",") for s in selected]
    if transitions:
        cases += [(b, "flat", True) for b in bodies.split(",")]
    results = []
    for body, surface, transition in cases:
        r, _, _ = run_case(
            net,
            body,
            surface,
            trials,
            12 if transition else seconds,
            seed,
            physics_backend,
            transition,
        )
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}), flush=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "seed": seed,
        "passed": sum(r["passed"] for r in results),
        "trials": sum(r["trials"] for r in results),
        "thresholds": {
            "idle_speed_mps": 0.06,
            "drift_m": 0.15,
            "tilt_deg": 20,
            "bad_support_n": 1,
            "penetration_m": 0.008,
            "unsupported_foot_fraction": 0.95,
        },
        "cases": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
