"""Fixed seeded tests; no resets inside a trial, all failures retained."""

import json
from pathlib import Path

import numpy as np
import torch

from .bodies import CONTROL_DT, LIMITS, PRESETS, BodySpec
from .env import DogEnv
from .limb_loss import LOSS_BODIES
from .paired import PAIR_CASES, training_pairs
from .train import load_checkpoint
from .motion_quality import measure

CASES = {
    "healthy": (PRESETS["healthy"], "flat", None),
    "short_fl": (PRESETS["short_fl"], "flat", None),
    "short_fr": (PRESETS["short_fr"], "flat", None),
    "short_rl": (PRESETS["short_rl"], "flat", None),
    "short_rr": (PRESETS["short_rr"], "flat", None),
    "unseen_short": (BodySpec("unseen_short", (0.58, 1, 1, 1)), "flat", None),
    "unseen_pair": (BodySpec("unseen_pair", (0.75, 1, 1, 0.65)), "flat", None),
    "unseen_weak": (PRESETS["healthy"], "flat", (3.0, 4, 0.25)),
    "short_steps": (PRESETS["short_fl"], "steps", None),
    "unseen_steps": (BodySpec("unseen_short", (0.82, 1, 1, 1)), "test_steps", None),
    "missing_calf": (PRESETS["missing_fl"], "flat", None),
}
CASES.update(PAIR_CASES)
CASES.update({body.name: (body, "flat", None) for body in training_pairs("mild")})
CASES["weak_fr_thigh_60"] = (PRESETS["healthy"], "flat", (3.0, 4, 0.60))
CASES.update({b.name: (b, "flat", None) for b in LOSS_BODIES})


def lane_command(env, speed=0.55):
    """Declared high-level lane follower using ideal pose localization.

    This supplies a body-frame velocity command to the learned locomotion policy;
    it is not learned navigation. It applies no forces or pose corrections.
    """
    g = env.groups[0]
    w, x, y, z = g.qpos[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    world_y = np.clip(-0.7 * env.pos[:, 1], -0.25, 0.25)
    env.commands[:, 0] = speed * np.cos(yaw) + world_y * np.sin(yaw)
    env.commands[:, 1] = -speed * np.sin(yaw) + world_y * np.cos(yaw)
    env.commands[:, 2] = np.clip(-1.5 * yaw, -0.5, 0.5)


def make_case(case, trials=16, seed=9137, timestep=0.002, support_substeps=False):
    body, terrain, fault = CASES[case]
    env = DogEnv(
        trials,
        seed,
        bodies=[body],
        terrain=terrain,
        randomize=False,
        faults=False,
        threads=min(trials, 12),
        timestep=timestep,
        support_substeps=support_substeps,
    )
    # Predetermined perturbations of the initial condition, identical across policies.
    rng = np.random.default_rng(seed)
    g = env.groups[0]
    g.qpos[:, g.qadr] += rng.uniform(-0.025, 0.025, (trials, len(g.slot)))
    g.qpos[:, 1] = rng.uniform(-0.1, 0.1, trials)
    g.qvel[:] = rng.uniform(-0.025, 0.025, g.qvel.shape)
    g.batch.forward()
    env.refresh()
    env.history[:] = env.obs()[:, None, :]
    env.commands[:] = [0.55, 0, 0]
    env.start_x[:] = env.last_x[:] = env.pos[:, 0]
    if fault:
        t, j, strength = fault
        env.fault_at[:] = round(t / CONTROL_DT)
        env.fault_joint[:] = j
        env.fault_strength[:] = strength
    return env


def rollout(
    net,
    case,
    trials=16,
    seed=9137,
    seconds=12,
    capture=False,
    timestep=0.002,
    support_substeps=True,
):
    torch.set_num_threads(1)
    env = make_case(case, trials, seed, timestep, support_substeps)
    frames = []
    motion = [[] for _ in range(6)]
    alive = np.ones(trials, bool)
    fail_time = np.full(trials, np.nan)
    max_distance = np.zeros(trials)
    end_distance = np.zeros(trials)
    velocity_sse = np.zeros(trials)
    alive_steps = np.zeros(trials, int)
    peak_torque = np.zeros(trials)
    peak_ratio = np.zeros(trials)
    controlled_after_goal = np.zeros(trials, int)
    completed = np.zeros(trials, bool)
    fell_once = np.zeros(trials, bool)
    off_course_once = np.zeros(trials, bool)
    support_group = env.groups[0]
    peak_support = np.zeros((trials, len(support_group.support_names)))
    support_impulse = np.zeros_like(peak_support)
    bad_support_windows = np.zeros(trials, int)
    for k in range(round(seconds / CONTROL_DT)):
        lane_command(env)
        with torch.no_grad():
            action, _ = net(
                torch.as_tensor(env.obs()),
                torch.as_tensor(env.context),
                torch.as_tensor(env.history),
            )
        _, _, fell, _ = env.step(action.numpy())
        peak_support = np.maximum(peak_support, support_group.support_peaks)
        support_impulse += support_group.support_impulses
        bad_peak = support_group.support_peaks[:, ~support_group.support_allowed].max(1)
        bad_support_windows += bad_peak > 1.0
        off_course = np.abs(env.pos[:, 1]) > 1.5
        fell_once |= alive & fell
        off_course_once |= alive & off_course
        newly_failed = alive & (fell | off_course)
        fail_time[newly_failed] = (k + 1) * CONTROL_DT
        alive &= ~(fell | off_course)
        distance = env.pos[:, 0] - env.start_x
        max_distance = np.maximum(max_distance, np.where(alive, distance, max_distance))
        end_distance = np.where(alive, distance, end_distance)
        velocity_sse += alive * (env.vel[:, 0] - 0.55) ** 2
        alive_steps += alive
        peak_torque = np.maximum(peak_torque, np.max(np.abs(env.torque), 1))
        effective = np.maximum(LIMITS[None, :] * env.strength, 1e-8)
        peak_ratio = np.maximum(peak_ratio, np.max(np.abs(env.torque) / effective, 1))
        in_goal = (distance >= 5) & alive & (np.abs(env.pos[:, 1]) <= 1)
        controlled_after_goal = np.where(in_goal, controlled_after_goal + 1, 0)
        completed |= controlled_after_goal >= 50
        if (k + 1) * CONTROL_DT >= 1.0:
            for samples, value in zip(
                motion,
                (
                    env.q[:, support_group.slot],
                    env.action[:, support_group.slot],
                    env.gyro,
                    env.vel,
                    env.pos[:, 2],
                    env.tip_forces,
                ),
                strict=True,
            ):
                samples.append(value.copy())
        if capture:
            g = env.groups[0]
            frames.append(
                {
                    "time": (k + 1) * CONTROL_DT,
                    "qpos": g.qpos[0].copy(),
                    "qvel": g.qvel[0].copy(),
                    "action": env.action[0].copy(),
                    "torque": env.torque[0].copy(),
                    "strength": env.strength[0].copy(),
                    "context": env.context[0].copy(),
                    "alive": bool(alive[0]),
                    "completed": bool(completed[0]),
                    "support_valid": bool(bad_support_windows[0] == 0),
                    "bad_support_peak_n": float(bad_peak[0]),
                    "support_peak_forces_n": support_group.support_peaks[0].copy(),
                }
            )
    rows = []
    for i in range(trials):
        rows.append(
            {
                "trial": i,
                "survived": bool(alive[i]),
                "completed_5m": bool(completed[i]),
                "completed_with_allowed_support": bool(
                    completed[i] and not bad_support_windows[i] and alive[i]
                ),
                "bad_support_control_windows": int(bad_support_windows[i]),
                "peak_unintended_support_n": float(
                    peak_support[i, ~support_group.support_allowed].max()
                ),
                "unintended_support_impulse_ns": float(
                    support_impulse[i, ~support_group.support_allowed].sum()
                ),
                "fell": bool(fell_once[i]),
                "left_course": bool(off_course_once[i]),
                "failure_time_s": None
                if np.isnan(fail_time[i])
                else float(fail_time[i]),
                "max_distance_m": float(max_distance[i]),
                "end_distance_before_failure_m": float(end_distance[i]),
                "velocity_rmse_while_alive": float(
                    np.sqrt(velocity_sse[i] / max(alive_steps[i], 1))
                ),
                "peak_actuator_torque_nm": float(peak_torque[i]),
                "peak_torque_limit_ratio": float(peak_ratio[i]),
            }
        )
    result = {
        "case": case,
        "seed": seed,
        "seconds": seconds,
        "trials": trials,
        "survived": int(alive.sum()),
        "completed_5m": int(completed.sum()),
        "completed_with_allowed_support": int(
            (completed & (bad_support_windows == 0) & alive).sum()
        ),
        "support_checked_every_physics_step": support_substeps,
        "support_force_threshold_n": 1.0,
        "support_geom_names": support_group.support_names,
        "allowed_support_geom_names": [
            n
            for n, valid in zip(
                support_group.support_names, support_group.support_allowed
            )
            if valid
        ],
        "maximum_force_by_geom_n": dict(
            zip(support_group.support_names, peak_support.max(0).tolist())
        ),
        "mean_distance_m": float(end_distance.mean()),
        "rows": rows,
    }
    if len(motion[0]) >= 3:
        result["motion_quality"] = measure(*motion)
        result["motion_quality"]["window_s"] = [1.0, seconds]
    model = env.groups[0].model
    env.close()
    return result, frames, model


def evaluate(checkpoint, output, cases="all", trials=16, seconds=12):
    net, saved = load_checkpoint(checkpoint)
    selected = list(CASES) if cases == "all" else cases.split(",")
    results = []
    for name in selected:
        result, _, _ = rollout(net, name, trials=trials, seconds=seconds)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    report = {
        "checkpoint": str(checkpoint),
        "mode": saved["mode"],
        "navigation": "scripted velocity commands from ideal pose localization",
        "evaluation_split": "development; separate final cases required after tuning",
        "cumulative_training_seconds": saved["cumulative_training_seconds"],
        "cases": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
