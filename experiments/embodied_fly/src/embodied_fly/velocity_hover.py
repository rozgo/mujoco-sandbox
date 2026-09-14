"""Physical hover objective and matched captures for the 391-input velocity actor.

All coordinates retain FlyBody's centimetre convention. Filtering is used only
to score velocity over three wingbeats; the force law still runs every 1 ms.
"""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment, post_row, pre_row
from embodied_fly.velocity_exercise import rolling_velocity
from embodied_fly.velocity_motor import VelocityPID, observation

EVALUATION_EPISODES = (0, 1, 8, 9)
RECIPE = {
    "version": "velocity_hover_v1",
    "control_dt": 0.002,
    "velocity_window_seconds": 0.1,
    "alive_rate": 1.0,
    "vertical_tracking_rate": 2.0,
    "horizontal_tracking_rate": 1.0,
    "angular_tracking_rate": 0.5,
    "upright_rate": 0.5,
    "velocity_scale_cm_s": 0.5,
    "angular_scale_rad_s": 0.5,
    "tracking_shape": "1 / (1 + squared normalized error)",
    "failure_penalty_once": 2.0,
    "failure_height_below_cm": 0.5,
    "failure_upright_below": 0.85,
    "failure_forbidden_body_weights_above": 0.1,
    "timing": "sum of reward rates times dt, then one failure penalty at termination",
    "height_or_position_target": False,
    "wing_speed_or_action_change_penalty": False,
    "minimum_live_rate": 1.0,
    "maximum_live_rate": 5.0,
}


def reward_rates(velocity, angular, upright, horizontal_scale=0.5):
    if not np.isfinite(horizontal_scale) or horizontal_scale <= 0:
        raise ValueError("Positive finite horizontal reward scale required")
    return {
        "alive": np.ones(len(velocity)),
        "vertical": 2 / (1 + (velocity[:, 2] / 0.5) ** 2),
        "horizontal": 1 / (1 + np.sum((velocity[:, :2] / horizontal_scale) ** 2, axis=1)),
        "angular": 0.5 / (1 + np.sum((angular / 0.5) ** 2, axis=1)),
        "upright": 0.5 * np.clip(upright, 0, 1),
    }


def failures(env):
    return (
        (env.fields["qpos"][:, 2] < 0.5)
        | (env.fields["xmat"][:, env.template.thorax_id, 8] < 0.85)
        | (env.forbidden_peak / env.body_weight > 0.1)
    )


class HoverReward:
    def __init__(self, env, horizontal_scale=0.5):
        self.env = env
        self.horizontal_scale = horizontal_scale
        self.recipe = RECIPE | {"horizontal_velocity_scale_cm_s": horizontal_scale}
        self.history = np.zeros((50, env.n, 6))
        self.total = np.zeros((env.n, 6))
        self.count = np.zeros(env.n, dtype=int)
        self.index = 0

    def reset(self, ids):
        self.history[:, ids] = 0
        self.total[ids] = 0
        self.count[ids] = 0

    def mean(self):
        return self.total / np.maximum(self.count, 1)[:, None]

    def __call__(self):
        env = self.env
        sample = np.column_stack((env.fields["qvel"][:, :3], env.velocity()[:, :3]))
        self.total += sample - self.history[self.index]
        self.history[self.index] = sample
        self.index = (self.index + 1) % 50
        self.count = np.minimum(self.count + 1, 50)
        mean = self.mean()
        terms = reward_rates(
            mean[:, :3],
            mean[:, 3:],
            env.fields["xmat"][:, env.template.thorax_id, 8],
            self.horizontal_scale,
        )
        failed = failures(env)
        reward = sum(terms.values()) * env.control_dt * ~failed - 2 * failed
        return reward.astype(np.float32), failed, terms


def start_states(dataset, episodes):
    records = []
    for episode in episodes:
        with np.load(dataset / f"episode_{episode:02d}.npz") as data:
            records.append({k: data[k][0].copy() for k in ("qpos", "qvel", "act", "ctrl")})
    return {k: np.stack([row[k] for row in records]) for k in records[0]}


def metrics(arrays, failure_seconds, seconds):
    stop = min(len(arrays["time"]), round((failure_seconds or seconds) * 500))
    velocity = rolling_velocity(arrays["measured_velocity"][:stop] * 10)
    angular = rolling_velocity(arrays["angular_velocity"][:stop])
    delta = (arrays["post_position"][:stop] - arrays["qpos"][0, :3]) * 10
    settled = slice(min(100, max(0, stop - 1)), stop)
    common = slice(0, min(stop, 1000))
    return {
        "first_failure_seconds": failure_seconds,
        "airborne_seconds": min(failure_seconds or seconds, seconds),
        "survived_ten_seconds": failure_seconds is None and seconds >= 10,
        "velocity_rms_mm_s": float(np.sqrt(np.mean(np.sum(velocity[settled] ** 2, axis=1)))),
        "vertical_velocity_rms_mm_s": float(np.sqrt(np.mean(velocity[settled, 2] ** 2))),
        "horizontal_velocity_rms_mm_s": float(
            np.sqrt(np.mean(np.sum(velocity[settled, :2] ** 2, axis=1)))
        ),
        "angular_rms_rad_s": float(np.sqrt(np.mean(np.sum(angular[settled] ** 2, axis=1)))),
        "peak_displacement_mm": float(np.linalg.norm(delta, axis=1).max()),
        "final_displacement_mm": delta[-1].tolist(),
        "height_span_mm": float(np.ptp(arrays["post_position"][:stop, 2]) * 10),
        "minimum_upright": float(arrays["upright"][:stop].min()),
        "common_first_two_seconds_complete": stop >= 1000,
        "first_two_seconds_velocity_rms_mm_s": float(
            np.sqrt(np.mean(np.sum(velocity[common] ** 2, axis=1)))
        ),
        "metrics_scope": "Before first failure only; early failure never counts as full hover",
    }


@torch.no_grad()
def evaluate_hover(actor, parent, dataset, output, device, seconds=10, teacher=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = environment(len(EVALUATION_EPISODES), len(EVALUATION_EPISODES))
    assert physical_contract(env.model) == parent["physical_contract"]
    mujoco.mj_saveModel(env.model, str(output / "model.mjb"))
    resets = start_states(dataset, EVALUATION_EPISODES)
    env.reset(np.arange(env.n), state=resets)
    commands = np.zeros((env.n, 4), np.float32)
    state = actor.initial_state(env.n)
    was_training = actor.training
    actor.eval()
    if teacher:
        saved_actions = np.load(dataset / "actions.npy", mmap_mode="r")
        controllers = [
            VelocityPID(env.template, saved_actions[e, 0].copy(), motion_feedforward=True)
            for e in EVALUATION_EPISODES
        ]
        # Original dataset phases; no phase is given to the neural actor.
        phases = [0, -0.35, 0.17, -0.17]
    setup_seconds = time.perf_counter() - started
    first_failure = np.full(env.n, np.nan)
    rows = []
    synchronize(device)
    begin = time.perf_counter()
    for step in range(round(seconds * 500)):
        row = pre_row(env, commands, step * 0.002, 0)
        row["angular_velocity"] = env.velocity()[:, :3].copy()
        obs = observation(env, commands)
        if teacher:
            actions = []
            for w, controller in enumerate(controllers):
                for key in resets:
                    getattr(env.template.data, key)[:] = env.fields[key][w]
                env.template.data.time = step * 0.002 + phases[w] / (2 * np.pi * 30)
                mujoco.mj_forward(env.model, env.template.data)
                actions.append(controller.act(commands[w]))
            action = np.asarray(actions, np.float32)
        else:
            result = actor(torch.as_tensor(obs, device=device), state)
            action, state = result.action.cpu().numpy(), result.state
        row["action"] = action
        env.step(action)
        row.update(post_row(env))
        rows.append(row)
        newly_failed = failures(env) & np.isnan(first_failure)
        first_failure[newly_failed] = (step + 1) * 0.002
        if (
            np.isfinite(first_failure).all()
            and (step + 1) * 0.002 >= np.max(first_failure) + 0.4
        ):
            break
    synchronize(device)
    capture_seconds = time.perf_counter() - begin
    cases = []
    for world, episode in enumerate(EVALUATION_EPISODES):
        failure = None if np.isnan(first_failure[world]) else float(first_failure[world])
        stop = len(rows) if failure is None else min(len(rows), round((failure + 0.4) * 500))
        arrays = {k: np.stack([row[k][world] for row in rows[:stop]]) for k in rows[0]}
        file = output / f"episode_{episode:02d}.npz"
        np.savez_compressed(file, **arrays)
        cases.append(
            {
                "episode": episode,
                "split": "validation" if episode >= 8 else "training",
                "file": file.name,
                "sha256": sha256(file),
                **metrics(arrays, failure, seconds),
            }
        )
    report = {
        "completed_utc": utc_now(),
        "seconds_requested": seconds,
        "controller": "PID reference" if teacher else "MaleCNS actor alone",
        "command": [0, 0, 0, 0],
        "physical_contract": parent["physical_contract"],
        "model_sha256": sha256(output / "model.mjb"),
        "cases": cases,
        "physical_resets": "Once at initialization; no rescue or resets during capture",
        "setup_seconds": setup_seconds,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    actor.train(was_training)
    return report
