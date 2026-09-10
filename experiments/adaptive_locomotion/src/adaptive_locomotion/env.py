"""Vectorized MuJoCo state, fault scheduling, sensor interface and task reward."""

import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from mjbatch import Batch

from .bodies import (
    ACTION_SCALE,
    CONTROL_DT,
    DT,
    LEGS,
    LIMITS,
    PRESETS,
    STAND,
    build_model,
    initialize,
    joint_mapping,
)

OBS_DIM, CONTEXT_DIM, HISTORY = 66, 20, 25


def rotation(q):
    w, x, y, z = q.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        1,
    ).reshape(-1, 3, 3)


class Group:
    def __init__(self, body, n, threads, terrain, sensing, timestep):
        self.body = body
        self.model = build_model(body, terrain, timestep, sensing)
        self.batch = Batch(self.model, n, num_threads=threads)
        self.n = n
        self.qpos, self.qvel, self.ctrl = (
            self.batch.bind(k) for k in ("qpos", "qvel", "ctrl")
        )
        self.torque = self.batch.bind("actuator_force")
        self.warnings = self.batch.bind("warning")
        self.slot, self.qadr, self.dadr, self.aadr = joint_mapping(self.model)
        self.gain, self.bias, self.force_range = (
            self.batch.expand(k)
            for k in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
        )
        self.original_gain, self.original_bias = self.gain.copy(), self.bias.copy()
        self.rays = (
            [self.batch.sensor(f"range_{i}") for i in range(9)] if sensing else []
        )
        data = mujoco.MjData(self.model)
        initialize(self.model, data)
        self.initial_qpos = data.qpos.copy()

    def set_strength(self, ids, strength):
        strength = strength[:, self.slot]
        cap = LIMITS[self.slot][None, :] * strength
        self.force_range[ids, :, 0] = -cap
        self.force_range[ids, :, 1] = cap
        active = (strength > 0)[:, :, None]
        self.gain[ids] = self.original_gain[ids] * active
        self.bias[ids] = self.original_bias[ids] * active


class DogEnv:
    def __init__(
        self,
        num_envs=512,
        seed=0,
        bodies=None,
        terrain="flat",
        randomize=True,
        faults=True,
        threads=12,
        timestep=DT,
        history=HISTORY,
        sensing=None,
    ):
        self.n, self.rng = num_envs, np.random.default_rng(seed)
        self.randomize, self.faults, self.terrain = randomize, faults, terrain
        self.timestep = timestep
        self.decimation = round(CONTROL_DT / timestep)
        if abs(self.decimation * timestep - CONTROL_DT) > 1e-10:
            raise ValueError("Inexact decimation")
        self.bodies = bodies or [
            PRESETS[k]
            for k in ("healthy", "short_fl", "short_fr", "short_rl", "short_rr")
        ]
        if num_envs < len(self.bodies):
            raise ValueError("At least one environment per body")
        self.groups, self.slices = [], []
        start = 0
        for i, body in enumerate(self.bodies):
            n = num_envs // len(self.bodies) + (i < num_envs % len(self.bodies))
            self.groups.append(
                Group(
                    body,
                    n,
                    max(1, threads // len(self.bodies)),
                    terrain,
                    True if sensing is None else sensing,
                    timestep,
                )
            )
            self.slices.append(slice(start, start + n))
            start += n
        self.pool = (
            ThreadPoolExecutor(max_workers=len(self.groups))
            if len(self.groups) > 1
            else None
        )
        self.q = np.zeros((num_envs, 12))
        self.dq = self.q.copy()
        self.torque = self.q.copy()
        self.pos = np.zeros((num_envs, 3))
        self.vel = self.pos.copy()
        self.gyro = self.pos.copy()
        self.up = self.pos.copy()
        self.valid = np.zeros_like(self.q)
        self.scan = np.ones((num_envs, 9), np.float32) * 2
        self.action = np.zeros((num_envs, 12), np.float32)
        self.commands = np.zeros((num_envs, 3), np.float32)
        self.strength = np.ones((num_envs, 12))
        self.context = np.ones((num_envs, CONTEXT_DIM), np.float32)
        self.steps = np.zeros(num_envs, int)
        self.fault_at = np.zeros(num_envs, int)
        self.fault_joint = np.zeros(num_envs, int)
        self.fault_strength = np.ones(num_envs)
        self.history = np.zeros((num_envs, history, OBS_DIM), np.float32)
        self.start_x = np.zeros(num_envs)
        self.last_x = np.zeros(num_envs)
        self.max_x = np.zeros(num_envs)
        self.returns = np.zeros(num_envs)
        self.reset(np.arange(num_envs))

    def refresh(self):
        self.q[:] = STAND
        self.dq[:] = 0
        self.torque[:] = 0
        for g, s in zip(self.groups, self.slices, strict=True):
            self.q[s, g.slot] = g.qpos[:, g.qadr]
            self.dq[s, g.slot] = g.qvel[:, g.dadr]
            self.torque[s, g.slot] = g.torque[:, g.aadr]
            self.pos[s] = g.qpos[:, :3]
            rot = rotation(g.qpos[:, 3:7])
            self.vel[s] = np.einsum("nji,nj->ni", rot, g.qvel[:, :3])
            self.gyro[s] = g.qvel[:, 3:6]
            self.up[s] = rot[:, 2, :]
            if g.rays:
                values = np.concatenate(g.rays, axis=1)
                self.scan[s] = np.where(values >= 0, np.minimum(values, 2), 2)

    def reset(self, ids):
        ids = np.asarray(ids, dtype=int)
        if not len(ids):
            return
        n = len(ids)
        self.action[ids] = 0
        self.steps[ids] = 0
        self.returns[ids] = 0
        self.commands[ids] = [0.55, 0, 0]
        self.strength[ids] = 1
        if self.randomize:
            self.commands[ids, 0] = self.rng.uniform(0.3, 0.8, n)
            self.commands[ids, 1] = self.rng.uniform(-0.1, 0.1, n)
            self.commands[ids, 2] = self.rng.uniform(-0.3, 0.3, n)
            affected = self.rng.random(n) < 0.5
            j = self.rng.integers(0, 12, n)
            self.strength[ids[affected], j[affected]] = self.rng.uniform(
                0.5, 1, affected.sum()
            )
        self.fault_at[ids] = self.rng.integers(100, 300, n) if self.faults else 100000
        self.fault_joint[ids] = self.rng.integers(0, 12, n)
        self.fault_strength[ids] = self.rng.uniform(0.4, 0.8, n)
        for g, s in zip(self.groups, self.slices, strict=True):
            global_ids = ids[(ids >= s.start) & (ids < s.stop)]
            local = global_ids - s.start
            if not len(local):
                continue
            g.batch.reset(local)
            g.qpos[local] = g.initial_qpos
            g.qvel[local] = 0
            if self.randomize:
                g.qpos[local[:, None], g.qadr] += self.rng.uniform(
                    -0.035, 0.035, (len(local), len(g.slot))
                )
                g.qpos[local, 1] = self.rng.uniform(-0.15, 0.15, len(local))
                g.qvel[local, :] = self.rng.uniform(
                    -0.05, 0.05, (len(local), g.model.nv)
                )
            g.ctrl[local] = STAND[g.slot]
            g.set_strength(local, self.strength[global_ids])
            g.batch.forward(local)
            self.valid[global_ids[:, None], g.slot] = 1
            self.context[global_ids, :4] = g.body.calf
            for leg in g.body.absent:
                self.context[global_ids, LEGS.index(leg)] = 0
        self.context[ids, 4:16] = self.strength[ids]
        self.context[ids, 16:] = 1
        self.refresh()
        self.start_x[ids] = self.pos[ids, 0]
        self.max_x[ids] = self.last_x[ids] = self.pos[ids, 0]
        self.history[ids] = self.obs()[ids, None, :]

    def obs(self):
        # Fixed nominal normalization; no private morphology/strength in this path.
        return np.concatenate(
            (
                (self.q - STAND) * self.valid,
                self.dq * 0.05,
                self.gyro * 0.25,
                self.up,
                self.action,
                self.commands,
                self.valid,
                self.scan * 0.5,
            ),
            1,
        ).astype(np.float32)

    def critic_obs(self):
        return np.concatenate(
            (self.obs(), self.context, self.vel, self.pos[:, 2:3]), 1
        ).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action), -3, 3)
        old_action = self.action.copy()
        events = np.flatnonzero(self.steps == self.fault_at)
        if len(events):
            self.strength[events, self.fault_joint[events]] = self.fault_strength[
                events
            ]
            self.context[events, 4:16] = self.strength[events]
        for g, s in zip(self.groups, self.slices, strict=True):
            local = events[(events >= s.start) & (events < s.stop)] - s.start
            if len(local):
                g.set_strength(local, self.strength[local + s.start])
            g.ctrl[:] = STAND[g.slot] + ACTION_SCALE * action[s][:, g.slot]
        if self.pool:
            list(
                self.pool.map(
                    lambda g: g.batch.step(nstep=self.decimation), self.groups
                )
            )
        else:
            self.groups[0].batch.step(nstep=self.decimation)
        if any(np.any(g.warnings[:, :, 1]) for g in self.groups):
            raise FloatingPointError(
                "MuJoCo numerical warning; refusing an automatically corrected trajectory"
            )
        self.steps += 1
        previous_scan = self.scan.copy()
        self.refresh()
        # Scan delivery alternates 2/3 control steps: exact average 20 Hz.
        new_scan = (self.steps * CONTROL_DT // 0.05) != (
            (self.steps - 1) * CONTROL_DT // 0.05
        )
        self.scan[~new_scan] = previous_scan[~new_scan]
        self.action[:] = action
        verror = np.sum((self.vel[:, :2] - self.commands[:, :2]) ** 2, 1)
        track = np.exp(-verror / 0.25)
        turn = np.exp(-((self.gyro[:, 2] - self.commands[:, 2]) ** 2) / 0.25)
        upright = self.up[:, 0] ** 2 + self.up[:, 1] ** 2
        progress = np.clip((self.pos[:, 0] - self.last_x) / CONTROL_DT, -1, 1)
        effort = np.sum((self.torque / LIMITS) ** 2, 1)
        rate = np.sum((action - old_action) ** 2, 1)
        reward = (
            2 * track
            + 0.5 * turn
            + 0.3
            + 0.3 * progress
            - 1.0 * upright
            - 0.025 * effort
            - 0.006 * rate
        )
        reward -= 0.02 * np.sum(self.gyro[:, :2] ** 2, 1)
        reward -= 0.1 * self.vel[:, 2] ** 2
        finite = np.isfinite(self.q).all(1) & np.isfinite(self.vel).all(1)
        fell = (self.up[:, 2] < 0.15) | (self.pos[:, 2] < 0.09) | ~finite
        timeout = self.steps >= 500
        reward[fell] -= 2
        self.last_x[:] = self.pos[:, 0]
        self.max_x = np.maximum(self.max_x, self.pos[:, 0])
        self.returns += reward
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = self.obs()
        return (
            reward.astype(np.float32),
            fell | timeout,
            fell,
            {
                "track": track,
                "turn": turn,
                "speed": self.vel[:, 0],
                "progress": self.pos[:, 0] - self.start_x,
                "faults": len(events),
            },
        )

    def close(self):
        # Native pools are released when their Batch objects are collected.
        if self.pool:
            self.pool.shutdown()
        self.groups.clear()


def benchmark(seconds=3, num_envs=512, threads=12, **kwargs):
    setup = time.perf_counter()
    env = DogEnv(num_envs=num_envs, threads=threads, **kwargs)
    setup = time.perf_counter() - setup
    t = time.perf_counter()
    steps = 0
    while time.perf_counter() - t < seconds:
        _, done, _, _ = env.step(np.zeros((env.n, 12)))
        env.reset(np.flatnonzero(done))
        steps += env.n
    elapsed = time.perf_counter() - t
    result = {
        "setup_seconds": setup,
        "seconds": elapsed,
        "control_transitions": steps,
        "transitions_per_second": steps / elapsed,
        "num_envs": num_envs,
        "threads": threads,
        "bodies": len(env.groups),
        "timestep": env.timestep,
        "full_body_contacts": True,
    }
    env.close()
    return result
